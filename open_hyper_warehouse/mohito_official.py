"""Official MOHITO networks with a warehouse inactive-slot adapter.

The Actor, Critic and GNN classes are imported from the authors' public UAI
2025 repository in ``external/mohito-public``.  This module only translates the
warehouse tensor observation into the published Levi incidence graph and masks
empty population slots.  It deliberately does not reimplement MOHITO as PPO.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .config import EnvConfig, EvaluationConfig, ExperimentConfig, ModelConfig, TrainConfig
from .buffer import stack_observations
from .baselines import baseline_metadata
from .env import AGENT_FEATURE_DIM, TASK_FEATURE_DIM
from .telemetry import TrainingTelemetry, write_training_log
from .vector_env import make_vector_env


MOHITO_COMMIT = "105ddccbacc92953a7aa6bd0d19f8ec299f7476d"


def _load_official_modules():
    root = Path(__file__).resolve().parents[1]
    package = root / "external" / "mohito-public" / "wildfire" / "mohito"
    if not package.exists():
        raise RuntimeError(
            "official MOHITO checkout is missing; run scripts/fetch_official_baselines.ps1"
        )
    source_marker = root / "external" / "mohito-public" / "UPSTREAM_SOURCE.txt"
    if source_marker.exists():
        marker_text = source_marker.read_text(encoding="utf-8")
        if MOHITO_COMMIT not in marker_text:
            raise RuntimeError(
                f"MOHITO source marker does not contain pinned commit {MOHITO_COMMIT}"
            )
    git_head = root / "external" / "mohito-public" / ".git" / "HEAD"
    if git_head.exists():
        head = git_head.read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref_path = git_head.parent / head.removeprefix("ref: ")
            head = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else head
        if head != MOHITO_COMMIT:
            raise RuntimeError(
                f"MOHITO source is at {head}, expected pinned commit {MOHITO_COMMIT}; "
                "run scripts/fetch_official_baselines.ps1"
            )
    package_text = str(package)
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    try:
        from mohito.mohito_layers import Actor, Critic
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise RuntimeError(
            "mohito_official requires torch-geometric and the official checkout"
        ) from exc
    return Actor, Critic, Data


Actor, Critic, Data = _load_official_modules()


def build_official_mohito_policy(
    env_cfg: EnvConfig, train_cfg: TrainConfig, model_cfg: ModelConfig
) -> "InactiveSlotMohito":
    return InactiveSlotMohito(
        ExperimentConfig(
            env=env_cfg,
            train=train_cfg,
            model=model_cfg,
            evaluation=EvaluationConfig(),
        )
    )


def _pad_node(identifier: float, values: np.ndarray | Tensor, width: int, device) -> Tensor:
    out = torch.zeros(width, dtype=torch.float32, device=device)
    out[0] = identifier
    value = torch.as_tensor(values, dtype=torch.float32, device=device).flatten()
    length = min(value.numel(), width - 1)
    out[1 : 1 + length] = value[:length]
    return out


class InactiveSlotMohito(nn.Module):
    """One official MOHITO actor/critic pair per maximum warehouse slot.

    Population openness is handled only by a wrapper: inactive slots emit NOOP,
    are absent from the critic graph and receive no gradient.  Consequently the
    official fixed-agent MOHITO mechanism can be evaluated without pretending
    that padded agents are real agents.
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.env_cfg = cfg.env
        self.node_width = 1 + max(AGENT_FEATURE_DIM, TASK_FEATURE_DIM, 3)
        common = dict(
            optimizer_fn=torch.optim.Adam,
            psi=cfg.model.mohito_tau,
            K=0,
            beta=0.01,
            regularize=True,
            gradient_clip=0.0,
            hidden_dim=cfg.model.mohito_hidden_dim,
            number_of_layers=cfg.model.mohito_layers,
            relu_slope=0.1,
            dropout_rate=0.0,
            num_heads=cfg.model.mohito_heads,
            node_feature_size=self.node_width,
            edge_feature_size=None,
            add_self_loops=True,
            use_graph_norm=True,
            layer_type="gatv2",
            layer_params={"dropout": 0.0},
            device=cfg.train.device if cfg.train.device != "auto" else "cpu",
        )
        self.actors = nn.ModuleList(
            Actor(
                lr=cfg.model.mohito_actor_lr,
                name=f"warehouse_actor_{slot}",
                rotate_hyperedges=True,
                exp_include_E_in_Gc=False,
                exp_log_soft_output=False,
                exp_linear_combination=False,
                fc_hyperedges=False,
                **common,
            )
            for slot in range(cfg.env.max_agents)
        )
        self.critics = nn.ModuleList(
            Critic(
                lr=cfg.model.mohito_critic_lr,
                name=f"warehouse_critic_{slot}",
                num_agents=cfg.env.max_agents,
                actor_hidden_dim=cfg.model.mohito_hidden_dim,
                exp_include_E_in_Gc=True,
                exp_linear_combination=False,
                fc_hyperedges=True,
                **common,
            )
            for slot in range(cfg.env.max_agents)
        )
        self.register_buffer(
            "epsilon", torch.full((cfg.env.max_agents,), cfg.model.mohito_initial_epsilon)
        )

    def set_runtime_device(self, device: torch.device | str) -> None:
        value = str(torch.device(device))
        for actor in self.actors:
            actor.main.device = value
            actor.target.device = value
        for critic in self.critics:
            critic.main[0].device = value
            critic.target[0].device = value

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.set_runtime_device(next(self.parameters()).device)
        return result

    def _graph(self, obs: dict[str, np.ndarray | Tensor], focal: int, device=None):
        device = device or next(self.parameters()).device
        # Replay observations live on the CPU.  Constructing each tiny graph on
        # CUDA and then calling ``item``/``tolist`` forced several device
        # synchronizations per agent.  Build the discrete topology and feature
        # matrix on CPU and transfer each finished graph once.  This is an
        # execution-only optimization: node order, identifiers and edges are
        # identical to the original adapter.
        def as_numpy(value):
            if isinstance(value, Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        agent_features = as_numpy(obs["agent_features"]).astype(np.float32, copy=False)
        task_features = as_numpy(obs["task_features"]).astype(np.float32, copy=False)
        agent_mask = as_numpy(obs["agent_mask"]).astype(bool, copy=False)
        action_mask = as_numpy(obs["action_mask"]).astype(bool, copy=False)
        valid_actions = np.flatnonzero(action_mask[focal]).astype(np.int64, copy=False)
        if valid_actions.size == 0 or valid_actions[0] != 0:
            valid_actions = np.concatenate((np.zeros(1, dtype=np.int64), valid_actions))

        nodes: list[np.ndarray] = []

        def append_node(identifier: float, values) -> int:
            row = np.zeros(self.node_width, dtype=np.float32)
            row[0] = identifier
            flat = np.asarray(values, dtype=np.float32).reshape(-1)
            length = min(flat.size, self.node_width - 1)
            if length:
                row[1 : 1 + length] = flat[:length]
            index = len(nodes)
            nodes.append(row)
            return index

        # One Levi node per feasible (agent, task, role-action) hyperedge.
        hyperedge_indices = []
        for _ in valid_actions:
            hyperedge_indices.append(append_node(5.0, ()))

        task_node: dict[int, int] = {}
        for action_value in valid_actions:
            action_id = int(action_value)
            task = action_id - 1
            values = np.zeros(TASK_FEATURE_DIM, dtype=np.float32) if action_id == 0 else task_features[task]
            task_node[action_id] = append_node(3.0, values)

        # MOHITO's third partition is action semantics, not the task identity.
        action_node: dict[int, int] = {}
        role = int(agent_features[focal, 1] > 0.5)
        for action_value in valid_actions:
            action_id = int(action_value)
            semantic = 0 if action_id == 0 else role + 1
            if semantic not in action_node:
                action_node[semantic] = append_node(4.0, (semantic, role))

        other_nodes = []
        for other in np.flatnonzero(agent_mask).tolist():
            if other == focal:
                continue
            other_nodes.append(append_node(2.0, agent_features[other]))
        focal_node = append_node(1.0, agent_features[focal])

        edges: list[tuple[int, int]] = []
        for ordinal, action_value in enumerate(valid_actions):
            action_id = int(action_value)
            h = hyperedge_indices[ordinal]
            semantic = 0 if action_id == 0 else role + 1
            edges.extend(
                [(focal_node, h), (h, focal_node), (task_node[action_id], h),
                 (h, task_node[action_id]), (action_node[semantic], h), (h, action_node[semantic])]
            )
        for other_node in other_nodes:
            edges.extend([(other_node, focal_node), (focal_node, other_node)])
        edge_array = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
        graph = Data(
            x=torch.as_tensor(np.stack(nodes), dtype=torch.float32, device=device),
            edge_index=torch.as_tensor(edge_array, dtype=torch.long, device=device).contiguous(),
        )
        graph.action_ids = torch.as_tensor(valid_actions, dtype=torch.long, device=device)
        graph.hyperedge_indices = torch.as_tensor(
            hyperedge_indices, dtype=torch.long, device=device
        )
        return graph

    @staticmethod
    def _as_numpy_sample(obs: dict[str, Tensor], batch: int) -> dict[str, np.ndarray]:
        sample = {}
        for key, value in obs.items():
            if isinstance(value, Tensor):
                sample[key] = value[batch].detach().cpu().numpy()
            else:
                sample[key] = np.asarray(value[batch])
        return sample

    @staticmethod
    def _split_actor_output(output, graph_count: int):
        """Split an official batched ActorReturn without detaching gradients."""

        pieces = []
        for batch_index in range(graph_count):
            mask = output.batch_indices == batch_index
            graph = type(output.graph)(
                x=output.graph.x[mask],
                agent_E_mask=output.graph.agent_E_mask[mask],
                agent_mask=output.graph.agent_mask[mask],
            )
            logits = None if output.logits is None else [output.logits[batch_index]]
            pieces.append(
                type(output)(
                    graph=graph,
                    batch_indices=torch.zeros(
                        int(mask.sum()), dtype=output.batch_indices.dtype,
                        device=output.batch_indices.device,
                    ),
                    task_action=output.task_action[batch_index : batch_index + 1],
                    hyperedge=output.hyperedge[batch_index : batch_index + 1],
                    logits=logits,
                )
            )
        return pieces

    def actor_outputs_many(self, observations, target: bool = False, detach: bool = False):
        """Evaluate each slot once on a PyG batch of replay observations.

        MOHITO owns a separate actor per padded slot, so batching is performed
        across replay samples for the same slot.  It changes neither parameter
        sharing nor the inactive-slot rule.
        """

        caches = []
        grouped: dict[int, list[tuple[int, Any]]] = {}
        for sample_id, obs in enumerate(observations):
            active = np.flatnonzero(obs["agent_mask"]).tolist()
            outputs: dict[int, Any] = {}
            graphs: dict[int, Any] = {}
            caches.append((active, outputs, graphs))
            for slot in active:
                graph = self._graph(obs, slot)
                graphs[slot] = graph
                grouped.setdefault(slot, []).append((sample_id, graph))

        for slot, entries in grouped.items():
            output = self.actors[slot]([graph for _, graph in entries], target=target)
            pieces = self._split_actor_output(output, len(entries))
            for (sample_id, _), piece in zip(entries, pieces):
                if detach:
                    piece.graph.x = piece.graph.x.detach()
                    piece.hyperedge = piece.hyperedge.detach()
                caches[sample_id][1][slot] = piece
        return caches

    def actor_outputs(self, obs: dict[str, np.ndarray], target: bool = False, detach=False):
        return self.actor_outputs_many([obs], target=target, detach=detach)[0]

    def critic_graph(self, active, outputs, graphs, action_ids: dict[int, int]):
        """Build the official disjunctive-join centralized critic graph."""
        kept_x, kept_edges, selected = [], [], []
        offset = 0
        for slot in active:
            graph = graphs[slot]
            x = outputs[slot].graph.x
            # Official wrapper removes 'other-agent' observer nodes from G_c.
            keep = x[:, 0] != 2
            old_to_new = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
            old_to_new[keep] = torch.arange(int(keep.sum()), device=x.device) + offset
            kept_x.append(x[keep])
            edge_keep = keep[graph.edge_index[0]] & keep[graph.edge_index[1]]
            kept_edges.append(old_to_new[graph.edge_index[:, edge_keep]])
            valid = graph.action_ids
            requested = int(action_ids[slot])
            match = torch.nonzero(valid == requested, as_tuple=False).flatten()
            ordinal = int(match[0]) if match.numel() else 0
            selected.append(old_to_new[graph.hyperedge_indices[ordinal]])
            offset += int(keep.sum())
        x = torch.cat(kept_x, dim=0)
        edge_index = torch.cat(kept_edges, dim=1) if kept_edges else torch.empty((2, 0), dtype=torch.long, device=x.device)
        # Disjunctive join: fully connect retained focal-agent nodes and all
        # hyperedge nodes, exactly as the public MOHITO critic wrapper does.
        extra = []
        for identifier in (1.0, 5.0):
            ids = torch.nonzero(x[:, 0] == identifier, as_tuple=False).flatten()
            if ids.numel() > 1:
                src = ids.repeat_interleave(ids.numel())
                dst = ids.repeat(ids.numel())
                nonself = src != dst
                extra.append(torch.stack([src[nonself], dst[nonself]]))
        if extra:
            edge_index = torch.cat([edge_index, *extra], dim=1)
        graph = Data(x=x, edge_index=edge_index)
        return graph, [torch.stack(selected)]

    def act(self, obs: dict[str, Tensor], deterministic: bool = False):
        batch_size, slots = obs["agent_mask"].shape
        device = next(self.parameters()).device
        actions = torch.zeros((batch_size, slots), dtype=torch.long, device=device)
        grouped: dict[int, list[tuple[int, Any]]] = {}
        for b in range(batch_size):
            sample = self._as_numpy_sample(obs, b)
            for slot in np.flatnonzero(sample["agent_mask"]).tolist():
                graph = self._graph(sample, slot)
                grouped.setdefault(slot, []).append((b, graph))
        for slot, entries in grouped.items():
            out = self.actors[slot]([graph for _, graph in entries], target=False)
            for local_index, (batch_index, graph) in enumerate(entries):
                ordinal = int(out.task_action[local_index])
                if not deterministic and random.random() < float(self.epsilon[slot]):
                    ordinal = random.randrange(graph.action_ids.numel())
                actions[batch_index, slot] = graph.action_ids[ordinal]
        zeros = torch.zeros(batch_size, dtype=torch.float32, device=actions.device)
        return actions, zeros, zeros, zeros

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class OfficialMohitoTrainer:
    """MADDPG-style replay/target-network training from the MOHITO paper."""

    def __init__(self, config: ExperimentConfig, seed: int, output_dir: str | Path):
        from .trainer import resolve_device, set_seed

        set_seed(seed)
        # Do not mutate the shared suite configuration when resolving a device.
        # Every method must receive the same immutable experimental factors.
        self.cfg, self.seed = deepcopy(config), seed
        self.device = resolve_device(self.cfg.train.device)
        self.policy = InactiveSlotMohito(self.cfg).to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_counts = np.zeros(self.cfg.train.num_envs, dtype=np.int64)
        initial_seeds = [self._episode_seed(i) for i in range(self.cfg.train.num_envs)]
        self.vector_env = make_vector_env(
            self.cfg.env,
            self.cfg.train.num_envs,
            self.cfg.train.parallel_env_workers,
            initial_seeds,
        )
        self.observations = self.vector_env.reset_all(initial_seeds)
        self.replay: deque[tuple[Any, ...]] = deque(maxlen=self.cfg.model.mohito_replay_size)
        self.rng = np.random.default_rng(seed)
        self.logs = []
        self.telemetry = TrainingTelemetry(
            self.output_dir,
            {
                "algorithm": "mohito_official",
                "training_seed": seed,
                "device": str(self.device),
                "environment_backend": self.vector_env.backend,
                "num_envs": self.cfg.train.num_envs,
                "parallel_env_workers": self.cfg.train.parallel_env_workers,
                "total_steps": self.cfg.train.total_steps,
            },
        )

    @staticmethod
    def _copy_obs(obs):
        return {key: value.copy() for key, value in obs.items()}

    def _episode_seed(self, env_index: int) -> int:
        # Identical to PPOTrainer: paired methods see the same reproducible
        # episode-seed schedule, while policy actions remain method-specific.
        return (
            self.seed * 10_000_000
            + env_index * 100_000
            + int(self.episode_counts[env_index])
        )

    @staticmethod
    def _offset_selected(graphs, selected):
        """Convert graph-local selected node ids to PyG batch-global ids."""

        result = []
        offset = 0
        for graph, local in zip(graphs, selected):
            result.append(local + offset)
            offset += int(graph.x.shape[0])
        return result

    def _update(self):
        batch_size = self.cfg.model.mohito_batch_size
        if len(self.replay) < max(batch_size, self.cfg.model.mohito_warmup_steps):
            return None
        indexes = self.rng.choice(len(self.replay), batch_size, replace=False)
        samples = [self.replay[int(i)] for i in indexes]
        # The public controller computes all actors' hyperedges once per replay
        # batch and reuses them across per-agent critic updates.  Cache the same
        # quantities here; rebuilding every observer graph inside every slot
        # would add an accidental O(N^2) adapter overhead.
        current_observations = [sample[0] for sample in samples]
        next_observations = [sample[3] for sample in samples]
        with torch.no_grad():
            current_cache = self.policy.actor_outputs_many(
                current_observations, detach=True
            )
            target_cache = self.policy.actor_outputs_many(
                next_observations, target=True, detach=True
            )
        losses = []
        # A slot is updated only from transitions in which it was active.  This
        # is the inactive-slot wrapper's essential fairness guarantee.
        for slot in range(self.cfg.env.max_agents):
            relevant_ids = [i for i, sample in enumerate(samples) if sample[0]["agent_mask"][slot]]
            if not relevant_ids:
                continue
            critic = self.policy.critics[slot]
            actor = self.policy.actors[slot]
            current_critic_graphs = []
            current_selected = []
            target_critic_graphs = []
            target_selected = []
            target_positions = []
            for sample_id in relevant_ids:
                obs, actions, reward, next_obs, done = samples[sample_id]
                active, outputs, graphs = current_cache[sample_id]
                current_ids = {a: int(actions[a]) for a in active}
                cg, selected = self.policy.critic_graph(active, outputs, graphs, current_ids)
                current_critic_graphs.append(cg)
                current_selected.append(selected[0])
                n_active, n_outputs, n_graphs = target_cache[sample_id]
                if n_active:
                    next_ids = {
                        a: int(n_graphs[a].action_ids[int(n_outputs[a].task_action[0])])
                        for a in n_active
                    }
                    ncg, nselected = self.policy.critic_graph(
                        n_active, n_outputs, n_graphs, next_ids
                    )
                    target_positions.append(len(current_critic_graphs) - 1)
                    target_critic_graphs.append(ncg)
                    target_selected.append(nselected[0])

            q = critic(
                current_critic_graphs,
                self._offset_selected(current_critic_graphs, current_selected),
                target=False,
            ).q
            with torch.no_grad():
                nq = torch.zeros(len(relevant_ids), dtype=q.dtype, device=self.device)
                if target_critic_graphs:
                    target_q = critic(
                        target_critic_graphs,
                        self._offset_selected(target_critic_graphs, target_selected),
                        target=True,
                    ).q
                    nq[torch.as_tensor(target_positions, device=self.device)] = target_q
                rewards = torch.as_tensor(
                    [float(samples[i][2]) for i in relevant_ids],
                    dtype=q.dtype,
                    device=self.device,
                )
                not_done = torch.as_tensor(
                    [1.0 - float(samples[i][4]) for i in relevant_ids],
                    dtype=q.dtype,
                    device=self.device,
                )
                target = rewards + self.cfg.train.gamma * not_done * nq
            critic_loss = (q - target).square().mean()
            critic.optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic.update()

            fresh_graphs = [
                self.policy._graph(samples[sample_id][0], slot)
                for sample_id in relevant_ids
            ]
            fresh_output = actor(fresh_graphs, target=False)
            fresh_pieces = self.policy._split_actor_output(
                fresh_output, len(fresh_graphs)
            )
            actor_critic_graphs = []
            actor_selected = []
            for local_index, sample_id in enumerate(relevant_ids):
                obs = samples[sample_id][0]
                active, cached_outputs, cached_graphs = current_cache[sample_id]
                graphs = dict(cached_graphs)
                outputs = dict(cached_outputs)
                graphs[slot] = fresh_graphs[local_index]
                outputs[slot] = fresh_pieces[local_index]
                chosen = {a: int(graphs[a].action_ids[int(outputs[a].task_action[0])]) for a in active}
                # Other actors are constants for this slot's policy gradient.
                local_outputs = {}
                for other in active:
                    result = outputs[other]
                    if other != slot:
                        result.graph.x = result.graph.x.detach()
                        result.hyperedge = result.hyperedge.detach()
                    local_outputs[other] = result
                cg, selected = self.policy.critic_graph(active, local_outputs, graphs, chosen)
                actor_critic_graphs.append(cg)
                actor_selected.append(selected[0])
            actor_loss = -critic(
                actor_critic_graphs,
                self._offset_selected(actor_critic_graphs, actor_selected),
                target=False,
            ).q.mean()
            actor.optimizer.zero_grad(set_to_none=True)
            critic.optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor.update()
            losses.append((float(actor_loss.detach()), float(critic_loss.detach())))
        self.policy.epsilon.mul_(self.cfg.model.mohito_epsilon_decay).clamp_(min=self.cfg.model.mohito_min_epsilon)
        if not losses:
            return None
        return {
            "actor_loss": float(np.mean([x[0] for x in losses])),
            "critic_loss": float(np.mean([x[1] for x in losses])),
            "updated_slots": len(losses),
        }

    def train(self):
        from .trainer import TrainingResult

        started = time.perf_counter()
        steps = updates = episode = 0
        experiences_since_update = 0
        next_checkpoint = self.cfg.train.checkpoint_interval
        logging_interval = self.cfg.train.num_envs * self.cfg.train.rollout_steps
        loss_window: list[dict[str, float]] = []
        completed_window: list[dict[str, float]] = []
        while steps < self.cfg.train.total_steps:
            stacked = stack_observations(self.observations)
            # Graph construction starts from CPU NumPy arrays.  Passing the
            # stacked observation directly avoids an otherwise redundant
            # host->device->host round trip before every action selection.
            with torch.inference_mode():
                action_batch = self.policy.act(stacked, deterministic=False)[0].cpu().numpy()
            transitions = self.vector_env.step(action_batch)
            next_observations = []
            reset_indexes: list[int] = []
            reset_seeds: list[int] = []
            for env_index, (next_obs, reward, done, info) in enumerate(transitions):
                obs = self.observations[env_index]
                actions = action_batch[env_index]
                self.replay.append(
                    (
                        self._copy_obs(obs),
                        actions.copy(),
                        reward,
                        self._copy_obs(next_obs),
                        done,
                    )
                )
                steps += 1
                experiences_since_update += 1
                cadence = self.cfg.model.mohito_experiences_before_update
                if experiences_since_update >= cadence:
                    experiences_since_update -= cadence
                    # Both controls are parameters of the official MOHITO
                    # training interface.  They change update-to-data ratio,
                    # not its Actor/Critic/GNN or target-network mechanism.
                    for _ in range(self.cfg.model.mohito_backups_per_update):
                        loss = self._update()
                        if loss:
                            updates += 1
                            loss_window.append(loss)
                if done:
                    episode += 1
                    completed_window.append(info)
                    self.episode_counts[env_index] += 1
                    reset_indexes.append(env_index)
                    reset_seeds.append(self._episode_seed(env_index))
                next_observations.append(next_obs)
            for index, obs in self.vector_env.reset_at(reset_indexes, reset_seeds).items():
                next_observations[index] = obs
            self.observations = next_observations
            if steps % logging_interval == 0:
                row: dict[str, Any] = {"steps": steps, "updates": updates}
                if loss_window:
                    for key in ("actor_loss", "critic_loss", "updated_slots"):
                        row[key] = float(np.mean([loss[key] for loss in loss_window]))
                if completed_window:
                    for key in (
                        "return",
                        "deliveries",
                        "completion_ratio",
                        "service_level",
                        "throughput_per_100_steps",
                        "deadline_miss_rate",
                        "mean_flow_time",
                        "contention_per_delivery",
                        "travel_per_delivery",
                        "unserved_orders",
                    ):
                        row[f"episode_{key}"] = float(
                            np.mean([metrics.get(key, 0.0) for metrics in completed_window])
                        )
                row["epsilon_mean"] = float(self.policy.epsilon.mean())
                row = self.telemetry.record(row, self.device)
                self.logs.append(row)
                loss_window.clear()
                completed_window.clear()
                print(json.dumps(row, ensure_ascii=False))
            if steps >= next_checkpoint:
                self._save(self.output_dir / f"step_{steps}.pt", steps, updates)
                write_training_log(self.output_dir, self.logs)
                next_checkpoint += self.cfg.train.checkpoint_interval
        wall = time.perf_counter() - started
        checkpoint = self.output_dir / "final.pt"
        self._save(checkpoint, steps, updates, wall)
        write_training_log(self.output_dir, self.logs)
        return TrainingResult(checkpoint, steps, updates, wall, None)

    def _save(
        self,
        checkpoint: Path,
        steps: int,
        updates: int,
        wall_seconds: float | None = None,
    ) -> None:
        torch.save(
            {
                "algorithm": "mohito_official",
                "config": self.cfg.to_dict(),
                "seed": self.seed,
                "steps": steps,
                "updates": updates,
                "wall_seconds": wall_seconds,
                "training_device": str(self.device),
                "torch_version": torch.__version__,
                "state_dict": self.policy.state_dict(),
                "parameter_count": self.policy.parameter_count(),
                "official_source": "external/mohito-public",
                "official_source_commit": MOHITO_COMMIT,
                "wrapper": "inactive slots emit NOOP and are excluded from critic/update",
                "sampling_protocol": "same parallel environments and episode seeds as MAPPO",
                "replay_update_cadence": {
                    "experiences_before_update": self.cfg.model.mohito_experiences_before_update,
                    "backups_per_update": self.cfg.model.mohito_backups_per_update,
                    "actor_graph_cache": "one current/target construction per sampled transition",
                },
                "execution_optimizations": {
                    "graph_construction": "CPU topology plus one device transfer per graph",
                    "actor_batching": "same slot batched across environments/replay samples",
                    "critic_batching": "same slot batched across replay samples",
                    "objective_change": False,
                },
                "method_metadata": baseline_metadata("mohito_official"),
                "environment_backend": self.vector_env.backend,
                "parallel_env_workers": self.cfg.train.parallel_env_workers,
            },
            checkpoint,
        )

    def close(self) -> None:
        self.vector_env.close()
