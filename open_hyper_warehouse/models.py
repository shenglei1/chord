from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .coalition import (
    COALITION_CAPACITY_DIM,
    CoalitionCandidates,
    build_coalition_candidates,
    task_capacity_features,
)
from .config import EnvConfig, ModelConfig, TrainConfig
from .env import AGENT_FEATURE_DIM, TASK_FEATURE_DIM
from .hypergraph import (
    DenseHypergraphEncoder,
    DensePairwiseGraphEncoder,
    IncidenceBatch,
    build_open_incidence,
    incidence_to_pairwise_adjacency,
)


def mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2) -> nn.Sequential:
    modules: list[nn.Module] = []
    current = input_dim
    for _ in range(max(layers - 1, 0)):
        modules.extend([nn.Linear(current, hidden_dim), nn.ReLU()])
        current = hidden_dim
    modules.append(nn.Linear(current, output_dim))
    return nn.Sequential(*modules)


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


@dataclass(slots=True)
class PolicyOutput:
    logits: Tensor
    value: Tensor
    decision_mask: Tensor
    deterministic_actions: Tensor | None = None


class WarehousePolicy(nn.Module, ABC):
    # PPO ratios are computed per agent. This flag only controls whether every
    # environment state or every active agent receives equal weight after
    # clipping; the no-popnorm ablation switches it off.
    population_normalized_actor_terms = True

    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.hidden_dim = train_cfg.hidden_dim
        self.agent_input = mlp(AGENT_FEATURE_DIM, self.hidden_dim, self.hidden_dim)
        self.task_input = mlp(TASK_FEATURE_DIM, self.hidden_dim, self.hidden_dim)
        self.noop_head = mlp(self.hidden_dim, self.hidden_dim, 1)
        self.pair_head = mlp(self.hidden_dim * 4, self.hidden_dim, 1)
        self.population_feature_dim = 8
        self.value_head = mlp(
            self.hidden_dim * 2 + self.population_feature_dim,
            self.hidden_dim,
            1,
        )

    @abstractmethod
    def encode(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _decode(
        self, obs: dict[str, Tensor], agents: Tensor, tasks: Tensor
    ) -> PolicyOutput:
        batch, num_agents, hidden = agents.shape
        num_tasks = tasks.shape[1]
        a = agents.unsqueeze(2).expand(batch, num_agents, num_tasks, hidden)
        t = tasks.unsqueeze(1).expand(batch, num_agents, num_tasks, hidden)
        pair = torch.cat([a, t, a - t, a * t], dim=-1)
        task_logits = self.pair_head(pair).squeeze(-1)
        task_logits = self.adjust_task_logits(obs, task_logits)
        logits, deterministic_actions = self._factorized_action_logits(
            agents, task_logits, obs["action_mask"].bool()
        )
        decision_mask = obs["agent_mask"].bool() & (obs["action_mask"].sum(dim=-1) > 1)
        agent_pool = masked_mean(agents, obs["agent_mask"].bool(), dim=1)
        task_pool = masked_mean(tasks, obs["task_mask"].bool(), dim=1)
        population = self._population_features(obs)
        value = self.value_head(
            torch.cat([agent_pool, task_pool, population], dim=-1)
        ).squeeze(-1)
        return PolicyOutput(logits, value, decision_mask, deterministic_actions)

    def adjust_task_logits(self, obs: dict[str, Tensor], task_logits: Tensor) -> Tensor:
        """Hook used by the capacity-only factorized decoder ablation."""

        return task_logits

    def _factorized_action_logits(
        self, agents: Tensor, task_logits: Tensor, action_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Idle gate followed by a conditional pointer over feasible tasks."""

        feasible_tasks = action_mask[..., 1:]
        has_task = feasible_tasks.any(dim=-1)
        masked_tasks = task_logits.masked_fill(~feasible_tasks, -1e9)
        conditional_log_prob = torch.log_softmax(masked_tasks, dim=-1)
        conditional_log_prob = conditional_log_prob.masked_fill(~feasible_tasks, -1e9)
        dispatch_logit = self.noop_head(agents).squeeze(-1)
        log_dispatch = torch.nn.functional.logsigmoid(dispatch_logit)
        log_idle = torch.nn.functional.logsigmoid(-dispatch_logit)
        log_idle = torch.where(has_task, log_idle, torch.zeros_like(log_idle))
        task_log_prob = log_dispatch.unsqueeze(-1) + conditional_log_prob
        task_log_prob = task_log_prob.masked_fill(~feasible_tasks, -1e9)
        logits = torch.cat([log_idle.unsqueeze(-1), task_log_prob], dim=-1)
        best_task = masked_tasks.argmax(dim=-1) + 1
        deterministic = torch.where(
            has_task & (dispatch_logit >= 0.0), best_task, torch.zeros_like(best_task)
        )
        return logits, deterministic

    def _population_features(self, obs: dict[str, Tensor]) -> Tensor:
        """Permutation-invariant cardinality and load context for the critic."""

        agent_features = obs["agent_features"]
        task_features = obs["task_features"]
        agent_mask = obs["agent_mask"].to(agent_features.dtype)
        task_mask = obs["task_mask"].to(task_features.dtype)
        agvs = (agent_mask * agent_features[..., 0]).sum(-1)
        pickers = (agent_mask * agent_features[..., 1]).sum(-1)
        tasks = task_mask.sum(-1)
        waiting = (task_mask * task_features[..., 7]).sum(-1)
        delivery = (task_mask * task_features[..., 8]).sum(-1)
        slack = (task_mask * task_features[..., 5]).sum(-1) / tasks.clamp_min(1.0)
        feasible = obs["action_mask"][..., 1:].to(agent_features.dtype).sum(dim=(-1, -2))
        feasible_scale = float(max(self.env_cfg.max_agents * self.env_cfg.max_tasks, 1))
        return torch.stack(
            [
                agvs / max(self.env_cfg.max_agvs, 1),
                pickers / max(self.env_cfg.max_pickers, 1),
                tasks / max(self.env_cfg.max_tasks, 1),
                torch.log1p(tasks) / math.log1p(max(self.env_cfg.max_tasks, 1)),
                waiting / tasks.clamp_min(1.0),
                delivery / tasks.clamp_min(1.0),
                slack,
                feasible / feasible_scale,
            ],
            dim=-1,
        )

    def forward(self, obs: dict[str, Tensor]) -> PolicyOutput:
        agents, tasks = self.encode(obs)
        return self._decode(obs, agents, tasks)

    def act(self, obs: dict[str, Tensor], deterministic: bool = False):
        output = self(obs)
        distribution = Categorical(logits=output.logits)
        if deterministic and output.deterministic_actions is not None:
            actions = output.deterministic_actions
        else:
            actions = output.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), distribution.entropy(), output.value

    def evaluate_actions(self, obs: dict[str, Tensor], actions: Tensor):
        output = self(obs)
        distribution = Categorical(logits=output.logits)
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            output.value,
            output.decision_mask,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class EntityPolicy(WarehousePolicy):
    """DeepSets control: variable entities without explicit relations."""

    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig):
        super().__init__(env_cfg, train_cfg)
        self.agent_update = mlp(self.hidden_dim * 3, self.hidden_dim, self.hidden_dim)
        self.task_update = mlp(self.hidden_dim * 3, self.hidden_dim, self.hidden_dim)

    def encode(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        agents = self.agent_input(obs["agent_features"])
        tasks = self.task_input(obs["task_features"])
        apool = masked_mean(agents, obs["agent_mask"].bool(), 1)
        tpool = masked_mean(tasks, obs["task_mask"].bool(), 1)
        global_context = torch.cat([apool, tpool], dim=-1)
        agents = self.agent_update(
            torch.cat([agents, global_context.unsqueeze(1).expand(-1, agents.shape[1], -1)], dim=-1)
        )
        tasks = self.task_update(
            torch.cat([tasks, global_context.unsqueeze(1).expand(-1, tasks.shape[1], -1)], dim=-1)
        )
        return agents, tasks


class GraphPolicy(WarehousePolicy):
    """Pairwise 2-section graph control for the proposed hypergraph."""

    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig, model_cfg: ModelConfig):
        super().__init__(env_cfg, train_cfg)
        self.model_cfg = model_cfg
        self.encoder = DensePairwiseGraphEncoder(self.hidden_dim, train_cfg.encoder_layers)

    def encode(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        agents = self.agent_input(obs["agent_features"])
        tasks = self.task_input(obs["task_features"])
        nodes = torch.cat([agents, tasks], dim=1)
        node_mask = torch.cat([obs["agent_mask"], obs["task_mask"]], dim=1).bool()
        incidence = build_open_incidence(
            obs["agent_features"],
            obs["task_features"],
            obs["agent_mask"].bool(),
            obs["task_mask"].bool(),
            obs["action_mask"].bool(),
            self.env_cfg.num_zones,
            self.model_cfg.coalition_top_k,
            include_tasks=True,
            include_zones=True,
        )
        adjacency = incidence_to_pairwise_adjacency(incidence)
        nodes = self.encoder(nodes, adjacency) * node_mask.unsqueeze(-1)
        return nodes[:, : agents.shape[1]], nodes[:, agents.shape[1] :]


class OpenHyperPolicy(WarehousePolicy):
    """CHORD: open task-coalition and shared-resource hypergraph policy."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        train_cfg: TrainConfig,
        model_cfg: ModelConfig,
        include_tasks: bool = True,
        include_zones: bool = True,
        grand_coalition: bool = False,
    ):
        super().__init__(env_cfg, train_cfg)
        self.model_cfg = model_cfg
        self.include_tasks = include_tasks
        self.include_zones = include_zones
        self.grand_coalition = grand_coalition
        self.encoder = DenseHypergraphEncoder(self.hidden_dim, train_cfg.encoder_layers)

    def build_incidence(self, obs: dict[str, Tensor]) -> IncidenceBatch:
        if self.grand_coalition:
            node_mask = torch.cat([obs["agent_mask"], obs["task_mask"]], dim=1).bool()
            matrix = obs["agent_features"].new_zeros(
                (node_mask.shape[0], node_mask.shape[1], 1)
            )
            matrix[:, :, 0] = node_mask.to(matrix.dtype)
            return IncidenceBatch(matrix, node_mask.any(dim=1, keepdim=True))
        return build_open_incidence(
            obs["agent_features"],
            obs["task_features"],
            obs["agent_mask"].bool(),
            obs["task_mask"].bool(),
            obs["action_mask"].bool(),
            self.env_cfg.num_zones,
            self.model_cfg.coalition_top_k,
            include_tasks=self.include_tasks,
            include_zones=self.include_zones,
        )

    def encode_with_incidence(
        self, obs: dict[str, Tensor], incidence: IncidenceBatch
    ) -> tuple[Tensor, Tensor]:
        agents = self.agent_input(obs["agent_features"])
        tasks = self.task_input(obs["task_features"])
        nodes = torch.cat([agents, tasks], dim=1)
        nodes = self.encoder(nodes, incidence)
        node_mask = torch.cat([obs["agent_mask"], obs["task_mask"]], dim=1).unsqueeze(-1)
        nodes = nodes * node_mask
        return nodes[:, : agents.shape[1]], nodes[:, agents.shape[1] :]

    def encode(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        return self.encode_with_incidence(obs, self.build_incidence(obs))

    def forward_with_incidence(
        self, obs: dict[str, Tensor], incidence: IncidenceBatch
    ) -> PolicyOutput:
        """Model-internal intervention used by semantic edge attribution."""

        agents, tasks = self.encode_with_incidence(obs, incidence)
        return self._decode(obs, agents, tasks)


class CapacityAwareOpenHyperPolicy(OpenHyperPolicy):
    """Ablation retaining independent actions but exposing task capacity."""

    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig, model_cfg: ModelConfig):
        super().__init__(env_cfg, train_cfg, model_cfg)
        self.capacity_bias = mlp(
            COALITION_CAPACITY_DIM, self.hidden_dim, 1
        )

    def adjust_task_logits(self, obs: dict[str, Tensor], task_logits: Tensor) -> Tensor:
        bias = self.capacity_bias(task_capacity_features(obs, self.env_cfg)).squeeze(-1)
        return task_logits + bias.unsqueeze(1)


class CoalitionReservationMixin:
    """Capacity-aware hyperedge reservation (CAHR) joint-action decoder.

    A decision is a variable-length sequence of mutually compatible
    ``{AGV, picker, task}`` coalitions terminated by STOP.  The stored action
    probability is the exact sum of conditional log probabilities in that
    sequence, so PPO optimizes a joint warehouse action rather than treating
    collision-prone per-agent proposals as independent samples.
    """

    uses_coalition_reservation = True

    def _init_reservation_decoder(
        self, model_cfg: ModelConfig, use_capacity_features: bool
    ) -> None:
        self.model_cfg = model_cfg
        self.use_capacity_features = use_capacity_features
        self.reservation_rounds = int(model_cfg.reservation_max_rounds) or min(
            self.env_cfg.max_agvs, self.env_cfg.max_pickers
        )
        # Project before candidate gathering: at the formal 512-state
        # minibatch, gathering full 128-D representations for every padded
        # triple would waste hundreds of MB on inactive candidate slots.
        self.reservation_dim = max(16, self.hidden_dim // 4)
        self.coalition_agv = nn.Linear(
            self.hidden_dim, self.reservation_dim, bias=False
        )
        self.coalition_picker = nn.Linear(
            self.hidden_dim, self.reservation_dim, bias=False
        )
        self.coalition_task = nn.Linear(
            self.hidden_dim, self.reservation_dim, bias=False
        )
        self.coalition_capacity = nn.Linear(
            COALITION_CAPACITY_DIM, self.reservation_dim, bias=False
        )
        self.coalition_output = nn.Linear(self.reservation_dim, 1)
        self.stop_head = mlp(self.hidden_dim * 2, self.hidden_dim, 1)
        self.last_action_diagnostics: dict[str, Tensor] = {}

    @staticmethod
    def _gather_embeddings(values: Tensor, indexes: Tensor) -> Tensor:
        hidden = values.shape[-1]
        gathered = torch.gather(
            values,
            1,
            indexes.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden),
        )
        return gathered * (indexes >= 0).unsqueeze(-1).to(gathered.dtype)

    def _reservation_components(
        self,
        obs: dict[str, Tensor],
        encoded: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, CoalitionCandidates, Tensor, Tensor]:
        agents, tasks = self.encode(obs) if encoded is None else encoded
        candidates = build_coalition_candidates(
            obs, self.env_cfg, self.model_cfg.coalition_top_k
        )
        agv = self._gather_embeddings(
            self.coalition_agv(agents), candidates.agv
        )
        picker = self._gather_embeddings(
            self.coalition_picker(agents), candidates.picker
        )
        task = self._gather_embeddings(
            self.coalition_task(tasks), candidates.task
        )
        capacity = candidates.capacity_features
        if not self.use_capacity_features:
            capacity = torch.zeros_like(capacity)
        capacity = self.coalition_capacity(capacity)
        scores = self.coalition_output(
            torch.relu(agv + picker + task + capacity)
        ).squeeze(-1)
        scores = scores.masked_fill(~candidates.mask, -1e9)
        agent_pool = masked_mean(agents, obs["agent_mask"].bool(), dim=1)
        task_pool = masked_mean(tasks, obs["task_mask"].bool(), dim=1)
        stop = self.stop_head(torch.cat([agent_pool, task_pool], dim=-1)).squeeze(-1)
        value = self._decode(obs, agents, tasks).value
        return scores, stop, candidates, value, agents

    def reservation_outputs_with_incidence(
        self, obs: dict[str, Tensor], incidence: IncidenceBatch
    ) -> tuple[Tensor, Tensor, CoalitionCandidates, Tensor]:
        """CAHR outputs under a semantic-incidence intervention.

        This method is intentionally limited to CHORD policies, whose encoder
        exposes ``encode_with_incidence``. It supports post-hoc structural
        attribution without adding or training an explanation network.
        """

        if not hasattr(self, "encode_with_incidence"):
            raise TypeError("incidence interventions require an OpenHyper encoder")
        encoded = self.encode_with_incidence(obs, incidence)
        scores, stop, candidates, value, _ = self._reservation_components(
            obs, encoded=encoded
        )
        return scores, stop, candidates, value

    def act(
        self,
        obs: dict[str, Tensor],
        deterministic: bool = False,
        return_trace: bool = False,
    ):
        scores, stop, candidates, value, _ = self._reservation_components(obs)
        batch, count = candidates.mask.shape
        device = scores.device
        actions = torch.zeros(
            (batch, self.env_cfg.max_agents), dtype=torch.long, device=device
        )
        available = candidates.mask.clone()
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        joint_log_prob = scores.new_zeros(batch)
        entropy_sum = scores.new_zeros(batch)
        active_round_count = scores.new_zeros(batch)
        sequence = torch.zeros(
            (batch, self.reservation_rounds), dtype=torch.long, device=device
        )
        conditional_masks = torch.zeros(
            (batch, self.reservation_rounds, count + 1),
            dtype=torch.bool,
            device=device,
        )
        round_mask = torch.zeros(
            (batch, self.reservation_rounds), dtype=torch.bool, device=device
        )
        masked_total = torch.zeros(batch, dtype=torch.long, device=device)
        selected_total = torch.zeros(batch, dtype=torch.long, device=device)

        for decision_round in range(self.reservation_rounds):
            ongoing = ~finished
            current_mask = torch.cat(
                [torch.ones((batch, 1), dtype=torch.bool, device=device), available],
                dim=-1,
            )
            # Finished rows retain a valid STOP distribution but do not enter
            # the joint probability or entropy objective again.
            current_mask[finished, 1:] = False
            conditional_masks[:, decision_round] = current_mask
            round_mask[:, decision_round] = ongoing
            logits = torch.cat([stop.unsqueeze(-1), scores], dim=-1).masked_fill(
                ~current_mask, -1e9
            )
            distribution = Categorical(logits=logits)
            choice = logits.argmax(-1) if deterministic else distribution.sample()
            choice = torch.where(ongoing, choice, torch.zeros_like(choice))
            sequence[:, decision_round] = choice
            joint_log_prob += distribution.log_prob(choice) * ongoing.to(scores.dtype)
            option_count = current_mask.sum(-1).to(scores.dtype)
            normalized_entropy = distribution.entropy() / torch.log(
                option_count.clamp_min(2.0)
            )
            entropy_sum += normalized_entropy * ongoing.to(scores.dtype)
            active_round_count += ongoing.to(scores.dtype)

            for b in torch.nonzero(ongoing & (choice > 0), as_tuple=False).flatten().tolist():
                selected = int(choice[b].item() - 1)
                a = int(candidates.agv[b, selected].item())
                p = int(candidates.picker[b, selected].item())
                task = int(candidates.task[b, selected].item())
                if a >= 0:
                    actions[b, a] = task + 1
                if p >= 0:
                    actions[b, p] = task + 1
                conflict = candidates.task[b] == task
                if a >= 0:
                    conflict |= candidates.agv[b] == a
                if p >= 0:
                    conflict |= candidates.picker[b] == p
                removed = available[b] & conflict
                masked_total[b] += removed.sum()
                available[b] &= ~conflict
                selected_total[b] += 1
            finished |= choice == 0
            finished |= ~available.any(dim=-1)

        entropy = entropy_sum / active_round_count.clamp_min(1.0)
        self.last_action_diagnostics = {
            "candidate_coalitions": candidates.mask.sum(-1),
            "selected_coalitions": selected_total,
            "masked_coalitions": masked_total,
            "reservation_rounds": active_round_count,
        }
        trace = {
            "sequence": sequence,
            "conditional_masks": conditional_masks,
            "round_mask": round_mask,
        }
        result = (actions, joint_log_prob, entropy, value)
        return (*result, trace) if return_trace else result

    def evaluate_coalition_sequence(
        self,
        obs: dict[str, Tensor],
        sequence: Tensor,
        conditional_masks: Tensor,
        round_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scores, stop, _, value, _ = self._reservation_components(obs)
        base_logits = torch.cat([stop.unsqueeze(-1), scores], dim=-1)
        joint_log_prob = scores.new_zeros(scores.shape[0])
        entropy_sum = scores.new_zeros(scores.shape[0])
        rounds = round_mask.to(scores.dtype).sum(-1).clamp_min(1.0)
        for decision_round in range(sequence.shape[1]):
            mask = conditional_masks[:, decision_round].bool()
            logits = base_logits.masked_fill(~mask, -1e9)
            distribution = Categorical(logits=logits)
            active = round_mask[:, decision_round].to(scores.dtype)
            joint_log_prob += distribution.log_prob(sequence[:, decision_round]) * active
            option_count = mask.sum(-1).to(scores.dtype)
            entropy_sum += (
                distribution.entropy() / torch.log(option_count.clamp_min(2.0))
            ) * active
        return joint_log_prob, entropy_sum / rounds, value


class OpenHyperCoalitionPolicy(CoalitionReservationMixin, OpenHyperPolicy):
    """CHORD-CAHR with the full capacity-aware reservation decoder."""

    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig, model_cfg: ModelConfig):
        OpenHyperPolicy.__init__(self, env_cfg, train_cfg, model_cfg)
        self._init_reservation_decoder(model_cfg, use_capacity_features=True)


class OpenHyperNoCapacityCoalitionPolicy(CoalitionReservationMixin, OpenHyperPolicy):
    """Decoder ablation: reservations without capacity descriptors."""

    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig, model_cfg: ModelConfig):
        OpenHyperPolicy.__init__(self, env_cfg, train_cfg, model_cfg)
        self._init_reservation_decoder(model_cfg, use_capacity_features=False)


class EntityCoalitionPolicy(CoalitionReservationMixin, EntityPolicy):
    """DeepSets encoder with the identical CAHR decoder control."""

    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig, model_cfg: ModelConfig):
        EntityPolicy.__init__(self, env_cfg, train_cfg)
        self._init_reservation_decoder(model_cfg, use_capacity_features=True)


class GraphCoalitionPolicy(CoalitionReservationMixin, GraphPolicy):
    """2-section graph encoder with the identical CAHR decoder control."""

    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig, model_cfg: ModelConfig):
        GraphPolicy.__init__(self, env_cfg, train_cfg, model_cfg)
        self._init_reservation_decoder(model_cfg, use_capacity_features=True)


class PlatoPolicy(WarehousePolicy):
    """Faithful clean-room implementation of PLATO's Algorithm 1.

    The decentralized pointer actor consumes a focal-agent view, builds its
    query from permutation-invariant mean/variance/min/max agent statistics,
    and scores the current task set plus NOOP.  The centralized critic is the
    paper's two-layer GCN on the feasible agent--task bipartite graph with an
    input skip connection and mean pooling.  Only the observation adapter is
    warehouse-specific; the published learning mechanism is not changed.
    """

    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig):
        # Do not instantiate WarehousePolicy's generic pair scorer: unused
        # parameters would make the parameter-count comparison misleading.
        nn.Module.__init__(self)
        self.env_cfg = env_cfg
        self.hidden_dim = train_cfg.hidden_dim
        h = train_cfg.hidden_dim
        self.pointer_query = mlp(AGENT_FEATURE_DIM * 4, h, h)
        self.pointer_task_encoder = mlp(TASK_FEATURE_DIM, h, h)
        self.pointer_key = nn.Linear(h, h)
        self.pointer_query_projection = nn.Linear(h, h, bias=False)
        self.pointer_score = nn.Linear(h, 1, bias=False)
        # PLATO encodes NOOP with the same task MLP as ordinary tasks.  A
        # learnable task-shaped observation preserves that weight sharing.
        self.noop_observation = nn.Parameter(torch.zeros(TASK_FEATURE_DIM))

        self.critic_agent_input = mlp(AGENT_FEATURE_DIM, h, h)
        self.critic_task_input = mlp(TASK_FEATURE_DIM, h, h)
        self.critic_gcn = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(2)])
        self.plato_value_head = mlp(h, h, 1)

    def encode(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        # Kept for the common policy API.  PLATO's actor itself uses the
        # per-focal local tensors returned by _local_views.
        return self.critic_agent_input(obs["agent_features"]), self.critic_task_input(
            obs["task_features"]
        )

    @staticmethod
    def _four_statistics(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        count = weights.sum(dim=-2).clamp_min(1.0)
        mean = (values * weights).sum(dim=-2) / count
        variance = ((values - mean.unsqueeze(-2)).square() * weights).sum(dim=-2) / count
        positive = torch.finfo(values.dtype).max
        negative = torch.finfo(values.dtype).min
        minimum = values.masked_fill(~mask.unsqueeze(-1), positive).amin(dim=-2)
        maximum = values.masked_fill(~mask.unsqueeze(-1), negative).amax(dim=-2)
        has_entity = mask.any(dim=-1, keepdim=True)
        minimum = torch.where(has_entity, minimum, torch.zeros_like(minimum))
        maximum = torch.where(has_entity, maximum, torch.zeros_like(maximum))
        return torch.cat([mean, variance, minimum, maximum], dim=-1)

    @staticmethod
    def _local_views(obs: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Factor the global simulator state into PLATO focal-agent views."""
        agents = obs["agent_features"]
        tasks = obs["task_features"]
        batch, count, _ = agents.shape
        focal_xy = agents[:, :, None, 2:4]
        local_agents = agents[:, None].expand(-1, count, -1, -1).clone()
        local_agents[..., 2:4] -= focal_xy
        local_tasks = tasks[:, None].expand(-1, count, -1, -1).clone()
        local_tasks[..., 0:2] -= agents[:, :, None, 2:4]
        local_tasks[..., 2:4] -= agents[:, :, None, 2:4]
        local_agent_mask = obs["agent_mask"].bool()[:, None].expand(batch, count, -1)
        return local_agents, local_tasks, local_agent_mask

    def _critic(self, obs: dict[str, Tensor]) -> Tensor:
        agent_mask = obs["agent_mask"].bool()
        task_mask = obs["task_mask"].bool()
        h0 = torch.cat(
            [
                self.critic_agent_input(obs["agent_features"]),
                self.critic_task_input(obs["task_features"]),
            ],
            dim=1,
        )
        batch, num_agents = agent_mask.shape
        num_tasks = task_mask.shape[1]
        node_mask = torch.cat([agent_mask, task_mask], dim=1)
        adjacency = h0.new_zeros((batch, num_agents + num_tasks, num_agents + num_tasks))
        # PLATO includes an edge precisely when the agent has an action for the
        # task.  Inactive padding slots are therefore never graph vertices.
        feasible = obs["action_mask"][..., 1:].bool()
        feasible &= agent_mask.unsqueeze(2) & task_mask.unsqueeze(1)
        adjacency[:, :num_agents, num_agents:] = feasible.to(h0.dtype)
        adjacency[:, num_agents:, :num_agents] = feasible.transpose(1, 2).to(h0.dtype)
        diagonal = torch.arange(num_agents + num_tasks, device=h0.device)
        adjacency[:, diagonal, diagonal] = node_mask.to(h0.dtype)
        degree = adjacency.sum(-1).clamp_min(1.0)
        normalizer = degree.rsqrt()
        adjacency = normalizer.unsqueeze(-1) * adjacency * normalizer.unsqueeze(-2)
        hidden = h0
        for layer in self.critic_gcn:
            hidden = torch.relu(layer(torch.bmm(adjacency, hidden)))
            hidden = hidden * node_mask.unsqueeze(-1)
        hidden = (hidden + h0) * node_mask.unsqueeze(-1)
        pooled = hidden.sum(1) / node_mask.sum(1, keepdim=True).to(hidden.dtype).clamp_min(1.0)
        return self.plato_value_head(pooled).squeeze(-1)

    def forward(self, obs: dict[str, Tensor]) -> PolicyOutput:
        local_agents, local_tasks, local_agent_mask = self._local_views(obs)
        query = self.pointer_query_projection(
            self.pointer_query(self._four_statistics(local_agents, local_agent_mask))
        )
        task_keys = self.pointer_key(self.pointer_task_encoder(local_tasks))
        task_logits = self.pointer_score(
            torch.tanh(query.unsqueeze(2) + task_keys)
        ).squeeze(-1)
        noop_encoded = self.pointer_task_encoder(self.noop_observation.view(1, 1, -1))
        noop_key = self.pointer_key(noop_encoded)
        noop_logits = self.pointer_score(
            torch.tanh(query + noop_key.expand(query.shape[0], query.shape[1], -1))
        )
        logits = torch.cat([noop_logits, task_logits], dim=-1)
        logits = logits.masked_fill(~obs["action_mask"].bool(), -1e9)
        decision_mask = obs["agent_mask"].bool() & (obs["action_mask"].sum(dim=-1) > 1)
        return PolicyOutput(logits, self._critic(obs), decision_mask)


def build_policy(
    name: str, env_cfg: EnvConfig, train_cfg: TrainConfig, model_cfg: ModelConfig
) -> nn.Module:
    def official_mohito():
        from .mohito_official import build_official_mohito_policy

        return build_official_mohito_policy(env_cfg, train_cfg, model_cfg)

    factories = {
        "chord_mappo": lambda: OpenHyperCoalitionPolicy(
            env_cfg, train_cfg, model_cfg
        ),
        # Decoder/capacity ablations. Together with chord_mappo these form the
        # preregistered four-row CAHR mechanism table.
        "chord_no_reservation": lambda: OpenHyperPolicy(
            env_cfg, train_cfg, model_cfg
        ),
        "chord_no_capacity": lambda: OpenHyperNoCapacityCoalitionPolicy(
            env_cfg, train_cfg, model_cfg
        ),
        "chord_capacity_factorized": lambda: CapacityAwareOpenHyperPolicy(
            env_cfg, train_cfg, model_cfg
        ),
        "entity_mappo": lambda: EntityPolicy(env_cfg, train_cfg),
        "graph_mappo": lambda: GraphPolicy(env_cfg, train_cfg, model_cfg),
        # Encoder controls with exactly the same CAHR action decoder.
        "entity_cahr": lambda: EntityCoalitionPolicy(
            env_cfg, train_cfg, model_cfg
        ),
        "graph_cahr": lambda: GraphCoalitionPolicy(
            env_cfg, train_cfg, model_cfg
        ),
        "plato_faithful": lambda: PlatoPolicy(env_cfg, train_cfg),
        "mohito_official": official_mohito,
    }
    if name not in factories:
        raise KeyError(f"unknown learned algorithm {name!r}; choose from {sorted(factories)}")
    return factories[name]()
