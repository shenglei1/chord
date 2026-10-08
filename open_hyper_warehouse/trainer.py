from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .buffer import RolloutBuffer, stack_observations, to_torch
from .baselines import baseline_metadata
from .config import ExperimentConfig
from .env import OpenWarehouseEnv
from .models import WarehousePolicy, build_policy
from .telemetry import TrainingTelemetry, write_training_log
from .vector_env import make_vector_env


@dataclass(slots=True)
class TrainingResult:
    checkpoint: Path
    steps: int
    updates: int
    wall_seconds: float
    peak_memory_mb: float | None


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class PPOTrainer:
    def __init__(
        self,
        algorithm: str,
        config: ExperimentConfig,
        seed: int,
        output_dir: str | Path,
    ):
        set_seed(seed)
        self.algorithm = algorithm
        self.cfg = config
        self.seed = seed
        self.device = resolve_device(config.train.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.policy: WarehousePolicy = build_policy(
            algorithm, config.env, config.train, config.model
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.train.learning_rate)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_counts = np.zeros(config.train.num_envs, dtype=np.int64)
        initial_seeds = [self._episode_seed(i) for i in range(config.train.num_envs)]
        self.vector_env = make_vector_env(
            config.env,
            config.train.num_envs,
            config.train.parallel_env_workers,
            initial_seeds,
        )
        self.observations = self.vector_env.reset_all(initial_seeds)
        self.logs: list[dict[str, float | int]] = []
        self.telemetry = TrainingTelemetry(
            self.output_dir,
            {
                "algorithm": algorithm,
                "training_seed": seed,
                "device": str(self.device),
                "environment_backend": self.vector_env.backend,
                "num_envs": config.train.num_envs,
                "parallel_env_workers": config.train.parallel_env_workers,
                "total_steps": config.train.total_steps,
            },
        )

    def _episode_seed(self, env_index: int) -> int:
        return self.seed * 10_000_000 + env_index * 100_000 + int(self.episode_counts[env_index])

    def train(self) -> TrainingResult:
        started = time.perf_counter()
        steps = 0
        updates = 0
        next_checkpoint = self.cfg.train.checkpoint_interval
        while steps < self.cfg.train.total_steps:
            remaining_vector_steps = (self.cfg.train.total_steps - steps) // self.cfg.train.num_envs
            if remaining_vector_steps <= 0:
                break
            rollout_steps = min(self.cfg.train.rollout_steps, remaining_vector_steps)
            buffer, completed = self._collect_rollout(rollout_steps)
            steps += rollout_steps * self.cfg.train.num_envs
            updates += 1
            entropy_coef = self._entropy_coefficient(steps)
            losses = self._update(buffer, entropy_coef)
            row: dict[str, float | int] = {"steps": steps, "updates": updates, **losses}
            if completed:
                for metric in (
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
                    row[f"episode_{metric}"] = float(
                        np.mean([m.get(metric, 0.0) for m in completed])
                    )
            row = self.telemetry.record(row, self.device)
            self.logs.append(row)
            if steps >= next_checkpoint:
                self.save(self.output_dir / f"step_{steps}.pt", steps, updates)
                write_training_log(self.output_dir, self.logs)
                next_checkpoint += self.cfg.train.checkpoint_interval
            if updates == 1 or updates % 10 == 0:
                print(json.dumps(row, ensure_ascii=False))
        checkpoint = self.output_dir / "final.pt"
        wall = time.perf_counter() - started
        peak_memory_mb = (
            float(torch.cuda.max_memory_allocated(self.device) / (1024**2))
            if self.device.type == "cuda"
            else None
        )
        self.save(checkpoint, steps, updates, wall, peak_memory_mb)
        write_training_log(self.output_dir, self.logs)
        return TrainingResult(checkpoint, steps, updates, wall, peak_memory_mb)

    def _collect_rollout(self, rollout_steps: int) -> tuple[RolloutBuffer, list[dict[str, float]]]:
        buffer = RolloutBuffer()
        completed: list[dict[str, float]] = []
        reservation_diagnostics: dict[str, list[float]] = {}
        for _ in range(rollout_steps):
            stacked = stack_observations(self.observations)
            with torch.no_grad():
                torch_obs = to_torch(stacked, self.device)
                if getattr(self.policy, "uses_coalition_reservation", False):
                    actions, log_probs, _, values, trace = self.policy.act(
                        torch_obs, return_trace=True
                    )
                    np_trace = {
                        key: value.cpu().numpy() for key, value in trace.items()
                    }
                    for key, value in self.policy.last_action_diagnostics.items():
                        reservation_diagnostics.setdefault(f"{key}_per_step", []).append(
                            float(value.to(torch.float32).mean().cpu())
                        )
                else:
                    actions, log_probs, _, values = self.policy.act(torch_obs)
                    np_trace = None
            np_actions = actions.cpu().numpy()
            rewards = np.zeros(len(self.observations), dtype=np.float32)
            dones = np.zeros(len(self.observations), dtype=np.bool_)
            transitions = self.vector_env.step(np_actions)
            next_observations: list[dict[str, np.ndarray]] = []
            reset_indexes: list[int] = []
            reset_seeds: list[int] = []
            for i, (obs, reward, done, info) in enumerate(transitions):
                rewards[i], dones[i] = reward, done
                if done:
                    completed.append(info)
                    self.episode_counts[i] += 1
                    reset_indexes.append(i)
                    reset_seeds.append(self._episode_seed(i))
                next_observations.append(obs)
            for index, obs in self.vector_env.reset_at(reset_indexes, reset_seeds).items():
                next_observations[index] = obs
            buffer.add(
                stacked,
                np_actions,
                log_probs.cpu().numpy(),
                values.cpu().numpy(),
                rewards,
                dones,
                coalition_trace=np_trace,
            )
            self.observations = next_observations
        with torch.no_grad():
            next_values = self.policy(to_torch(stack_observations(self.observations), self.device)).value
        advantages, returns = buffer.compute_gae(
            next_values.cpu().numpy(), self.cfg.train.gamma, self.cfg.train.gae_lambda
        )
        buffer._advantages = advantages
        buffer._returns = returns
        buffer._reservation_metrics = {
            key: float(np.mean(values))
            for key, values in reservation_diagnostics.items()
        }
        return buffer, completed

    def _entropy_coefficient(self, steps: int) -> float:
        anneal_steps = max(
            1.0,
            self.cfg.train.total_steps * self.cfg.train.entropy_anneal_fraction,
        )
        progress = min(max(steps / anneal_steps, 0.0), 1.0)
        start = self.cfg.train.entropy_coef
        end = self.cfg.train.entropy_coef_final
        return float(start + progress * (end - start))

    @staticmethod
    def _reduce_agent_terms(
        terms: torch.Tensor,
        decision_mask: torch.Tensor,
        population_normalized: bool,
    ) -> torch.Tensor:
        weights = decision_mask.to(terms.dtype)
        per_state = (terms * weights).sum(dim=-1)
        if population_normalized:
            per_state = per_state / weights.sum(dim=-1).clamp_min(1.0)
        valid_states = decision_mask.any(dim=-1)
        if not valid_states.any():
            return terms.sum() * 0.0
        return per_state[valid_states].mean()

    def _update(self, buffer: RolloutBuffer, entropy_coef: float) -> dict[str, float]:
        data = buffer.tensors(buffer._advantages, buffer._returns, self.device)
        advantages = data["advantages"]
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        data["advantages"] = advantages
        size = advantages.shape[0]
        accumulated = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "policy_entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        batches = 0
        population_normalized = bool(
            getattr(self.policy, "population_normalized_actor_terms", True)
        )
        uses_reservation = bool(
            getattr(self.policy, "uses_coalition_reservation", False)
        )
        for _ in range(self.cfg.train.update_epochs):
            permutation = torch.randperm(size, device=self.device)
            for start in range(0, size, self.cfg.train.minibatch_size):
                index = permutation[start : start + self.cfg.train.minibatch_size]
                obs = {key: value[index] for key, value in data["obs"].items()}
                if uses_reservation:
                    trace = data["coalition_trace"]
                    new_log_prob, entropy_mean_per_state, values = (
                        self.policy.evaluate_coalition_sequence(
                            obs,
                            trace["sequence"][index],
                            trace["conditional_masks"][index],
                            trace["round_mask"][index],
                        )
                    )
                    log_ratio = new_log_prob - data["old_log_probs"][index]
                    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                    state_advantages = data["advantages"][index]
                    unclipped = ratio * state_advantages
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.cfg.train.ppo_clip,
                        1.0 + self.cfg.train.ppo_clip,
                    ) * state_advantages
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    entropy_mean = entropy_mean_per_state.mean()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.cfg.train.ppo_clip
                    ).to(ratio.dtype).mean()
                else:
                    new_log_prob, entropy, values, decision_mask = self.policy.evaluate_actions(
                        obs, data["actions"][index]
                    )
                    log_ratio = new_log_prob - data["old_log_probs"][index]
                    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                    agent_advantages = data["advantages"][index].unsqueeze(-1)
                    unclipped = ratio * agent_advantages
                    clipped = torch.clamp(
                        ratio, 1.0 - self.cfg.train.ppo_clip, 1.0 + self.cfg.train.ppo_clip
                    ) * agent_advantages
                    policy_loss = -self._reduce_agent_terms(
                        torch.minimum(unclipped, clipped),
                        decision_mask,
                        population_normalized,
                    )
                    action_count = obs["action_mask"].sum(dim=-1).to(entropy.dtype)
                    normalized_entropy = entropy / torch.log(action_count.clamp_min(2.0))
                    entropy_mean = self._reduce_agent_terms(
                        normalized_entropy,
                        decision_mask,
                        population_normalized,
                    )
                    approx_kl = self._reduce_agent_terms(
                        (ratio - 1.0) - log_ratio,
                        decision_mask,
                        population_normalized=True,
                    )
                    clip_fraction = self._reduce_agent_terms(
                        ((ratio - 1.0).abs() > self.cfg.train.ppo_clip).to(ratio.dtype),
                        decision_mask,
                        population_normalized=True,
                    )
                value_loss = 0.5 * (values - data["returns"][index]).pow(2).mean()
                loss = (
                    policy_loss
                    + self.cfg.train.value_coef * value_loss
                    - entropy_coef * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.max_grad_norm)
                self.optimizer.step()
                accumulated["policy_loss"] += float(policy_loss.detach())
                accumulated["value_loss"] += float(value_loss.detach())
                accumulated["policy_entropy"] += float(entropy_mean.detach())
                accumulated["approx_kl"] += float(approx_kl.detach())
                accumulated["clip_fraction"] += float(clip_fraction.detach())
                accumulated["grad_norm"] += float(grad_norm)
                batches += 1
        output = {key: value / max(batches, 1) for key, value in accumulated.items()}
        returns = data["returns"]
        residual = returns - data["old_values"]
        target_variance = torch.var(returns, unbiased=False)
        output["explained_variance_before_update"] = float(
            (1.0 - torch.var(residual, unbiased=False) / target_variance.clamp_min(1e-8)).detach()
        )
        if uses_reservation:
            trace = data["coalition_trace"]
            selected = (trace["sequence"] > 0) & trace["round_mask"].bool()
            output["reservation_stop_rate"] = float(
                (trace["sequence"][:, 0] == 0).to(torch.float32).mean().detach()
            )
            output["selected_coalitions_per_step"] = float(
                selected.sum(-1).to(torch.float32).mean().detach()
            )
            output.update(getattr(buffer, "_reservation_metrics", {}))
        else:
            decision_mask = data["obs"]["agent_mask"].bool() & (
                data["obs"]["action_mask"].sum(dim=-1) > 1
            )
            decision_count = decision_mask.sum().clamp_min(1)
            output["noop_rate"] = float(
                (((data["actions"] == 0) & decision_mask).sum() / decision_count).detach()
            )
            output["mean_decision_agents"] = float(
                decision_mask.sum(dim=-1).to(torch.float32).mean().detach()
            )
        output["entropy_coefficient"] = entropy_coef
        return output

    def save(
        self,
        path: Path,
        steps: int,
        updates: int,
        wall_seconds: float | None = None,
        peak_memory_mb: float | None = None,
    ) -> None:
        torch.save(
            {
                "algorithm": self.algorithm,
                "config": self.cfg.to_dict(),
                "seed": self.seed,
                "steps": steps,
                "updates": updates,
                "wall_seconds": wall_seconds,
                "peak_memory_mb": peak_memory_mb,
                "training_device": str(self.device),
                "torch_version": torch.__version__,
                "cuda_device_name": (
                    torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None
                ),
                "state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "parameter_count": self.policy.parameter_count(),
                "method_metadata": baseline_metadata(self.algorithm),
                "environment_backend": self.vector_env.backend,
                "parallel_env_workers": self.cfg.train.parallel_env_workers,
                "learner_revision": (
                    "cahr-joint-conditional-ppo-v1"
                    if getattr(self.policy, "uses_coalition_reservation", False)
                    else "agentwise-ppo-cardinality-critic-feasible-coalitions-v2"
                ),
                "action_decoder": (
                    "capacity-aware-autoregressive-hyperedge-reservation"
                    if getattr(self.policy, "uses_coalition_reservation", False)
                    else "independent-factorized-pointer"
                ),
            },
            path,
        )

    def close(self) -> None:
        self.vector_env.close()


def train_algorithm(
    algorithm: str,
    config: ExperimentConfig,
    seed: int,
    output_root: str | Path = "results",
) -> TrainingResult:
    output = Path(output_root) / algorithm / f"seed_{seed}"
    if algorithm == "mohito_official":
        # MOHITO is an off-policy deterministic actor--critic with replay and
        # target networks.  Routing it through the common PPO trainer would be
        # a mechanism-changing adaptation, not a valid baseline reproduction.
        from .mohito_official import OfficialMohitoTrainer

        trainer = OfficialMohitoTrainer(config, seed, output)
    else:
        trainer = PPOTrainer(algorithm, config, seed, output)
    try:
        return trainer.train()
    finally:
        trainer.close()
