from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor


@dataclass
class RolloutBuffer:
    observations: dict[str, list[np.ndarray]] = field(default_factory=dict)
    actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    rewards: list[np.ndarray] = field(default_factory=list)
    dones: list[np.ndarray] = field(default_factory=list)
    coalition_traces: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(
        self,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        coalition_trace: dict[str, np.ndarray] | None = None,
    ) -> None:
        for key, value in obs.items():
            self.observations.setdefault(key, []).append(value.copy())
        self.actions.append(actions.copy())
        self.log_probs.append(log_probs.copy())
        self.values.append(values.copy())
        self.rewards.append(rewards.copy())
        self.dones.append(dones.copy())
        if coalition_trace is not None:
            for key, value in coalition_trace.items():
                self.coalition_traces.setdefault(key, []).append(value.copy())

    def compute_gae(
        self, next_values: np.ndarray, gamma: float, gae_lambda: float
    ) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
        for step in reversed(range(rewards.shape[0])):
            following = next_values if step == rewards.shape[0] - 1 else values[step + 1]
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + gamma * following * nonterminal - values[step]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values
        return advantages, returns

    def tensors(
        self, advantages: np.ndarray, returns: np.ndarray, device: torch.device
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        time_steps, num_envs = advantages.shape
        flat_obs = {
            key: torch.as_tensor(np.asarray(values), device=device).reshape(
                time_steps * num_envs, *np.asarray(values).shape[2:]
            )
            for key, values in self.observations.items()
        }
        action_array = np.asarray(self.actions)
        log_prob_array = np.asarray(self.log_probs)
        output: dict[str, Tensor | dict[str, Tensor]] = {
            "obs": flat_obs,
            "actions": torch.as_tensor(action_array, dtype=torch.long, device=device).reshape(
                time_steps * num_envs, *action_array.shape[2:]
            ),
            "old_log_probs": torch.as_tensor(
                log_prob_array, dtype=torch.float32, device=device
            ).reshape(time_steps * num_envs, *log_prob_array.shape[2:]),
            "old_values": torch.as_tensor(np.asarray(self.values), dtype=torch.float32, device=device).reshape(-1),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=device).reshape(-1),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=device).reshape(-1),
        }
        if self.coalition_traces:
            output["coalition_trace"] = {
                key: torch.as_tensor(np.asarray(values), device=device).reshape(
                    time_steps * num_envs, *np.asarray(values).shape[2:]
                )
                for key, values in self.coalition_traces.items()
            }
        return output


def stack_observations(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}


def to_torch(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    return {key: torch.as_tensor(value, device=device) for key, value in obs.items()}
