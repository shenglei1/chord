from __future__ import annotations

import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import numpy as np
import torch

from open_hyper_warehouse.buffer import to_torch
from open_hyper_warehouse.config import EnvConfig, ModelConfig, TrainConfig
from open_hyper_warehouse.env import OpenWarehouseEnv
from open_hyper_warehouse.models import OpenHyperPolicy, build_policy


def permute_observation(obs, agent_perm, task_perm):
    result = {
        "agent_features": obs["agent_features"][agent_perm],
        "task_features": obs["task_features"][task_perm],
        "agent_mask": obs["agent_mask"][agent_perm],
        "task_mask": obs["task_mask"][task_perm],
    }
    result["action_mask"] = obs["action_mask"][agent_perm]
    result["action_mask"] = np.concatenate(
        [result["action_mask"][:, :1], result["action_mask"][:, 1:][:, task_perm]], axis=1
    )
    return result


class EquivarianceTests(unittest.TestCase):
    def test_open_hyper_policy_is_type_permutation_equivariant(self):
        cfg = EnvConfig(
            max_agvs=4, max_pickers=3, max_tasks=8,
            initial_agvs_min=4, initial_agvs_max=4,
            initial_pickers_min=3, initial_pickers_max=3,
            initial_tasks_min=6, initial_tasks_max=6,
            deadline_min=20, deadline_max=30, num_zones=2,
        )
        obs = OpenWarehouseEnv(cfg, 12).reset(12)
        agent_perm = np.array([2, 0, 3, 1, 6, 4, 5])
        task_perm = np.array([3, 0, 5, 1, 2, 4, 7, 6])
        permuted = permute_observation(obs, agent_perm, task_perm)
        torch.manual_seed(0)
        policy = OpenHyperPolicy(cfg, TrainConfig(hidden_dim=32, encoder_layers=2), ModelConfig()).eval()
        with torch.no_grad():
            original_logits = policy(to_torch({k: v[None] for k, v in obs.items()}, torch.device("cpu"))).logits[0]
            permuted_logits = policy(
                to_torch({k: v[None] for k, v in permuted.items()}, torch.device("cpu"))
            ).logits[0]
        expected = original_logits[agent_perm]
        expected = torch.cat([expected[:, :1], expected[:, 1:][:, task_perm]], dim=1)
        valid = torch.as_tensor(permuted["action_mask"])
        self.assertTrue(torch.allclose(permuted_logits[valid], expected[valid], atol=1e-5, rtol=1e-5))

    def test_cahr_deterministic_coalition_set_is_type_equivariant(self):
        cfg = EnvConfig(
            max_agvs=4, max_pickers=3, max_tasks=8,
            initial_agvs_min=4, initial_agvs_max=4,
            initial_pickers_min=3, initial_pickers_max=3,
            initial_tasks_min=6, initial_tasks_max=6,
            deadline_min=20, deadline_max=30, num_zones=2,
        )
        obs = OpenWarehouseEnv(cfg, 19).reset(19)
        # Remove exact-distance depot ties. Under such ties the corresponding
        # robots are physically exchangeable and deterministic argmax may pick
        # a different but equivalent representative after relabelling.
        obs["agent_features"][:, 2] = np.linspace(0.05, 0.65, cfg.max_agents)
        agent_perm = np.array([2, 0, 3, 1, 6, 4, 5])
        task_perm = np.array([3, 0, 5, 1, 2, 4, 7, 6])
        permuted = permute_observation(obs, agent_perm, task_perm)
        torch.manual_seed(4)
        policy = build_policy(
            "chord_mappo", cfg, TrainConfig(hidden_dim=32, encoder_layers=2),
            ModelConfig(coalition_top_k=3),
        ).eval()
        with torch.no_grad():
            policy.stop_head[-1].bias.fill_(-100.0)
            policy.coalition_output.bias.fill_(100.0)
            original = policy.act(
                to_torch({k: v[None] for k, v in obs.items()}, torch.device("cpu")),
                deterministic=True,
            )[0][0]
            changed = policy.act(
                to_torch({k: v[None] for k, v in permuted.items()}, torch.device("cpu")),
                deterministic=True,
            )[0][0]
        inverse_tasks = np.argsort(task_perm)
        expected = original[torch.as_tensor(agent_perm)].clone()
        positive = expected > 0
        expected[positive] = torch.as_tensor(inverse_tasks, dtype=torch.long)[
            expected[positive] - 1
        ] + 1
        self.assertTrue(torch.equal(changed, expected))


if __name__ == "__main__":
    unittest.main()
