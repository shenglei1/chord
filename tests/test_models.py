from __future__ import annotations

import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from open_hyper_warehouse.buffer import to_torch
from open_hyper_warehouse.config import EnvConfig, ModelConfig, TrainConfig
from open_hyper_warehouse.env import OpenWarehouseEnv
from open_hyper_warehouse.models import build_policy


class ModelTests(unittest.TestCase):
    def test_all_controlled_baselines_share_interface(self):
        env_cfg = EnvConfig(
            max_agvs=4, max_pickers=2, max_tasks=8,
            initial_agvs_min=3, initial_agvs_max=3,
            initial_pickers_min=2, initial_pickers_max=2,
            initial_tasks_min=4, initial_tasks_max=4,
            deadline_min=20, deadline_max=30, num_zones=2,
        )
        obs = OpenWarehouseEnv(env_cfg, 0).reset(0)
        batch = to_torch({key: value[None] for key, value in obs.items()}, torch.device("cpu"))
        names = [
            "chord_mappo", "entity_cahr", "graph_cahr",
            "chord_no_capacity", "chord_no_reservation", "chord_capacity_factorized",
            "entity_mappo", "graph_mappo", "plato_faithful",
        ]
        for name in names:
            with self.subTest(name=name):
                torch.manual_seed(0)
                policy = build_policy(
                    name, env_cfg, TrainConfig(hidden_dim=32, encoder_layers=1), ModelConfig(num_groups=2)
                )
                output = policy(batch)
                self.assertEqual(tuple(output.logits.shape), (1, 6, 9))
                self.assertEqual(tuple(output.value.shape), (1,))
                valid = batch["action_mask"].bool()
                self.assertTrue(torch.isfinite(output.logits[valid]).all())
                self.assertTrue((output.logits[~valid] < -1e8).all())
                actions, log_prob, entropy, value = policy.act(batch)
                self.assertEqual(tuple(actions.shape), (1, 6))
                expected = (1,) if getattr(policy, "uses_coalition_reservation", False) else (1, 6)
                self.assertEqual(tuple(log_prob.shape), expected)
                self.assertEqual(tuple(entropy.shape), expected)
                self.assertTrue(torch.isfinite(log_prob).all())
                self.assertTrue(torch.isfinite(entropy).all())
                self.assertTrue(torch.isfinite(value).all())

    def test_factorized_distribution_is_normalized_and_gate_has_task_semantics(self):
        env_cfg = EnvConfig(
            max_agvs=2, max_pickers=1, max_tasks=4,
            initial_agvs_min=2, initial_agvs_max=2,
            initial_pickers_min=1, initial_pickers_max=1,
            initial_tasks_min=3, initial_tasks_max=3,
        )
        obs = OpenWarehouseEnv(env_cfg, 5).reset(5)
        batch = to_torch({key: value[None] for key, value in obs.items()}, torch.device("cpu"))
        policy = build_policy(
            "chord_no_reservation", env_cfg, TrainConfig(hidden_dim=16, encoder_layers=1), ModelConfig()
        )
        output = policy(batch)
        probabilities = torch.softmax(output.logits, dim=-1)
        self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[..., 0])))
        deterministic = policy.act(batch, deterministic=True)[0]
        self.assertTrue(batch["action_mask"].gather(-1, deterministic.unsqueeze(-1)).all())

    def test_cahr_trace_reproduces_joint_probability_and_is_conflict_free(self):
        env_cfg = EnvConfig(
            max_agvs=3, max_pickers=2, max_tasks=6,
            initial_agvs_min=3, initial_agvs_max=3,
            initial_pickers_min=2, initial_pickers_max=2,
            initial_tasks_min=4, initial_tasks_max=4,
        )
        obs = OpenWarehouseEnv(env_cfg, 13).reset(13)
        batch = to_torch({key: value[None] for key, value in obs.items()}, torch.device("cpu"))
        policy = build_policy(
            "chord_mappo", env_cfg, TrainConfig(hidden_dim=16, encoder_layers=1),
            ModelConfig(coalition_top_k=2),
        )
        with torch.no_grad():
            policy.stop_head[-1].bias.fill_(-100.0)
            policy.coalition_output.bias.fill_(100.0)
        actions, old_log_prob, _, _, trace = policy.act(
            batch, deterministic=True, return_trace=True
        )
        new_log_prob, entropy, _ = policy.evaluate_coalition_sequence(
            batch, trace["sequence"], trace["conditional_masks"], trace["round_mask"]
        )
        self.assertTrue(torch.allclose(old_log_prob, new_log_prob, atol=1e-6))
        self.assertTrue(torch.isfinite(entropy).all())
        chosen = actions[0]
        self.assertGreater(int((chosen > 0).sum()), 0)
        for task_action in torch.unique(chosen[chosen > 0]).tolist():
            slots = torch.nonzero(chosen == task_action, as_tuple=False).flatten()
            roles = batch["agent_features"][0, slots, :2].argmax(-1)
            self.assertEqual(int((roles == 0).sum()), 1)
            self.assertEqual(int((roles == 1).sum()), 1)


if __name__ == "__main__":
    unittest.main()
