from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from open_hyper_warehouse.config import EnvConfig
from open_hyper_warehouse.env import OpenWarehouseEnv
from open_hyper_warehouse.heuristics import closest_feasible_actions, rolling_horizon_actions
from open_hyper_warehouse.scenarios import SCENARIOS, make_scenario


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig(
            width=10,
            height=8,
            max_agvs=5,
            max_pickers=3,
            max_tasks=10,
            initial_agvs_min=3,
            initial_agvs_max=3,
            initial_pickers_min=2,
            initial_pickers_max=2,
            initial_tasks_min=4,
            initial_tasks_max=4,
            episode_steps=20,
            deadline_min=12,
            deadline_max=20,
            num_zones=2,
        )

    def test_shapes_and_masks(self):
        env = OpenWarehouseEnv(self.cfg, seed=3)
        obs = env.reset(seed=3)
        self.assertEqual(obs["agent_features"].shape, (8, 12))
        self.assertEqual(obs["task_features"].shape, (10, 10))
        self.assertEqual(obs["action_mask"].shape, (8, 11))
        self.assertTrue(obs["action_mask"][:, 0].all())
        self.assertFalse(obs["action_mask"][~obs["agent_mask"], 1:].any())

    def test_seeded_trajectories_are_identical(self):
        env_a, env_b = OpenWarehouseEnv(self.cfg, 11), OpenWarehouseEnv(self.cfg, 11)
        obs_a, obs_b = env_a.reset(11), env_b.reset(11)
        for _ in range(self.cfg.episode_steps):
            actions_a = closest_feasible_actions(obs_a)
            actions_b = closest_feasible_actions(obs_b)
            np.testing.assert_array_equal(actions_a, actions_b)
            obs_a, reward_a, done_a, info_a = env_a.step(actions_a)
            obs_b, reward_b, done_b, info_b = env_b.step(actions_b)
            self.assertEqual(reward_a, reward_b)
            self.assertEqual(done_a, done_b)
        self.assertEqual(info_a, info_b)

    def test_order_event_tape_is_policy_independent(self):
        cfg = replace(self.cfg, order_arrival_rate=0.8, order_cancel_probability=0.1)
        env_idle, env_dispatch = OpenWarehouseEnv(cfg, 17), OpenWarehouseEnv(cfg, 17)
        obs_idle, obs_dispatch = env_idle.reset(17), env_dispatch.reset(17)
        for _ in range(cfg.episode_steps):
            obs_idle, _, _, info_idle = env_idle.step(
                np.zeros(cfg.max_agents, dtype=np.int64)
            )
            obs_dispatch, _, _, info_dispatch = env_dispatch.step(
                rolling_horizon_actions(obs_dispatch)
            )
        self.assertEqual(info_idle["orders_offered"], info_dispatch["orders_offered"])

    def test_open_population_changes_inside_episode(self):
        cfg = replace(self.cfg, agent_join_probability=1.0, agent_leave_probability=0.0)
        env = OpenWarehouseEnv(cfg, 5)
        obs = env.reset(5)
        initial = int(obs["agent_mask"].sum())
        for _ in range(4):
            obs, _, _, info = env.step(np.zeros(cfg.max_agents, dtype=np.int64))
        self.assertGreater(int(obs["agent_mask"].sum()), initial)
        self.assertGreater(info["agent_joins"], 0)

    def test_invalid_action_is_rejected(self):
        env = OpenWarehouseEnv(self.cfg, 7)
        obs = env.reset(7)
        inactive = int(np.flatnonzero(~obs["agent_mask"])[0])
        actions = np.zeros(self.cfg.max_agents, dtype=np.int64)
        actions[inactive] = 1
        with self.assertRaises(ValueError):
            env.step(actions)

    def test_scheduled_disruptions_and_operational_metrics(self):
        cfg = replace(
            self.cfg,
            scheduled_breakdown_step=1,
            scheduled_breakdown_count=1,
            scheduled_repair_step=3,
            scheduled_repair_count=1,
            picker_shift_step=2,
            picker_shift_leave_count=1,
            picker_shift_join_count=1,
        )
        env = OpenWarehouseEnv(cfg, 9)
        obs = env.reset(9)
        for _ in range(4):
            obs, _, _, info = env.step(rolling_horizon_actions(obs))
        self.assertEqual(info["forced_failures"], 1.0)
        self.assertEqual(info["repairs"], 1.0)
        self.assertEqual(info["shift_changes"], 1.0)
        self.assertIn("travel_per_delivery", info)
        self.assertIn("mean_flow_time", info)
        self.assertIn("service_level", info)
        self.assertIn("throughput_per_100_steps", info)
        self.assertIn("deadline_miss_rate", info)

    def test_all_application_scenarios_validate(self):
        for name in SCENARIOS:
            with self.subTest(name=name):
                make_scenario(self.cfg, name).validate()


if __name__ == "__main__":
    unittest.main()
