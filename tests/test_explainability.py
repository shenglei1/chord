from __future__ import annotations

import torch

from open_hyper_warehouse.buffer import to_torch
from open_hyper_warehouse.config import EnvConfig, ModelConfig, TrainConfig
from open_hyper_warehouse.env import OpenWarehouseEnv
from open_hyper_warehouse.explainability import semantic_edge_attribution
from open_hyper_warehouse.models import OpenHyperPolicy, build_policy


def test_breakdown_capture_preserves_event_timing_and_order_identity():
    from open_hyper_warehouse.config import ExperimentConfig
    from open_hyper_warehouse.explainability import capture_breakdown_episodes

    cfg = ExperimentConfig()
    cfg.env.episode_steps = 12
    cfg.env.agent_join_probability = 0.0

    class IdlePolicy:
        def act(self, obs, deterministic=False):
            return (torch.zeros_like(obs["agent_mask"], dtype=torch.long),)

    result = capture_breakdown_episodes(IdlePolicy(), cfg, torch.device("cpu"), 1)[0]
    scenes = result["scene_states"]
    assert [scenes[e]["time"] for e in ("pre_breakdown", "post_breakdown", "post_repair")] == [4, 5, 9]
    assert len(result["timeline"]) == 12
    assert scenes["pre_breakdown"]["metrics"]["forced_failures"] == 0
    assert scenes["post_breakdown"]["metrics"]["forced_failures"] > 0
    assert scenes["post_repair"]["metrics"]["repairs"] > 0
    for scene in scenes.values():
        assert all("order_id" in task and "slot" in task for task in scene["tasks"])
        assert all("joined_at" in agent for agent in scene["agents"])


def test_explanation_exports_scene_identity_and_timeline():
    import csv
    import json
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from open_hyper_warehouse.config import ExperimentConfig
    from open_hyper_warehouse.explainability import explain_checkpoint, write_explanation

    cfg = ExperimentConfig()
    cfg.env = EnvConfig(
        max_agvs=3, max_pickers=2, max_tasks=6,
        initial_agvs_min=3, initial_agvs_max=3,
        initial_pickers_min=2, initial_pickers_max=2,
        initial_tasks_min=4, initial_tasks_max=4, episode_steps=12,
    )
    cfg.train.hidden_dim = 16
    cfg.train.encoder_layers = 1
    cfg.model.coalition_top_k = 2
    policy = build_policy("chord_mappo", cfg.env, cfg.train, cfg.model).eval()
    metadata = {"algorithm": "chord_mappo", "seed": 1}
    with patch("open_hyper_warehouse.explainability.load_policy",
               return_value=(policy, cfg, metadata, torch.device("cpu"))):
        summary, rows = explain_checkpoint("synthetic-test-only.pt", episodes=3, device="cpu")
    assert all("order_ids" in row for row in rows)
    with tempfile.TemporaryDirectory(prefix="chord-explanation-test-") as temporary:
        write_explanation([summary], rows, temporary)
        root = Path(temporary)
        exported = json.loads((root / "semantic_attribution.json").read_text(encoding="utf-8"))
        assert exported["checkpoints"][0]["scenario"] == "robot_breakdown"
        with (root / "case_timelines.csv").open(encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 36
        assert (root / "semantic_case_study.md").exists()


def test_semantic_attribution_masks_real_edges_without_retraining():
    env_cfg = EnvConfig(
        max_agvs=3,
        max_pickers=2,
        max_tasks=6,
        initial_agvs_min=3,
        initial_agvs_max=3,
        initial_pickers_min=2,
        initial_pickers_max=2,
        initial_tasks_min=4,
        initial_tasks_max=4,
        num_zones=2,
    )
    policy = build_policy(
        "chord_mappo",
        env_cfg,
        TrainConfig(hidden_dim=16, encoder_layers=1, device="cpu"),
        ModelConfig(coalition_top_k=2),
    ).eval()
    assert isinstance(policy, OpenHyperPolicy)
    obs = OpenWarehouseEnv(env_cfg, seed=7).reset(seed=7)
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    report = semantic_edge_attribution(policy, obs, torch.device("cpu"))
    assert report["edge_attributions"]
    assert {row["edge_type"] for row in report["edge_attributions"]} == {
        "task_coalition",
        "shared_zone",
    }
    assert set(report["group_attributions"]) == {
        "all_task_coalitions",
        "all_shared_zones",
    }
    assert all(torch.equal(before[name], value) for name, value in policy.state_dict().items())


def test_task_coalitions_exclude_agents_that_cannot_accept_the_task():
    env_cfg = EnvConfig(
        max_agvs=2,
        max_pickers=1,
        max_tasks=4,
        initial_agvs_min=2,
        initial_agvs_max=2,
        initial_pickers_min=1,
        initial_pickers_max=1,
        initial_tasks_min=3,
        initial_tasks_max=3,
    )
    env = OpenWarehouseEnv(env_cfg, seed=11)
    obs = env.reset(seed=11)
    actions = torch.zeros(env_cfg.max_agents, dtype=torch.long).numpy()
    actions[0] = 1
    actions[env_cfg.max_agvs] = 1
    obs, _, _, _ = env.step(actions)
    policy = build_policy(
        "chord_mappo",
        env_cfg,
        TrainConfig(hidden_dim=16, encoder_layers=1),
        ModelConfig(coalition_top_k=2),
    )
    batch = to_torch({key: value[None] for key, value in obs.items()}, torch.device("cpu"))
    incidence = policy.build_incidence(batch).matrix[0]
    assert incidence[0, : env_cfg.max_tasks].sum() == 0
    assert incidence[env_cfg.max_agvs, : env_cfg.max_tasks].sum() == 0
