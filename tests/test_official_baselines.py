from pathlib import Path

import numpy as np
import torch

from open_hyper_warehouse.buffer import to_torch
from open_hyper_warehouse.baselines import (
    EAAI_MAIN_LEARNED_KEYS,
    EAAI_MAIN_METHODS,
    EAAI_MAIN_OPERATIONAL_KEYS,
    validate_main_registry,
)
from open_hyper_warehouse.config import (
    EnvConfig, EvaluationConfig, ExperimentConfig, ModelConfig, TrainConfig
)
from open_hyper_warehouse.compatibility import audit_main_baseline_compatibility
from open_hyper_warehouse.env import OpenWarehouseEnv
from open_hyper_warehouse.heuristics import rolling_horizon_actions
from open_hyper_warehouse.models import build_policy
from open_hyper_warehouse.mohito_official import OfficialMohitoTrainer


def test_eaai_main_registry_is_exact_and_auditable():
    validate_main_registry()
    assert EAAI_MAIN_LEARNED_KEYS == (
        "chord_mappo",
        "plato_faithful",
        "mohito_official",
        "entity_mappo",
        "graph_mappo",
    )
    assert EAAI_MAIN_OPERATIONAL_KEYS == ("rolling_horizon",)
    assert all(spec.implementation_level and spec.adaptation_boundary for spec in EAAI_MAIN_METHODS)


def tiny_config(total_steps: int = 3) -> ExperimentConfig:
    env = EnvConfig(
        max_agvs=2, max_pickers=1, max_tasks=4,
        initial_agvs_min=2, initial_agvs_max=2,
        initial_pickers_min=1, initial_pickers_max=1,
        initial_tasks_min=2, initial_tasks_max=2,
        episode_steps=5,
    )
    train = TrainConfig(
        total_steps=total_steps, num_envs=1, rollout_steps=1,
        minibatch_size=2, hidden_dim=16, encoder_layers=1, device="cpu",
    )
    model = ModelConfig(
        mohito_hidden_dim=8, mohito_layers=2, mohito_heads=1,
        mohito_replay_size=8, mohito_batch_size=2, mohito_warmup_steps=2,
        mohito_experiences_before_update=1, mohito_backups_per_update=1,
    )
    return ExperimentConfig(env, train, model, EvaluationConfig(episodes=1))


def test_plato_faithful_and_official_mohito_interfaces():
    cfg = tiny_config()
    obs = OpenWarehouseEnv(cfg.env, seed=0).reset(seed=0)
    batch = to_torch({k: v[None] for k, v in obs.items()}, torch.device("cpu"))
    plato = build_policy("plato_faithful", cfg.env, cfg.train, cfg.model)
    output = plato(batch)
    assert output.logits.shape == (1, cfg.env.max_agents, cfg.env.max_tasks + 1)
    mohito = build_policy("mohito_official", cfg.env, cfg.train, cfg.model)
    actions = mohito.act(batch, deterministic=True)[0][0].numpy()
    assert np.all(obs["action_mask"][np.arange(cfg.env.max_agents), actions])
    assert np.all(actions[~obs["agent_mask"]] == 0)


def test_graph_control_matches_hypergraph_parameter_budget():
    cfg = tiny_config()
    proposed = build_policy("chord_mappo", cfg.env, cfg.train, cfg.model)
    graph = build_policy("graph_cahr", cfg.env, cfg.train, cfg.model)
    assert proposed.parameter_count() == graph.parameter_count()


def test_rolling_horizon_milp_returns_joint_feasible_assignments():
    cfg = tiny_config()
    obs = OpenWarehouseEnv(cfg.env, seed=2).reset(seed=2)
    actions = rolling_horizon_actions(obs, horizon=3, candidate_limit=4)
    assert np.all(obs["action_mask"][np.arange(cfg.env.max_agents), actions])
    selected = actions[actions > 0]
    assert selected.size >= 2
    for task_action in np.unique(selected):
        workers = np.flatnonzero(actions == task_action)
        roles = obs["agent_features"][workers, :2].argmax(axis=1)
        assert set(roles.tolist()) == {0, 1}


def test_official_mohito_executes_replay_updates(tmp_path: Path):
    result = OfficialMohitoTrainer(tiny_config(), seed=0, output_dir=tmp_path).train()
    checkpoint = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
    assert result.steps == 3
    assert result.updates > 0
    assert checkpoint["algorithm"] == "mohito_official"
    assert checkpoint["sampling_protocol"] == "same parallel environments and episode seeds as MAPPO"


def test_every_main_method_completes_every_locked_scenario():
    cfg = tiny_config(total_steps=2)
    report = audit_main_baseline_compatibility(cfg, execution_seed=91_000)
    assert report["method_count"] == 6
    assert report["scenario_count"] == 12
    assert report["episode_count"] == 72
    assert report["all_valid"]
    breakdown = [
        row for row in report["rows"] if row["scenario"] == "robot_breakdown"
    ]
    assert all(row["forced_failures"] > 0 and row["repairs"] > 0 for row in breakdown)
