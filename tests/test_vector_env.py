from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from open_hyper_warehouse.config import (
    EnvConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from open_hyper_warehouse.trainer import train_algorithm
from open_hyper_warehouse.vector_env import make_vector_env


def _tiny_env() -> EnvConfig:
    return EnvConfig(
        max_agvs=2,
        max_pickers=1,
        max_tasks=4,
        initial_agvs_min=2,
        initial_agvs_max=2,
        initial_pickers_min=1,
        initial_pickers_max=1,
        initial_tasks_min=2,
        initial_tasks_max=2,
        episode_steps=4,
    )


def test_process_vector_env_matches_serial_transition_tape():
    seeds = [101, 202]
    serial = make_vector_env(_tiny_env(), 2, 0, seeds)
    parallel = make_vector_env(_tiny_env(), 2, 2, seeds)
    try:
        serial_obs = serial.reset_all(seeds)
        parallel_obs = parallel.reset_all(seeds)
        for left, right in zip(serial_obs, parallel_obs):
            assert all(np.array_equal(left[key], right[key]) for key in left)
        actions = np.zeros((2, 3), dtype=np.int64)
        for _ in range(4):
            serial_rows = serial.step(actions)
            parallel_rows = parallel.step(actions)
            for left, right in zip(serial_rows, parallel_rows):
                assert left[1:] == right[1:]
                assert all(np.array_equal(left[0][key], right[0][key]) for key in left[0])
    finally:
        serial.close()
        parallel.close()


def test_parallel_trainer_persists_incremental_telemetry(tmp_path):
    cfg = ExperimentConfig(
        env=_tiny_env(),
        train=TrainConfig(
            total_steps=8,
            num_envs=2,
            parallel_env_workers=2,
            rollout_steps=2,
            update_epochs=1,
            minibatch_size=4,
            hidden_dim=16,
            encoder_layers=1,
            checkpoint_interval=4,
            device="cpu",
        ),
        model=ModelConfig(coalition_top_k=2),
        evaluation=EvaluationConfig(episodes=1),
    )
    result = train_algorithm("chord_mappo", cfg, seed=4, output_root=tmp_path)
    run_dir = result.checkpoint.parent
    progress = json.loads((run_dir / "progress.json").read_text())
    checkpoint = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
    assert progress["steps"] == 8
    assert progress["environment_steps_per_second"] > 0
    assert checkpoint["environment_backend"] == "process"
    assert checkpoint["parallel_env_workers"] == 2
    assert len((run_dir / "training_metrics.jsonl").read_text().splitlines()) == 2
    for key in (
        "approx_kl",
        "clip_fraction",
        "explained_variance_before_update",
        "reservation_stop_rate",
        "candidate_coalitions_per_step",
        "masked_coalitions_per_step",
        "policy_entropy",
        "entropy_coefficient",
    ):
        assert key in progress


@pytest.mark.parametrize(
    "algorithm",
    ["plato_faithful", "mohito_official", "entity_mappo", "graph_mappo"],
)
def test_every_learned_baseline_runs_with_process_environments(tmp_path, algorithm):
    cfg = ExperimentConfig(
        env=_tiny_env(),
        train=TrainConfig(
            total_steps=4,
            num_envs=2,
            parallel_env_workers=2,
            rollout_steps=1,
            update_epochs=1,
            minibatch_size=2,
            hidden_dim=16,
            encoder_layers=1,
            checkpoint_interval=4,
            device="cpu",
        ),
        model=ModelConfig(
            coalition_top_k=2,
            mohito_hidden_dim=8,
            mohito_layers=2,
            mohito_heads=1,
            mohito_replay_size=8,
            mohito_batch_size=2,
            mohito_warmup_steps=2,
            mohito_experiences_before_update=1,
            mohito_backups_per_update=1,
        ),
        evaluation=EvaluationConfig(episodes=1),
    )
    result = train_algorithm(algorithm, cfg, seed=8, output_root=tmp_path)
    checkpoint = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["algorithm"] == algorithm
    assert checkpoint["environment_backend"] == "process"
    assert checkpoint["parallel_env_workers"] == 2
