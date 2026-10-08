from __future__ import annotations

from open_hyper_warehouse.config import EnvConfig, load_config
from open_hyper_warehouse.eaai_protocol import (
    EAAI_BLOCKS,
    EAAI_EXCLUDED_DIAGNOSTICS,
    EAAI_FORMAL_SEEDS,
    EAAI_PRELIMINARY_SEEDS,
    PRIMARY_METRICS,
    all_eaai_scenarios,
    core8_eaai_scenarios,
    infer_protocol,
    validate_core8_protocol,
    validate_eaai_protocol,
)
from open_hyper_warehouse.scenarios import make_scenario


def test_eaai_blocks_are_locked_disjoint_and_constructible():
    validate_eaai_protocol()
    scenarios = all_eaai_scenarios()
    assert len(EAAI_BLOCKS) == 4
    assert len(scenarios) == 12
    assert len(scenarios) == len(set(scenarios))
    assert not set(scenarios) & set(EAAI_EXCLUDED_DIAGNOSTICS)
    base = EnvConfig(max_agvs=24, max_pickers=12, max_tasks=80)
    for name in scenarios:
        make_scenario(base, name).validate()


def test_eaai_config_matches_primary_protocol():
    cfg = load_config("configs/eaai_core8_cuda.toml")
    assert cfg.evaluation.scenario_names == core8_eaai_scenarios()
    assert set(PRIMARY_METRICS) == {
        "service_level",
        "throughput_per_100_steps",
        "deadline_miss_rate",
        "mean_flow_time",
    }


def test_core8_config_and_six_seed_protocol_are_locked():
    validate_core8_protocol()
    scenarios = core8_eaai_scenarios()
    cfg = load_config("configs/eaai_core8_cuda.toml")
    assert cfg.evaluation.scenario_names == scenarios
    assert infer_protocol(scenarios) == "core8"
    assert EAAI_PRELIMINARY_SEEDS == (1, 45, 789)
    assert EAAI_FORMAL_SEEDS == (1, 45, 789, 2027, 4096, 8191)
    assert cfg.train.total_steps == 1_024_000
    assert cfg.train.num_envs * cfg.train.rollout_steps == 2048
    assert cfg.train.parallel_env_workers == 8


def test_repairable_breakdown_is_temporally_valid():
    cfg = make_scenario(EnvConfig(), "robot_breakdown")
    assert cfg.scheduled_breakdown_count == cfg.scheduled_repair_count
    assert cfg.scheduled_repair_step > cfg.scheduled_breakdown_step
    assert cfg.scheduled_breakdown_count <= 3


def test_core8_main_stresses_are_moderate_not_near_capacity():
    base = EnvConfig(max_agvs=24, max_pickers=12, max_tasks=80)
    large = make_scenario(base, "unseen_large")
    agv_heavy = make_scenario(base, "agv_heavy_ratio")
    picker_heavy = make_scenario(base, "picker_heavy_ratio")
    churn = make_scenario(base, "high_churn")
    assert base.initial_tasks_max < large.initial_tasks_max < base.max_tasks
    assert base.initial_agvs_max < large.initial_agvs_max < base.max_agvs
    assert agv_heavy.initial_agvs_min > base.initial_agvs_max
    assert agv_heavy.initial_pickers_min < base.initial_pickers_min
    assert picker_heavy.initial_agvs_min < base.initial_agvs_min
    assert picker_heavy.initial_pickers_min > base.initial_pickers_max
    assert churn.agent_join_probability == 0.03
    assert churn.agent_leave_probability == 0.02
