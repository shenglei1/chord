from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from .buffer import to_torch
from .baselines import baseline_metadata
from .config import ExperimentConfig, config_from_dict
from .env import OpenWarehouseEnv
from .heuristics import (
    closest_feasible_actions,
    earliest_deadline_actions,
    fcfs_actions,
    rolling_horizon_actions,
)
from .models import WarehousePolicy, build_policy
from .scenarios import make_scenario
from .trainer import resolve_device


METRICS = (
    "return",
    "deliveries",
    "completion_ratio",
    "service_level",
    "throughput_per_100_steps",
    "deadline_misses",
    "deadline_miss_rate",
    "contentions",
    "travel",
    "utilization",
    "mean_active_agents",
    "agent_joins",
    "agent_leaves",
    "mean_flow_time",
    "travel_per_delivery",
    "max_backlog",
    "forced_failures",
    "repairs",
    "shift_changes",
    "cancellations",
    "orders_offered",
    "contention_per_delivery",
    "unserved_orders",
    "candidate_coalitions_per_step",
    "selected_coalitions_per_step",
    "masked_coalitions_per_step",
    "reservation_rounds_per_step",
    "decoder_latency_ms_per_step",
)


def bootstrap_interval(
    values: np.ndarray, samples: int, seed: int, confidence: float = 0.95
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def load_policy(checkpoint: str | Path, device: str | None = None):
    requested = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = config_from_dict(requested["config"])
    if device is not None:
        cfg.train.device = device
    resolved = resolve_device(cfg.train.device)
    build_name = requested["algorithm"]
    if (
        build_name == "chord_mappo"
        and requested.get("learner_revision")
        == "agentwise-ppo-cardinality-critic-feasible-coalitions-v2"
    ):
        # Pilot checkpoints predate CAHR. Keeping the compatibility route lets
        # them remain auditable while their result folders are explicitly
        # marked pilot_factorized_decoder.
        build_name = "chord_no_reservation"
    policy = build_policy(build_name, cfg.env, cfg.train, cfg.model).to(resolved)
    policy.load_state_dict(requested["state_dict"])
    policy.eval()
    return policy, cfg, requested, resolved


def _policy_action(policy: WarehousePolicy, device: torch.device, deterministic: bool):
    def choose(obs: dict[str, np.ndarray]) -> np.ndarray:
        batch = {key: value[None] for key, value in obs.items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            actions, _, _, _ = policy.act(to_torch(batch, device), deterministic=deterministic)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        diagnostics: dict[str, float] = {
            "decoder_latency_ms_per_step": (time.perf_counter() - started) * 1_000.0
        }
        for key, value in getattr(policy, "last_action_diagnostics", {}).items():
            diagnostics[f"{key}_per_step"] = float(
                value.to(torch.float32).mean().detach().cpu()
            )
        choose.last_diagnostics = diagnostics
        return actions[0].cpu().numpy()

    choose.last_diagnostics = {}
    return choose


def evaluate_action_fn(
    action_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
    config: ExperimentConfig,
    scenario: str,
    episodes: int,
    seed_offset: int = 1_000_000,
) -> list[dict[str, float]]:
    env_cfg = make_scenario(config.env, scenario)
    results: list[dict[str, float]] = []
    for episode in range(episodes):
        seed = seed_offset + episode
        env = OpenWarehouseEnv(env_cfg, seed=seed)
        obs = env.reset(seed=seed)
        done = False
        info = {}
        diagnostic_rows: list[dict[str, float]] = []
        while not done:
            obs, _, done, info = env.step(action_fn(obs))
            diagnostics = getattr(action_fn, "last_diagnostics", None)
            if diagnostics:
                diagnostic_rows.append(dict(diagnostics))
        merged = dict(info)
        for key in METRICS:
            values = [row[key] for row in diagnostic_rows if key in row]
            if values:
                merged[key] = float(np.mean(values))
        results.append({key: float(merged.get(key, 0.0)) for key in METRICS})
    return results


def summarize(
    rows: list[dict[str, float]], bootstrap_samples: int, seed: int
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        low, high = bootstrap_interval(values, bootstrap_samples, seed)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "ci95_low": low,
            "ci95_high": high,
        }
    return summary


def evaluate_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path | None = None,
    scenarios: list[str] | None = None,
    episodes: int | None = None,
    device: str | None = None,
) -> dict:
    policy, cfg, metadata, resolved = load_policy(checkpoint, device)
    scenario_names = scenarios or cfg.evaluation.scenario_names
    num_episodes = episodes or cfg.evaluation.episodes
    latency_obs = OpenWarehouseEnv(cfg.env, seed=999).reset(seed=999)
    choose = _policy_action(policy, resolved, cfg.evaluation.deterministic)
    for _ in range(10):
        choose(latency_obs)
    started = time.perf_counter()
    for _ in range(100):
        choose(latency_obs)
    latency_ms = (time.perf_counter() - started) * 10.0

    report = {
        "algorithm": metadata["algorithm"],
        "training_seed": metadata["seed"],
        "checkpoint": str(Path(checkpoint).resolve()),
        "parameter_count": int(metadata.get("parameter_count", policy.parameter_count())),
        "inference_latency_ms": latency_ms,
        "training_wall_seconds": metadata.get("wall_seconds"),
        "training_peak_memory_mb": metadata.get("peak_memory_mb"),
        "training_device": metadata.get("training_device"),
        "torch_version": metadata.get("torch_version"),
        "cuda_device_name": metadata.get("cuda_device_name"),
        "method_metadata": metadata.get("method_metadata"),
        "episodes_per_scenario": num_episodes,
        "scenarios": {},
    }
    raw_rows: list[dict[str, str | float | int]] = []
    target = Path(output_dir) if output_dir else Path(checkpoint).parent / "evaluation"
    target.mkdir(parents=True, exist_ok=True)
    for scenario_index, scenario in enumerate(scenario_names):
        rows = evaluate_action_fn(
            choose,
            cfg,
            scenario,
            num_episodes,
            seed_offset=1_000_000 + scenario_index * 100_000,
        )
        report["scenarios"][scenario] = summarize(
            rows, cfg.evaluation.bootstrap_samples, seed=42 + scenario_index
        )
        raw_rows.extend(
            {"scenario": scenario, "episode": i, **row} for i, row in enumerate(rows)
        )
        # A crash during a later scenario must not erase completed evaluation
        # evidence from earlier scenarios.
        temporary = target / "summary.partial.json.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(target / "summary.partial.json")
        _write_csv(target / "episodes.partial.csv", raw_rows)
    (target / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(target / "episodes.csv", raw_rows)
    return report


HEURISTICS = {
    "closest": closest_feasible_actions,
    "fcfs": fcfs_actions,
    "earliest_deadline": earliest_deadline_actions,
    "rolling_horizon": rolling_horizon_actions,
}


def evaluate_heuristic(
    name: str,
    config: ExperimentConfig,
    output_dir: str | Path,
    scenarios: list[str] | None = None,
) -> dict:
    if name not in HEURISTICS:
        raise KeyError(f"unknown heuristic {name!r}; choose from {sorted(HEURISTICS)}")
    scenario_names = scenarios or config.evaluation.scenario_names
    metadata = baseline_metadata(name)
    report = {
        "algorithm": name,
        "method_class": metadata["method_class"],
        "method_metadata": metadata,
        "scenarios": {},
    }
    raw_rows: list[dict[str, str | float | int]] = []
    for scenario_index, scenario in enumerate(scenario_names):
        rows = evaluate_action_fn(
            HEURISTICS[name],
            config,
            scenario,
            config.evaluation.episodes,
            seed_offset=1_000_000 + scenario_index * 100_000,
        )
        report["scenarios"][scenario] = summarize(
            rows, config.evaluation.bootstrap_samples, seed=42 + scenario_index
        )
        raw_rows.extend(
            {"scenario": scenario, "episode": i, **row} for i, row in enumerate(rows)
        )
    target = Path(output_dir) / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(target / "episodes.csv", raw_rows)
    return report


def evaluate_closest(
    config: ExperimentConfig, output_dir: str | Path, scenarios: list[str] | None = None
) -> dict:
    return evaluate_heuristic("closest", config, output_dir, scenarios)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
