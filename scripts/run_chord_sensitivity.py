"""Evaluate exactly six trained CHORD policies on the locked sensitivity sweep."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, ".")

from open_hyper_warehouse.analysis import aggregate_reports, write_aggregate
from open_hyper_warehouse.config import load_config
from open_hyper_warehouse.evaluation import evaluate_checkpoint


FORMAL_SEEDS = (1, 45, 789, 2027, 4096, 8191)
EXPECTED_SCENARIOS = (
    "breakdown_000", "breakdown_010", "breakdown_025", "breakdown_040",
    "demand_050", "demand_100", "demand_200", "demand_350",
)


def evaluate_one(
    checkpoint: str,
    target: str,
    scenarios: list[str],
    episodes: int,
    device: str,
) -> dict[str, Any]:
    return evaluate_checkpoint(
        checkpoint,
        output_dir=target,
        scenarios=scenarios,
        episodes=episodes,
        device=device,
    )


def validate_checkpoint(path: Path, seed: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing CHORD checkpoint for seed {seed}: {path}")
    metadata = torch.load(path, map_location="cpu", weights_only=False)
    if metadata.get("algorithm") != "chord_mappo":
        raise ValueError(f"not a CHORD checkpoint: {path}")
    if int(metadata.get("seed", -1)) != seed:
        raise ValueError(f"seed mismatch in {path}")
    if int(metadata.get("steps", -1)) != 1_024_000:
        raise ValueError(f"checkpoint is not the formal 1.024M-step run: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-root", default="results/main_comparison/chord_mappo")
    parser.add_argument("--config", default="configs/cie_eaai_sensitivity.toml")
    parser.add_argument("--output", default="results/sensitivity/chord")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if tuple(args.seeds) != FORMAL_SEEDS:
        raise ValueError(f"formal sensitivity requires seeds {FORMAL_SEEDS}")
    if args.workers < 1 or args.workers > len(FORMAL_SEEDS):
        raise ValueError("workers must be between 1 and 6")

    config_path = Path(args.config)
    config = load_config(config_path)
    scenarios = tuple(config.evaluation.scenario_names)
    if scenarios != EXPECTED_SCENARIOS:
        raise ValueError(f"sensitivity scenario mismatch: {scenarios}")
    checkpoint_root = Path(args.checkpoints_root)
    output = Path(args.output)
    checkpoints = {
        seed: checkpoint_root / f"seed_{seed}" / "final.pt" for seed in FORMAL_SEEDS
    }
    for seed, checkpoint in checkpoints.items():
        validate_checkpoint(checkpoint, seed)

    plan = {
        "algorithm": "chord_mappo",
        "seeds": list(FORMAL_SEEDS),
        "scenarios": list(scenarios),
        "episodes_per_scenario": config.evaluation.episodes,
        "total_episodes": len(FORMAL_SEEDS) * len(scenarios) * config.evaluation.episodes,
        "workers": args.workers,
        "device": args.device,
        "checkpoint_root": str(checkpoint_root),
        "output": str(output),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    reports: dict[int, dict[str, Any]] = {}
    pending = []
    for seed, checkpoint in checkpoints.items():
        target = output / f"seed_{seed}"
        summary = target / "summary.json"
        if summary.exists():
            if not args.reuse_existing:
                raise FileExistsError(
                    f"existing result would be overwritten: {summary}; use --reuse-existing"
                )
            report = json.loads(summary.read_text(encoding="utf-8"))
            if report.get("algorithm") != "chord_mappo" or int(report.get("training_seed", -1)) != seed:
                raise ValueError(f"incompatible existing result: {summary}")
            reports[seed] = report
        else:
            pending.append((seed, checkpoint, target))

    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            futures = {
                executor.submit(
                    evaluate_one,
                    str(checkpoint),
                    str(target),
                    list(scenarios),
                    config.evaluation.episodes,
                    args.device,
                ): seed
                for seed, checkpoint, target in pending
            }
            for future in as_completed(futures):
                seed = futures[future]
                reports[seed] = future.result()
                print(f"completed CHORD sensitivity seed {seed}", flush=True)

    ordered = [reports[seed] for seed in FORMAL_SEEDS]
    (output / "sensitivity_suite.json").write_text(
        json.dumps(ordered, indent=2), encoding="utf-8"
    )
    write_aggregate(
        aggregate_reports(
            ordered,
            reference="chord_mappo",
            bootstrap_samples=config.evaluation.bootstrap_samples,
        ),
        output / "aggregate",
    )
    manifest = {
        **plan,
        "evaluation_only": True,
        "zero_shot": True,
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "statistical_unit": "independent CHORD policy-training seed",
        "operational_baselines_excluded": True,
        "other_learning_baselines_excluded": True,
    }
    (output / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote {output / 'sensitivity_suite.json'}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
