"""Run the three-scene EAAI mechanism ablation without changing the main runner.

This runner intentionally shares the existing trainer, checkpoint validation,
evaluation, aggregation, table writer and export code.  Its only protocol
change is the locked three-scene evaluation set used for the ablation table.
The proposed method and the Entity/Graph CAHR controls are excluded by design:
their six-seed comparison checkpoints are produced by the main comparison run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_hyper_warehouse.analysis import aggregate_reports, write_aggregate
from open_hyper_warehouse.config import load_config
from open_hyper_warehouse.evaluation import evaluate_checkpoint
from open_hyper_warehouse.eaai_protocol import EAAI_FORMAL_SEEDS
from open_hyper_warehouse.paper_results import write_eaai_tables
from open_hyper_warehouse.scenarios import make_scenario
from open_hyper_warehouse.trainer import train_algorithm
from open_hyper_warehouse.training_exports import export_training_data


ABLATION_SCENARIOS = (
    "in_distribution",
    "agv_heavy_ratio",
    "robot_breakdown",
)
ABLATION_METHODS = (
    "chord_no_reservation",
    "chord_capacity_factorized",
    "chord_no_capacity",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-scene CPU/GPU EAAI mechanism ablation runner"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(EAAI_FORMAL_SEEDS)
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=list(ABLATION_METHODS),
        help="subset of the three ablation methods; main/CAHR controls are rejected",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    unknown = set(args.algorithms) - set(ABLATION_METHODS)
    if unknown:
        raise ValueError(
            "This runner accepts only the three ablations: "
            f"{sorted(ABLATION_METHODS)}; got {sorted(unknown)}"
        )
    algorithms = list(dict.fromkeys(args.algorithms))
    config_path = Path(args.config)
    cfg = load_config(config_path)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    partial = root / "ablation3_suite.partial.json"

    def persist() -> None:
        temporary = partial.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        temporary.replace(partial)

    for algorithm in algorithms:
        for seed in args.seeds:
            checkpoint = root / algorithm / f"seed_{seed}" / "final.pt"
            run_dir = checkpoint.parent
            incomplete = list(run_dir.glob("step_*.pt")) + [
                path for path in (
                    run_dir / "progress.json",
                    run_dir / "training_metrics.jsonl",
                ) if path.exists()
            ]
            if incomplete and not checkpoint.exists():
                raise RuntimeError(
                    f"refusing to overwrite incomplete run {run_dir}; use a new output root"
                )
            if args.reuse_existing and checkpoint.exists():
                metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if (
                    metadata.get("algorithm") != algorithm
                    or int(metadata.get("seed", -1)) != seed
                    or metadata.get("config") != cfg.to_dict()
                ):
                    raise ValueError(f"incompatible checkpoint: {checkpoint}")
            else:
                checkpoint = train_algorithm(algorithm, cfg, seed, str(root)).checkpoint
            reports.append(
                evaluate_checkpoint(
                    checkpoint,
                    scenarios=list(ABLATION_SCENARIOS),
                    episodes=cfg.evaluation.episodes,
                )
            )
            persist()

    suite = root / "ablation3_suite.json"
    suite.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    write_eaai_tables(reports, root / "ablation3_paper_tables")
    write_aggregate(
        aggregate_reports(reports, reference="chord_no_reservation"),
        root / "ablation3_aggregate",
    )
    manifest = {
        "runner": "scripts/run_eaai_ablation3.py",
        "protocol": "EAAI mechanism ablation, locked three-scene evaluation",
        "scenarios": list(ABLATION_SCENARIOS),
        "algorithms": algorithms,
        "excluded_existing_comparison_methods": [
            "chord_mappo", "entity_cahr", "graph_cahr"
        ],
        "training_seeds": args.seeds,
        "statistical_unit": "independent policy-training seed",
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "scenario_parameters": {
            name: asdict(make_scenario(cfg.env, name))
            for name in ABLATION_SCENARIOS
        },
        "checkpoint_reuse": args.reuse_existing,
    }
    (root / "ablation3_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    export_training_data(root)
    print(f"wrote {suite}")


if __name__ == "__main__":
    main()
