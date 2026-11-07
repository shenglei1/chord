from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, ".")

from open_hyper_warehouse.analysis import aggregate_reports, write_aggregate
from open_hyper_warehouse.baselines import (
    EAAI_MAIN_LEARNED_KEYS,
    EAAI_MAIN_METHODS,
    EAAI_MAIN_OPERATIONAL_KEYS,
    validate_main_registry,
)
from open_hyper_warehouse.config import config_from_dict, load_config
from open_hyper_warehouse.eaai_protocol import (
    EAAI_BLOCKS,
    EAAI_FORMAL_SEEDS,
    EAAI_PRELIMINARY_SEEDS,
    PRIMARY_METRICS,
    REALISM_ASSUMPTIONS,
    SECONDARY_METRICS,
    all_eaai_scenarios,
    infer_protocol,
    protocol_blocks,
    protocol_scenarios,
    validate_eaai_protocol,
)
from open_hyper_warehouse.evaluation import evaluate_checkpoint, evaluate_heuristic
from open_hyper_warehouse.paper_results import write_eaai_tables
from open_hyper_warehouse.scenarios import make_scenario
from open_hyper_warehouse.trainer import train_algorithm
from open_hyper_warehouse.training_exports import export_training_data


# Main comparison: the nearest open-system learning mechanisms and one
# operational optimization reference.
OPEN_MAIN_LEARNED = list(EAAI_MAIN_LEARNED_KEYS)
OPEN_MAIN_OPERATIONAL = list(EAAI_MAIN_OPERATIONAL_KEYS)

# Decoder-package and capacity-context variants. Removing reservation also
# changes joint-action factorization and PPO likelihood, so these variants are
# not presented as an orthogonal two-factor causal experiment.
ABLATIONS = [
    "chord_no_reservation",       # neither capacity context nor joint reservation
    "chord_capacity_factorized",  # capacity context, independent agent actions
    "chord_no_capacity",          # joint reservation, no capacity descriptors
]

# Table C2: identical CAHR decoder with weaker entity/pairwise encoders.  This
# prevents a performance gain from being attributed to the matching layer
# alone and is deliberately separate from the main published baselines.
DECODER_CONTROLS = ["entity_cahr", "graph_cahr"]

def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locked EAAI Applied Robotic Cases experiment matrix"
    )
    parser.add_argument("--config", default="configs/eaai_core6_cuda.toml")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EAAI_FORMAL_SEEDS))
    parser.add_argument(
        "--scenario-set",
        choices=("auto", "core6"),
        default="auto",
        help="auto infers the locked protocol from the config scenario list",
    )
    parser.add_argument(
        "--section", choices=("main", "ablation", "all"), default="all"
    )
    parser.add_argument("--algorithms", nargs="+", help="optional method subset")
    parser.add_argument("--output", default="eaai_robotics_results")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse a final checkpoint only after exact algorithm/seed/config validation",
    )
    args = parser.parse_args()

    validate_eaai_protocol()
    validate_main_registry()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    inferred_protocol = infer_protocol(cfg.evaluation.scenario_names)
    scenario_protocol = inferred_protocol if args.scenario_set == "auto" else args.scenario_set
    if scenario_protocol != inferred_protocol:
        raise ValueError(
            f"config contains {inferred_protocol}, but --scenario-set requested {scenario_protocol}"
        )
    scenarios = protocol_scenarios(scenario_protocol)
    scenario_blocks = protocol_blocks(scenario_protocol)

    section_methods = {
        "main": OPEN_MAIN_LEARNED,
        "ablation": ["chord_mappo", *ABLATIONS, *DECODER_CONTROLS],
        # "all" means every required primary-paper learned run.
        "all": OPEN_MAIN_LEARNED + ABLATIONS + DECODER_CONTROLS,
    }
    selected = _unique(section_methods[args.section])
    allowed = {method for methods in section_methods.values() for method in methods}
    if args.algorithms:
        unknown = set(args.algorithms) - allowed
        if unknown:
            raise ValueError(f"unknown EAAI controlled algorithms: {sorted(unknown)}")
        selected = _unique(args.algorithms)

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    partial_suite = root / f"{args.section}_suite.partial.json"
    reports = []

    def persist_partial() -> None:
        temporary = partial_suite.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        temporary.replace(partial_suite)

    for algorithm in selected:
        for seed in args.seeds:
            expected_checkpoint = Path(args.output) / algorithm / f"seed_{seed}" / "final.pt"
            run_dir = expected_checkpoint.parent
            incomplete_artifacts = (
                list(run_dir.glob("step_*.pt"))
                + [path for path in (run_dir / "progress.json", run_dir / "training_metrics.jsonl") if path.exists()]
            )
            if incomplete_artifacts and not expected_checkpoint.exists():
                raise RuntimeError(
                    f"refusing to overwrite incomplete run {run_dir}; the trainer cannot "
                    "resume an exact environment/RNG state. Archive that directory or use "
                    "a new --output root before restarting."
                )
            if args.reuse_existing and expected_checkpoint.exists():
                metadata = torch.load(expected_checkpoint, map_location="cpu", weights_only=False)
                if (
                    metadata.get("algorithm") != algorithm
                    or int(metadata.get("seed", -1)) != seed
                    or config_from_dict(metadata.get("config", {})).to_dict() != cfg.to_dict()
                ):
                    raise ValueError(
                        f"refusing to reuse incompatible checkpoint {expected_checkpoint}"
                    )
                checkpoint = expected_checkpoint
            else:
                checkpoint = train_algorithm(algorithm, cfg, seed, args.output).checkpoint
            reports.append(
                evaluate_checkpoint(checkpoint, scenarios=scenarios, episodes=cfg.evaluation.episodes)
            )
            persist_partial()

    operational: list[str] = []
    if args.algorithms is None and args.section in {"main", "all"}:
        operational.extend(OPEN_MAIN_OPERATIONAL)
    for method in _unique(operational):
        reports.append(evaluate_heuristic(method, cfg, args.output, scenarios=scenarios))
        persist_partial()

    suite_path = root / f"{args.section}_suite.json"
    suite_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    write_eaai_tables(reports, root / f"{args.section}_paper_tables")
    write_aggregate(
        aggregate_reports(reports, reference="chord_mappo"),
        root / f"{args.section}_aggregate",
    )

    manifest = {
        "venue": "Engineering Applications of Artificial Intelligence",
        "special_issue": "Intelligent Systems and Control for Applied Robotic Cases",
        "submission_deadline": "2027-01-31",
        "research_question": (
            "Can an agent-task-resource hypergraph controller retain service quality when "
            "AGVs, pickers and orders form changing finite sets?"
        ),
        "algorithm_revision": {
            "id": "cahr-joint-conditional-ppo-v1",
            "joint_ppo": (
                "clip the exact autoregressive joint-action ratio formed by the sum "
                "of conditional coalition-selection log probabilities"
            ),
            "critic": "masked entity pools plus permutation-invariant cardinality/load statistics",
            "chord_incidence": "semantically constructed action-feasible AGV-picker-task coalitions",
            "decoder": (
                "capacity-aware autoregressive hyperedge reservation with dynamic "
                "agent, picker and task exclusivity masks"
            ),
            "entropy": "conditional entropy normalized per reservation round and linearly annealed",
            "potential_based_shaping": False,
        },
        "tables": {
            "main_comparison": {
                "learned": OPEN_MAIN_LEARNED,
                "operational": OPEN_MAIN_OPERATIONAL,
                "purpose": "agent-and-task openness under a common interaction budget",
            },
            "decoder_package_and_capacity_context": {
                "methods": ["chord_mappo", *ABLATIONS],
                "purpose": (
                    "compare the complete decoder package with factorized and "
                    "capacity-removed variants"
                ),
            },
            "decoder_matched_encoder_controls": {
                "methods": DECODER_CONTROLS,
                "purpose": "hold the CAHR decoder fixed while changing the relational encoder",
            },
        },
        "executed_section": args.section,
        "executed_learned_methods": selected,
        "executed_operational_methods": _unique(operational),
        "reuse_existing_checkpoints": args.reuse_existing,
        "training_seeds": args.seeds,
        "seed_protocol": (
            "formal_six"
            if tuple(args.seeds) == EAAI_FORMAL_SEEDS
            else "preliminary_three"
            if tuple(args.seeds) == EAAI_PRELIMINARY_SEEDS
            else "custom_debug_or_sensitivity"
        ),
        "formal_training_seeds": list(EAAI_FORMAL_SEEDS),
        "scenario_protocol": scenario_protocol,
        "statistical_unit": "independent policy-training seed",
        "common_evaluation_episode_seeds": True,
        "primary_metrics": list(PRIMARY_METRICS),
        "secondary_metrics": list(SECONDARY_METRICS),
        "scenario_blocks": [
            {
                "name": block.name,
                "scenarios": list(block.scenarios),
                "engineering_question": block.engineering_question,
            }
            for block in scenario_blocks
        ],
        "realized_scenario_parameters": {
            scenario: asdict(make_scenario(cfg.env, scenario)) for scenario in scenarios
        },
        "semantic_explanation": {
            "scenario": "robot_breakdown",
            "events": ["pre_breakdown", "post_breakdown", "post_repair"],
            "candidate_episodes_per_training_seed": 20,
            "selection_rule": "median service level; execution seed breaks ties",
            "intervention": "post-hoc task/zone hyperedge masking; no retraining",
            "claim_boundary": "model-internal structural transparency, not physical causality",
            "runner": "scripts/explain_chord.py",
        },
        "realism_assumptions": REALISM_ASSUMPTIONS,
        "fairness": {
            "shared": [
                "environment transitions and reward",
                "training interactions and final-checkpoint rule",
                "discount factor and reward scale",
                "PPO optimizer, GAE, clipping, epochs and minibatch budget for all MAPPO methods",
                "training seeds, evaluation seeds and scenario instances",
            ],
            "method_specific_only": [
                "published representation mechanism",
                "MOHITO's published replay, target-network, actor/critic learning rates and exploration",
                "MOHITO update-to-data cadence exposed by its official configuration",
                "coalition top-k, attention heads or grouping count when intrinsic",
            ],
            "selection_rule": "validation-only model selection; locked test scenarios never select a checkpoint",
        },
        "provenance": {
            spec.paper_label: spec.to_dict() for spec in EAAI_MAIN_METHODS
        },
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    (root / f"{args.section}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    export_training_data(root)
    print(f"wrote {suite_path}")


if __name__ == "__main__":
    main()
