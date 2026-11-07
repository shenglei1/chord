"""Copy generated PDF panels to their final Figure_* filenames.

The source experiment files and unrelated manuscript artwork are never
modified.  Only the explicitly mapped Figure_* files are replaced.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PAPER_MAP = {
    "fig_scale_generalization_A_service_level.pdf": "Figure_4A.pdf",
    "fig_scale_generalization_B_throughput_per_100_steps.pdf": "Figure_4B.pdf",
    "fig_encoder_controls_A_service_level.pdf": "Figure_6A.pdf",
    "fig_encoder_controls_B_throughput_per_100_steps.pdf": "Figure_6B.pdf",
    "fig_encoder_controls_C_deadline_miss_rate.pdf": "Figure_6C.pdf",
    "fig_encoder_controls_D_mean_flow_time.pdf": "Figure_6D.pdf",
    "fig_encoder_controls_legend.pdf": "Figure_6L.pdf",
    "fig_component_ablations_A_service_level.pdf": "Figure_7A.pdf",
    "fig_component_ablations_B_throughput_per_100_steps.pdf": "Figure_7B.pdf",
    "fig_component_ablations_C_contention_per_delivery.pdf": "Figure_7C.pdf",
    "fig_component_ablations_D_deadline_miss_rate.pdf": "Figure_7D.pdf",
    "fig_component_ablations_legend.pdf": "Figure_7L.pdf",
    "fig_training_curves_unsmoothed_A_service_level.pdf": "Figure_B_1A.pdf",
    "fig_training_curves_unsmoothed_B_throughput_per_100_steps.pdf": "Figure_B_1B.pdf",
}

SEMANTIC_MAP = {
    "fig_semantic_masking_sensitivity_A.pdf": "Figure_9A.pdf",
    "fig_semantic_representative_timeline_B.pdf": "Figure_9B.pdf",
    "fig_appendix_policy_js_divergence.pdf": "Figure_D_1A.pdf",
    "fig_appendix_group_value_change.pdf": "Figure_D_1B.pdf",
}

NUMBERED_MAP = {
    "FIG4A.pdf": "Figure_5A.pdf",
    "FIG4B.pdf": "Figure_5B.pdf",
    "FIG4C.pdf": "Figure_5C.pdf",
    "FIG4D.pdf": "Figure_5D.pdf",
    "FIG7A.pdf": "Figure_8A.pdf",
    "FIG7B.pdf": "Figure_8B.pdf",
    "FIG7C.pdf": "Figure_8C.pdf",
    "FIG7D.pdf": "Figure_8D.pdf",
}

GROUP_TARGETS = {
    "figure4": {"Figure_4A.pdf", "Figure_4B.pdf"},
    "figure5": {"Figure_5A.pdf", "Figure_5B.pdf", "Figure_5C.pdf", "Figure_5D.pdf"},
    "figure6": {
        "Figure_6A.pdf", "Figure_6B.pdf", "Figure_6C.pdf", "Figure_6D.pdf", "Figure_6L.pdf",
    },
    "figure7": {
        "Figure_7A.pdf", "Figure_7B.pdf", "Figure_7C.pdf", "Figure_7D.pdf", "Figure_7L.pdf",
    },
    "figure8": {"Figure_8A.pdf", "Figure_8B.pdf", "Figure_8C.pdf", "Figure_8D.pdf"},
    "figure9": {"Figure_9A.pdf", "Figure_9B.pdf"},
    "appendix-b": {"Figure_B_1A.pdf", "Figure_B_1B.pdf"},
    "appendix-d": {"Figure_D_1A.pdf", "Figure_D_1B.pdf"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-artifacts", type=Path, default=Path("figs/paper_artifacts")
    )
    parser.add_argument(
        "--semantic-artifacts", type=Path, default=Path("figs/semantic_analysis")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figs"))
    parser.add_argument(
        "--group", choices=("all", *GROUP_TARGETS), default="all",
        help="Synchronize all mapped panels or one manuscript figure group.",
    )
    return parser.parse_args()


def copy_checked(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"missing derived artwork: {source}")
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    paper = args.paper_artifacts
    semantic = args.semantic_artifacts
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)

    numbered = paper / "eaai_numbered_figures"
    written: list[Path] = []
    for source, target in NUMBERED_MAP.items():
        if args.group != "all" and target not in GROUP_TARGETS[args.group]:
            continue
        output = destination / target
        copy_checked(numbered / source, output)
        written.append(output)
    for source, target in PAPER_MAP.items():
        if args.group != "all" and target not in GROUP_TARGETS[args.group]:
            continue
        output = destination / target
        copy_checked(paper / "figures" / source, output)
        written.append(output)
    for source, target in SEMANTIC_MAP.items():
        if args.group != "all" and target not in GROUP_TARGETS[args.group]:
            continue
        output = destination / target
        copy_checked(semantic / source, output)
        written.append(output)

    if len(written) != len(set(written)):
        raise RuntimeError("duplicate manuscript destination in figure map")
    expected_count = (
        sum(len(targets) for targets in GROUP_TARGETS.values())
        if args.group == "all"
        else len(GROUP_TARGETS[args.group])
    )
    if len(written) != expected_count:
        raise RuntimeError(
            f"group {args.group!r} synchronized {len(written)} files; expected {expected_count}"
        )
    print(f"synchronized {len(written)} mapped PDF files to {destination}")


if __name__ == "__main__":
    main()
