"""Regenerate every English paper figure from the archived experiment results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "figs"

GROUP_OUTPUTS = {
    "figure3": ("Figure_3A.pdf", "Figure_3B.pdf", "Figure_3L.pdf"),
    "figure4": ("Figure_4A.pdf", "Figure_4B.pdf", "Figure_4L.pdf"),
    "figure5": (
        "Figure_5A.pdf", "Figure_5B.pdf", "Figure_5C.pdf",
        "Figure_5D.pdf", "Figure_5L.pdf",
    ),
    "figure6": (
        "Figure_6A.pdf", "Figure_6B.pdf", "Figure_6C.pdf",
        "Figure_6D.pdf", "Figure_6L.pdf",
    ),
    "figure7": (
        "Figure_7A.pdf", "Figure_7B.pdf", "Figure_7C.pdf",
        "Figure_7D.pdf", "Figure_7L.pdf",
    ),
    "figure8": (
        "Figure_8A.pdf", "Figure_8B.pdf", "Figure_8C.pdf", "Figure_8D.pdf",
    ),
    "figure9": ("Figure_9A.pdf", "Figure_9B.pdf"),
    "appendix-b": ("Figure_B_1A.pdf", "Figure_B_1B.pdf", "Figure_B_1L.pdf"),
    "appendix-d": ("Figure_D_1A.pdf", "Figure_D_1B.pdf"),
    "appendix-j": (
        "Figure_J_1A.pdf", "Figure_J_1B.pdf", "Figure_J_2A.pdf", "Figure_J_2B.pdf",
    ),
}

ALL_STEPS = (
    ("generate_paper_artifacts.py",),
    ("generate_eaai_numbered_figures.py",),
    ("build_semantic_paper_artifacts.py", "--figures-only"),
    ("regenerate_retained_scenario_plots.py",),
    ("regenerate_method_legends.py",),
    ("sync_eaai_manuscript_figures.py",),
)

GROUP_STEPS = {
    "figure3": (
        ("regenerate_retained_scenario_plots.py", "--group", "figure3"),
        ("regenerate_method_legends.py", "--group", "figure3"),
    ),
    "figure4": (
        ("generate_paper_artifacts.py",),
        ("regenerate_method_legends.py", "--group", "figure4"),
        ("sync_eaai_manuscript_figures.py", "--group", "figure4"),
    ),
    "figure5": (
        ("generate_eaai_numbered_figures.py",),
        ("regenerate_method_legends.py", "--group", "figure5"),
        ("sync_eaai_manuscript_figures.py", "--group", "figure5"),
    ),
    "figure6": (
        ("generate_paper_artifacts.py",),
        ("sync_eaai_manuscript_figures.py", "--group", "figure6"),
    ),
    "figure7": (
        ("generate_paper_artifacts.py",),
        ("sync_eaai_manuscript_figures.py", "--group", "figure7"),
    ),
    "figure8": (
        ("generate_eaai_numbered_figures.py",),
        ("sync_eaai_manuscript_figures.py", "--group", "figure8"),
    ),
    "figure9": (
        ("build_semantic_paper_artifacts.py", "--figures-only"),
        ("sync_eaai_manuscript_figures.py", "--group", "figure9"),
    ),
    "appendix-b": (
        ("generate_paper_artifacts.py",),
        ("regenerate_method_legends.py", "--group", "appendix-b"),
        ("sync_eaai_manuscript_figures.py", "--group", "appendix-b"),
    ),
    "appendix-d": (
        ("build_semantic_paper_artifacts.py", "--figures-only"),
        ("sync_eaai_manuscript_figures.py", "--group", "appendix-d"),
    ),
    "appendix-j": (
        ("regenerate_retained_scenario_plots.py", "--group", "appendix-j"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", choices=("all", *GROUP_STEPS), default="all",
        help="Generate all figures or one manuscript figure group.",
    )
    return parser.parse_args()


def run(script_name: str, *arguments: str) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script_name), *arguments]
    print(f"[figures-en] {script_name}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    steps = ALL_STEPS if args.group == "all" else GROUP_STEPS[args.group]
    for step in steps:
        run(*step)
    expected = (
        tuple(name for names in GROUP_OUTPUTS.values() for name in names)
        if args.group == "all"
        else GROUP_OUTPUTS[args.group]
    )
    missing = [name for name in expected if not (OUTPUT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing generated figure files: {missing}")
    print(f"English figure group '{args.group}' written under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
