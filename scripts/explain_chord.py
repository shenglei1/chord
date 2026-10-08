from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_hyper_warehouse.explainability import explain_checkpoint, write_explanation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc semantic hyperedge attribution on AGV breakdown and capacity restoration"
    )
    checkpoint_source = parser.add_mutually_exclusive_group(required=True)
    checkpoint_source.add_argument("--checkpoints", nargs="+")
    checkpoint_source.add_argument(
        "--checkpoints-root",
        help="discover chord_mappo/seed_*/final.pt below a completed suite root",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device")
    parser.add_argument("--output", default="eaai_semantic_explanations")
    args = parser.parse_args()
    if args.episodes < 3:
        raise ValueError("use at least three candidate episodes for median-case selection")
    checkpoints = (
        [str(path) for path in sorted(Path(args.checkpoints_root).glob("chord_mappo/seed_*/final.pt"))]
        if args.checkpoints_root
        else args.checkpoints
    )
    if not checkpoints:
        raise FileNotFoundError("no CHORD-CAHR final checkpoints found")
    summaries, rows = [], []
    for checkpoint in checkpoints:
        summary, checkpoint_rows = explain_checkpoint(checkpoint, args.episodes, args.device)
        summaries.append(summary)
        rows.extend(checkpoint_rows)
    write_explanation(summaries, rows, args.output)
    print(f"wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
