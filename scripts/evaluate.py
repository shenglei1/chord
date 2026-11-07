from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, ".")

from open_hyper_warehouse.evaluation import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--device")
    args = parser.parse_args()
    report = evaluate_checkpoint(
        args.checkpoint, args.output, args.scenarios, args.episodes, args.device
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
