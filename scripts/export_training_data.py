from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, ".")

from open_hyper_warehouse.training_exports import export_training_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export complete and in-progress EAAI training telemetry to CSV"
    )
    parser.add_argument("--root", default="eaai_core6_results")
    args = parser.parse_args()
    print(json.dumps(export_training_data(args.root), indent=2))


if __name__ == "__main__":
    main()
