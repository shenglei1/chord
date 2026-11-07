from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, ".")

from open_hyper_warehouse.analysis import aggregate_reports, write_aggregate
from open_hyper_warehouse.paper_results import write_eaai_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="results/paper_suite.json")
    parser.add_argument("--reference", default="chord_mappo")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    suite_path = Path(args.suite)
    reports = json.loads(suite_path.read_text(encoding="utf-8"))
    aggregate = aggregate_reports(
        reports,
        reference=args.reference,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = Path(args.output) if args.output else suite_path.parent
    write_aggregate(aggregate, output)
    write_eaai_tables(reports, output / "paper_tables")
    print(f"wrote {output / 'aggregate_by_training_seed.json'}")
    print(f"wrote {output / 'aggregate_by_training_seed.csv'}")
    print(f"wrote {output / 'paper_tables'}")


if __name__ == "__main__":
    main()
