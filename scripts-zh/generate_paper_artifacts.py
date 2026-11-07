"""Audit final CSVs and generate publication-ready tables and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, ".")

from paper_artifacts import generate_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="figs-zh/paper_artifacts")
    parser.add_argument("--uncertainty", choices=("ci95", "std"), default="ci95")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--formats", nargs="+", choices=("pdf", "svg", "png"),
        default=("pdf", "svg", "png"),
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help="write the audit, metric dictionary and table data without importing matplotlib",
    )
    args = parser.parse_args()
    audit = generate_artifacts(
        Path(args.results_root),
        Path(args.output),
        uncertainty=args.uncertainty,
        formats=args.formats,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        audit_only=args.audit_only,
    )
    print(json.dumps({
        "status": audit["status"],
        "errors": len(audit["errors"]),
        "warnings": len(audit["warnings"]),
        "output": str(Path(args.output)),
    }, indent=2))


if __name__ == "__main__":
    main()
