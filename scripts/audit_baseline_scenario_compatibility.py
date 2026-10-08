from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_hyper_warehouse.compatibility import audit_main_baseline_compatibility
from open_hyper_warehouse.config import load_config
from open_hyper_warehouse.eaai_protocol import infer_protocol, protocol_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every EAAI main method through every locked scenario"
    )
    parser.add_argument("--config", default="configs/eaai_robotics_smoke.toml")
    parser.add_argument("--seed", type=int, default=73_000)
    parser.add_argument("--scenario-set", choices=("auto", "core8", "full12"), default="auto")
    parser.add_argument("--output", default="eaai_compatibility_audit.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    inferred = infer_protocol(cfg.evaluation.scenario_names)
    protocol = inferred if args.scenario_set == "auto" else args.scenario_set
    if protocol != inferred:
        raise ValueError(f"config contains {inferred}, but --scenario-set requested {protocol}")
    report = audit_main_baseline_compatibility(
        cfg, args.seed, scenarios=protocol_scenarios(protocol)
    )
    report["scenario_protocol"] = protocol
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"validated {report['method_count']} methods x {report['scenario_count']} scenarios")
    print(f"wrote {target.resolve()}")


if __name__ == "__main__":
    main()
