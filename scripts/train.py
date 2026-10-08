from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_hyper_warehouse.config import load_config
from open_hyper_warehouse.trainer import train_algorithm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=[
        "chord_mappo", "entity_mappo", "graph_mappo",
        "plato_faithful", "mohito_official",
        "chord_no_reservation", "chord_capacity_factorized", "chord_no_capacity",
        "entity_cahr", "graph_cahr",
    ])
    parser.add_argument("--config", default="configs/eaai_core8_smoke.toml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    result = train_algorithm(args.algorithm, load_config(args.config), args.seed, args.output)
    print(f"checkpoint={result.checkpoint} steps={result.steps} wall_seconds={result.wall_seconds:.2f}")


if __name__ == "__main__":
    main()
