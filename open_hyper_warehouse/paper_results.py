"""Direction-aware EAAI result tables generated without manual cherry-picking."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import EAAI_MAIN_BY_KEY
from .eaai_protocol import PRIMARY_METRICS


METRIC_DIRECTION = {
    "service_level": "max",
    "throughput_per_100_steps": "max",
    "deadline_miss_rate": "min",
    "mean_flow_time": "min",
}

ABLATION_KEYS = (
    "chord_mappo",
    "chord_no_reservation",
    "chord_capacity_factorized",
    "chord_no_capacity",
    "entity_cahr",
    "graph_cahr",
)

DECODER_DIAGNOSTICS = (
    "candidate_coalitions_per_step",
    "selected_coalitions_per_step",
    "masked_coalitions_per_step",
    "reservation_rounds_per_step",
    "decoder_latency_ms_per_step",
    "contention_per_delivery",
)


def _method_means(reports: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[float]]:
    values: dict[tuple[str, str, str], list[float]] = {}
    for report in reports:
        for scenario, metrics in report.get("scenarios", {}).items():
            for metric in PRIMARY_METRICS:
                if metric in metrics:
                    values.setdefault((report["algorithm"], scenario, metric), []).append(
                        float(metrics[metric]["mean"])
                    )
    return values


def build_eaai_tables(reports: list[dict[str, Any]]) -> dict[str, Any]:
    values = _method_means(reports)
    available = {key[0] for key in values}
    main_methods = [key for key in EAAI_MAIN_BY_KEY if key in available]
    ablation_methods = [key for key in ABLATION_KEYS if key in available]
    scenarios = sorted({key[1] for key in values})

    rows = []
    ranks = []
    wins = {method: 0 for method in main_methods}
    comparisons = 0
    for scenario in scenarios:
        for metric in PRIMARY_METRICS:
            records = []
            for method in main_methods:
                samples = values.get((method, scenario, metric), [])
                if not samples:
                    continue
                mean = float(np.mean(samples))
                rows.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "direction": METRIC_DIRECTION[metric],
                        "method": method,
                        "mean": mean,
                        "std_across_training_seeds": (
                            float(np.std(samples, ddof=1)) if len(samples) > 1 else None
                        ),
                        "n_training_seeds": len(samples) if method != "rolling_horizon" else 0,
                    }
                )
                records.append((method, mean))
            if records:
                comparisons += 1
                reverse = METRIC_DIRECTION[metric] == "max"
                ordered = sorted(records, key=lambda item: item[1], reverse=reverse)
                best = ordered[0][1]
                for rank, (method, value) in enumerate(ordered, start=1):
                    is_best = bool(np.isclose(value, best, rtol=1e-9, atol=1e-12))
                    ranks.append(
                        {
                            "scenario": scenario,
                            "metric": metric,
                            "method": method,
                            "rank": rank,
                            "is_best_mean": is_best,
                        }
                    )
                    if is_best:
                        wins[method] += 1

    ablation_rows = []
    for (method, scenario, metric), samples in values.items():
        if method in ablation_methods:
            ablation_rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "direction": METRIC_DIRECTION[metric],
                    "method": method,
                    "mean": float(np.mean(samples)),
                    "std_across_training_seeds": (
                        float(np.std(samples, ddof=1)) if len(samples) > 1 else None
                    ),
                    "n_training_seeds": len(samples),
                }
            )

    nominal = {
        (row["method"], row["metric"]): row["mean"]
        for row in rows
        if row["scenario"] == "in_distribution"
    }
    degradation_rows = []
    for row in rows:
        if row["scenario"] == "in_distribution":
            continue
        baseline = nominal.get((row["method"], row["metric"]))
        if baseline is None:
            continue
        raw_change = row["mean"] - baseline
        # Positive means worse than the method's own nominal result for every
        # metric, which prevents lower-is-better KPIs from being reversed.
        oriented = -raw_change if row["direction"] == "max" else raw_change
        degradation_rows.append(
            {
                "scenario": row["scenario"],
                "metric": row["metric"],
                "method": row["method"],
                "nominal_mean": baseline,
                "scenario_mean": row["mean"],
                "oriented_degradation": oriented,
                "interpretation": "positive=worse_than_own_nominal",
            }
        )

    diagnostic_rows = []
    for report in reports:
        for scenario, metrics in report.get("scenarios", {}).items():
            for metric in DECODER_DIAGNOSTICS:
                if metric not in metrics:
                    continue
                diagnostic_rows.append(
                    {
                        "method": report["algorithm"],
                        "training_seed": report.get("training_seed"),
                        "scenario": scenario,
                        "metric": metric,
                        **metrics[metric],
                    }
                )

    return {
        "main_rows": rows,
        "rank_rows": ranks,
        "ablation_rows": ablation_rows,
        "degradation_rows": degradation_rows,
        "diagnostic_rows": diagnostic_rows,
        "summary": {
            "metric_directions": METRIC_DIRECTION,
            "main_methods_present": main_methods,
            "ablation_methods_present": ablation_methods,
            "scenario_count": len(scenarios),
            "method_best_mean_counts": wins,
            "scenario_metric_comparisons": comparisons,
            "warning": (
                "Best-mean counts are descriptive. Confirmatory claims require seed-level "
                "confidence intervals, effect sizes and Holm-adjusted paired tests."
            ),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_eaai_tables(reports: list[dict[str, Any]], output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tables = build_eaai_tables(reports)
    _write_csv(target / "main_comparison.csv", tables["main_rows"])
    _write_csv(target / "direction_aware_ranks.csv", tables["rank_rows"])
    _write_csv(target / "ablation_comparison.csv", tables["ablation_rows"])
    _write_csv(target / "degradation_from_nominal.csv", tables["degradation_rows"])
    _write_csv(target / "decoder_diagnostics.csv", tables["diagnostic_rows"])
    (target / "comparison_summary.json").write_text(
        json.dumps(tables["summary"], indent=2), encoding="utf-8"
    )
