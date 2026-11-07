from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _seed_bootstrap(
    values: np.ndarray, samples: int, seed: int
) -> tuple[float | None, float | None]:
    """Bootstrap independent training seeds, not correlated evaluation episodes."""
    if values.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _paired_sign_flip_pvalue(values: np.ndarray, seed: int) -> float | None:
    """Two-sided paired randomization test over independent training seeds."""
    if values.size < 2:
        return None
    observed = abs(float(values.mean()))
    if values.size <= 20:
        indexes = np.arange(1 << values.size, dtype=np.uint64)[:, None]
        bits = ((indexes >> np.arange(values.size, dtype=np.uint64)) & 1).astype(np.float64)
        signs = bits * 2.0 - 1.0
        null = np.abs((signs * values[None, :]).mean(axis=1))
        # This branch enumerates the complete null distribution, so a
        # Monte-Carlo +1 correction would be inappropriate.
        return float(np.count_nonzero(null >= observed) / null.size)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(100_000, values.size))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (null.size + 1))


def _holm_adjust(entries: list[tuple[dict[str, Any], float]]) -> None:
    ordered = sorted(entries, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (record, raw) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * raw)
        running = max(running, adjusted)
        record["holm_adjusted_p"] = running


def aggregate_reports(
    reports: list[dict[str, Any]],
    reference: str = "chord_mappo",
    bootstrap_samples: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Aggregate scenario means across independent policy-training seeds.

    Every policy is evaluated on identical episode seeds, so baseline comparisons
    use paired differences at the training-seed level.
    """
    learned = [report for report in reports if "training_seed" in report]
    descriptive = [report for report in reports if "training_seed" not in report]
    algorithms = sorted({report["algorithm"] for report in learned})
    scenarios = sorted(
        {scenario for report in learned for scenario in report.get("scenarios", {})}
    )
    output: dict[str, Any] = {
        "statistical_unit": "independent training seed",
        "reference": reference,
        "algorithms": {},
        "paired_differences": {},
        "operational_differences": {},
        "descriptive_nonlearned": descriptive,
    }

    lookup: dict[tuple[str, int, str, str], float] = {}
    for report in learned:
        algorithm = report["algorithm"]
        training_seed = int(report["training_seed"])
        for scenario, metrics in report.get("scenarios", {}).items():
            for metric, stats in metrics.items():
                lookup[(algorithm, training_seed, scenario, metric)] = float(stats["mean"])

    row_index = 0
    for algorithm in algorithms:
        algorithm_output: dict[str, Any] = {}
        for scenario in scenarios:
            metric_names = sorted(
                {
                    metric
                    for (alg, _, scen, metric) in lookup
                    if alg == algorithm and scen == scenario
                }
            )
            scenario_output: dict[str, Any] = {}
            for metric in metric_names:
                keyed = sorted(
                    (training_seed, value)
                    for (alg, training_seed, scen, met), value in lookup.items()
                    if alg == algorithm and scen == scenario and met == metric
                )
                values = np.asarray([value for _, value in keyed], dtype=np.float64)
                low, high = _seed_bootstrap(values, bootstrap_samples, seed + row_index)
                row_index += 1
                scenario_output[metric] = {
                    "n_training_seeds": int(values.size),
                    "mean": float(values.mean()),
                    "std_across_seeds": (
                        float(values.std(ddof=1)) if values.size > 1 else None
                    ),
                    "ci95_low": low,
                    "ci95_high": high,
                    "training_seeds": [training_seed for training_seed, _ in keyed],
                }
            algorithm_output[scenario] = scenario_output
        output["algorithms"][algorithm] = algorithm_output

    if reference not in algorithms:
        return output
    for algorithm in algorithms:
        if algorithm == reference:
            continue
        comparison: dict[str, Any] = {}
        for scenario in scenarios:
            comparison[scenario] = {}
            metric_names = sorted(
                {
                    metric
                    for (alg, _, scen, metric) in lookup
                    if alg in {reference, algorithm} and scen == scenario
                }
            )
            for metric in metric_names:
                reference_by_seed = {
                    training_seed: value
                    for (alg, training_seed, scen, met), value in lookup.items()
                    if alg == reference and scen == scenario and met == metric
                }
                baseline_by_seed = {
                    training_seed: value
                    for (alg, training_seed, scen, met), value in lookup.items()
                    if alg == algorithm and scen == scenario and met == metric
                }
                common = sorted(reference_by_seed.keys() & baseline_by_seed.keys())
                differences = np.asarray(
                    [reference_by_seed[s] - baseline_by_seed[s] for s in common],
                    dtype=np.float64,
                )
                low, high = _seed_bootstrap(
                    differences, bootstrap_samples, seed + row_index
                )
                row_index += 1
                comparison[scenario][metric] = {
                    "definition": f"{reference} minus {algorithm}",
                    "n_paired_training_seeds": int(differences.size),
                    "mean_difference": (
                        float(differences.mean()) if differences.size else None
                    ),
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_effect_size_dz": (
                        float(differences.mean() / differences.std(ddof=1))
                        if differences.size > 1 and differences.std(ddof=1) > 0
                        else None
                    ),
                    "sign_flip_p": _paired_sign_flip_pvalue(
                        differences, seed + row_index
                    ),
                    "minimum_attainable_two_sided_p": (
                        2.0 / float(1 << differences.size)
                        if 1 <= differences.size <= 20
                        else None
                    ),
                    "training_seeds": common,
                }
        output["paired_differences"][algorithm] = comparison

    # A deterministic operational method has no policy-training seed. Compare
    # its seed-matched episode mean with every independently trained reference
    # policy, so uncertainty remains over policy-training seeds rather than
    # pretending evaluation episodes are independent trained policies.
    reference_seeds = sorted(
        {training_seed for (algorithm, training_seed, _, _) in lookup if algorithm == reference}
    )
    for report in descriptive:
        operational = report["algorithm"]
        comparison: dict[str, Any] = {}
        for scenario, metrics in report.get("scenarios", {}).items():
            comparison[scenario] = {}
            for metric, stats in metrics.items():
                baseline_mean = float(stats["mean"])
                differences = np.asarray(
                    [
                        lookup[(reference, training_seed, scenario, metric)] - baseline_mean
                        for training_seed in reference_seeds
                        if (reference, training_seed, scenario, metric) in lookup
                    ],
                    dtype=np.float64,
                )
                low, high = _seed_bootstrap(
                    differences, bootstrap_samples, seed + row_index
                )
                row_index += 1
                comparison[scenario][metric] = {
                    "definition": f"{reference} minus {operational}",
                    "n_reference_training_seeds": int(differences.size),
                    "mean_difference": float(differences.mean()) if differences.size else None,
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_effect_size_dz": (
                        float(differences.mean() / differences.std(ddof=1))
                        if differences.size > 1 and differences.std(ddof=1) > 0
                        else None
                    ),
                    "sign_flip_p": _paired_sign_flip_pvalue(differences, seed + row_index),
                    "minimum_attainable_two_sided_p": (
                        2.0 / float(1 << differences.size)
                        if 1 <= differences.size <= 20
                        else None
                    ),
                    "training_seeds": reference_seeds,
                }
        output["operational_differences"][operational] = comparison
    # Control family-wise error separately for every scenario/metric family.
    for scenario in scenarios:
        metric_names = {
            metric
            for comparison in output["paired_differences"].values()
            for metric in comparison.get(scenario, {})
        }
        for metric in metric_names:
            family = []
            for comparison in output["paired_differences"].values():
                record = comparison.get(scenario, {}).get(metric)
                if record is not None and record["sign_flip_p"] is not None:
                    family.append((record, float(record["sign_flip_p"])))
            _holm_adjust(family)
    # The operational comparison is a separate confirmatory family.
    for scenario in scenarios:
        metric_names = {
            metric
            for comparison in output["operational_differences"].values()
            for metric in comparison.get(scenario, {})
        }
        for metric in metric_names:
            family = []
            for comparison in output["operational_differences"].values():
                record = comparison.get(scenario, {}).get(metric)
                if record is not None and record["sign_flip_p"] is not None:
                    family.append((record, float(record["sign_flip_p"])))
            _holm_adjust(family)
    return output


def write_aggregate(report: dict[str, Any], output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "aggregate_by_training_seed.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    for algorithm, scenarios in report["algorithms"].items():
        for scenario, metrics in scenarios.items():
            for metric, stats in metrics.items():
                rows.append(
                    {"algorithm": algorithm, "scenario": scenario, "metric": metric, **stats}
                )
    if rows:
        with (target / "aggregate_by_training_seed.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for section, baseline_field in (
        ("paired_differences", "baseline"),
        ("operational_differences", "operational_baseline"),
    ):
        difference_rows: list[dict[str, Any]] = []
        for baseline, scenarios in report[section].items():
            for scenario, metrics in scenarios.items():
                for metric, statistics in metrics.items():
                    difference_rows.append(
                        {
                            baseline_field: baseline,
                            "scenario": scenario,
                            "metric": metric,
                            **statistics,
                        }
                    )
        if difference_rows:
            with (target / f"{section}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(difference_rows[0]))
                writer.writeheader()
                writer.writerows(difference_rows)
