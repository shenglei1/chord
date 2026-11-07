"""
The realized conflict-pressure proxy uses episode-mean decoder counts and
removes the selected coalition itself from the masked count:

    max(masked_coalitions - selected_coalitions, 0) / candidate_coalitions

It estimates the fraction of additional candidate coalitions removed because
they share a task, AGV, or picker with selected reservations. Because it is a
ratio of episode means rather than a mean of per-step ratios, it is a coarse
pressure proxy. It remains policy-trajectory dependent and supports
association, not causality.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FORMAL_SEEDS = (1, 45, 789, 2027, 4096, 8191)
METHODS = ("chord_mappo", "entity_cahr", "graph_cahr")
CONTROLS = ("entity_cahr", "graph_cahr")
DEFAULT_SCENARIOS = ("unseen_small", "fixed_team_dynamic_orders")
OUTCOME_DIRECTIONS = {
    "service_level": 1.0,
    "throughput_per_100_steps": 1.0,
    "deadline_miss_rate": -1.0,
    "mean_flow_time": -1.0,
}
PROXIES = (
    "pooled_conflict_elimination_fraction",
    "pooled_candidate_coalitions_per_step",
)
STRATA = ("low", "medium", "high")


def _method_root(results_root: Path, method: str) -> Path:
    if method == "chord_mappo":
        return results_root / "main_comparison" / method
    return results_root / "mechanism_controls" / method


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_bootstrap(
    values: np.ndarray, samples: int, seed: int
) -> tuple[float | None, float | None]:
    if values.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _sign_flip_pvalue(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    observed = abs(float(values.mean()))
    indexes = np.arange(1 << values.size, dtype=np.uint64)[:, None]
    bits = ((indexes >> np.arange(values.size, dtype=np.uint64)) & 1).astype(float)
    signs = bits * 2.0 - 1.0
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.count_nonzero(null >= observed) / null.size)


def _aggregate_seed_values(
    values: list[float], samples: int, seed: int
) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    low, high = _seed_bootstrap(array, samples, seed)
    std = float(array.std(ddof=1)) if array.size > 1 else None
    return {
        "n_training_seeds": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "std_across_training_seeds": std,
        "ci95_low": low,
        "ci95_high": high,
        "paired_effect_size_dz": (
            float(array.mean() / std) if std is not None and std > 0 else None
        ),
        "sign_flip_p": _sign_flip_pvalue(array),
        "minimum_attainable_two_sided_p": (
            2.0 / float(1 << array.size) if 1 <= array.size <= 20 else None
        ),
    }


def _rank_strata(values: list[tuple[int, float]]) -> dict[int, str]:
    """Assign equal-count within-seed strata using deterministic ranks."""

    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    count = len(ordered)
    output: dict[int, str] = {}
    for rank, (episode, _) in enumerate(ordered):
        index = min(len(STRATA) - 1, rank * len(STRATA) // max(count, 1))
        output[episode] = STRATA[index]
    return output


def _episode_record(
    row: dict[str, str], method: str, training_seed: int
) -> dict[str, Any]:
    candidate = float(row["candidate_coalitions_per_step"])
    selected = float(row["selected_coalitions_per_step"])
    masked = float(row["masked_coalitions_per_step"])
    additionally_eliminated = max(masked - selected, 0.0)
    return {
        "method": method,
        "training_seed": training_seed,
        "scenario": row["scenario"],
        "episode": int(row["episode"]),
        "candidate_coalitions_per_step": candidate,
        "selected_coalitions_per_step": selected,
        "masked_coalitions_per_step": masked,
        "reservation_rounds_per_step": float(row["reservation_rounds_per_step"]),
        "additional_conflicting_coalitions_per_step": additionally_eliminated,
        "conflict_elimination_fraction": (
            additionally_eliminated / candidate if candidate > 0 else 0.0
        ),
        "conflicting_alternatives_per_selection": (
            additionally_eliminated / selected if selected > 0 else 0.0
        ),
        **{metric: float(row[metric]) for metric in OUTCOME_DIRECTIONS},
    }


def collect_episodes(
    results_root: Path,
    scenarios: Iterable[str],
    seeds: Iterable[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_scenarios = set(scenarios)
    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for method in METHODS:
        for training_seed in seeds:
            source = (
                _method_root(results_root, method)
                / f"seed_{training_seed}"
                / "evaluation"
                / "episodes.csv"
            )
            if not source.exists():
                raise FileNotFoundError(f"missing evaluation episodes: {source}")
            rows = [row for row in _read_csv(source) if row["scenario"] in selected_scenarios]
            by_scenario: dict[str, int] = defaultdict(int)
            for row in rows:
                by_scenario[row["scenario"]] += 1
                records.append(_episode_record(row, method, training_seed))
            for scenario in scenarios:
                audit.append(
                    {
                        "method": method,
                        "training_seed": training_seed,
                        "scenario": scenario,
                        "episode_count": by_scenario.get(scenario, 0),
                        "source": str(source),
                    }
                )
                if by_scenario.get(scenario, 0) != 100:
                    raise ValueError(
                        f"expected 100 episodes for {method}, seed {training_seed}, "
                        f"scenario {scenario}; found {by_scenario.get(scenario, 0)}"
                    )
    return records, audit


def add_pooled_proxies_and_strata(records: list[dict[str, Any]]) -> None:
    paired: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        paired[(row["scenario"], row["training_seed"], row["episode"])].append(row)
    for key, group in paired.items():
        if {row["method"] for row in group} != set(METHODS):
            raise ValueError(f"incomplete paired episode: {key}")
        pooled_conflict = float(
            np.mean([row["conflict_elimination_fraction"] for row in group])
        )
        pooled_candidate = float(
            np.mean([row["candidate_coalitions_per_step"] for row in group])
        )
        for row in group:
            row["pooled_conflict_elimination_fraction"] = pooled_conflict
            row["pooled_candidate_coalitions_per_step"] = pooled_candidate

    for scenario, training_seed in sorted(
        {(row["scenario"], row["training_seed"]) for row in records}
    ):
        reference_rows = {
            row["episode"]: row
            for row in records
            if row["scenario"] == scenario
            and row["training_seed"] == training_seed
            and row["method"] == "chord_mappo"
        }
        for proxy in PROXIES:
            assignment = _rank_strata(
                [(episode, float(row[proxy])) for episode, row in reference_rows.items()]
            )
            for row in records:
                if row["scenario"] == scenario and row["training_seed"] == training_seed:
                    row[f"{proxy}_stratum"] = assignment[row["episode"]]


def summarize_strata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        *OUTCOME_DIRECTIONS,
        "candidate_coalitions_per_step",
        "selected_coalitions_per_step",
        "masked_coalitions_per_step",
        "reservation_rounds_per_step",
        "additional_conflicting_coalitions_per_step",
        "conflict_elimination_fraction",
        "conflicting_alternatives_per_selection",
    )
    groups: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        for proxy in PROXIES:
            key = (
                proxy,
                row["scenario"],
                row["training_seed"],
                row["method"],
                row[f"{proxy}_stratum"],
            )
            groups[key].append(row)
    output: list[dict[str, Any]] = []
    for (proxy, scenario, seed, method, stratum), rows in sorted(groups.items()):
        output.append(
            {
                "proxy": proxy,
                "scenario": scenario,
                "training_seed": seed,
                "method": method,
                "stratum": stratum,
                "n_paired_episodes": len(rows),
                "stratum_proxy_mean": float(np.mean([row[proxy] for row in rows])),
                **{
                    field: float(np.mean([row[field] for row in rows]))
                    for field in fields
                },
            }
        )
    return output


def stratified_effects(
    seed_means: list[dict[str, Any]], samples: int, seed: int
) -> list[dict[str, Any]]:
    lookup = {
        (
            row["proxy"],
            row["scenario"],
            row["training_seed"],
            row["method"],
            row["stratum"],
        ): row
        for row in seed_means
    }
    output: list[dict[str, Any]] = []
    row_index = 0
    scenarios = sorted({row["scenario"] for row in seed_means})
    seeds = sorted({int(row["training_seed"]) for row in seed_means})
    for proxy in PROXIES:
        for scenario in scenarios:
            for control in CONTROLS:
                for stratum in STRATA:
                    for metric, direction in OUTCOME_DIRECTIONS.items():
                        per_seed = []
                        for training_seed in seeds:
                            chord = lookup[(proxy, scenario, training_seed, "chord_mappo", stratum)]
                            baseline = lookup[(proxy, scenario, training_seed, control, stratum)]
                            per_seed.append(direction * (chord[metric] - baseline[metric]))
                        statistics = _aggregate_seed_values(
                            per_seed, samples, seed + row_index
                        )
                        row_index += 1
                        output.append(
                            {
                                "proxy": proxy,
                                "scenario": scenario,
                                "control": control,
                                "stratum": stratum,
                                "metric": metric,
                                "definition": "positive favors CHORD",
                                **statistics,
                                "training_seed_values": json.dumps(per_seed),
                            }
                        )
    return output


def high_minus_low_effects(
    seed_means: list[dict[str, Any]], samples: int, seed: int
) -> list[dict[str, Any]]:
    lookup = {
        (
            row["proxy"],
            row["scenario"],
            row["training_seed"],
            row["method"],
            row["stratum"],
        ): row
        for row in seed_means
    }
    output: list[dict[str, Any]] = []
    row_index = 0
    scenarios = sorted({row["scenario"] for row in seed_means})
    seeds = sorted({int(row["training_seed"]) for row in seed_means})
    for proxy in PROXIES:
        for scenario in scenarios:
            for control in CONTROLS:
                for metric, direction in OUTCOME_DIRECTIONS.items():
                    interactions = []
                    for training_seed in seeds:
                        chord_high = lookup[(proxy, scenario, training_seed, "chord_mappo", "high")]
                        control_high = lookup[(proxy, scenario, training_seed, control, "high")]
                        chord_low = lookup[(proxy, scenario, training_seed, "chord_mappo", "low")]
                        control_low = lookup[(proxy, scenario, training_seed, control, "low")]
                        high_effect = direction * (chord_high[metric] - control_high[metric])
                        low_effect = direction * (chord_low[metric] - control_low[metric])
                        interactions.append(high_effect - low_effect)
                    statistics = _aggregate_seed_values(
                        interactions, samples, seed + row_index
                    )
                    row_index += 1
                    output.append(
                        {
                            "proxy": proxy,
                            "scenario": scenario,
                            "control": control,
                            "metric": metric,
                            "definition": (
                                "(CHORD-minus-control oriented effect in high stratum) "
                                "minus (effect in low stratum); positive supports concentration"
                            ),
                            **statistics,
                            "training_seed_values": json.dumps(interactions),
                        }
                    )
    return output


def continuous_effect_slopes(
    records: list[dict[str, Any]], samples: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["scenario"], row["training_seed"], row["episode"], row["method"]): row
        for row in records
    }
    scenarios = sorted({row["scenario"] for row in records})
    seeds = sorted({int(row["training_seed"]) for row in records})
    seed_rows: list[dict[str, Any]] = []
    for proxy in PROXIES:
        for scenario in scenarios:
            for training_seed in seeds:
                episodes = sorted(
                    row["episode"]
                    for row in records
                    if row["scenario"] == scenario
                    and row["training_seed"] == training_seed
                    and row["method"] == "chord_mappo"
                )
                x = np.asarray(
                    [
                        lookup[(scenario, training_seed, episode, "chord_mappo")][proxy]
                        for episode in episodes
                    ],
                    dtype=float,
                )
                x_std = float(x.std(ddof=0))
                xz = (x - x.mean()) / x_std if x_std > 0 else np.zeros_like(x)
                for control in CONTROLS:
                    for metric, direction in OUTCOME_DIRECTIONS.items():
                        y = np.asarray(
                            [
                                direction
                                * (
                                    lookup[(scenario, training_seed, episode, "chord_mappo")][metric]
                                    - lookup[(scenario, training_seed, episode, control)][metric]
                                )
                                for episode in episodes
                            ],
                            dtype=float,
                        )
                        slope = float(np.mean(xz * (y - y.mean()))) if x_std > 0 else 0.0
                        correlation = (
                            float(np.corrcoef(xz, y)[0, 1])
                            if x_std > 0 and float(y.std(ddof=0)) > 0
                            else 0.0
                        )
                        seed_rows.append(
                            {
                                "proxy": proxy,
                                "scenario": scenario,
                                "training_seed": training_seed,
                                "control": control,
                                "metric": metric,
                                "slope_per_one_sd_proxy": slope,
                                "episode_level_pearson_r": correlation,
                                "n_paired_episodes": len(episodes),
                            }
                        )
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["proxy"], row["scenario"], row["control"], row["metric"])].append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for row_index, (key, rows) in enumerate(sorted(groups.items())):
        proxy, scenario, control, metric = key
        slopes = [float(row["slope_per_one_sd_proxy"]) for row in rows]
        correlations = [float(row["episode_level_pearson_r"]) for row in rows]
        aggregate_rows.append(
            {
                "proxy": proxy,
                "scenario": scenario,
                "control": control,
                "metric": metric,
                "definition": (
                    "change in positive-favors-CHORD paired outcome per one-SD "
                    "increase in the pooled episode proxy"
                ),
                **_aggregate_seed_values(slopes, samples, seed + row_index),
                "mean_within_seed_episode_correlation": float(np.mean(correlations)),
                "training_seed_slopes": json.dumps(slopes),
            }
        )
    return seed_rows, aggregate_rows


def build_verdict(
    high_low: list[dict[str, Any]], slopes: list[dict[str, Any]]
) -> dict[str, Any]:
    primary_metrics = ("service_level", "throughput_per_100_steps")
    conflict_high_low = {
        (row["scenario"], row["control"], row["metric"]): row
        for row in high_low
        if row["proxy"] == "pooled_conflict_elimination_fraction"
    }
    conflict_slopes = {
        (row["scenario"], row["control"], row["metric"]): row
        for row in slopes
        if row["proxy"] == "pooled_conflict_elimination_fraction"
    }
    scenarios = sorted({key[0] for key in conflict_high_low})
    summary: dict[str, Any] = {}
    for scenario in scenarios:
        rows = []
        for control in CONTROLS:
            for metric in primary_metrics:
                high_low_row = conflict_high_low[(scenario, control, metric)]
                slope_row = conflict_slopes[(scenario, control, metric)]
                rows.append(
                    {
                        "control": control,
                        "metric": metric,
                        "high_minus_low_mean": high_low_row["mean"],
                        "high_minus_low_ci95_low": high_low_row["ci95_low"],
                        "high_minus_low_ci95_high": high_low_row["ci95_high"],
                        "continuous_slope_mean": slope_row["mean"],
                        "continuous_slope_ci95_low": slope_row["ci95_low"],
                        "continuous_slope_ci95_high": slope_row["ci95_high"],
                    }
                )
        confirmed = all(
            row["high_minus_low_ci95_low"] is not None
            and row["high_minus_low_ci95_low"] > 0
            and row["continuous_slope_ci95_low"] is not None
            and row["continuous_slope_ci95_low"] > 0
            for row in rows
        )
        directional = sum(
            row["high_minus_low_mean"] > 0 and row["continuous_slope_mean"] > 0
            for row in rows
        )
        summary[scenario] = {
            "confirmatory_pattern_for_all_primary_comparisons": confirmed,
            "directionally_consistent_primary_comparisons": directional,
            "total_primary_comparisons": len(rows),
            "details": rows,
        }
    return summary


def paper_summary_rows(
    effects: list[dict[str, Any]],
    high_low: list[dict[str, Any]],
    slopes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one compact row per scenario/control/primary outcome."""

    proxy = "pooled_conflict_elimination_fraction"
    primary_metrics = ("service_level", "throughput_per_100_steps")
    effect_lookup = {
        (row["scenario"], row["control"], row["metric"], row["stratum"]): row
        for row in effects
        if row["proxy"] == proxy and row["metric"] in primary_metrics
    }
    high_low_lookup = {
        (row["scenario"], row["control"], row["metric"]): row
        for row in high_low
        if row["proxy"] == proxy and row["metric"] in primary_metrics
    }
    slope_lookup = {
        (row["scenario"], row["control"], row["metric"]): row
        for row in slopes
        if row["proxy"] == proxy and row["metric"] in primary_metrics
    }
    output: list[dict[str, Any]] = []
    keys = sorted(high_low_lookup)
    for scenario, control, metric in keys:
        low = effect_lookup[(scenario, control, metric, "low")]
        medium = effect_lookup[(scenario, control, metric, "medium")]
        high = effect_lookup[(scenario, control, metric, "high")]
        interaction = high_low_lookup[(scenario, control, metric)]
        slope = slope_lookup[(scenario, control, metric)]
        output.append(
            {
                "scenario": scenario,
                "control": control,
                "metric": metric,
                "effect_definition": "positive favors CHORD",
                "low_overlap_effect": low["mean"],
                "medium_overlap_effect": medium["mean"],
                "high_overlap_effect": high["mean"],
                "high_minus_low": interaction["mean"],
                "high_minus_low_ci95_low": interaction["ci95_low"],
                "high_minus_low_ci95_high": interaction["ci95_high"],
                "continuous_slope_per_sd": slope["mean"],
                "continuous_slope_ci95_low": slope["ci95_low"],
                "continuous_slope_ci95_high": slope["ci95_high"],
                "n_training_seeds": interaction["n_training_seeds"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/mechanism_controls/overlap_posthoc")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output = Path(args.output)
    records, audit = collect_episodes(results_root, args.scenarios, args.seeds)
    add_pooled_proxies_and_strata(records)
    seed_means = summarize_strata(records)
    effects = stratified_effects(
        seed_means, args.bootstrap_samples, args.bootstrap_seed
    )
    high_low = high_minus_low_effects(
        seed_means, args.bootstrap_samples, args.bootstrap_seed + 10_000
    )
    seed_slopes, aggregate_slopes = continuous_effect_slopes(
        records, args.bootstrap_samples, args.bootstrap_seed + 20_000
    )
    verdict = build_verdict(high_low, aggregate_slopes)
    paper_summary = paper_summary_rows(effects, high_low, aggregate_slopes)

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "data_audit.csv", audit)
    _write_csv(output / "episode_overlap_proxy.csv", records)
    _write_csv(output / "stratified_seed_means.csv", seed_means)
    _write_csv(output / "stratified_paired_effects.csv", effects)
    _write_csv(output / "high_minus_low_effects.csv", high_low)
    _write_csv(output / "continuous_seed_slopes.csv", seed_slopes)
    _write_csv(output / "continuous_slope_summary.csv", aggregate_slopes)
    _write_csv(output / "paper_overlap_summary.csv", paper_summary)
    summary = {
        "analysis_type": "post-hoc associative mechanism analysis",
        "statistical_unit": "independent policy-training seed",
        "training_seeds": list(args.seeds),
        "scenarios": list(args.scenarios),
        "methods": list(METHODS),
        "episodes_per_method_seed_scenario": 100,
        "proxies": {
            "pooled_conflict_elimination_fraction": (
                "Across-method mean of max(mean_masked-mean_selected, 0)/mean_candidate "
                "for a paired episode. This ratio of episode-mean counts estimates the "
                "share of additional candidates removed by task/AGV/picker exclusivity "
                "after a reservation."
            ),
            "pooled_candidate_coalitions_per_step": (
                "Across-method mean number of feasible coalition candidates per step for "
                "a paired episode. It measures candidate-set load, not overlap alone."
            ),
        },
        "stratification": (
            "Equal-count low/medium/high ranks within each scenario and training seed; "
            "episode indices are paired across methods."
        ),
        "limitations": [
            "The proxies depend on policy-induced trajectories and cannot establish physical causality.",
            "Evaluation episodes are used only within seed; n remains six independent training seeds.",
            "Scenario and proxy analyses are post-hoc and should be labeled exploratory.",
            "A positive high-minus-low contrast supports concentration of an encoder effect but does not prove hypergraph necessity.",
        ],
        "verdict": verdict,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
