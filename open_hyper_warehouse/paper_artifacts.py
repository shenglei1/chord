"""Audited statistics and publication figures for the CHORD warehouse study.

The independent statistical unit is an independently trained policy seed.  The
100 evaluation episodes associated with one checkpoint are used to estimate
that checkpoint's scenario mean; they are never treated as 100 independent
trained policies.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FORMAL_SEEDS = (1, 45, 789, 2027, 4096, 8191)
CORE_SCENARIOS = (
    "in_distribution",
    "unseen_small",
    "unseen_large",
    "agv_heavy_ratio",
    "picker_heavy_ratio",
    "picker_shift",
    "fixed_team_dynamic_orders",
    "robot_breakdown",
)
# Final-policy comparison for the application section.  Population-size
# extrapolation is shown separately as an ordered scale curve, so the two
# unseen-size scenarios are not repeated in this categorical application plot.
APPLICATION_SCENARIOS = (
    "in_distribution",
    "agv_heavy_ratio",
    "picker_heavy_ratio",
    "picker_shift",
    "fixed_team_dynamic_orders",
    "robot_breakdown",
)
ABLATION_SCENARIOS = ("in_distribution", "agv_heavy_ratio", "robot_breakdown")
LEARNED_METHODS = (
    "chord_mappo",
    "plato_faithful",
    "mohito_official",
    "entity_mappo",
    "graph_mappo",
)
PRIMARY_METRICS = (
    "service_level",
    "throughput_per_100_steps",
    "deadline_miss_rate",
    "mean_flow_time",
)

# The reservation ablation must include its direct mechanism-level outcome.
# Mean flow time remains part of the main comparison and encoder-control plots,
# while contention/delivery tests whether the reservation decoder actually
# removes mutually incompatible same-role proposals.
COMPONENT_ABLATION_METRICS = (
    "service_level",
    "throughput_per_100_steps",
    "contention_per_delivery",
    "deadline_miss_rate",
)

METHOD_LABELS = {
    "chord_mappo": "CHORD-CAHR",
    "plato_faithful": "PLATO-faithful",
    "mohito_official": "MOHITO-official",
    "entity_mappo": "Entity-MAPPO",
    "graph_mappo": "Graph-MAPPO",
    "rolling_horizon": "Rolling-horizon MILP",
    "entity_cahr": "Entity+CAHR",
    "graph_cahr": "Graph+CAHR",
    "chord_no_reservation": "No reservation / no capacity",
    "chord_capacity_factorized": "Capacity + factorized decoder",
    "chord_no_capacity": "Reservation without capacity",
}

METHOD_STYLES = {
    "chord_mappo": ("#0072B2", "o", "-"),
    "plato_faithful": ("#D55E00", "s", "--"),
    "mohito_official": ("#CC79A7", "P", "-."),
    "entity_mappo": ("#009E73", "^", ":"),
    "graph_mappo": ("#E69F00", "D", "--"),
    "rolling_horizon": ("#4D4D4D", "X", ":"),
    "entity_cahr": ("#56B4E9", "v", "--"),
    # Darkened Okabe--Ito yellow so the marker remains legible on white paper.
    "graph_cahr": ("#B79F00", "<", "-."),
    "chord_no_reservation": ("#999999", "s", "--"),
    "chord_capacity_factorized": ("#009E73", "^", "-."),
    "chord_no_capacity": ("#D55E00", "D", ":"),
}

SCENARIO_LABELS = {
    "in_distribution": "In-distribution",
    "unseen_small": "Unseen small",
    "unseen_large": "Unseen large",
    "agv_heavy_ratio": "AGV-heavy",
    "picker_heavy_ratio": "Picker-heavy",
    "picker_shift": "Picker shift",
    "fixed_team_dynamic_orders": "Dynamic orders",
    "robot_breakdown": "Robot breakdown",
}

METRIC_DEFINITIONS = {
    "service_level": {
        "label": "Service level",
        "direction": "max",
        "unit": "ratio",
        "definition": "deliveries / max(orders_created - cancellations, 1)",
    },
    "throughput_per_100_steps": {
        "label": "Throughput / 100 steps",
        "direction": "max",
        "unit": "deliveries per 100 simulator steps",
        "definition": "100 * deliveries / max(episode_steps, 1)",
    },
    "deadline_miss_rate": {
        "label": "Deadline-miss rate",
        "direction": "min",
        "unit": "ratio",
        "definition": "deadline_misses / max(orders_created - cancellations, 1)",
    },
    "mean_flow_time": {
        "label": "Mean flow time",
        "direction": "min",
        "unit": "simulator steps",
        "definition": "sum(delivery_time - creation_time + 1) / max(deliveries, 1)",
    },
    "contention_per_delivery": {
        "label": "Rejected duplicate proposals / delivery",
        "direction": "min",
        "unit": "proposal conflicts per delivery",
        "definition": "rejected same-role, same-task proposals / max(deliveries, 1)",
    },
    "travel_per_delivery": {
        "label": "Travel / delivery",
        "direction": "min",
        "unit": "grid moves per delivery",
        "definition": "agent grid moves / max(deliveries, 1)",
    },
    "utilization": {
        "label": "Agent utilization",
        "direction": "context",
        "unit": "ratio",
        "definition": "busy active-agent steps / active-agent steps",
    },
    "unserved_orders": {
        "label": "Unserved orders at horizon",
        "direction": "min",
        "unit": "orders",
        "definition": "active tasks remaining at episode termination",
    },
}

# Self-contained axis labels for standalone panels.  A bare unit such as
# "ratio" is insufficient once a panel is separated from a combined figure.
METRIC_AXIS_LABELS = {
    "service_level": "Service level (ratio)",
    "throughput_per_100_steps": "Deliveries per 100 simulator steps",
    "deadline_miss_rate": "Deadline-miss rate (ratio)",
    "mean_flow_time": "Mean flow time (simulator steps)",
    "contention_per_delivery": "Rejected duplicate proposals per delivery",
}

REQUIRED_EPISODE_COLUMNS = {
    "scenario",
    "episode",
    "return",
    "deliveries",
    "completion_ratio",
    "service_level",
    "throughput_per_100_steps",
    "deadline_misses",
    "deadline_miss_rate",
    "contentions",
    "travel",
    "utilization",
    "mean_flow_time",
    "cancellations",
    "orders_offered",
    "contention_per_delivery",
    "travel_per_delivery",
    "unserved_orders",
}


def bootstrap_ci(
    values: Iterable[float], samples: int = 10_000, seed: int = 2026
) -> tuple[float | None, float | None]:
    """Percentile CI over independent training seeds."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def interval(
    values: Iterable[float], mode: str, samples: int = 10_000, seed: int = 2026
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    mean = float(array.mean())
    if mode == "std":
        spread = float(array.std(ddof=1)) if array.size > 1 else 0.0
        return mean, mean - spread, mean + spread
    if mode != "ci95":
        raise ValueError(f"unknown uncertainty mode: {mode}")
    low, high = bootstrap_ci(array, samples=samples, seed=seed)
    return mean, mean if low is None else low, mean if high is None else high


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_seed_summaries(
    method_root: Path,
) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
    """Recompute each checkpoint's scenario means directly from episodes.csv.

    JSON summaries remain audit targets, not plotting inputs.  This keeps every
    paper mark traceable to the row-level CSV evidence requested by the paper
    protocol.
    """
    summaries: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    for seed_dir in sorted(method_root.glob("seed_*")):
        episodes = seed_dir / "evaluation" / "episodes.csv"
        if not episodes.exists():
            continue
        rows = _read_csv(episodes)
        by_scenario: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_scenario[row["scenario"]].append(row)
        scenario_summaries: dict[str, dict[str, dict[str, float]]] = {}
        for scenario, scenario_rows in by_scenario.items():
            metric_names = [
                key for key in scenario_rows[0]
                if key not in {"scenario", "episode"}
            ]
            scenario_summaries[scenario] = {
                metric: {
                    "mean": float(np.mean([float(row[metric]) for row in scenario_rows]))
                }
                for metric in metric_names
            }
        summaries[int(seed_dir.name.rsplit("_", 1)[-1])] = scenario_summaries
    return summaries


def collect_main_seed_means(
    results_root: Path,
) -> dict[str, dict[int, dict[str, dict[str, dict[str, float]]]]]:
    root = results_root / "main_comparison"
    return {method: load_seed_summaries(root / method) for method in LEARNED_METHODS}


def _episode_formula_issues(rows: list[dict[str, str]], source: str) -> list[str]:
    issues: list[str] = []
    for row_index, row in enumerate(rows, start=2):
        deliveries = float(row["deliveries"])
        offered = float(row["orders_offered"])
        cancellations = float(row["cancellations"])
        checks = {
            "contention_per_delivery": float(row["contentions"])
            / max(deliveries, 1.0),
            "travel_per_delivery": float(row["travel"]) / max(deliveries, 1.0),
        }
        # orders_created is not exported.  When a delivery exists it can be
        # recovered from completion_ratio, allowing the two eligible-order
        # rates to be checked without incorrectly treating orders_offered as
        # orders_created under a future saturated task buffer.
        completion = float(row["completion_ratio"])
        if deliveries > 0 and completion > 0:
            inferred_created = deliveries / completion
            eligible = max(inferred_created - cancellations, 1.0)
            checks["service_level"] = deliveries / eligible
            checks["deadline_miss_rate"] = float(row["deadline_misses"]) / eligible
            if inferred_created > offered + 1e-7:
                issues.append(f"{source}:{row_index}: inferred created orders exceed offers")
        for metric, expected in checks.items():
            if not np.isclose(float(row[metric]), expected, atol=1e-10, rtol=1e-9):
                issues.append(f"{source}:{row_index}: {metric} formula mismatch")
                if len(issues) >= 20:
                    return issues
    return issues


def audit_results(results_root: Path) -> dict[str, Any]:
    """Reconcile episode CSVs, JSON summaries and seed-level aggregates."""
    main = results_root / "main_comparison"
    errors: list[str] = []
    warnings: list[str] = []
    algorithms: dict[str, Any] = {}
    recomputed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    hardware: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inferred_rejected_offer_episodes = 0

    for method in LEARNED_METHODS:
        method_root = main / method
        seen_seeds: list[int] = []
        for seed_dir in sorted(method_root.glob("seed_*")):
            episode_path = seed_dir / "evaluation" / "episodes.csv"
            summary_path = seed_dir / "evaluation" / "summary.json"
            if not episode_path.exists() or not summary_path.exists():
                errors.append(f"{seed_dir}: missing episodes.csv or summary.json")
                continue
            report = json.loads(summary_path.read_text(encoding="utf-8"))
            seed = int(report["training_seed"])
            seen_seeds.append(seed)
            rows = _read_csv(episode_path)
            if not rows:
                errors.append(f"{episode_path}: empty")
                continue
            missing = REQUIRED_EPISODE_COLUMNS - set(rows[0])
            if missing:
                errors.append(f"{episode_path}: missing columns {sorted(missing)}")
                continue
            counts = Counter(row["scenario"] for row in rows)
            if tuple(counts) != CORE_SCENARIOS or set(counts.values()) != {100}:
                errors.append(f"{episode_path}: expected Core8 x 100, got {dict(counts)}")
            errors.extend(_episode_formula_issues(rows, str(episode_path)))
            for row in rows:
                deliveries = float(row["deliveries"])
                completion = float(row["completion_ratio"])
                if deliveries > 0 and completion > 0:
                    inferred_created = deliveries / completion
                    if float(row["orders_offered"]) - inferred_created > 1e-7:
                        inferred_rejected_offer_episodes += 1
            for scenario in CORE_SCENARIOS:
                scenario_rows = [row for row in rows if row["scenario"] == scenario]
                for metric in PRIMARY_METRICS:
                    value = float(np.mean([float(row[metric]) for row in scenario_rows]))
                    reported = float(report["scenarios"][scenario][metric]["mean"])
                    if not np.isclose(value, reported, atol=1e-10, rtol=1e-9):
                        errors.append(
                            f"{summary_path}: {scenario}/{metric} does not match episodes.csv"
                        )
                    recomputed[(method, scenario, metric)].append(value)
            hardware[method].append(
                {
                    "training_seed": seed,
                    "device": report.get("training_device"),
                    "gpu": report.get("cuda_device_name"),
                    "parameter_count": report.get("parameter_count"),
                    "training_wall_seconds": report.get("training_wall_seconds"),
                    "inference_latency_ms": report.get("inference_latency_ms"),
                }
            )
        if sorted(seen_seeds) != sorted(FORMAL_SEEDS):
            errors.append(f"{method}: expected seeds {FORMAL_SEEDS}, got {sorted(seen_seeds)}")
        algorithms[method] = {
            "training_seeds": sorted(seen_seeds),
            "n_training_seeds": len(seen_seeds),
            "episodes_per_seed": 800,
        }

    aggregate_path = main / "aggregate" / "aggregate_by_training_seed.csv"
    if not aggregate_path.exists():
        errors.append(f"missing {aggregate_path}")
    else:
        for row in _read_csv(aggregate_path):
            key = (row["algorithm"], row["scenario"], row["metric"])
            if key not in recomputed:
                continue
            expected = float(np.mean(recomputed[key]))
            if not np.isclose(expected, float(row["mean"]), atol=1e-10, rtol=1e-9):
                errors.append(f"{aggregate_path}: aggregate mismatch for {key}")

    rolling = main / "rolling_horizon" / "episodes.csv"
    if not rolling.exists() or len(_read_csv(rolling)) != 800:
        errors.append("rolling_horizon: expected one deterministic Core8 x 100 evaluation")

    gpu_names = {
        row["gpu"]
        for rows in hardware.values()
        for row in rows
        if row.get("gpu")
    }
    if len(gpu_names) > 1 or any(
        row.get("gpu") is None for rows in hardware.values() for row in rows
    ):
        warnings.append(
            "Inference latency and training wall time were recorded on mixed or unidentified "
            "GPUs; do not make cross-method efficiency claims without a same-machine benchmark."
        )

    sensitivity = results_root / "sensitivity" / "chord" / "aggregate" / "aggregate_by_training_seed.csv"
    sensitivity_rejected_offer_episodes = None
    if not sensitivity.exists():
        warnings.append(
            "Sensitivity results are absent. Evaluate the six existing CHORD checkpoints; "
            "retraining is not required."
        )
    else:
        # completion_ratio = deliveries / orders_created in the environment.
        # Every completed sensitivity row has positive deliveries, so task-slot
        # rejection can be audited without inventing orders_created values.
        sensitivity_rejected_offer_episodes = 0
        sensitivity_root = results_root / "sensitivity" / "chord"
        for episode_path in sensitivity_root.glob("seed_*/episodes.csv"):
            for row in _read_csv(episode_path):
                deliveries = float(row["deliveries"])
                completion = float(row["completion_ratio"])
                if deliveries <= 0 or completion <= 0:
                    warnings.append(
                        f"{episode_path}: cannot infer created orders for a zero-delivery row"
                    )
                    continue
                inferred_created = deliveries / completion
                if float(row["orders_offered"]) - inferred_created > 1e-7:
                    sensitivity_rejected_offer_episodes += 1
    warnings.append(
        "The three-scene ablation runner changes scenario indices, hence evaluation episode "
        "seeds, relative to Core8. Means are usable; strict paired-trajectory claims require "
        "reevaluating existing checkpoints under one common scenario order, not retraining."
    )
    semantic_summary_path = results_root / "semantic_analysis" / "semantic_paper_summary.json"
    semantic_available = semantic_summary_path.exists()
    if semantic_available:
        semantic_summary = json.loads(semantic_summary_path.read_text(encoding="utf-8"))
        semantic_seeds = tuple(int(value) for value in semantic_summary["formal_training_seeds"])
        if semantic_seeds != FORMAL_SEEDS:
            errors.append(
                f"{semantic_summary_path}: expected semantic seeds {FORMAL_SEEDS}, "
                f"got {semantic_seeds}"
            )
        if int(semantic_summary["execution_episodes_per_seed"]) != 20:
            errors.append(f"{semantic_summary_path}: expected 20 execution episodes per seed")
        warnings.append(
            "Semantic masking is model-internal structural sensitivity, not physical causality."
        )
    else:
        warnings.append("Six-seed semantic-analysis summary is absent.")

    return {
        "status": "pass_with_warnings" if not errors else "fail",
        "statistical_unit": "independent policy-training seed",
        "evaluation_episodes_per_seed_scenario": 100,
        "algorithms": algorithms,
        "hardware_records": hardware,
        "core8_episodes_with_inferred_rejected_offers": inferred_rejected_offer_episodes,
        "sensitivity_episodes_with_inferred_rejected_offers": sensitivity_rejected_offer_episodes,
        "errors": errors,
        "warnings": warnings,
        "sensitivity_available": sensitivity.exists(),
        "semantic_available": semantic_available,
    }


def _setup_matplotlib():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Paper plotting requires matplotlib. Install with `python -m pip install -e .[plot]`."
        ) from exc
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )
    return plt


def _save(fig, base: Path, formats: Iterable[str]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(base.with_suffix(f".{extension}"))


def _ordered_legend_entries(
    handles: Iterable[Any], labels: Iterable[str], ncol: int
) -> tuple[list[Any], list[str]]:
    """Return unique, method-ordered entries displayed row-major by Matplotlib."""
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in unique:
            unique[label] = handle
    priority = {label: index for index, label in enumerate(METHOD_LABELS.values())}
    ordered = sorted(unique, key=lambda label: (priority.get(label, 10_000), label))
    if not ordered:
        return [], []
    columns = max(1, min(ncol, len(ordered)))
    rows = int(np.ceil(len(ordered) / columns))
    # Matplotlib fills legends down columns. Reorder input so the visual result
    # reads left-to-right across rows: CHORD first, then the named baselines.
    column_major: list[str] = []
    for column in range(columns):
        for row in range(rows):
            index = row * columns + column
            if index < len(ordered):
                column_major.append(ordered[index])
    return [unique[label] for label in column_major], column_major


def _save_individual_panels(
    fig,
    axes,
    base: Path,
    panel_names: Iterable[str],
    formats: Iterable[str],
    *,
    legend_ncol: int | None = None,
) -> None:
    """Export clean subplot crops for assembly with a shared legend in LaTeX."""
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    panel_names = tuple(panel_names)
    if len(flat_axes) != len(panel_names):
        raise ValueError("panel_names must match the number of axes")
    base.parent.mkdir(parents=True, exist_ok=True)
    # Hide the combined figure legend while cropping panels.  A separate shared
    # legend is exported below, which avoids repeated method names and prevents
    # legend text from colliding with the x-axis title in half-width panels.
    figure_legends = list(fig.legends)
    previous_visibility = [legend.get_visible() for legend in figure_legends]
    for figure_legend in figure_legends:
        figure_legend.set_visible(False)
    try:
        for panel_index, (ax, panel_name) in enumerate(zip(flat_axes, panel_names)):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = ax.get_tightbbox(renderer)
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).expanded(1.04, 1.06)
            panel_letter = chr(ord("A") + panel_index)
            panel_base = base.parent / f"{base.name}_{panel_letter}_{panel_name}"
            for extension in formats:
                fig.savefig(panel_base.with_suffix(f".{extension}"), bbox_inches=bbox)
    finally:
        for figure_legend, visible in zip(figure_legends, previous_visibility):
            figure_legend.set_visible(visible)


def _save_shared_legend(
    fig,
    axes,
    base: Path,
    formats: Iterable[str],
    *,
    ncol: int,
) -> None:
    """Export one non-overlapping legend for a group of standalone panels."""
    import matplotlib.pyplot as plt

    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    handles, labels = _ordered_legend_entries(handles, labels, ncol)
    if not handles:
        return
    legend_fig = plt.figure(figsize=(7.25, 0.42))
    legend_fig.legend(
        handles,
        labels,
        loc="center",
        ncol=min(ncol, len(handles)),
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.55,
    )
    _save(legend_fig, base.parent / f"{base.name}_legend", formats)
    plt.close(legend_fig)


def _values(
    data: dict[str, dict[int, dict[str, dict[str, dict[str, float]]]]],
    method: str,
    scenario: str,
    metric: str,
) -> list[float]:
    return [
        float(data[method][seed][scenario][metric]["mean"])
        for seed in FORMAL_SEEDS
        if seed in data.get(method, {})
    ]


def _rolling_means(results_root: Path) -> dict[str, dict[str, float]]:
    path = results_root / "main_comparison" / "rolling_horizon" / "summary.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        scenario: {metric: float(stats["mean"]) for metric, stats in metrics.items()}
        for scenario, metrics in report["scenarios"].items()
    }


def plot_main_metrics(
    results_root: Path,
    output: Path,
    uncertainty: str,
    formats: Iterable[str],
    samples: int,
    seed: int,
) -> None:
    plt = _setup_matplotlib()
    data = collect_main_seed_means(results_root)
    rolling = _rolling_means(results_root)
    methods = (*LEARNED_METHODS, "rolling_horizon")
    metrics = ("service_level", "throughput_per_100_steps")
    x = np.arange(len(APPLICATION_SCENARIOS), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), constrained_layout=True)
    for panel, (ax, metric) in enumerate(zip(axes, metrics)):
        for method, offset in zip(methods, offsets):
            color, marker, _ = METHOD_STYLES[method]
            if method == "rolling_horizon":
                means = [rolling[s][metric] for s in APPLICATION_SCENARIOS]
                ax.scatter(x + offset, means, color=color, marker=marker, s=28,
                           label=METHOD_LABELS[method], zorder=4)
                continue
            means, lows, highs = [], [], []
            for index, scenario in enumerate(APPLICATION_SCENARIOS):
                mean, low, high = interval(
                    _values(data, method, scenario, metric), uncertainty,
                    samples=samples, seed=seed + panel * 100 + index,
                )
                means.append(mean); lows.append(low); highs.append(high)
            yerr = np.vstack((np.asarray(means) - lows, np.asarray(highs) - means))
            ax.errorbar(x + offset, means, yerr=yerr, fmt=marker, color=color,
                        markersize=4, capsize=2, linewidth=1, label=METHOD_LABELS[method])
        ax.set_title(METRIC_DEFINITIONS[metric]["label"])
        ax.set_xticks(
            x,
            [SCENARIO_LABELS[s] for s in APPLICATION_SCENARIOS],
            rotation=30,
            ha="right",
        )
        ax.set_xlabel("Warehouse operating scenario")
        ax.set_ylabel(METRIC_AXIS_LABELS[metric])
        ax.set_ylim(bottom=0)
        if metric in {"service_level", "deadline_miss_rate"}:
            ax.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = _ordered_legend_entries(handles, labels, 3)
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    _save(fig, output / "fig_application_comparison", formats)
    _save_individual_panels(
        fig, axes, output / "fig_application_comparison",
        metrics, formats, legend_ncol=3,
    )
    _save_shared_legend(fig, axes, output / "fig_application_comparison", formats, ncol=3)
    plt.close(fig)


def plot_chord_effects(
    results_root: Path,
    output: Path,
    uncertainty: str,
    formats: Iterable[str],
    samples: int,
    seed: int,
) -> None:
    plt = _setup_matplotlib()
    data = collect_main_seed_means(results_root)
    rolling = _rolling_means(results_root)
    baselines = (*LEARNED_METHODS[1:], "rolling_horizon")
    x = np.arange(len(CORE_SCENARIOS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(baselines))
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.2), constrained_layout=True)
    for panel, (ax, metric) in enumerate(zip(axes.flat, PRIMARY_METRICS)):
        sign = 1.0 if METRIC_DEFINITIONS[metric]["direction"] == "max" else -1.0
        for baseline, offset in zip(baselines, offsets):
            means, lows, highs = [], [], []
            for index, scenario in enumerate(CORE_SCENARIOS):
                chord = _values(data, "chord_mappo", scenario, metric)
                if baseline == "rolling_horizon":
                    differences = [sign * (value - rolling[scenario][metric]) for value in chord]
                else:
                    other = _values(data, baseline, scenario, metric)
                    differences = [sign * (left - right) for left, right in zip(chord, other)]
                mean, low, high = interval(
                    differences, uncertainty, samples=samples,
                    seed=seed + panel * 100 + index,
                )
                means.append(mean); lows.append(low); highs.append(high)
            color, marker, _ = METHOD_STYLES[baseline]
            yerr = np.vstack((np.asarray(means) - lows, np.asarray(highs) - means))
            ax.errorbar(x + offset, means, yerr=yerr, fmt=marker, color=color,
                        markersize=4, capsize=2, linewidth=1, label=METHOD_LABELS[baseline])
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.set_title(METRIC_DEFINITIONS[metric]["label"])
        ax.set_xticks(x, [SCENARIO_LABELS[s] for s in CORE_SCENARIOS], rotation=35, ha="right")
        ax.set_ylabel("Oriented difference (positive favors CHORD)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles, labels = _ordered_legend_entries(handles, labels, 3)
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    _save(fig, output / "fig_chord_oriented_effects", formats)
    _save_individual_panels(
        fig, axes, output / "fig_chord_oriented_effects",
        PRIMARY_METRICS, formats, legend_ncol=3,
    )
    _save_shared_legend(fig, axes, output / "fig_chord_oriented_effects", formats, ncol=3)
    plt.close(fig)


def plot_scale_generalization(
    results_root: Path,
    output: Path,
    uncertainty: str,
    formats: Iterable[str],
    samples: int,
    seed: int,
) -> None:
    plt = _setup_matplotlib()
    data = collect_main_seed_means(results_root)
    rolling = _rolling_means(results_root)
    scenarios = ("unseen_small", "in_distribution", "unseen_large")
    metrics = ("service_level", "throughput_per_100_steps")
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.8), constrained_layout=True)
    for panel, (ax, metric) in enumerate(zip(axes, metrics)):
        for method in LEARNED_METHODS:
            means, lows, highs = [], [], []
            for index, scenario in enumerate(scenarios):
                mean, low, high = interval(
                    _values(data, method, scenario, metric), uncertainty,
                    samples=samples, seed=seed + panel * 100 + index,
                )
                means.append(mean); lows.append(low); highs.append(high)
            color, marker, linestyle = METHOD_STYLES[method]
            ax.plot(x, means, color=color, marker=marker, linestyle=linestyle,
                    markersize=4, label=METHOD_LABELS[method])
            ax.fill_between(x, lows, highs, color=color, alpha=0.10, linewidth=0)
        color, marker, linestyle = METHOD_STYLES["rolling_horizon"]
        ax.plot(x, [rolling[s][metric] for s in scenarios], color=color,
                marker=marker, linestyle=linestyle, markersize=4,
                label=METHOD_LABELS["rolling_horizon"])
        ax.set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
        ax.set_xlabel("Population-scale regime")
        ax.set_title(METRIC_DEFINITIONS[metric]["label"])
        ax.set_ylabel(METRIC_AXIS_LABELS[metric])
        ax.set_ylim(bottom=0)
        if metric == "service_level": ax.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = _ordered_legend_entries(handles, labels, 3)
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    _save(fig, output / "fig_scale_generalization", formats)
    _save_individual_panels(
        fig, axes, output / "fig_scale_generalization",
        metrics, formats, legend_ncol=3,
    )
    _save_shared_legend(fig, axes, output / "fig_scale_generalization", formats, ncol=3)
    plt.close(fig)


def _collect_variant_data(results_root: Path, section: str, methods: Iterable[str]):
    root = results_root / section
    data = {"chord_mappo": load_seed_summaries(results_root / "main_comparison" / "chord_mappo")}
    for method in methods:
        data[method] = load_seed_summaries(root / method)
    return data


def _plot_mechanism_grid(
    data: dict[str, dict[int, dict[str, dict[str, dict[str, float]]]]],
    methods: tuple[str, ...],
    scenarios: tuple[str, ...],
    output_base: Path,
    uncertainty: str,
    formats: Iterable[str],
    samples: int,
    seed: int,
    metrics: tuple[str, ...] = PRIMARY_METRICS,
) -> None:
    plt = _setup_matplotlib()
    x = np.arange(len(scenarios), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(methods))
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.7), constrained_layout=True)
    if len(metrics) != len(axes.flat):
        raise ValueError("mechanism grid requires exactly four metrics")
    for panel, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        for method, offset in zip(methods, offsets):
            means, lows, highs = [], [], []
            for index, scenario in enumerate(scenarios):
                mean, low, high = interval(
                    _values(data, method, scenario, metric), uncertainty,
                    samples=samples, seed=seed + panel * 100 + index,
                )
                means.append(mean); lows.append(low); highs.append(high)
            color, marker, _ = METHOD_STYLES[method]
            yerr = np.vstack((np.asarray(means) - lows, np.asarray(highs) - means))
            ax.errorbar(x + offset, means, yerr=yerr, fmt=marker, color=color,
                        markersize=4, capsize=2, linewidth=1, label=METHOD_LABELS[method])
        ax.set_title(METRIC_DEFINITIONS[metric]["label"])
        ax.set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios], rotation=20, ha="right")
        ax.set_xlabel("Evaluation scenario")
        ax.set_ylabel(METRIC_AXIS_LABELS[metric])
        ax.set_ylim(bottom=0)
        if metric in {"service_level", "deadline_miss_rate"}: ax.set_ylim(0, 1)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles, labels = _ordered_legend_entries(handles, labels, 2)
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    _save(fig, output_base, formats)
    _save_individual_panels(
        fig, axes, output_base, metrics, formats, legend_ncol=2,
    )
    _save_shared_legend(fig, axes, output_base, formats, ncol=2)
    plt.close(fig)


def plot_mechanisms(
    results_root: Path,
    output: Path,
    uncertainty: str,
    formats: Iterable[str],
    samples: int,
    seed: int,
) -> None:
    ablations = ("chord_no_reservation", "chord_capacity_factorized", "chord_no_capacity")
    data = _collect_variant_data(results_root, "mechanism_ablations", ablations)
    _plot_mechanism_grid(
        data, ("chord_mappo", *ablations), ABLATION_SCENARIOS,
        output / "fig_component_ablations", uncertainty, formats, samples, seed,
        metrics=COMPONENT_ABLATION_METRICS,
    )
    controls = ("entity_cahr", "graph_cahr")
    data = _collect_variant_data(results_root, "mechanism_controls", controls)
    _plot_mechanism_grid(
        data, ("chord_mappo", *controls), ABLATION_SCENARIOS,
        output / "fig_encoder_controls", uncertainty, formats, samples, seed + 500,
    )


def plot_training_curves(
    results_root: Path,
    output: Path,
    uncertainty: str,
    formats: Iterable[str],
    samples: int,
    seed: int,
) -> None:
    plt = _setup_matplotlib()
    metrics = ("episode_service_level", "episode_throughput_per_100_steps")
    labels = (
        "Training service level (ratio)",
        "Training throughput (deliveries per 100 steps)",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.8), constrained_layout=True)
    for panel, (ax, metric, label) in enumerate(zip(axes, metrics, labels)):
        for method in LEARNED_METHODS:
            by_seed: dict[int, dict[int, float]] = {}
            for seed_dir in (results_root / "main_comparison" / method).glob("seed_*"):
                training = seed_dir / "training_metrics.jsonl"
                rows = [json.loads(line) for line in training.read_text(encoding="utf-8").splitlines() if line]
                by_seed[int(seed_dir.name.split("_")[-1])] = {
                    int(row["steps"]): float(row[metric]) for row in rows
                }
            common_steps = sorted(set.intersection(*(set(values) for values in by_seed.values())))
            # Display decimation only: every fifth raw checkpoint, no moving average.
            common_steps = common_steps[::5]
            means, lows, highs = [], [], []
            for index, step in enumerate(common_steps):
                values = [by_seed[s][step] for s in FORMAL_SEEDS]
                mean, low, high = interval(
                    values, uncertainty, samples=samples,
                    seed=seed + panel * 1000 + index,
                )
                means.append(mean); lows.append(low); highs.append(high)
            x = np.asarray(common_steps) / 1_000_000.0
            color, _, linestyle = METHOD_STYLES[method]
            ax.plot(x, means, color=color, linestyle=linestyle, label=METHOD_LABELS[method])
            ax.fill_between(x, lows, highs, color=color, alpha=0.10, linewidth=0)
        ax.set_xlabel("Environment interactions (million)")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        if metric == "episode_service_level": ax.set_ylim(0, 1)
    handles, labels_out = axes[0].get_legend_handles_labels()
    handles, labels_out = _ordered_legend_entries(handles, labels_out, 3)
    fig.legend(handles, labels_out, loc="outside lower center", ncol=3, frameon=False)
    _save(fig, output / "fig_training_curves_unsmoothed", formats)
    _save_individual_panels(
        fig, axes, output / "fig_training_curves_unsmoothed",
        ("service_level", "throughput_per_100_steps"), formats, legend_ncol=3,
    )
    _save_shared_legend(fig, axes, output / "fig_training_curves_unsmoothed", formats, ncol=3)
    plt.close(fig)


def plot_sensitivity_if_available(
    results_root: Path,
    output: Path,
    uncertainty: str,
    formats: Iterable[str],
    samples: int,
    seed: int,
) -> bool:
    path = results_root / "sensitivity" / "chord" / "aggregate" / "aggregate_by_training_seed.csv"
    if not path.exists():
        return False
    rows = _read_csv(path)
    look = {(r["scenario"], r["metric"]): r for r in rows if r["algorithm"] == "chord_mappo"}
    plt = _setup_matplotlib()
    sweeps = (
        (
            "Scheduled AGV breakdown fraction",
            ("breakdown_000", "breakdown_010", "breakdown_025", "breakdown_040"),
            (0, .10, .25, .40),
            ("0%", "10%", "25%", "40%"),
        ),
        (
            "Order-arrival-rate multiplier",
            ("demand_050", "demand_100", "demand_200", "demand_350"),
            (.5, 1, 2, 3.5),
            ("0.5×", "1.0×", "2.0×", "3.5×"),
        ),
    )
    metrics = ("service_level", "throughput_per_100_steps")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.8), constrained_layout=True)
    for column, (xlabel, scenarios, doses, tick_labels) in enumerate(sweeps):
        for row_index, metric in enumerate(metrics):
            ax = axes[row_index, column]
            means, lows, highs = [], [], []
            for scenario in scenarios:
                record = look[(scenario, metric)]
                means.append(float(record["mean"]))
                if uncertainty == "std":
                    spread = float(record["std_across_seeds"])
                    lows.append(float(record["mean"]) - spread)
                    highs.append(float(record["mean"]) + spread)
                else:
                    lows.append(float(record["ci95_low"])); highs.append(float(record["ci95_high"]))
            ax.plot(doses, means, color=METHOD_STYLES["chord_mappo"][0], marker="o")
            ax.fill_between(doses, lows, highs, color=METHOD_STYLES["chord_mappo"][0], alpha=.15)
            ax.set_xticks(doses, tick_labels)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(METRIC_AXIS_LABELS[metric])
            ax.set_ylim(bottom=0)
            if metric == "service_level": ax.set_ylim(0, 1)
    _save(fig, output / "fig_sensitivity_dose_response", formats)
    _save_individual_panels(
        fig, axes, output / "fig_sensitivity_dose_response",
        (
            "breakdown_service_level",
            "demand_service_level",
            "breakdown_throughput_per_100_steps",
            "demand_throughput_per_100_steps",
        ),
        formats,
        legend_ncol=1,
    )
    plt.close(fig)
    return True


def plot_semantic_case(
    results_root: Path, output: Path, formats: Iterable[str]
) -> bool:
    path = results_root / "supplementary_semantic_analysis" / "semantic_event_summary.csv"
    if not path.exists():
        return False
    rows = [
        row for row in _read_csv(path)
        if row["attribution_scope"] in {"task_coalition", "shared_zone"}
    ]
    plt = _setup_matplotlib()
    events = ("pre_breakdown", "post_breakdown", "post_repair")
    scopes = ("task_coalition", "shared_zone")
    x = np.arange(len(events), dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 2.8), constrained_layout=True)
    for offset, scope, color, marker in (
        (-.08, scopes[0], "#0072B2", "o"),
        (.08, scopes[1], "#D55E00", "s"),
    ):
        means = []
        for event_index, event in enumerate(events):
            values = [
                float(row["mean_edge_abs_probability_change"])
                for row in rows if row["event"] == event and row["attribution_scope"] == scope
            ]
            means.append(float(np.mean(values)))
            ax.scatter(np.full(len(values), x[event_index] + offset), values,
                       color=color, marker=marker, s=20, alpha=.7)
        ax.plot(x + offset, means, color=color, marker=marker,
                label=scope.replace("_", " "))
    ax.set_xticks(x, ["Pre-breakdown", "Post-breakdown", "Post-repair"])
    ax.set_ylabel("Mean absolute action-probability change")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    _save(fig, output / "fig_semantic_edge_masking", formats)
    plt.close(fig)
    return True


def write_tables(
    results_root: Path,
    output: Path,
    uncertainty: str,
    samples: int,
    seed: int,
) -> None:
    data = collect_main_seed_means(results_root)
    rolling = _rolling_means(results_root)
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(CORE_SCENARIOS):
        for method in (*LEARNED_METHODS, "rolling_horizon"):
            row: dict[str, Any] = {"scenario": scenario, "method": method}
            for metric_index, metric in enumerate(PRIMARY_METRICS):
                if method == "rolling_horizon":
                    mean, low, high, n = rolling[scenario][metric], None, None, 0
                else:
                    values = _values(data, method, scenario, metric)
                    mean, low, high = interval(
                        values, uncertainty, samples=samples,
                        seed=seed + scenario_index * 100 + metric_index,
                    )
                    n = len(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_low"] = low
                row[f"{metric}_high"] = high
                row[f"{metric}_n_seeds"] = n
            rows.append(row)
    _write_csv(output / "table_main_results_seed_level.csv", rows)

    metric_rows = [
        {"metric": metric, **definition} for metric, definition in METRIC_DEFINITIONS.items()
    ]
    _write_csv(output / "table_metric_dictionary.csv", metric_rows)

    effects: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(CORE_SCENARIOS):
        for baseline in (*LEARNED_METHODS[1:], "rolling_horizon"):
            for metric_index, metric in enumerate(PRIMARY_METRICS):
                sign = 1.0 if METRIC_DEFINITIONS[metric]["direction"] == "max" else -1.0
                chord = _values(data, "chord_mappo", scenario, metric)
                if baseline == "rolling_horizon":
                    diffs = [sign * (v - rolling[scenario][metric]) for v in chord]
                else:
                    diffs = [
                        sign * (left - right)
                        for left, right in zip(chord, _values(data, baseline, scenario, metric))
                    ]
                mean, low, high = interval(
                    diffs, uncertainty, samples=samples,
                    seed=seed + scenario_index * 100 + metric_index,
                )
                effects.append(
                    {
                        "scenario": scenario,
                        "baseline": baseline,
                        "metric": metric,
                        "oriented_mean_difference_positive_favors_chord": mean,
                        "interval_low": low,
                        "interval_high": high,
                        "n_training_seeds": len(diffs),
                    }
                )
    _write_csv(output / "table_chord_oriented_effects.csv", effects)


def write_figure_manifest(
    output: Path, sensitivity_available: bool, semantic_available: bool
) -> None:
    rows = [
        {
            "artifact": "fig_application_comparison",
            "standalone_panels": "fig_application_comparison_[A-B]_<metric>",
            "claim": "absolute final-policy performance under application disruptions",
            "x_axis": "six categorical warehouse operating scenarios",
            "y_axis": "service level and throughput in native units",
            "statistics": "mean and selected interval across six training seeds; rolling MILP is a point estimate",
            "placement": "main text",
        },
        {
            "artifact": "fig_chord_oriented_effects",
            "standalone_panels": "fig_chord_oriented_effects_[A-D]_<metric>",
            "claim": "diagnostic direction-normalized CHORD-minus-baseline differences",
            "x_axis": "eight categorical warehouse scenarios",
            "y_axis": "direction-normalized CHORD-minus-baseline difference; positive favors CHORD",
            "statistics": "paired by training seed for learned methods; uncertainty over CHORD seeds for deterministic MILP",
            "placement": "diagnostic only; omit from manuscript to avoid duplicating the absolute comparison and tables",
        },
        {
            "artifact": "fig_scale_generalization",
            "standalone_panels": "fig_scale_generalization_[A-B]_<metric>",
            "claim": "bounded compound population-and-initial-load extrapolation",
            "x_axis": "unseen-small, in-distribution, unseen-large regimes",
            "y_axis": "service level and throughput",
            "statistics": "mean and selected interval across six training seeds",
            "placement": "main text",
        },
        {
            "artifact": "fig_component_ablations",
            "standalone_panels": "fig_component_ablations_[A-D]_<metric>",
            "claim": "separate value of capacity context and conflict-free reservation",
            "x_axis": "in-distribution, AGV-heavy, robot-breakdown scenarios",
            "y_axis": "service level, throughput, contention/delivery and deadline-miss rate",
            "statistics": "mean and selected interval across six training seeds; no strict paired-trajectory claim",
            "placement": "main text",
        },
        {
            "artifact": "fig_encoder_controls",
            "standalone_panels": "fig_encoder_controls_[A-D]_<metric>",
            "claim": "benefit of hypergraph coalition representation with the decoder held fixed",
            "x_axis": "in-distribution, AGV-heavy, robot-breakdown scenarios",
            "y_axis": "four primary metrics",
            "statistics": "mean and selected interval across six training seeds",
            "placement": "main text",
        },
        {
            "artifact": "fig_training_curves_unsmoothed",
            "standalone_panels": "fig_training_curves_unsmoothed_[A-B]_<metric>",
            "claim": "learning-process trajectory within the fixed interaction budget",
            "x_axis": "environment interactions",
            "y_axis": "training service level and throughput",
            "statistics": "mean and 95% bootstrap interval across six seeds; every fifth logged point, no smoothing",
            "placement": "appendix",
        },
        {
            "artifact": "fig_sensitivity_dose_response",
            "standalone_panels": "fig_sensitivity_dose_response_[A-D]_<sweep_metric>",
            "claim": "response to increasing AGV failure and demand intensity within tested ranges",
            "x_axis": "failure fraction or arrival-rate multiplier",
            "y_axis": "service level and throughput",
            "statistics": "mean and selected interval across six fixed trained CHORD policies",
            "placement": "main text",
            "available": sensitivity_available,
        },
        {
            "artifact": "semantic_analysis/fig_semantic_analysis",
            "standalone_panels": "fig_semantic_masking_sensitivity_A and fig_semantic_representative_timeline_B",
            "claim": "structural transparency around resource breakdown and restoration",
            "x_axis": "event stage in panel A; simulator transition in panel B",
            "y_axis": "top-edge probability sensitivity in panel A; resources and deliveries in panel B",
            "statistics": "six training seeds after within-seed aggregation of 20 execution episodes; model-internal, not causal",
            "placement": "main text",
            "available": semantic_available,
        },
    ]
    _write_csv(output / "figure_table_manifest.csv", rows)


def generate_artifacts(
    results_root: Path,
    output: Path,
    uncertainty: str = "ci95",
    formats: Iterable[str] = ("pdf", "svg", "png"),
    bootstrap_samples: int = 10_000,
    seed: int = 2026,
    audit_only: bool = False,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    figure_output = output / "figures"
    figure_output.mkdir(parents=True, exist_ok=True)
    audit = audit_results(results_root)
    (output / "audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_tables(results_root, output, uncertainty, bootstrap_samples, seed)
    sensitivity_available = (
        results_root / "sensitivity" / "chord" / "aggregate" / "aggregate_by_training_seed.csv"
    ).exists()
    semantic_available = (
        results_root / "semantic_analysis" / "semantic_paper_summary.json"
    ).exists()
    write_figure_manifest(output, sensitivity_available, semantic_available)
    if audit_only:
        return audit
    plot_main_metrics(
        results_root, figure_output, uncertainty, formats, bootstrap_samples, seed
    )
    # The CHORD-minus-baseline diagnostic is intentionally not regenerated by
    # the paper pipeline: it hides CHORD as the zero reference and duplicates
    # the absolute comparison/table evidence.  Keep the function available for
    # analysis, but exclude the figure from the manuscript build.
    plot_scale_generalization(
        results_root, figure_output, uncertainty, formats, bootstrap_samples, seed
    )
    plot_mechanisms(
        results_root, figure_output, uncertainty, formats, bootstrap_samples, seed
    )
    plot_training_curves(
        results_root, figure_output, uncertainty, formats, bootstrap_samples, seed
    )
    plot_sensitivity_if_available(
        results_root, figure_output, uncertainty, formats, bootstrap_samples, seed
    )
    # Semantic figures are generated by build_semantic_paper_artifacts.py from
    # the audited six-seed, 20-episode-per-seed result set.  Do not regenerate
    # the obsolete three-seed plot here.
    return audit
