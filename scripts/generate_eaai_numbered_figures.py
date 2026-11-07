"""Generate the twelve numbered EAAI artwork panels used by the manuscript.

This is a read-only post-processing entry point.  It reads the archived
six-training-seed aggregate CSV files and writes derived PDF, SVG, and PNG
artwork.  It never edits checkpoints, episode records, or aggregate CSVs.

The filenames follow Elsevier's numbered-artwork convention and the current
manuscript order:

* FIG4A--FIG4D: unseen-small matched-encoder comparison;
* FIG7A--FIG7D: breakdown and demand dose-response evaluation;
* FIG11A--FIG12B: complete CHORD-minus-control effect plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


FORMAL_SEEDS = (1, 45, 789, 2027, 4096, 8191)
FORMATS = ("pdf", "svg", "png")

METHOD_LABELS = {
    "chord_mappo": "CHORD-CAHR",
    "entity_cahr": "Entity+CAHR",
    "graph_cahr": "Graph+CAHR",
}

METHOD_STYLES = {
    "chord_mappo": ("#0072B2", "o"),
    "entity_cahr": ("#56B4E9", "v"),
    "graph_cahr": ("#B79F00", "<"),
}

SCENARIOS = (
    "in_distribution",
    "unseen_small",
    "unseen_large",
    "agv_heavy_ratio",
    "fixed_team_dynamic_orders",
    "robot_breakdown",
)

SCENARIO_LABELS = {
    "in_distribution": "Nominal",
    "unseen_small": "Unseen small",
    "unseen_large": "Unseen large",
    "agv_heavy_ratio": "AGV-heavy",
    "fixed_team_dynamic_orders": "Dynamic orders",
    "robot_breakdown": "Breakdown",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    axis_label: str
    direction: str
    ratio: bool = False


METRICS = {
    "service_level": MetricSpec(
        "service_level", "Service level", "max", True
    ),
    "throughput_per_100_steps": MetricSpec(
        "throughput_per_100_steps",
        "Logged throughput",
        "max",
    ),
    "deadline_miss_rate": MetricSpec(
        "deadline_miss_rate", "Deadline-miss rate", "min", True
    ),
    "mean_flow_time": MetricSpec(
        "mean_flow_time", "Flow time (ticks)", "min"
    ),
}

EXPLORATORY_PANELS = {
    "FIG4A": "service_level",
    "FIG4B": "throughput_per_100_steps",
    "FIG4C": "deadline_miss_rate",
    "FIG4D": "mean_flow_time",
}

STRESS_PANELS = {
    "FIG7A": ("breakdown", "service_level"),
    "FIG7B": ("demand", "service_level"),
    "FIG7C": ("breakdown", "throughput_per_100_steps"),
    "FIG7D": ("demand", "throughput_per_100_steps"),
}

CONTROL_PANELS = {
    "FIG11A": "service_level",
    "FIG11B": "throughput_per_100_steps",
    "FIG12A": "deadline_miss_rate",
    "FIG12B": "mean_flow_time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figs/paper_artifacts/eaai_numbered_figures"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=FORMATS,
        default=FORMATS,
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_unique(
    rows: Iterable[dict[str, str]], keys: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, str]]:
    index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[column] for column in keys)
        if key in index:
            raise ValueError(f"duplicate aggregate row for {key}")
        index[key] = row
    return index


def _validate_seed_summary(row: dict[str, str], context: str) -> None:
    count = int(row["n_training_seeds"])
    if count != len(FORMAL_SEEDS):
        raise ValueError(f"{context}: expected six training seeds, found {count}")
    recorded = tuple(json.loads(row["training_seeds"]))
    if recorded != FORMAL_SEEDS:
        raise ValueError(
            f"{context}: expected seeds {FORMAL_SEEDS}, found {recorded}"
        )
    for field in ("mean", "ci95_low", "ci95_high"):
        value = float(row[field])
        if not np.isfinite(value):
            raise ValueError(f"{context}: non-finite {field}")


def orient_interval(
    mean: float, low: float, high: float, direction: str
) -> tuple[float, float, float]:
    """Orient an interval so positive differences always favour CHORD."""
    if direction == "max":
        return mean, low, high
    if direction == "min":
        return -mean, -high, -low
    raise ValueError(f"unknown metric direction: {direction}")


def _setup_matplotlib():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Install the plotting dependency with `python -m pip install -e .[plot]`."
        ) from exc

    font_manager.findfont("Arial", fallback_to_default=False)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            # Numbered panels are inserted at 0.91--0.93 native scale; 8 pt
            # source labels remain above 7 pt at final publication size.
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
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
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    return plt


def _save(fig, base: Path, formats: Iterable[str]) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in formats:
        path = base.with_suffix(f".{extension}")
        fig.savefig(path)
        written.append(path)
    return written


def _set_native_y_range(ax, metric: MetricSpec) -> None:
    if metric.ratio:
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(bottom=0.0)


def plot_exploratory_panels(
    rows: list[dict[str, str]], output: Path, formats: Iterable[str]
) -> list[dict[str, str]]:
    plt = _setup_matplotlib()
    index = _index_unique(rows, ("algorithm", "scenario", "metric"))
    methods = ("chord_mappo", "entity_cahr", "graph_cahr")
    x = np.arange(len(methods), dtype=float)
    manifest: list[dict[str, str]] = []

    for figure_name, metric_key in EXPLORATORY_PANELS.items():
        metric = METRICS[metric_key]
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for method in methods:
            key = (method, "unseen_small", metric_key)
            if key not in index:
                raise KeyError(f"missing aggregate row: {key}")
            row = index[key]
            _validate_seed_summary(row, str(key))
            means.append(float(row["mean"]))
            lows.append(float(row["ci95_low"]))
            highs.append(float(row["ci95_high"]))

        fig, ax = plt.subplots(figsize=(3.45, 2.70), constrained_layout=True)
        for position, method, mean, low, high in zip(
            x, methods, means, lows, highs
        ):
            color, marker = METHOD_STYLES[method]
            ax.errorbar(
                position,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt=marker,
                color=color,
                markersize=5.0,
                capsize=2.5,
                linewidth=1.1,
                zorder=3,
            )
        ax.set_xticks(
            x,
            ("CHORD-\nCAHR", "Entity+\nCAHR", "Graph+\nCAHR"),
        )

        ax.set_ylabel(metric.axis_label)
        _set_native_y_range(ax, metric)
        _save(fig, output / figure_name, formats)
        plt.close(fig)
        manifest.append(
            {
                "figure": figure_name,
                "source": "mechanism_controls/aggregate/aggregate_by_training_seed.csv",
                "filter": f"scenario=unseen_small; metric={metric_key}",
                "statistic": "mean and 95% CI across six training-seed means",
            }
        )
    return manifest


def plot_stress_panels(
    rows: list[dict[str, str]], output: Path, formats: Iterable[str]
) -> list[dict[str, str]]:
    plt = _setup_matplotlib()
    index = _index_unique(rows, ("algorithm", "scenario", "metric"))
    sweep_specs = {
        "breakdown": {
            "scenarios": ("breakdown_000", "breakdown_010", "breakdown_025", "breakdown_040"),
            "x": np.asarray((0.0, 0.10, 0.25, 0.40)),
            "ticks": ("0%", "10%", "25%", "40%"),
            "xlabel": "Configured AGV removal fraction",
        },
        "demand": {
            "scenarios": ("demand_050", "demand_100", "demand_200", "demand_350"),
            "x": np.asarray((0.5, 1.0, 2.0, 3.5)),
            "ticks": ("0.5", "1.0", "2.0", "3.5"),
            "xlabel": "Order-arrival-rate multiplier",
        },
    }
    manifest: list[dict[str, str]] = []

    for figure_name, (sweep, metric_key) in STRESS_PANELS.items():
        metric = METRICS[metric_key]
        spec = sweep_specs[sweep]
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for scenario in spec["scenarios"]:
            key = ("chord_mappo", scenario, metric_key)
            if key not in index:
                raise KeyError(f"missing sensitivity row: {key}")
            row = index[key]
            _validate_seed_summary(row, str(key))
            means.append(float(row["mean"]))
            lows.append(float(row["ci95_low"]))
            highs.append(float(row["ci95_high"]))

        x = np.asarray(spec["x"], dtype=float)
        mean_array = np.asarray(means)
        low_array = np.asarray(lows)
        high_array = np.asarray(highs)
        color, marker = METHOD_STYLES["chord_mappo"]
        fig, ax = plt.subplots(figsize=(3.45, 2.65), constrained_layout=True)
        ax.plot(x, mean_array, color=color, marker=marker, zorder=3)
        ax.fill_between(
            x,
            low_array,
            high_array,
            color=color,
            alpha=0.18,
            linewidth=0.7,
            edgecolor=color,
        )
        ax.set_xticks(x, spec["ticks"])
        ax.set_xlabel(str(spec["xlabel"]))
        ax.set_ylabel(metric.axis_label)
        _set_native_y_range(ax, metric)
        _save(fig, output / figure_name, formats)
        plt.close(fig)
        manifest.append(
            {
                "figure": figure_name,
                "source": "sensitivity/chord/aggregate/aggregate_by_training_seed.csv",
                "filter": f"sweep={sweep}; metric={metric_key}; algorithm=chord_mappo",
                "statistic": "mean and 95% CI across six fixed trained policies",
            }
        )
    return manifest


def plot_control_effect_panels(
    rows: list[dict[str, str]], output: Path, formats: Iterable[str]
) -> list[dict[str, str]]:
    plt = _setup_matplotlib()
    index = _index_unique(rows, ("baseline", "scenario", "metric"))
    baselines = ("entity_cahr", "graph_cahr")
    x = np.arange(len(SCENARIOS), dtype=float)
    offsets = {"entity_cahr": -0.09, "graph_cahr": 0.09}
    manifest: list[dict[str, str]] = []

    for figure_name, metric_key in CONTROL_PANELS.items():
        metric = METRICS[metric_key]
        fig, ax = plt.subplots(figsize=(6.85, 2.85), constrained_layout=True)
        for baseline in baselines:
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for scenario in SCENARIOS:
                key = (baseline, scenario, metric_key)
                if key not in index:
                    raise KeyError(f"missing paired-effect row: {key}")
                row = index[key]
                if int(row["n_paired_training_seeds"]) != len(FORMAL_SEEDS):
                    raise ValueError(f"{key}: expected six paired training seeds")
                mean, low, high = orient_interval(
                    float(row["mean_difference"]),
                    float(row["ci95_low"]),
                    float(row["ci95_high"]),
                    metric.direction,
                )
                means.append(mean)
                lows.append(low)
                highs.append(high)
            mean_array = np.asarray(means)
            low_array = np.asarray(lows)
            high_array = np.asarray(highs)
            color, marker = METHOD_STYLES[baseline]
            ax.errorbar(
                x + offsets[baseline],
                mean_array,
                yerr=np.vstack((mean_array - low_array, high_array - mean_array)),
                fmt=marker,
                color=color,
                markersize=4.5,
                capsize=2.2,
                linewidth=1.0,
                label=METHOD_LABELS[baseline],
                zorder=3,
            )
        ax.axhline(0.0, color="#333333", linewidth=0.8, zorder=2)
        ax.set_xticks(
            x,
            [SCENARIO_LABELS[scenario] for scenario in SCENARIOS],
            rotation=27,
            ha="right",
        )
        short_metric = {
            "service_level": "service-level",
            "throughput_per_100_steps": "logged-throughput",
            "deadline_miss_rate": "deadline-miss",
            "mean_flow_time": "flow-time",
        }[metric_key]
        ax.set_ylabel(
            f"Oriented {short_metric} difference\n(positive favours CHORD)"
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.5,
        )
        _save(fig, output / figure_name, formats)
        plt.close(fig)
        manifest.append(
            {
                "figure": figure_name,
                "source": "mechanism_controls/aggregate/paired_differences.csv",
                "filter": f"metric={metric_key}; baselines=entity_cahr,graph_cahr; scenarios=Core6",
                "statistic": "paired CHORD-minus-control mean and 95% CI over six training seeds; lower-is-better metrics sign-reversed",
            }
        )
    return manifest


def _write_manifest(
    output: Path,
    rows: list[dict[str, str]],
    source_paths: list[Path],
    formats: Iterable[str],
) -> None:
    csv_path = output / "figure_provenance.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("figure", "source", "filter", "statistic")
        )
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "statistical_unit": "independent policy-training seed",
        "training_seeds": list(FORMAL_SEEDS),
        "source_files_are_read_only": True,
        "formats": list(formats),
        "source_sha256": {
            str(path.as_posix()): _sha256(path) for path in source_paths
        },
        "figures": [row["figure"] for row in rows],
    }
    (output / "figure_provenance.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    formats = tuple(dict.fromkeys(args.formats))
    results_root = args.results_root
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    control_summary = (
        results_root
        / "mechanism_controls"
        / "aggregate"
        / "aggregate_by_training_seed.csv"
    )
    paired_effects = (
        results_root
        / "mechanism_controls"
        / "aggregate"
        / "paired_differences.csv"
    )
    sensitivity_summary = (
        results_root
        / "sensitivity"
        / "chord"
        / "aggregate"
        / "aggregate_by_training_seed.csv"
    )
    source_paths = [control_summary, paired_effects, sensitivity_summary]

    manifest: list[dict[str, str]] = []
    manifest.extend(
        plot_exploratory_panels(_read_csv(control_summary), output, formats)
    )
    manifest.extend(
        plot_stress_panels(_read_csv(sensitivity_summary), output, formats)
    )
    manifest.extend(
        plot_control_effect_panels(_read_csv(paired_effects), output, formats)
    )
    _write_manifest(output, manifest, source_paths, formats)

    expected_figures = set(EXPLORATORY_PANELS) | set(STRESS_PANELS) | set(CONTROL_PANELS)
    if len(expected_figures) != 12 or {row["figure"] for row in manifest} != expected_figures:
        raise RuntimeError("the numbered-artwork manifest is incomplete")

    print(
        json.dumps(
            {
                "status": "ok",
                "source_files": [str(path) for path in source_paths],
                "figures": sorted(expected_figures),
                "formats": list(formats),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
