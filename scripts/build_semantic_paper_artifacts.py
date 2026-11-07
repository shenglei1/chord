"""Build six-seed semantic-explanation tables and publication figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable
import zlib

import numpy as np


FORMAL_SEEDS = (1, 45, 789, 2027, 4096, 8191)
EVENT_ORDER = ("pre_breakdown", "post_breakdown", "post_repair")
EVENT_LABELS = {
    "pre_breakdown": "Before\nbreakdown",
    "post_breakdown": "After\nbreakdown",
    "post_repair": "After\nrestoration",
}
RELATION_ORDER = ("task_coalition", "shared_zone")
RELATION_LABELS = {
    "task_coalition": "Task-coalition relation",
    "shared_zone": "Shared-zone relation",
}
GROUP_SCOPE_TO_RELATION = {
    "all_task_coalitions": "task_coalition",
    "all_shared_zones": "shared_zone",
}
RAW_RUN_DIRS = (
    "seed_1_45_789_combined",
    "seed_2027",
    "seed_4096",
    "seed_8191",
)
CSV_NAMES = (
    "semantic_event_summary.csv",
    "edge_attributions.csv",
    "case_timelines.csv",
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260929
WRITE_DERIVED_DATA = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/semantic_analysis"),
        help="Semantic-analysis directory containing raw/ (default: results/semantic_analysis).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("pdf", "svg", "png"),
        choices=("pdf", "svg", "png"),
        help="Figure formats. PDF is the preferred LaTeX artifact.",
    )
    parser.add_argument(
        "--figures-output",
        type=Path,
        default=Path("figs/semantic_analysis"),
        help="Directory for generated figures (default: figs/semantic_analysis).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Generate all processed CSV/JSON/LaTeX outputs without importing matplotlib.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate figures without rewriting processed result tables or summaries.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    if not WRITE_DERIVED_DATA:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError(f"cannot infer CSV fields for empty output {path}")
        fields = rows[0].keys()
    field_list = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def stable_seed(label: str) -> int:
    return BOOTSTRAP_SEED + zlib.crc32(label.encode("utf-8"))


def bootstrap_ci(values: list[float], label: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"no values for {label}")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(stable_seed(label))
    indexes = rng.integers(0, array.size, size=(BOOTSTRAP_SAMPLES, array.size))
    means = array[indexes].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = root / "raw"
    summaries: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []
    timelines: list[dict[str, str]] = []
    checkpoint_records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    expected_headers: dict[str, list[str]] = {}
    for run_name in RAW_RUN_DIRS:
        run_dir = raw / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"missing raw semantic run: {run_dir}")
        for csv_name in CSV_NAMES:
            path = run_dir / csv_name
            rows = read_csv(path)
            if not rows:
                raise ValueError(f"empty semantic input: {path}")
            header = list(rows[0].keys())
            if csv_name in expected_headers and header != expected_headers[csv_name]:
                raise ValueError(f"schema mismatch for {path}")
            expected_headers.setdefault(csv_name, header)
            if csv_name == "semantic_event_summary.csv":
                summaries.extend(rows)
            elif csv_name == "edge_attributions.csv":
                edges.extend(rows)
            else:
                timelines.extend(rows)
            manifest.append(
                {
                    "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

        json_path = run_dir / "semantic_attribution.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        checkpoint_records.extend(payload.get("checkpoints", []))
        manifest.append(
            {
                "relative_path": str(json_path.relative_to(root)).replace("\\", "/"),
                "bytes": json_path.stat().st_size,
                "sha256": sha256(json_path),
            }
        )

    observed_seeds = sorted({int(row["training_seed"]) for row in summaries})
    if observed_seeds != list(FORMAL_SEEDS):
        raise ValueError(f"expected formal seeds {FORMAL_SEEDS}, found {observed_seeds}")
    if sorted({int(row["training_seed"]) for row in edges}) != list(FORMAL_SEEDS):
        raise ValueError("edge-attribution seeds do not match the formal six-seed protocol")
    if sorted({int(row["training_seed"]) for row in timelines}) != list(FORMAL_SEEDS):
        raise ValueError("timeline seeds do not match the formal six-seed protocol")

    expected_summary_keys = {
        (seed, event, scope)
        for seed in FORMAL_SEEDS
        for event in EVENT_ORDER
        for scope in (*RELATION_ORDER, *GROUP_SCOPE_TO_RELATION)
    }
    actual_summary_keys = {
        (int(row["training_seed"]), row["event"], row["attribution_scope"])
        for row in summaries
    }
    if actual_summary_keys != expected_summary_keys:
        missing = sorted(expected_summary_keys - actual_summary_keys)
        extra = sorted(actual_summary_keys - expected_summary_keys)
        raise ValueError(f"semantic summary grid mismatch; missing={missing}, extra={extra}")
    return summaries, edges, timelines, checkpoint_records, manifest


def aggregate_seed_metric(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    metric: str,
    output_metric: str | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(metric))
        if value is None:
            continue
        grouped[tuple(row[field] for field in key_fields)].append(value)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        low, high = bootstrap_ci(values, f"{metric}:{key}")
        record = {field: value for field, value in zip(key_fields, key)}
        record.update(
            {
                "metric": output_metric or metric,
                "n_training_seeds": len(values),
                "mean": mean(values),
                "std_across_training_seeds": sample_std(values),
                "ci95_low": low,
                "ci95_high": high,
                "training_seeds": json.dumps(list(FORMAL_SEEDS)),
            }
        )
        output.append(record)
    return output


def build_main_tables(summaries: list[dict[str, str]], tables: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relation_rows: list[dict[str, Any]] = []
    for row in summaries:
        scope = row["attribution_scope"]
        if scope not in RELATION_ORDER:
            continue
        relation_rows.append(
            {
                "training_seed": int(row["training_seed"]),
                "event": row["event"],
                "event_label": EVENT_LABELS[row["event"]].replace("\n", " "),
                "relation": scope,
                "relation_label": RELATION_LABELS[scope],
                "n_execution_episodes": int(row["n_execution_episodes"]),
                "mean_edge_abs_probability_change": as_float(row["mean_edge_abs_probability_change"]),
                "mean_top_edge_abs_probability_change": as_float(row["mean_top_edge_abs_probability_change"]),
            }
        )

    main_stats = aggregate_seed_metric(
        relation_rows,
        ("event", "event_label", "relation", "relation_label"),
        "mean_top_edge_abs_probability_change",
        "mean top-edge absolute selection-probability change",
    )
    mean_edge_stats = aggregate_seed_metric(
        relation_rows,
        ("event", "event_label", "relation", "relation_label"),
        "mean_edge_abs_probability_change",
        "mean edge absolute selection-probability change",
    )
    write_csv(tables / "table_main_semantic_sensitivity.csv", main_stats)
    write_csv(tables / "table_appendix_mean_edge_sensitivity.csv", mean_edge_stats)

    lookup = {
        (int(row["training_seed"]), row["event"], row["relation"]): float(row["mean_top_edge_abs_probability_change"])
        for row in relation_rows
    }
    contrast_rows: list[dict[str, Any]] = []
    contrasts = (
        ("post_minus_pre", "post_breakdown", "pre_breakdown"),
        ("restoration_minus_post", "post_repair", "post_breakdown"),
        ("restoration_minus_pre", "post_repair", "pre_breakdown"),
    )
    for relation in RELATION_ORDER:
        for contrast, left, right in contrasts:
            values = [lookup[(seed, left, relation)] - lookup[(seed, right, relation)] for seed in FORMAL_SEEDS]
            low, high = bootstrap_ci(values, f"contrast:{relation}:{contrast}")
            contrast_rows.append(
                {
                    "relation": relation,
                    "relation_label": RELATION_LABELS[relation],
                    "contrast": contrast,
                    "n_paired_training_seeds": len(values),
                    "mean_paired_difference": mean(values),
                    "std_paired_difference": sample_std(values),
                    "ci95_low": low,
                    "ci95_high": high,
                    "training_seeds": json.dumps(list(FORMAL_SEEDS)),
                }
            )
    write_csv(tables / "table_main_semantic_stage_contrasts.csv", contrast_rows)
    return main_stats, contrast_rows


def build_group_masking_table(summaries: list[dict[str, str]], tables: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summaries:
        scope = row["attribution_scope"]
        if scope not in GROUP_SCOPE_TO_RELATION:
            continue
        rows.append(
            {
                "training_seed": int(row["training_seed"]),
                "event": row["event"],
                "event_label": EVENT_LABELS[row["event"]].replace("\n", " "),
                "relation": GROUP_SCOPE_TO_RELATION[scope],
                "relation_label": RELATION_LABELS[GROUP_SCOPE_TO_RELATION[scope]],
                "mean_abs_group_probability_drop": as_float(row["mean_abs_group_probability_drop"]),
                "mean_abs_group_value_drop": as_float(row["mean_abs_group_value_drop"]),
            }
        )
    output: list[dict[str, Any]] = []
    for metric in ("mean_abs_group_probability_drop", "mean_abs_group_value_drop"):
        output.extend(
            aggregate_seed_metric(
                rows,
                ("event", "event_label", "relation", "relation_label"),
                metric,
            )
        )
    write_csv(tables / "table_appendix_group_masking.csv", output)
    return output


def build_top_edges(edges: list[dict[str, str]], tables: Path, top_k: int = 5) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in edges:
        if not as_bool(row["is_median_case"]):
            continue
        edge_type = row["edge_type"]
        if edge_type not in RELATION_ORDER:
            continue
        grouped[(int(row["training_seed"]), row["event"], edge_type)].append(row)
    output: list[dict[str, Any]] = []
    for (seed, event, edge_type), rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                -(as_float(row["absolute_probability_change"]) or 0.0),
                int(row["edge_index"]),
            ),
        )[:top_k]
        for rank, row in enumerate(ordered, start=1):
            output.append(
                {
                    "training_seed": seed,
                    "execution_seed": int(row["execution_seed"]),
                    "event": event,
                    "edge_type": edge_type,
                    "top_k_rank_within_edge_type": rank,
                    "semantic_id": row["semantic_id"],
                    "agent_slots": row["agent_slots"],
                    "agv_slots": row["agv_slots"],
                    "picker_slots": row["picker_slots"],
                    "task_slots": row["task_slots"],
                    "order_ids": row["order_ids"],
                    "cardinality": int(row["cardinality"]),
                    "selected_probability_drop": as_float(row["selected_probability_drop"]),
                    "absolute_probability_change": as_float(row["absolute_probability_change"]),
                    "policy_js_divergence": as_float(row["policy_js_divergence"]),
                    "critic_value_drop": as_float(row["critic_value_drop"]),
                }
            )
    write_csv(tables / "table_appendix_top5_semantic_edges.csv", output)
    return output


def build_js_value_table(edges: list[dict[str, str]], tables: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_groups: dict[tuple[int, int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in edges:
        if row["edge_type"] not in RELATION_ORDER:
            continue
        episode_groups[
            (int(row["training_seed"]), int(row["episode"]), row["event"], row["edge_type"])
        ].append(row)

    episode_rows: list[dict[str, Any]] = []
    for (seed, episode, event, edge_type), group in sorted(episode_groups.items()):
        js = [value for value in (as_float(row["policy_js_divergence"]) for row in group) if value is not None]
        critic = [value for value in (as_float(row["critic_value_drop"]) for row in group) if value is not None]
        probability = [value for value in (as_float(row["absolute_probability_change"]) for row in group) if value is not None]
        episode_rows.append(
            {
                "training_seed": seed,
                "episode": episode,
                "event": event,
                "edge_type": edge_type,
                "n_edges": len(group),
                "mean_policy_js_divergence": mean(js) if js else None,
                "mean_signed_critic_value_drop": mean(critic) if critic else None,
                "mean_abs_critic_value_change": mean([abs(value) for value in critic]) if critic else None,
                "mean_absolute_probability_change": mean(probability) if probability else None,
            }
        )
    write_csv(tables / "table_appendix_episode_edge_diagnostics.csv", episode_rows)

    seed_groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        seed_groups[(int(row["training_seed"]), row["event"], row["edge_type"])].append(row)
    seed_rows: list[dict[str, Any]] = []
    metrics = (
        "mean_policy_js_divergence",
        "mean_signed_critic_value_drop",
        "mean_abs_critic_value_change",
        "mean_absolute_probability_change",
    )
    for (seed, event, edge_type), group in sorted(seed_groups.items()):
        record: dict[str, Any] = {
            "training_seed": seed,
            "event": event,
            "event_label": EVENT_LABELS[event].replace("\n", " "),
            "edge_type": edge_type,
            "relation_label": RELATION_LABELS[edge_type],
            "n_execution_episodes_with_valid_edges": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            record[metric] = mean(values) if values else None
            record[f"within_seed_episode_std_{metric}"] = sample_std(values) if values else None
        seed_rows.append(record)
    write_csv(tables / "table_appendix_seed_js_value_diagnostics.csv", seed_rows)

    across_seed: list[dict[str, Any]] = []
    for metric in metrics:
        across_seed.extend(
            aggregate_seed_metric(
                seed_rows,
                ("event", "event_label", "edge_type", "relation_label"),
                metric,
            )
        )
    write_csv(tables / "table_appendix_six_seed_js_value_diagnostics.csv", across_seed)
    return seed_rows, across_seed


def build_case_table(timelines: list[dict[str, str]], tables: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in timelines:
        groups[(int(row["training_seed"]), int(row["execution_seed"]))].append(row)
    terminal_rows: list[dict[str, Any]] = []
    normalized_timelines: list[dict[str, Any]] = []
    for (seed, execution_seed), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["time"]))
        terminal = ordered[-1]
        is_median = as_bool(terminal["is_median_case"])
        terminal_rows.append(
            {
                "training_seed": seed,
                "execution_seed": execution_seed,
                "is_seed_median_service_case": is_median,
                "terminal_time": int(terminal["time"]),
                "terminal_service_level": as_float(terminal["service_level"]),
                "terminal_deliveries": as_float(terminal["deliveries"]),
                "terminal_deadline_miss_rate": as_float(terminal["deadline_miss_rate"]),
                "terminal_contentions": as_float(terminal["contentions"]),
                "terminal_forced_failures": as_float(terminal["forced_failures"]),
                "terminal_repairs": as_float(terminal["repairs"]),
            }
        )
        for row in ordered:
            normalized_timelines.append(
                {
                    "training_seed": seed,
                    "execution_seed": execution_seed,
                    "is_median_case": is_median,
                    "time": int(row["time"]),
                    "service_level": as_float(row["service_level"]),
                    "deliveries": as_float(row["deliveries"]),
                    "deadline_miss_rate": as_float(row["deadline_miss_rate"]),
                    "active_agents": as_float(row["active_agents"]),
                    "active_tasks": as_float(row["active_tasks"]),
                    "contentions": as_float(row["contentions"]),
                    "forced_failures": as_float(row["forced_failures"]),
                    "repairs": as_float(row["repairs"]),
                }
            )
    write_csv(tables / "table_appendix_case_terminal_summary.csv", terminal_rows)

    seed_median = [row for row in terminal_rows if row["is_seed_median_service_case"]]
    if sorted(row["training_seed"] for row in seed_median) != list(FORMAL_SEEDS):
        raise ValueError("each training seed must contribute exactly one median-service case")
    service_median = median(float(row["terminal_service_level"]) for row in seed_median)
    representative = min(
        seed_median,
        key=lambda row: (
            abs(float(row["terminal_service_level"]) - service_median),
            int(row["training_seed"]),
        ),
    )
    return terminal_rows, representative, normalized_timelines


def scenario_events(checkpoints: list[dict[str, Any]], representative_seed: int) -> tuple[int, int]:
    record = next(
        item for item in checkpoints if int(item.get("training_seed")) == representative_seed
    )
    config = record["scenario_config"]
    return int(config["scheduled_breakdown_step"]), int(config["scheduled_repair_step"])


def write_latex_main_table(path: Path, rows: list[dict[str, Any]]) -> None:
    order = {(event, relation): i for i, (event, relation) in enumerate(
        (event, relation) for event in EVENT_ORDER for relation in RELATION_ORDER
    )}
    rows = sorted(rows, key=lambda row: order[(row["event"], row["relation"])])
    lines = [
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"Disruption stage & Relation type & Top-edge $|\Delta p|$ (mean, 95\% CI) \\",
        r"\midrule",
    ]
    for row in rows:
        event = row["event_label"].replace("_", r"\_")
        relation = row["relation_label"].replace("_", r"\_")
        value = f"{float(row['mean']):.4f} [{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}]"
        lines.append(f"{event} & {relation} & {value} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_matplotlib() -> Any:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for figures. Install the project 'plot' extra "
            "or rerun with --skip-plots to generate only tables."
        ) from exc
    font_manager.findfont("Arial", fallback_to_default=False)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            # These panels are inserted at about 97% of their native size.
            # Eight-point source text consequently remains comfortably above
            # 7 pt in the final two-column manuscript.
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )
    return plt


def save_figure(fig: Any, base: Path, formats: Iterable[str]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(base.with_suffix(f".{extension}"), dpi=400)


def plot_masking_sensitivity(plt: Any, rows: list[dict[str, Any]], figures: Path, formats: Iterable[str]) -> None:
    lookup = {(row["event"], row["relation"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(3.15, 2.45), constrained_layout=True)
    x = np.arange(len(EVENT_ORDER), dtype=float)
    styles = {
        "task_coalition": ("#0072B2", "o"),
        "shared_zone": ("#D55E00", "s"),
    }
    for relation, offset in zip(RELATION_ORDER, (-0.08, 0.08)):
        selected = [lookup[(event, relation)] for event in EVENT_ORDER]
        center = np.asarray([float(row["mean"]) for row in selected])
        low = np.asarray([float(row["ci95_low"]) for row in selected])
        high = np.asarray([float(row["ci95_high"]) for row in selected])
        color, marker = styles[relation]
        ax.errorbar(
            x + offset,
            center,
            yerr=np.vstack([center - low, high - center]),
            color=color,
            marker=marker,
            linestyle="-",
            capsize=2.2,
            label=RELATION_LABELS[relation],
        )
    ax.set_xticks(x, [EVENT_LABELS[event] for event in EVENT_ORDER])
    ax.set_xlabel("Resource-disruption stage")
    # Use the Unicode glyph so the exported PDF contains only the selected
    # Arial/Helvetica family instead of Matplotlib's DejaVu math fallback.
    ax.set_ylabel("Top-edge change in selection probability |Δp|")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=1, frameon=False)
    save_figure(fig, figures / "fig_semantic_masking_sensitivity_A", formats)
    plt.close(fig)


def plot_representative_timeline(
    plt: Any,
    representative: dict[str, Any],
    timelines: list[dict[str, Any]],
    breakdown_step: int,
    repair_step: int,
    figures: Path,
    formats: Iterable[str],
) -> None:
    rows = [
        row for row in timelines
        if int(row["training_seed"]) == int(representative["training_seed"])
        and int(row["execution_seed"]) == int(representative["execution_seed"])
    ]
    rows.sort(key=lambda row: int(row["time"]))
    time = np.asarray([int(row["time"]) for row in rows])
    active = np.asarray([float(row["active_agents"]) for row in rows])
    deliveries = np.asarray([float(row["deliveries"]) for row in rows])

    fig, ax = plt.subplots(figsize=(3.15, 2.45), constrained_layout=True)
    ax2 = ax.twinx()
    active_line = ax.step(time, active, where="post", color="#0072B2", label="Active resources")[0]
    delivery_line = ax2.step(time, deliveries, where="post", color="#D55E00", label="Cumulative deliveries")[0]
    breakdown = ax.axvline(breakdown_step, color="#222222", linestyle="--", linewidth=1.0, label="Breakdown")
    restoration = ax.axvline(repair_step, color="#777777", linestyle=":", linewidth=1.1, label="Capacity restoration")
    ax.set_xlabel("Simulator transition")
    ax.set_ylabel("Active resources", color="#0072B2")
    ax2.set_ylabel("Cumulative deliveries", color="#D55E00")
    ax.tick_params(axis="y", colors="#0072B2")
    ax2.tick_params(axis="y", colors="#D55E00")
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(bottom=0.0)
    ax2.set_ylim(bottom=0.0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.legend(
        [active_line, delivery_line, breakdown, restoration],
        ["Active resources", "Cumulative deliveries", "Breakdown", "Capacity restoration"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.8,
    )
    save_figure(fig, figures / "fig_semantic_representative_timeline_B", formats)
    plt.close(fig)


def plot_appendix_metric(
    plt: Any,
    rows: list[dict[str, Any]],
    metric_name: str,
    y_label: str,
    output_name: str,
    figures: Path,
    formats: Iterable[str],
) -> None:
    selected = [row for row in rows if row["metric"] == metric_name]
    lookup = {(row["event"], row.get("edge_type", row.get("relation"))): row for row in selected}
    fig, ax = plt.subplots(figsize=(3.15, 2.45), constrained_layout=True)
    x = np.arange(len(EVENT_ORDER), dtype=float)
    for relation, offset, color, marker in (
        ("task_coalition", -0.08, "#0072B2", "o"),
        ("shared_zone", 0.08, "#D55E00", "s"),
    ):
        points = [lookup[(event, relation)] for event in EVENT_ORDER]
        center = np.asarray([float(row["mean"]) for row in points])
        low = np.asarray([float(row["ci95_low"]) for row in points])
        high = np.asarray([float(row["ci95_high"]) for row in points])
        ax.errorbar(
            x + offset,
            center,
            yerr=np.vstack([center - low, high - center]),
            color=color,
            marker=marker,
            linestyle="-",
            capsize=2.2,
            label=RELATION_LABELS[relation],
        )
    ax.set_xticks(x, [EVENT_LABELS[event] for event in EVENT_ORDER])
    ax.set_xlabel("Resource-disruption stage")
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), frameon=False)
    save_figure(fig, figures / output_name, formats)
    plt.close(fig)


def write_summary(
    root: Path,
    representative: dict[str, Any],
    breakdown_step: int,
    repair_step: int,
    main_stats: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    plots_generated: bool,
) -> None:
    summary = {
        "scenario": "robot_breakdown",
        "statistical_unit": "independently trained policy seed",
        "formal_training_seeds": list(FORMAL_SEEDS),
        "execution_episodes_per_seed": 20,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "main_masking_metric": "mean top-edge absolute selection-probability change",
        "representative_case_selection": "among each seed's median-service case, choose the case closest to the median terminal service level across the six training seeds; training seed breaks ties",
        "representative_case": representative,
        "configured_breakdown_step": breakdown_step,
        "configured_repair_step": repair_step,
        "claim_boundary": "model-internal structural sensitivity; not physical or causal explanation",
        "paper_claim": "CHORD provides quantitative structural transparency by exposing how semantically defined coalition and shared-zone relations influence policy predictions during resource disruption.",
        "main_statistics": main_stats,
        "paired_stage_contrasts": contrasts,
        "plots_generated": plots_generated,
        "raw_file_manifest": manifest,
    }
    (root / "semantic_paper_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    global WRITE_DERIVED_DATA
    args = parse_args()
    WRITE_DERIVED_DATA = not args.figures_only
    root = args.root
    tables = root / "tables"
    processed = root / "processed"
    figures = args.figures_output
    if WRITE_DERIVED_DATA:
        tables.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)

    summaries, edges, timelines, checkpoints, manifest = load_inputs(root)
    write_csv(processed / "semantic_event_summary_6seeds.csv", summaries)
    write_csv(processed / "edge_attributions_6seeds.csv", edges)
    write_csv(processed / "case_timelines_6seeds.csv", timelines)

    main_stats, contrasts = build_main_tables(summaries, tables)
    group_stats = build_group_masking_table(summaries, tables)
    build_top_edges(edges, tables)
    _, js_value_stats = build_js_value_table(edges, tables)
    _, representative, normalized_timelines = build_case_table(timelines, tables)
    breakdown_step, repair_step = scenario_events(
        checkpoints, int(representative["training_seed"])
    )
    if WRITE_DERIVED_DATA:
        write_latex_main_table(tables / "table_main_semantic_sensitivity.tex", main_stats)

    plots_generated = not args.skip_plots
    if plots_generated:
        plt = configure_matplotlib()
        plot_masking_sensitivity(plt, main_stats, figures, args.formats)
        plot_representative_timeline(
            plt,
            representative,
            normalized_timelines,
            breakdown_step,
            repair_step,
            figures,
            args.formats,
        )
        plot_appendix_metric(
            plt,
            js_value_stats,
            "mean_policy_js_divergence",
            "Policy Jensen–Shannon divergence",
            "fig_appendix_policy_js_divergence",
            figures,
            args.formats,
        )
        plot_appendix_metric(
            plt,
            group_stats,
            "mean_abs_group_value_drop",
            "Absolute critic-value change",
            "fig_appendix_group_value_change",
            figures,
            args.formats,
        )

    if WRITE_DERIVED_DATA:
        write_summary(
            root,
            representative,
            breakdown_step,
            repair_step,
            main_stats,
            contrasts,
            manifest,
            plots_generated,
        )
    print(
        json.dumps(
            {
                "root": str(root),
                "training_seeds": list(FORMAL_SEEDS),
                "summary_rows": len(summaries),
                "edge_rows": len(edges),
                "timeline_rows": len(timelines),
                "representative_case": representative,
                "plots_generated": plots_generated,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
