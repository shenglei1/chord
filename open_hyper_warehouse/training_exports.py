"""Export crash-tolerant training telemetry into paper-ready flat tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_training_data(root: str | Path) -> dict[str, Any]:
    """Collect complete and in-progress runs without modifying their contents."""

    root = Path(root)
    curve_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []

    for jsonl_path in sorted(root.glob("*/seed_*/training_metrics.jsonl")):
        run_dir = jsonl_path.parent
        algorithm = run_dir.parent.name
        seed_text = run_dir.name.removeprefix("seed_")
        metadata_path = run_dir / "run_metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            curve_rows.append(
                {
                    "algorithm": algorithm,
                    "training_seed": int(seed_text),
                    **json.loads(line),
                }
            )

        progress_path = run_dir / "progress.json"
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.exists()
            else {}
        )
        checkpoint_path = run_dir / "final.pt"
        checkpoint: dict[str, Any] = {}
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        efficiency_rows.append(
            {
                "algorithm": algorithm,
                "training_seed": int(seed_text),
                "run_complete": checkpoint_path.exists(),
                "steps": checkpoint.get("steps", progress.get("steps")),
                "updates": checkpoint.get("updates", progress.get("updates")),
                "wall_seconds": checkpoint.get(
                    "wall_seconds", progress.get("elapsed_seconds")
                ),
                "environment_steps_per_second": progress.get(
                    "environment_steps_per_second"
                ),
                "parameter_count": checkpoint.get("parameter_count"),
                "peak_memory_mb": checkpoint.get(
                    "peak_memory_mb", progress.get("cuda_peak_memory_allocated_mb")
                ),
                "device": checkpoint.get("training_device", metadata.get("device")),
                "cuda_device_name": checkpoint.get("cuda_device_name"),
                "environment_backend": checkpoint.get(
                    "environment_backend", metadata.get("environment_backend")
                ),
                "parallel_env_workers": checkpoint.get(
                    "parallel_env_workers", metadata.get("parallel_env_workers")
                ),
            }
        )

    curves_path = root / "training_curves.csv"
    efficiency_path = root / "training_efficiency.csv"
    _write_csv(curves_path, curve_rows)
    _write_csv(efficiency_path, efficiency_rows)
    manifest = {
        "training_curve_rows": len(curve_rows),
        "training_runs": len(efficiency_rows),
        "complete_runs": sum(bool(row["run_complete"]) for row in efficiency_rows),
        "training_curves": str(curves_path.resolve()),
        "training_efficiency": str(efficiency_path.resolve()),
    }
    target = root / "training_export_manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(target)
    return manifest
