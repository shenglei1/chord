"""Crash-tolerant incremental training telemetry for paper experiments."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import torch


class TrainingTelemetry:
    def __init__(self, output_dir: str | Path, metadata: dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "training_metrics.jsonl"
        self.progress_path = self.output_dir / "progress.json"
        self.started = time.perf_counter()
        self.started_wall = time.time()
        # A fresh non-reused training run owns its telemetry files.
        self.jsonl_path.write_text("", encoding="utf-8")
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    **metadata,
                    "started_unix_seconds": self.started_wall,
                    "telemetry_schema": 1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def record(self, row: dict[str, Any], device: torch.device) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        enriched = {
            **row,
            "elapsed_seconds": elapsed,
            "environment_steps_per_second": float(row.get("steps", 0) / elapsed),
            "cuda_memory_allocated_mb": (
                float(torch.cuda.memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else None
            ),
            "cuda_peak_memory_allocated_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else None
            ),
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            handle.flush()
        temporary = self.progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
        temporary.replace(self.progress_path)
        return enriched


def write_training_log(output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(output_dir) / "training_log.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
