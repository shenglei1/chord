from __future__ import annotations

import csv
import json

from open_hyper_warehouse.training_exports import export_training_data


def test_export_in_progress_training_data(tmp_path):
    run = tmp_path / "chord_mappo" / "seed_45"
    run.mkdir(parents=True)
    (run / "run_metadata.json").write_text(
        json.dumps(
            {
                "device": "cuda",
                "environment_backend": "process",
                "parallel_env_workers": 8,
            }
        ),
        encoding="utf-8",
    )
    row = {"steps": 2048, "updates": 1, "elapsed_seconds": 2.0,
           "environment_steps_per_second": 1024.0, "episode_service_level": 0.8}
    (run / "training_metrics.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    (run / "progress.json").write_text(json.dumps(row), encoding="utf-8")

    manifest = export_training_data(tmp_path)

    assert manifest["training_runs"] == 1
    assert manifest["complete_runs"] == 0
    with (tmp_path / "training_curves.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["algorithm"] == "chord_mappo"
    assert rows[0]["episode_service_level"] == "0.8"
    with (tmp_path / "training_efficiency.csv").open(encoding="utf-8-sig") as handle:
        efficiency = list(csv.DictReader(handle))
    assert efficiency[0]["environment_backend"] == "process"
    assert efficiency[0]["run_complete"] == "False"
