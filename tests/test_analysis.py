import numpy as np

from open_hyper_warehouse.analysis import _paired_sign_flip_pvalue, aggregate_reports


def _report(algorithm: str, seed: int, score: float) -> dict:
    return {
        "algorithm": algorithm,
        "training_seed": seed,
        "scenarios": {"shift": {"deliveries": {"mean": score}}},
    }


def test_aggregation_uses_paired_training_seeds() -> None:
    reports = [
        _report("chord_mappo", 0, 5.0),
        _report("chord_mappo", 1, 7.0),
        _report("graph_mappo", 0, 4.0),
        _report("graph_mappo", 1, 3.0),
    ]
    aggregate = aggregate_reports(reports, bootstrap_samples=100, seed=0)
    proposed = aggregate["algorithms"]["chord_mappo"]["shift"]["deliveries"]
    paired = aggregate["paired_differences"]["graph_mappo"]["shift"]["deliveries"]
    assert proposed["n_training_seeds"] == 2
    assert proposed["mean"] == 6.0
    assert paired["training_seeds"] == [0, 1]
    assert paired["mean_difference"] == 2.5
    assert paired["sign_flip_p"] is not None
    assert paired["holm_adjusted_p"] is not None


def test_operational_comparison_uses_reference_training_seeds() -> None:
    reports = [
        _report("chord_mappo", 0, 5.0),
        _report("chord_mappo", 1, 7.0),
        {
            "algorithm": "rolling_horizon",
            "scenarios": {"shift": {"deliveries": {"mean": 4.0}}},
        },
    ]
    aggregate = aggregate_reports(reports, bootstrap_samples=100, seed=0)
    record = aggregate["operational_differences"]["rolling_horizon"]["shift"]["deliveries"]
    assert record["n_reference_training_seeds"] == 2
    assert record["mean_difference"] == 2.0
    assert record["training_seeds"] == [0, 1]


def test_six_seed_exact_sign_flip_resolution_is_reported_honestly() -> None:
    assert _paired_sign_flip_pvalue(np.ones(6), seed=0) == 0.03125
