from open_hyper_warehouse.paper_results import build_eaai_tables


def _report(method, seed, service, misses):
    report = {
        "algorithm": method,
        "scenarios": {
            "robot_breakdown": {
                "service_level": {"mean": service},
                "throughput_per_100_steps": {"mean": service * 10},
                "deadline_miss_rate": {"mean": misses},
                "mean_flow_time": {"mean": 10 + misses},
            }
        },
    }
    if seed is not None:
        report["training_seed"] = seed
    return report


def test_direction_aware_tables_do_not_treat_lower_cost_as_worse():
    reports = [
        _report("chord_mappo", 0, 0.8, 0.1),
        _report("chord_mappo", 1, 0.9, 0.2),
        _report("graph_mappo", 0, 0.7, 0.3),
        _report("graph_mappo", 1, 0.6, 0.4),
        _report("rolling_horizon", None, 0.5, 0.5),
    ]
    result = build_eaai_tables(reports)
    best = {
        (row["metric"], row["method"])
        for row in result["rank_rows"]
        if row["is_best_mean"]
    }
    assert ("service_level", "chord_mappo") in best
    assert ("deadline_miss_rate", "chord_mappo") in best


def test_degradation_is_oriented_so_positive_always_means_worse():
    nominal = _report("chord_mappo", 0, 0.8, 0.1)
    nominal["scenarios"]["in_distribution"] = nominal["scenarios"].pop(
        "robot_breakdown"
    )
    stress = _report("chord_mappo", 1, 0.6, 0.3)
    result = build_eaai_tables([nominal, stress])
    degradation = {
        row["metric"]: row["oriented_degradation"]
        for row in result["degradation_rows"]
    }
    assert degradation["service_level"] > 0
    assert degradation["deadline_miss_rate"] > 0
