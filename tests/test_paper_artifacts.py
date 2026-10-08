import numpy as np

from open_hyper_warehouse.paper_artifacts import bootstrap_ci, interval


def test_interval_uses_training_seed_sample() -> None:
    values = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
    mean, low, high = interval(values, "ci95", samples=2_000, seed=7)
    assert np.isclose(mean, 0.75)
    assert low < mean < high


def test_std_interval_is_mean_plus_or_minus_seed_std() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    mean, low, high = interval(values, "std")
    assert mean == 2.0
    assert np.isclose(low, mean - values.std(ddof=1))
    assert np.isclose(high, mean + values.std(ddof=1))


def test_bootstrap_requires_at_least_two_training_seeds() -> None:
    assert bootstrap_ci([1.0]) == (None, None)

