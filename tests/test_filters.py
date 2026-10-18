"""Tests for the One Euro filter (parapara.pose.filters)."""

import numpy as np
import pytest

from parapara.pose.filters import OneEuroFilter, filter_series

MS = 1_000_000


def test_first_sample_passes_through() -> None:
    f = OneEuroFilter()
    np.testing.assert_array_equal(f([1.0, 2.0], 0), [1.0, 2.0])


def test_matches_hand_computed_low_pass_with_beta_zero() -> None:
    """With beta = 0 it is a plain exponential filter at min_cutoff."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f(0.0, 0)
    out = f(1.0, 40 * MS)  # 25 Hz
    tau, te = 1 / (2 * np.pi), 0.04
    assert float(out) == pytest.approx(1 / (1 + tau / te))


def test_reduces_jitter_at_rest() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(200) * 43 * MS  # ~23 fps, like the recordings
    noisy = 0.5 + rng.normal(0, 0.01, size=200)
    smooth = filter_series(noisy, t, min_cutoff=1.0, beta=0.0)
    assert smooth[50:].std() < noisy[50:].std() / 2


def test_beta_reduces_lag_on_fast_moves() -> None:
    """A fast ramp: the higher beta is, the closer the output follows."""
    t = np.arange(60) * 43 * MS
    ramp = np.linspace(0, 1.5, 60)  # ~0.6 m/s
    slow = filter_series(ramp, t, min_cutoff=0.5, beta=0.0)
    fast = filter_series(ramp, t, min_cutoff=0.5, beta=5.0)
    assert abs(fast[-1] - ramp[-1]) < abs(slow[-1] - ramp[-1]) / 3


def test_handles_irregular_timestamps() -> None:
    """Skipped camera frames: a longer gap moves the output further."""
    a, b = OneEuroFilter(), OneEuroFilter()
    for f in (a, b):
        f(0.0, 0)
    assert float(b(1.0, 200 * MS)) > float(a(1.0, 40 * MS))


def test_repeated_timestamp_keeps_last_output() -> None:
    f = OneEuroFilter()
    f(0.0, 0)
    first = f(1.0, 40 * MS)
    np.testing.assert_array_equal(f(5.0, 40 * MS), first)


def test_missing_values_reset_only_those_elements() -> None:
    f = OneEuroFilter(min_cutoff=0.5)
    f([0.0, 0.0], 0)
    out = f([np.nan, 1.0], 40 * MS)
    assert np.isnan(out[0]) and 0 < out[1] < 1
    out = f([3.0, 1.0], 80 * MS)
    assert out[0] == 3.0  # restarted from its new value, no smoothing towards stale data


def test_shape_change_restarts() -> None:
    f = OneEuroFilter()
    f(np.zeros(3), 0)
    np.testing.assert_array_equal(f(np.ones((2, 2)), 40 * MS), np.ones((2, 2)))


@pytest.mark.parametrize("kwargs", [{"min_cutoff": 0}, {"d_cutoff": -1}, {"beta": -0.1}])
def test_rejects_bad_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        OneEuroFilter(**kwargs)
