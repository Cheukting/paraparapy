"""Tests for parapara.routine.geometry."""

import math

import pytest

from parapara.routine.geometry import (
    FORWARD,
    RIGHT,
    UP,
    angle_between_deg,
    mirror_x,
    normalize,
    slerp,
    unit,
)


def test_unit_normalizes() -> None:
    v = unit(3, 0, 4)
    assert v == pytest.approx((0.6, 0.0, 0.8))
    assert math.hypot(*v) == pytest.approx(1.0)


def test_unit_rejects_zero_vector() -> None:
    with pytest.raises(ValueError):
        unit(0, 0, 0)


def test_normalize_is_idempotent() -> None:
    v = normalize((1, 2, 3))
    assert normalize(v) == v  # exactly, not approximately


def test_normalize_preserves_unit_vectors_bit_for_bit() -> None:
    # Regression: re-normalizing a unit vector used to change its last bits,
    # so mirroring a pose twice did not compare equal to the original.
    v = unit(1, 1, 0)
    assert normalize(mirror_x(mirror_x(v))) == v


def test_mirror_x_negates_only_x() -> None:
    assert mirror_x((1.0, 2.0, 3.0)) == (-1.0, 2.0, 3.0)
    assert mirror_x(mirror_x((0.3, -0.2, 0.9))) == (0.3, -0.2, 0.9)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (RIGHT, RIGHT, 0.0),
        (RIGHT, UP, 90.0),
        (RIGHT, mirror_x(RIGHT), 180.0),
        (RIGHT, (1, 1, 0), 45.0),
        (FORWARD, (0, 0, 5), 0.0),  # length does not matter
    ],
)
def test_angle_between_deg(a, b, expected) -> None:
    assert angle_between_deg(a, b) == pytest.approx(expected)


def test_slerp_end_points_and_midpoint() -> None:
    assert slerp(RIGHT, UP, 0) == pytest.approx(RIGHT)
    assert slerp(RIGHT, UP, 1) == pytest.approx(UP)
    mid = slerp(RIGHT, UP, 0.5)
    assert angle_between_deg(mid, RIGHT) == pytest.approx(45)
    assert angle_between_deg(mid, UP) == pytest.approx(45)


def test_slerp_constant_angular_speed() -> None:
    steps = [slerp(RIGHT, FORWARD, t / 3) for t in range(4)]
    gaps = [angle_between_deg(a, b) for a, b in zip(steps, steps[1:])]
    assert gaps == pytest.approx([30, 30, 30])


def test_slerp_rejects_opposite_directions() -> None:
    with pytest.raises(ValueError):
        slerp(RIGHT, mirror_x(RIGHT), 0.5)


def test_angle_between_is_robust_to_rounding() -> None:
    # Nearly identical vectors can make the dot product exceed 1 by rounding.
    v = unit(0.1, 0.2, 0.3)
    assert angle_between_deg(v, v) == pytest.approx(0.0, abs=1e-6)
