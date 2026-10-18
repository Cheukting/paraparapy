"""Tests for the draft block library.

These pin down the *interpretation* of each move written in library.py (and
checked against the EKS-D "How to Dance" reference), so a later refinement,
e.g. from motion capture, that changes a move's meaning is a visible,
deliberate test change rather than a silent one.
"""

import pytest

from parapara.routine.geometry import LEFT, RIGHT, UP, angle_between_deg
from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import ALLOWED_DURATIONS, BlockKind, HandShape

L = BLOCK_LIBRARY


def test_every_kind_and_allowed_duration_is_available() -> None:
    available = {(b.kind, b.duration) for b in L.values()}
    for kind, durations in ALLOWED_DURATIONS.items():
        for d in durations:
            assert (kind, d) in available, f"no {kind} block lasting {d} counts"


def test_ids_match_keys() -> None:
    assert all(key == block.id for key, block in L.items())


@pytest.mark.parametrize(
    ("right_id", "left_id"),
    [
        ("chop_right", "chop_left"),
        ("knock_right", "knock_left"),
        ("flick_clap_to_right", "flick_clap_to_left"),
        ("sweep_to_right_4", "sweep_to_left_4"),
        ("sweep_to_right_2", "sweep_to_left_2"),
        ("sweep_diagonal_down_to_right_4", "sweep_diagonal_down_to_left_4"),
        ("sweep_diagonal_up_to_right_4", "sweep_diagonal_up_to_left_4"),
    ],
)
def test_left_variants_are_mirror_images(right_id: str, left_id: str) -> None:
    assert L[right_id].mirrored(left_id).keyframes == L[left_id].keyframes


# --- Sweeps ------------------------------------------------------------------


def _fores(block_id: str) -> list[tuple[float, float, float]]:
    return [kf.right.fore for kf in L[block_id].keyframes]


def test_sweep_arms_are_parallel() -> None:
    for block in L.values():
        if block.kind is BlockKind.SWEEP:
            assert all(kf.right == kf.left for kf in block.keyframes), block.id


@pytest.mark.parametrize(
    ("block_id", "axis", "increasing"),
    [
        ("sweep_to_right_4", 0, True),
        ("sweep_to_left_4", 0, False),
        ("sweep_down_4", 1, False),
        ("sweep_up_4", 1, True),
    ],
)
def test_sweep_travels_monotonically(block_id: str, axis: int, increasing: bool) -> None:
    values = [f[axis] for f in _fores(block_id)]
    assert values == sorted(values, reverse=not increasing)
    assert values[0] * values[-1] < 0  # crosses the midline / shoulder level


def test_diagonal_sweep_goes_upper_left_to_lower_right() -> None:
    fores = _fores("sweep_diagonal_down_to_right_4")
    assert fores[0][0] < 0 < fores[0][1]  # starts upper left
    assert fores[-1][1] < 0 < fores[-1][0]  # ends lower right


# --- Knocks ------------------------------------------------------------------


def test_knock_is_same_hand_twice_with_fist_at_right_angle() -> None:
    """Reference: same hand on beats 1 & 2, fist, upper arm horizontal,
    forearm up from the elbow."""
    first, second = L["knock_right"].keyframes
    assert first.right == second.right
    assert first.left is None and second.left is None
    arm = first.right
    assert arm.hand_shape is HandShape.FIST
    assert arm.upper[1] == pytest.approx(0)  # horizontal upper arm
    assert angle_between_deg(arm.fore, UP) < 10  # forearm up
    assert angle_between_deg(arm.upper, arm.fore) == pytest.approx(90, abs=10)


# --- Chops -------------------------------------------------------------------


def test_chop_reaches_45_degrees_up_and_out_then_salutes() -> None:
    out, salute = L["chop_right"].keyframes
    assert angle_between_deg(out.right.fore, RIGHT) == pytest.approx(45)
    assert angle_between_deg(out.right.fore, UP) == pytest.approx(45)
    assert out.left is None and salute.left is None  # one-armed move
    assert out.right.hand_shape is HandShape.OPEN
    # Salute: forearm across the front of the chest, roughly level...
    arm = salute.right
    assert arm.fore[0] < 0  # pointing inwards, across the body
    assert arm.upper[2] > 0 and arm.fore[2] > 0  # in front of the chest
    assert abs(arm.fore[1]) < 0.2
    # ...with the flat hand continuing the forearm, not pointing down.
    assert arm.hand is not None and arm.hand[1] > -0.2
    assert angle_between_deg(arm.hand, arm.fore) < 15


# --- Flick + Clap --------------------------------------------------------------


def test_flick_clap_travels_across_the_body_then_claps() -> None:
    kfs = L["flick_clap_to_right"].keyframes
    assert [kf.beat for kf in kfs] == [0, 1, 2, 3]
    assert [kf.clap for kf in kfs] == [False, False, False, True]
    # One-armed flicks: the left arm is never scored during the flicks.
    assert all(kf.left is None for kf in kfs[:3])
    # Beat 1 crossed over the body, beat 3 on the opposite (right) side.
    assert kfs[0].right.fore[0] < 0 and kfs[0].right.upper[0] < 0
    assert kfs[2].right.upper[0] > 0
    # Every flick has a hand direction; flicks are just above shoulder height.
    assert all(kf.right.hand is not None and kf.right.fore[1] > 0 for kf in kfs[:3])


def test_clap_is_scored_only_as_a_clap() -> None:
    """Any clap counts, wherever it happens: no arm targets on the clap."""
    for block_id in ("flick_clap_to_right", "flick_clap_to_left"):
        clap = L[block_id].keyframes[-1]
        assert clap.clap and clap.right is None and clap.left is None


# --- Wrist roll (unverified draft) ---------------------------------------------


def test_wrist_roll_alternates_every_half_count() -> None:
    roll = L["wrist_roll_4"]
    assert [kf.beat for kf in roll.keyframes] == [i / 2 for i in range(8)]
    over = [kf.right.fore[1] > kf.left.fore[1] for kf in roll.keyframes]
    assert over == [True, False] * 4  # right hand over, under, over, ...


# --- Anatomical sanity -------------------------------------------------------


def test_upper_arms_stay_anatomically_plausible() -> None:
    """No draft asks for an upper arm pointing nearly straight up, or far
    across the body towards the other side."""
    for block in L.values():
        for kf in block.keyframes:
            for side, arm in (("right", kf.right), ("left", kf.left)):
                if arm is None:
                    continue
                assert angle_between_deg(arm.upper, UP) > 30, (block.id, kf.beat)
                inward = LEFT if side == "right" else RIGHT
                assert angle_between_deg(arm.upper, inward) > 45, (block.id, kf.beat, side)
