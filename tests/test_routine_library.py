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


def test_sweep_forearms_are_parallel() -> None:
    for block in L.values():
        if block.kind is BlockKind.SWEEP:
            assert all(kf.right.fore == kf.left.fore for kf in block.keyframes), (
                block.id
            )


def test_sweep_crossing_arm_keeps_upper_arm_forward() -> None:
    """From the recording: with both forearms pointing left, the left arm
    extends to the left while the right arm crosses the body with its upper
    arm pointing forward."""
    first = L["sweep_to_right_4"].keyframes[0]  # forearms point to the left
    assert first.left.upper[0] < -0.4  # leading arm extends sideways
    crossing = first.right.upper
    assert abs(crossing[0]) < 0.2 and crossing[2] > 0.9  # crossing arm: forward


@pytest.mark.parametrize(
    ("block_id", "axis", "increasing"),
    [
        ("sweep_to_right_4", 0, True),
        ("sweep_to_left_4", 0, False),
        ("sweep_down_4", 1, False),
        ("sweep_up_4", 1, True),
    ],
)
def test_sweep_travels_monotonically(
    block_id: str, axis: int, increasing: bool
) -> None:
    values = [f[axis] for f in _fores(block_id)]
    assert values == sorted(values, reverse=not increasing)
    assert values[0] * values[-1] < 0  # crosses the midline / shoulder level


def test_diagonal_sweep_goes_upper_left_to_lower_right() -> None:
    fores = _fores("sweep_diagonal_down_to_right_4")
    assert fores[0][0] < 0 < fores[0][1]  # starts upper left
    assert fores[-1][1] < 0 < fores[-1][0]  # ends lower right


# --- Knocks ------------------------------------------------------------------


def test_knock_is_same_hand_twice_with_fist_and_bent_elbow() -> None:
    """Reference: same hand on beats 1 & 2, fist, upper arm level from the
    shoulder, forearm up. Recording: upper arm ~45° forward, forearm tilted
    forward towards the "door"."""
    first, second = L["knock_right"].keyframes
    assert first.right == second.right
    assert first.left is None and second.left is None
    arm = first.right
    assert arm.hand_shape is HandShape.FIST
    assert abs(arm.upper[1]) < 0.2  # roughly level upper arm
    assert arm.upper[0] > 0.5 and arm.upper[2] > 0.5  # out and forward
    assert (
        angle_between_deg(arm.fore, UP) < 50 and arm.fore[2] > 0
    )  # up, tilted forward
    assert 55 < angle_between_deg(arm.upper, arm.fore) < 100  # clearly bent elbow


# --- Chops -------------------------------------------------------------------


def test_chop_reaches_up_and_out_then_salutes() -> None:
    chop = L["chop_right"]
    out, salute = chop.keyframes
    # 25° up and out, as danced at game speed in every recording (the
    # reference's 45° was never reached).
    assert angle_between_deg(out.right.fore, RIGHT) == pytest.approx(25, abs=0.5)
    assert angle_between_deg(out.right.fore, UP) == pytest.approx(65, abs=0.5)
    # More allowance than other sharp keyframes: dancers often reach lower.
    assert chop.tolerance_for(out) == 40 > chop.tolerance_for(salute)
    assert out.left is None and salute.left is None  # one-armed move
    assert out.right.hand_shape is HandShape.OPEN
    # Salute: elbow at shoulder height, forearm across the front of the chest...
    arm = salute.right
    assert abs(arm.upper[1]) < 0.2 and arm.upper[2] > 0  # level elbow, in front
    assert arm.fore[0] < 0 and arm.fore[2] > 0  # inwards, in front of the chest
    assert abs(arm.fore[1]) < 0.35
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

    def wrist_x(arm) -> float:  # in shoulder widths, segment ratios ~0.62
        return 0.5 + 0.62 * arm.upper[0] + 0.62 * arm.fore[0]

    # The hand travels from crossed (left of the right shoulder) to the right.
    xs = [wrist_x(kf.right) for kf in kfs[:3]]
    assert xs == sorted(xs) and xs[0] < 0.25 < xs[-1]
    # From the recordings: forearm roughly level, pointing inwards, in front.
    for kf in kfs[:3]:
        assert abs(kf.right.fore[1]) < 0.3 and kf.right.fore[0] < 0 < kf.right.fore[2]
        assert kf.right.hand is not None  # the flick direction
    # Recalibrated 2026-10-23 (two sessions at 150 BPM): on the front and side
    # flicks the forearm turns from across the chest to forward and slightly
    # up, most of all on the side flick.
    crossed, front, side = (kf.right.fore for kf in kfs[:3])
    assert front[1] > 0 and side[1] > 0
    assert (
        side[2] > front[2] > crossed[2] - 0.1
        and abs(side[0]) < abs(front[0]) < abs(crossed[0]) + 0.1
    )


def test_clap_is_scored_only_as_a_clap() -> None:
    """Any clap counts, wherever it happens: no arm targets on the clap."""
    for block_id in ("flick_clap_to_right", "flick_clap_to_left"):
        clap = L[block_id].keyframes[-1]
        assert clap.clap and clap.right is None and clap.left is None


# --- Wrist roll (unverified draft) ---------------------------------------------


def test_wrist_roll_holds_position_every_count() -> None:
    """The rolling is a wrist rotation the body landmarks cannot see, so the
    keyframes hold one position per count (Step 4 adds a "keep moving" check)."""
    roll = L["wrist_roll_4"]
    assert [kf.beat for kf in roll.keyframes] == [0, 1, 2, 3]
    first = roll.keyframes[0]
    assert all(
        kf.right == first.right and kf.left == first.left for kf in roll.keyframes
    )
    assert first.left == first.right.mirrored()
    # Forearms level and pointing inwards, elbows in front of the body.
    assert first.right.fore[0] < -0.6 and abs(first.right.fore[1]) < 0.2
    assert first.right.upper[2] > 0.6


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
                assert angle_between_deg(arm.upper, inward) > 45, (
                    block.id,
                    kf.beat,
                    side,
                )
