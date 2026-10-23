"""Tests for parapara.routine.model: construction rules, phrase rules, timeline."""

import math

import pytest

from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import (
    COUNTS_PER_PHRASE,
    ArmPose,
    Block,
    BlockKind,
    Difficulty,
    HandShape,
    Keyframe,
    Phrase,
    Routine,
    RoutineError,
    phrase_issues,
)

L = BLOCK_LIBRARY

# A valid 32-count phrase used as the baseline for rule-violation tests.
VALID_IDS = [
    "sweep_to_right_4", "sweep_to_left_4",                          # bar 1
    "knock_right", "knock_left", "chop_right", "chop_right",        # bar 2
    "chop_left", "chop_left", "wrist_roll_4",                       # bar 3
    "sweep_diagonal_down_to_right_4", "flick_clap_to_right",        # bar 4
]  # fmt: skip


def _blocks(ids: list[str]) -> tuple[Block, ...]:
    return tuple(L[i] for i in ids)


POSE = ArmPose(upper=(1, 0, 0), fore=(0, 1, 0))


# --- ArmPose / Keyframe ------------------------------------------------------


def test_arm_pose_normalizes_directions() -> None:
    pose = ArmPose(upper=(2, 0, 0), fore=(0, 0, 3), hand=(0, 4, 0))
    assert pose.upper == (1.0, 0.0, 0.0)
    assert pose.fore == (0.0, 0.0, 1.0)
    assert pose.hand == (0.0, 1.0, 0.0)


def test_arm_pose_rejects_zero_direction() -> None:
    with pytest.raises(ValueError):
        ArmPose(upper=(0, 0, 0), fore=(0, 1, 0))


def test_arm_pose_mirror_keeps_hand_shape() -> None:
    pose = ArmPose(
        upper=(1, 1, 0), fore=(1, 0, 0), hand_shape=HandShape.FIST, hand=(1, 0, 1)
    )
    m = pose.mirrored()
    assert m.upper[0] == pytest.approx(-pose.upper[0])
    assert m.hand is not None and m.hand[0] == pytest.approx(-pose.hand[0])
    assert m.hand_shape is HandShape.FIST
    assert m.mirrored() == pose


def test_keyframe_mirror_swaps_arms() -> None:
    kf = Keyframe(beat=0, right=POSE)
    m = kf.mirrored()
    assert m.right is None
    assert m.left == POSE.mirrored()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beat": -1, "right": POSE},
        {"beat": math.nan, "right": POSE},
        {"beat": 0},  # nothing to score
        {"beat": 0, "right": POSE, "tolerance_deg": 0},
        {"beat": 0, "right": POSE, "tolerance_deg": 120},
    ],
)
def test_invalid_keyframes_rejected(kwargs) -> None:
    with pytest.raises(RoutineError):
        Keyframe(**kwargs)


def test_clap_only_keyframe_is_valid() -> None:
    assert Keyframe(beat=0, clap=True).clap


# --- Block -------------------------------------------------------------------


def test_block_rejects_wrong_duration() -> None:
    with pytest.raises(RoutineError, match="must last"):
        Block(
            id="x",
            kind=BlockKind.CHOP,
            duration=4,
            keyframes=(Keyframe(beat=0, right=POSE),),
        )


def test_block_reports_all_issues_together() -> None:
    with pytest.raises(RoutineError) as err:
        Block(
            id="x",
            kind=BlockKind.KNOCK,
            duration=2,
            keyframes=(Keyframe(beat=1, right=POSE), Keyframe(beat=0.5, right=POSE),
                       Keyframe(beat=2, right=POSE)),
        )  # fmt: skip
    assert len(err.value.issues) == 2  # out of range + not increasing


def test_flick_clap_must_end_with_clap() -> None:
    with pytest.raises(RoutineError, match="clap"):
        Block(id="x", kind=BlockKind.FLICK_CLAP, duration=4,
              keyframes=(Keyframe(beat=0, right=POSE),))  # fmt: skip


def test_tolerance_defaults_depend_on_style() -> None:
    sweep, chop = L["sweep_to_right_4"], L["chop_right"]
    assert sweep.is_flowing and not chop.is_flowing
    # chop.keyframes[1] uses the default; keyframes[0] has its own override.
    assert sweep.tolerance_for(sweep.keyframes[0]) > chop.tolerance_for(
        chop.keyframes[1]
    )
    override = Keyframe(beat=0, right=POSE, tolerance_deg=10)
    assert chop.tolerance_for(override) == 10


# --- Phrase rules ----------------------------------------------------------


def test_valid_phrase() -> None:
    blocks = _blocks(VALID_IDS)
    assert sum(b.duration for b in blocks) == COUNTS_PER_PHRASE
    assert phrase_issues(blocks) == []
    Phrase(blocks)


def test_empty_phrase() -> None:
    assert phrase_issues([]) == ["phrase has no blocks"]


def test_phrase_missing_kind() -> None:
    ids = [
        i if not i.startswith("wrist_roll") else "sweep_to_right_4" for i in VALID_IDS
    ]
    assert any("missing block kinds" in i for i in phrase_issues(_blocks(ids)))


def test_phrase_must_end_with_flick_clap() -> None:
    # Same blocks, but the finisher swapped with the sweep before it.
    ids = VALID_IDS[:-2] + ["flick_clap_to_right", "sweep_diagonal_down_to_right_4"]
    assert phrase_issues(_blocks(ids)) == [
        "phrase must end with flick + clap, ends with sweep_diagonal_down_to_right_4"
    ]


def test_phrase_wrong_length() -> None:
    issues = phrase_issues(_blocks(VALID_IDS[:-2] + ["flick_clap_to_right"]))
    assert issues == ["phrase must last 32 counts, got 28"]


def test_block_crossing_bar_line_rejected() -> None:
    # Three 2-count chops push a 4-count sweep to start on bar 1 count 7, so it
    # runs over into bar 2. Everything else is valid and totals 32 counts.
    blocks = _blocks(["chop_right", "chop_right", "chop_left", "sweep_to_right_4",
                      "sweep_to_left_2", "knock_right", "knock_left",
                      "wrist_roll_4", "knock_right", "knock_left",
                      "chop_left", "chop_left", "flick_clap_to_right"])  # fmt: skip
    assert phrase_issues(blocks) == [
        "sweep_to_right_4 at bar 1 count 7 crosses a bar line",
        "sweep_to_right_4 at bar 1 count 7 must start on a multiple of 4 counts",
    ]


def test_phrase_constructor_raises_with_issues() -> None:
    with pytest.raises(RoutineError) as err:
        Phrase(_blocks(["chop_right"]))
    assert len(err.value.issues) >= 3  # missing kinds, no finisher, wrong length


# --- Routine & timeline ------------------------------------------------------


def _routine(n_phrases: int = 2) -> Routine:
    phrase = Phrase(_blocks(VALID_IDS))
    return Routine(
        id="r", song_id="s", difficulty=Difficulty.NORMAL, phrases=(phrase,) * n_phrases
    )


def test_routine_requires_phrases_and_ids() -> None:
    with pytest.raises(RoutineError) as err:
        Routine(id="", song_id="", difficulty=Difficulty.EASY, phrases=())
    assert len(err.value.issues) == 3


def test_timeline_is_sorted_and_within_routine() -> None:
    routine = _routine()
    beats = [t.beat for t in routine.timeline()]
    assert beats == sorted(beats)
    assert len(beats) == len(set(beats))  # never two keyframes on the same beat
    assert 0 <= beats[0] and beats[-1] < routine.total_counts == 64


def test_timeline_musical_positions() -> None:
    timeline = list(_routine().timeline())
    first = timeline[0]
    assert (first.beat, first.phrase_index, first.bar, first.count) == (0, 0, 0, 1)
    # The phrase's final clap lands on bar 4, count 8 of each phrase.
    claps = [t for t in timeline if t.keyframe.clap]
    assert [(t.beat, t.phrase_index, t.bar, t.count) for t in claps] == [
        (31, 0, 3, 8),
        (63, 1, 3, 8),
    ]


def test_timeline_off_beat_counts() -> None:
    # A block with keyframes every half count, in place of wrist_roll_4
    # (bar 3, count 5): off-beat keyframes get fractional counts.
    half = Block(id="half_roll", kind=BlockKind.WRIST_ROLL, duration=4,
                 keyframes=tuple(Keyframe(beat=i / 2, right=POSE) for i in range(8)))  # fmt: skip
    blocks = tuple(half if b.id == "wrist_roll_4" else b for b in _blocks(VALID_IDS))
    routine = Routine(
        id="r", song_id="s", difficulty=Difficulty.NORMAL, phrases=(Phrase(blocks),)
    )
    rolls = [t for t in routine.timeline() if t.block is half]
    assert [t.count for t in rolls] == [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]
    assert {t.bar for t in rolls} == {2}
