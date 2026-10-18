"""Draft definitions of the five building blocks (PLAN.md §1.2).

⚠ DRAFTS. Poses are written from textual descriptions, not motion capture,
and will be refined from recordings of a real dancer. They are deliberately
lenient: an arm that is not part of the move is left unscored (``None``).

Reference
---------
Checked against EKS-D "How to Dance" (https://kirakira.org/XD/learn/how-to-dance/),
retrieved 2026-10-18. Quotes below come from that page. It describes sweeps,
knocks, chops and flick+clap, but **not the wrist roll**, which therefore
remains an unverified draft.

Conventions
-----------
- Body frame: ``+x`` dancer's right, ``+y`` up, ``+z`` forward (see
  :mod:`parapara.routine.geometry`). Directions need not be normalized;
  :class:`ArmPose` normalizes them.
- One-sided moves are authored for the **right** arm, because moves "with rare
  exception, begin on the right-hand side". Left variants are generated with
  :meth:`Block.mirrored` so both sides stay exactly symmetric.
- Pairing ("right on 1–2, left on 3–4") is choreography, not part of a block:
  routines (and the Step 7 generator) arrange single-sided blocks into pairs.

Interpretation choices (reviewed with the project owner 2026-10-18):

- **Knock**: the upper arm points out to the side and slightly forward (the
  page only says "horizontal from the shoulder"). Confirmed.
- **Chop** beat 2: the forearm folds across **in front of the chest** with the
  flat hand continuing the forearm line (not pointing down), leading into the
  salute shape.
- **Flick + Clap**: the right arm travels left → right; the mirrored block
  travels right → left. **The clap is scored only as a clap** (heard by the
  microphone, confirmed by the wrists meeting), wherever it happens -- arm
  positions and side are ignored. Hand direction
  during the backhanded flick is approximated as up and towards the direction
  of travel.
- **Hand shapes** (fist for Knocks, open for Chops) are recorded for the demo
  animation only; they are not scored because hands are not tracked.
- **Sweep**: both arms parallel; the upper arm follows the forearm (see
  :func:`_sweep_upper`). One-armed sweeps are not drafted yet.
"""

from __future__ import annotations

from collections.abc import Iterable

from parapara.routine.geometry import Vec3, normalize, slerp
from parapara.routine.model import ArmPose, Block, BlockKind, HandShape, Keyframe

# ---------------------------------------------------------------------------
# Sweep (flowing, 2 or 4 counts)
#   "Arms move in a sweeping motion from one point to another", horizontal,
#   vertical or diagonal; one arm or both.
# ---------------------------------------------------------------------------

# Forearm end points of the sweeps, at roughly arm's length in front.
_HIGH, _LOW = (0, 1, 0.4), (0, -1, 0.4)
_SIDE_LEFT, _SIDE_RIGHT = (-0.87, 0, 0.5), (0.87, 0, 0.5)  # 60° either side
_UPPER_LEFT, _UPPER_RIGHT = (-0.7, 0.7, 0.4), (0.7, 0.7, 0.4)
_LOWER_LEFT, _LOWER_RIGHT = (-0.7, -0.7, 0.4), (0.7, -0.7, 0.4)


def _sweep_upper(fore: Vec3) -> Vec3:
    """Plausible upper-arm direction for a forearm direction during a sweep.

    The upper arm follows the forearm but stays lower and further forward
    (elbows soft and in front of the body), which keeps it anatomically
    reachable for both arms even when they point across the body.
    """
    f = normalize(fore)
    return normalize((0.7 * f[0], 0.7 * f[1] - 0.3, 0.7 * f[2] + 0.3))


def _sweep(block_id: str, duration: int, start: Vec3, end: Vec3, label: str) -> Block:
    """Both arms parallel, turning from ``start`` to ``end``, one keyframe per
    count and arriving on the block's last count."""
    keyframes = []
    for i in range(duration):
        fore = slerp(start, end, i / (duration - 1))
        pose = ArmPose(upper=_sweep_upper(fore), fore=fore)
        keyframes.append(Keyframe(beat=float(i), right=pose, left=pose))
    return Block(
        id=block_id,
        kind=BlockKind.SWEEP,
        duration=duration,
        keyframes=tuple(keyframes),
        description=f"Both arms sweep parallel {label} over {duration} counts; soft wrists.",
    )


# ---------------------------------------------------------------------------
# Knock (sharp, 2 counts)
#   "Like banging on an imaginary door": "a closed fist with your thumb facing
#   inward toward your body"; upper arm horizontal from the shoulder, forearm
#   up from the elbow (a right angle). "Usually in pairs (ie: on beats 1 & 2)
#   and mirrored": right hand on 1–2, left hand on 3–4.
# ---------------------------------------------------------------------------

_KNOCK = ArmPose(upper=(1, 0, 0.5), fore=(0, 1, 0.1), hand_shape=HandShape.FIST)

KNOCK_RIGHT = Block(
    id="knock_right",
    kind=BlockKind.KNOCK,
    duration=2,
    keyframes=(
        Keyframe(beat=0, right=_KNOCK),  # knock 1
        Keyframe(beat=1, right=_KNOCK),  # knock 2, same hand, same spot
    ),
    description="Right fist knocks twice (counts 1-2): upper arm horizontal, forearm "
    "up, thumb towards the body.",
)

# ---------------------------------------------------------------------------
# Chop (sharp, 2 counts)
#   Beat 1: "upward at a 45° angle out to the side of your body".
#   Beat 2: fold the forearm in toward the chest "with your fingertips resting
#   at your collarbone" -- like a salute. Often right on 1–2 and 3–4, then
#   left on 5–6 and 7–8.
# ---------------------------------------------------------------------------

_CHOP_OUT = ArmPose(upper=(1, 1, 0), fore=(1, 1, 0), hand_shape=HandShape.OPEN)
# Salute fold: elbow out and in front, forearm roughly horizontal across the
# front of the chest, flat hand continuing the forearm line (not drooping).
_CHOP_IN = ArmPose(
    upper=(0.5, -0.6, 0.6),
    fore=(-1, 0.1, 0.2),
    hand_shape=HandShape.OPEN,
    hand=(-1, 0.1, 0.2),
)

CHOP_RIGHT = Block(
    id="chop_right",
    kind=BlockKind.CHOP,
    duration=2,
    keyframes=(
        Keyframe(beat=0, right=_CHOP_OUT),
        Keyframe(beat=1, right=_CHOP_IN),
    ),
    description="Right arm reaches 45° up and out (count 1), then folds in front of "
    "the chest, flat hand level, like a salute (count 2).",
)

# ---------------------------------------------------------------------------
# Wrist Roll (flowing, 2 or 4 counts) -- NOT in the reference page; unverified.
#   Hands close to the chest at shoulder height, wrists rolling over each
#   other: forearms horizontal pointing inwards, hands alternate over/under
#   every half count.
# ---------------------------------------------------------------------------


def _wrist_roll_pose(right_over: bool) -> dict[str, ArmPose]:
    tilt = 0.35  # forearm tilt up (over) or down (under)
    over = ArmPose(upper=(0.4, -0.7, 0.5), fore=(-1, tilt, 0.3))
    under = ArmPose(upper=(0.4, -0.7, 0.5), fore=(-1, -tilt, 0.3))
    right = over if right_over else under
    left = (under if right_over else over).mirrored()
    return {"right": right, "left": left}


def _wrist_roll(block_id: str, duration: int) -> Block:
    keyframes = tuple(
        Keyframe(beat=i * 0.5, **_wrist_roll_pose(right_over=i % 2 == 0))
        for i in range(duration * 2)  # every half count
    )
    return Block(
        id=block_id,
        kind=BlockKind.WRIST_ROLL,
        duration=duration,
        keyframes=keyframes,
        description=f"Hands at the chest, wrists rolling over each other for "
        f"{duration} counts. (Unverified draft.)",
    )


# ---------------------------------------------------------------------------
# Flick + Clap (phrase finisher, 4 counts) -- "nami + nami + nami + pon!"
#   Beat 1: "cross the arm over the body", hand just above shoulder height,
#   backhanded flick by rotating the wrist. Beat 2: same flick directly in
#   front. Beat 3: same flick on the side opposite beat 1. Beat 4: clap "at
#   the spot where the third flick ended". Often done in pairs going opposite
#   directions. It is effectively a one-armed sweep accented by flicks.
# ---------------------------------------------------------------------------

_TRAVEL_RIGHT = (0.7, 0.7, 0)  # backhanded flick: hand up and towards travel

_FLICK_CROSSED = ArmPose(upper=(-0.4, -0.1, 1), fore=(-1, 0.4, 0.4), hand=_TRAVEL_RIGHT)
_FLICK_FRONT = ArmPose(upper=(0.3, -0.2, 1), fore=(-0.2, 0.7, 0.7), hand=_TRAVEL_RIGHT)
_FLICK_SIDE = ArmPose(upper=(1, -0.2, 0.4), fore=(0.2, 1, 0.3), hand=_TRAVEL_RIGHT)

FLICK_CLAP_TO_RIGHT = Block(
    id="flick_clap_to_right",
    kind=BlockKind.FLICK_CLAP,
    duration=4,
    keyframes=(
        Keyframe(beat=0, right=_FLICK_CROSSED),
        Keyframe(beat=1, right=_FLICK_FRONT),
        Keyframe(beat=2, right=_FLICK_SIDE),
        # Any clap counts: arms are unscored, so the side does not matter.
        Keyframe(beat=3, clap=True),
    ),
    description="Right hand flicks crossed over the body, in front, then on the "
    "right (counts 1-3); clap (count 4).",
)

# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------


def _index(blocks: Iterable[Block]) -> dict[str, Block]:
    library: dict[str, Block] = {}
    for block in blocks:
        if block.id in library:
            raise ValueError(f"duplicate block id {block.id!r}")
        library[block.id] = block
    return library


BLOCK_LIBRARY: dict[str, Block] = _index(
    [
        # Horizontal sweeps
        _sweep("sweep_to_right_4", 4, _SIDE_LEFT, _SIDE_RIGHT, "from left to right"),
        _sweep("sweep_to_left_4", 4, _SIDE_RIGHT, _SIDE_LEFT, "from right to left"),
        _sweep("sweep_to_right_2", 2, _SIDE_LEFT, _SIDE_RIGHT, "from left to right"),
        _sweep("sweep_to_left_2", 2, _SIDE_RIGHT, _SIDE_LEFT, "from right to left"),
        # Vertical sweeps
        _sweep("sweep_down_4", 4, _HIGH, _LOW, "from high to low"),
        _sweep("sweep_up_4", 4, _LOW, _HIGH, "from low to high"),
        # Diagonal sweeps across the body
        _sweep("sweep_diagonal_down_to_right_4", 4, _UPPER_LEFT, _LOWER_RIGHT,
               "from upper left to lower right"),
        _sweep("sweep_diagonal_down_to_left_4", 4, _UPPER_RIGHT, _LOWER_LEFT,
               "from upper right to lower left"),
        _sweep("sweep_diagonal_up_to_right_4", 4, _LOWER_LEFT, _UPPER_RIGHT,
               "from lower left to upper right"),
        _sweep("sweep_diagonal_up_to_left_4", 4, _LOWER_RIGHT, _UPPER_LEFT,
               "from lower right to upper left"),
        # Knocks
        KNOCK_RIGHT,
        KNOCK_RIGHT.mirrored(
            "knock_left",
            "Left fist knocks twice (counts 1-2): upper arm horizontal, forearm "
            "up, thumb towards the body.",
        ),
        # Chops
        CHOP_RIGHT,
        CHOP_RIGHT.mirrored(
            "chop_left",
            "Left arm reaches 45° up and out (count 1), then folds in front of "
            "the chest, flat hand level, like a salute (count 2).",
        ),
        # Wrist rolls
        _wrist_roll("wrist_roll_4", 4),
        _wrist_roll("wrist_roll_2", 2),
        # Flick + Clap
        FLICK_CLAP_TO_RIGHT,
        FLICK_CLAP_TO_RIGHT.mirrored(
            "flick_clap_to_left",
            "Left hand flicks crossed over the body, in front, then on the "
            "left (counts 1-3); clap (count 4).",
        ),
    ]
)  # fmt: skip
"""All built-in blocks, keyed by the id used in routine files."""
