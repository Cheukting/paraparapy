"""Draft definitions of the five building blocks (PLAN.md §1.2).

⚠ DRAFTS. Poses were first written from textual descriptions, then
**calibrated against a recording of a real dancer** (session
``recordings/20261018-144829``, analysed with ``uv run parapara-analyze``).
To recalibrate, record new clips (``uv run parapara-record``), run the
analysis, and move each target towards the reported "seen" direction. The
drafts are deliberately lenient: an arm that is not part of the move is left
unscored (``None``).

Calibration notes (2026-10-18 recording):

- Depth (``z``) is MediaPipe's least reliable axis and reads biased towards
  the camera, so the targets below follow the data mostly in ``x``/``y`` and
  only partly in ``z``.
- One take by one dancer; treat values as a starting point.

Reference
---------
Checked against EKS-D "How to Dance" (https://kirakira.org/XD/learn/how-to-dance/),
retrieved 2026-10-18. Quotes below come from that page. It describes sweeps,
knocks, chops and flick+clap, but **not the wrist roll**, which was instead
checked against recordings of a real dancer (2026-10-18, 2026-10-23).

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

- **Knock**: upper arm out to the side and forward (~45°, from the recording);
  the forearm points up and tilts forward towards the "door".
- **Chop** beat 1 keeps the reference's 45° up-and-out target, with a wider
  tolerance (the dancer reached 15-25°; a 25° target waits for the held-pose
  timing fix, see the note at ``_CHOP_OUT``). Beat 2: the elbow stays at shoulder
  height, out and forward, and the forearm folds across **in front of the
  chest** with the flat hand continuing the forearm line (salute shape).
- **Flick + Clap**: the right arm travels left → right; the mirrored block
  travels right → left. Forearm targets of the front and side flicks
  recalibrated from two 2026-10-23 recordings (forward and slightly up). **The clap is scored only as a clap** (heard by the
  microphone, confirmed by the wrists meeting), wherever it happens -- arm
  positions and side are ignored. During the flicks the forearm stays
  roughly **level across the chest** and the elbow carries it from side to
  side (from the recording); the backhanded flick itself is a wrist rotation,
  approximated as the hand pointing up and towards the direction of travel.
- **Hand shapes** (fist for Knocks, open for Chops) are recorded for the demo
  animation only; they are not scored because hands are not tracked.
- **Sweep**: forearms parallel, ±45° arc. The arm on the side the forearms
  point to extends that way; the arm crossing the body keeps its upper arm
  pointing forward (see :func:`_sweep_upper`). One-armed sweeps are not
  drafted yet.
- **Wrist Roll**: hands in front of the chest, elbows forward. The rolling
  itself is a wrist rotation that the body landmarks cannot see, so keyframes
  only hold the position (one per count); Step 4 will add a "keep moving"
  check.
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
_SIDE_LEFT, _SIDE_RIGHT = (-0.71, 0, 0.71), (0.71, 0, 0.71)  # 45° either side
_UPPER_LEFT, _UPPER_RIGHT = (-0.7, 0.7, 0.4), (0.7, 0.7, 0.4)
_LOWER_LEFT, _LOWER_RIGHT = (-0.7, -0.7, 0.4), (0.7, -0.7, 0.4)


def _sweep_upper(fore: Vec3, side: float) -> Vec3:
    """Upper-arm direction for one arm during a sweep.

    Args:
        fore: the (shared, parallel) forearm direction.
        side: ``+1`` for the right arm, ``-1`` for the left.

    Fitted to the recording: when the forearms point to the arm's own side
    the upper arm extends that way too; the arm crossing the body keeps its
    upper arm pointing forward; in vertical sweeps both elbows are slightly
    out. The formula is odd in ``x`` under ``side -> -side``, so mirrored
    sweeps stay exact mirror images.
    """
    f = normalize(fore)
    own_side = max(0.0, f[0] * side)  # how far the forearm points to this arm's side
    x = side * (0.3 * (1 - abs(f[0])) + 0.8 * own_side)
    return normalize((x, 0.7 * f[1] - 0.15, 0.8 - 0.5 * own_side))


def _sweep(block_id: str, duration: int, start: Vec3, end: Vec3, label: str) -> Block:
    """Both arms parallel, turning from ``start`` to ``end``, one keyframe per
    count and arriving on the block's last count."""
    keyframes = []
    for i in range(duration):
        fore = slerp(start, end, i / (duration - 1))
        keyframes.append(Keyframe(
            beat=float(i),
            right=ArmPose(upper=_sweep_upper(fore, +1), fore=fore),
            left=ArmPose(upper=_sweep_upper(fore, -1), fore=fore),
        ))  # fmt: skip
    return Block(
        id=block_id,
        kind=BlockKind.SWEEP,
        duration=duration,
        keyframes=tuple(keyframes),
        description=f"Both forearms sweep parallel {label} over {duration} counts; soft wrists.",
    )


# ---------------------------------------------------------------------------
# Knock (sharp, 2 counts)
#   "Like banging on an imaginary door": "a closed fist with your thumb facing
#   inward toward your body"; upper arm horizontal from the shoulder, forearm
#   up from the elbow (a right angle). "Usually in pairs (ie: on beats 1 & 2)
#   and mirrored": right hand on 1–2, left hand on 3–4.
# ---------------------------------------------------------------------------

# From the recording: upper arm out and ~45° forward, forearm up and tilted
# forward towards the "door".
_KNOCK = ArmPose(upper=(0.7, 0.1, 0.7), fore=(-0.15, 0.75, 0.65), hand_shape=HandShape.FIST)

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
# The 45° target is kept, but dancers often reach lower (15-25° in every
# recording so far), so this keyframe is judged more leniently than other
# sharp ones. A 25° target fits the recordings better (error 18° -> 9°) but
# was put on hold on 2026-10-23: the judge then times the reach ~50 ms later
# (a held pose's arrival is detected late through the smoothing filter), which
# cost ~5 points. Re-apply it after the held-pose timing fix (PLAN.md §10).
CHOP_OUT_TOLERANCE_DEG = 40.0
# Salute fold (from the recording): elbow at shoulder height, out and forward;
# forearm across the front of the chest, flat hand continuing its line.
_CHOP_IN = ArmPose(
    upper=(0.7, 0.05, 0.7),
    fore=(-0.65, 0.25, 0.7),
    hand_shape=HandShape.OPEN,
    hand=(-0.65, 0.25, 0.7),
)

CHOP_RIGHT = Block(
    id="chop_right",
    kind=BlockKind.CHOP,
    duration=2,
    keyframes=(
        Keyframe(beat=0, right=_CHOP_OUT, tolerance_deg=CHOP_OUT_TOLERANCE_DEG),
        Keyframe(beat=1, right=_CHOP_IN),
    ),
    description="Right arm reaches 45° up and out (count 1), then folds in front of "
    "the chest, flat hand level, like a salute (count 2).",
)

# ---------------------------------------------------------------------------
# Wrist Roll (flowing, 2 or 4 counts) -- NOT in the reference page, but
#   checked against a real dancer: 13.8°, 14.6° and 13.6° mean error in the
#   2026-10-18 and two 2026-10-23 recordings (98-100% accuracy).
#   Hands close to the chest at shoulder height, wrists rolling over each
#   other. From the recording: elbows forward, forearms level and pointing
#   inwards. The over/under rolling is invisible to body landmarks, so the
#   keyframes hold this position once per count.
# ---------------------------------------------------------------------------

_WRIST_ROLL = ArmPose(upper=(0.43, -0.32, 0.85), fore=(-0.8, 0.08, 0.6))


def _wrist_roll(block_id: str, duration: int) -> Block:
    keyframes = tuple(
        Keyframe(beat=float(i), right=_WRIST_ROLL, left=_WRIST_ROLL.mirrored())
        for i in range(duration)
    )
    return Block(
        id=block_id,
        kind=BlockKind.WRIST_ROLL,
        duration=duration,
        keyframes=keyframes,
        description=f"Hands at the chest, wrists rolling over each other for "
        f"{duration} counts.",
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

# From the recordings: the elbow carries the hand from the left (crossed) to
# the right side. On the crossed flick the forearm lies across the chest; on
# the front and side flicks it turns to point forward and slightly up (two
# 2026-10-23 sessions at 150 BPM agreed within ~10°; the 2026-10-18 draft had
# it level across the chest, which scored 26-39%). Targets follow the
# recordings in x/y and only partly in z (depth reads biased towards the
# camera).
_FLICK_CROSSED = ArmPose(upper=(0.0, -0.55, 0.85), fore=(-0.6, 0.25, 0.75), hand=_TRAVEL_RIGHT)
_FLICK_FRONT = ArmPose(upper=(0.23, -0.45, 0.86), fore=(-0.49, 0.2, 0.81), hand=_TRAVEL_RIGHT)
_FLICK_SIDE = ArmPose(upper=(0.7, -0.55, 0.45), fore=(-0.22, 0.25, 0.85), hand=_TRAVEL_RIGHT)

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
