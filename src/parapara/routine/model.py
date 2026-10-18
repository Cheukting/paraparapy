"""Routine data model: keyframes → blocks → phrases → routines.

Design (PLAN.md §2–§3)
----------------------
- A **Keyframe** is a target pose at a beat offset inside a block. Each arm's
  target is an :class:`ArmPose` of unit direction vectors in the body frame
  (see :mod:`parapara.routine.geometry`). An arm set to ``None`` is not scored
  at that keyframe, which keeps drafts lenient where the choreography is loose.
- A **Block** is one building-block move (Sweep, Knock, Chop, Wrist Roll,
  Flick + Clap) with a fixed length in counts and its keyframes.
- A **Phrase** is exactly 32 counts (4 bars of 8) of blocks, obeying the
  phrase rules. A **Routine** is a sequence of phrases for one song and
  difficulty.

Every object validates itself on construction, so an invalid routine cannot
exist at runtime. The rules are also exposed as ``*_issues()`` functions that
return a list of human-readable problems instead of raising; the routine
generator (Step 7) uses those to reject candidates cheaply.

Beats here are *musical* positions only (counts from the start of the
routine). Converting beats to seconds is the job of the song's beat map
(Step 5), so routines are independent of tempo.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from parapara.routine.geometry import Vec3, mirror_x, normalize

COUNTS_PER_BAR = 8
BARS_PER_PHRASE = 4
COUNTS_PER_PHRASE = COUNTS_PER_BAR * BARS_PER_PHRASE  # 32


class BlockKind(StrEnum):
    """The five Para Para building blocks (PLAN.md §1.2)."""

    SWEEP = "sweep"
    KNOCK = "knock"
    CHOP = "chop"
    WRIST_ROLL = "wrist_roll"
    FLICK_CLAP = "flick_clap"


# Lengths (in counts) each kind may have; taken from the move descriptions.
ALLOWED_DURATIONS: dict[BlockKind, frozenset[int]] = {
    BlockKind.SWEEP: frozenset({2, 4}),
    BlockKind.KNOCK: frozenset({2}),
    BlockKind.CHOP: frozenset({2}),
    BlockKind.WRIST_ROLL: frozenset({2, 4}),
    BlockKind.FLICK_CLAP: frozenset({4}),
}

# Flowing moves are judged on the whole trajectory (DTW + smoothness) and get
# looser per-keyframe tolerance; sharp moves are judged on the hit positions.
FLOWING_KINDS = frozenset({BlockKind.SWEEP, BlockKind.WRIST_ROLL})
SHARP_TOLERANCE_DEG = 25.0
FLOWING_TOLERANCE_DEG = 35.0


class Difficulty(StrEnum):
    EASY = "easy"
    NORMAL = "normal"
    HARD = "hard"


class HandShape(StrEnum):
    """Hand shape of the choreography (fists for Knocks, open for Chops).

    **Descriptive only, never scored**: hand tracking was removed (too slow and
    unreliable at dancing distance, 2026-10-18 benchmark). It is still used to
    animate the demo avatar correctly. ``ANY`` means unspecified.
    """

    ANY = "any"
    OPEN = "open"
    FIST = "fist"


class RoutineError(ValueError):
    """Raised when a keyframe, block, phrase or routine breaks a rule.

    ``issues`` holds every problem found, not just the first, so authors can
    fix a hand-written routine in one pass.
    """

    def __init__(self, issues: list[str]) -> None:
        self.issues = list(issues)
        super().__init__("; ".join(self.issues))


def _raise_if(issues: list[str]) -> None:
    if issues:
        raise RoutineError(issues)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArmPose:
    """Target configuration of one arm, as unit directions in the body frame.

    Attributes:
        upper: shoulder → elbow direction.
        fore: elbow → wrist direction.
        hand_shape: hand shape for the demo animation (not scored, see
            :class:`HandShape`).
        hand: optional wrist → knuckles direction; only set where the wrist
            angle *is* the move (Flicks). Scored from the pose model's coarse
            index/pinky points at the wrist (no separate hand tracking).

    Directions are normalized on construction, so authors may pass any
    non-zero vector, e.g. ``(1, 1, 0)`` for 45° up-and-out.
    """

    upper: Vec3
    fore: Vec3
    hand_shape: HandShape = HandShape.ANY
    hand: Vec3 | None = None

    def __post_init__(self) -> None:
        # frozen dataclass: normalize via object.__setattr__.
        object.__setattr__(self, "upper", normalize(self.upper))
        object.__setattr__(self, "fore", normalize(self.fore))
        if self.hand is not None:
            object.__setattr__(self, "hand", normalize(self.hand))

    def mirrored(self) -> ArmPose:
        """The same pose reflected through the body's midline (``x → -x``)."""
        return ArmPose(
            upper=mirror_x(self.upper),
            fore=mirror_x(self.fore),
            hand_shape=self.hand_shape,
            hand=None if self.hand is None else mirror_x(self.hand),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Keyframe:
    """A target pose at ``beat`` counts after the start of its block.

    ``beat`` may be fractional (e.g. ``0.5``) for moves on the off-beat such
    as the Wrist Roll. ``left``/``right`` are the dancer's own arms; ``None``
    means "not scored". ``clap`` requires a clap, detected by sound and
    confirmed by the wrists meeting (PLAN.md §5.1). ``tolerance_deg``
    overrides the block's default angular tolerance.
    """

    beat: float
    right: ArmPose | None = None
    left: ArmPose | None = None
    clap: bool = False
    tolerance_deg: float | None = None

    def __post_init__(self) -> None:
        _raise_if(keyframe_issues(self))

    def mirrored(self) -> Keyframe:
        """Swap the arms and reflect both (``x → -x``)."""
        return dataclasses.replace(
            self,
            right=None if self.left is None else self.left.mirrored(),
            left=None if self.right is None else self.right.mirrored(),
        )


def keyframe_issues(kf: Keyframe) -> list[str]:
    issues: list[str] = []
    if not math.isfinite(kf.beat) or kf.beat < 0:
        issues.append(f"keyframe beat must be a finite value >= 0, got {kf.beat}")
    if kf.right is None and kf.left is None and not kf.clap:
        issues.append(f"keyframe at beat {kf.beat} has nothing to score")
    if kf.tolerance_deg is not None and not 0 < kf.tolerance_deg <= 90:
        issues.append(f"tolerance_deg must be in (0, 90], got {kf.tolerance_deg}")
    return issues


@dataclass(frozen=True, slots=True, kw_only=True)
class Block:
    """One building-block move with a fixed length and its keyframes.

    ``id`` is the stable name used in routine files (e.g. ``"chop_right"``).
    """

    id: str
    kind: BlockKind
    duration: int
    keyframes: tuple[Keyframe, ...]
    description: str = ""

    def __post_init__(self) -> None:
        _raise_if(block_issues(self))

    @property
    def is_flowing(self) -> bool:
        return self.kind in FLOWING_KINDS

    @property
    def default_tolerance_deg(self) -> float:
        return FLOWING_TOLERANCE_DEG if self.is_flowing else SHARP_TOLERANCE_DEG

    def tolerance_for(self, kf: Keyframe) -> float:
        """Effective angular tolerance of one of this block's keyframes."""
        return self.default_tolerance_deg if kf.tolerance_deg is None else kf.tolerance_deg

    def mirrored(self, new_id: str, description: str | None = None) -> Block:
        """The left/right mirror image of this block, e.g. Chop-right → Chop-left."""
        return dataclasses.replace(
            self,
            id=new_id,
            keyframes=tuple(kf.mirrored() for kf in self.keyframes),
            description=self.description if description is None else description,
        )


def block_issues(block: Block) -> list[str]:
    issues: list[str] = []
    if not block.id:
        issues.append("block id must not be empty")
    allowed = ALLOWED_DURATIONS[block.kind]
    if block.duration not in allowed:
        issues.append(
            f"{block.id}: {block.kind} must last {sorted(allowed)} counts, got {block.duration}"
        )
    if not block.keyframes:
        issues.append(f"{block.id}: needs at least one keyframe")
    beats = [kf.beat for kf in block.keyframes]
    if any(b >= block.duration for b in beats):
        issues.append(f"{block.id}: keyframe beats must be < duration {block.duration}")
    if any(a >= b for a, b in zip(beats, beats[1:])):
        issues.append(f"{block.id}: keyframe beats must be strictly increasing")
    if block.kind is BlockKind.FLICK_CLAP and not (block.keyframes and block.keyframes[-1].clap):
        issues.append(f"{block.id}: flick + clap must finish with a clap keyframe")
    return issues


@dataclass(frozen=True, slots=True)
class Phrase:
    """Exactly 32 counts of blocks, played in order. See :func:`phrase_issues`."""

    blocks: tuple[Block, ...]

    def __post_init__(self) -> None:
        _raise_if(phrase_issues(self.blocks))

    def placements(self) -> Iterator[tuple[int, Block]]:
        """Yield ``(start_count, block)`` with ``start_count`` 0-based in the phrase."""
        start = 0
        for block in self.blocks:
            yield start, block
            start += block.duration


def phrase_issues(blocks: tuple[Block, ...] | list[Block]) -> list[str]:
    """Check the phrase rules from PLAN.md §2 and return every violation.

    1. contains at least one block of each of the five kinds;
    2. ends with a Flick + Clap;
    3. lasts exactly 32 counts;
    4. no block crosses a bar line, and every block starts on a multiple of
       its own length (4-count blocks on count 1 or 5; 2-count blocks on odd
       counts). This keeps every move locked to the musical grid.
    """
    issues: list[str] = []
    if not blocks:
        return ["phrase has no blocks"]

    missing = set(BlockKind) - {b.kind for b in blocks}
    if missing:
        issues.append(f"phrase is missing block kinds: {sorted(missing)}")

    if blocks[-1].kind is not BlockKind.FLICK_CLAP:
        issues.append(f"phrase must end with flick + clap, ends with {blocks[-1].id}")

    total = sum(b.duration for b in blocks)
    if total != COUNTS_PER_PHRASE:
        issues.append(f"phrase must last {COUNTS_PER_PHRASE} counts, got {total}")

    start = 0
    for block in blocks:
        end = start + block.duration - 1  # last count the block occupies
        position = f"{block.id} at bar {start // COUNTS_PER_BAR + 1} count {start % COUNTS_PER_BAR + 1}"
        if start // COUNTS_PER_BAR != end // COUNTS_PER_BAR:
            issues.append(f"{position} crosses a bar line")
        if start % block.duration != 0:
            issues.append(f"{position} must start on a multiple of {block.duration} counts")
        start += block.duration
    return issues


@dataclass(frozen=True, slots=True)
class TimedKeyframe:
    """A keyframe placed on the routine's timeline (output of ``Routine.timeline``).

    Attributes:
        beat: absolute beat from the start of the routine (0-based).
        phrase_index, bar, count: musical position for display; ``bar`` is
            0-based within the phrase, ``count`` is 1-based within the bar
            (fractional for off-beat keyframes, e.g. 1.5).
        block, keyframe: what to score, and with which tolerance.
    """

    beat: float
    phrase_index: int
    bar: int
    count: float
    block: Block
    keyframe: Keyframe


@dataclass(frozen=True, slots=True, kw_only=True)
class Routine:
    """A full routine: the phrases danced to one song at one difficulty."""

    id: str
    song_id: str
    difficulty: Difficulty
    phrases: tuple[Phrase, ...]

    def __post_init__(self) -> None:
        issues: list[str] = []
        if not self.id:
            issues.append("routine id must not be empty")
        if not self.song_id:
            issues.append("routine song_id must not be empty")
        if not self.phrases:
            issues.append("routine needs at least one phrase")
        _raise_if(issues)

    @property
    def total_counts(self) -> int:
        return len(self.phrases) * COUNTS_PER_PHRASE

    def timeline(self) -> Iterator[TimedKeyframe]:
        """Every keyframe of the routine in time order, with absolute beats.

        This is the single schedule consumed by the scorer, the on-screen move
        cues and the guide figure.
        """
        for p_index, phrase in enumerate(self.phrases):
            phrase_start = p_index * COUNTS_PER_PHRASE
            for block_start, block in phrase.placements():
                for kf in block.keyframes:
                    in_phrase = block_start + kf.beat
                    yield TimedKeyframe(
                        beat=phrase_start + in_phrase,
                        phrase_index=p_index,
                        bar=int(in_phrase // COUNTS_PER_BAR),
                        count=in_phrase % COUNTS_PER_BAR + 1,
                        block=block,
                        keyframe=kf,
                    )
