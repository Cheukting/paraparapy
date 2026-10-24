"""Routine + beat position → the demo dancer's full pose.

The dancer dances **exactly what is scored**: on every keyframe's beat each
scored arm is precisely at the keyframe's target (the same targets the judge
compares the player with). In between, motion styles make it look like
dancing rather than jumping between poses:

- **snap** (knocks, chops, flick positions, transitions between blocks):
  hold the current pose, then move in the last :data:`SNAP_WINDOW_BEATS` of
  the beat before the next target with a smoothstep (zero speed at both
  ends), landing exactly on the beat;
- **glide** (inside a sweep): constant-speed spherical interpolation from
  one target to the next, passing through each on its beat;
- **roll** (wrist roll): the held position plus the hands circling each
  other in opposite phase, one circle per beat; the offset is zero on whole
  beats, so the scored pose is exact on the beat;
- **knock**: the fist pulls back between beats and hits the target on each
  beat (``sin²`` pulse, zero on whole beats: exact when scored);
- **flick**: the hand is cocked back and whips out to its target direction
  in the last :data:`FLICK_WINDOW_BEATS` before the beat;
- **clap**: both hands come together on the beat (a display pose: the clap
  keyframe has no arm targets because any clap counts), with a short flash;
- **rest**: an arm the keyframe does not use relaxes down.

Body: a side-to-side two-step (step right on count 1, close on 2, step left
on 3, close on 4; the Para Para "engine", shown as a guide and never scored),
a small knee bounce dipping on every beat, and grooving in place before the
routine starts. The pose also says which block is being danced and which is
next (for the "next move" card, previewed :data:`PREVIEW_BEATS` ahead).

Beats are **routine beats**: 0 is the routine's first count (``song.routine_start_beat``
in song beats); negative values are the intro.

Pure NumPy (no pygame), so it is fully testable and renderer-independent.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from parapara.routine.geometry import Vec3, angle_between_deg, normalize, slerp
from parapara.routine.model import (
    COUNTS_PER_BAR,
    COUNTS_PER_PHRASE,
    ArmPose,
    Block,
    BlockKind,
    Routine,
)

SNAP_WINDOW_BEATS = 0.4
FLICK_WINDOW_BEATS = 0.22
CLAP_FLASH_BEATS = 0.3
ROLL_RADIUS = 0.35  # size of the hand circles during a wrist roll
KNOCK_PULL = 0.6  # how far the fist pulls back between knocks
PREVIEW_BEATS = COUNTS_PER_BAR  # show the next block one bar ahead
BOUNCE = 0.06  # knee dip, in shoulder widths
STEP_WIDTH = 0.35  # foot travel of the two-step, in shoulder widths

REST = ArmPose(
    upper=(0.12, -1, 0.05), fore=(0.05, -1, 0.15)
)  # right arm hanging relaxed
CLAP = ArmPose(
    upper=(0.35, -0.75, 0.6), fore=(-0.75, 0.55, 0.4)
)  # right arm; hands meet in front

_FORWARD: Vec3 = (0.0, 0.0, 1.0)


def smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def turn(a: Vec3, b: Vec3, t: float) -> Vec3:
    """Direction ``t`` of the way from ``a`` to ``b``; nearly opposite
    directions turn through the front of the body (never through the torso)."""
    if t <= 0:
        return normalize(a)
    if t >= 1:
        return normalize(b)
    if angle_between_deg(a, b) > 170:
        return slerp(a, _FORWARD, 2 * t) if t < 0.5 else slerp(_FORWARD, b, 2 * t - 1)
    return slerp(a, b, t)


@dataclass(frozen=True, slots=True)
class ArmState:
    upper: Vec3
    fore: Vec3
    hand: Vec3


@dataclass(frozen=True, slots=True)
class DancerPose:
    """Everything a renderer needs for one frame (dancer's anatomical frame:
    +x = dancer's right, +y up, +z forward)."""

    beat: float
    right: ArmState
    left: ArmState
    clap_flash: float  # 0..1, peaks on a clap
    body_x: float  # sideways sway from the two-step, shoulder widths
    bounce: float  # knee dip (>= 0), shoulder widths
    right_foot: tuple[float, float]  # (x, lift) in shoulder widths
    left_foot: tuple[float, float]
    count: int  # 1-8 within the bar (0 before the routine)
    bar: int  # 1-based within the phrase (0 before the routine)
    phrase: int  # 1-based (0 before the routine)
    block: Block | None  # being danced now
    next_block: Block | None  # coming up (within PREVIEW_BEATS)
    next_block_beat: float | None


@dataclass(frozen=True, slots=True)
class _Target:
    beat: float
    pose: ArmPose
    block: Block
    flick: bool = False


@dataclass(frozen=True, slots=True)
class _Placed:
    start: float
    end: float
    block: Block


def _side_pose(arm: ArmPose | None, clap: bool, side: str) -> ArmPose:
    if arm is not None:
        return arm
    base = CLAP if clap else REST
    return base if side == "right" else base.mirrored()


class Animator:
    """Pose of the demo dancer for a routine at any routine beat."""

    def __init__(self, routine: Routine) -> None:
        self.routine: Routine | None = routine
        placements = [(p_index * COUNTS_PER_PHRASE + start, block)
                      for p_index, phrase in enumerate(routine.phrases)
                      for start, block in phrase.placements()]  # fmt: skip
        self._place(placements, float(routine.total_counts))

    @classmethod
    def for_blocks(cls, blocks: Sequence[Block]) -> Animator:
        """Animator for ``blocks`` danced back to back from beat 0, without the
        phrase rules of a routine (32 counts, every kind, ...).

        Used to show single moves: the "Learn the moves" screen loops one
        block (``[block] * n``) at a slow tempo.
        """
        self = cls.__new__(cls)
        self.routine = None
        starts = np.cumsum([0] + [b.duration for b in blocks])
        self._place(
            list(zip(starts[:-1].tolist(), blocks, strict=True)), float(starts[-1])
        )
        return self

    def _place(self, placements: list[tuple[int, Block]], end_beat: float) -> None:
        """Index ``(start_beat, block)`` placements into per-arm targets."""
        self._blocks: list[_Placed] = []
        self._targets: dict[str, list[_Target]] = {"right": [], "left": []}
        self._claps: list[float] = []
        for start, block in placements:
            self._blocks.append(_Placed(start, start + block.duration, block))
            for kf in block.keyframes:
                beat = start + kf.beat
                if kf.clap:
                    self._claps.append(beat)
                for side in ("right", "left"):
                    arm = getattr(kf, side)
                    flick = (
                        block.kind is BlockKind.FLICK_CLAP
                        and arm is not None
                        and arm.hand is not None
                    )
                    self._targets[side].append(
                        _Target(beat, _side_pose(arm, kf.clap, side), block, flick)
                    )
        self._beats = {s: [t.beat for t in ts] for s, ts in self._targets.items()}
        self._block_starts = [b.start for b in self._blocks]
        self.end_beat = end_beat

    # -- arms --------------------------------------------------------------

    def _arm(self, side: str, beat: float) -> ArmState:
        targets, beats = self._targets[side], self._beats[side]
        rest = REST if side == "right" else REST.mirrored()
        k = bisect.bisect_right(beats, beat) - 1  # last target at or before `beat`
        prev = targets[k].pose if k >= 0 else rest
        if k + 1 >= len(targets):  # after the last target: relax after one beat
            u = smoothstep((beat - (beats[-1] + 1)) / 1.0) if beats else 1.0
            return self._blend(prev, rest, u)
        nxt = targets[k + 1]
        gap = nxt.beat - (beats[k] if k >= 0 else nxt.beat - 1.0)
        same_sweep = (
            k >= 0
            and targets[k].block is nxt.block
            and nxt.block.kind is BlockKind.SWEEP
        )
        if same_sweep:  # glide through the sweep at constant speed
            u = (beat - beats[k]) / gap
        else:  # snap onto the beat
            window = min(SNAP_WINDOW_BEATS, gap)
            u = smoothstep((beat - (nxt.beat - window)) / window)
        state = self._blend(prev, nxt.pose, u)
        if nxt.flick:
            # After the previous hit the hand cocks back gradually, then whips
            # out to the target in the last FLICK_WINDOW_BEATS: exact on both
            # beats, no jumps in between.
            hand_target = nxt.pose.hand
            if hand_target is None:
                hand_target = state.hand
            vx = 2.0 * state.fore[0] - hand_target[0]
            vy = 2.0 * state.fore[1] - hand_target[1]
            vz = 2.0 * state.fore[2] - hand_target[2]
            cocked = normalize((vx, vy, vz))
            whip_start = nxt.beat - min(FLICK_WINDOW_BEATS, gap)
            start = beats[k] if k >= 0 else nxt.beat - 1.0
            if beat < whip_start:
                c = smoothstep((beat - start) / max(whip_start - start, 1e-9))
                hand = turn(state.hand, cocked, c)
            else:
                hand = turn(
                    cocked,
                    hand_target,
                    smoothstep((beat - whip_start) / (nxt.beat - whip_start)),
                )
            state = ArmState(state.upper, state.fore, hand)
        return self._knock(side, beat, self._roll(side, beat, state))

    def _knock(self, side: str, beat: float, state: ArmState) -> ArmState:
        block = self.block_at(beat)
        if block is None or block.block.kind is not BlockKind.KNOCK:
            return state
        if all(getattr(kf, side) is None for kf in block.block.keyframes):
            return state  # the other arm is not knocking
        pull = KNOCK_PULL * math.sin(math.pi * beat) ** 2  # 0 on the beat, max half-way

        def back(v: Vec3) -> Vec3:
            return normalize((v[0], v[1], v[2] - pull))

        # Offset (not replace) the hand too, so leaving the knock has no jump.
        return ArmState(state.upper, back(state.fore), back(state.hand))

    @staticmethod
    def _blend(a: ArmPose, b: ArmPose, u: float) -> ArmState:
        hand_a = a.hand if a.hand is not None else a.fore
        hand_b = b.hand if b.hand is not None else b.fore
        fore = turn(a.fore, b.fore, u)
        hand = (
            turn(hand_a, hand_b, u)
            if (a.hand is not None or b.hand is not None)
            else fore
        )
        return ArmState(turn(a.upper, b.upper, u), fore, hand)

    def _roll(self, side: str, beat: float, state: ArmState) -> ArmState:
        block = self.block_at(beat)
        if block is None or block.block.kind is not BlockKind.WRIST_ROLL:
            return state
        phase = 2 * math.pi * beat
        sign = (
            1.0 if side == "right" else -1.0
        )  # opposite phase: one hand over, one under
        # Zero on whole beats (sin 0 = 0, 1 - cos 0 = 0): the scored pose stays exact.
        offset = (
            np.array([0.0, sign * math.sin(phase), 1 - math.cos(phase)]) * ROLL_RADIUS
        )

        def roll(v: Vec3) -> Vec3:
            return normalize(tuple(np.asarray(v) + offset))

        # Offset (not replace) the hand too, so leaving the roll has no jump.
        return ArmState(state.upper, roll(state.fore), roll(state.hand))

    # -- body & context ----------------------------------------------------

    @staticmethod
    def two_step(beat: float) -> tuple[float, tuple[float, float], tuple[float, float]]:
        """Body sway and feet ``(x, lift)`` for the two-step at ``beat``.

        Feet positions (shoulder widths, dancer's right = +x) after landing on
        each count of the 4-count cycle: step right (1), close the left foot
        (2), step left (3), close the right foot (4). A foot moves in the
        last :data:`SNAP_WINDOW_BEATS` before its beat, lifting a little, and
        the body stays centred between the feet.
        """
        whole = math.floor(beat)
        u = smoothstep((beat - whole - (1 - SNAP_WINDOW_BEATS)) / SNAP_WINDOW_BEATS)

        def foot(xs: tuple[float, float, float, float]) -> tuple[float, float]:
            here, there = xs[whole % 4], xs[(whole + 1) % 4]
            return here + (there - here) * u, (
                0.1 * math.sin(math.pi * u) if there != here else 0.0
            )

        w = STEP_WIDTH
        right = foot((0.1 + w, 0.1 + w, 0.1 + w, 0.1))
        left = foot((-0.1, -0.1 + w, -0.1, -0.1))
        # The steps travel right then back; centre the sway around the body.
        right, left = (right[0] - w / 2, right[1]), (left[0] - w / 2, left[1])
        return (right[0] + left[0]) / 2, right, left

    def block_at(self, beat: float) -> _Placed | None:
        i = bisect.bisect_right(self._block_starts, beat) - 1
        if i >= 0 and beat < self._blocks[i].end:
            return self._blocks[i]
        return None

    def next_block(self, beat: float) -> _Placed | None:
        i = bisect.bisect_right(self._block_starts, beat)
        if i < len(self._blocks) and self._blocks[i].start - beat <= PREVIEW_BEATS:
            return self._blocks[i]
        return None

    def pose_at(self, beat: float) -> DancerPose:
        body_x, rf, lf = self.two_step(beat)
        bounce = BOUNCE * (0.5 + 0.5 * math.cos(2 * math.pi * beat))
        flash = max((1 - abs(beat - c) / CLAP_FLASH_BEATS for c in self._claps
                     if abs(beat - c) < CLAP_FLASH_BEATS), default=0.0)  # fmt: skip
        current, upcoming = self.block_at(beat), self.next_block(beat)
        in_routine = 0 <= beat < self.end_beat
        whole = math.floor(beat)
        return DancerPose(
            beat=beat,
            right=self._arm("right", beat),
            left=self._arm("left", beat),
            clap_flash=float(flash),
            body_x=body_x,
            bounce=bounce,
            right_foot=rf,
            left_foot=lf,
            count=(whole % COUNTS_PER_BAR) + 1 if in_routine else 0,
            bar=(whole % COUNTS_PER_PHRASE) // COUNTS_PER_BAR + 1 if in_routine else 0,
            phrase=whole // COUNTS_PER_PHRASE + 1 if in_routine else 0,
            block=current.block if current else None,
            next_block=upcoming.block if upcoming else None,
            next_block_beat=upcoming.start if upcoming else None,
        )


def block_title(block: Block) -> str:
    """'chop_right' -> 'Chop (right)'; 'sweep_to_left_4' -> 'Sweep to left (4)'."""
    words = block.id.split("_")
    count = words.pop() if words[-1].isdigit() else None
    side = words.pop() if words[-1] in ("right", "left") and len(words) == 2 else None
    title = " ".join(words).capitalize()
    return title + (f" ({side})" if side else "") + (f" ({count})" if count else "")
