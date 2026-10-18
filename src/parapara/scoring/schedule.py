"""When each target pose is due: the scoring schedule.

The judge does not care where keyframes come from; it works on a list of
:class:`ScheduledKeyframe`, each with an absolute expected time. Two
builders:

- :func:`schedule_routine`: a full routine on a song, via a
  :class:`BeatClock` (tempo + time of the first beat + latency offset);
- :func:`schedule_block_clip`: one block repeated over the beats of a
  recorded clip (``beat_times_ns`` in the recording metadata), used to score
  recordings and in tests.

**Hold keyframes**: when a keyframe asks for the same pose as the previous
one of the same block instance (e.g. the second knock, or the wrist roll's
held position), its timing cannot be measured (the recordings showed the
lag fit is flat), so it is marked ``hold`` and judged on the pose alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parapara.routine.geometry import angle_between_deg
from parapara.routine.model import Block, Keyframe, Routine

HOLD_MAX_CHANGE_DEG = 10.0  # targets this similar count as "the same pose"


@dataclass(frozen=True, slots=True)
class BeatClock:
    """Maps musical beats to ``time.monotonic_ns()`` values.

    Attributes:
        bpm: tempo.
        first_beat_ns: time of beat 0 (the routine's first count).
        latency_ns: added to every expected time; the measured delay between
            the music and the player's movement *as seen by the camera*
            (from the clap-to-the-beat calibration, PLAN.md §4). It must be
            measured through the same feature pipeline, so it also absorbs
            the landmark filter's lag (~28 ms on fast moves).
    """

    bpm: float
    first_beat_ns: int
    latency_ns: int = 0

    @property
    def beat_ns(self) -> float:
        return 60e9 / self.bpm

    def time_of(self, beat: float) -> int:
        return int(self.first_beat_ns + beat * self.beat_ns + self.latency_ns)


@dataclass(frozen=True, slots=True)
class ScheduledKeyframe:
    """A keyframe with its expected time and context.

    Attributes:
        index: position in the schedule.
        t_ns: when the pose should be hit.
        block, keyframe: what to hit (and with which tolerance).
        block_start_ns, block_end_ns: the block instance this belongs to.
        instance: number of the block instance in the schedule.
        beat_ns: duration of one beat (for timing windows).
        hold: same pose as the previous keyframe; judged without timing.
    """

    index: int
    t_ns: int
    block: Block
    keyframe: Keyframe
    block_start_ns: int
    block_end_ns: int
    instance: int
    beat_ns: float
    hold: bool = False


def _same_pose(a: Keyframe, b: Keyframe) -> bool:
    if a.clap or b.clap:
        return False
    for side in ("right", "left"):
        pa, pb = getattr(a, side), getattr(b, side)
        if (pa is None) != (pb is None):
            return False
        if pa is not None and (angle_between_deg(pa.upper, pb.upper) > HOLD_MAX_CHANGE_DEG
                               or angle_between_deg(pa.fore, pb.fore) > HOLD_MAX_CHANGE_DEG):  # fmt: skip
            return False
    return True


def _with_holds(items: list[ScheduledKeyframe]) -> list[ScheduledKeyframe]:
    out = []
    for i, item in enumerate(items):
        prev = items[i - 1] if i else None
        hold = prev is not None and prev.instance == item.instance and _same_pose(prev.keyframe, item.keyframe)
        out.append(ScheduledKeyframe(item.index, item.t_ns, item.block, item.keyframe, item.block_start_ns,
                                     item.block_end_ns, item.instance, item.beat_ns, hold))  # fmt: skip
    return out


def schedule_routine(routine: Routine, clock: BeatClock) -> list[ScheduledKeyframe]:
    items: list[ScheduledKeyframe] = []
    instance, last_block_start = -1, None
    for tk in routine.timeline():
        block_start_beat = tk.beat - tk.keyframe.beat
        if block_start_beat != last_block_start:
            instance, last_block_start = instance + 1, block_start_beat
        items.append(ScheduledKeyframe(
            index=len(items), t_ns=clock.time_of(tk.beat), block=tk.block, keyframe=tk.keyframe,
            block_start_ns=clock.time_of(block_start_beat),
            block_end_ns=clock.time_of(block_start_beat + tk.block.duration),
            instance=instance, beat_ns=clock.beat_ns,
        ))  # fmt: skip
    return _with_holds(items)


def schedule_block_clip(block: Block, beat_times_ns: list[int], latency_ns: int = 0) -> list[ScheduledKeyframe]:
    """``block`` repeated over ``beat_times_ns`` (the recorded counts, without
    count-in). Fractional keyframe beats are interpolated between beats."""
    beats = np.asarray(beat_times_ns, dtype=np.float64)
    beat_ns = float(np.median(np.diff(beats))) if len(beats) > 1 else 5e8

    def at(beat: float) -> int:
        i = int(np.floor(beat))
        base = beats[i] if i < len(beats) else beats[-1] + (i - len(beats) + 1) * beat_ns
        return int(base + (beat - i) * beat_ns + latency_ns)

    items: list[ScheduledKeyframe] = []
    for instance, start in enumerate(range(0, len(beats) - block.duration + 1, block.duration)):
        for kf in block.keyframes:
            items.append(ScheduledKeyframe(
                index=len(items), t_ns=at(start + kf.beat), block=block, keyframe=kf,
                block_start_ns=at(start), block_end_ns=at(start + block.duration),
                instance=instance, beat_ns=beat_ns,
            ))  # fmt: skip
    return _with_holds(items)
