"""Judging keyframes as the features stream in (PLAN.md §5).

:class:`Judge` receives :class:`~parapara.pose.features.PoseFeatures` in
time order and, as soon as a keyframe's timing window has passed, emits a
:class:`Judgment` (so the game can show "Perfect!" right after the beat).
The same code scores recorded clips offline (``parapara-score``).

Per keyframe, within ``±window`` of its expected time:

- **Normal keyframes**: the sample with the lowest pose error (front-weighted,
  :mod:`parapara.scoring.matching`) is the hit; a tiny penalty for distance
  from the beat breaks ties. Its time is refined between samples with a
  parabola through the neighbouring errors (the camera delivers only ~23 fps,
  i.e. 43 ms between samples). The offset grades the timing:
  Perfect ≤ 60 ms, Great ≤ 120 ms, Good ≤ the window (190 ms, capped at
  0.45 beat so neighbouring windows don't overlap).
- **Hold keyframes** (same pose as the previous one, e.g. the second knock):
  timing is unmeasurable, so the sample nearest the beat is judged on pose
  only, graded ``HOLD``.
- **Clap keyframes**: hands together (wrist distance, vision). If clap sound
  onsets are supplied (:meth:`Judge.add_clap_onset`, PLAN.md §5.1), the onset
  nearest the beat gives the timing and the vision check gates it; otherwise
  the moment the wrists are closest gives the timing.
- A pose score below :data:`MIN_POSE_SCORE`, or nothing in the window, is a
  ``MISS``.

Points per keyframe = pose score × timing factor (Perfect 1.0, Great 0.8,
Good 0.5, Hold 1.0).

**Flowing blocks** (sweeps, wrist roll) also get a :class:`FlowJudgment`
when they end: the share of the block during which the wrists keep moving
faster than :data:`MOVING_SPEED` shoulder widths per second (measured on the
2026-10-18 recordings: held poses stay below ~0.6 for 90% of samples, the
wrist roll stays above it for 90%). This is the "keep moving" check: the
wrist roll's rolling is invisible as a direction but visible as motion.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from parapara.pose.features import PoseFeatures
from parapara.scoring.matching import clap_score, keyframe_error_deg, pose_score
from parapara.scoring.schedule import ScheduledKeyframe

MIN_POSE_SCORE = 0.3
TIE_BREAK_DEG = 2.0  # error added at the edge of the window, to prefer hits near the beat
MOVING_SPEED = 0.6  # shoulder widths / s
MOVING_SHARE_FOR_FULL = 0.8


class Grade(StrEnum):
    PERFECT = "perfect"
    GREAT = "great"
    GOOD = "good"
    HOLD = "hold"
    MISS = "miss"


TIMING_FACTOR = {Grade.PERFECT: 1.0, Grade.GREAT: 0.8, Grade.GOOD: 0.5, Grade.HOLD: 1.0, Grade.MISS: 0.0}


@dataclass(frozen=True, slots=True)
class TimingWindows:
    perfect_ms: float = 60.0
    great_ms: float = 120.0
    good_ms: float = 190.0
    max_beat_fraction: float = 0.45

    def window_ns(self, beat_ns: float) -> float:
        return min(self.good_ms * 1e6, self.max_beat_fraction * beat_ns)

    def grade(self, offset_ns: float, window_ns: float) -> Grade:
        off = abs(offset_ns)
        if off <= self.perfect_ms * 1e6:
            return Grade.PERFECT
        if off <= self.great_ms * 1e6:
            return Grade.GREAT
        return Grade.GOOD if off <= window_ns else Grade.MISS


@dataclass(frozen=True, slots=True)
class Judgment:
    item: ScheduledKeyframe
    grade: Grade
    pose_score: float
    points: float
    offset_ms: float | None = None  # + = late; None for holds and empty windows
    error_deg: float | None = None  # None for claps
    reason: str = ""  # why a MISS (for feedback)


@dataclass(frozen=True, slots=True)
class FlowJudgment:
    item: ScheduledKeyframe  # first keyframe of the flowing block instance
    moving_share: float
    score: float


def _refine_time(ts: np.ndarray, es: np.ndarray, i: int) -> float:
    """Vertex of the parabola through samples i-1, i, i+1 (time of minimum
    error), clamped to the neighbours; ``ts[i]`` at the edges."""
    if i == 0 or i == len(ts) - 1:
        return float(ts[i])
    t0, t1, t2 = ts[i - 1:i + 2]
    e0, e1, e2 = es[i - 1:i + 2]
    denom = (t0 - t1) * (t0 - t2) * (t1 - t2)
    a = (t2 * (e1 - e0) + t1 * (e0 - e2) + t0 * (e2 - e1)) / denom
    b = (t2 ** 2 * (e0 - e1) + t1 ** 2 * (e2 - e0) + t0 ** 2 * (e1 - e2)) / denom
    if a <= 0:
        return float(t1)
    return float(np.clip(-b / (2 * a), t0, t2))


class Judge:
    """Streaming judge for a schedule (see module docstring)."""

    def __init__(self, schedule: list[ScheduledKeyframe], windows: TimingWindows | None = None) -> None:
        self.schedule = schedule
        self.windows = windows or TimingWindows()
        self._next = 0  # index of the next keyframe to judge
        self._buffer: deque[PoseFeatures] = deque()
        self._onsets: list[int] = []
        self._flow_done: set[int] = set()
        self._flow_pending = sorted({(s.block_end_ns, s.instance): s for s in schedule
                                     if s.block.is_flowing}.items())  # fmt: skip

    def add_clap_onset(self, t_ns: int) -> None:
        """A clap sound detected at ``t_ns`` (from the audio module)."""
        self._onsets.append(t_ns)

    def add(self, features: PoseFeatures | None, now_ns: int | None = None) -> list[Judgment | FlowJudgment]:
        """Add one sample (``None`` = no person; only advances time) and
        return the judgments whose windows have closed."""
        if features is not None:
            self._buffer.append(features)
            now_ns = features.t_ns if now_ns is None else now_ns
        if now_ns is None:
            return []
        return self._judge_until(now_ns)

    def finish(self) -> list[Judgment | FlowJudgment]:
        """Judge everything left (end of the routine)."""
        return self._judge_until(None)

    def _judge_until(self, now_ns: int | None) -> list[Judgment | FlowJudgment]:
        out: list[Judgment | FlowJudgment] = []
        while self._next < len(self.schedule):
            item = self.schedule[self._next]
            window = self.windows.window_ns(item.beat_ns)
            if now_ns is not None and now_ns <= item.t_ns + window:
                break
            out.append(self._judge(item, window))
            self._next += 1
        for (end_ns, instance), item in self._flow_pending:
            if instance not in self._flow_done and (now_ns is None or now_ns > end_ns):
                self._flow_done.add(instance)
                out.append(self._judge_flow(item))
        self._trim()
        return out

    def _trim(self) -> None:
        """Forget samples no pending keyframe or flowing block can still use."""
        starts = []
        if self._next < len(self.schedule):
            item = self.schedule[self._next]
            starts.append(item.t_ns - self.windows.window_ns(item.beat_ns))
        starts += [item.block_start_ns for (_, inst), item in self._flow_pending if inst not in self._flow_done]
        if not starts:
            return
        horizon = min(starts)
        while self._buffer and self._buffer[0].t_ns < horizon:
            self._buffer.popleft()

    def _window_samples(self, item: ScheduledKeyframe, window: float) -> list[PoseFeatures]:
        return [f for f in self._buffer if abs(f.t_ns - item.t_ns) <= window]

    def _judge(self, item: ScheduledKeyframe, window: float) -> Judgment:
        samples = self._window_samples(item, window)
        if not samples:
            return Judgment(item, Grade.MISS, 0.0, 0.0, reason="not seen")
        if item.keyframe.clap:
            return self._judge_clap(item, samples, window)

        tol = item.block.tolerance_for(item.keyframe)
        ts = np.array([f.t_ns for f in samples], dtype=np.float64)
        es = np.array([keyframe_error_deg(f, item.keyframe) for f in samples])
        if item.hold:
            i = int(np.argmin(np.abs(ts - item.t_ns)))
            score = pose_score(float(es[i]), tol)
            grade = Grade.HOLD if score >= MIN_POSE_SCORE else Grade.MISS
            return Judgment(item, grade, score, score * TIMING_FACTOR[grade], error_deg=float(es[i]),
                            reason="" if grade is Grade.HOLD else "pose")  # fmt: skip

        cost = es + TIE_BREAK_DEG * np.abs(ts - item.t_ns) / window
        i = int(np.argmin(cost))
        hit_ns = _refine_time(ts, es, i)
        score = pose_score(float(es[i]), tol)
        offset = hit_ns - item.t_ns
        grade = self.windows.grade(offset, window) if score >= MIN_POSE_SCORE else Grade.MISS
        return Judgment(item, grade, score, score * TIMING_FACTOR[grade], offset / 1e6, float(es[i]),
                        "" if grade is not Grade.MISS else "pose")  # fmt: skip

    def _judge_clap(self, item: ScheduledKeyframe, samples: list[PoseFeatures], window: float) -> Judgment:
        ts = np.array([f.t_ns for f in samples], dtype=np.float64)
        onsets = [t for t in self._onsets if abs(t - item.t_ns) <= window]
        if onsets:
            hit_ns = float(min(onsets, key=lambda t: abs(t - item.t_ns)))
            score = clap_score(samples[int(np.argmin(np.abs(ts - hit_ns)))])  # vision gate
        else:
            dist = np.array([f.wrist_distance for f in samples])
            i = int(np.argmin(dist + 0.05 * np.abs(ts - item.t_ns) / window))
            hit_ns = _refine_time(ts, dist, i)
            score = clap_score(samples[i])
        offset = hit_ns - item.t_ns
        grade = self.windows.grade(offset, window) if score >= MIN_POSE_SCORE else Grade.MISS
        return Judgment(item, grade, score, score * TIMING_FACTOR[grade], offset / 1e6,
                        reason="" if grade is not Grade.MISS else "no clap")  # fmt: skip

    def _judge_flow(self, item: ScheduledKeyframe) -> FlowJudgment:
        samples = [f for f in self._buffer if item.block_start_ns <= f.t_ns <= item.block_end_ns]
        if len(samples) < 3:
            return FlowJudgment(item, 0.0, 0.0)
        t = np.array([f.t_ns for f in samples]) / 1e9
        speeds = []
        for side in ("right_wrist", "left_wrist"):
            p = np.stack([getattr(f, side) for f in samples])
            speeds.append(np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6))
        moving = float(np.mean(np.mean(speeds, axis=0) >= MOVING_SPEED))
        return FlowJudgment(item, moving, float(min(1.0, moving / MOVING_SHARE_FOR_FULL)))


def judge_all(schedule: list[ScheduledKeyframe], features: Iterable[PoseFeatures | None],
              windows: TimingWindows | None = None) -> list[Judgment | FlowJudgment]:  # fmt: skip
    """Offline convenience: stream ``features`` through a :class:`Judge`."""
    judge = Judge(schedule, windows)
    out: list[Judgment | FlowJudgment] = []
    for f in features:
        out += judge.add(f)
    return out + judge.finish()
