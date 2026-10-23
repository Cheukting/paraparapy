"""Latency calibration: "clap to the beat" (PLAN.md §4).

The player claps along to metronome clicks. For each click, the moment the
wrists come together is found in the **same signal the scoring times hits
on**: the unsmoothed timing copy of the features
(:meth:`parapara.pose.features.PoseFeatures.for_timing`, as used by the
judge since 2026-10-23), so the measured offset contains everything between
the music and the judged movement: audio output delay, camera and pose-model
delay, and the player's own habit of clapping slightly early or late. (Before
that change both used the smoothed landmarks, which added ~28 ms.) That is exactly the
``latency_ns`` the scoring schedule needs (:class:`~parapara.scoring.schedule.BeatClock`).

Per click, the clap is the minimum of the wrist distance within a window from
a quarter beat before to half a beat after the click (latency is normally
positive), refined between samples (~23 fps camera). It counts only if the
wrists really met (below 1.3 × the clap distance).

The result is the **median** offset of the claps found; the spread is the
scaled median absolute deviation. Because each clap is searched only within
its click's window, wild claps are clipped and the spread *underestimates*
large jitter (synthetic test: true ±40 ms → 22 ms, ±120 ms → 50 ms), so the
limit is set in measured units: 40 ms accepts careful clapping and rejects
claps scattered by about ±100 ms or more. A calibration is accepted with at least
:data:`MIN_CLAPS` claps (and half of the clicks) and a spread of at most
:data:`MAX_SPREAD_MS`; otherwise the player is asked to try again.

Clap detection by sound (PLAN.md §5.1) can later replace the vision timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from parapara.pose.features import PoseFeatures
from parapara.scoring.judge import refine_time
from parapara.scoring.matching import CLAP_DISTANCE

MIN_CLAPS = 6
MAX_SPREAD_MS = 40.0
CLAP_GATE = 1.3  # wrists closer than CLAP_GATE x CLAP_DISTANCE count as a clap


@dataclass(frozen=True)
class LatencyResult:
    latency_ms: float
    spread_ms: float
    claps_found: int
    clicks: int
    offsets_ms: list[float] = field(default_factory=list)
    ok: bool = False
    reason: str = ""


def estimate_latency(features: list[PoseFeatures | None], click_ns: list[int], beat_ns: float) -> LatencyResult:
    """Median clap offset relative to ``click_ns`` (see module docstring)."""
    samples = [f.for_timing() for f in features if f is not None]
    samples = [f for f in samples if np.isfinite(f.wrist_distance)]
    ts = np.array([f.t_ns for f in samples], dtype=np.float64)
    dist = np.array([f.wrist_distance for f in samples], dtype=np.float64)
    offsets = []
    for click in click_ns:
        in_window = np.flatnonzero((ts >= click - 0.25 * beat_ns) & (ts <= click + 0.5 * beat_ns))
        if len(in_window) == 0:
            continue
        i = int(in_window[np.argmin(dist[in_window])])
        if dist[i] > CLAP_GATE * CLAP_DISTANCE:
            continue  # the wrists never met: no clap for this click
        lo, hi = max(0, i - 1), min(len(ts), i + 2)
        hit = refine_time(ts[lo:hi], dist[lo:hi], i - lo)
        offsets.append((hit - click) / 1e6)

    if not offsets:
        return LatencyResult(0.0, 0.0, 0, len(click_ns), [], False, "no claps seen: clap with your hands in view")
    arr = np.array(offsets)
    median = float(np.median(arr))
    spread = float(1.4826 * np.median(np.abs(arr - median)))
    needed = max(MIN_CLAPS, len(click_ns) // 2)
    if len(arr) < needed:
        reason = f"only {len(arr)} of {len(click_ns)} claps seen (need {needed}): clap clearly, hands in view"
    elif spread > MAX_SPREAD_MS:
        reason = f"claps too uneven (spread {spread:.0f} ms): try to clap exactly on the clicks"
    else:
        reason = ""
    return LatencyResult(round(median, 1), round(spread, 1), len(arr), len(click_ns),
                         [round(o, 1) for o in offsets], not reason, reason)  # fmt: skip
