"""Pure helpers for drawing landmark overlays and HUD statistics.

Kept free of ``pygame`` so they can be unit-tested in the main pytest process
(which also imports ``cv2`` for the capture tests; the two must never be
loaded together, see the package docstring).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from parapara.pose.landmarks import PoseSample

VISIBILITY_THRESHOLD = 0.5


def to_screen(
    points: npt.ArrayLike, size: tuple[int, int], mirrored: bool
) -> list[tuple[int, int]]:
    """Normalized image landmarks (``x, y`` in ``[0, 1]``) → pixel positions.

    ``mirrored`` flips x, matching a horizontally flipped camera picture, so
    that the overlay stays on top of the body in a "selfie" view.
    """
    xy = np.asarray(points, dtype=np.float64)[:, :2].copy()
    if mirrored:
        xy[:, 0] = 1.0 - xy[:, 0]
    width, height = size
    xy *= (width, height)
    return [(round(x), round(y)) for x, y in xy]


def visible(pose: npt.NDArray[np.float32], index: int, threshold: float = VISIBILITY_THRESHOLD) -> bool:
    """Whether a pose landmark is confidently visible (NaN counts as not)."""
    v = float(pose[index, 3])
    return v == v and v >= threshold  # v == v is False for NaN


class RateMeter:
    """Events per second over a sliding time window (e.g. tracking FPS)."""

    def __init__(self, window_s: float = 1.0) -> None:
        self._window_ns = int(window_s * 1e9)
        self._times: deque[int] = deque()

    def tick(self, t_ns: int) -> None:
        self._times.append(t_ns)
        while self._times and t_ns - self._times[0] > self._window_ns:
            self._times.popleft()

    @property
    def rate(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = (self._times[-1] - self._times[0]) / 1e9
        return (len(self._times) - 1) / span if span > 0 else 0.0


class RollingMean:
    """Mean of the last ``size`` values (for jittery timings)."""

    def __init__(self, size: int = 30) -> None:
        self._values: deque[float] = deque(maxlen=size)

    def add(self, value: float) -> None:
        self._values.append(value)

    @property
    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


@dataclass
class CaptureStats:
    """Rolling statistics the debug view displays."""

    tracking_rate: RateMeter
    pose_ms: RollingMean
    hands_ms: RollingMean
    processing_ms: RollingMean
    display_latency_ms: RollingMean
    dropped: int = 0
    samples: int = 0
    with_pose: int = 0

    @classmethod
    def new(cls) -> CaptureStats:
        return cls(RateMeter(), RollingMean(), RollingMean(), RollingMean(), RollingMean())

    def add_sample(self, sample: PoseSample) -> None:
        self.tracking_rate.tick(sample.processed_ns)
        self.pose_ms.add(sample.pose_ms)
        self.hands_ms.add(sample.hands_ms)
        self.processing_ms.add(sample.processing_ms)
        self.dropped = sample.dropped_samples
        self.samples += 1
        self.with_pose += sample.has_pose

    def hud_lines(self) -> list[str]:
        detected = 100 * self.with_pose / self.samples if self.samples else 0.0
        return [
            f"tracking  {self.tracking_rate.rate:5.1f} fps   person found {detected:3.0f}%",
            f"pose      {self.pose_ms.mean:5.1f} ms    hands {self.hands_ms.mean:5.1f} ms",
            f"camera→landmarks {self.processing_ms.mean:5.1f} ms",
            f"camera→screen    {self.display_latency_ms.mean:5.1f} ms",
            f"dropped samples  {self.dropped}",
        ]
