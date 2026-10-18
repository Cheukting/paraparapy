"""Rolling statistics shown live in the capture debug view's HUD.

Pure Python (no ``cv2``/``mediapipe``/``pygame``). For whole-run summaries
written to log files see :mod:`parapara.diagnostics.recorder`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from parapara.pose.landmarks import PoseSample


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


class CameraRateMeter:
    """Frames per second the *camera* delivers, measured from sample data.

    The grabber numbers every camera frame, and tracking skips frames when
    it is slower than the camera, so over a time window
    ``(last_index - first_index) / (last_time - first_time)`` is the camera's
    real delivery rate even though only some frames are processed. Comparing
    it with the tracking rate tells whether low FPS comes from the camera
    (e.g. a webcam lowering its frame rate in dim light) or from inference.
    """

    def __init__(self, window_s: float = 2.0) -> None:
        self._window_ns = int(window_s * 1e9)
        self._points: deque[tuple[int, int]] = deque()  # (captured_ns, frame_index)

    def add(self, captured_ns: int, frame_index: int) -> None:
        self._points.append((captured_ns, frame_index))
        while self._points and captured_ns - self._points[0][0] > self._window_ns:
            self._points.popleft()

    @property
    def rate(self) -> float:
        if len(self._points) < 2:
            return 0.0
        (t0, i0), (t1, i1) = self._points[0], self._points[-1]
        return (i1 - i0) / ((t1 - t0) / 1e9) if t1 > t0 else 0.0

    @property
    def skipped_fraction(self) -> float:
        """Share of camera frames in the window that were not processed."""
        if len(self._points) < 2:
            return 0.0
        delivered = self._points[-1][1] - self._points[0][1]
        processed = len(self._points) - 1
        return max(0.0, 1 - processed / delivered) if delivered > 0 else 0.0


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
    camera_rate: CameraRateMeter
    pose_ms: RollingMean
    processing_ms: RollingMean
    display_latency_ms: RollingMean
    dropped: int = 0
    samples: int = 0
    with_pose: int = 0

    @classmethod
    def new(cls) -> CaptureStats:
        return cls(
            RateMeter(), CameraRateMeter(), RollingMean(), RollingMean(), RollingMean()
        )

    def add_sample(self, sample: PoseSample) -> None:
        self.tracking_rate.tick(sample.processed_ns)
        self.camera_rate.add(sample.captured_ns, sample.frame_index)
        self.pose_ms.add(sample.pose_ms)
        self.processing_ms.add(sample.processing_ms)
        self.dropped = sample.dropped_samples
        self.samples += 1
        self.with_pose += sample.has_pose

    def hud_lines(self) -> list[str]:
        detected = 100 * self.with_pose / self.samples if self.samples else 0.0
        return [
            f"tracking  {self.tracking_rate.rate:5.1f} fps   person found {detected:3.0f}%",
            f"camera    {self.camera_rate.rate:5.1f} fps   skipped "
            f"{100 * self.camera_rate.skipped_fraction:3.0f}%",
            f"pose model       {self.pose_ms.mean:5.1f} ms",
            f"camera→landmarks {self.processing_ms.mean:5.1f} ms",
            f"camera→screen    {self.display_latency_ms.mean:5.1f} ms",
            f"dropped samples  {self.dropped}",
        ]
