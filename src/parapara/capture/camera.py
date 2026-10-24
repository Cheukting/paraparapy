"""Webcam reader that always hands out the *newest* frame.

Why a grabber thread
--------------------
``cv2.VideoCapture.read()`` returns frames in order from the driver's queue.
If landmark detection is slower than the camera for a moment, a naive
"read → detect" loop falls further and further behind real time, which is
fatal for a rhythm game. :class:`LatestFrameGrabber` reads continuously on a
background thread and keeps only the newest frame; the detection loop asks
for "the next frame newer than the one I processed" and skipped frames simply
show up as gaps in ``frame_index``.

Timestamps are taken with ``time.monotonic_ns()`` right after ``read()``
returns. That excludes the camera's own exposure/transfer delay, which the
player calibration step (PLAN.md §3) measures and compensates.

OpenCV 5 API reference: https://docs.opencv.org/5.x/d8/dfe/classcv_1_1VideoCapture.html
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CameraFrame:
    index: int
    captured_ns: int
    image: npt.NDArray[np.uint8]  # BGR, as delivered by OpenCV


def _backend() -> int:
    # AVFoundation is the native macOS backend; let OpenCV pick elsewhere.
    return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY


class LatestFrameGrabber:
    """Background thread that keeps the newest frame from a camera.

    Args:
        camera_index: OpenCV device index.
        width, height, fps: requested capture format (best effort; read the
            actual values from :attr:`actual_format` after :meth:`start`).
    """

    _MAX_CONSECUTIVE_FAILURES = 30  # ~1 s at 30 fps

    def __init__(self, camera_index: int, width: int, height: int, fps: int) -> None:
        self._camera_index = camera_index
        self._requested = (width, height, fps)
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._new_frame = threading.Condition()
        self._latest: CameraFrame | None = None
        self._error: str | None = None
        self.actual_format: tuple[int, int, float] = (0, 0, 0.0)

    def start(self) -> None:
        capture = cv2.VideoCapture(self._camera_index, _backend())
        if not capture.isOpened():
            raise CameraError(
                f"could not open camera {self._camera_index}. On macOS, allow camera "
                "access for your terminal/IDE in System Settings → Privacy & Security → Camera."
            )
        width, height, fps = self._requested
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        # Ask the driver for the smallest queue; not every backend honours it.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.actual_format = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
        )
        self._capture = capture
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="camera-grabber", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        assert self._capture is not None
        index = 0
        failures = 0
        while self._running.is_set():
            ok, image = self._capture.read()
            captured_ns = time.monotonic_ns()
            if not ok or image is None:
                failures += 1
                if failures >= self._MAX_CONSECUTIVE_FAILURES:
                    self._fail("camera stopped delivering frames")
                    return
                time.sleep(0.01)
                continue
            failures = 0
            with self._new_frame:
                self._latest = CameraFrame(
                    index, captured_ns, np.asarray(image, dtype=np.uint8)
                )
                self._new_frame.notify_all()
            index += 1

    def _fail(self, message: str) -> None:
        with self._new_frame:
            self._error = message
            self._new_frame.notify_all()

    def next_frame(self, after: int, timeout: float) -> CameraFrame | None:
        """Wait up to ``timeout`` s for a frame with ``index > after``.

        Returns ``None`` on timeout. Raises :class:`CameraError` if the camera
        failed.
        """
        with self._new_frame:
            self._new_frame.wait_for(
                lambda: (
                    self._error is not None
                    or (self._latest is not None and self._latest.index > after)
                ),
                timeout,
            )
            if self._error is not None:
                raise CameraError(self._error)
            if self._latest is None or self._latest.index <= after:
                return None
            return self._latest

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
        self._thread = None
        self._capture = None
