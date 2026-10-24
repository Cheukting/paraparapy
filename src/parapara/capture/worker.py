"""Capture process entry point: camera → landmarks → game.

:func:`run` is started by :class:`parapara.ipc.client.CaptureClient` in its
own OS process (see that module for the worker contract). It:

1. reports ``STARTING`` and makes sure the model bundles are cached
   (downloading and verifying them on first use);
2. opens the camera and the landmark detector, then reports ``READY`` with the
   camera's actual format;
3. runs :func:`capture_loop` until the game sets ``stop_event``;
4. reports ``STOPPED`` (or ``ERROR`` with a readable message) and releases
   the camera.

:func:`capture_loop` takes its collaborators as arguments so the loop logic
can be unit-tested without a camera or MediaPipe.
"""

from __future__ import annotations

import queue
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from parapara.capture.camera import CameraFrame, LatestFrameGrabber
from parapara.capture.detector import Detection, LandmarkDetector
from parapara.capture.models import MODELS, ModelSpec, ensure_model
from parapara.ipc.frame_buffer import SharedFrameBuffer
from parapara.ipc.messages import CaptureConfig, CaptureStatus, StatusKind
from parapara.pose.landmarks import PoseSample

type RGBFrame = npt.NDArray[np.uint8]


class FrameSource(Protocol):
    def next_frame(self, after: int, timeout: float) -> CameraFrame | None: ...


class Detector(Protocol):
    def detect(self, rgb: RGBFrame, captured_ns: int) -> Detection: ...


class FrameSink(Protocol):
    def write(self, frame: RGBFrame, frame_index: int, captured_ns: int) -> None: ...


def to_rgb(image_bgr: npt.NDArray[np.uint8], width: int, height: int) -> RGBFrame:
    """OpenCV BGR camera image → RGB of the configured size."""
    img = image_bgr
    if img.shape[1] != width or img.shape[0] != height:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.asarray(rgb, dtype=np.uint8)


class QueueEmitter:
    """Puts messages on the result queue without ever blocking the loop.

    When the queue is full (the game is not draining it), the sample is
    dropped and counted; the count travels with the next sample so the game
    can display it.
    """

    def __init__(self, results: Any) -> None:
        self._results = results
        self.dropped = 0

    def __call__(self, message: PoseSample | CaptureStatus) -> None:
        try:
            self._results.put_nowait(message)
        except queue.Full:
            self.dropped += 1


def capture_loop(
    *,
    source: FrameSource,
    detector: Detector,
    frames: FrameSink,
    emit: Callable[[PoseSample], None],
    should_stop: Callable[[], bool],
    convert: Callable[[npt.NDArray[np.uint8]], RGBFrame],
    frame_size: tuple[int, int],
    dropped: Callable[[], int] = lambda: 0,
    poll_timeout: float = 0.2,
) -> int:
    """Process frames until ``should_stop()``; returns the number processed.

    The frame is published to the shared buffer *after* detection, under the
    same ``frame_index`` as its sample, so the debug view can draw a skeleton
    that exactly matches the picture.
    """
    last_index = -1
    processed = 0
    while not should_stop():
        frame = source.next_frame(after=last_index, timeout=poll_timeout)
        if frame is None:
            continue
        last_index = frame.index
        rgb = convert(frame.image)
        detection = detector.detect(rgb, frame.captured_ns)
        frames.write(rgb, frame.index, frame.captured_ns)
        emit(
            PoseSample(
                frame_index=frame.index,
                captured_ns=frame.captured_ns,
                processed_ns=time.monotonic_ns(),
                frame_size=frame_size,
                pose_image=detection.pose_image,
                pose_world=detection.pose_world,
                pose_ms=detection.pose_ms,
                dropped_samples=dropped(),
            )
        )
        processed += 1
    return processed


def run(
    *,
    config: CaptureConfig,
    results: Any,
    stop_event: Any,
    frame_buffer_name: str,
    frame_shape: tuple[int, int, int],
    frame_lock: Any,
) -> None:
    """Capture process main function (see module docstring)."""
    emit = QueueEmitter(results)
    grabber: LatestFrameGrabber | None = None
    detector: LandmarkDetector | None = None
    frames: SharedFrameBuffer | None = None
    try:
        emit(CaptureStatus(StatusKind.STARTING, "loading models"))
        model_dir = Path(config.model_dir) if config.model_dir else None

        def downloading(spec: ModelSpec) -> None:
            emit(CaptureStatus(StatusKind.STARTING, f"downloading {spec.filename}"))

        pose_path = ensure_model(
            MODELS[f"pose_{config.pose_model}"], model_dir, downloading
        )

        emit(CaptureStatus(StatusKind.STARTING, "opening camera"))
        frames = SharedFrameBuffer.attach(frame_buffer_name, frame_shape, frame_lock)
        detector = LandmarkDetector(pose_path)
        grabber = LatestFrameGrabber(
            config.camera_index, config.width, config.height, config.fps
        )
        grabber.start()

        width, height, fps = grabber.actual_format
        emit(
            CaptureStatus(
                StatusKind.READY,
                "capturing",
                {"camera_width": width, "camera_height": height, "camera_fps": fps},
            )
        )
        capture_loop(
            source=grabber,
            detector=detector,
            frames=frames,
            emit=emit,
            should_stop=stop_event.is_set,
            convert=lambda bgr: to_rgb(bgr, config.width, config.height),
            frame_size=(config.width, config.height),
            dropped=lambda: emit.dropped,
        )
        emit(CaptureStatus(StatusKind.STOPPED))
    except Exception as err:  # report *any* failure to the game, then exit
        emit(
            CaptureStatus(
                StatusKind.ERROR, str(err), {"traceback": traceback.format_exc()}
            )
        )
    finally:
        if grabber is not None:
            grabber.stop()
        if detector is not None:
            detector.close()
        if frames is not None:
            frames.close()
