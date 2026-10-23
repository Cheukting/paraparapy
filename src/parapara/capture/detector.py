"""MediaPipe Pose landmark detection, converted to plain NumPy.

Uses the MediaPipe **Tasks** API (``mediapipe.tasks.python.vision``), checked
against mediapipe 1.1.1 on 2026-10-18:

- ``PoseLandmarker`` created with ``RunningMode.VIDEO`` and called with
  ``detect_for_video(image, ts_ms)``;
- results expose ``pose_landmarks`` / ``pose_world_landmarks``;
- each landmark has ``x, y, z, visibility, presence``.

Why VIDEO mode rather than LIVE_STREAM
--------------------------------------
LIVE_STREAM returns results asynchronously via a callback and silently drops
frames while busy. VIDEO mode is synchronous, so every result belongs to a
known frame and its timing is easy to measure. Frame dropping is handled
explicitly upstream by :class:`~parapara.capture.camera.LatestFrameGrabber`.

VIDEO mode requires **strictly increasing** timestamps (verified: a repeated
timestamp raises ``ValueError``), which :class:`LandmarkDetector` guarantees.

The Hand Landmarker is deliberately not used (too slow and unreliable at
dancing distance; see ``parapara.pose.landmarks``).

The converter uses duck typing, so tests can exercise it with fake objects.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
import numpy.typing as npt
from mediapipe.tasks.python import BaseOptions, vision

from parapara.pose.landmarks import NUM_POSE_LANDMARKS, FloatArray


def pose_landmarks_to_array(landmarks: Sequence[Any]) -> FloatArray:
    """MediaPipe landmarks → ``(33, 5)`` array of ``x, y, z, visibility, presence``.

    Missing ``visibility``/``presence`` values become NaN.
    """
    if len(landmarks) != NUM_POSE_LANDMARKS:
        raise ValueError(
            f"expected {NUM_POSE_LANDMARKS} pose landmarks, got {len(landmarks)}"
        )

    def value(v: float | None) -> float:
        return np.nan if v is None else float(v)

    return np.array(
        [
            (lm.x, lm.y, lm.z, value(lm.visibility), value(lm.presence))
            for lm in landmarks
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True, slots=True)
class Detection:
    """Landmarks found in one frame."""

    pose_image: FloatArray | None
    pose_world: FloatArray | None
    pose_ms: float = 0.0


class LandmarkDetector:
    """Runs the Pose Landmarker on RGB frames.

    Args:
        pose_model: path to a ``pose_landmarker_*.task`` bundle.
    """

    def __init__(self, pose_model: Path) -> None:
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(pose_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,  # single player (PLAN.md §10)
            )
        )
        self._last_ts_ms = -1

    def _timestamp_ms(self, captured_ns: int) -> int:
        # VIDEO mode rejects non-increasing timestamps; two frames captured
        # within the same millisecond would otherwise crash the worker.
        ts = max(captured_ns // 1_000_000, self._last_ts_ms + 1)
        self._last_ts_ms = ts
        return ts

    def detect(self, rgb: npt.NDArray[np.uint8], captured_ns: int) -> Detection:
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        ts = self._timestamp_ms(captured_ns)
        t0 = time.perf_counter()
        result = self._pose.detect_for_video(image, ts)
        pose_ms = (time.perf_counter() - t0) * 1000
        if not result.pose_landmarks:
            return Detection(None, None, pose_ms)
        return Detection(
            pose_image=pose_landmarks_to_array(result.pose_landmarks[0]),
            pose_world=pose_landmarks_to_array(result.pose_world_landmarks[0]),
            pose_ms=pose_ms,
        )

    def close(self) -> None:
        self._pose.close()
