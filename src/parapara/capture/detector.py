"""MediaPipe Pose + Hand landmark detection, converted to plain NumPy.

Uses the MediaPipe **Tasks** API (``mediapipe.tasks.python.vision``), checked
against mediapipe 1.1.1 on 2026-10-18:

- ``PoseLandmarker`` / ``HandLandmarker`` created with
  ``RunningMode.VIDEO`` and called with ``detect_for_video(image, ts_ms)``;
- results expose ``pose_landmarks`` / ``pose_world_landmarks`` and
  ``hand_landmarks`` / ``hand_world_landmarks`` / ``handedness``;
- each landmark has ``x, y, z, visibility, presence``.

Why VIDEO mode rather than LIVE_STREAM
--------------------------------------
LIVE_STREAM returns results asynchronously via a callback and silently drops
frames while busy. VIDEO mode is synchronous, so every result belongs to a
known frame and its timing is easy to measure. Frame dropping is handled
explicitly upstream by :class:`~parapara.capture.camera.LatestFrameGrabber`.

VIDEO mode requires **strictly increasing** timestamps (verified: a repeated
timestamp raises ``ValueError``), which :class:`LandmarkDetector` guarantees.

The ``*_to_array`` converters use duck typing, so tests can exercise them
with simple fake objects.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
import numpy.typing as npt
from mediapipe.tasks.python import BaseOptions, vision

from parapara.pose.landmarks import (
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
    FloatArray,
    HandObservation,
)


def pose_landmarks_to_array(landmarks: Sequence[Any]) -> FloatArray:
    """MediaPipe landmarks → ``(33, 5)`` array of ``x, y, z, visibility, presence``.

    Missing ``visibility``/``presence`` values become NaN.
    """
    if len(landmarks) != NUM_POSE_LANDMARKS:
        raise ValueError(f"expected {NUM_POSE_LANDMARKS} pose landmarks, got {len(landmarks)}")

    def value(v: float | None) -> float:
        return np.nan if v is None else float(v)

    return np.array(
        [(lm.x, lm.y, lm.z, value(lm.visibility), value(lm.presence)) for lm in landmarks],
        dtype=np.float32,
    )


def hand_landmarks_to_array(landmarks: Sequence[Any]) -> FloatArray:
    """MediaPipe hand landmarks → ``(21, 3)`` array of ``x, y, z``."""
    if len(landmarks) != NUM_HAND_LANDMARKS:
        raise ValueError(f"expected {NUM_HAND_LANDMARKS} hand landmarks, got {len(landmarks)}")
    return np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)


def hands_from_result(result: Any) -> list[HandObservation]:
    """Convert a ``HandLandmarkerResult`` into :class:`HandObservation` objects."""
    hands = []
    for image_lms, world_lms, categories in zip(
        result.hand_landmarks, result.hand_world_landmarks, result.handedness, strict=True
    ):
        top = categories[0] if categories else None
        hands.append(
            HandObservation(
                image=hand_landmarks_to_array(image_lms),
                world=hand_landmarks_to_array(world_lms),
                score=float(top.score) if top is not None else 0.0,
                label=str(top.category_name) if top is not None else "",
            )
        )
    return hands


@dataclass(frozen=True, slots=True)
class Detection:
    """Landmarks found in one frame (before left/right hand assignment)."""

    pose_image: FloatArray | None
    pose_world: FloatArray | None
    hands: list[HandObservation] = field(default_factory=list)
    pose_ms: float = 0.0
    hands_ms: float = 0.0


class LandmarkDetector:
    """Runs the Pose (and optionally Hand) Landmarker on RGB frames.

    Args:
        pose_model: path to a ``pose_landmarker_*.task`` bundle.
        hand_model: path to ``hand_landmarker.task``, or ``None`` to skip
            hand detection (faster; fists/flicks then cannot be scored).
    """

    def __init__(self, pose_model: Path, hand_model: Path | None) -> None:
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(pose_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,  # single player (PLAN.md §10)
            )
        )
        self._hands = None
        if hand_model is not None:
            self._hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(hand_model)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2,
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
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = self._timestamp_ms(captured_ns)

        t0 = time.perf_counter()
        pose_result = self._pose.detect_for_video(image, ts)
        t1 = time.perf_counter()
        pose_image = pose_world = None
        if pose_result.pose_landmarks:
            pose_image = pose_landmarks_to_array(pose_result.pose_landmarks[0])
            pose_world = pose_landmarks_to_array(pose_result.pose_world_landmarks[0])

        hands: list[HandObservation] = []
        if self._hands is not None:
            hands = hands_from_result(self._hands.detect_for_video(image, ts))
        t2 = time.perf_counter()

        return Detection(
            pose_image=pose_image,
            pose_world=pose_world,
            hands=hands,
            pose_ms=(t1 - t0) * 1000,
            hands_ms=(t2 - t1) * 1000,
        )

    def close(self) -> None:
        self._pose.close()
        if self._hands is not None:
            self._hands.close()
