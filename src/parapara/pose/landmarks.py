"""Landmark data shared by the capture and game processes (pure NumPy).

The capture process converts MediaPipe results into the plain dataclasses
below and sends them to the game process through a ``multiprocessing.Queue``
(they pickle cheaply). Nothing here imports ``mediapipe`` or ``cv2``, so the
game process can use these types without loading OpenCV's bundled SDL2.

Landmark layout follows MediaPipe's Pose Landmarker (33 points), as
documented at
https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
(checked 2026-10-18). Index names are duplicated here, rather than imported
from ``mediapipe``, for the same import-boundary reason.

Array conventions
-----------------
- Pose arrays have shape ``(33, 5)`` with columns ``x, y, z, visibility,
  presence``. Image landmarks have ``x``/``y`` normalized to ``[0, 1]`` by the
  frame width/height (they may slightly exceed that range for off-screen
  joints); world landmarks are metres with the hip midpoint as origin.
- Frames are **not** mirrored: "left" means the person's anatomical left
  (MediaPipe pose labels), which appears on the *right* side of an
  unmirrored webcam image. Confirmed in the first human test (2026-10-18).

No hand tracking
----------------
The separate Hand Landmarker was removed after the 2026-10-18 benchmark: it
took ~55 ms per frame (halving the tracking rate to ~12 fps) and found each
hand only ~20% of the time at dancing distance. Hand *direction* (for flicks)
comes from the pose model's own index/pinky/thumb points at each wrist; hand
*shape* (fist/open) is not tracked; claps are detected by sound, confirmed by
the wrists meeting (PLAN.md §5.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float32]


class PoseLandmark(IntEnum):
    """MediaPipe Pose Landmarker indices."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


NUM_POSE_LANDMARKS = len(PoseLandmark)
POSE_COLUMNS = ("x", "y", "z", "visibility", "presence")

_P = PoseLandmark

# Only the upper body is scored (PLAN.md §1.1), so only it is drawn.
UPPER_BODY_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (_P.LEFT_SHOULDER, _P.RIGHT_SHOULDER),
    (_P.LEFT_SHOULDER, _P.LEFT_ELBOW),
    (_P.LEFT_ELBOW, _P.LEFT_WRIST),
    (_P.RIGHT_SHOULDER, _P.RIGHT_ELBOW),
    (_P.RIGHT_ELBOW, _P.RIGHT_WRIST),
    # Hands, from the pose model's coarse hand points.
    (_P.LEFT_WRIST, _P.LEFT_INDEX),
    (_P.LEFT_WRIST, _P.LEFT_PINKY),
    (_P.LEFT_INDEX, _P.LEFT_PINKY),
    (_P.LEFT_WRIST, _P.LEFT_THUMB),
    (_P.RIGHT_WRIST, _P.RIGHT_INDEX),
    (_P.RIGHT_WRIST, _P.RIGHT_PINKY),
    (_P.RIGHT_INDEX, _P.RIGHT_PINKY),
    (_P.RIGHT_WRIST, _P.RIGHT_THUMB),
    (_P.LEFT_SHOULDER, _P.LEFT_HIP),
    (_P.RIGHT_SHOULDER, _P.RIGHT_HIP),
    (_P.LEFT_HIP, _P.RIGHT_HIP),
    (_P.LEFT_EAR, _P.LEFT_EYE),
    (_P.LEFT_EYE, _P.NOSE),
    (_P.NOSE, _P.RIGHT_EYE),
    (_P.RIGHT_EYE, _P.RIGHT_EAR),
)


@dataclass(frozen=True, slots=True)
class PoseSample:
    """Everything the capture process knows about one camera frame.

    Timestamps are ``time.monotonic_ns()`` values. On macOS and Linux the
    monotonic clock is system-wide, so the game process can compare them
    with its own clock to measure end-to-end latency.

    Attributes:
        frame_index: increasing camera frame counter (gaps = skipped frames).
        captured_ns: when the frame was read from the camera.
        processed_ns: when landmark detection finished.
        frame_size: ``(width, height)`` of the frame the landmarks refer to.
        pose_image / pose_world: ``(33, 5)`` arrays, or ``None`` when no
            person was detected.
        pose_ms: pose model inference time.
        dropped_samples: samples the capture process discarded so far
            because the game was not consuming them fast enough.
    """

    frame_index: int
    captured_ns: int
    processed_ns: int
    frame_size: tuple[int, int]
    pose_image: FloatArray | None
    pose_world: FloatArray | None
    pose_ms: float = 0.0
    dropped_samples: int = 0

    @property
    def has_pose(self) -> bool:
        return self.pose_image is not None

    @property
    def processing_ms(self) -> float:
        """Camera read → landmarks ready, in milliseconds."""
        return (self.processed_ns - self.captured_ns) / 1e6
