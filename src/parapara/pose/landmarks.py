"""Landmark data shared by the capture and game processes (pure NumPy).

The capture process converts MediaPipe results into the plain dataclasses
below and sends them to the game process through a ``multiprocessing.Queue``
(they pickle cheaply). Nothing here imports ``mediapipe`` or ``cv2``, so the
game process can use these types without loading OpenCV's bundled SDL2.

Landmark layouts follow MediaPipe's Pose Landmarker (33 points) and Hand
Landmarker (21 points) models, as documented at
https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
(checked 2026-10-18). Index names are duplicated here, rather than imported
from ``mediapipe``, for the same import-boundary reason.

Array conventions
-----------------
- Pose arrays have shape ``(33, 5)`` with columns ``x, y, z, visibility,
  presence``. Image landmarks have ``x``/``y`` normalized to ``[0, 1]`` by the
  frame width/height (they may slightly exceed that range for off-screen
  joints); world landmarks are metres with the hip midpoint as origin.
- Hand arrays have shape ``(21, 3)`` with columns ``x, y, z``.
- Frames are **not** mirrored: "left" means the person's anatomical left
  (MediaPipe pose labels), which appears on the *right* side of an
  unmirrored webcam image. To be confirmed in the real-dancer test run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
NUM_HAND_LANDMARKS = 21
POSE_COLUMNS = ("x", "y", "z", "visibility", "presence")

_P = PoseLandmark

# Only the upper body is scored (PLAN.md §1.1), so only it is drawn.
UPPER_BODY_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (_P.LEFT_SHOULDER, _P.RIGHT_SHOULDER),
    (_P.LEFT_SHOULDER, _P.LEFT_ELBOW),
    (_P.LEFT_ELBOW, _P.LEFT_WRIST),
    (_P.RIGHT_SHOULDER, _P.RIGHT_ELBOW),
    (_P.RIGHT_ELBOW, _P.RIGHT_WRIST),
    (_P.LEFT_SHOULDER, _P.LEFT_HIP),
    (_P.RIGHT_SHOULDER, _P.RIGHT_HIP),
    (_P.LEFT_HIP, _P.RIGHT_HIP),
    (_P.LEFT_EAR, _P.LEFT_EYE),
    (_P.LEFT_EYE, _P.NOSE),
    (_P.NOSE, _P.RIGHT_EYE),
    (_P.RIGHT_EYE, _P.RIGHT_EAR),
)

# Standard 21-point hand topology: wrist (0), then 4 joints per finger.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class HandObservation:
    """One detected hand.

    Attributes:
        image: ``(21, 3)`` normalized image landmarks; index 0 is the wrist.
        world: ``(21, 3)`` metres, origin at the hand's geometric centre.
        score: handedness confidence from MediaPipe.
        label: MediaPipe's raw handedness label ("Left"/"Right"). Kept for
            debugging only; left/right assignment uses the pose wrists
            instead (see :func:`assign_hands_to_wrists`).
    """

    image: FloatArray
    world: FloatArray
    score: float
    label: str


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
        left_hand / right_hand: the dancer's anatomical hands, or ``None``.
        pose_ms / hands_ms: inference time of each model.
        dropped_samples: samples the capture process discarded so far
            because the game was not consuming them fast enough.
    """

    frame_index: int
    captured_ns: int
    processed_ns: int
    frame_size: tuple[int, int]
    pose_image: FloatArray | None
    pose_world: FloatArray | None
    left_hand: HandObservation | None = None
    right_hand: HandObservation | None = None
    pose_ms: float = 0.0
    hands_ms: float = 0.0
    dropped_samples: int = field(default=0)

    @property
    def has_pose(self) -> bool:
        return self.pose_image is not None

    @property
    def processing_ms(self) -> float:
        """Camera read → landmarks ready, in milliseconds."""
        return (self.processed_ns - self.captured_ns) / 1e6


def assign_hands_to_wrists(
    hands: list[HandObservation],
    pose_image: FloatArray | None,
    frame_size: tuple[int, int],
    max_distance: float = 0.2,
) -> tuple[HandObservation | None, HandObservation | None]:
    """Decide which detected hand is the dancer's left and which the right.

    MediaPipe's handedness label depends on whether the input image is
    mirrored, which is easy to get wrong, so instead each hand's wrist
    (hand landmark 0) is matched to the nearest **pose** wrist. With two hands
    the assignment minimizing the total distance wins.

    Distances are measured in units of frame *height* (x is rescaled by the
    aspect ratio so both axes use the same unit). A hand further than
    ``max_distance`` from its wrist is rejected as a false detection.

    Returns:
        ``(left_hand, right_hand)``; either may be ``None``. Without a pose
        no assignment is possible and both are ``None``.
    """
    if pose_image is None or not hands:
        return None, None

    width, height = frame_size
    aspect = width / height

    def point(xy: npt.ArrayLike) -> np.ndarray:
        x, y = np.asarray(xy, dtype=np.float64)[:2]
        return np.array([x * aspect, y])

    wrists = {
        "left": point(pose_image[PoseLandmark.LEFT_WRIST]),
        "right": point(pose_image[PoseLandmark.RIGHT_WRIST]),
    }
    candidates = hands[:2]

    def dist(hand: HandObservation, side: str) -> float:
        return float(np.linalg.norm(point(hand.image[0]) - wrists[side]))

    if len(candidates) == 1:
        hand = candidates[0]
        side = min(wrists, key=lambda s: dist(hand, s))
        assignment = {side: hand}
    else:
        a, b = candidates
        straight = dist(a, "left") + dist(b, "right")
        swapped = dist(a, "right") + dist(b, "left")
        assignment = {"left": a, "right": b} if straight <= swapped else {"left": b, "right": a}

    accepted = {s: h for s, h in assignment.items() if dist(h, s) <= max_distance}
    return accepted.get("left"), accepted.get("right")
