"""Messages exchanged between the game and capture processes.

The capture worker sends two kinds of objects over the result queue:

- :class:`~parapara.pose.landmarks.PoseSample` for every processed frame;
- :class:`CaptureStatus` for lifecycle events (starting, ready, error,
  stopped), so the game can show progress such as "downloading models" or a
  camera-permission error instead of silently showing nothing.

:class:`CaptureConfig` travels the other way, once, when the worker starts.
All types are small frozen dataclasses so they pickle cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

type PoseModel = Literal["lite", "full"]


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Settings for the capture worker.

    Attributes:
        camera_index: OpenCV camera index (0 = default webcam).
        width, height: size of the frames delivered to the game; camera
            frames of another size are resized to this.
        fps: frame rate requested from the camera (the camera may ignore it).
        pose_model: Pose Landmarker bundle. ``"full"`` is more accurate,
            ``"lite"`` is faster. See ``parapara.capture.models``.
        model_dir: where to cache downloaded models; ``None`` uses the
            platform default (see ``parapara.capture.models.default_model_dir``).
    """

    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    pose_model: PoseModel = "full"
    model_dir: str | None = None

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """NumPy shape of one RGB frame: ``(height, width, 3)``."""
        return (self.height, self.width, 3)


class StatusKind(StrEnum):
    STARTING = "starting"  # loading / downloading models, opening the camera
    READY = "ready"  # first frames are flowing
    ERROR = "error"  # fatal; the worker exits after sending this
    STOPPED = "stopped"  # clean shutdown


@dataclass(frozen=True, slots=True)
class CaptureStatus:
    """A lifecycle event from the capture worker.

    ``details`` carries extra facts, e.g. the camera's actual resolution and
    frame rate on ``READY``.
    """

    kind: StatusKind
    message: str = ""
    details: dict[str, str | int | float] = field(default_factory=dict)
