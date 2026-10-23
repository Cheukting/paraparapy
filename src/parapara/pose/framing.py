"""Is the player well placed in the camera picture? (framing check)

Used before calibration (neutral pose / T-pose) and, later, before a game, so
that every reference measurement is taken with the whole torso in view and
enough room to stretch the arms in every direction the routines use.

Checks, on the normalized image landmarks of one sample (unmirrored camera
image, so the player's *left* appears on the *right* of the picture):

1. **person found**, shoulders and hips visible (the body frame needs both);
2. **not too far**: shoulders span at least ``MIN_SHOULDER_WIDTH`` of the
   frame width, otherwise landmarks get noisy;
3. **arm room**: a fully stretched arm (``ARM_REACH`` shoulder widths, the
   wrist-from-shoulder distance measured in the 2026-10-18 T-pose was 1.26)
   fits on both sides and above the shoulders, with a ``MARGIN``;
4. **centred** horizontally (within ``CENTRE_TOLERANCE``);
5. **facing the camera**: the shoulder line turned by less than
   ``MAX_YAW_DEG`` (needs world landmarks).

Messages talk about the player's **own** left/right, which is independent of
whether the screen shows a mirrored picture.

Pure NumPy (no cv2/mediapipe/pygame).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from parapara.pose.body_frame import torso_motion
from parapara.pose.landmarks import PoseLandmark as P
from parapara.pose.landmarks import PoseSample

MARGIN = 0.03  # fraction of the frame kept free at the edges
MIN_SHOULDER_WIDTH = 0.10  # of frame width (recorded: 0.19)
ARM_REACH = 1.3  # shoulder → stretched wrist, in shoulder widths (recorded: 1.26)
CENTRE_TOLERANCE = 0.12  # allowed offset of the shoulder midpoint from centre
MAX_YAW_DEG = 20.0
VISIBLE = 0.5


@dataclass(frozen=True, slots=True)
class FramingIssue:
    code: str  # stable identifier, e.g. "too_far"
    message: str  # instruction for the player


@dataclass(frozen=True, slots=True)
class FramingResult:
    """Outcome of :func:`check_framing`.

    ``needed_box`` is the picture area the player needs (normalized image
    ``x0, y0, x1, y1``: arm span left/right, raised hand above, hips below),
    for drawing a guide rectangle; ``None`` when no person is found.
    """

    issues: tuple[FramingIssue, ...]
    needed_box: tuple[float, float, float, float] | None = None
    shoulder_width: float = 0.0  # of frame width

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def codes(self) -> set[str]:
        return {i.code for i in self.issues}


def _visible(pose: npt.NDArray[np.float32], *indices: int) -> bool:
    v = pose[list(indices), 3]
    return bool(np.all(np.nan_to_num(v, nan=0.0) >= VISIBLE))


def check_framing(sample: PoseSample) -> FramingResult:
    """Check one sample; issues are ordered by what to fix first."""
    pose = sample.pose_image
    if pose is None:
        return FramingResult(
            (FramingIssue("no_person", "Step into the picture, facing the camera."),)
        )
    if not _visible(pose, P.LEFT_SHOULDER, P.RIGHT_SHOULDER):
        return FramingResult(
            (
                FramingIssue(
                    "no_shoulders", "Face the camera with your shoulders in view."
                ),
            )
        )

    width, height = sample.frame_size
    aspect = width / height  # converts frame-width fractions to frame-height fractions
    rs, ls = (
        pose[P.RIGHT_SHOULDER, :2].astype(float),
        pose[P.LEFT_SHOULDER, :2].astype(float),
    )
    mid_x, shoulder_y = (rs[0] + ls[0]) / 2, (rs[1] + ls[1]) / 2
    shoulder_w = float(
        np.hypot((ls[0] - rs[0]), (ls[1] - rs[1]) / aspect)
    )  # in frame widths
    reach_x = shoulder_w / 2 + ARM_REACH * shoulder_w
    reach_up = ARM_REACH * shoulder_w * aspect  # same length in frame heights

    hips_seen = _visible(pose, P.LEFT_HIP, P.RIGHT_HIP)
    hip_y = float(pose[[P.LEFT_HIP, P.RIGHT_HIP], 1].mean()) if hips_seen else 1.0
    box = (mid_x - reach_x, shoulder_y - reach_up, mid_x + reach_x, hip_y)

    issues: list[FramingIssue] = []
    room_left = box[0] >= MARGIN  # picture's left = the player's right
    room_right = box[2] <= 1 - MARGIN
    room_up = box[1] >= MARGIN
    hips_in = hips_seen and hip_y <= 1 - MARGIN

    if not hips_in:
        issues.append(
            FramingIssue("no_hips", "Step back until your hips are in the picture.")
        )
    if shoulder_w < MIN_SHOULDER_WIDTH:
        issues.append(FramingIssue("too_far", "Come a little closer to the camera."))
    elif (not room_left and not room_right) or (
        not room_up and (not room_left or not room_right)
    ):
        issues.append(
            FramingIssue("too_close", "Step back: leave room to stretch your arms.")
        )
    else:
        offset = mid_x - 0.5
        if not room_left or offset < -CENTRE_TOLERANCE:
            issues.append(FramingIssue("move_left", "Move a little to your LEFT."))
        elif not room_right or offset > CENTRE_TOLERANCE:
            issues.append(FramingIssue("move_right", "Move a little to your RIGHT."))
        if not room_up:
            issues.append(
                FramingIssue(
                    "no_headroom",
                    "Step back: leave room above your head for raised arms.",
                )
            )

    if sample.pose_world is not None:
        yaw = float(torso_motion(sample.pose_world[None])["yaw"][0])
        if abs(yaw) > MAX_YAW_DEG:
            issues.append(FramingIssue("turned", "Turn to face the camera squarely."))
    return FramingResult(tuple(issues), box, shoulder_w)


@dataclass
class FramingMonitor:
    """Debounces framing over recent samples so prompts don't flicker.

    ``stable`` is True once the last ``window`` checks were all OK (about
    0.4 s at 23 fps); ``latest`` is the most recent result for display.
    """

    window: int = 10
    latest: FramingResult | None = None
    _recent: deque[bool] = field(default_factory=deque)

    def update(self, sample: PoseSample) -> FramingResult:
        self.latest = check_framing(sample)
        self._recent.append(self.latest.ok)
        while len(self._recent) > self.window:
            self._recent.popleft()
        return self.latest

    @property
    def stable(self) -> bool:
        return len(self._recent) == self.window and all(self._recent)

    def reset(self) -> None:
        self._recent.clear()
        self.latest = None
