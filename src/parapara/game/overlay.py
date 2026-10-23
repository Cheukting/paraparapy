"""Pure helpers for drawing landmark overlays and the guide figure.

HUD statistics live in :mod:`parapara.diagnostics.live_stats`.

Kept free of ``pygame`` so they can be unit-tested in the main pytest process
(which also imports ``cv2`` for the capture tests; the two must never be
loaded together, see the package docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from parapara.routine.model import ArmPose

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


def visible(
    pose: npt.NDArray[np.float32], index: int, threshold: float = VISIBILITY_THRESHOLD
) -> bool:
    """Whether a pose landmark is confidently visible (NaN counts as not)."""
    v = float(pose[index, 3])
    return v == v and v >= threshold  # v == v is False for NaN


# --- guide figure ------------------------------------------------------------

# Stick-figure proportions in units of shoulder width (rough adult averages).
GUIDE_SHOULDER_HALF = 0.5
GUIDE_UPPER_ARM = 0.75
GUIDE_FOREARM = 0.7
GUIDE_HAND = 0.25
GUIDE_TORSO = 1.3
GUIDE_HEAD_RADIUS = 0.28


@dataclass(frozen=True, slots=True)
class GuideArm:
    shoulder: tuple[int, int]
    elbow: tuple[int, int]
    wrist: tuple[int, int]
    hand: tuple[int, int]
    scored: bool  # False: arm not part of the move, drawn hanging and dimmed


@dataclass(frozen=True, slots=True)
class GuideFigure:
    head: tuple[int, int]
    head_radius: int
    neck: tuple[int, int]
    hip: tuple[int, int]
    right: GuideArm
    left: GuideArm


def guide_figure(
    right: ArmPose | None, left: ArmPose | None, centre: tuple[int, int], scale: float
) -> GuideFigure:
    """Front-view stick figure of a target pose.

    Body-frame directions (``+x`` dancer's right, ``+y`` up) are projected by
    dropping ``z``, with the dancer's right drawn on the **right of the
    screen**, matching the mirrored camera view: the player copies the figure
    like a mirror (PLAN.md §7). Arms pointing at the camera therefore look
    short; the instructions say so.

    Args:
        right, left: target arm poses; ``None`` draws the arm hanging down
            and marks it unscored.
        centre: screen position of the mid-shoulder point.
        scale: pixels per shoulder width.
    """
    cx, cy = centre

    def px(x: float, y: float) -> tuple[int, int]:
        return (round(cx + x * scale), round(cy - y * scale))

    def arm(pose: ArmPose | None, side: float) -> GuideArm:
        scored = pose is not None
        upper, fore = (
            (pose.upper, pose.fore) if pose is not None else ((0, -1, 0), (0, -1, 0))
        )
        hand = pose.hand if pose is not None and pose.hand is not None else fore
        sx, sy = side * GUIDE_SHOULDER_HALF, 0.0
        ex, ey = sx + upper[0] * GUIDE_UPPER_ARM, sy + upper[1] * GUIDE_UPPER_ARM
        wx, wy = ex + fore[0] * GUIDE_FOREARM, ey + fore[1] * GUIDE_FOREARM
        hx, hy = wx + hand[0] * GUIDE_HAND, wy + hand[1] * GUIDE_HAND
        return GuideArm(px(sx, sy), px(ex, ey), px(wx, wy), px(hx, hy), scored)

    return GuideFigure(
        head=px(0, GUIDE_HEAD_RADIUS + 0.12),
        head_radius=max(1, round(GUIDE_HEAD_RADIUS * scale)),
        neck=px(0, 0),
        hip=px(0, -GUIDE_TORSO),
        right=arm(right, +1.0),
        left=arm(left, -1.0),
    )
