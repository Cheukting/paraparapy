"""How close is a pose to a keyframe? (pose accuracy)

Angular error between the player's arm segments (from
:class:`parapara.pose.features.PoseFeatures`) and the keyframe's target
directions, **weighted towards the front plane**: the depth (``z``)
component of both vectors is scaled by :data:`DEPTH_WEIGHT` before the angle
is measured, because depth is MediaPipe's least reliable axis (wrist jitter
12-59 mm vs 4-17 mm in x/y, and a consistent bias towards the camera, in the
2026-10-18 recordings). ``DEPTH_WEIGHT = 1`` is plain 3-D, ``0`` front-only.

Error → score: full credit up to the block's tolerance (25° sharp, 35°
flowing, 40° for the chop reach), falling linearly to zero at twice the
tolerance. Forearms count more than upper arms (they define the visible
shape); the hand direction counts a little when the keyframe sets one.
"""

from __future__ import annotations

import numpy as np

from parapara.pose.features import ArmFeatures, PoseFeatures
from parapara.routine.model import ArmPose, Keyframe

DEPTH_WEIGHT = 0.5
SEGMENT_WEIGHTS = {"upper": 0.35, "fore": 0.5, "hand": 0.15}
CLAP_DISTANCE = 0.6  # wrists closer than this many shoulder widths = hands together


def weighted_angle_deg(a: np.ndarray, b: tuple[float, float, float] | np.ndarray,
                       depth_weight: float = DEPTH_WEIGHT) -> float:  # fmt: skip
    """Angle between ``a`` and ``b`` after scaling their z by ``depth_weight``."""
    scale = np.array([1.0, 1.0, depth_weight])
    u, v = np.asarray(a, float) * scale, np.asarray(b, float) * scale
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 90.0  # a pure-depth direction with depth ignored: undecidable
    return float(np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0))))


def arm_error_deg(
    arm: ArmFeatures, target: ArmPose, depth_weight: float = DEPTH_WEIGHT
) -> float | None:
    """Weighted mean segment error of one arm, or ``None`` if not visible."""
    if not arm.valid or not np.all(np.isfinite(arm.fore)):
        return None
    parts = [("upper", target.upper, arm.upper), ("fore", target.fore, arm.fore)]
    if target.hand is not None and np.all(np.isfinite(arm.hand)):
        parts.append(("hand", target.hand, arm.hand))
    total = sum(SEGMENT_WEIGHTS[name] for name, _, _ in parts)
    return sum(SEGMENT_WEIGHTS[name] * weighted_angle_deg(seen, want, depth_weight)
               for name, want, seen in parts) / total  # fmt: skip


def keyframe_error_deg(features: PoseFeatures, keyframe: Keyframe,
                       depth_weight: float = DEPTH_WEIGHT) -> float:  # fmt: skip
    """Mean error over the keyframe's scored arms; an invisible scored arm
    counts as 180°. Clap keyframes are judged by :func:`clap_score`."""
    errors = []
    for side in ("right", "left"):
        target = getattr(keyframe, side)
        if target is None:
            continue
        err = arm_error_deg(getattr(features, side), target, depth_weight)
        errors.append(180.0 if err is None else err)
    return float(np.mean(errors)) if errors else 0.0


def pose_score(error_deg: float, tolerance_deg: float) -> float:
    """1.0 within tolerance, linear to 0.0 at twice the tolerance."""
    return float(np.clip((2 * tolerance_deg - error_deg) / tolerance_deg, 0.0, 1.0))


def clap_score(features: PoseFeatures) -> float:
    """1.0 when the wrists are together, linear to 0 at twice the distance."""
    return pose_score(features.wrist_distance, CLAP_DISTANCE)
