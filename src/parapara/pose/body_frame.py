"""MediaPipe world landmarks → arm directions in the dancer's body frame.

This is where camera-space landmarks become comparable with routine
keyframes, which are unit directions in the body frame of
:mod:`parapara.routine.geometry` (``+x`` dancer's right, ``+y`` up, ``+z``
forward).

MediaPipe world axes (measured on 2026-10-18 recordings: right arm up / T-pose
/ right arm pointing at the camera): ``+x`` = dancer's left, ``+y`` = down,
``+z`` = away from the camera; origin at the hip midpoint, metres.

The body frame is rebuilt **for every sample** from the torso, so leaning or
turning slightly does not change the measured arm directions:

- ``right`` = unit(right shoulder − left shoulder)
- ``up``    = unit(mid-shoulder − mid-hip), made orthogonal to ``right``
- ``forward`` = ``up × right``. MediaPipe's basis is physically right-handed,
  so a numeric cross product in its coordinates is the physical one; the body
  frame itself is physically left-handed (see ``geometry.py``), hence
  ``up × right`` rather than ``right × up``.

If the hips are not reliably visible (player close to the camera), ``up``
falls back to a given direction, by default MediaPipe's "up" (``-y``); the
player calibration (Step 3) will supply a measured one.

Hand direction comes from the pose model's coarse hand points: wrist → the
midpoint of the index and pinky knuckle landmarks.

Frame variants (:data:`FRAME_METHODS`), compared on the 2026-10-18
recordings with ``parapara-analyze`` (mean angle error per block clip):

- ``"torso"`` (default): left/right from the **shoulder** line, up from the
  hips. Best or near-best on 10 of 12 clips; essential for sweeps and flicks,
  where dancers twist their shoulders by ±26° (error 31° → 18°). Costs a few
  degrees on chops, because raising an arm also lifts that shoulder.
- ``"camera"``: no compensation (camera axes). Best on chops only.
- ``"yaw"``: camera vertical as up, shoulder line for turning only.
- ``"hips"``: left/right from the hip line. Fails on sweeps: shoulders twist
  relative to the hips.
- ``"blend"``: average of shoulder and hip lines.

The variants are kept so the comparison can be repeated on new recordings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from parapara.pose.landmarks import PoseLandmark as P

type Vectors = npt.NDArray[np.float64]  # (N, 3)

MEDIAPIPE_UP: tuple[float, float, float] = (0.0, -1.0, 0.0)
VISIBILITY_THRESHOLD = 0.5

_SIDES = {
    "right": (P.RIGHT_SHOULDER, P.RIGHT_ELBOW, P.RIGHT_WRIST, P.RIGHT_INDEX, P.RIGHT_PINKY),
    "left": (P.LEFT_SHOULDER, P.LEFT_ELBOW, P.LEFT_WRIST, P.LEFT_INDEX, P.LEFT_PINKY),
}


def _unit(v: Vectors) -> Vectors:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / norm  # zero-length -> NaN, flagged by callers' validity masks


@dataclass(frozen=True, slots=True)
class BodyFrames:
    """Per-sample body axes, expressed in MediaPipe world coordinates."""

    right: Vectors
    up: Vectors
    forward: Vectors

    def to_body(self, v: Vectors) -> Vectors:
        """Express MediaPipe-world vectors ``v`` (N, 3) in the body frame."""
        return np.stack(
            [(v * self.right).sum(-1), (v * self.up).sum(-1), (v * self.forward).sum(-1)], -1
        )


def body_frames(
    world: npt.ArrayLike,
    fallback_up: tuple[float, float, float] = MEDIAPIPE_UP,
    method: str = "torso",
) -> BodyFrames:
    """Body axes for each sample of ``world`` (N, 33, ≥3; column 3 = visibility).

    ``method`` selects a frame variant from :data:`FRAME_METHODS` (see the
    module docstring); the default ``"torso"`` is the one used for scoring.
    """
    if method not in FRAME_METHODS:
        raise ValueError(f"unknown frame method {method!r}, expected one of {FRAME_METHODS}")
    w = np.asarray(world, dtype=np.float64)
    n = len(w)
    rs, ls = w[:, P.RIGHT_SHOULDER, :3], w[:, P.LEFT_SHOULDER, :3]
    rh, lh = w[:, P.RIGHT_HIP, :3], w[:, P.LEFT_HIP, :3]
    vertical = np.tile(np.asarray(fallback_up, dtype=np.float64), (n, 1))
    if method == "camera":
        right = np.tile(np.array([-1.0, 0.0, 0.0]), (n, 1))
        return BodyFrames(right, vertical, np.cross(vertical, right))
    if method == "yaw":
        right = rs - ls
        right = _unit(right - (right * vertical).sum(-1, keepdims=True) * vertical)
        return BodyFrames(right, vertical, np.cross(vertical, right))
    shoulder_line, hip_line = _unit(rs - ls), _unit(rh - lh)
    right = {"torso": shoulder_line, "hips": hip_line,
             "blend": _unit(shoulder_line + hip_line)}[method]  # fmt: skip
    up = (rs + ls) / 2 - (rh + lh) / 2
    if w.shape[-1] > 3:
        hips_seen = np.minimum(w[:, P.RIGHT_HIP, 3], w[:, P.LEFT_HIP, 3]) >= VISIBILITY_THRESHOLD
        up = np.where(hips_seen[:, None], up, np.asarray(fallback_up, dtype=np.float64))
    up = _unit(up - (up * right).sum(-1, keepdims=True) * right)
    forward = np.cross(up, right)
    return BodyFrames(right, up, forward)


FRAME_METHODS = ("torso", "camera", "yaw", "hips", "blend")


def torso_motion(world: npt.ArrayLike) -> dict[str, npt.NDArray[np.float64]]:
    """Per-sample torso orientation relative to the camera, in degrees:

    - ``yaw``: turning (shoulder line rotating towards/away from the camera),
    - ``side_tilt``: leaning sideways (e.g. hip sway from stepping),
    - ``lean``: leaning towards the camera.
    """
    frames = body_frames(world)
    r, u = frames.right, frames.up
    return {
        "yaw": np.degrees(np.arctan2(r[:, 2], -r[:, 0])),
        "side_tilt": np.degrees(np.arctan2(u[:, 0], -u[:, 1])),
        "lean": np.degrees(np.arctan2(-u[:, 2], -u[:, 1])),
    }


@dataclass(frozen=True, slots=True)
class ArmDirections:
    """Unit directions of one arm's segments in the body frame, per sample.

    ``valid`` is False where a needed landmark was not visible enough (or the
    sample had no person); those rows are NaN.
    """

    upper: Vectors  # shoulder → elbow
    fore: Vectors  # elbow → wrist
    hand: Vectors  # wrist → index/pinky midpoint
    valid: npt.NDArray[np.bool_]


def arm_directions(world: npt.ArrayLike, frames: BodyFrames | None = None) -> dict[str, ArmDirections]:
    """Per-arm segment directions for ``world`` landmarks (N, 33, ≥3).

    Returns ``{"right": ArmDirections, "left": ArmDirections}`` for the
    dancer's anatomical arms.
    """
    w = np.asarray(world, dtype=np.float64)
    frames = frames or body_frames(w)
    out = {}
    for side, (s, e, wr, ix, pk) in _SIDES.items():
        shoulder, elbow, wrist = w[:, s, :3], w[:, e, :3], w[:, wr, :3]
        knuckles = (w[:, ix, :3] + w[:, pk, :3]) / 2
        valid = np.all(np.isfinite(w[:, [s, e, wr], :3]), axis=(1, 2))
        if w.shape[-1] > 3:
            valid &= np.min(w[:, [s, e, wr], 3], axis=1) >= VISIBILITY_THRESHOLD
        out[side] = ArmDirections(
            upper=_unit(frames.to_body(elbow - shoulder)),
            fore=_unit(frames.to_body(wrist - elbow)),
            hand=_unit(frames.to_body(knuckles - wrist)),
            valid=valid,
        )
    return out


def angle_deg(a: Vectors, b: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Row-wise angle in degrees between unit vectors ``a`` (N, 3) and ``b``."""
    dot = (np.asarray(a) * np.asarray(b, dtype=np.float64)).sum(-1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
