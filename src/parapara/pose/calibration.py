"""Per-player calibration from a neutral pose and a T-pose (PLAN.md §3).

Measured once per session, after the position check (``pose.framing``):

- **shoulder width** (metres, MediaPipe world units) and upper-arm / forearm
  lengths as ratios of it: they turn "wrists close together" (the clap gate)
  and other distances into body-relative quantities;
- **neutral torso up** (MediaPipe coordinates): the camera's view of
  "upright" for this player and camera angle. It replaces the vertical
  fallback when the hips are hidden, and posture is judged relative to it
  (a laptop camera looking up made the torso appear to lean 4-12° towards
  the camera in the 2026-10-18 recordings);
- **neutral head height** (nose above the shoulders, in shoulder widths):
  the reference for "head up".

Both poses are validated (arms hanging down / stretched sideways, person
well framed most of the time), so a bad calibration is rejected with a
clear message instead of silently skewing every score. Arm checks use the
**front plane** only (depth dropped): depth is MediaPipe's noisiest axis and
made hanging forearms read ~40° forward in the 2026-10-18 neutral pose.

Pure NumPy (no cv2/mediapipe/pygame).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

import numpy as np

from parapara.pose.body_frame import angle_deg, arm_directions, body_frames
from parapara.pose.framing import check_framing
from parapara.pose.landmarks import PoseLandmark as P
from parapara.pose.recording import Recording
from parapara.routine.geometry import DOWN, LEFT, RIGHT

FORMAT_VERSION = 1
MAX_POSE_ERROR_DEG = 35.0  # arms must be within this of the expected direction
MIN_FRAMED_SHARE = 0.5  # share of samples that must pass the framing check
MIN_SAMPLES = 10


class CalibrationError(ValueError):
    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


@dataclass(frozen=True)
class Calibration:
    shoulder_width_m: float
    upper_arm_ratio: float  # upper-arm length / shoulder width
    forearm_ratio: float
    neutral_up: tuple[float, float, float]  # torso up in MediaPipe world coords
    neutral_lean_deg: float  # apparent lean towards the camera when upright
    neutral_head_height: float  # nose above mid-shoulder, in shoulder widths
    neutral_arm_error_deg: float  # quality: forearms vs. straight down (front plane)
    t_pose_arm_error_deg: float  # quality: forearms vs. straight out (front plane)
    created: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"format_version": FORMAT_VERSION, **asdict(self)}, indent=2)
            + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        data = json.loads(path.read_text())
        if data.pop("format_version", None) != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported calibration format")
        data["neutral_up"] = tuple(data["neutral_up"])
        return cls(**data)


def head_height(world: np.ndarray, up: np.ndarray, shoulder_width: float) -> np.ndarray:
    """Nose height above the mid-shoulder point along ``up``, in shoulder widths."""
    mid = (world[:, P.RIGHT_SHOULDER, :3] + world[:, P.LEFT_SHOULDER, :3]) / 2
    return ((world[:, P.NOSE, :3] - mid) * up).sum(-1) / shoulder_width


def front_angle_deg(directions: np.ndarray, target) -> np.ndarray:
    """Row-wise angle to ``target`` after dropping depth (z) from both."""
    d = np.array(directions, dtype=np.float64)
    t = np.array(target, dtype=np.float64)
    d[..., 2] = 0.0
    t[2] = 0.0
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return angle_deg(d, t / np.linalg.norm(t))


def _usable(rec: Recording) -> tuple[np.ndarray, np.ndarray]:
    samples = rec.samples()
    return np.array([s.has_pose for s in samples]), np.array(
        [check_framing(s).ok for s in samples]
    )


def calibrate(neutral: Recording, t_pose: Recording) -> Calibration:
    """Compute a :class:`Calibration`; raises :class:`CalibrationError` with
    every problem found."""
    issues: list[str] = []
    for name, rec, expected in (("neutral pose", neutral, (DOWN, DOWN)),
                                ("T-pose", t_pose, (RIGHT, LEFT))):  # fmt: skip
        has, framed = _usable(rec)
        if has.sum() < MIN_SAMPLES:
            issues.append(f"{name}: no person found in enough samples")
            continue
        if framed[has].mean() < MIN_FRAMED_SHARE:
            issues.append(
                f"{name}: poorly framed (hips, arms or head out of the picture)"
            )
        arms = arm_directions(rec.pose_world[has])
        for side, target in zip(("right", "left"), expected, strict=True):
            err = float(np.nanmedian(front_angle_deg(arms[side].fore, target)))
            if err > MAX_POSE_ERROR_DEG:
                issues.append(
                    f"{name}: {side} arm is {err:.0f}° away from the expected pose"
                )
    if issues:
        raise CalibrationError(issues)

    n_world = neutral.pose_world[neutral.has_pose].astype(np.float64)
    t_world = t_pose.pose_world[t_pose.has_pose].astype(np.float64)
    both = np.concatenate([n_world, t_world])

    def length(w: np.ndarray, a: int, b: int) -> float:
        return float(np.nanmedian(np.linalg.norm(w[:, a, :3] - w[:, b, :3], axis=-1)))

    shoulder = length(both, P.RIGHT_SHOULDER, P.LEFT_SHOULDER)
    # Segment lengths from the T-pose: straight arms, least foreshortened.
    upper = np.mean([length(t_world, P.RIGHT_SHOULDER, P.RIGHT_ELBOW),
                     length(t_world, P.LEFT_SHOULDER, P.LEFT_ELBOW)])  # fmt: skip
    fore = np.mean([length(t_world, P.RIGHT_ELBOW, P.RIGHT_WRIST),
                    length(t_world, P.LEFT_ELBOW, P.LEFT_WRIST)])  # fmt: skip

    up = np.nanmedian(body_frames(n_world).up, axis=0)
    up /= np.linalg.norm(up)
    lean = float(np.degrees(np.arctan2(-up[2], -up[1])))
    n_arms, t_arms = arm_directions(n_world), arm_directions(t_world)

    def arm_error(arms, targets) -> float:
        return float(np.mean([np.nanmedian(front_angle_deg(arms[s].fore, t))
                              for s, t in zip(("right", "left"), targets, strict=True)]))  # fmt: skip

    neutral_up_t = (float(round(float(up[0]), 4)), float(round(float(up[1]), 4)), float(round(float(up[2]), 4)))
    return Calibration(
        shoulder_width_m=round(shoulder, 4),
        upper_arm_ratio=round(float(upper / shoulder), 3),
        forearm_ratio=round(float(fore / shoulder), 3),
        neutral_up=neutral_up_t,
        neutral_lean_deg=round(lean, 1),
        neutral_head_height=round(
            float(np.nanmedian(head_height(n_world, up, shoulder))), 3
        ),
        neutral_arm_error_deg=round(arm_error(n_arms, (DOWN, DOWN)), 1),
        t_pose_arm_error_deg=round(arm_error(t_arms, (RIGHT, LEFT)), 1),
        created=datetime.now().isoformat(timespec="seconds"),
    )


def calibrate_session(session: Path) -> Calibration:
    """Calibrate from ``neutral.npz`` and ``t_pose.npz`` in a recording session."""
    return calibrate(
        Recording.load(session / "neutral.npz"), Recording.load(session / "t_pose.npz")
    )
