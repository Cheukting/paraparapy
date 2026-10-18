"""Per-sample pose features for scoring (PLAN.md §5, steps 1-4).

:class:`FeatureExtractor` turns the stream of :class:`PoseSample` into
:class:`PoseFeatures`:

1. **Smooth** the world landmarks with a One Euro filter
   (:data:`LANDMARK_FILTER`, tuned on the 2026-10-18 recordings: forearm
   jitter at rest 1.19° → 0.47° per frame, ~28 ms added lag on fast moves).
2. Build the **body frame** from the smoothed torso
   (:mod:`parapara.pose.body_frame`, ``"torso"`` method), with the
   calibrated neutral "up" as fallback when the hips are hidden.
3. Smooth the body frame **more slowly** (:data:`TORSO_FILTER`): the torso
   turns slowly while arms move fast, and a raised arm lifting its shoulder
   should not swing the whole frame.
4. Extract features: arm segment directions in the body frame, the wrist
   distance in shoulder widths (clap gate), and posture (lean, side tilt,
   shoulder tilt, head height; relative to the calibrated neutral pose when a
   :class:`~parapara.pose.calibration.Calibration` is given).

Mirroring for the on-screen demo is a display concern: features always use
the dancer's anatomical left/right, the same convention as routine keyframes.

Pure NumPy (no cv2/mediapipe/pygame), so it runs in the game process.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from parapara.pose.body_frame import (
    MEDIAPIPE_UP,
    VISIBILITY_THRESHOLD,
    ArmDirections,
    BodyFrames,
    arm_directions,
    body_frames,
)
from parapara.pose.calibration import Calibration, head_height
from parapara.pose.filters import OneEuroFilter
from parapara.pose.landmarks import PoseLandmark as P
from parapara.pose.landmarks import PoseSample
from parapara.pose.recording import Recording

type Vec = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FilterParams:
    min_cutoff: float
    beta: float
    d_cutoff: float = 1.0

    def make(self) -> OneEuroFilter:
        return OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)


# Tuned on recordings/20261018-144829 (static clips for jitter, block clips
# for lag; see CHANGELOG). Units: metres and m/s.
LANDMARK_FILTER = FilterParams(min_cutoff=0.5, beta=10.0)
# Torso axes are unit vectors (speeds in 1/s); tuned on the same session.
# Adaptive on purpose: a fixed slow filter lagged the ±26° torso turns of
# sweeps (sweep error 18° -> 26-32°); with beta 0.5 it smooths a steady torso
# but follows turns (mean block error 19.8° raw -> 19.6°, chop 23° -> 21.4°).
TORSO_FILTER = FilterParams(min_cutoff=0.5, beta=0.5)


@dataclass(frozen=True, slots=True)
class ArmFeatures:
    upper: Vec  # (3,) unit direction, body frame
    fore: Vec
    hand: Vec
    valid: bool


@dataclass(frozen=True, slots=True)
class PoseFeatures:
    """Features of one sample (see module docstring).

    Posture angles are in degrees; with a calibration, ``lean_deg`` and
    ``head_height`` are relative to the player's neutral pose (0 = as upright
    as when calibrating; negative head height = head lower than neutral).
    """

    t_ns: int
    right: ArmFeatures
    left: ArmFeatures
    wrist_distance: float  # between the wrists, in shoulder widths
    lean_deg: float  # towards the camera
    side_tilt_deg: float  # torso tilting sideways (+ = towards the dancer's left)
    shoulder_tilt_deg: float  # shoulder line vs. horizontal (+ = right shoulder higher)
    head_height: float  # nose above shoulders, in shoulder widths
    hips_visible: bool


@dataclass
class FeatureExtractor:
    """Streaming feature extraction; one instance per player and session."""

    calibration: Calibration | None = None
    landmark_filter: FilterParams = LANDMARK_FILTER
    torso_filter: FilterParams = TORSO_FILTER
    _landmarks: OneEuroFilter = field(init=False)
    _torso: OneEuroFilter = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._landmarks = self.landmark_filter.make()
        self._torso = self.torso_filter.make()

    @property
    def _up_reference(self) -> tuple[float, float, float]:
        return self.calibration.neutral_up if self.calibration else MEDIAPIPE_UP

    def update(self, sample: PoseSample) -> PoseFeatures | None:
        """Features for ``sample``, or ``None`` if no person was detected
        (the filters restart when the person reappears)."""
        if sample.pose_world is None:
            self.reset()
            return None
        raw = np.asarray(sample.pose_world, dtype=np.float64)
        world = raw.copy()
        world[:, :3] = self._landmarks(raw[:, :3], sample.captured_ns)
        w = world[None]

        frames = body_frames(w, fallback_up=self._up_reference)
        axes = self._torso(np.stack([frames.right[0], frames.up[0]]), sample.captured_ns)
        right = axes[0] / np.linalg.norm(axes[0])
        up = axes[1] - (axes[1] @ right) * right
        up /= np.linalg.norm(up)
        smooth = BodyFrames(right[None], up[None], np.cross(up, right)[None])
        arms = arm_directions(w, smooth)
        return self._features(sample.captured_ns, world, smooth, arms)

    def _features(self, t_ns: int, world: np.ndarray, frames: BodyFrames,
                  arms: dict[str, ArmDirections]) -> PoseFeatures:  # fmt: skip
        rs, ls = world[P.RIGHT_SHOULDER, :3], world[P.LEFT_SHOULDER, :3]
        shoulder_w = (self.calibration.shoulder_width_m if self.calibration
                      else float(np.linalg.norm(rs - ls)))  # fmt: skip
        wrists = np.linalg.norm(world[P.RIGHT_WRIST, :3] - world[P.LEFT_WRIST, :3]) / shoulder_w

        up = frames.up[0]
        lean = float(np.degrees(np.arctan2(-up[2], -up[1])))
        side_tilt = float(np.degrees(np.arctan2(up[0], -up[1])))
        # Shoulder line in the camera's front plane (MediaPipe y points down).
        d = rs - ls  # towards the dancer's right
        shoulder_tilt = float(np.degrees(np.arctan2(-d[1], np.hypot(d[0], d[2]))))
        head = float(head_height(world[None], up, shoulder_w)[0])
        if self.calibration:
            lean -= self.calibration.neutral_lean_deg
            head -= self.calibration.neutral_head_height

        hips_visible = bool(min(world[P.RIGHT_HIP, 3], world[P.LEFT_HIP, 3]) >= VISIBILITY_THRESHOLD)

        def arm(side: str) -> ArmFeatures:
            a = arms[side]
            return ArmFeatures(a.upper[0], a.fore[0], a.hand[0], bool(a.valid[0]))

        return PoseFeatures(t_ns, arm("right"), arm("left"), float(wrists), lean, side_tilt,
                            shoulder_tilt, head, hips_visible)  # fmt: skip


def extract_recording(
    rec: Recording, extractor: FeatureExtractor | None = None
) -> list[PoseFeatures | None]:
    """Run a fresh (or the given) extractor over a recorded clip, in order."""
    extractor = extractor or FeatureExtractor()
    return [extractor.update(s) for s in rec.samples()]


def as_arm_directions(features: list[PoseFeatures | None]) -> dict[str, ArmDirections]:
    """Stack per-sample features into :class:`ArmDirections` arrays (NaN where
    no features), the input format of the clip analysis."""
    nan = np.full(3, np.nan)
    out = {}
    for side in ("right", "left"):
        arms = [getattr(f, side) if f is not None else None for f in features]
        out[side] = ArmDirections(
            upper=np.stack([a.upper if a else nan for a in arms]),
            fore=np.stack([a.fore if a else nan for a in arms]),
            hand=np.stack([a.hand if a else nan for a in arms]),
            valid=np.array([bool(a and a.valid) for a in arms]),
        )
    return out
