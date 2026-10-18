"""Tests for parapara.pose.calibration and parapara.pose.features."""

import numpy as np
import pytest
from pose_synth import image_pose, recording, world_pose

from parapara.pose.calibration import Calibration, CalibrationError, calibrate
from parapara.pose.features import FeatureExtractor, as_arm_directions, extract_recording
from parapara.pose.landmarks import PoseSample
from parapara.routine.geometry import DOWN, LEFT, RIGHT, angle_between_deg
from parapara.routine.model import ArmPose

T_ARMS = (ArmPose(upper=RIGHT, fore=RIGHT), ArmPose(upper=LEFT, fore=LEFT))
MS = 1_000_000


def _clip(right=None, left=None, n=20, lean=0.0, noise=0.0, seed=0, **image_kw):
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        p = world_pose(right, left, lean_deg=lean)
        p[:, :3] += rng.normal(0, noise, size=(33, 3))
        poses.append(p)
    images = [image_pose(**image_kw)] * n
    return recording(poses, [i * 43 * MS for i in range(n)], {"kind": "static"}, images)


# --- calibration ---------------------------------------------------------------


def test_calibration_from_neutral_and_t_pose(tmp_path) -> None:
    cal = calibrate(_clip(lean=8), _clip(*T_ARMS, lean=8))
    assert cal.shoulder_width_m == pytest.approx(0.35, abs=1e-3)
    assert cal.upper_arm_ratio == pytest.approx(0.22 / 0.35, abs=0.01)
    assert cal.forearm_ratio == pytest.approx(0.22 / 0.35, abs=0.01)
    assert cal.neutral_lean_deg == pytest.approx(8, abs=0.1)
    assert cal.neutral_head_height == pytest.approx(0.18 / 0.35, abs=0.01)
    assert cal.neutral_arm_error_deg < 1 and cal.t_pose_arm_error_deg < 1
    path = tmp_path / "calibration.json"
    cal.save(path)
    assert Calibration.load(path) == cal


def test_calibration_rejects_wrong_poses() -> None:
    with pytest.raises(CalibrationError) as err:
        calibrate(_clip(*T_ARMS), _clip())  # poses swapped
    text = " ".join(err.value.issues)
    assert "neutral pose: right arm" in text and "T-pose: left arm" in text


def test_calibration_rejects_poor_framing() -> None:
    with pytest.raises(CalibrationError, match="poorly framed"):
        calibrate(_clip(hip_y=1.05), _clip(*T_ARMS))


def test_calibration_ignores_depth_bias_of_hanging_forearms() -> None:
    """Recorded: hanging forearms read ~40° towards the camera (depth bias).
    The arm check uses the front plane, so this neutral pose is accepted."""
    biased = ArmPose(upper=DOWN, fore=(0.1, -0.8, 0.6))
    cal = calibrate(_clip(biased, biased.mirrored()), _clip(*T_ARMS))
    assert cal.neutral_arm_error_deg < 10


# --- features ------------------------------------------------------------------


def test_features_of_a_t_pose() -> None:
    feats = [f for f in extract_recording(_clip(*T_ARMS)) if f is not None]
    last = feats[-1]
    assert angle_between_deg(tuple(last.right.fore), RIGHT) < 1e-3
    assert angle_between_deg(tuple(last.left.upper), LEFT) < 1e-3
    assert last.right.valid and last.hips_visible
    # Wrists ~ shoulder width + 2 * (upper + fore) apart.
    assert last.wrist_distance == pytest.approx((0.35 + 4 * 0.22) / 0.35, abs=0.01)
    assert abs(last.lean_deg) < 0.1 and abs(last.shoulder_tilt_deg) < 0.1


def test_filters_reduce_frame_to_frame_jitter() -> None:
    rec = _clip(*T_ARMS, n=60, noise=0.01)
    smooth = as_arm_directions(extract_recording(rec))["right"].fore
    from parapara.pose.body_frame import arm_directions

    raw = arm_directions(rec.pose_world)["right"].fore

    def jitter(d):
        return np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(-1), -1, 1)))[10:].mean()

    assert jitter(smooth) < jitter(raw) / 2


def test_posture_relative_to_calibration() -> None:
    cal = calibrate(_clip(lean=10), _clip(*T_ARMS, lean=10))
    extractor = FeatureExtractor(calibration=cal)
    upright = extract_recording(_clip(lean=10), extractor)[-1]
    assert abs(upright.lean_deg) < 0.5 and abs(upright.head_height) < 0.01
    extractor.reset()
    leaning = extract_recording(_clip(lean=25), extractor)[-1]
    assert leaning.lean_deg == pytest.approx(15, abs=0.5)


def test_no_person_resets_and_returns_none() -> None:
    extractor = FeatureExtractor()
    assert extractor.update(PoseSample(0, 0, 0, (640, 480), None, None)) is None
    p = world_pose(*T_ARMS)
    feats = extractor.update(PoseSample(1, 40 * MS, 0, (640, 480), p, p))
    assert feats is not None and feats.right.valid


def test_as_arm_directions_marks_missing_samples() -> None:
    p = world_pose(*T_ARMS)
    extractor = FeatureExtractor()
    feats = [extractor.update(PoseSample(0, 0, 0, (640, 480), p, p)), None]
    arms = as_arm_directions(feats)
    assert arms["right"].valid.tolist() == [True, False]
    assert np.isnan(arms["right"].fore[1]).all()


def test_analysis_saves_calibration_and_tuning_runs(tmp_path) -> None:
    from parapara.diagnostics import filter_tuning, pose_analysis

    session = tmp_path / "20990101-000000"
    session.mkdir()
    (session / "session.json").write_text("{}")
    _clip(n=12).save(session / "neutral.npz")
    _clip(*T_ARMS, n=12).save(session / "t_pose.npz")
    report = pose_analysis.analyze_session(session)
    assert report["calibration"]["shoulder_width_m"] == pytest.approx(0.35, abs=1e-3)
    assert Calibration.load(session / "calibration.json").shoulder_width_m == pytest.approx(0.35, abs=1e-3)

    tuning = filter_tuning.tune(session)  # static clips only: jitter, no lag
    assert tuning["raw_jitter_deg"] == 0.0
    assert all("jitter_deg" in row and "lag_ms" not in row for row in tuning["landmarks"])


def test_analysis_reports_rejected_calibration(tmp_path) -> None:
    from parapara.diagnostics import pose_analysis

    session = tmp_path / "s"
    session.mkdir()
    _clip(*T_ARMS, n=12).save(session / "neutral.npz")  # wrong pose
    _clip(*T_ARMS, n=12).save(session / "t_pose.npz")
    report = pose_analysis.analyze_session(session)
    assert any("neutral pose" in issue for issue in report["calibration"]["rejected"])
    assert not (session / "calibration.json").exists()
