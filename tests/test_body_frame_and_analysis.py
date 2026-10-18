"""Tests for parapara.pose.body_frame and parapara.diagnostics.pose_analysis,
using synthetic skeletons from tests/pose_synth.py."""

import json

import numpy as np
import pytest
from pose_synth import recording, world_pose

from parapara.diagnostics import pose_analysis
from parapara.pose.body_frame import (
    FRAME_METHODS,
    angle_deg,
    arm_directions,
    body_frames,
    torso_motion,
)
from parapara.routine.geometry import DOWN, FORWARD, LEFT, RIGHT, UP, angle_between_deg, slerp
from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import ArmPose

T_POSE = (ArmPose(upper=RIGHT, fore=RIGHT), ArmPose(upper=LEFT, fore=LEFT))


# --- body frame --------------------------------------------------------------


def test_mediapipe_axes_map_to_body_frame() -> None:
    """Measured on 2026-10-18: MediaPipe +x = dancer's left, +y = down,
    +z = away from the camera. Body frame: right, up, forward."""
    frames = body_frames(world_pose()[None])
    np.testing.assert_allclose(frames.right[0], [-1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(frames.up[0], [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(frames.forward[0], [0, 0, -1], atol=1e-6)


@pytest.mark.parametrize(
    ("right", "expected_upper", "expected_fore"),
    [
        (ArmPose(upper=RIGHT, fore=RIGHT), RIGHT, RIGHT),
        (ArmPose(upper=UP, fore=UP), UP, UP),
        (ArmPose(upper=FORWARD, fore=FORWARD), FORWARD, FORWARD),
        (ArmPose(upper=RIGHT, fore=UP), RIGHT, UP),  # knock-like right angle
        (None, DOWN, DOWN),
    ],
)
def test_arm_directions_recovered(right, expected_upper, expected_fore) -> None:
    arms = arm_directions(world_pose(right=right)[None])
    np.testing.assert_allclose(arms["right"].upper[0], expected_upper, atol=1e-6)
    np.testing.assert_allclose(arms["right"].fore[0], expected_fore, atol=1e-6)
    assert arms["right"].valid[0]


@pytest.mark.parametrize(("lean", "turn"), [(12, 0), (0, 20), (8, -15)])
def test_directions_are_relative_to_the_body(lean, turn) -> None:
    """Leaning towards the camera or turning must not change arm directions:
    the frame is rebuilt from the torso every sample."""
    chop = BLOCK_LIBRARY["chop_right"].keyframes[1].right
    arms = arm_directions(world_pose(right=chop, lean_deg=lean, turn_deg=turn)[None])
    assert angle_deg(arms["right"].upper, chop.upper)[0] < 1e-4
    assert angle_deg(arms["right"].fore, chop.fore)[0] < 1e-4


def test_left_arm_and_hand_direction() -> None:
    left = ArmPose(upper=LEFT, fore=UP, hand=(-1, 1, 0))
    arms = arm_directions(world_pose(left=left)[None])
    np.testing.assert_allclose(arms["left"].fore[0], UP, atol=1e-6)
    np.testing.assert_allclose(arms["left"].hand[0], np.array([-1, 1, 0]) / np.sqrt(2), atol=1e-6)


def test_hidden_hips_fall_back_to_vertical_up() -> None:
    pose = world_pose(lean_deg=30, hips_visible=False)  # hips unreliable
    frames = body_frames(pose[None])
    np.testing.assert_allclose(frames.up[0], [0, -1, 0], atol=1e-6)


def test_invisible_arm_is_invalid_and_missing_pose_is_nan() -> None:
    pose = world_pose()
    pose[14, 3] = 0.1  # right elbow barely visible
    missing = np.full_like(pose, np.nan)
    arms = arm_directions(np.stack([pose, missing]))
    assert not arms["right"].valid[0] and arms["left"].valid[0]
    assert not arms["left"].valid[1] and np.isnan(arms["left"].upper[1]).all()


@pytest.mark.parametrize("method", FRAME_METHODS)
def test_every_frame_method_is_exact_for_an_upright_dancer(method) -> None:
    chop = BLOCK_LIBRARY["chop_right"].keyframes[1].right
    world = world_pose(right=chop)[None]
    arms = arm_directions(world, body_frames(world, method=method))
    assert angle_deg(arms["right"].fore, chop.fore)[0] < 1e-4


def test_turning_is_compensated_by_body_relative_frames_only() -> None:
    """A 25° turn (as in the recorded sweeps) changes the camera-relative
    direction of a level arm by exactly 25°, but none of the body-relative
    ones. (A level arm is perpendicular to the turning axis; other directions
    change by less.)"""
    arm = ArmPose(upper=RIGHT, fore=RIGHT)
    world = world_pose(right=arm, turn_deg=25)[None]
    errors = {m: angle_deg(arm_directions(world, body_frames(world, method=m))["right"].fore,
                           arm.fore)[0] for m in FRAME_METHODS}  # fmt: skip
    assert errors["camera"] == pytest.approx(25, abs=0.5)
    for m in ("torso", "yaw", "hips", "blend"):
        assert errors[m] < 1e-4, m


def test_unknown_frame_method() -> None:
    with pytest.raises(ValueError, match="frame method"):
        body_frames(world_pose()[None], method="gravity")


def test_torso_motion_measures_turn_and_lean() -> None:
    world = np.stack([world_pose(turn_deg=20), world_pose(lean_deg=10)])
    motion = torso_motion(world)
    assert motion["yaw"] == pytest.approx([20, 0], abs=0.1)
    assert motion["lean"] == pytest.approx([0, 10], abs=0.1)
    assert motion["side_tilt"] == pytest.approx([0, 0], abs=0.1)


# --- analysis: static ------------------------------------------------------------


def test_analyze_static_t_pose() -> None:
    poses = [world_pose(*T_POSE, lean_deg=10) for _ in range(5)]
    rec = recording(poses, [i * 40_000_000 for i in range(5)], {"clip_id": "t_pose", "kind": "static"})
    report = pose_analysis.analyze_static(rec)
    assert report.arms["right"]["upper"] == [1.0, 0.0, 0.0]
    assert report.arms["left"]["fore"] == [-1.0, 0.0, 0.0]
    assert report.shoulder_width_m == pytest.approx(0.35, abs=1e-3)
    assert report.upper_arm_ratio == pytest.approx(0.22 / 0.35, abs=0.01)
    assert report.torso_lean_deg == pytest.approx(10, abs=0.1)
    assert report.wrist_jitter_m["right"] == [0.0, 0.0, 0.0]


# --- analysis: blocks ------------------------------------------------------------


def _danced_block(block_id: str, lag_s: float, bpm: float = 120, fps: float = 50, bars: int = 2):
    """A recording of a dancer moving smoothly through ``block_id``: each arm
    segment turns at constant speed from one keyframe's direction to the
    next, arriving exactly ``lag_s`` after each keyframe's beat."""
    block = BLOCK_LIBRARY[block_id]
    beat_s = 60 / bpm
    count_in = 4
    first_beat_ns = 1_000_000_000
    beat_times = [first_beat_ns + int((i - count_in) * beat_s * 1e9) for i in range(count_in + bars * 8)]
    times = np.arange(0, bars * 8 * beat_s, 1 / fps)
    kfs = block.keyframes

    def arm(kf, side):
        return getattr(kf, side) or ArmPose(upper=DOWN, fore=DOWN)

    poses = []
    for t in times:
        position = ((t - lag_s) / beat_s) % block.duration
        k = max(i for i, kf in enumerate(kfs) if kf.beat <= position + 1e-9) if position >= kfs[0].beat else len(kfs) - 1
        a, nxt = kfs[k], kfs[(k + 1) % len(kfs)]
        span = (nxt.beat - a.beat) % block.duration or block.duration
        frac = ((position - a.beat) % block.duration) / span
        both = {}
        for side in ("right", "left"):
            pa, pb = arm(a, side), arm(nxt, side)
            both[side] = ArmPose(upper=slerp_safe(pa.upper, pb.upper, frac),
                                 fore=slerp_safe(pa.fore, pb.fore, frac))  # fmt: skip
        poses.append(world_pose(both["right"], both["left"]))
    meta = {"clip_id": f"block_{block_id}", "kind": "block", "block_id": block_id, "bpm": bpm,
            "count_in_beats": count_in, "beat_times_ns": beat_times}  # fmt: skip
    return recording(poses, [first_beat_ns + int(t * 1e9) for t in times], meta)


def slerp_safe(a, b, t):
    """Interpolate directions; nearly opposite pairs go via a mid direction."""
    if angle_between_deg(a, b) > 170:
        return slerp(a, (0.0, 0.0, 1.0), t)
    return slerp(a, b, t)


@pytest.mark.parametrize("block_id", ["chop_right", "sweep_to_right_4", "flick_clap_to_right"])
def test_analyze_block_finds_lag_and_perfect_fit(block_id) -> None:
    report = pose_analysis.analyze_block(_danced_block(block_id, lag_s=0.3))
    assert report.best_lag_ms == pytest.approx(300, abs=25)
    assert report.mean_error_deg < 3
    for fit in report.keyframes:
        assert fit.error_upper_deg <= 3 and fit.error_fore_deg <= 3


@pytest.mark.parametrize("block_id", ["knock_right", "wrist_roll_4"])
def test_lag_not_measurable_when_pose_does_not_change(block_id) -> None:
    """Knocks hit the same pose twice and the wrist roll holds one position:
    every lag fits equally well, so no lag is reported."""
    report = pose_analysis.analyze_block(_danced_block(block_id, lag_s=0.3))
    assert report.best_lag_ms is None
    assert report.mean_error_deg < 1


def test_analyze_block_reports_a_wrongly_danced_pose() -> None:
    """Danced as the old (pre-calibration) chop salute: the fit shows the
    difference, which is how the library gets recalibrated."""
    rec = _danced_block("chop_right", lag_s=0.0, fps=100)
    salute_target = BLOCK_LIBRARY["chop_right"].keyframes[1].right
    wrong = ArmPose(upper=(0.5, -0.6, 0.6), fore=(-1, 0.1, 0.2))
    beat_s = 0.5
    t = (rec.captured_ns - rec.metadata["beat_times_ns"][4]) / 1e9
    for i in np.where((t % (2 * beat_s)) >= beat_s)[0]:  # every count 2
        rec.pose_world[i] = world_pose(right=wrong)
    report = pose_analysis.analyze_block(rec)
    salute = next(f for f in report.keyframes if f.beat == 1)
    assert salute.seen_upper == pytest.approx(list(np.round(np.array(wrong.upper), 2)), abs=0.01)
    assert salute.error_upper_deg == pytest.approx(
        float(angle_deg(np.array(wrong.upper), salute_target.upper)), abs=1
    )


def test_analyze_session_cli_writes_report(tmp_path, capsys) -> None:
    session = tmp_path / "20990101-000000"
    session.mkdir()
    (session / "session.json").write_text("{}")
    _danced_block("knock_right", 0.1).save(session / "block_knock_right.npz")
    poses = [world_pose(*T_POSE)] * 3
    recording(poses, [0, 40_000_000, 80_000_000], {"clip_id": "t_pose", "kind": "static"}).save(
        session / "t_pose.npz"
    )
    report = pose_analysis.run([str(session)])
    assert set(report["static"]) == {"t_pose"} and set(report["blocks"]) == {"block_knock_right"}
    saved = json.loads((session / "analysis.json").read_text())
    block = saved["blocks"]["block_knock_right"]
    assert block["block_id"] == "knock_right"
    assert set(block["frame_comparison"]) == set(FRAME_METHODS)
    assert block["torso_motion"]["yaw"] == {"mean": 0.0, "std": 0.0}
    assert "== knock_right" in capsys.readouterr().out
    assert pose_analysis.latest_session(tmp_path) == session
