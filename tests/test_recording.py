"""Tests for landmark recordings, the recording script/timeline, the guide
figure projection, and a headless run of the guided recorder."""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from parapara.diagnostics.clips import CLIP_IDS, CLIPS, ClipTimeline
from parapara.game.overlay import guide_figure
from parapara.pose.landmarks import NUM_POSE_LANDMARKS, PoseSample
from parapara.pose.recording import Recording
from parapara.routine.geometry import FORWARD, RIGHT, UP
from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import ArmPose

# --- Recording -------------------------------------------------------------------


def _samples(n: int) -> list[PoseSample]:
    out = []
    for i in range(n):
        pose = None if i == 1 else np.full((NUM_POSE_LANDMARKS, 5), i, np.float32)
        out.append(PoseSample(i * 2, i * 1_000_000_000, i * 1_000_000_000 + 5, (640, 480),
                              pose, pose, pose_ms=3.0))  # fmt: skip
    return out


def test_recording_round_trip(tmp_path) -> None:
    rec = Recording.from_samples(_samples(3), {"clip_id": "t_pose", "beat_times_ns": [1, 2]})
    assert len(rec) == 3 and rec.metadata["frame_size"] == [640, 480]
    assert rec.person_found_pct == pytest.approx(200 / 3)
    assert rec.duration_s == pytest.approx(2.0)
    assert np.isnan(rec.pose_world[1]).all()  # no person -> NaN, has_pose False

    path = tmp_path / "t_pose.npz"
    rec.save(path)
    loaded = Recording.load(path)
    assert loaded.metadata == rec.metadata
    np.testing.assert_array_equal(loaded.captured_ns, rec.captured_ns)
    np.testing.assert_array_equal(loaded.pose_image, rec.pose_image)

    samples = loaded.samples()
    assert [s.has_pose for s in samples] == [True, False, True]
    assert samples[2].pose_world[0, 0] == 2.0 and samples[2].frame_size == (640, 480)


def test_empty_recording(tmp_path) -> None:
    rec = Recording.from_samples([], {"frame_size": [640, 480]})
    assert len(rec) == 0 and rec.pose_image.shape == (0, 33, 5)
    rec.save(tmp_path / "e.npz")
    assert len(Recording.load(tmp_path / "e.npz")) == 0


def test_recording_rejects_unknown_version(tmp_path) -> None:
    path = tmp_path / "x.npz"
    Recording.from_samples(_samples(1), {}).save(path)
    with np.load(path) as data:
        arrays = dict(data)
    arrays["metadata"] = np.array(json.dumps({"format_version": 99}))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="format_version"):
        Recording.load(path)


# --- clip script & timeline ------------------------------------------------------


def test_clip_script() -> None:
    ids = [c.id for c in CLIPS]
    assert len(ids) == len(set(ids)) and set(ids) == set(CLIP_IDS)
    assert ids[:2] == ["neutral", "t_pose"]  # calibration first
    for clip in CLIPS:
        if clip.kind == "block":
            assert clip.block_id in BLOCK_LIBRARY and clip.guide is None
        else:
            assert clip.guide is not None and clip.block is None
    blocks = {c.block.kind for c in CLIPS if c.kind == "block"}
    assert len(blocks) == 5  # every building block is recorded


def test_axis_clips_point_along_the_axes() -> None:
    assert CLIP_IDS["right_arm_up"].guide.right.fore == UP
    assert CLIP_IDS["right_arm_forward"].guide.right.fore == FORWARD
    assert CLIP_IDS["t_pose"].guide.right.fore == RIGHT


def test_static_timeline() -> None:
    tl = ClipTimeline(CLIP_IDS["t_pose"], countdown_s=3, static_seconds=4)
    assert (tl.recording_start, tl.recording_end) == (3, 7) and tl.beats() == []
    assert [tl.phase_at(t).label for t in (0, 1.5, 2.9)] == ["3", "2", "1"]
    rec = tl.phase_at(5)
    assert rec.name == "recording" and rec.label == "REC" and rec.progress == pytest.approx(0.5)
    assert rec.guide is CLIP_IDS["t_pose"].guide
    assert tl.phase_at(7).name == "done"


def test_block_timeline_count_in_beats_and_keyframes() -> None:
    tl = ClipTimeline(CLIP_IDS["block_chop_right"], bpm=120)  # 0.5 s per beat
    assert tl.recording_start == 2.0 and tl.recording_end == 10.0  # 4 + 16 beats
    beats = tl.beats()
    assert len(beats) == 20 and beats[1] == (0.5, False)
    assert [i for i, (_, accent) in enumerate(beats) if accent] == [0, 4, 12]
    assert [tl.phase_at(t).label for t in (0, 0.6, 1.1, 1.6)] == ["4", "3", "2", "1"]

    out, salute = BLOCK_LIBRARY["chop_right"].keyframes
    p1, p2, p3 = tl.phase_at(2.1), tl.phase_at(2.6), tl.phase_at(3.1)
    assert (p1.bar, p1.count, p1.guide) == (1, 1, out)  # count 1: arm out
    assert (p2.count, p2.guide) == (2, salute)  # count 2: salute
    assert (p3.count, p3.guide) == (3, out)  # block repeats
    p_bar2 = tl.phase_at(6.1)
    assert (p_bar2.bar, p_bar2.count) == (2, 1)


def test_block_timeline_half_beat_keyframes() -> None:
    tl = ClipTimeline(CLIP_IDS["block_wrist_roll_4"], bpm=120)
    kfs = BLOCK_LIBRARY["wrist_roll_4"].keyframes
    assert tl.phase_at(2.0 + 0.25).guide is kfs[1]  # 0.5 counts in


# --- guide figure ----------------------------------------------------------------


def test_guide_figure_projection() -> None:
    fig = guide_figure(ArmPose(upper=UP, fore=UP), None, centre=(100, 100), scale=100)
    # Dancer's right is drawn on the right of the screen; up is up.
    assert fig.right.shoulder == (150, 100)
    assert fig.right.elbow == (150, 25) and fig.right.wrist == (150, -45)
    assert fig.right.scored and not fig.left.scored
    assert fig.left.wrist[1] > fig.left.shoulder[1]  # unscored arm hangs down
    assert fig.hip[1] > fig.neck[1] > fig.head[1]


def test_guide_figure_uses_hand_direction() -> None:
    pose = ArmPose(upper=RIGHT, fore=RIGHT, hand=UP)
    fig = guide_figure(pose, None, (0, 0), 100)
    assert fig.right.hand[0] == fig.right.wrist[0] and fig.right.hand[1] < fig.right.wrist[1]


# --- headless recorder run ---------------------------------------------------------

RECORD_SCRIPT = textwrap.dedent(
    """
    import sys
    from parapara.game import recorder
    recorder.main(["--auto", "--out", sys.argv[1], "--bpm", "960", "--countdown", "0.2",
                   "--static-seconds", "0.4", "--width", "64", "--height", "48",
                   "--clips", "t_pose,block_knock_right"], worker="fake_capture_worker:stream")
    """
)


def test_recorder_records_and_saves_clips_headless(tmp_path) -> None:
    tests_dir = os.path.dirname(__file__)
    env = os.environ | {
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
        "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, [tests_dir, os.environ.get("PYTHONPATH")])),
    }
    result = subprocess.run(
        [sys.executable, "-c", RECORD_SCRIPT, str(tmp_path)],
        capture_output=True, text=True, env=env, timeout=90,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "SDL_" not in result.stderr
    (session,) = tmp_path.iterdir()
    manifest = json.loads((session / "session.json").read_text())
    assert list(manifest["clips"]) == ["t_pose", "block_knock_right"]

    static = Recording.load(session / "t_pose.npz")
    assert len(static) > 5 and static.person_found_pct == 100.0
    start, end = static.metadata["recording_window_ns"]
    assert start <= static.captured_ns.min() and static.captured_ns.max() < end

    block = Recording.load(session / "block_knock_right.npz")
    assert block.metadata["block_id"] == "knock_right" and block.metadata["bpm"] == 960
    assert len(block.metadata["beat_times_ns"]) == 4 + 16  # count-in + 2 bars
    assert len(block) > 0
