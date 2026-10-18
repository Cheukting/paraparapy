"""Tests for the pygame-free overlay helpers, plus a headless render of the
debug view's drawing functions in a subprocess (pygame must not be imported
into the pytest process, see conftest.py)."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from parapara.game.overlay import CaptureStats, RateMeter, RollingMean, to_screen, visible
from parapara.pose.landmarks import PoseSample


def test_to_screen_scales_and_mirrors() -> None:
    pts = [(0.0, 0.0), (0.25, 0.5), (1.0, 1.0)]
    assert to_screen(pts, (640, 480), mirrored=False) == [(0, 0), (160, 240), (640, 480)]
    assert to_screen(pts, (640, 480), mirrored=True) == [(640, 0), (480, 240), (0, 480)]


def test_to_screen_ignores_extra_columns_and_does_not_modify_input() -> None:
    pts = np.array([[0.5, 0.5, 0.1, 0.9, 0.9]], np.float32)
    assert to_screen(pts, (100, 100), mirrored=True) == [(50, 50)]
    assert pts[0, 0] == 0.5


def test_visible_threshold_and_nan() -> None:
    pose = np.zeros((3, 5), np.float32)
    pose[:, 3] = [0.9, 0.2, np.nan]
    assert [visible(pose, i) for i in range(3)] == [True, False, False]


def test_rate_meter() -> None:
    meter = RateMeter(window_s=1.0)
    assert meter.rate == 0.0
    for i in range(31):  # 30 intervals of 1/30 s
        meter.tick(int(i * 1e9 / 30))
    assert meter.rate == pytest.approx(30.0)
    meter.tick(int(5e9))  # long pause: old ticks fall out of the window
    assert meter.rate == 0.0


def test_rolling_mean() -> None:
    mean = RollingMean(size=2)
    assert mean.mean == 0.0
    for v in (1.0, 3.0, 5.0):
        mean.add(v)
    assert mean.mean == pytest.approx(4.0)


def test_capture_stats_hud() -> None:
    stats = CaptureStats.new()
    pose = np.zeros((33, 5), np.float32)
    for i in range(4):
        stats.add_sample(
            PoseSample(
                frame_index=i, captured_ns=i * 33_000_000, processed_ns=i * 33_000_000 + 20_000_000,
                frame_size=(640, 480), pose_image=pose if i % 2 else None, pose_world=None,
                pose_ms=12.0, hands_ms=6.0, dropped_samples=i,
            )
        )  # fmt: skip
    text = "\n".join(stats.hud_lines())
    assert "person found  50%" in text
    assert "12.0 ms" in text and "6.0 ms" in text
    assert "camera→landmarks  20.0 ms" in text
    assert "dropped samples  3" in text


RENDER_SCRIPT = textwrap.dedent(
    """
    import numpy as np, pygame
    from parapara.game import debug_view as dv
    from parapara.pose.landmarks import HandObservation, PoseLandmark as P, PoseSample

    pose = np.zeros((33, 5), np.float32)
    pose[:, 3:] = 1.0
    pose[P.LEFT_SHOULDER, :2] = (0.6, 0.3); pose[P.LEFT_WRIST, :2] = (0.8, 0.5)
    pose[P.RIGHT_SHOULDER, :2] = (0.4, 0.3); pose[P.RIGHT_WRIST, :2] = (0.2, 0.5)
    hand = np.zeros((21, 3), np.float32); hand[:, :2] = (0.8, 0.5)
    sample = PoseSample(0, 0, 0, (200, 100), pose, pose,
                        left_hand=HandObservation(hand, hand, 1.0, "x"))
    pygame.init()
    for mirrored, x in ((False, 160), (True, 40)):
        s = pygame.Surface((200, 100)); s.fill((0, 0, 0))
        dv.draw_pose(s, sample, mirrored)
        dv.draw_hud(s, ["hello"])
        print(mirrored, tuple(s.get_at((x, 50)))[:3] == dv.LEFT_COLOUR)
    """
)


def test_debug_view_draws_left_wrist_on_correct_side_headless() -> None:
    env = os.environ | {"SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
    out = subprocess.run(
        [sys.executable, "-c", RENDER_SCRIPT], capture_output=True, text=True, env=env, check=True
    ).stdout.split("\n")
    # Unmirrored: the dancer's left wrist appears on the right of the image
    # (x=160); mirrored it appears on the left (x=40). Drawn in LEFT_COLOUR.
    assert out[:2] == ["False True", "True True"]


SMOKE_SCRIPT = textwrap.dedent(
    """
    from parapara.game import debug_view
    debug_view.main(["--width", "64", "--height", "48"],
                    worker="fake_capture_worker:run", max_frames=40)
    print("ok")
    """
)


def test_debug_view_main_loop_runs_headless_with_fake_worker() -> None:
    """End-to-end: window loop + CaptureClient + spawned (fake) worker."""
    tests_dir = os.path.dirname(__file__)
    env = os.environ | {
        "SDL_VIDEODRIVER": "dummy",
        "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, [tests_dir, os.environ.get("PYTHONPATH")])),
    }
    result = subprocess.run(
        [sys.executable, "-c", SMOKE_SCRIPT], capture_output=True, text=True, env=env, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    assert "SDL_" not in result.stderr  # no duplicate-SDL warning: cv2 never loaded
