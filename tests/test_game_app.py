"""Headless run of the game app (`parapara-play`) with a streaming fake camera:
menu skipped (--auto), song played for a few seconds, results saved."""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from parapara.audio import importer
from parapara.audio.synth import render_demo_song, write_wav
from parapara.pose.recording import Recording

TESTS = Path(__file__).resolve().parent


def accept_privacy(data_dir: Path) -> None:
    """The privacy notice was accepted (it is shown before the camera starts)."""
    from parapara.notices import PRIVACY_VERSION

    data_dir.mkdir(parents=True, exist_ok=True)
    ui = data_dir / "ui.json"
    prefs = json.loads(ui.read_text()) if ui.exists() else {}
    ui.write_text(json.dumps(prefs | {"privacy_notice": PRIVACY_VERSION}))


def _run_game(tmp_path, args: list[str], privacy_accepted: bool = True,
              worker: str = "fake_capture_worker:stream") -> subprocess.CompletedProcess:
    if privacy_accepted:
        accept_privacy(tmp_path / "data")
    if not (tmp_path / "lib").exists():
        track = render_demo_song()
        write_wav(tmp_path / "t.wav", track.samples, track.sample_rate)
        importer.import_song(tmp_path / "t.wav", importer.SongInfo("Test"), library=tmp_path / "lib")
    script = f"from parapara.game import app; app.run({args!r}, worker={worker!r})"
    env = os.environ | {
        "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        "PARAPARA_SONGS_DIR": str(tmp_path / "lib"), "PARAPARA_DATA_DIR": str(tmp_path / "data"),
        "PYTHONPATH": os.pathsep.join(filter(None, [str(TESTS), os.environ.get("PYTHONPATH")])),
    }  # fmt: skip
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            cwd=tmp_path, env=env, timeout=60)  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "SDL_" not in result.stderr and "Traceback" not in result.stderr, result.stderr
    return result


def test_game_plays_a_song_and_saves_results(tmp_path) -> None:
    _run_game(tmp_path, ["--auto", "--auto-seconds", "6", "--count-in", "0"])
    (log,) = glob.glob(str(tmp_path / "logs" / "game-*.json"))
    summary = json.loads(Path(log).read_text())["result"]
    assert summary["song_id"] == "test" and summary["keyframes"] > 0
    assert sum(summary["grades"].values()) == summary["keyframes"]  # every keyframe judged
    (play,) = glob.glob(str(tmp_path / "recordings" / "game-*" / "play.npz"))
    rec = Recording.load(Path(play))
    assert len(rec) > 50 and rec.metadata["kind"] == "play"  # camera samples were recorded


def test_menu_and_body_calibration_flow(tmp_path) -> None:
    """Menu -> body calibration -> (the fake camera holds arms down, so the
    T-pose check fails and the issues are shown) -> menu -> quit."""
    _run_game(tmp_path, ["--fast", "--keys", "10:return 300:escape 320:escape"])
    assert not (tmp_path / "data" / "calibration.json").exists()  # rejected, not saved


def test_latency_calibration_then_play_and_results_flow(tmp_path) -> None:
    """With a saved body calibration reused (--reuse-calibration): menu ->
    latency (no claps from the fake camera: rejected, skipped with S) -> play
    -> end early -> results -> menu."""
    from parapara.pose.calibration import Calibration

    (tmp_path / "data").mkdir(exist_ok=True)
    cal = Calibration(0.35, 0.62, 0.63, (0.0, -1.0, 0.0), 5.0, 0.5, 3.0, 4.0)
    cal.save(tmp_path / "data" / "calibration.json")
    # The latency clicks + evaluation take ~2.5 s (~150 frames): S is sent twice
    # with slack (S is ignored once playing).
    _run_game(tmp_path, ["--fast", "--reuse-calibration", "--keys",
                         "10:return 20:space 240:s 300:s 460:escape 490:return 510:escape"])
    (log,) = glob.glob(str(tmp_path / "logs" / "game-*.json"))
    result = json.loads(Path(log).read_text())["result"]
    assert result["ended_early"] and result["calibrated"] and result["latency_ms"] == 0.0
    assert not (tmp_path / "data" / "latency.json").exists()  # skipped, nothing saved


def test_text_size_keys_change_and_save_the_scale(tmp_path) -> None:
    """+ / - resize the text from the dancing spot; the choice is kept in
    ui.json and clamped to UI.MIN..UI.MAX (the first test run: text too small)."""
    _run_game(tmp_path, ["--fast", "--keys", "10:= 12:= 14:- 30:escape"])
    assert json.loads((tmp_path / "data" / "ui.json").read_text())["text_scale"] == 2.2
    _run_game(tmp_path, ["--fast", "--text-scale", "2.6", "--keys", "10:= 30:escape"])
    assert json.loads((tmp_path / "data" / "ui.json").read_text())["text_scale"] == 2.6  # clamped


def test_learn_the_moves_flow(tmp_path) -> None:
    """Menu -> T (learn the moves) -> next move, faster, previous move (wrapping
    round to the last one) -> Esc back to the menu -> quit; nothing is scored
    or saved."""
    _run_game(tmp_path, ["--fast", "--keys", "10:t 40:right 70:up 90:left 92:left 120:escape 140:escape"])
    assert not (tmp_path / "logs").exists()


def _save_calibration(tmp_path) -> None:
    from parapara.pose.calibration import Calibration

    (tmp_path / "data").mkdir(exist_ok=True)
    Calibration(0.35, 0.62, 0.63, (0.0, -1.0, 0.0), 5.0, 0.5, 3.0, 4.0).save(tmp_path / "data" / "calibration.json")


def test_calibrates_at_every_launch_but_not_between_songs(tmp_path) -> None:
    """Saved calibrations are not reused at launch: Enter goes to the body
    calibration even though one is saved (third test run)."""
    _save_calibration(tmp_path)
    (tmp_path / "data" / "latency.json").write_text('{"latency_ms": 230.0, "spread_ms": 10, "claps": 16}')
    # Body calibration fails with the fake camera (arms never in a T-pose), so
    # no game is played and no log is written.
    _run_game(tmp_path, ["--fast", "--keys", "10:return 300:escape 320:escape"])
    assert not (tmp_path / "logs").exists()


def test_latency_skip_keeps_the_value_and_songs_in_a_row_need_no_calibration(tmp_path) -> None:
    """L (redo latency) then S keeps the current latency; after the first song
    the second one starts straight away, without any calibration screen."""
    _save_calibration(tmp_path)
    (tmp_path / "data" / "latency.json").write_text('{"latency_ms": 230.0, "spread_ms": 10, "claps": 16}')
    keys = "10:l 12:return 30:s 120:escape 150:return 170:return 260:escape 290:return 310:escape"
    _run_game(tmp_path, ["--fast", "--reuse-calibration", "--count-in", "0", "--keys", keys])
    logs = sorted(glob.glob(str(tmp_path / "logs" / "game-*.json")))
    assert len(logs) == 2  # two songs; the second needed no calibration
    for log in logs:
        assert json.loads(Path(log).read_text())["result"]["latency_ms"] == 230.0


def test_layout_checker_reports_overlaps(tmp_path) -> None:
    """layout_issues finds overlapping text, text crossing a box and cut-off
    text (and accepts text inside a box), so the layout test below can trust it."""
    script = """
import pygame
from parapara.game.app import Drawn, layout_issues
R, screen = pygame.Rect, pygame.Rect(0, 0, 100, 100)
ok = [Drawn(R(0, 0, 10, 10), "a", screen), Drawn(R(0, 12, 10, 10), "b", screen),
      Drawn(R(50, 50, 40, 40), "[box]", screen, box=True), Drawn(R(55, 55, 10, 10), "in", screen)]
assert layout_issues(ok) == [], layout_issues(ok)
bad = [Drawn(R(0, 0, 10, 10), "a", screen), Drawn(R(5, 5, 10, 10), "b", screen),
       Drawn(R(50, 50, 40, 40), "[box]", screen, box=True), Drawn(R(45, 55, 10, 10), "edge", screen),
       Drawn(R(95, 0, 10, 10), "off", screen)]
issues = layout_issues(bad)
assert any("overlap" in i for i in issues) and any("crosses [box]" in i for i in issues)
assert any("cut off: 'off'" in i for i in issues), issues
"""
    env = os.environ | {"SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("scale", ["1.0", "2.0", "2.6"])
def test_no_overlapping_text_on_any_screen(tmp_path, scale) -> None:
    """Every screen at the smallest, default and largest text size, with a
    song list longer than the screen and a long title: no text overlaps or is
    cut off (--check-layout prints LAYOUT lines otherwise)."""
    track = render_demo_song()
    write_wav(tmp_path / "t.wav", track.samples, track.sample_rate)
    for title in ["Test", "A Very Long Song Title That Needs Wrapping On The Menu", *[f"Song {i}" for i in range(8)]]:
        importer.import_song(tmp_path / "t.wav", importer.SongInfo(title), library=tmp_path / "lib")
    check = ["--check-layout", "--text-scale", scale]
    runs = [
        # body calibration (all steps, rejected T-pose); menu scrolled to a long title
        ["--fast", "--keys", "5:down 10:up 15:return 300:escape 310:up 330:escape"],
        # latency (rejected) -> play with the count-in -> results -> learn the moves
        ["--fast", "--reuse-calibration", "--keys",
         "10:return 20:space 240:s 300:s 460:escape 490:return 510:t 540:right 560:up 580:escape 600:escape"],
        # play with grades and the next-move box
        ["--auto", "--auto-seconds", "9", "--count-in", "4"],
    ]
    _save_calibration(tmp_path)
    for args in runs:
        result = _run_game(tmp_path, args + check)
        assert "LAYOUT" not in result.stderr, result.stderr


def test_branding_logo_and_icon(tmp_path) -> None:
    """The ParaParaPy wordmark is the logo's artwork (transparent margin
    cropped, aspect kept); the icon is round (transparent corners, opaque
    centre); a missing logo file gives None (the game falls back to text)."""
    script = """
import pygame
from pathlib import Path
from parapara.game import branding
pygame.init(); pygame.display.set_mode((1, 1))
mark = branding.wordmark(120)
assert mark is not None and mark.get_height() == 120
assert abs(mark.get_width() / 120 - 1877 / 613) < 0.02, mark.get_size()
icon = branding.icon(64)
assert icon.get_size() == (64, 64)
assert icon.get_at((0, 0)).a == 0 and icon.get_at((63, 63)).a == 0 and icon.get_at((32, 32)).a > 240
assert branding.wordmark(50, Path("missing.png")) is None and branding.icon(64, Path("missing.png")) is None
"""
    env = os.environ | {"SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env,
                            cwd=tmp_path, timeout=60)  # fmt: skip
    assert result.returncode == 0, result.stderr


def test_latency_test_tool_runs_hands_free_and_logs(tmp_path) -> None:
    """parapara-latency-test: position check, rounds alternating 8 and 16
    claps (no claps from the fake camera: every round is "not accepted"),
    a summary, closes by itself; every round is logged with the click times,
    both estimates and the wrist-distance series; no overlapping text."""
    script = ("from parapara.game import latency_check\n"
              "latency_check.run(['--fast', '--bpm', '600', '--rounds', '3', '--check-layout'],"
              " worker='fake_capture_worker:stream')\n")  # fmt: skip
    env = os.environ | {
        "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        "PARAPARA_DATA_DIR": str(tmp_path / "data"),
        "PYTHONPATH": os.pathsep.join(filter(None, [str(TESTS), os.environ.get("PYTHONPATH")])),
    }  # fmt: skip
    accept_privacy(tmp_path / "data")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            cwd=tmp_path, env=env, timeout=90)  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "LAYOUT" not in result.stderr and "Traceback" not in result.stderr, result.stderr
    (log,) = glob.glob(str(tmp_path / "logs" / "latency-test-*.json"))
    data = json.loads(Path(log).read_text())
    rounds = data["rounds"]
    assert [r["claps_requested"] for r in rounds] == [8, 16, 8]
    assert all(len(r["click_ms"]) == r["claps_requested"] and len(r["count_in_ms"]) == 4 for r in rounds)
    assert all(r["result"]["ok"] is False and r["result_smoothed"] is not None for r in rounds)
    assert all(len(r["series"]["t_ms"]) == len(r["series"]["wrist_distance_timing"]) > 0 for r in rounds)
    assert data["summary"]["8"]["rounds"] == 2 and data["summary"]["16"]["accepted"] == 0


def test_privacy_notice_comes_before_the_camera(tmp_path) -> None:
    """First start: the privacy notice is shown before the camera starts.
    Esc quits (nothing accepted, nothing played); Enter accepts, remembers it
    and goes on to the menu; the next start skips the notice."""
    _run_game(tmp_path, ["--keys", "10:escape"], privacy_accepted=False)
    assert not (tmp_path / "data" / "ui.json").exists()
    _run_game(tmp_path, ["--check-layout", "--keys", "10:down 12:return 40:escape"], privacy_accepted=False)
    from parapara.notices import PRIVACY_VERSION

    assert json.loads((tmp_path / "data" / "ui.json").read_text())["privacy_notice"] == PRIVACY_VERSION
    result = _run_game(tmp_path, ["--keys", "10:escape"], privacy_accepted=False)  # straight to the menu
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("scale", ["1.0", "2.6"])
def test_credits_and_privacy_screens(tmp_path, scale) -> None:
    """Menu -> C (credits; scroll down and up) -> back -> P (privacy) -> back
    -> quit, with no overlapping or cut-off text."""
    keys = "10:c 30:down 40:down 50:up 70:escape 90:p 110:down 130:escape 150:escape"
    result = _run_game(tmp_path, ["--check-layout", "--text-scale", scale, "--keys", keys])
    assert "LAYOUT" not in result.stderr, result.stderr


def test_low_light_hint_shows_and_is_logged(tmp_path) -> None:
    """A camera delivering ~14 fps (dim light): the "more light helps" hint
    appears on the camera view without overlapping the other text, and the
    play's log records the camera rate and how long the hint was shown."""
    result = _run_game(tmp_path, ["--auto", "--auto-seconds", "8", "--count-in", "0", "--check-layout",
                                  "--text-scale", "2.6"], worker="fake_capture_worker:slow_stream")  # fmt: skip
    assert "LAYOUT" not in result.stderr, result.stderr
    (log,) = glob.glob(str(tmp_path / "logs" / "game-*.json"))
    summary = json.loads(Path(log).read_text())["result"]
    assert summary["camera_fps"] < 20 and summary["low_light_pct"] > 20
    normal = _run_game(tmp_path / "ok", ["--auto", "--auto-seconds", "4", "--count-in", "0"])
    (log,) = glob.glob(str(tmp_path / "ok" / "logs" / "game-*.json"))
    assert json.loads(Path(log).read_text())["result"]["low_light_pct"] == 0.0 and normal.returncode == 0


def test_score_board(tmp_path) -> None:
    """Two plays to the end of the (short) test song: both on the board
    (scores.json, best first) with their placement in the log; the results
    screen shows the board (checked at text sizes 1.0 and 2.6), the menu the
    best score, and S the score board (Left/Right level); no overlaps."""
    for scale in ("1.0", "2.6"):
        result = _run_game(tmp_path, ["--auto", "--auto-seconds", "60", "--count-in", "0", "--check-layout",
                                      "--text-scale", scale])  # fmt: skip
        assert "LAYOUT" not in result.stderr, result.stderr
    board = json.loads((tmp_path / "data" / "scores.json").read_text())["scores"]
    assert len(board) == 2
    logs = sorted(glob.glob(str(tmp_path / "logs" / "game-*.json")))
    placements = [json.loads(Path(log).read_text())["result"]["placement"] for log in logs]
    assert placements[0]["rank"] == 1 and placements[1]["entries"] == 2
    for scale in ("1.0", "2.6"):
        keys = "10:s 30:right 50:left 70:escape 90:escape"
        result = _run_game(tmp_path, ["--check-layout", "--text-scale", scale, "--keys", keys])
        assert "LAYOUT" not in result.stderr, result.stderr
