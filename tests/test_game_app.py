"""Headless run of the game app (`parapara-play`) with a streaming fake camera:
menu skipped (--auto), song played for a few seconds, results saved."""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from parapara.audio import importer
from parapara.audio.synth import render_demo_song, write_wav
from parapara.pose.recording import Recording

TESTS = Path(__file__).resolve().parent


def _run_game(tmp_path, args: list[str]) -> subprocess.CompletedProcess:
    if not (tmp_path / "lib").exists():
        track = render_demo_song()
        write_wav(tmp_path / "t.wav", track.samples, track.sample_rate)
        importer.import_song(tmp_path / "t.wav", importer.SongInfo("Test"), library=tmp_path / "lib")
    script = f"from parapara.game import app; app.run({args!r}, worker='fake_capture_worker:stream')"
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
    _run_game(tmp_path, ["--auto", "--auto-seconds", "6"])
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
    """With a saved body calibration: menu -> latency (no claps from the fake
    camera: rejected, skipped with S) -> play -> end early -> results -> menu."""
    from parapara.pose.calibration import Calibration

    (tmp_path / "data").mkdir()
    cal = Calibration(0.35, 0.62, 0.63, (0.0, -1.0, 0.0), 5.0, 0.5, 3.0, 4.0)
    cal.save(tmp_path / "data" / "calibration.json")
    # The latency clicks + evaluation take ~2.5 s (~150 frames): S is sent twice
    # with slack (S is ignored once playing).
    _run_game(tmp_path, ["--fast", "--keys", "10:return 20:space 240:s 300:s 460:escape 490:return 510:escape"])
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
