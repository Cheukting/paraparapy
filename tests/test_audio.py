"""Tests for the demo track generator, the song library and beat-map estimation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from parapara.audio.beatgrid import BeatGrid, low_band_envelope, render_click_track
from parapara.audio.beatmap import (
    DETECTION_LAG_S,
    estimate_beat_map,
    fit_constant_tempo,
    fold_tempo,
    update_song,
)
from parapara.audio.demo_song import DEMO_ID, demo_song_metadata, make_demo_song
from parapara.audio.song import Song, SongError, load_library
from parapara.audio.synth import read_wav, render_demo_song, write_wav
from parapara.routine.serialization import load_routine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo_library(tmp_path_factory) -> Path:
    library = tmp_path_factory.mktemp("songs")
    make_demo_song(library)
    return library


# --- synth ---------------------------------------------------------------------


def test_demo_track_layout_and_determinism() -> None:
    a, b = render_demo_song(), render_demo_song()
    np.testing.assert_array_equal(a.samples, b.samples)  # fixed seed
    assert a.bpm == 150 and a.beat_s == pytest.approx(0.4)
    assert a.duration_beats == 8 + 64 + 8
    assert a.duration_s == pytest.approx(0.5 + 80 * 0.4, abs=1e-3)
    assert a.sections == {"intro": (0, 8), "main": (8, 72), "outro": (72, 80)}
    assert 0.85 < np.abs(a.samples).max() <= 0.9  # about -1 dBFS


def test_kicks_fall_exactly_on_the_beats() -> None:
    song = render_demo_song()
    sr = song.sample_rate
    x = song.samples.astype(np.float64)
    low = np.convolve(x, np.ones(64) / 64, mode="same")  # crude low-pass: keep the kick
    for beat in (0, 1, 17, 79):
        t = int(round(song.beat_time(beat) * sr))
        before = np.mean(low[t - int(0.03 * sr):t - int(0.005 * sr)] ** 2)
        after = np.mean(low[t:t + int(0.025 * sr)] ** 2)
        assert after > 10 * before, beat
    assert np.abs(x[: int(0.45 * sr)]).max() < 1e-6  # silent lead-in


def test_wav_round_trip(tmp_path) -> None:
    samples = np.sin(np.linspace(0, 100, 4410)).astype(np.float32) * 0.5
    write_wav(tmp_path / "x.wav", samples, 44100)
    back, sr = read_wav(tmp_path / "x.wav")
    assert sr == 44100 and back.shape == samples.shape
    np.testing.assert_allclose(back, samples, atol=1 / 32767 * 1.5)


# --- song library ----------------------------------------------------------------


def test_generated_track_is_not_in_the_game_library() -> None:
    """Machine testing only: the generated track is never offered for dancing."""
    library = ROOT / "assets" / "songs"
    for song_json in library.glob("*/song.json"):
        assert json.loads(song_json.read_text())["beat_map"]["method"] != "generated", song_json
    assert not (library / DEMO_ID).exists() and not (library / "demo").exists()


def test_demo_song_loads_and_maps_beats(demo_library) -> None:
    song = load_library(demo_library)[DEMO_ID]
    assert song.warnings == () and song.beat_map.verified
    assert song.beat_time_s(8) == pytest.approx(0.5 + 8 * 0.4)
    assert song.beat_at(song.beat_time_s(13.5)) == pytest.approx(13.5)
    clock = song.routine_clock(playback_start_ns=10**9, latency_ns=50_000_000)
    assert clock.time_of(0) == 10**9 + int(3.7e9) + 50_000_000  # routine starts after the intro


def test_sample_routine_fits_the_test_track(demo_library) -> None:
    routine = load_routine(ROOT / "tests/fixtures/sample-routine.json")
    test_track = load_library(demo_library)[DEMO_ID]
    assert test_track.routine_start_beat + routine.total_counts <= test_track.duration_beats


def _song_dict(**changes) -> dict:
    return demo_song_metadata(render_demo_song()).to_dict() | changes


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"license": "CC-BY"}, "license must be an object"),
        ({"bpm": 20}, "not a plausible tempo"),
        ({"audio": "song.txt"}, "audio must be one of"),
        ({"routine_start_beat": 90}, "routine_start_beat"),
        ({"first_beat_s": -1}, "first_beat_s"),
        ({"beat_map": {"method": "guess"}}, "beat_map.method"),
        ({"format_version": 2}, "format_version"),
    ],
)
def test_song_validation(changes, message, tmp_path) -> None:
    with pytest.raises(SongError, match=message):
        Song.from_dict(_song_dict(**changes), tmp_path)


def test_licence_and_source_are_optional(tmp_path) -> None:
    """User songs are for personal local use: no licence or source needed."""
    data = _song_dict()
    for key in ("license", "source", "artist", "attribution"):
        del data[key]
    song = Song.from_dict(data, tmp_path)
    assert song.license.name == "unspecified (personal use)" and song.source == "" and song.artist == ""


def test_missing_fields_and_audio(tmp_path) -> None:
    data = _song_dict()
    del data["bpm"]
    with pytest.raises(SongError, match="missing field"):
        Song.from_dict(data, tmp_path)
    (tmp_path / "song.json").write_text(json.dumps(_song_dict()))
    with pytest.raises(SongError, match="audio file missing"):
        Song.load(tmp_path)


def test_song_warnings(tmp_path) -> None:
    song = Song.from_dict(_song_dict(bpm=128, beat_map={"method": "librosa", "verified": False}), tmp_path)
    assert any("design range" in w for w in song.warnings)
    assert any("not verified" in w for w in song.warnings)


def test_library_can_skip_songs_without_audio(demo_library, tmp_path) -> None:
    (tmp_path / DEMO_ID).mkdir()
    (tmp_path / DEMO_ID / "song.json").write_bytes((demo_library / DEMO_ID / "song.json").read_bytes())
    with pytest.raises(SongError, match="audio file missing"):
        load_library(tmp_path)
    assert load_library(tmp_path, skip_missing_audio=True) == {}


def test_save_round_trip(demo_library, tmp_path) -> None:
    song = load_library(demo_library)[DEMO_ID]
    song.save(tmp_path)
    assert Song.load(tmp_path, require_audio=False) == song


# --- beat maps -------------------------------------------------------------------


def test_fold_tempo() -> None:
    assert fold_tempo(75) == 150 and fold_tempo(300) == 150 and fold_tempo(150) == 150
    with pytest.raises(ValueError):
        fold_tempo(0)


def test_grid_fit_handles_frame_quantization_missing_beats_and_outliers() -> None:
    frame = 256 / 22050  # librosa onset frames
    true = 0.731 + np.arange(200) * 0.4  # 150 BPM, 80 s
    beats = np.round(true / frame) * frame  # quantized like librosa's output
    beats = np.delete(beats, [10, 11, 50])  # missed beats
    beats[100] += 0.17  # one bad detection
    fit = fit_constant_tempo(beats, bpm_hint=152.0)  # librosa's quantized tempo
    assert fit.bpm == pytest.approx(150.0, abs=0.01)
    assert fit.first_beat_s == pytest.approx(0.731, abs=0.004)
    assert fit.drift_ms < 4 and fit.beats_used == len(beats) - 1


def test_grid_fit_needs_beats() -> None:
    with pytest.raises(ValueError):
        fit_constant_tempo([0.1, 0.5])


def test_estimate_beat_map_on_the_demo_track(demo_library) -> None:
    """The generated track's beats are known exactly: detection must recover
    them (this test caught the off-beat lock-on and the median-gap bias)."""
    fit = estimate_beat_map(demo_library / DEMO_ID / "demo.wav")
    assert fit.bpm == pytest.approx(150.0, abs=0.05)
    assert fit.first_beat_s == pytest.approx(0.5, abs=0.005)
    assert fit.drift_ms < 8 and fit.beats_used >= 76
    assert 0 < DETECTION_LAG_S < 0.05


def test_update_song_creates_a_loadable_skeleton(demo_library, tmp_path) -> None:
    song_dir = tmp_path / "new_track"
    song_dir.mkdir()
    (song_dir / "track.wav").write_bytes((demo_library / DEMO_ID / "demo.wav").read_bytes())
    fit = update_song(song_dir, "track.wav")
    data = json.loads((song_dir / "song.json").read_text())
    assert data["bpm"] == pytest.approx(150, abs=0.05) and data["beat_map"]["verified"] is False
    assert data["duration_beats"] == pytest.approx(80, abs=1)
    assert Song.load(song_dir).title == "new_track"  # loads without a licence
    # Updating an existing song keeps the human-entered fields.
    data |= {"license": {"name": "CC-BY-4.0", "url": "https://example.org"}, "source": "test"}
    (song_dir / "song.json").write_text(json.dumps(data))
    update_song(song_dir)
    song = Song.load(song_dir)
    assert song.license.name == "CC-BY-4.0" and song.bpm == pytest.approx(fit.bpm, abs=1e-3)


# --- beat grid & check app ---------------------------------------------------------

def test_beat_grid_nudges() -> None:
    g = BeatGrid(150.0, 0.5, 8)
    assert g.nudge_offset_ms(5).first_beat_s == pytest.approx(0.505)
    assert g.nudge_offset_ms(-1000).first_beat_s == 0.0  # never before the file starts
    assert g.nudge_bpm(0.01).bpm == pytest.approx(150.01)
    assert g.shift_downbeat(-1).routine_start_beat == 7
    assert g.is_bar_start(8) and g.is_bar_start(16) and g.is_bar_start(0) and not g.is_bar_start(9)
    assert g.beats_between(0.0, 1.31) == [0, 1, 2]  # beats at 0.5, 0.9, 1.3
    assert g.beat_at(g.time_of(12.25)) == pytest.approx(12.25)


def test_click_track_is_sample_accurate() -> None:
    sr = 22050
    grid = BeatGrid(120.0, 0.25, 0)
    mix = render_click_track(np.zeros(sr * 2), sr, grid)
    onsets = np.flatnonzero(np.abs(mix) > 1e-3)
    starts = onsets[np.insert(np.diff(onsets) > 100, 0, True)]
    assert starts.tolist() == [round(grid.time_of(b) * sr) + 1 for b in range(4)]  # sin(0)=0 at the very first sample


def test_low_band_envelope_peaks_on_the_kicks() -> None:
    song = render_demo_song()
    env = low_band_envelope(song.samples, song.sample_rate, hop_s=0.005)
    for beat in (2, 30):
        i = int(song.beat_time(beat) / 0.005)
        assert env[i + 1:i + 10].max() > 4 * env[i - 10:i - 2].mean()


def test_beatcheck_app_saves_a_verified_beat_map_headless(demo_library, tmp_path) -> None:
    song_dir = tmp_path / DEMO_ID
    song_dir.mkdir()
    for f in ("song.json", "demo.wav"):
        (song_dir / f).write_bytes((demo_library / DEMO_ID / f).read_bytes())
    env = os.environ | {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
    script = "import sys; from parapara.game import beatcheck; beatcheck.run([sys.argv[1], '--auto'])"
    result = subprocess.run([sys.executable, "-c", script, str(song_dir)], capture_output=True,
                            text=True, env=env, timeout=60)  # fmt: skip
    assert result.returncode == 0, result.stderr
    saved = json.loads((song_dir / "song.json").read_text())
    assert saved["first_beat_s"] == pytest.approx(0.505) and saved["bpm"] == pytest.approx(150.01)
    assert saved["beat_map"]["method"] == "manual" and saved["beat_map"]["verified"] is True
    assert "verified by ear" in saved["beat_map"]["notes"]
