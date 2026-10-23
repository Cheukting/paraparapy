"""Tests for the play session, latency calibration and player settings."""

from dataclasses import replace

import numpy as np
import pytest
from pose_synth import danced_block, routine_play_samples

from parapara import settings
from parapara.audio.demo_song import demo_song_metadata
from parapara.audio.synth import render_demo_song
from parapara.generator.routine_generator import generate_routine
from parapara.pose.calibration import Calibration
from parapara.pose.features import extract_recording
from parapara.routine.model import Difficulty
from parapara.scoring.judge import Grade, Judgment
from parapara.scoring.latency import estimate_latency
from parapara.scoring.session import PlaySession

START = 10**9
SONG = demo_song_metadata(render_demo_song())  # 150 BPM, routine from beat 8, 80 beats
ROUTINE = generate_routine(SONG.id, 2, Difficulty.NORMAL)


def _play(lag_s: float, latency_ms: float) -> PlaySession:
    session = PlaySession(SONG, ROUTINE, START, latency_ns=int(latency_ms * 1e6))
    for sample in routine_play_samples(SONG, ROUTINE, START, lag_s=lag_s):
        session.add(sample)
    session.finish()
    return session


# --- play session ---------------------------------------------------------------


def test_perfect_player_scores_near_full_marks() -> None:
    s = _play(lag_s=0.0, latency_ms=0).summary()
    assert s["keyframes"] == len(list(ROUTINE.timeline()))
    assert s["accuracy_pct"] > 90 and s["grades"]["miss"] == 0
    assert s["posture_pct"] == 100.0 and s["score"] > 0


def test_perfect_player_is_timed_on_the_beat() -> None:
    """Hits are timed on the unsmoothed arms: the demo dancer, exact on every
    beat, is graded within ±40 ms on every sharp move (it was 40-130 ms late
    while timing used the smoothed arms, 2026-10-23)."""
    from parapara.scoring.judge import Judgment

    session = PlaySession(SONG, ROUTINE, START)
    judgments = []
    for sample in routine_play_samples(SONG, ROUTINE, START):
        judgments += [e.judgment for e in session.add(sample)]
    judgments += [e.judgment for e in session.finish()]
    offsets = [j.offset_ms for j in judgments
               if isinstance(j, Judgment) and j.offset_ms is not None and not j.item.block.is_flowing]  # fmt: skip
    assert offsets and max(abs(o) for o in offsets) < 40, sorted(offsets)


def test_latency_compensation_rescues_a_late_player() -> None:
    late = _play(lag_s=0.2, latency_ms=0).summary()
    compensated = _play(lag_s=0.2, latency_ms=200).summary()
    assert compensated["accuracy_pct"] > late["accuracy_pct"] + 10
    assert compensated["accuracy_pct"] > 85


def test_events_stream_during_play_and_session_finishes_once() -> None:
    session = PlaySession(SONG, ROUTINE, START)
    events = []
    for sample in routine_play_samples(SONG, ROUTINE, START):
        events += session.add(sample)
    assert events and isinstance(events[0].judgment, Judgment)
    assert events[0].points > 0 and events[0].judgment.grade is not Grade.MISS
    session.finish()
    assert session.finish() == [] and session.add(session.samples[0]) == []


def test_framing_warning_and_recording() -> None:
    session = _play(0.0, 0)
    assert session.framing_warning is None
    away = replace(session.samples[0], pose_image=None, pose_world=None)
    fresh = PlaySession(SONG, ROUTINE, START)
    fresh.add(away)
    assert fresh.framing_warning is not None and "no_person" in fresh.framing_warning.codes
    rec = session.recording()
    assert len(rec) == len(session.samples) and rec.metadata["routine_id"] == ROUTINE.id
    assert len(rec.metadata["keyframe_times_ns"]) == len(session.schedule)
    assert rec.metadata["summary"]["accuracy_pct"] == session.summary()["accuracy_pct"]


# --- latency -------------------------------------------------------------------------


def _clapping(delay_ms: float, clicks_ns: list[int], jitter_ms: float = 0.0, miss: set[int] = frozenset()):
    """Features of a player clapping ``delay_ms`` after each click (sampled at 23 fps)."""
    base = next(f for f in extract_recording(danced_block("chop_right", 0.0, fps=30)) if f is not None)
    rng = np.random.default_rng(1)
    claps = [c + (delay_ms + rng.normal(0, jitter_ms)) * 1e6 for i, c in enumerate(clicks_ns) if i not in miss]
    out = []
    for t in np.arange(clicks_ns[0] - 5e8, clicks_ns[-1] + 5e8, 1e9 / 23):
        d = min(abs(t - c) for c in claps) / 1e9
        out.append(replace(base, t_ns=int(t), wrist_distance=0.15 + 2.0 * min(1.0, d / 0.12)))
    return out


BEAT_NS = 5e8
CLICKS = [START + int(i * BEAT_NS) for i in range(16)]


@pytest.mark.parametrize("delay_ms", [0, 80, 160, -30])
def test_latency_recovered(delay_ms) -> None:
    r = estimate_latency(_clapping(delay_ms, CLICKS), CLICKS, BEAT_NS)
    assert r.ok and r.claps_found == 16
    assert r.latency_ms == pytest.approx(delay_ms, abs=12)  # 43 ms frames, refined


def test_latency_from_8_claps() -> None:
    """The game asks for 8 claps (fourth test run: 16 felt too long): careful
    clapping is still recovered, too few or scattered claps rejected."""
    clicks = CLICKS[:8]
    r = estimate_latency(_clapping(200, clicks, jitter_ms=25), clicks, BEAT_NS)
    assert r.ok and r.claps_found == 8
    assert r.latency_ms == pytest.approx(200, abs=25)
    few = estimate_latency(_clapping(200, clicks, miss={0, 2, 4}), clicks, BEAT_NS)
    assert not few.ok  # 5 of 8 is below MIN_CLAPS
    assert not estimate_latency(_clapping(200, clicks, jitter_ms=120), clicks, BEAT_NS).ok


def test_latency_rejects_too_few_or_uneven_claps() -> None:
    few = estimate_latency(_clapping(100, CLICKS, miss=set(range(12))), CLICKS, BEAT_NS)
    assert not few.ok and "of 16 claps" in few.reason
    uneven = estimate_latency(_clapping(100, CLICKS, jitter_ms=120), CLICKS, BEAT_NS)
    assert not uneven.ok and ("uneven" in uneven.reason or "of 16" in uneven.reason)
    none = estimate_latency([], CLICKS, BEAT_NS)
    assert not none.ok and none.claps_found == 0


# --- settings --------------------------------------------------------------------------


def test_settings_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PARAPARA_DATA_DIR", str(tmp_path))
    assert settings.load_latency() is None and settings.load_calibration() is None
    settings.save_latency(123.4, 15.0, 14)
    assert settings.load_latency()["latency_ms"] == 123.4
    cal = Calibration(0.35, 0.62, 0.63, (0.0, -1.0, 0.0), 5.0, 0.5, 3.0, 4.0)
    settings.save_calibration(cal)
    assert settings.load_calibration() == cal
    (tmp_path / "calibration.json").write_text("{}")
    assert settings.load_calibration() is None  # broken file: calibrate again
    assert settings.load_ui() == {}
    settings.save_ui(text_scale=2.2)
    settings.save_ui(other=1)  # merged, not replaced
    assert settings.load_ui() == {"text_scale": 2.2, "other": 1}
    (tmp_path / "ui.json").write_text("not json")
    assert settings.load_ui() == {}  # broken file: defaults


def test_ending_early_drops_moves_not_yet_due() -> None:
    session = PlaySession(SONG, ROUTINE, START)
    samples = routine_play_samples(SONG, ROUTINE, START)
    half = len(samples) // 2
    for sample in samples[:half]:
        session.add(sample)
    session.finish(now_ns=samples[half].captured_ns)
    s = session.summary()
    assert s["ended_early"] and 0 < s["keyframes"] < len(list(ROUTINE.timeline()))
    assert s["grades"]["miss"] <= 1 and s["accuracy_pct"] > 85
    full = _play(0.0, 0)
    assert not full.summary()["ended_early"]
