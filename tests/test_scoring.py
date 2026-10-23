"""Tests for the scoring engine (parapara.scoring), with synthetic dancers
from tests/pose_synth.py moving smoothly through a block, on or off the beat."""

from dataclasses import replace

import numpy as np
import pytest
from pose_synth import danced_block

from parapara.pose.features import FeatureExtractor, extract_recording
from parapara.routine.geometry import DOWN, RIGHT
from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import ArmPose, Keyframe
from parapara.routine.serialization import load_routine
from parapara.scoring.judge import (
    FlowJudgment,
    Grade,
    Judge,
    Judgment,
    TimingWindows,
    judge_all,
)
from parapara.scoring.matching import (
    clap_score,
    keyframe_error_deg,
    pose_score,
    weighted_angle_deg,
)
from parapara.scoring.schedule import BeatClock, schedule_block_clip, schedule_routine
from parapara.scoring.scorekeeper import ScoreKeeper

MS = 1_000_000
ROUTINE = load_routine(
    __import__("pathlib").Path(__file__).resolve().parent
    / "fixtures/sample-routine.json"
)


def _score_clip(block_id: str, lag_s: float, latency_ms: float = 0.0):
    rec = danced_block(block_id, lag_s=lag_s, fps=30)
    m = rec.metadata
    schedule = schedule_block_clip(BLOCK_LIBRARY[block_id], m["beat_times_ns"][m["count_in_beats"]:],
                                   int(latency_ms * MS))  # fmt: skip
    features = extract_recording(rec, FeatureExtractor())
    return schedule, features, judge_all(schedule, features)


def _keyframes(results):
    return [j for j in results if isinstance(j, Judgment)]


# --- schedule ----------------------------------------------------------------


def test_block_clip_schedule_and_holds() -> None:
    beats = [i * 500 * MS for i in range(16)]
    knock = schedule_block_clip(BLOCK_LIBRARY["knock_right"], beats)
    assert [s.t_ns for s in knock[:4]] == [0, 500 * MS, 1000 * MS, 1500 * MS]
    assert [s.hold for s in knock[:4]] == [
        False,
        True,
        False,
        True,
    ]  # second knock = same pose
    chop = schedule_block_clip(BLOCK_LIBRARY["chop_right"], beats)
    assert not any(s.hold for s in chop)
    roll = schedule_block_clip(BLOCK_LIBRARY["wrist_roll_4"], beats)
    assert [s.hold for s in roll[:5]] == [
        False,
        True,
        True,
        True,
        False,
    ]  # new instance
    assert roll[4].instance == 1 and roll[4].block_start_ns == 2000 * MS


def test_routine_schedule_with_latency() -> None:
    clock = BeatClock(
        bpm=150, first_beat_ns=10**9, latency_ns=80 * MS
    )  # 400 ms per beat
    schedule = schedule_routine(ROUTINE, clock)
    timeline = list(ROUTINE.timeline())
    assert len(schedule) == len(timeline)
    assert schedule[0].t_ns == 10**9 + 80 * MS
    assert schedule[-1].t_ns == clock.time_of(timeline[-1].beat)
    assert (
        schedule[0].block_end_ns - schedule[0].block_start_ns == 4 * 400 * MS
    )  # 4-count sweep
    assert [s.instance for s in schedule] == sorted(s.instance for s in schedule)


# --- matching -------------------------------------------------------------------


def test_weighted_angle_discounts_depth() -> None:
    a, b = np.array([0.0, -0.8, 0.6]), (0.0, -1.0, 0.0)  # hanging arm read ~37° forward
    assert weighted_angle_deg(a, b, depth_weight=1.0) == pytest.approx(36.9, abs=0.1)
    assert weighted_angle_deg(a, b) < 25  # default: depth counts half
    assert weighted_angle_deg(a, b, depth_weight=0.0) == pytest.approx(0.0, abs=1e-6)
    assert weighted_angle_deg(np.array([1.0, 0, 0]), RIGHT) == pytest.approx(0.0)


def test_pose_score_shape() -> None:
    assert [pose_score(e, 25) for e in (0, 25, 37.5, 50, 90)] == [1, 1, 0.5, 0, 0]


def test_keyframe_error_counts_invisible_arm_as_full_miss() -> None:
    feats = extract_recording(danced_block("chop_right", lag_s=0.0, fps=30))[5]
    hidden = replace(feats, right=replace(feats.right, valid=False))
    kf = Keyframe(beat=0, right=ArmPose(upper=DOWN, fore=DOWN))
    assert keyframe_error_deg(hidden, kf) == 180.0
    assert (
        keyframe_error_deg(feats, Keyframe(beat=0, clap=True)) == 0.0
    )  # no arms to compare


def test_clap_score() -> None:
    feats = extract_recording(danced_block("chop_right", lag_s=0.0, fps=30))[5]
    assert clap_score(replace(feats, wrist_distance=0.3)) == 1.0
    assert clap_score(replace(feats, wrist_distance=1.5)) == 0.0


# --- judge ---------------------------------------------------------------------


@pytest.mark.parametrize("block_id", ["chop_right", "sweep_to_right_4", "knock_right"])
def test_on_beat_dancer_scores_perfect(block_id) -> None:
    _, _, results = _score_clip(block_id, lag_s=0.0)
    keyframes = _keyframes(results)
    assert {j.grade for j in keyframes} <= {Grade.PERFECT, Grade.HOLD}
    assert np.mean([j.points for j in keyframes]) > 0.95
    for j in keyframes:
        if j.offset_ms is not None:
            # Refined between 33 ms samples; in continuous moves (sweeps) the
            # landmark filter adds its ~28 ms lag, which the latency
            # calibration absorbs because it measures through the same pipeline.
            assert -25 <= j.offset_ms <= 45


def test_late_dancer_offset_is_measured() -> None:
    _, _, results = _score_clip("chop_right", lag_s=0.1)
    offsets = [j.offset_ms for j in _keyframes(results) if j.offset_ms is not None]
    assert np.median(offsets) == pytest.approx(
        100 + 28, abs=30
    )  # + landmark filter lag
    assert {j.grade for j in _keyframes(results)} <= {
        Grade.GREAT,
        Grade.GOOD,
        Grade.PERFECT,
    }


def test_latency_compensation_restores_perfect() -> None:
    _, _, late = _score_clip("chop_right", lag_s=0.1)
    _, _, compensated = _score_clip("chop_right", lag_s=0.1, latency_ms=128)
    assert np.mean([j.points for j in _keyframes(compensated)]) > np.mean(
        [j.points for j in _keyframes(late)]
    )
    assert Grade.PERFECT in {j.grade for j in _keyframes(compensated)}


def test_far_too_late_is_a_miss() -> None:
    _, _, results = _score_clip(
        "chop_right", lag_s=0.5
    )  # a whole beat late = wrong pose on the beat
    assert (
        sum(j.grade is Grade.MISS for j in _keyframes(results))
        >= len(_keyframes(results)) // 2
    )


def test_nothing_seen_is_a_miss() -> None:
    schedule = schedule_block_clip(
        BLOCK_LIBRARY["chop_right"], [i * 500 * MS for i in range(4)]
    )
    results = judge_all(schedule, [None, None])
    assert all(j.grade is Grade.MISS and j.reason == "not seen" for j in results)


def test_streaming_judges_after_the_window_closes() -> None:
    schedule, features, _ = _score_clip("chop_right", lag_s=0.0)
    judge = Judge(schedule)
    window_ns = TimingWindows().window_ns(schedule[0].beat_ns)
    first_emitted_at = None
    for f in features:
        out = judge.add(f)
        if out and first_emitted_at is None:
            first_emitted_at = f.t_ns
            assert isinstance(out[0], Judgment) and out[0].item.index == 0
    assert first_emitted_at > schedule[0].t_ns + window_ns
    assert first_emitted_at < schedule[0].t_ns + window_ns + 60 * MS  # promptly after


def test_window_is_capped_at_part_of_a_beat() -> None:
    w = TimingWindows()
    assert w.window_ns(60e9 / 120) == 190 * MS  # 500 ms beat: full Good window
    assert w.window_ns(60e9 / 160) == pytest.approx(
        0.45 * 375 * MS
    )  # fast songs: capped


def test_clap_timing_from_sound_onset_with_vision_gate() -> None:
    schedule, features, _ = _score_clip("flick_clap_to_right", lag_s=0.0)
    clap = next(s for s in schedule if s.keyframe.clap)
    near = min(
        (f for f in features if f is not None), key=lambda f: abs(f.t_ns - clap.t_ns)
    )
    together = [replace(f, wrist_distance=0.2) if f is near else f for f in features]

    judge = Judge([clap])
    judge.add_clap_onset(clap.t_ns + 30 * MS)
    out = [j for f in together for j in judge.add(f)] + judge.finish()
    (j,) = [x for x in out if isinstance(x, Judgment)]
    assert j.offset_ms == pytest.approx(30) and j.grade is Grade.PERFECT

    apart = [replace(f, wrist_distance=2.0) for f in features if f is not None]
    judge = Judge([clap])
    judge.add_clap_onset(clap.t_ns)  # the song's own clap: hands are apart
    (j,) = [x for x in [*judge.add(apart[0]), *[y for f in apart[1:] for y in judge.add(f)], *judge.finish()]
            if isinstance(x, Judgment)]  # fmt: skip
    assert j.grade is Grade.MISS and j.reason == "no clap"


def test_flow_rewards_moving_and_penalizes_standing_still() -> None:
    _, _, moving = _score_clip("sweep_to_right_4", lag_s=0.0)
    flows = [j for j in moving if isinstance(j, FlowJudgment)]
    assert len(flows) == 4 and all(f.score == 1.0 for f in flows)  # 4 sweeps in 2 bars

    schedule, features, _ = _score_clip("sweep_to_right_4", lag_s=0.0)
    frozen = [replace(features[10], t_ns=f.t_ns) for f in features if f is not None]
    still = [j for j in judge_all(schedule, frozen) if isinstance(j, FlowJudgment)]
    assert all(f.score == 0.0 for f in still)


# --- score keeper ------------------------------------------------------------------


def _judgment(grade: Grade, points: float = 1.0) -> Judgment:
    schedule = schedule_block_clip(BLOCK_LIBRARY["chop_right"], [0, 500 * MS])
    return Judgment(schedule[0], grade, points, points)


def test_combo_multiplier_and_reset() -> None:
    keeper = ScoreKeeper()
    earned = [keeper.add(_judgment(Grade.PERFECT)) for _ in range(20)]
    assert earned[:9] == [100] * 9 and earned[9] == 110  # 10th hit: x1.1
    assert keeper.combo == 20 and keeper.multiplier == pytest.approx(1.2)
    keeper.add(_judgment(Grade.MISS, 0.0))
    assert keeper.combo == 0 and keeper.max_combo == 20 and keeper.multiplier == 1.0


def test_multiplier_is_capped() -> None:
    keeper = ScoreKeeper()
    for _ in range(300):
        keeper.add(_judgment(Grade.PERFECT))
    assert keeper.multiplier == 2.0


def test_summary_accuracy_posture_and_flow() -> None:
    _, features, results = _score_clip("sweep_to_right_4", lag_s=0.0)
    keeper = ScoreKeeper()
    for j in results:
        keeper.add(j)
    for f in features:
        keeper.add_posture(f)
    s = keeper.summary()
    assert s["accuracy_pct"] > 95 and s["flow_pct"] == 100.0
    assert s["posture_pct"] == 100.0  # synthetic dancer stands upright
    assert s["posture_bonus"] == round(0.1 * 100 * s["keyframes"])
    assert s["score"] == s["keyframe_score"] + s["posture_bonus"]
    assert sum(s["grades"].values()) == s["keyframes"] == 16


def test_posture_rules() -> None:
    f = extract_recording(danced_block("chop_right", lag_s=0.0, fps=30))[5]
    keeper = ScoreKeeper(calibrated=True)
    for posture in (
        {},
        {"lean_deg": 25.0},
        {"side_tilt_deg": -20.0},
        {"head_height": -0.5},
    ):
        keeper.add_posture(replace(f, **posture))
    keeper.add_posture(None)  # ignored
    assert keeper.posture_share == pytest.approx(0.25)


def test_score_session_cli(tmp_path, capsys) -> None:
    from parapara.diagnostics import score_clips

    session = tmp_path / "20990101-000000"
    session.mkdir()
    (session / "session.json").write_text("{}")
    danced_block("chop_right", lag_s=0.1, fps=30).save(session / "block_chop_right.npz")
    fixed = score_clips.run([str(session), "--latency-ms", "0"])
    auto = score_clips.run([str(session), "--auto-latency"])
    assert fixed["calibrated"] is False
    assert auto["clips"]["block_chop_right"]["latency_ms"] == pytest.approx(100, abs=25)
    assert (auto["clips"]["block_chop_right"]["accuracy_pct"]
            >= fixed["clips"]["block_chop_right"]["accuracy_pct"])  # fmt: skip
    assert (session / "score.json").exists() and "chop_right" in capsys.readouterr().out
