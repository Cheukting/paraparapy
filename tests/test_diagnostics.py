"""Tests for parapara.diagnostics: run recorder, log files and the benchmark."""

import json

import numpy as np
import pytest

from parapara.diagnostics import bench
from parapara.diagnostics.recorder import RunRecorder, environment, write_log
from parapara.pose.landmarks import PoseSample

MS = 1_000_000


def _sample(i: int, *, frame_step: int = 2, period_ms: float = 50, pose=True):
    """Samples every ``period_ms``; camera frames numbered ``frame_step`` apart
    (i.e. the camera delivers ``frame_step`` frames per processed sample)."""
    captured = int(i * period_ms * MS)
    arr = np.zeros((33, 5), np.float32) if pose else None
    return PoseSample(
        frame_index=i * frame_step, captured_ns=captured, processed_ns=captured + 30 * MS,
        frame_size=(640, 480), pose_image=arr, pose_world=arr,
        pose_ms=20.0 + i % 2, dropped_samples=i // 10,
    )  # fmt: skip


def test_summary_rates_timings_and_detection() -> None:
    rec = RunRecorder(warmup_s=0)
    for i in range(41):  # 2 s at 20 fps, camera at 40 fps
        rec.add(_sample(i, pose=i % 4 != 0))
    s = rec.summary()
    assert s["samples"] == 41 and s["duration_s"] == pytest.approx(2.0)
    assert s["tracking_fps"] == pytest.approx(20.0)
    assert s["camera_fps"] == pytest.approx(40.0)
    assert s["skipped_pct"] == pytest.approx(50.0)
    assert (
        s["pose_ms"]["mean"] == pytest.approx(20.5, abs=0.1)
        and s["pose_ms"]["max"] == 21.0
    )
    assert s["camera_to_landmarks_ms"]["p50"] == pytest.approx(30.0)
    assert s["person_found_pct"] == pytest.approx(100 * 30 / 41, abs=0.1)
    assert s["dropped_samples"] == 4
    assert "camera_to_screen_ms" not in s


def test_warmup_samples_excluded() -> None:
    rec = RunRecorder(warmup_s=1.0)
    for i in range(40):  # 50 ms apart: the first 20 fall in the warm-up
        rec.add(_sample(i))
    assert rec.sample_count == 20


def test_summary_needs_two_samples() -> None:
    rec = RunRecorder(warmup_s=0)
    rec.add(_sample(0))
    assert "error" in rec.summary()


def test_display_latency_included_when_measured() -> None:
    rec = RunRecorder(warmup_s=0)
    for i in range(3):
        rec.add(_sample(i))
        rec.add_display_latency(40.0 + i)
    assert rec.summary()["camera_to_screen_ms"]["p50"] == pytest.approx(41.0)


def test_environment_reads_versions_without_importing_mediapipe() -> None:
    env = environment()
    assert env["python"].startswith("3.")
    assert env["mediapipe"][0].isdigit()


def test_write_log(tmp_path) -> None:
    path = write_log("unit", {"answer": 42}, tmp_path / "logs")
    assert path.name.startswith("unit-") and path.suffix == ".json"
    record = json.loads(path.read_text())
    assert (
        record["kind"] == "unit" and record["answer"] == 42 and "environment" in record
    )


# --- benchmark, with fake workers in real spawned processes --------------------


def test_bench_runs_configs_and_logs(tmp_path, capsys) -> None:
    path = bench.run(
        ["--seconds", "0.5", "--warmup", "0", "--pause", "0", "--width", "8", "--height", "6",
         "--configs", "full,lite", "--log-dir", str(tmp_path)],
        worker="fake_capture_worker:run",
    )  # fmt: skip
    record = json.loads(path.read_text())
    assert list(record["results"]) == ["full", "lite"]
    full = record["results"]["full"]
    assert (
        full["config"]["pose_model"] == "full" and "detect_hands" not in full["config"]
    )
    assert full["summary"]["samples"] == 3  # the fake worker emits 3 samples
    assert record["results"]["lite"]["config"]["pose_model"] == "lite"
    out = capsys.readouterr().out
    assert "track fps" in out and "log written to" in out


def test_bench_records_worker_failure(tmp_path) -> None:
    path = bench.run(
        [
            "--seconds",
            "0.2",
            "--pause",
            "0",
            "--configs",
            "full",
            "--log-dir",
            str(tmp_path),
        ],
        worker="fake_capture_worker:crash",
    )
    result = json.loads(path.read_text())["results"]["full"]
    assert "did not become ready" in result["error"]
    assert "ERROR" in bench.format_table({"full": result})


def test_bench_rejects_unknown_config(tmp_path) -> None:
    with pytest.raises(SystemExit):
        bench.run(["--configs", "turbo", "--log-dir", str(tmp_path)])


def test_low_light_monitor_waits_then_warns_and_recovers() -> None:
    """Below 25 fps for 3 s -> active; above again for 3 s -> cleared; a
    short dip does not trigger it."""
    from parapara.diagnostics.live_stats import LowLightMonitor

    m = LowLightMonitor(min_fps=25.0, hold_s=3.0)
    t, frame = 0, 0

    def run(fps: float, seconds: float) -> None:
        nonlocal t, frame
        for _ in range(int(fps * seconds)):
            t += int(1e9 / fps)
            frame += 1
            m.add(t, frame)

    run(30, 4)
    assert not m.active
    run(20, 1)  # short dip
    run(30, 3)
    assert not m.active
    run(20, 5)
    assert m.active and 18 < m.fps < 22
    run(30, 5)
    assert not m.active
    assert 0.0 < m.low_share < 0.5
