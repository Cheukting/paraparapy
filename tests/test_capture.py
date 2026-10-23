"""Tests for parapara.capture: models, detector conversion, worker loop.

These import ``cv2``/``mediapipe`` in the pytest process, which is fine as
long as no test imports ``pygame`` in-process (see conftest.py).

The detector integration test runs the real MediaPipe models and is skipped
when they are not cached yet (``uv run parapara-capture-debug`` downloads
them on first use).
"""

import hashlib
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from parapara.capture.camera import CameraFrame
from parapara.capture.detector import (
    Detection,
    LandmarkDetector,
    pose_landmarks_to_array,
)
from parapara.capture.models import (
    MODELS,
    ModelDownloadError,
    ModelSpec,
    default_model_dir,
    ensure_model,
)
from parapara.capture.worker import QueueEmitter, capture_loop, to_rgb
from parapara.pose.landmarks import NUM_POSE_LANDMARKS

# --- models -----------------------------------------------------------------


def _local_spec(tmp_path: Path, content: bytes, sha: str | None = None) -> ModelSpec:
    src = tmp_path / "src.task"
    src.write_bytes(content)
    return ModelSpec(
        filename="m.task",
        url=src.as_uri(),  # urllib handles file:// URLs: no network needed
        sha256=sha or hashlib.sha256(content).hexdigest(),
        size=len(content),
    )


def test_pinned_model_specs_look_valid() -> None:
    assert set(MODELS) == {"pose_lite", "pose_full"}  # no hand model (removed)
    for spec in MODELS.values():
        assert spec.url.startswith("https://storage.googleapis.com/mediapipe-models/")
        assert spec.url.endswith(spec.filename)
        assert len(spec.sha256) == 64 and int(spec.sha256, 16) >= 0
        assert spec.size > 1_000_000


def test_default_model_dir_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PARAPARA_MODEL_DIR", str(tmp_path))
    assert default_model_dir() == tmp_path


def test_ensure_model_downloads_once_then_uses_cache(tmp_path) -> None:
    spec = _local_spec(tmp_path, b"model-bytes")
    cache = tmp_path / "cache"
    calls = []
    path = ensure_model(spec, cache, on_download=calls.append)
    assert path.read_bytes() == b"model-bytes" and calls == [spec]
    assert ensure_model(spec, cache, on_download=calls.append) == path
    assert calls == [spec]  # second call hit the cache


def test_ensure_model_replaces_corrupt_cache(tmp_path) -> None:
    spec = _local_spec(tmp_path, b"good")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / spec.filename).write_bytes(b"evil")  # same size, wrong hash
    assert ensure_model(spec, cache).read_bytes() == b"good"


def test_ensure_model_rejects_hash_mismatch_and_leaves_nothing(tmp_path) -> None:
    spec = _local_spec(tmp_path, b"tampered", sha="0" * 64)
    cache = tmp_path / "cache"
    with pytest.raises(ModelDownloadError, match="SHA-256 mismatch"):
        ensure_model(spec, cache)
    assert list(cache.iterdir()) == []  # no final file, no leftover .part file


def test_ensure_model_network_error(tmp_path) -> None:
    spec = ModelSpec("m.task", (tmp_path / "missing.task").as_uri(), "0" * 64, 1)
    with pytest.raises(ModelDownloadError, match="could not download"):
        ensure_model(spec, tmp_path / "cache")


# --- detector conversion (duck-typed fakes) ------------------------------------


def _lm(i: float, visibility=0.9, presence=0.8):
    return SimpleNamespace(
        x=i, y=i + 0.5, z=-i, visibility=visibility, presence=presence
    )


def test_pose_landmarks_to_array() -> None:
    lms = [_lm(i / 100) for i in range(NUM_POSE_LANDMARKS)]
    lms[3] = _lm(0.03, visibility=None, presence=None)
    arr = pose_landmarks_to_array(lms)
    assert arr.shape == (33, 5) and arr.dtype == np.float32
    assert arr[10, :3] == pytest.approx([0.1, 0.6, -0.1])
    assert np.isnan(arr[3, 3]) and np.isnan(arr[3, 4])


def test_landmark_count_checked() -> None:
    with pytest.raises(ValueError):
        pose_landmarks_to_array([_lm(0)] * 5)


# --- real MediaPipe integration (skipped until models are cached) --------------


def _cached(name: str) -> Path | None:
    path = default_model_dir() / MODELS[name].filename
    return path if path.exists() else None


@pytest.mark.skipif(
    _cached("pose_lite") is None,
    reason="MediaPipe models not cached; run `uv run parapara-capture-debug` once",
)
def test_detector_on_empty_frame_finds_no_one_and_tolerates_equal_timestamps() -> None:
    pose_path = ensure_model(MODELS["pose_lite"])  # verifies the cached hash
    detector = LandmarkDetector(pose_path)
    try:
        frame = np.zeros((240, 320, 3), np.uint8)
        # Same capture time twice: VIDEO mode would raise without the guard.
        for _ in range(2):
            detection = detector.detect(frame, captured_ns=5_000_000)
            assert detection.pose_image is None and detection.pose_world is None
            assert detection.pose_ms > 0
    finally:
        detector.close()


# --- worker loop ---------------------------------------------------------------


class FakeSource:
    def __init__(self, n: int) -> None:
        self.frames = [
            CameraFrame(i * 2, 1_000_000 * i, np.full((4, 6, 3), i, np.uint8))
            for i in range(n)
        ]  # indices 0, 2, 4: gaps = frames skipped by the grabber
        self.calls = 0

    def next_frame(self, after: int, timeout: float):
        self.calls += 1
        if self.calls == 2:
            return None  # simulate a timeout; the loop must just retry
        return next((f for f in self.frames if f.index > after), None)


class FakeDetector:
    def __init__(self) -> None:
        self.timestamps: list[int] = []

    def detect(self, rgb, captured_ns):
        self.timestamps.append(captured_ns)
        pose = np.zeros((NUM_POSE_LANDMARKS, 5), np.float32)
        return Detection(pose, pose, pose_ms=5.0)


class FakeSink:
    def __init__(self) -> None:
        self.written: list[int] = []

    def write(self, frame, frame_index, captured_ns) -> None:
        self.written.append(frame_index)


def test_capture_loop_processes_newest_frames() -> None:
    source, detector, sink, emitted = FakeSource(3), FakeDetector(), FakeSink(), []
    processed = capture_loop(
        source=source,
        detector=detector,
        frames=sink,
        emit=emitted.append,
        should_stop=lambda: len(emitted) >= 3,
        convert=lambda img: img,
        frame_size=(6, 4),
        dropped=lambda: 7,
    )
    assert processed == 3
    assert [s.frame_index for s in emitted] == sink.written == [0, 2, 4]
    assert detector.timestamps == [0, 1_000_000, 2_000_000]
    sample = emitted[0]
    assert sample.has_pose
    assert (sample.pose_ms, sample.dropped_samples) == (5.0, 7)
    assert sample.processed_ns >= sample.captured_ns


def test_queue_emitter_drops_instead_of_blocking() -> None:
    q: queue.Queue = queue.Queue(maxsize=1)
    emit = QueueEmitter(q)
    emit("a")
    emit("b")
    assert q.get_nowait() == "a" and emit.dropped == 1


def test_to_rgb_converts_and_resizes() -> None:
    bgr = np.zeros((8, 12, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue in BGR
    rgb = to_rgb(bgr, width=6, height=4)
    assert rgb.shape == (4, 6, 3)
    assert rgb[0, 0].tolist() == [0, 0, 255]
