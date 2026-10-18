"""Tests for parapara.ipc: shared frame buffer and the capture client.

The client tests start real ``spawn`` processes running the fake workers in
tests/fake_capture_worker.py, so they exercise the actual queue, shared
memory and shutdown paths without a camera.
"""

import multiprocessing as mp
import time

import numpy as np
import pytest

from parapara.ipc.client import CaptureClient
from parapara.ipc.frame_buffer import SharedFrameBuffer
from parapara.ipc.messages import CaptureConfig, CaptureStatus, StatusKind
from parapara.pose.landmarks import PoseSample

SHAPE = (4, 6, 3)
SMALL = CaptureConfig(width=6, height=4)


def _poll_until(client: CaptureClient, predicate, timeout: float = 15.0) -> list:
    """Collect messages until ``predicate(messages)`` holds or time runs out."""
    messages: list = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        messages += client.poll()
        if predicate(messages):
            return messages
        time.sleep(0.02)
    raise AssertionError(f"timed out; got {messages!r}")


# --- SharedFrameBuffer ---------------------------------------------------------


def test_frame_buffer_starts_empty_then_returns_copies() -> None:
    with SharedFrameBuffer.create(SHAPE, mp.Lock()) as buf:
        assert buf.read() is None
        frame = np.arange(np.prod(SHAPE), dtype=np.uint8).reshape(SHAPE)
        buf.write(frame, frame_index=5, captured_ns=123)
        snap = buf.read()
        assert (snap.frame_index, snap.captured_ns) == (5, 123)
        np.testing.assert_array_equal(snap.frame, frame)
        snap.frame[...] = 0  # a copy: must not affect the buffer
        np.testing.assert_array_equal(buf.read().frame, frame)


def test_frame_buffer_newer_than() -> None:
    with SharedFrameBuffer.create(SHAPE, mp.Lock()) as buf:
        buf.write(np.zeros(SHAPE, np.uint8), 3, 0)
        assert buf.read(newer_than=3) is None
        assert buf.read(newer_than=2) is not None


def test_frame_buffer_rejects_wrong_frames() -> None:
    with SharedFrameBuffer.create(SHAPE, mp.Lock()) as buf:
        with pytest.raises(ValueError):
            buf.write(np.zeros((4, 6), np.uint8), 0, 0)
        with pytest.raises(ValueError):
            buf.write(np.zeros(SHAPE, np.float32), 0, 0)


def test_attach_shares_memory_and_checks_size() -> None:
    lock = mp.Lock()
    with SharedFrameBuffer.create(SHAPE, lock) as owner:
        other = SharedFrameBuffer.attach(owner.name, SHAPE, lock)
        other.write(np.full(SHAPE, 7, np.uint8), 1, 99)
        assert owner.read().frame[0, 0, 0] == 7
        other.close()  # non-owner close must not free the segment
        assert owner.read().captured_ns == 99
        with pytest.raises(ValueError, match="too small"):
            SharedFrameBuffer.attach(owner.name, (100, 100, 3), lock)


# --- CaptureClient (real processes, fake workers) ------------------------------


def test_client_receives_samples_and_frames_across_processes() -> None:
    with CaptureClient(SMALL, worker="fake_capture_worker:run") as client:
        messages = _poll_until(client, lambda ms: sum(isinstance(m, PoseSample) for m in ms) >= 3)
        statuses = [m for m in messages if isinstance(m, CaptureStatus)]
        assert statuses[0].kind is StatusKind.READY
        assert statuses[0].details == {"camera_width": 6}
        samples = [m for m in messages if isinstance(m, PoseSample)]
        assert [s.frame_index for s in samples] == [0, 1, 2]
        snap = client.latest_frame()
        assert snap.frame_index == 2 and snap.frame.shape == SMALL.frame_shape
        assert np.all(snap.frame == 2)
        assert client.latest_frame(newer_than=2) is None
        assert client.is_alive
    assert not client.is_alive


def test_client_stop_is_graceful_and_idempotent() -> None:
    client = CaptureClient(SMALL, worker="fake_capture_worker:run")
    client.start()
    _poll_until(client, lambda ms: any(isinstance(m, PoseSample) for m in ms))
    client.stop()
    assert client.exitcode is None  # process handle released after stop
    client.stop()  # second call is a no-op
    assert client.poll() == [] and client.latest_frame() is None


def test_client_reports_crashed_worker() -> None:
    client = CaptureClient(SMALL, worker="fake_capture_worker:crash")
    client.start()
    deadline = time.monotonic() + 15
    while client.is_alive and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not client.is_alive and client.exitcode == 1
    client.stop()


def test_client_terminates_hung_worker() -> None:
    client = CaptureClient(SMALL, worker="fake_capture_worker:hang")
    client.start()
    started = time.monotonic()
    client.stop(timeout=0.5)
    assert time.monotonic() - started < 10
    assert not client.is_alive


def test_client_cannot_start_twice() -> None:
    with CaptureClient(SMALL, worker="fake_capture_worker:run") as client:
        with pytest.raises(RuntimeError):
            client.start()
