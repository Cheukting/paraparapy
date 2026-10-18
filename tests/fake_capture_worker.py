"""Stand-in capture workers for testing CaptureClient without a camera.

They follow the same contract as ``parapara.capture.worker.run`` (see
``parapara.ipc.client``) but synthesize frames and samples, and never import
``cv2``/``mediapipe``. The child process finds this module because the spawn
start method copies the parent's ``sys.path`` (which pytest has extended
with the tests directory).
"""

import time

import numpy as np

from parapara.ipc.frame_buffer import SharedFrameBuffer
from parapara.ipc.messages import CaptureStatus, StatusKind
from parapara.pose.landmarks import NUM_POSE_LANDMARKS, PoseSample

FRAMES = 3


def run(*, config, results, stop_event, frame_buffer_name, frame_shape, frame_lock):
    """Emit READY, then FRAMES frames whose pixels equal their index, then
    wait for the stop event and emit STOPPED."""
    frames = SharedFrameBuffer.attach(frame_buffer_name, frame_shape, frame_lock)
    try:
        results.put(CaptureStatus(StatusKind.READY, "fake", {"camera_width": config.width}))
        for index in range(FRAMES):
            captured = time.monotonic_ns()
            frames.write(np.full(frame_shape, index, dtype=np.uint8), index, captured)
            pose = np.zeros((NUM_POSE_LANDMARKS, 5), dtype=np.float32)
            results.put(
                PoseSample(
                    frame_index=index,
                    captured_ns=captured,
                    processed_ns=time.monotonic_ns(),
                    frame_size=(config.width, config.height),
                    pose_image=pose,
                    pose_world=pose,
                )
            )
        stop_event.wait(timeout=10)
        results.put(CaptureStatus(StatusKind.STOPPED))
    finally:
        frames.close()


def crash(**kwargs):
    """A worker that dies immediately with an exception."""
    raise RuntimeError("boom")


def hang(**kwargs):
    """A worker that ignores the stop event (must be terminated)."""
    time.sleep(60)


def stream(*, config, results, stop_event, frame_buffer_name, frame_shape, frame_lock):
    """Like a real camera: READY, then a sample (and frame) every ~20 ms with
    a well-framed person in it (passes the framing check), until the stop
    event is set."""
    from pose_synth import image_pose, world_pose  # tests dir is on sys.path

    image, world = image_pose(), world_pose()
    frames = SharedFrameBuffer.attach(frame_buffer_name, frame_shape, frame_lock)
    try:
        results.put(CaptureStatus(StatusKind.READY, "fake stream", {"camera_width": config.width}))
        index = 0
        while not stop_event.wait(timeout=0.02):
            captured = time.monotonic_ns()
            frames.write(np.full(frame_shape, index % 256, dtype=np.uint8), index, captured)
            results.put(
                PoseSample(
                    frame_index=index, captured_ns=captured, processed_ns=time.monotonic_ns(),
                    frame_size=(config.width, config.height), pose_image=image, pose_world=world,
                    pose_ms=1.0,
                )
            )
            index += 1
        results.put(CaptureStatus(StatusKind.STOPPED))
    finally:
        frames.close()
