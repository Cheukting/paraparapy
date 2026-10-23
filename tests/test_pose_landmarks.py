"""Tests for parapara.pose.landmarks: landmark tables and the sample type."""

import pickle

import numpy as np
import pytest

from parapara.pose.landmarks import (
    NUM_POSE_LANDMARKS,
    UPPER_BODY_CONNECTIONS,
    PoseLandmark,
    PoseSample,
)


def test_landmark_tables_are_consistent() -> None:
    assert NUM_POSE_LANDMARKS == 33
    assert all(0 <= i < NUM_POSE_LANDMARKS for c in UPPER_BODY_CONNECTIONS for i in c)


def test_upper_body_drawing_includes_pose_hand_points_but_no_legs() -> None:
    used = {PoseLandmark(i) for c in UPPER_BODY_CONNECTIONS for i in c}
    for side in ("LEFT", "RIGHT"):
        for joint in ("SHOULDER", "ELBOW", "WRIST", "INDEX", "PINKY", "THUMB"):
            assert PoseLandmark[f"{side}_{joint}"] in used
    assert not any(
        p.name.endswith(("KNEE", "ANKLE", "HEEL", "FOOT_INDEX")) for p in used
    )


def test_pose_sample_pickles_and_reports_timing() -> None:
    pose = np.zeros((NUM_POSE_LANDMARKS, 5), dtype=np.float32)
    sample = PoseSample(
        frame_index=1, captured_ns=1_000_000, processed_ns=21_000_000, frame_size=(640, 480),
        pose_image=pose, pose_world=pose, pose_ms=12.5,
    )  # fmt: skip
    copy = pickle.loads(pickle.dumps(sample))
    assert copy.processing_ms == pytest.approx(20.0)
    assert copy.has_pose and copy.pose_ms == 12.5
    np.testing.assert_array_equal(copy.pose_image, sample.pose_image)


def test_sample_without_person() -> None:
    sample = PoseSample(0, 0, 0, (640, 480), None, None)
    assert not sample.has_pose
