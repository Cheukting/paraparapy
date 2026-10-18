"""Tests for parapara.pose.landmarks: data types and hand-to-wrist assignment."""

import pickle

import numpy as np
import pytest

from parapara.pose.landmarks import (
    HAND_CONNECTIONS,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
    UPPER_BODY_CONNECTIONS,
    HandObservation,
    PoseLandmark,
    PoseSample,
    assign_hands_to_wrists,
)

FRAME = (640, 480)


def _pose(left_wrist=(0.7, 0.5), right_wrist=(0.3, 0.5)) -> np.ndarray:
    """Pose with the given wrist image positions (unmirrored camera image:
    the dancer's left wrist appears on the right of the picture)."""
    pose = np.zeros((NUM_POSE_LANDMARKS, 5), dtype=np.float32)
    pose[:, 3:] = 1.0
    pose[PoseLandmark.LEFT_WRIST, :2] = left_wrist
    pose[PoseLandmark.RIGHT_WRIST, :2] = right_wrist
    return pose


def _hand(wrist_xy, label="?") -> HandObservation:
    image = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
    image[:, :2] = wrist_xy
    return HandObservation(image=image, world=image.copy(), score=0.9, label=label)


def test_landmark_tables_are_consistent() -> None:
    assert NUM_POSE_LANDMARKS == 33
    assert all(0 <= i < NUM_POSE_LANDMARKS for c in UPPER_BODY_CONNECTIONS for i in c)
    assert all(0 <= i < NUM_HAND_LANDMARKS for c in HAND_CONNECTIONS for i in c)
    assert {i for c in HAND_CONNECTIONS for i in c} == set(range(NUM_HAND_LANDMARKS))


def test_pose_sample_pickles_and_reports_timing() -> None:
    sample = PoseSample(
        frame_index=1, captured_ns=1_000_000, processed_ns=21_000_000, frame_size=FRAME,
        pose_image=_pose(), pose_world=_pose(), left_hand=_hand((0.7, 0.5)),
    )  # fmt: skip
    copy = pickle.loads(pickle.dumps(sample))
    assert copy.processing_ms == pytest.approx(20.0)
    assert copy.has_pose
    np.testing.assert_array_equal(copy.pose_image, sample.pose_image)


def test_no_pose_means_no_assignment() -> None:
    assert assign_hands_to_wrists([_hand((0.5, 0.5))], None, FRAME) == (None, None)


def test_no_hands() -> None:
    assert assign_hands_to_wrists([], _pose(), FRAME) == (None, None)


def test_two_hands_assigned_by_nearest_wrist_ignoring_labels() -> None:
    near_left, near_right = _hand((0.69, 0.5), "Right"), _hand((0.31, 0.5), "Left")
    # Labels deliberately contradict the geometry: geometry must win.
    for order in ([near_left, near_right], [near_right, near_left]):
        left, right = assign_hands_to_wrists(order, _pose(), FRAME)
        assert left is near_left and right is near_right


def test_single_hand_goes_to_nearest_wrist() -> None:
    hand = _hand((0.32, 0.52))
    assert assign_hands_to_wrists([hand], _pose(), FRAME) == (None, hand)


def test_far_away_hand_rejected() -> None:
    # 0.25 frame heights away from the nearest wrist (max is 0.2).
    hand = _hand((0.3, 0.75))
    assert assign_hands_to_wrists([hand], _pose(), FRAME) == (None, None)


def test_distance_uses_frame_aspect_ratio() -> None:
    # 0.14 in normalized x on a 4:3 frame is 0.187 frame heights: accepted.
    # On a 16:9 frame it is 0.249 frame heights: rejected.
    hand = _hand((0.44, 0.5))
    assert assign_hands_to_wrists([hand], _pose(), (640, 480))[1] is hand
    assert assign_hands_to_wrists([hand], _pose(), (1280, 720))[1] is None


def test_extra_hands_ignored() -> None:
    hands = [_hand((0.7, 0.5)), _hand((0.3, 0.5)), _hand((0.5, 0.5))]
    left, right = assign_hands_to_wrists(hands, _pose(), FRAME)
    assert left is hands[0] and right is hands[1]
