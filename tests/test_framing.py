"""Tests for parapara.pose.framing (the position check before calibration)."""

import pytest
from pose_synth import image_pose, world_pose

from parapara.pose.framing import FramingMonitor, check_framing
from parapara.pose.landmarks import PoseSample


def _sample(image=None, world=None, size=(640, 480)) -> PoseSample:
    return PoseSample(0, 0, 0, size, image, world)


def test_recorded_neutral_position_is_ok() -> None:
    result = check_framing(_sample(image_pose(), world_pose()))
    assert result.ok and result.issues == ()
    x0, y0, x1, y1 = result.needed_box
    assert 0.03 <= x0 < x1 <= 0.97 and 0.03 <= y0 < y1 <= 0.97
    assert result.shoulder_width == pytest.approx(0.19, abs=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"hip_y": 1.02}, "no_hips"),  # as in the 2026-10-18 second session
        ({"hips_visible": False}, "no_hips"),
        ({"shoulder_w": 0.08}, "too_far"),
        ({"shoulder_w": 0.35}, "too_close"),
        # Player on the picture's left = their own right: they must move LEFT.
        ({"mid_x": 0.33}, "move_left"),
        ({"mid_x": 0.67}, "move_right"),
        ({"shoulder_y": 0.25, "hip_y": 0.6}, "no_headroom"),
    ],
)
def test_framing_issues(kwargs, code) -> None:
    result = check_framing(_sample(image_pose(**kwargs), world_pose()))
    assert not result.ok
    assert code in result.codes, result.issues


def test_turned_player() -> None:
    assert (
        "turned" in check_framing(_sample(image_pose(), world_pose(turn_deg=30))).codes
    )
    assert check_framing(_sample(image_pose(), world_pose(turn_deg=10))).ok


def test_no_person_and_no_shoulders() -> None:
    assert check_framing(_sample()).codes == {"no_person"}
    pose = image_pose()
    pose[11:13, 3] = 0.0  # shoulders invisible
    assert check_framing(_sample(pose)).codes == {"no_shoulders"}


def test_messages_use_the_players_own_left_and_right() -> None:
    (issue,) = check_framing(_sample(image_pose(mid_x=0.33), world_pose())).issues
    assert "LEFT" in issue.message


def test_monitor_needs_a_stable_run_of_good_samples() -> None:
    good, bad = (
        _sample(image_pose(), world_pose()),
        _sample(image_pose(hip_y=1.05), world_pose()),
    )
    monitor = FramingMonitor(window=3)
    for s in (good, good):
        monitor.update(s)
    assert not monitor.stable  # not enough samples yet
    monitor.update(good)
    assert monitor.stable and monitor.latest.ok
    monitor.update(bad)
    assert not monitor.stable and "no_hips" in monitor.latest.codes
    monitor.reset()
    assert monitor.latest is None and not monitor.stable
