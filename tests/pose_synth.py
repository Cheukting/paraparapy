"""Synthetic MediaPipe world landmarks built from body-frame arm poses.

Test helper: the inverse of ``parapara.pose.body_frame``. It places a
skeleton in MediaPipe's world axes (+x = dancer's left, +y = down, +z = away
from the camera, origin at the hip midpoint; measured 2026-10-18) with arm
segments pointing along given body-frame directions, optionally with the
torso rotated (leaning / turned), so tests can check that analysis recovers
exactly what was put in.
"""

import numpy as np

from parapara.pose.landmarks import NUM_POSE_LANDMARKS
from parapara.pose.landmarks import PoseLandmark as P
from parapara.pose.recording import Recording
from parapara.routine.geometry import DOWN, angle_between_deg, slerp
from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import ArmPose

SHOULDER_HALF = 0.175  # metres
TORSO = 0.47
UPPER, FORE, HAND = 0.22, 0.22, 0.08

# Body-frame axes (right, up, forward) in MediaPipe coordinates, unrotated.
BODY_TO_MP = np.array(
    [[-1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]
)  # rows: right, up, forward


def rotation(lean_deg: float = 0.0, turn_deg: float = 0.0) -> np.ndarray:
    """Rotation of the torso in MediaPipe coordinates: lean towards the
    camera (about x), then turn (about the vertical y axis)."""
    a, b = np.radians(lean_deg), np.radians(turn_deg)
    lean = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    turn = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return turn @ lean


def world_pose(
    right=None, left=None, lean_deg=0.0, turn_deg=0.0, hips_visible=True
) -> np.ndarray:
    """One (33, 5) MediaPipe world pose. ``right``/``left`` are ArmPose-like
    objects (``upper``, ``fore``, optional ``hand``) or ``None`` (arm down)."""
    rot = rotation(lean_deg, turn_deg)
    axes = BODY_TO_MP @ rot.T  # rows: body right/up/forward in MediaPipe coords

    def mp(v) -> np.ndarray:  # body-frame direction -> MediaPipe vector
        return np.asarray(v, dtype=np.float64) @ axes

    pose = np.zeros((NUM_POSE_LANDMARKS, 5))
    pose[:, 3:] = 1.0
    up = mp((0, 1, 0)) * TORSO
    for side, sign, (s, e, w, ix, pk, h) in (
        ("right", 1, (P.RIGHT_SHOULDER, P.RIGHT_ELBOW, P.RIGHT_WRIST, P.RIGHT_INDEX, P.RIGHT_PINKY, P.RIGHT_HIP)),
        ("left", -1, (P.LEFT_SHOULDER, P.LEFT_ELBOW, P.LEFT_WRIST, P.LEFT_INDEX, P.LEFT_PINKY, P.LEFT_HIP)),
    ):  # fmt: skip
        arm = right if side == "right" else left
        upper = getattr(arm, "upper", (0, -1, 0))
        fore = getattr(arm, "fore", (0, -1, 0))
        hand = getattr(arm, "hand", None) or fore
        unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)  # noqa: E731
        pose[h, :3] = mp((sign * 0.11, 0, 0))
        pose[s, :3] = up + mp((sign * SHOULDER_HALF, 0, 0))
        pose[e, :3] = pose[s, :3] + mp(unit(upper)) * UPPER
        pose[w, :3] = pose[e, :3] + mp(unit(fore)) * FORE
        pose[ix, :3] = pose[pk, :3] = pose[w, :3] + mp(unit(hand)) * HAND
        if not hips_visible:
            pose[h, 3] = 0.1
    pose[P.NOSE, :3] = up + mp((0, 0.18, 0.05))  # head above the shoulders
    return pose.astype(np.float32)


def recording(
    poses: list[np.ndarray],
    times_ns: list[int],
    metadata: dict,
    images: list[np.ndarray] | None = None,
) -> Recording:
    """A Recording of world ``poses``; image landmarks default to a copy of the
    world ones (fine unless the framing check is involved: pass ``images``)."""
    n = len(poses)
    arr = np.stack(poses).astype(np.float32)
    image_arr = (
        np.stack(images).astype(np.float32) if images is not None else arr.copy()
    )
    return Recording(
        frame_index=np.arange(n, dtype=np.int64),
        captured_ns=np.asarray(times_ns, dtype=np.int64),
        processed_ns=np.asarray(times_ns, dtype=np.int64) + 20_000_000,
        pose_ms=np.full(n, 20.0, np.float32),
        has_pose=np.ones(n, bool),
        pose_image=image_arr,
        pose_world=arr,
        metadata={"frame_size": [640, 480], **metadata},
    )


def image_pose(
    mid_x=0.52, shoulder_y=0.49, shoulder_w=0.19, hip_y=0.87, hips_visible=True
) -> np.ndarray:
    """(33, 5) normalized image landmarks of a player standing upright, facing
    an unmirrored camera (their right shoulder at the smaller x). Defaults are
    the neutral pose measured on 2026-10-18; arms hang down."""
    pose = np.zeros((NUM_POSE_LANDMARKS, 5))
    pose[:, 3:] = 1.0
    pose[:, :2] = (mid_x, shoulder_y)  # head and anything unset sit near the shoulders
    half = shoulder_w / 2
    for side, sign in (("RIGHT", -1), ("LEFT", 1)):
        x = mid_x + sign * half
        pose[P[f"{side}_SHOULDER"], :2] = (x, shoulder_y)
        pose[P[f"{side}_ELBOW"], :2] = (x + sign * 0.02, shoulder_y + 0.2)
        pose[P[f"{side}_WRIST"], :2] = (x + sign * 0.03, shoulder_y + 0.38)
        pose[P[f"{side}_HIP"], :2] = (mid_x + sign * half * 0.65, hip_y)
        if not hips_visible:
            pose[P[f"{side}_HIP"], 3] = 0.1
    pose[P.NOSE, :2] = (mid_x, shoulder_y - 0.15)
    return pose.astype(np.float32)


def danced_block(
    block_id: str, lag_s: float, bpm: float = 120, fps: float = 50, bars: int = 2
):
    """A recording of a dancer moving smoothly through ``block_id``: each arm
    segment turns at constant speed from one keyframe's direction to the
    next, arriving exactly ``lag_s`` after each keyframe's beat."""
    block = BLOCK_LIBRARY[block_id]
    beat_s = 60 / bpm
    count_in = 4
    first_beat_ns = 1_000_000_000
    beat_times = [
        first_beat_ns + int((i - count_in) * beat_s * 1e9)
        for i in range(count_in + bars * 8)
    ]
    times = np.arange(0, bars * 8 * beat_s, 1 / fps)
    kfs = block.keyframes

    def arm(kf, side):
        return getattr(kf, side) or ArmPose(upper=DOWN, fore=DOWN)

    poses = []
    for t in times:
        position = ((t - lag_s) / beat_s) % block.duration
        k = (
            max(i for i, kf in enumerate(kfs) if kf.beat <= position + 1e-9)
            if position >= kfs[0].beat
            else len(kfs) - 1
        )
        a, nxt = kfs[k], kfs[(k + 1) % len(kfs)]
        span = (nxt.beat - a.beat) % block.duration or block.duration
        frac = ((position - a.beat) % block.duration) / span
        both = {}
        for side in ("right", "left"):
            pa, pb = arm(a, side), arm(nxt, side)
            both[side] = ArmPose(upper=slerp_safe(pa.upper, pb.upper, frac),
                                 fore=slerp_safe(pa.fore, pb.fore, frac))  # fmt: skip
        poses.append(world_pose(both["right"], both["left"]))
    meta = {"clip_id": f"block_{block_id}", "kind": "block", "block_id": block_id, "bpm": bpm,
            "count_in_beats": count_in, "beat_times_ns": beat_times}  # fmt: skip
    return recording(poses, [first_beat_ns + int(t * 1e9) for t in times], meta)


def slerp_safe(a, b, t):
    """Interpolate directions; nearly opposite pairs go via a mid direction."""
    if angle_between_deg(a, b) > 170:
        return slerp(a, (0.0, 0.0, 1.0), t)
    return slerp(a, b, t)


def routine_play_samples(
    song, routine, playback_start_ns: int, lag_s: float = 0.0, fps: float = 30.0
):
    """PoseSamples of a perfect player: the demo dancer's own motion (which is
    exact on every scored keyframe), performed ``lag_s`` late, seen by a
    camera at ``fps`` from audio time 0 to the end of the routine + 1 bar."""
    from parapara.animation.animator import Animator
    from parapara.pose.landmarks import PoseSample

    animator = Animator(routine)
    end_s = song.beat_time_s(song.routine_start_beat + routine.total_counts + 8)
    image = image_pose()
    samples = []
    for i, t in enumerate(np.arange(0.0, end_s, 1.0 / fps)):
        beat = song.beat_at(t - lag_s) - song.routine_start_beat
        p = animator.pose_at(beat)
        arm = lambda s: ArmPose(upper=s.upper, fore=s.fore, hand=s.hand)  # noqa: E731
        world = world_pose(arm(p.right), arm(p.left))
        if p.clap_flash > 0.9:  # hands together on the clap
            world[16, :3] = world[15, :3]
        t_ns = playback_start_ns + int(t * 1e9)
        samples.append(
            PoseSample(
                i, t_ns, t_ns + 20_000_000, (640, 480), image, world, pose_ms=20.0
            )
        )
    return samples
