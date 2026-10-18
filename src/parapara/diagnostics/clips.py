"""The guided recording script: which clips to record and their timing.

Used by the guided recorder (``uv run parapara-record``). Kept free of
``pygame`` so the timing logic is unit-testable.

Clips
-----
- **Static clips** (hold a pose for a few seconds after a countdown):
  ``neutral`` and ``t_pose`` feed the player calibration (PLAN.md §3);
  ``right_arm_up`` and ``right_arm_forward`` pin down the direction of
  MediaPipe's world axes (up and towards the camera), which the docs leave
  unspecified. Together with the T-pose (sideways) they cover all three axes.
- **Block clips** (dance one building block to a metronome): a count-in, then
  the block repeated for ``bars`` 8-count bars. The guide shows the target
  keyframe of the current count, and the beat times are saved with the clip so
  Step 4 can test timing against them.

All times in a :class:`ClipTimeline` are seconds relative to the moment the
player starts the clip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from parapara.routine.geometry import DOWN, FORWARD, LEFT, RIGHT, UP
from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import COUNTS_PER_BAR, ArmPose, Block, Keyframe

_ARM_DOWN = ArmPose(upper=DOWN, fore=DOWN)


@dataclass(frozen=True, slots=True)
class Clip:
    """One clip of the recording script.

    Attributes:
        id: file name stem, e.g. ``"t_pose"`` or ``"block_chop_right"``.
        title, instructions: shown in the guide panel.
        kind: ``"static"`` (hold ``guide`` for ``seconds``) or ``"block"``
            (dance ``block_id`` for ``bars`` bars to the metronome).
    """

    id: str
    title: str
    instructions: tuple[str, ...]
    kind: Literal["static", "block"]
    guide: Keyframe | None = None
    block_id: str | None = None
    bars: int = 2

    @property
    def block(self) -> Block | None:
        return None if self.block_id is None else BLOCK_LIBRARY[self.block_id]


def _static(id: str, title: str, instructions: tuple[str, ...], right: ArmPose, left: ArmPose) -> Clip:
    return Clip(id, title, instructions, "static", guide=Keyframe(beat=0, right=right, left=left))


def _block(block_id: str, title: str, *instructions: str) -> Clip:
    return Clip(f"block_{block_id}", title, instructions, "block", block_id=block_id)


CLIPS: tuple[Clip, ...] = (
    _static("neutral", "Neutral pose", ("Stand relaxed, facing the camera.",
            "Arms hanging down by your sides."), _ARM_DOWN, _ARM_DOWN),
    _static("t_pose", "T-pose", ("Both arms straight out to the sides,",
            "level with your shoulders."),
            ArmPose(upper=RIGHT, fore=RIGHT), ArmPose(upper=LEFT, fore=LEFT)),
    _static("right_arm_up", "Right arm up", ("Raise your RIGHT arm straight up.",
            "Left arm relaxed."), ArmPose(upper=UP, fore=UP), _ARM_DOWN),
    _static("right_arm_forward", "Right arm forward", ("Point your RIGHT arm straight",
            "at the camera, at shoulder height.", "(It looks short from the front.)"),
            ArmPose(upper=FORWARD, fore=FORWARD), _ARM_DOWN),
    _block("sweep_to_right_4", "Sweep to the right", "Both arms parallel, sweep from",
           "your left to your right over 4 counts."),
    _block("sweep_down_4", "Sweep down", "Both arms parallel, sweep from",
           "high to low over 4 counts."),
    _block("knock_right", "Knock (right)", "Right fist knocks twice: upper arm",
           "level, forearm up. Like knocking on a door."),
    _block("chop_right", "Chop (right)", "Count 1: right arm 45° up and out.",
           "Count 2: fold in front of the chest, like a salute."),
    _block("wrist_roll_4", "Wrist roll", "Hands in front of the chest,",
           "rolling over each other every half count."),
    _block("flick_clap_to_right", "Flick + Clap", "Right hand flicks: crossed, front,",
           "right side (counts 1-3), then CLAP (4)."),
)  # fmt: skip
"""The default recording script, in recording order."""

CLIP_IDS = {clip.id: clip for clip in CLIPS}


@dataclass(frozen=True, slots=True)
class Phase:
    """What the recorder should show at a given time.

    Attributes:
        name: ``"countdown"``, ``"count_in"``, ``"recording"`` or ``"done"``.
        label: big text for the guide panel (e.g. ``"3"``, ``"5"``, ``"REC"``).
        guide: target pose to draw, if any.
        bar, count: musical position while recording a block clip (1-based).
        progress: fraction of the recording completed, ``0..1``.
    """

    name: Literal["countdown", "count_in", "recording", "done"]
    label: str = ""
    guide: Keyframe | None = None
    bar: int = 0
    count: int = 0
    progress: float = 0.0


@dataclass(frozen=True, slots=True)
class ClipTimeline:
    """Timing of one clip run.

    Args:
        clip: the clip.
        bpm: metronome tempo for block clips.
        countdown_s: countdown before a static clip.
        static_seconds: how long a static pose is recorded.
        count_in_beats: metronome beats before a block clip's recording.
    """

    clip: Clip
    bpm: float = 120.0
    countdown_s: float = 3.0
    static_seconds: float = 4.0
    count_in_beats: int = 4

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm

    @property
    def recording_start(self) -> float:
        if self.clip.kind == "static":
            return self.countdown_s
        return self.count_in_beats * self.beat_s

    @property
    def recording_end(self) -> float:
        if self.clip.kind == "static":
            return self.countdown_s + self.static_seconds
        return self.recording_start + self.clip.bars * COUNTS_PER_BAR * self.beat_s

    def beats(self) -> list[tuple[float, bool]]:
        """Metronome clicks as ``(time_s, accented)``; empty for static clips.

        Accented clicks mark the first count-in beat and count 1 of every bar.
        """
        if self.clip.kind == "static":
            return []
        total = self.count_in_beats + self.clip.bars * COUNTS_PER_BAR
        clicks = []
        for i in range(total):
            recording_index = i - self.count_in_beats
            accent = i == 0 or (recording_index >= 0 and recording_index % COUNTS_PER_BAR == 0)
            clicks.append((i * self.beat_s, accent))
        return clicks

    def phase_at(self, t: float) -> Phase:
        if t >= self.recording_end:
            return Phase("done", progress=1.0)
        if t < self.recording_start:
            if self.clip.kind == "static":
                return Phase("countdown", str(math.ceil(self.countdown_s - t)), self.clip.guide)
            beat = int(t / self.beat_s)
            label = str(self.count_in_beats - beat)  # 4, 3, 2, 1
            first = self._keyframe_at(0.0)
            return Phase("count_in", label, first)
        progress = (t - self.recording_start) / (self.recording_end - self.recording_start)
        if self.clip.kind == "static":
            return Phase("recording", "REC", self.clip.guide, progress=progress)
        position = (t - self.recording_start) / self.beat_s  # counts since recording start
        index = int(position)
        bar, count = index // COUNTS_PER_BAR + 1, index % COUNTS_PER_BAR + 1
        return Phase("recording", str(count), self._keyframe_at(position), bar, count, progress)

    def _keyframe_at(self, position: float) -> Keyframe | None:
        """Target keyframe at ``position`` counts into the repeated block: the
        latest keyframe at or before the position within the block."""
        block = self.clip.block
        if block is None:
            return None
        within = position % block.duration
        current = block.keyframes[0]
        for kf in block.keyframes:
            if kf.beat <= within + 1e-9:
                current = kf
        return current
