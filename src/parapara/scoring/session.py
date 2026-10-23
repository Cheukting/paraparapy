"""One play of a routine: features → judgments → score (the game's core loop).

:class:`PlaySession` wires the pieces together for one song and routine, so
the game screen only has to feed camera samples and draw:

- :class:`~parapara.pose.features.FeatureExtractor` (filters, body frame,
  calibration-relative posture);
- :class:`~parapara.scoring.judge.Judge` on the schedule of the routine on the
  song's beat clock **shifted by the measured latency**;
- :class:`~parapara.scoring.scorekeeper.ScoreKeeper` (points, combo, posture);
- :class:`~parapara.pose.framing.FramingMonitor` (warn when the player leaves
  the picture).

Every sample is kept so the play can be saved as a
:class:`~parapara.pose.recording.Recording` (with the schedule's beat times)
and examined later with ``parapara-score`` / ``parapara-analyze``.

Pure Python + NumPy, so it is tested without camera or pygame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from parapara.audio.song import Song
from parapara.pose.calibration import Calibration
from parapara.pose.features import FeatureExtractor, PoseFeatures
from parapara.pose.framing import FramingMonitor, FramingResult
from parapara.pose.landmarks import PoseSample
from parapara.pose.recording import Recording
from parapara.routine.model import Routine
from parapara.scoring.judge import FlowJudgment, Judge, Judgment
from parapara.scoring.schedule import BeatClock, schedule_routine
from parapara.scoring.scorekeeper import ScoreKeeper


@dataclass(frozen=True)
class ScoreEvent:
    """A judgment and the points it earned (for on-screen pop-ups)."""

    judgment: Judgment | FlowJudgment
    points: int


@dataclass
class PlaySession:
    """Score one play (see module docstring).

    Args:
        song, routine: what is danced.
        playback_start_ns: ``time.monotonic_ns()`` when audio time 0 played.
        latency_ns: from the latency calibration.
        calibration: player calibration (posture relative to neutral).
    """

    song: Song
    routine: Routine
    playback_start_ns: int
    latency_ns: int = 0
    calibration: Calibration | None = None
    samples: list[PoseSample] = field(default_factory=list)
    finished: bool = False
    ended_early: bool = False

    def __post_init__(self) -> None:
        self.clock: BeatClock = self.song.routine_clock(
            self.playback_start_ns, self.latency_ns
        )
        self.schedule = schedule_routine(self.routine, self.clock)
        self.extractor = FeatureExtractor(calibration=self.calibration)
        self.judge = Judge(self.schedule)
        self.keeper = ScoreKeeper(calibrated=self.calibration is not None)
        self.framing = FramingMonitor()
        self.latest_features: PoseFeatures | None = None

    @property
    def end_ns(self) -> int:
        """When the last keyframe's window has certainly closed."""
        return self.clock.time_of(self.routine.total_counts + 1)

    def add(self, sample: PoseSample) -> list[ScoreEvent]:
        """Feed one camera sample; returns the judgments it completed."""
        if self.finished:
            return []
        self.samples.append(sample)
        self.framing.update(sample)
        features = self.extractor.update(sample)
        self.latest_features = features
        in_routine = self.clock.time_of(0) <= sample.captured_ns <= self.end_ns
        if in_routine:
            self.keeper.add_posture(features)
        return self._score(self.judge.add(features, now_ns=sample.captured_ns))

    def finish(self, now_ns: int | None = None) -> list[ScoreEvent]:
        """Judge whatever is left.

        At the end of the song call it without arguments. If the player stops
        early, pass the current time: keyframes not yet due are dropped
        instead of counting as misses, and the summary says ``ended_early``.
        """
        if self.finished:
            return []
        self.finished = True
        until = None
        if now_ns is not None and now_ns < self.end_ns:
            self.ended_early, until = True, now_ns
        return self._score(self.judge.finish(until))

    def _score(self, judgments: list[Judgment | FlowJudgment]) -> list[ScoreEvent]:
        return [ScoreEvent(j, self.keeper.add(j)) for j in judgments]

    @property
    def framing_warning(self) -> FramingResult | None:
        """The latest framing result if the player is (partly) out of the picture."""
        latest = self.framing.latest
        if latest is not None and latest.codes & {
            "no_person",
            "no_shoulders",
            "no_hips",
        }:
            return latest
        return None

    def summary(self) -> dict[str, Any]:
        return {"song_id": self.song.id, "routine_id": self.routine.id,
                "difficulty": self.routine.difficulty.value, "latency_ms": self.latency_ns / 1e6,
                "calibrated": self.calibration is not None, "ended_early": self.ended_early,
                **self.keeper.summary()}  # fmt: skip

    def recording(self) -> Recording:
        """The play as a recording (for ``parapara-score`` / ``-analyze``)."""
        return Recording.from_samples(self.samples, {
            "clip_id": f"play-{self.routine.id}", "kind": "play", "song_id": self.song.id,
            "routine_id": self.routine.id, "bpm": self.song.bpm,
            "latency_ns": self.latency_ns, "playback_start_ns": self.playback_start_ns,
            "keyframe_times_ns": [s.t_ns for s in self.schedule], "summary": self.summary(),
        })  # fmt: skip
