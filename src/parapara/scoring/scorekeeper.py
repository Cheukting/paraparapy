"""Turning judgments into a score (PLAN.md §5: "keyframe scores × combo
multiplier + posture bonus").

- **Keyframe points**: each :class:`~parapara.scoring.judge.Judgment` is worth
  up to :data:`POINTS_PER_KEYFRAME` × pose score × timing factor, times the
  current **combo multiplier**: +0.1 for every 10 consecutive non-miss
  keyframes, up to ×2.0. A miss resets the combo.
- **Flow bonus**: flowing blocks add up to :data:`FLOW_BONUS` × flow score.
- **Posture bonus** (the "strong, confident posture" requirement, PLAN.md §2):
  the share of samples with the torso upright (lean and side tilt within
  :data:`POSTURE_MAX_TILT_DEG` of the calibrated neutral) and, when
  calibrated, the head not dropped below the neutral height by more than
  :data:`HEAD_DROP_MAX`; worth up to 10% of the maximum keyframe points.
  Shoulder tilt is not used: raising an arm lifts its shoulder by design.
- **Accuracy**: mean keyframe points (0-100%), independent of combos, for the
  results screen and for comparing recordings.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from parapara.pose.features import PoseFeatures
from parapara.scoring.judge import FlowJudgment, Grade, Judgment

POINTS_PER_KEYFRAME = 100
FLOW_BONUS = 50
POSTURE_SHARE_OF_MAX = 0.10
POSTURE_MAX_TILT_DEG = 15.0
HEAD_DROP_MAX = 0.2  # shoulder widths
COMBO_STEP = 10
COMBO_INCREMENT = 0.1
MAX_MULTIPLIER = 2.0


@dataclass
class ScoreKeeper:
    """Accumulates judgments and posture samples into a score.

    Args:
        calibrated: whether features are relative to a player calibration
            (enables the head-height check).
    """

    calibrated: bool = False
    score: int = 0
    combo: int = 0
    max_combo: int = 0
    grades: Counter = field(default_factory=Counter)
    _points: list[float] = field(default_factory=list)
    _flow: list[float] = field(default_factory=list)
    _posture_ok: int = 0
    _posture_n: int = 0

    @property
    def multiplier(self) -> float:
        return min(MAX_MULTIPLIER, 1.0 + COMBO_INCREMENT * (self.combo // COMBO_STEP))

    def add(self, judgment: Judgment | FlowJudgment) -> int:
        """Add a judgment; returns the points it earned (for on-screen popups)."""
        if isinstance(judgment, FlowJudgment):
            self._flow.append(judgment.score)
            earned = round(FLOW_BONUS * judgment.score)
        else:
            self.grades[judgment.grade] += 1
            self._points.append(judgment.points)
            if judgment.grade is Grade.MISS:
                self.combo = 0
            else:
                self.combo += 1
                self.max_combo = max(self.max_combo, self.combo)
            earned = round(POINTS_PER_KEYFRAME * judgment.points * self.multiplier)
        self.score += earned
        return earned

    def add_posture(self, features: PoseFeatures | None) -> None:
        if features is None:
            return
        ok = abs(features.lean_deg) <= POSTURE_MAX_TILT_DEG and abs(features.side_tilt_deg) <= POSTURE_MAX_TILT_DEG
        if self.calibrated:
            ok = ok and features.head_height >= -HEAD_DROP_MAX
        self._posture_n += 1
        self._posture_ok += ok

    @property
    def posture_share(self) -> float:
        return self._posture_ok / self._posture_n if self._posture_n else 0.0

    @property
    def posture_bonus(self) -> int:
        return round(POSTURE_SHARE_OF_MAX * POINTS_PER_KEYFRAME * len(self._points) * self.posture_share)

    def summary(self) -> dict[str, Any]:
        n = len(self._points)
        return {
            "score": self.score + self.posture_bonus,
            "keyframe_score": self.score,
            "posture_bonus": self.posture_bonus,
            "accuracy_pct": round(100 * sum(self._points) / n, 1) if n else 0.0,
            "keyframes": n,
            "grades": {g.value: self.grades.get(g, 0) for g in Grade},
            "max_combo": self.max_combo,
            "posture_pct": round(100 * self.posture_share, 1),
            "flow_pct": round(100 * sum(self._flow) / len(self._flow), 1) if self._flow else None,
        }
