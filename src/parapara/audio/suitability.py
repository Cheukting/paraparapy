"""Is a track suitable for Para Para? (checked on every song import)

Para Para is danced on a strict, steady 4/4 grid at 140-160 BPM (PLAN.md §2),
and the game's scoring needs a beat map accurate to a few milliseconds. A
track is **rejected** if any of these fails, with the reason shown:

======================  ============================  ==========================
check                   reject                        warn
======================  ============================  ==========================
tempo                   outside 135-165 BPM           outside 140-160 BPM
steady tempo (drift)    RMS > 15 ms off the grid      > 10 ms
beats on the grid       < 80 % of detected beats      -
beat clarity            < 1.3                         < 1.8
length                  < 40 beats                    < 72 beats
======================  ============================  ==========================

- *Tempo, drift, beats on the grid* come from the beat-map fit
  (:mod:`parapara.audio.beatmap`).
- *Beat clarity* = median kick-band onset strength **on** the beats divided
  by the median **half-way between** beats. Four-on-the-floor dance music has
  a strong kick on every beat (clarity well above 1.5); music without a
  steady kick stays near 1.
- *Length*: at least an 8-count intro plus one 32-count phrase.

Thresholds were set from measurements of 2026-10-18: the generated demo track
and CC BY dance tracks from Free Music Archive (``assets/songs/catalog.json``).
Clean dance tracks had drift ≤ 9 ms, ≥ 80 % of beats on the grid and clarity
≥ 1.5 (but a player found 1.6 hard to dance to, hence the 1.8 warning); tracks with tempo changes or no steady kick had drift ≥ 15 ms, as few
as a third of beats on the grid, and clarity ≈ 1.0.

Licences are **not** checked (decided 2026-10-18): imported songs are
assumed to be the user's own music for personal, local use. If a licence is
given, NC / ND / SA terms are noted as information only (see
:func:`license_warnings`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from parapara.audio import beatmap as bm

BPM_REJECT = (135.0, 165.0)
BPM_WARN = (140.0, 160.0)
DRIFT_REJECT_MS, DRIFT_WARN_MS = 15.0, 10.0
MIN_ON_GRID = 0.8
# Warn below 1.8: "broke" (clarity 1.61) passed the earlier 1.5 but was
# reported hard to dance to (2026-10-18); "Lets Go To A Rave" (1.94) was fine.
CLARITY_REJECT, CLARITY_WARN = 1.3, 1.8
MIN_BEATS, SHORT_BEATS = 40, 72


@dataclass(frozen=True)
class TrackAnalysis:
    bpm: float
    first_beat_s: float
    drift_ms: float
    beats_used: int
    beats_detected: int
    beat_clarity: float
    duration_s: float

    @property
    def on_grid_share(self) -> float:
        return self.beats_used / self.beats_detected if self.beats_detected else 0.0

    @property
    def duration_beats(self) -> int:
        return max(0, int((self.duration_s - self.first_beat_s) * self.bpm / 60.0))


@dataclass(frozen=True)
class SuitabilityReport:
    analysis: TrackAnalysis | None
    rejections: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.rejections


def beat_clarity(onset_env: npt.ArrayLike, frames_per_s: float, first_beat_s: float,
                 bpm: float) -> float:  # fmt: skip
    """Median onset strength on the beats / half-way between (see module docstring)."""
    env = np.asarray(onset_env, dtype=np.float64)
    period = 60.0 / bpm
    duration = len(env) / frames_per_s
    beats = first_beat_s + period * np.arange(
        max(0, int((duration - first_beat_s) / period))
    )

    def strength(times: np.ndarray) -> np.ndarray:
        frames = np.round((times + bm.DETECTION_LAG_S) * frames_per_s).astype(int)
        frames = frames[(frames >= 1) & (frames < len(env) - 1)]
        return np.maximum.reduce(
            [env[frames - 1], env[frames], env[frames + 1]]
        )  # ±1 frame

    on, off = strength(beats), strength(beats + period / 2)
    if len(on) == 0 or len(off) == 0:
        return 0.0
    return float(np.median(on) / (np.median(off) + 1e-9))


def analyze_track(audio: Path) -> TrackAnalysis:
    """Beat-map fit, beat clarity and duration of an audio file (librosa)."""
    import librosa  # tools dependency

    y, sr = librosa.load(audio, sr=bm.ANALYSIS_SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=bm.HOP,
                                       fmax=bm.ONSET_FMAX_HZ, n_mels=bm.ONSET_N_MELS)  # fmt: skip
    tempo, beats = bm.detect_beats(audio)
    fit = bm.fit_constant_tempo(beats, bpm_hint=tempo)
    clarity = beat_clarity(env, sr / bm.HOP, fit.first_beat_s, fit.bpm)
    return TrackAnalysis(fit.bpm, fit.first_beat_s, fit.drift_ms, fit.beats_used,
                         fit.beats_detected, clarity, len(y) / sr)  # fmt: skip


def license_warnings(license_name: str) -> list[str]:
    name = license_name.upper().replace("-", " ")
    out = []
    if "NC" in name.split() or "NONCOMMERCIAL" in name.replace(" ", ""):
        out.append(
            "non-commercial licence: fine for private play, not for a commercial release"
        )
    if "ND" in name.split() or "NODERIV" in name.replace(" ", ""):
        out.append("no-derivatives licence: play the track unmodified only")
    if "SA" in name.split() or "SHAREALIKE" in name.replace(" ", ""):
        out.append("share-alike licence: changed versions must use the same licence")
    return out


def judge_suitability(
    analysis: TrackAnalysis | None, license_name: str = ""
) -> SuitabilityReport:
    """Apply the table in the module docstring to an analysis."""
    rejections: list[str] = []
    warnings: list[str] = license_warnings(license_name) if license_name.strip() else []
    if analysis is None:
        return SuitabilityReport(
            None,
            tuple(rejections + ["no steady beat could be detected"]),
            tuple(warnings),
        )

    a = analysis
    if not BPM_REJECT[0] <= a.bpm <= BPM_REJECT[1]:
        rejections.append(f"tempo {a.bpm:.1f} BPM: Para Para needs about 140-160 BPM")
    elif not BPM_WARN[0] <= a.bpm <= BPM_WARN[1]:
        warnings.append(f"tempo {a.bpm:.1f} BPM is outside the ideal 140-160 BPM")
    if a.drift_ms > DRIFT_REJECT_MS:
        rejections.append(f"tempo not steady: beats drift {a.drift_ms:.0f} ms from a constant grid "
                          "(tempo changes, rubato or unclear beat)")  # fmt: skip
    elif a.drift_ms > DRIFT_WARN_MS:
        warnings.append(
            f"beats drift {a.drift_ms:.0f} ms from a constant grid; check by ear"
        )
    if a.on_grid_share < MIN_ON_GRID:
        rejections.append(
            f"only {100 * a.on_grid_share:.0f}% of beats follow one constant tempo"
        )
    if a.beat_clarity < CLARITY_REJECT:
        rejections.append(
            f"beat not clear enough (clarity {a.beat_clarity:.2f}): needs a steady kick on every beat"
        )
    elif a.beat_clarity < CLARITY_WARN:
        warnings.append(
            f"beat clarity {a.beat_clarity:.2f} is low; the beat may be hard to follow"
        )
    if a.duration_beats < MIN_BEATS:
        rejections.append(
            f"too short: {a.duration_beats} beats (needs at least {MIN_BEATS})"
        )
    elif a.duration_beats < SHORT_BEATS:
        warnings.append(f"short track: {a.duration_beats} beats")
    metrics = asdict(a) | {
        "on_grid_share": round(a.on_grid_share, 3),
        "duration_beats": a.duration_beats,
    }
    return SuitabilityReport(a, tuple(rejections), tuple(warnings), metrics)


def check_track(audio: Path, license_name: str = "") -> SuitabilityReport:
    """Analyze ``audio`` and judge it; undetectable beats are a rejection."""
    try:
        analysis = analyze_track(audio)
    except ValueError:  # fewer than 4 beats: no grid to fit
        analysis = None
    return judge_suitability(analysis, license_name)
