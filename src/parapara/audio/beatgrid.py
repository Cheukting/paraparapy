"""Editable beat grid and click-track rendering for the beat-map check.

Pure NumPy, used by ``parapara-beatcheck`` (:mod:`parapara.game.beatcheck`):

- :class:`BeatGrid`: ``bpm``, ``first_beat_s`` and ``routine_start_beat`` with
  the nudges the check app offers (offset in ms, tempo in BPM, phrase start
  in whole beats);
- :func:`render_click_track`: the song with metronome clicks **mixed into the
  samples** at the grid's beat times. Clicks fired from the 60 Hz display loop
  would jitter by up to ~17 ms, audible as a flam and larger than the ±5 ms
  nudges; mixed into the audio they are sample-accurate. Count 1 of every
  8-count from the routine start gets a higher, accented click;
- :func:`low_band_envelope`: kick-drum energy over time, drawn under the grid
  lines so a misaligned beat map is visible as well as audible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

COUNTS_PER_BAR = 8


@dataclass(frozen=True)
class BeatGrid:
    bpm: float
    first_beat_s: float
    routine_start_beat: int

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm

    def time_of(self, beat: float) -> float:
        return self.first_beat_s + beat * self.beat_s

    def beat_at(self, t: float) -> float:
        return (t - self.first_beat_s) / self.beat_s

    def nudge_offset_ms(self, ms: float) -> BeatGrid:
        return replace(self, first_beat_s=max(0.0, self.first_beat_s + ms / 1000))

    def nudge_bpm(self, delta: float) -> BeatGrid:
        return replace(self, bpm=round(max(30.0, self.bpm + delta), 4))

    def shift_downbeat(self, beats: int) -> BeatGrid:
        return replace(self, routine_start_beat=max(0, self.routine_start_beat + beats))

    def is_bar_start(self, beat: int) -> bool:
        """Count 1 of an 8-count, aligned with the routine start."""
        return (beat - self.routine_start_beat) % COUNTS_PER_BAR == 0

    def beats_between(self, t0: float, t1: float) -> list[int]:
        """Beat indices (>= 0) whose time lies in ``[t0, t1)``."""
        first = max(0, math.ceil(self.beat_at(t0) - 1e-9))
        out = []
        b = first
        while self.time_of(b) < t1:
            out.append(b)
            b += 1
        return out


def _click(sr: int, pitch: float, dur_s: float = 0.03) -> np.ndarray:
    t = np.arange(int(sr * dur_s)) / sr
    return np.sin(2 * np.pi * pitch * t) * np.exp(-t * 150)


def render_click_track(samples: npt.ArrayLike, sr: int, grid: BeatGrid,
                       music_gain: float = 0.7, click_gain: float = 0.6) -> npt.NDArray[np.float32]:  # fmt: skip
    """Mono mix of the song and sample-accurate clicks on the grid's beats."""
    music = np.asarray(samples, dtype=np.float64) * music_gain
    out = music.copy()
    normal, accent = _click(sr, 1200.0), _click(sr, 1900.0)
    for beat in grid.beats_between(0.0, len(music) / sr):
        start = int(round(grid.time_of(beat) * sr))
        click = accent if grid.is_bar_start(beat) else normal
        end = min(start + len(click), len(out))
        out[start:end] += click_gain * click[: end - start]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def low_band_envelope(samples: npt.ArrayLike, sr: int, hop_s: float = 0.005,
                      cutoff_hz: float = 150.0) -> npt.NDArray[np.float32]:  # fmt: skip
    """Low-frequency (kick) energy per ``hop_s``, normalized to its 99th percentile."""
    x = np.asarray(samples, dtype=np.float64)
    width = max(1, int(sr / cutoff_hz))
    low = np.convolve(x, np.ones(width) / width, mode="same")  # moving average = low-pass
    hop = max(1, int(sr * hop_s))
    n = len(low) // hop
    energy = np.sqrt(np.mean(low[: n * hop].reshape(n, hop) ** 2, axis=1))
    scale = np.percentile(energy, 99) or 1.0
    return np.clip(energy / scale, 0.0, 1.0).astype(np.float32)
