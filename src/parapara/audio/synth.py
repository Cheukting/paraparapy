"""A procedurally generated "Eurobeat-like" demo track for development.

**For machine testing only, not for dancing** (decided 2026-10-18): the
track is generated entirely in code, so its beat times are known *exactly*,
which makes it the ground truth for automated tests of beat detection
(``parapara-beatmap``), the suitability check, song import and the beat-map
check app (see :mod:`parapara.audio.demo_song`). It is not meant to sound
like real Eurobeat and is not in the game's song library.

Arrangement at ``bpm`` (default 150, inside the game's 140-160 range):

- a short silent **lead-in** (so the first beat is not at time 0, which
  tests offset handling);
- **intro** (8 counts): drums and bass;
- **main** (``phrases`` × 32 counts, where the routine is danced): drums,
  bass and a synth lead arpeggio;
- **outro** (8 counts): drums only.

Instruments (all synthesized with NumPy):

- kick on every beat (four-on-the-floor), pitch sweep 120 → 45 Hz;
- off-beat hi-hat (filtered noise);
- clap on counts 2 and 4 of every measure, like many Eurobeat tracks; also
  realistic test material for clap detection, which must not mistake the
  song's claps for the player's (PLAN.md §5.1);
- the typical Eurobeat octave-jumping bass (root on the beat, octave on the
  "and"), chords Am - F - G - E, one per 4 counts;
- a saw-wave lead playing 16th-note chord arpeggios.

Output: mono float32 samples, written as 16-bit PCM WAV with the standard
library's :mod:`wave` (no extra dependency). The generator is deterministic
(fixed noise seed), so regenerating yields an identical file.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 44100
COUNTS_PER_PHRASE = 32

# Chords (MIDI root, minor/major third), one per 4 counts.
_PROGRESSION = ((57, 3), (53, 4), (55, 4), (52, 4))  # A minor, F, G, E


def _midi_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12)


@dataclass(frozen=True)
class DemoSong:
    """A rendered demo track and its exact beat map."""

    samples: npt.NDArray[np.float32]
    sample_rate: int
    bpm: float
    first_beat_s: float  # time of beat 0 (the first intro count)
    routine_start_beat: int  # first count of the main section
    duration_beats: int
    sections: dict[str, tuple[int, int]] = field(
        default_factory=dict
    )  # name -> (start, end) beat

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm

    def beat_time(self, beat: float) -> float:
        return self.first_beat_s + beat * self.beat_s

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


class _Mixer:
    """Places short sounds into one long buffer."""

    def __init__(self, length_s: float, sr: int) -> None:
        self.sr = sr
        self.buf = np.zeros(int(length_s * sr) + sr, dtype=np.float64)

    def add(self, sound: np.ndarray, t_s: float, gain: float = 1.0) -> None:
        start = int(round(t_s * self.sr))
        end = min(start + len(sound), len(self.buf))
        if start < end:
            self.buf[start:end] += gain * sound[: end - start]


def _env(n: int, sr: int, decay_s: float, attack_s: float = 0.002) -> np.ndarray:
    t = np.arange(n) / sr
    return np.minimum(1.0, t / attack_s) * np.exp(-t / decay_s)


def _kick(sr: int) -> np.ndarray:
    t = np.arange(int(0.25 * sr)) / sr
    freq = 45 + 75 * np.exp(-t / 0.03)  # 120 Hz -> 45 Hz sweep
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * _env(len(t), sr, 0.09)


def _hat(sr: int, rng: np.random.Generator) -> np.ndarray:
    noise = np.diff(
        rng.standard_normal(int(0.05 * sr) + 1)
    )  # differencing = crude high-pass
    return noise * _env(len(noise), sr, 0.012) * 0.25


def _clap(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.18 * sr)
    noise = rng.standard_normal(n)
    noise = noise - np.convolve(noise, np.ones(8) / 8, mode="same")  # take off the lows
    env = np.zeros(n)
    for burst_ms in (0, 9, 18):  # three quick hand slaps, then a short tail
        b = int(burst_ms / 1000 * sr)
        env[b:] += _env(n - b, sr, 0.006 if burst_ms < 18 else 0.05)
    return noise * env * 0.35


def _saw(
    freq: float, dur_s: float, sr: int, decay_s: float, brightness: float = 1.0
) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    wave_ = 2.0 * ((t * freq) % 1.0) - 1.0
    # One-pole low-pass to tame the edges (brightness 1 = open).
    alpha = np.clip(brightness, 0.01, 1.0)
    out = np.empty_like(wave_)
    acc = 0.0
    for i, x in enumerate(wave_):
        acc += alpha * (x - acc)
        out[i] = acc
    return out * _env(len(t), sr, decay_s)


def render_demo_song(
    bpm: float = 150.0,
    phrases: int = 2,
    intro_counts: int = 8,
    outro_counts: int = 8,
    lead_in_s: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 7,
) -> DemoSong:
    """Render the demo track (see module docstring)."""
    rng = np.random.default_rng(seed)
    beat_s = 60.0 / bpm
    main_counts = phrases * COUNTS_PER_PHRASE
    total = intro_counts + main_counts + outro_counts
    mix = _Mixer(lead_in_s + total * beat_s, sample_rate)
    kick, clap = _kick(sample_rate), _clap(sample_rate, rng)
    hats = [_hat(sample_rate, rng) for _ in range(8)]

    def t(beat: float) -> float:
        return lead_in_s + beat * beat_s

    main_start, main_end = intro_counts, intro_counts + main_counts
    for b in range(total):
        mix.add(kick, t(b), 0.9)
        mix.add(hats[b % 8], t(b + 0.5))
        if b % 2 == 1:  # counts 2 and 4 of each measure
            mix.add(clap, t(b), 0.8)
        if b >= main_end:
            continue  # outro: drums only
        root, third = _PROGRESSION[(b // 4) % len(_PROGRESSION)]
        bass_root = root - 24
        mix.add(
            _saw(_midi_hz(bass_root), beat_s * 0.45, sample_rate, 0.12, 0.25),
            t(b),
            0.35,
        )
        mix.add(
            _saw(_midi_hz(bass_root + 12), beat_s * 0.45, sample_rate, 0.12, 0.25),
            t(b + 0.5),
            0.35,
        )
        if b >= main_start:
            arpeggio = (root + 12, root + 12 + third, root + 19, root + 24)
            for k, note in enumerate(arpeggio):
                mix.add(_saw(_midi_hz(note), beat_s / 4 * 0.9, sample_rate, 0.08, 0.5),
                        t(b + k / 4), 0.12)  # fmt: skip

    samples = mix.buf[: int(round(t(total) * sample_rate))]
    samples = samples / (np.max(np.abs(samples)) / 0.89)  # peak at about -1 dBFS
    return DemoSong(
        samples=samples.astype(np.float32),
        sample_rate=sample_rate,
        bpm=bpm,
        first_beat_s=lead_in_s,
        routine_start_beat=main_start,
        duration_beats=total,
        sections={
            "intro": (0, main_start),
            "main": (main_start, main_end),
            "outro": (main_end, total),
        },
    )


def write_wav(path: Path, samples: npt.ArrayLike, sample_rate: int) -> None:
    """Write mono float samples in [-1, 1] as 16-bit PCM WAV."""
    data = (np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0) * 32767).astype(
        "<i2"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data.tobytes())


def read_wav(path: Path) -> tuple[npt.NDArray[np.float32], int]:
    """Read a mono or stereo 16-bit PCM WAV as mono float32 samples."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        sr, channels = w.getframerate(), w.getnchannels()
        data = (
            np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
            / 32767
        )
    return data.reshape(-1, channels).mean(axis=1), sr
