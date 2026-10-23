"""Automatic beat maps: ``uv run parapara-beatmap <song dir>``.

1. **Detect beats** with librosa 1.0 (``librosa.onset.onset_strength`` +
   ``librosa.beat.beat_track``; API checked 2026-10-18), seeded at 150 BPM
   (Eurobeat; librosa's default prior is 120), with tempo octave errors
   (75 / 300 BPM) folded back into the 100-200 range.

   The onset strength uses only the **low band** (below :data:`ONSET_FMAX_HZ`,
   the kick drum). On the full spectrum the tracker locked onto the off-beat
   hi-hats and octave bass typical of Eurobeat: on the generated demo track
   beats came out ~150 ms off (mostly on the off-beats); with the low band
   they sit within 4 ms (std) of the true beats. Detection lags the attacks
   by a constant :data:`DETECTION_LAG_S` (measured on the demo track), which
   is subtracted.
2. **Fit one constant tempo**: Eurobeat does not change tempo, so the beat
   grid is ``t_i = first_beat + i · 60 / bpm``. Each detected beat gets its
   grid index (from the mean beat spacing: librosa's tempo estimate is
   quantized, 152 for a 150 BPM track, which mis-numbers beats over a song), a least-squares line is fitted, beats more than
   :data:`OUTLIER_MS` off are dropped, and the fit is repeated. The RMS
   residual of the remaining beats is the **drift**: a few ms for a clean,
   constant-tempo track; much more means tempo changes or bad detection.
3. Write ``bpm``, ``first_beat_s`` and ``beat_map`` (``method: librosa``,
   ``verified: false``) into ``song.json``. **Always verify by ear** with
   ``uv run parapara-beatcheck``: detection does not know where the musical
   phrase starts (``routine_start_beat``) and can be off by a constant few ms.

librosa is a *tools* dependency (``uv sync`` installs it by default for
development), imported lazily so the game process never needs it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from parapara.audio.song import FORMAT_VERSION, Song, SongError

ANALYSIS_SR = 22050
HOP = 256  # ~11.6 ms per onset frame at 22.05 kHz
START_BPM = 150.0
OUTLIER_MS = 50.0
ONSET_FMAX_HZ = 150.0  # kick-drum band
ONSET_N_MELS = 12  # ~14 FFT bins below 150 Hz: more bands would leave some empty
DETECTION_LAG_S = 0.019  # onset peaks trail the kick attacks (demo track: +19 ms)


@dataclass(frozen=True)
class GridFit:
    bpm: float
    first_beat_s: float  # time of the first detected (inlier) beat
    drift_ms: float  # RMS residual of the inliers
    beats_used: int
    beats_detected: int


def fold_tempo(bpm: float, low: float = 100.0, high: float = 200.0) -> float:
    """Undo tempo octave errors: halve/double until inside ``[low, high)``."""
    if bpm <= 0:
        raise ValueError("tempo must be positive")
    while bpm < low:
        bpm *= 2
    while bpm >= high:
        bpm /= 2
    return bpm


def fit_constant_tempo(
    beat_times_s: npt.ArrayLike, bpm_hint: float | None = None
) -> GridFit:
    """Fit ``t = first + i · period`` to detected beat times (see module docstring)."""
    t = np.sort(np.asarray(beat_times_s, dtype=np.float64))
    if len(t) < 4:
        raise ValueError("need at least 4 beats to fit a tempo")
    # Mean (not median) spacing: beat times are quantized to onset frames
    # (11.6 ms), so a 400 ms beat shows up as alternating 395 / 406 ms gaps
    # and the median is biased. Gaps far from typical (missed beats) are
    # left out of the mean.
    gaps = np.diff(t)
    typical = np.median(gaps)
    period = float(np.mean(gaps[np.abs(gaps / typical - 1) < 0.2]))
    if bpm_hint and abs(period / (60.0 / bpm_hint) - 1) > 0.1:
        period = 60.0 / bpm_hint  # many missed beats: trust the tempo estimate instead
    keep = np.ones(len(t), bool)
    for _ in range(3):
        idx = np.round((t - t[keep][0]) / period)
        A = np.stack([np.ones(keep.sum()), idx[keep]], axis=1)
        (first, period), *_ = np.linalg.lstsq(A, t[keep], rcond=None)
        residual_ms = (t - (first + idx * period)) * 1000
        new_keep = np.abs(residual_ms) <= OUTLIER_MS
        if new_keep.sum() < 4 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    idx = np.round((t - first) / period)
    first_used = float(first + idx[keep].min() * period)
    drift = float(np.sqrt(np.mean(residual_ms[keep] ** 2)))
    return GridFit(60.0 / period, first_used, drift, int(keep.sum()), len(t))


def detect_beats(
    audio: Path, start_bpm: float = START_BPM
) -> tuple[float, npt.NDArray[np.float64]]:
    """Run librosa's beat tracker; returns (tempo estimate, beat times in s)."""
    import librosa  # tools dependency: imported only when building beat maps

    y, sr = librosa.load(audio, sr=ANALYSIS_SR, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP,
                                             fmax=ONSET_FMAX_HZ, n_mels=ONSET_N_MELS)  # fmt: skip
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP,
                                           start_bpm=start_bpm, units="time")  # fmt: skip
    beats = np.asarray(beats, dtype=np.float64) - DETECTION_LAG_S
    return fold_tempo(float(np.atleast_1d(tempo)[0])), beats


def estimate_beat_map(audio: Path, start_bpm: float = START_BPM) -> GridFit:
    tempo, beats = detect_beats(audio, start_bpm)
    return fit_constant_tempo(beats, bpm_hint=tempo)


def update_song(
    directory: Path, audio_name: str | None = None, start_bpm: float = START_BPM
) -> GridFit:
    """Estimate the beat map and write it into ``directory/song.json``.

    An existing ``song.json`` keeps all other fields. Without one, a skeleton
    is written (title = folder name; licence and source left empty, which is
    fine for personal use).
    """
    path = directory / "song.json"
    data = json.loads(path.read_text()) if path.exists() else None
    if audio_name is None:
        if data is None:
            raise SongError(["no song.json yet: pass the audio file name"])
        audio_name = data["audio"]
    fit = estimate_beat_map(directory / audio_name, start_bpm)
    if data is None:
        data = {"format_version": FORMAT_VERSION, "id": directory.name, "title": directory.name,
                "artist": "", "audio": audio_name, "license": {"name": "", "url": None},
                "source": "", "attribution": "", "routine_start_beat": 8}  # fmt: skip
    data |= {
        "bpm": round(fit.bpm, 3),
        "first_beat_s": round(fit.first_beat_s, 4),
        "duration_beats": int(data.get("duration_beats") or _beats_in(directory / audio_name, fit)),
        "beat_map": {"method": "librosa", "verified": False, "drift_ms": round(fit.drift_ms, 2),
                     "notes": f"{fit.beats_used}/{fit.beats_detected} detected beats on the grid"},
    }  # fmt: skip
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return fit


def _beats_in(audio: Path, fit: GridFit) -> int:
    import soundfile  # installed with librosa

    duration = soundfile.info(str(audio)).duration
    return int((duration - fit.first_beat_s) / (60.0 / fit.bpm))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Estimate a song's beat map with librosa"
    )
    parser.add_argument("song_dir", type=Path)
    parser.add_argument(
        "--audio", help="audio file name (needed when there is no song.json yet)"
    )
    parser.add_argument("--start-bpm", type=float, default=START_BPM)
    args = parser.parse_args(argv)
    fit = update_song(args.song_dir, args.audio, args.start_bpm)
    print(f"bpm {fit.bpm:.3f}, first beat {fit.first_beat_s * 1000:.1f} ms, drift {fit.drift_ms:.1f} ms "
          f"({fit.beats_used}/{fit.beats_detected} beats)")  # fmt: skip
    print(
        f"written to {args.song_dir / 'song.json'}; now verify by ear: uv run parapara-beatcheck {args.song_dir}"
    )
    try:
        Song.load(args.song_dir)
    except SongError as err:
        print("song.json still needs:", "; ".join(err.issues))


if __name__ == "__main__":
    main()
