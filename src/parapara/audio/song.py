"""The song library: ``assets/songs/<song_id>/song.json`` + an audio file.

``song.json`` (``format_version`` 1)::

    {
      "format_version": 1,
      "id": "demo",
      "title": "...", "artist": "...",
      "audio": "demo.wav",
      "license": {"name": "CC-BY-4.0", "url": "https://..."},
      "source": "where the track came from",
      "attribution": "text to show in the credits ('' if none needed)",
      "bpm": 150.0,
      "first_beat_s": 0.5,
      "routine_start_beat": 8,
      "duration_beats": 80,
      "beat_map": {"method": "generated|librosa|manual", "verified": true,
                   "drift_ms": 0.0, "notes": ""}
    }

- **Licence and source are optional** (decided 2026-10-18): songs users
  import are assumed to be their own music for personal, local use, so the
  game does not police licences. They are recorded when known (the CC BY demo
  tracks from ``catalog.json`` carry licence and attribution for the credits);
  a missing licence loads as :data:`UNSPECIFIED_LICENSE`.
- **Beat map**: Eurobeat keeps a constant tempo, so a song's beat grid is two
  numbers: ``bpm`` and ``first_beat_s`` (time of beat 0 in the audio file).
  ``routine_start_beat`` is the beat on which the routine's first count falls
  (after the intro). ``beat_map.verified`` must be set by a person after
  listening (``parapara-beatcheck``): a 30 ms error ruins the scoring.
- Tempos outside 140-160 BPM are allowed but reported as warnings (the game's
  design range, PLAN.md §2).

Loading validates everything and reports all problems in one
:class:`SongError`. Pure Python (no audio libraries).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

from parapara.scoring.schedule import BeatClock

FORMAT_VERSION = 1
DEFAULT_LIBRARY = Path("assets/songs")
DESIGN_BPM = (140.0, 160.0)
UNSPECIFIED_LICENSE = "unspecified (personal use)"
AUDIO_SUFFIXES = (".wav", ".ogg", ".flac", ".mp3")


class SongError(ValueError):
    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


@dataclass(frozen=True)
class License:
    name: str
    url: str | None = None


@dataclass(frozen=True)
class BeatMapInfo:
    method: Literal["generated", "librosa", "manual"]
    verified: bool = False
    drift_ms: float | None = None  # RMS deviation of detected beats from the grid
    notes: str = ""


@dataclass(frozen=True)
class Song:
    id: str
    title: str
    artist: str
    audio: str  # file name inside the song directory
    license: License
    source: str
    bpm: float
    first_beat_s: float
    routine_start_beat: int
    duration_beats: int
    beat_map: BeatMapInfo
    attribution: str = ""
    directory: Path = field(default=Path("."), compare=False)
    warnings: tuple[str, ...] = field(default=(), compare=False)

    @property
    def audio_path(self) -> Path:
        return self.directory / self.audio

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm

    def beat_time_s(self, beat: float) -> float:
        """Time of ``beat`` in the audio file (seconds)."""
        return self.first_beat_s + beat * self.beat_s

    def beat_at(self, time_s: float) -> float:
        """Beat position at audio time ``time_s`` (fractional)."""
        return (time_s - self.first_beat_s) / self.beat_s

    def routine_clock(self, playback_start_ns: int, latency_ns: int = 0) -> BeatClock:
        """Beat clock for the routine when playback started at
        ``playback_start_ns`` (``time.monotonic_ns()``): routine beat 0 is the
        song's ``routine_start_beat``."""
        first = playback_start_ns + int(self.beat_time_s(self.routine_start_beat) * 1e9)
        return BeatClock(self.bpm, first, latency_ns)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("directory"), d.pop("warnings")
        return {"format_version": FORMAT_VERSION, **d}

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or self.directory
        path = directory / "song.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any], directory: Path) -> Self:
        issues: list[str] = []
        if data.get("format_version") != FORMAT_VERSION:
            raise SongError(
                [f"unsupported format_version {data.get('format_version')!r}"]
            )
        required = ("id", "title", "audio", "bpm", "first_beat_s", "routine_start_beat",
                    "duration_beats", "beat_map")  # fmt: skip
        missing = [k for k in required if k not in data]
        if missing:
            raise SongError([f"missing field(s): {', '.join(missing)}"])

        lic = data.get("license") or {}
        if not isinstance(lic, dict):
            issues.append("license must be an object with a name (and optional url)")
            lic = {}
        if Path(data["audio"]).suffix.lower() not in AUDIO_SUFFIXES:
            issues.append(
                f"audio must be one of {AUDIO_SUFFIXES}, got {data['audio']!r}"
            )
        bpm = float(data["bpm"])
        if not 60 <= bpm <= 240:
            issues.append(f"bpm {bpm} is not a plausible tempo")
        if float(data["first_beat_s"]) < 0:
            issues.append("first_beat_s must be >= 0")
        start, total = int(data["routine_start_beat"]), int(data["duration_beats"])
        if not 0 <= start < total:
            issues.append(
                f"routine_start_beat {start} must be within the song's {total} beats"
            )
        bm = data["beat_map"]
        if bm.get("method") not in ("generated", "librosa", "manual"):
            issues.append(
                f"beat_map.method {bm.get('method')!r} is not generated/librosa/manual"
            )
        if issues:
            raise SongError(issues)

        warnings = []
        if not DESIGN_BPM[0] <= bpm <= DESIGN_BPM[1]:
            warnings.append(f"bpm {bpm} is outside the design range {DESIGN_BPM}")
        if not bm.get("verified"):
            warnings.append(
                "beat map not verified by ear yet (see docs/verify-beat-maps.md)"
            )
        return cls(
            id=data["id"], title=data["title"], artist=data.get("artist", ""), audio=data["audio"],
            license=License(lic.get("name") or UNSPECIFIED_LICENSE, lic.get("url")),
            source=data.get("source", ""), bpm=bpm,
            first_beat_s=float(data["first_beat_s"]), routine_start_beat=start, duration_beats=total,
            beat_map=BeatMapInfo(bm["method"], bool(bm.get("verified")), bm.get("drift_ms"),
                                 bm.get("notes", "")),
            attribution=data.get("attribution", ""), directory=directory, warnings=tuple(warnings),
        )  # fmt: skip

    @classmethod
    def load(cls, directory: Path, require_audio: bool = True) -> Self:
        song = cls.from_dict(
            json.loads((directory / "song.json").read_text(encoding="utf-8")), directory
        )
        if require_audio and not song.audio_path.exists():
            hint = (
                " (run `uv run parapara-fetch-demo-songs`)"
                if song.license.name.startswith("CC")
                else ""
            )
            raise SongError([f"audio file missing: {song.audio_path}{hint}"])
        return song


def load_library(root: Path = DEFAULT_LIBRARY, require_audio: bool = True,
                 skip_missing_audio: bool = False) -> dict[str, Song]:  # fmt: skip
    """All songs under ``root`` keyed by id; raises on the first invalid song.

    ``skip_missing_audio`` leaves out songs whose audio file is absent (e.g.
    git-ignored demo tracks not fetched yet on a fresh clone) instead of raising.
    """
    songs: dict[str, Song] = {}
    for directory in sorted(p for p in root.iterdir() if (p / "song.json").exists()):
        if skip_missing_audio:
            song = Song.load(directory, require_audio=False)
            if not song.audio_path.exists():
                continue
        else:
            song = Song.load(directory, require_audio)
        if song.id in songs:
            raise SongError([f"duplicate song id {song.id!r}"])
        songs[song.id] = song
    return songs
