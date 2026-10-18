"""Import a player's own song: ``uv run parapara-import-song <audio file> ...``.

Lets users bring their own music, so the project never has to distribute
(or clear the rights to) anyone's tracks:

1. **Suitability check** (:mod:`parapara.audio.suitability`): tempo, steady
   beat, beat clarity and length. Unsuitable tracks are **rejected and not
   imported**; every reason is printed.
2. The audio is copied into the **user song library**, outside the git
   repository (:func:`default_user_library`), as ``<library>/<id>/``.
3. A ``song.json`` is written with the automatic beat map, marked
   **unverified**: check it by ear with ``uv run parapara-beatcheck
   <library>/<id>`` before playing.
4. Easy / normal / hard **routines are generated** for the song
   (:func:`parapara.routine.store.ensure_generated`); players can add their
   own with ``parapara-routines add``.

Licence and source are **optional** (decided 2026-10-18): imported songs are
assumed to be the user's own music, used locally for personal play. Only the
title is needed, and it defaults to the file name.
``--catalog <id>`` fills title, artist, licence, source and attribution from
``assets/songs/catalog.json`` (the vetted CC BY demo candidates).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from parapara.audio.song import (
    AUDIO_SUFFIXES,
    DEFAULT_LIBRARY,
    UNSPECIFIED_LICENSE,
    BeatMapInfo,
    License,
    Song,
    SongError,
    load_library,
)
from parapara.audio.suitability import SuitabilityReport, check_track

CATALOG = DEFAULT_LIBRARY / "catalog.json"
DEFAULT_ROUTINE_START_BEAT = 8  # after an 8-count intro; adjust with parapara-beatcheck


class ImportRejected(Exception):
    def __init__(self, report: SuitabilityReport) -> None:
        self.report = report
        super().__init__("; ".join(report.rejections))


def default_user_library() -> Path:
    """``$PARAPARA_SONGS_DIR``, else ``~/Library/Application Support/parapara/songs``
    on macOS, else ``$XDG_DATA_HOME/parapara/songs`` (``~/.local/share``)."""
    if env := os.environ.get("PARAPARA_SONGS_DIR"):
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "parapara" / "songs"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "parapara" / "songs"


def load_all_songs(include_user: bool = True, builtin: Path = DEFAULT_LIBRARY) -> dict[str, Song]:
    """Playable songs: the built-in library (``assets/songs``: the fetched CC
    BY demo tracks) plus the user library. Songs whose audio is missing are
    skipped; ids must be unique across libraries."""
    songs = load_library(builtin, skip_missing_audio=True) if builtin.exists() else {}
    user = default_user_library()
    if include_user and user.exists():
        for song_id, song in load_library(user, skip_missing_audio=True).items():
            if song_id in songs:
                raise SongError([f"song id {song_id!r} exists in both {builtin} and {user}"])
            songs[song_id] = song
    return songs


def slugify(*parts: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", " ".join(parts).lower()).strip("-")
    return slug[:60] or "song"


def catalog_entry(catalog_id: str, catalog: Path = CATALOG) -> dict:
    entries = {e["id"]: e for e in json.loads(catalog.read_text())["candidates"]}
    if catalog_id not in entries:
        raise KeyError(f"{catalog_id!r} is not in {catalog} (known: {', '.join(entries)})")
    return entries[catalog_id]


@dataclass(frozen=True)
class SongInfo:
    title: str
    artist: str = ""
    license_name: str = ""
    source: str = ""
    license_url: str | None = None
    attribution: str = ""

    @classmethod
    def from_catalog(cls, entry: dict) -> SongInfo:
        return cls(entry["title"], entry["artist"], entry["license"]["name"], entry["page_url"],
                   entry["license"].get("url"), entry.get("attribution", ""))  # fmt: skip


def import_song(audio: Path, info: SongInfo, song_id: str | None = None,
                library: Path | None = None, replace: bool = False,
                routines: bool = True) -> tuple[Song, SuitabilityReport]:  # fmt: skip
    """Check and import ``audio`` (see module docstring).

    Raises:
        ImportRejected: the track is unsuitable (nothing is written).
        SongError: bad input (unsupported format, missing file, id taken).
    """
    if audio.suffix.lower() not in AUDIO_SUFFIXES:
        raise SongError([f"unsupported audio format {audio.suffix!r}; use one of {AUDIO_SUFFIXES}"])
    if not audio.exists():
        raise SongError([f"no such file: {audio}"])
    report = check_track(audio, info.license_name)
    if not report.ok:
        raise ImportRejected(report)

    song_id = song_id or slugify(info.artist, info.title)
    directory = (library or default_user_library()) / song_id
    if directory.exists() and not replace:
        raise SongError([f"song {song_id!r} already exists in {directory.parent} (use --replace)"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{song_id}{audio.suffix.lower()}"
    shutil.copyfile(audio, target)

    a = report.analysis
    assert a is not None
    song = Song(
        id=song_id, title=info.title, artist=info.artist, audio=target.name,
        license=License(info.license_name or UNSPECIFIED_LICENSE, info.license_url), source=info.source,
        bpm=round(a.bpm, 3), first_beat_s=round(a.first_beat_s, 4),
        routine_start_beat=DEFAULT_ROUTINE_START_BEAT, duration_beats=a.duration_beats,
        beat_map=BeatMapInfo("librosa", verified=False, drift_ms=round(a.drift_ms, 2),
                             notes="imported; " + "; ".join(report.warnings)),
        attribution=info.attribution, directory=directory,
    )  # fmt: skip
    song.save()
    song = Song.load(directory)
    if routines:
        from parapara.routine.store import ensure_generated  # routine.store imports audio.song

        ensure_generated(song, force=replace)
    return song, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check a track and import it into your song library")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--catalog", help="fill in the details from assets/songs/catalog.json")
    parser.add_argument("--title", help="default: the file name")
    parser.add_argument("--artist", default="")
    parser.add_argument("--license", dest="license_name", default="", help="optional, e.g. CC-BY-4.0")
    parser.add_argument("--license-url")
    parser.add_argument("--source", default="", help="optional: where the track came from")
    parser.add_argument("--attribution", default="", help="credit text to show in the game")
    parser.add_argument("--id", dest="song_id")
    parser.add_argument("--library", type=Path, help=f"default: {default_user_library()}")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)

    if args.catalog:
        info = SongInfo.from_catalog(catalog_entry(args.catalog))
        song_id = args.song_id or args.catalog
    else:
        title = args.title or args.audio.stem.replace("_", " ")
        info = SongInfo(title, args.artist, args.license_name, args.source, args.license_url,
                        args.attribution)  # fmt: skip
        song_id = args.song_id
    try:
        song, report = import_song(args.audio, info, song_id, args.library, args.replace)
    except ImportRejected as err:
        print(f"REJECTED: {args.audio.name} is not suitable for Para Para:")
        for reason in err.report.rejections:
            print(f"  - {reason}")
        sys.exit(2)
    except SongError as err:
        print("ERROR: " + "; ".join(err.issues))
        sys.exit(1)
    print(f"imported {song.title!r} as {song.id!r}: {song.bpm:.2f} BPM, {song.duration_beats} beats; "
          "generated easy/normal/hard routines")  # fmt: skip
    for warning in report.warnings:
        print(f"  warning: {warning}")
    print(f"next: verify the beat map by ear: uv run parapara-beatcheck {song.directory}")


if __name__ == "__main__":
    main()
