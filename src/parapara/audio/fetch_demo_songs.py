"""``uv run parapara-fetch-demo-songs``: download the CC BY demo tracks.

Downloads every candidate in ``assets/songs/catalog.json`` from its
``audio_url`` and imports it into the built-in library as
``assets/songs/<catalog id>/`` through :func:`parapara.audio.importer.import_song`
(suitability check, automatic beat map, licence and attribution from the
catalog). These folders are **git-ignored**: third-party audio is never
committed, each developer fetches it (or copies it from the team's shared
drive with ``--from-dir``).

Already present songs are skipped unless ``--replace`` is given.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from parapara.audio.importer import CATALOG, ImportRejected, SongInfo, import_song
from parapara.audio.song import DEFAULT_LIBRARY


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "parapara-demo-fetch"})
    with urllib.request.urlopen(request, timeout=120) as resp, path.open("wb") as out:  # noqa: S310
        while chunk := resp.read(1 << 20):
            out.write(chunk)


def fetch_demo_songs(
    library: Path = DEFAULT_LIBRARY,
    catalog: Path = CATALOG,
    ids: list[str] | None = None,
    replace: bool = False,
    from_dir: Path | None = None,
    download: Callable[[str, Path], None] = _download,
) -> dict[str, str]:
    """Fetch and import catalog songs; returns ``{id: outcome}`` for each."""
    entries = json.loads(catalog.read_text())["candidates"]
    outcomes: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for entry in entries:
            song_id = entry["id"]
            if ids and song_id not in ids:
                continue
            if (library / song_id / "song.json").exists() and not replace:
                outcomes[song_id] = "already present"
                continue
            suffix = Path(entry["audio_url"]).suffix or ".mp3"
            local = from_dir / f"{song_id}{suffix}" if from_dir else Path(tmp) / f"{song_id}{suffix}"
            if not from_dir:
                download(entry["audio_url"], local)
            try:
                song, _ = import_song(local, SongInfo.from_catalog(entry), song_id, library, replace=True)
                outcomes[song_id] = f"imported ({song.bpm:.1f} BPM, beat map unverified)"
            except ImportRejected as err:
                outcomes[song_id] = "REJECTED: " + "; ".join(err.report.rejections)
    return outcomes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the CC BY demo songs from the catalog")
    parser.add_argument("ids", nargs="*", help="catalog ids (default: all)")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--from-dir", type=Path, help="use <id>.<ext> files from this folder instead of downloading")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    for song_id, outcome in fetch_demo_songs(args.library, ids=args.ids or None, replace=args.replace,
                                             from_dir=args.from_dir).items():  # fmt: skip
        print(f"{song_id:42s} {outcome}")
    print(f"\nsongs are in {args.library}/ (git-ignored); credits: see each song.json 'attribution'.")
    print("verify each beat map by ear: uv run parapara-beatcheck <song folder>")


if __name__ == "__main__":
    main()
