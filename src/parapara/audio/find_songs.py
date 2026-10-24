"""Search Jamendo for Para Para candidates: ``JAMENDO_CLIENT_ID=... uv run parapara-find-songs``.

Uses the Jamendo API v3 tracks endpoint
(https://developer.jamendo.com/v3.0/tracks, checked 2026-10-18), which needs
a free developer ``client_id`` from https://devportal.jamendo.com:

- ``fuzzytags`` (OR) over dance-music tags, ``speed=high+veryhigh``,
  ``include=licenses+musicinfo``, singles and album tracks;
- keeps **plain CC BY** only (licence URL ``/licenses/by/``: no NC, ND or SA)
  and tracks whose artist allows downloads (``audiodownload_allowed``);
- downloads each into ``--download-dir`` (outside git) and runs the
  suitability check (:mod:`parapara.audio.suitability`), printing which pass.

The result is written as JSON in the same shape as the catalog's
``candidates`` (``assets/songs/catalog.json``), so good finds can be added
there by hand after listening.

(Free Music Archive has no public API any more; its search with the CC BY
filter was used manually for the catalog, see ``catalog.json`` ``searched``.)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from parapara.audio.suitability import check_track

API = "https://api.jamendo.com/v3.0/tracks/"
DEFAULT_TAGS = (
    "eurobeat",
    "eurodance",
    "trance",
    "hardstyle",
    "techno",
    "dance",
    "electronic",
)
CC_BY = "https://creativecommons.org/licenses/by/4.0/"


def is_plain_cc_by(license_url: str | None) -> bool:
    return (
        bool(license_url) and "/licenses/by/" in license_url
    )  # by-nc, by-sa ... don't match


def search_url(
    client_id: str, tags: tuple[str, ...], limit: int, offset: int = 0
) -> str:
    params = {"client_id": client_id, "format": "json", "limit": limit, "offset": offset,
              "fuzzytags": " ".join(tags), "speed": "high veryhigh", "include": "licenses musicinfo",
              "type": "single albumtrack", "audiodlformat": "mp32"}  # fmt: skip
    return API + "?" + urllib.parse.urlencode(params).replace("%20", "+")


def _fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (fixed https API)
        return json.load(resp)


def candidates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Plain CC BY, downloadable tracks, as catalog-shaped entries."""
    out = []
    for r in results:
        if not is_plain_cc_by(r.get("license_ccurl")) or not r.get(
            "audiodownload_allowed"
        ):
            continue
        out.append({
            "id": f"jamendo-{r['id']}", "title": r["name"], "artist": r["artist_name"],
            "platform": "Jamendo", "page_url": r.get("shareurl", ""), "audio_url": r.get("audiodownload", ""),
            "license": {"name": "CC-BY", "url": r["license_ccurl"]},
            "attribution": f"\"{r['name']}\" by {r['artist_name']} ({r.get('shareurl', '')}), "
                           f"licensed under CC BY ({r['license_ccurl']})",
        })  # fmt: skip
    return out


def find(client_id: str, download_dir: Path, tags: tuple[str, ...] = DEFAULT_TAGS, limit: int = 50,
         fetch: Callable[[str], dict] = _fetch_json,
         download: Callable[[str, Path], None] | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """Search, download and check; returns the entries with a ``check`` field."""
    if download is None:
        def _download(url: str, path: Path) -> None:
            urllib.request.urlretrieve(url, path)
        download_fn = _download
    else:
        download_fn = download
    found = candidates(fetch(search_url(client_id, tags, limit)).get("results", []))
    download_dir.mkdir(parents=True, exist_ok=True)
    for entry in found:
        path = download_dir / f"{entry['id']}.mp3"
        if not path.exists():
            download_fn(entry["audio_url"], path)
        report = check_track(path, entry["license"]["name"])
        entry["check"] = {"ok": report.ok, "rejections": list(report.rejections),
                          "warnings": list(report.warnings), "measured": report.metrics}  # fmt: skip
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Search Jamendo for CC BY Para Para candidates"
    )
    parser.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "parapara-candidates",
    )
    parser.add_argument("--out", type=Path, default=Path("jamendo-candidates.json"))
    args = parser.parse_args(argv)
    client_id = os.environ.get("JAMENDO_CLIENT_ID")
    if not client_id:
        parser.error(
            "set JAMENDO_CLIENT_ID (free developer id from https://devportal.jamendo.com)"
        )
    entries = find(
        client_id, args.download_dir, tuple(args.tags.split(",")), args.limit
    )
    for e in entries:
        m = e["check"]["measured"]
        status = "OK " if e["check"]["ok"] else "rej"
        print(f"{status} {e['artist'][:20]:20s} {e['title'][:34]:34s} {m.get('bpm', 0):7.2f} BPM  "
              + "; ".join(e["check"]["rejections"] or e["check"]["warnings"]))  # fmt: skip
    args.out.write_text(json.dumps(entries, indent=2) + "\n")
    print(
        f"\n{sum(e['check']['ok'] for e in entries)}/{len(entries)} suitable; written to {args.out}"
    )


if __name__ == "__main__":
    main()
