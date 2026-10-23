"""The texts of the privacy & licence notice and of the credits.

Shown by the game (:mod:`parapara.game.app`: before the camera starts for the
first time, and from the menu with ``P`` / ``C``), and by the recorder and the
latency test before they start the camera. Pure Python (no pygame), so the
wording is unit-tested.

Facts behind the privacy notice (checked 2026-10-23):

- MediaPipe processes the camera images on the device and does not send them
  to Google, but MediaPipe Tasks (Python included) sends Google metrics about
  the performance and use of its APIs, and apps are responsible for telling
  their users; there is no documented way to switch it off
  (https://developers.google.com/edge/mediapipe/solutions/tasks#mediapipe_tasks_privacy_notice,
  last modified 2026-06-05).
- The pose models (BlazePose GHUM 3D, used by the Pose Landmarker) are
  licensed under Apache 2.0; surveillance and identity recognition are out
  of scope (model card linked from
  https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).

Each entry is ``(style, text)``; styles: ``title``, ``heading``, ``text``,
``small``. :data:`PRIVACY_VERSION` is stored in ``ui.json`` once the player
has accepted the notice; changing the notice's substance means bumping it, so
everyone sees the new version once.
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse

from parapara.audio.song import UNSPECIFIED_LICENSE, Song
from parapara.settings import data_dir

PRIVACY_VERSION = "2026-10-23"
MEDIAPIPE_PRIVACY_URL = "goo.gle/mediapipe-privacy"

type Entry = tuple[str, str]


def privacy_entries() -> list[Entry]:
    """The privacy & licence notice."""
    return [
        ("title", "Privacy and licences"),
        ("heading", "Your camera picture"),
        ("text", "is processed on this computer by Google MediaPipe to find your body pose. "
                 "It is never saved and never sent anywhere."),
        ("heading", "Usage statistics"),
        ("text", "MediaPipe sends Google statistics about how its pose tracking performs and is "
                 "used (no images and no body data). It cannot be switched off."),
        ("small", f"Google's notice: {MEDIAPIPE_PRIVACY_URL}"),
        ("heading", "Saved on this computer"),
        ("text", "Each song you play is kept as body positions (no images) in recordings/ and a "
                 "score log in logs/; calibrations and settings in the data folder. "
                 "Delete them whenever you like."),
        ("small", f"data folder: {data_dir()}"),
        ("heading", "Licences"),
        ("text", "Pose models: BlazePose GHUM 3D by Google, Apache 2.0. Demo songs: CC BY "
                 "(see Credits, C in the menu)."),
    ]  # fmt: skip


def _site(url: str) -> str:
    """``https://freemusicarchive.org/music/...`` -> ``freemusicarchive.org``."""
    host = urlparse(url).netloc if "://" in url else url.split("/")[0]
    return host.removeprefix("www.")


def song_credit(song: Song) -> list[Entry]:
    """Title, artist, licence and source of one song (CC BY needs all four)."""
    who = f"{song.title} by {song.artist}" if song.artist else song.title
    if song.license.name == UNSPECIFIED_LICENSE:
        return [("text", who), ("small", "your own music")]
    licence = song.license.name.replace("-", " ")  # "CC-BY-4.0" -> "CC BY 4.0"
    source = f", from {_site(song.source)}" if song.source else ""
    return [("text", who), ("small", f"{licence}{source}")]


def credits_entries(songs: Iterable[Song]) -> list[Entry]:
    """The credits: music, pose tracking, software, references, logo."""
    entries: list[Entry] = [("title", "Credits"), ("heading", "Music")]
    for song in sorted(songs, key=lambda s: s.title.lower()):
        entries += song_credit(song)
    entries.append(("small", "Full attributions with links: CREDITS.md"))
    entries += [
        ("heading", "Pose tracking"),
        ("text", "Google MediaPipe Pose Landmarker with the BlazePose GHUM 3D models, Apache 2.0"),
        ("heading", "Software"),
        ("text", "pygame-ce (LGPL 2.1), NumPy (BSD), OpenCV (Apache 2.0, used by MediaPipe)"),
        ("heading", "Dance moves"),
        ("text", "Move descriptions follow EKS-D's \"How to Dance\" guide (kirakira.org)"),
        ("heading", "Logo"),
        ("text", "ParaParaPy logo designed by the ParaParaPy creator"),
    ]  # fmt: skip
    return entries
