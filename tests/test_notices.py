"""The privacy & licence notice and the credits (pure text, parapara.notices)."""

from pathlib import Path

from parapara.audio.song import UNSPECIFIED_LICENSE, BeatMapInfo, License, Song
from parapara.notices import credits_entries, privacy_entries, song_credit


def _song(title: str, artist: str, licence: str, source: str = "") -> Song:
    return Song(id=title.lower().replace(" ", "-"), title=title, artist=artist, audio="a.ogg",
                license=License(licence), source=source, bpm=150.0, first_beat_s=0.5,
                routine_start_beat=8, duration_beats=200, beat_map=BeatMapInfo("librosa", True),
                directory=Path("."))  # fmt: skip


def test_privacy_notice_says_what_happens_to_camera_images_and_data() -> None:
    text = " ".join(s for _, s in privacy_entries()).lower()
    assert "never saved and never sent" in text  # camera images stay on the computer
    assert (
        "mediapipe sends google statistics" in text and "cannot be switched off" in text
    )
    assert "goo.gle/mediapipe-privacy" in text  # Google's own notice
    assert "recordings/" in text and "logs/" in text and "no images" in text
    assert "apache 2.0" in text and "cc by" in text
    styles = {style for style, _ in privacy_entries()}
    assert styles <= {"title", "heading", "text", "small"}


def test_credits_name_title_artist_licence_and_source_of_every_song() -> None:
    songs = [
        _song(
            "Lets Go To A Rave",
            "Defrini",
            "CC-BY-4.0",
            "https://freemusicarchive.org/music/defrini/x/y/",
        ),
        _song("My Own Track", "Me", UNSPECIFIED_LICENSE),
    ]
    entries = credits_entries(songs)
    text = [s for _, s in entries]
    assert (
        "Lets Go To A Rave by Defrini" in text
        and "CC BY 4.0, from freemusicarchive.org" in text
    )
    assert "My Own Track by Me" in text and "your own music" in text
    joined = " ".join(text)
    assert "MediaPipe" in joined and "Apache 2.0" in joined and "pygame-ce" in joined
    assert "CREDITS.md" in joined and "logo" in joined.lower()
    assert song_credit(songs[0])[0] == ("text", "Lets Go To A Rave by Defrini")
