"""``uv run parapara-make-demo-song``: (re)generate the demo track.

Writes ``assets/songs/demo/demo.wav`` (not committed: ~3 MB, regenerated
deterministically) and ``song.json`` (committed). The beat map is exact by
construction, so it is marked ``generated`` and verified.

The demo routine ``assets/routines/demo-normal.json`` (two 32-count phrases)
is danced in the track's main section.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from parapara.audio.song import DEFAULT_LIBRARY, BeatMapInfo, License, Song
from parapara.audio.synth import render_demo_song, write_wav

DEMO_ID = "demo"


def make_demo_song(library: Path = DEFAULT_LIBRARY, bpm: float = 150.0, phrases: int = 2) -> Song:
    track = render_demo_song(bpm=bpm, phrases=phrases)
    directory = library / DEMO_ID
    write_wav(directory / "demo.wav", track.samples, track.sample_rate)
    song = Song(
        id=DEMO_ID,
        title="Demo Eurobeat (generated)",
        artist="parapara.audio.synth",
        audio="demo.wav",
        license=License("project-generated (no third-party rights)"),
        source=f"generated in code by parapara.audio.synth.render_demo_song(bpm={bpm}, phrases={phrases})",
        bpm=track.bpm,
        first_beat_s=track.first_beat_s,
        routine_start_beat=track.routine_start_beat,
        duration_beats=track.duration_beats,
        beat_map=BeatMapInfo("generated", verified=True, drift_ms=0.0,
                             notes="exact by construction; sections " + str(track.sections)),  # fmt: skip
        directory=directory,
    )
    song.save()
    return song


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the demo song")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    args = parser.parse_args(argv)
    song = make_demo_song(args.library)
    print(f"wrote {song.audio_path} and {song.directory / 'song.json'}")


if __name__ == "__main__":
    main()
