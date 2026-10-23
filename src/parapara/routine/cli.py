"""``uv run parapara-routines <command>``: generate, list, write and install routines.

Commands::

    generate [SONG ...] [--force]   generated easy/normal/hard routines for the given
                                    songs (default: every song that is missing them)
    list                            every song and its routines
    blocks                          the moves you can use, with counts and descriptions
    template --song ID [-o FILE]    start your own routine from a generated one
    check FILE [--song ID]          validate a routine (and that it fits the song)
    add FILE --song ID [--name N]   validate and install your own routine

Writing your own routine
------------------------
A routine is JSON: phrases of exactly 32 counts, each a list of move names
(see ``blocks``). Every phrase must contain all five kinds of move, end with a
flick + clap, and keep moves inside the 8-count bars. Easiest is to start
from ``template``, edit the move names, then ``check`` and ``add``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parapara.audio.importer import load_all_songs
from parapara.audio.song import Song
from parapara.generator.routine_generator import generate_routine
from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import Difficulty, RoutineError
from parapara.routine.serialization import routine_to_dict
from parapara.routine.store import (
    check_routine_file,
    ensure_generated,
    install_routine,
    max_phrases,
    routines_for,
)


def _song(songs: dict[str, Song], song_id: str) -> Song:
    if song_id not in songs:
        sys.exit(f"unknown song {song_id!r}; known: {', '.join(sorted(songs))}")
    return songs[song_id]


def cmd_generate(args: argparse.Namespace, songs: dict[str, Song]) -> None:
    for song_id in args.songs or sorted(songs):
        song = _song(songs, song_id)
        written = ensure_generated(song, force=args.force)
        status = (
            f"wrote {', '.join(p.stem for p in written)}" if written else "up to date"
        )
        print(f"{song_id:42s} {max_phrases(song):3d} phrases  {status}")


def cmd_list(args: argparse.Namespace, songs: dict[str, Song]) -> None:
    for song_id, song in sorted(songs.items()):
        routines = routines_for(song)
        print(f"{song_id} ({song.bpm:.0f} BPM, up to {max_phrases(song)} phrases)")
        for routine in routines.values():
            print(
                f"    {routine.id:50s} {routine.difficulty.value:7s} {len(routine.phrases)} phrases"
            )
        if not routines:
            print("    (no routines: run `parapara-routines generate`)")


def cmd_blocks(args: argparse.Namespace, songs: dict[str, Song]) -> None:
    for block_id, block in sorted(
        BLOCK_LIBRARY.items(), key=lambda kv: (kv[1].kind, kv[0])
    ):
        print(
            f"{block_id:32s} {block.kind.value:11s} {block.duration} counts  {block.description}"
        )


def cmd_template(args: argparse.Namespace, songs: dict[str, Song]) -> None:
    song = _song(songs, args.song)
    routine = generate_routine(song.id, max(1, min(args.phrases or max_phrases(song), max_phrases(song))),
                               Difficulty(args.difficulty))  # fmt: skip
    data = routine_to_dict(routine) | {"id": f"{song.id}-mine", "difficulty": "custom"}
    text = json.dumps(data, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
        print(
            f"wrote {args.output}; edit it, then: parapara-routines check {args.output} --song {song.id}"
        )
    else:
        print(text, end="")


def cmd_check(args: argparse.Namespace, songs: dict[str, Song]) -> None:
    song = _song(songs, args.song) if args.song else None
    routine, issues = check_routine_file(args.file, song)
    if issues:
        print(f"{args.file}: {len(issues)} problem(s)")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    assert routine is not None
    print(
        f"{args.file}: OK, {len(routine.phrases)} phrases ({routine.total_counts} counts)"
    )


def cmd_add(args: argparse.Namespace, songs: dict[str, Song]) -> None:
    song = _song(songs, args.song)
    try:
        target = install_routine(args.file, song, args.name)
    except RoutineError as err:
        print(f"not installed, {len(err.issues)} problem(s):")
        for issue in err.issues:
            print(f"  - {issue}")
        sys.exit(1)
    print(f"installed {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parapara-routines", description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate", help="generate easy/normal/hard routines")
    p.add_argument("songs", nargs="*")
    p.add_argument("--force", action="store_true", help="regenerate even if present")
    sub.add_parser("list", help="songs and their routines")
    sub.add_parser("blocks", help="moves available for routines")
    p = sub.add_parser("template", help="a generated routine to edit")
    p.add_argument("--song", required=True)
    p.add_argument("--difficulty", choices=[d.value for d in Difficulty if d is not Difficulty.CUSTOM],
                   default="normal")  # fmt: skip
    p.add_argument("--phrases", type=int)
    p.add_argument("-o", "--output", type=Path)
    p = sub.add_parser("check", help="validate a routine file")
    p.add_argument("file", type=Path)
    p.add_argument("--song")
    p = sub.add_parser("add", help="validate and install your own routine")
    p.add_argument("file", type=Path)
    p.add_argument("--song", required=True)
    p.add_argument("--name", help="routine name (default: file name)")
    return parser


COMMANDS = {"generate": cmd_generate, "list": cmd_list, "blocks": cmd_blocks,
            "template": cmd_template, "check": cmd_check, "add": cmd_add}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    COMMANDS[args.command](args, load_all_songs())


if __name__ == "__main__":
    main()
