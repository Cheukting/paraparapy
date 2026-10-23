"""Where routines live, and how they are generated, checked and installed.

Routines are stored **next to their song**: ``<song folder>/routines/<name>.json``
(JSON format v1, :mod:`parapara.routine.serialization`). Songs fetched or
imported locally therefore keep their routines locally too (git-ignored with
the song). This is the only place routines live.

- :func:`routines_for` lists a song's routines;
- :func:`fit_issues` checks that a routine fits the song: it starts on the
  song's ``routine_start_beat`` and must end before the song does;
- :func:`max_phrases` is how many 32-count phrases fit, keeping
  :data:`END_MARGIN_COUNTS` counts free at the end for the music to finish;
- :func:`ensure_generated` writes the generated easy / normal / hard routines
  a song is missing (called after fetching or importing a song);
- :func:`install_routine` validates a player's own routine and installs it.

Pure Python (no cv2/mediapipe/pygame).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from parapara.audio.song import Song
from parapara.generator.routine_generator import generate_routine
from parapara.routine.model import COUNTS_PER_PHRASE, Difficulty, Routine, RoutineError
from parapara.routine.serialization import load_routine, routine_from_dict, save_routine

END_MARGIN_COUNTS = 8
GENERATED = (Difficulty.EASY, Difficulty.NORMAL, Difficulty.HARD)


def routine_dir(song: Song) -> Path:
    return song.directory / "routines"


def max_phrases(song: Song) -> int:
    available = song.duration_beats - song.routine_start_beat - END_MARGIN_COUNTS
    return max(0, available // COUNTS_PER_PHRASE)


def fit_issues(routine: Routine, song: Song) -> list[str]:
    issues = []
    if routine.song_id != song.id:
        issues.append(f"routine is for song {routine.song_id!r}, not {song.id!r}")
    end = song.routine_start_beat + routine.total_counts
    if end > song.duration_beats:
        issues.append(f"routine is {len(routine.phrases)} phrases ({routine.total_counts} counts) but only "
                      f"{max_phrases(song)} fit in the song (it starts on beat {song.routine_start_beat} "
                      f"of {song.duration_beats})")  # fmt: skip
    return issues


def routines_for(song: Song) -> dict[str, Routine]:
    """All routines for ``song`` (``<song folder>/routines/*.json``), keyed by id."""
    if not routine_dir(song).exists():
        return {}
    routines = (load_routine(p) for p in sorted(routine_dir(song).glob("*.json")))
    return {r.id: r for r in routines}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "routine"


def ensure_generated(song: Song, force: bool = False) -> list[Path]:
    """Write the generated routines the song is missing; returns written files."""
    n = max_phrases(song)
    if n < 1:
        return []
    written = []
    routine_dir(song).mkdir(parents=True, exist_ok=True)
    for difficulty in GENERATED:
        path = routine_dir(song) / f"{difficulty.value}.json"
        if path.exists() and not force:
            continue
        save_routine(generate_routine(song.id, n, difficulty), path)
        written.append(path)
    return written


def check_routine_file(
    path: Path, song: Song | None = None
) -> tuple[Routine | None, list[str]]:
    """Parse and validate a routine file (and its fit to ``song``)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        return None, [f"cannot read {path}: {err}"]
    if song is not None:
        data = {**data, "song_id": song.id}  # a template written for this song
    try:
        routine = routine_from_dict(data)
    except RoutineError as err:
        return None, err.issues
    return routine, fit_issues(routine, song) if song else []


def install_routine(path: Path, song: Song, name: str | None = None) -> Path:
    """Validate a player's routine and install it as a ``custom`` routine.

    Raises:
        RoutineError: with every problem found (nothing is installed).
    """
    routine, issues = check_routine_file(path, song)
    if issues or routine is None:
        raise RoutineError(issues)
    name = _slug(name or path.stem)
    if name in {d.value for d in GENERATED}:
        name = f"my-{name}"  # never overwrite a generated routine
    installed = Routine(id=f"{song.id}-{name}", song_id=song.id, difficulty=Difficulty.CUSTOM,
                        phrases=routine.phrases)  # fmt: skip
    target = routine_dir(song) / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    save_routine(installed, target)
    return target
