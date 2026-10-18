"""Load and save routines as JSON.

File format (``format_version`` 1)::

    {
      "format_version": 1,
      "id": "demo-normal",
      "song_id": "demo",
      "difficulty": "normal",
      "phrases": [
        ["sweep_to_right_4", "sweep_to_left_4", ..., "flick_clap"],
        ...
      ]
    }

Routines reference blocks **by id** from a block library rather than
embedding keyframes. Refining a block's poses (e.g. after recording a real
dancer) therefore updates every routine that uses it, and routine files stay
short enough to write and review by hand.

Loading runs the full validation from :mod:`parapara.routine.model`; all
problems are reported together in one :class:`RoutineError`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import Block, Difficulty, Phrase, Routine, RoutineError

FORMAT_VERSION = 1


def routine_to_dict(routine: Routine) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "id": routine.id,
        "song_id": routine.song_id,
        "difficulty": routine.difficulty.value,
        "phrases": [[block.id for block in phrase.blocks] for phrase in routine.phrases],
    }


def routine_from_dict(
    data: Mapping[str, Any], library: Mapping[str, Block] = BLOCK_LIBRARY
) -> Routine:
    """Build and validate a :class:`Routine` from parsed JSON data.

    Raises:
        RoutineError: listing every problem found (unknown version, missing
            fields, unknown block ids, or phrase-rule violations).
    """
    version = data.get("format_version")
    if version != FORMAT_VERSION:
        raise RoutineError([f"unsupported format_version {version!r}, expected {FORMAT_VERSION}"])

    missing = [k for k in ("id", "song_id", "difficulty", "phrases") if k not in data]
    if missing:
        raise RoutineError([f"missing field(s): {', '.join(missing)}"])

    issues: list[str] = []
    try:
        difficulty = Difficulty(data["difficulty"])
    except ValueError:
        issues.append(
            f"unknown difficulty {data['difficulty']!r}, expected one of {[d.value for d in Difficulty]}"
        )

    phrases: list[Phrase] = []
    for p_index, block_ids in enumerate(data["phrases"], start=1):
        unknown = [b for b in block_ids if b not in library]
        if unknown:
            issues.append(f"phrase {p_index}: unknown block id(s) {unknown}")
            continue
        try:
            phrases.append(Phrase(tuple(library[b] for b in block_ids)))
        except RoutineError as err:
            issues.extend(f"phrase {p_index}: {issue}" for issue in err.issues)

    if issues:
        raise RoutineError(issues)
    return Routine(
        id=data["id"], song_id=data["song_id"], difficulty=difficulty, phrases=tuple(phrases)
    )


def save_routine(routine: Routine, path: Path) -> None:
    path.write_text(json.dumps(routine_to_dict(routine), indent=2) + "\n", encoding="utf-8")


def load_routine(path: Path, library: Mapping[str, Block] = BLOCK_LIBRARY) -> Routine:
    return routine_from_dict(json.loads(path.read_text(encoding="utf-8")), library)
