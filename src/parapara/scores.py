"""The score board: best scores per song and level, kept on this computer.

Stored as ``scores.json`` in the user data folder (:func:`parapara.settings.data_dir`,
next to the calibrations). Single player for now (PLAN.md §10), so entries
carry no names; a leaderboard with names / online is future work.

- :func:`record`: add a finished play and return its :class:`Placement`.
  Plays ended early (Esc) are not ranked, so a short perfect start cannot
  top the board.
- :func:`top`: the best plays of one song + routine (level), best first
  (ties: the earlier play first).
- :func:`best`: the best play of one song + routine, or ``None``.

At most :data:`KEEP` plays are kept per song + routine. Pure Python (no pygame).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from parapara.settings import data_dir

KEEP = 20


@dataclass(frozen=True)
class Placement:
    rank: int  # 1 = best on this song and level
    entries: int  # plays on the board for this song and level
    previous_best: int | None  # the best score before this play

    @property
    def new_best(self) -> bool:
        return self.rank == 1


def _path():
    return data_dir() / "scores.json"


def _load() -> list[dict[str, Any]]:
    try:
        return (
            json.loads(_path().read_text()).get("scores", [])
            if _path().exists()
            else []
        )
    except (json.JSONDecodeError, AttributeError):
        return []  # a broken file starts a new board


def _save(entries: list[dict[str, Any]]) -> None:
    _path().parent.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps({"scores": entries}, indent=2) + "\n")


def top(song_id: str, routine_id: str, n: int = 10) -> list[dict[str, Any]]:
    mine = [
        e for e in _load() if e["song_id"] == song_id and e["routine_id"] == routine_id
    ]
    return sorted(mine, key=lambda e: (-e["score"], e["played"]))[:n]


def best(song_id: str, routine_id: str) -> dict[str, Any] | None:
    board = top(song_id, routine_id, 1)
    return board[0] if board else None


def record(summary: dict[str, Any], played: datetime | None = None) -> Placement | None:
    """Add a finished play (a :meth:`PlaySession.summary`); ``None`` if it was
    ended early (not ranked)."""
    if summary.get("ended_early"):
        return None
    song_id, routine_id = summary["song_id"], summary["routine_id"]
    before = best(song_id, routine_id)
    entry = {
        "song_id": song_id, "routine_id": routine_id, "difficulty": summary.get("difficulty", ""),
        "score": int(summary["score"]), "accuracy_pct": summary.get("accuracy_pct", 0.0),
        "max_combo": summary.get("max_combo", 0),
        "played": (played or datetime.now()).isoformat(timespec="seconds"),
    }  # fmt: skip
    entries = _load() + [entry]
    board = sorted((e for e in entries if e["song_id"] == song_id and e["routine_id"] == routine_id),
                   key=lambda e: (-e["score"], e["played"]))  # fmt: skip
    rank = board.index(entry) + 1
    dropped = board[KEEP:]  # keep the best KEEP per song and level
    _save([e for e in entries if e not in dropped])
    return Placement(rank, min(len(board), KEEP), before["score"] if before else None)


def placement_dict(p: Placement | None) -> dict[str, Any] | None:
    return asdict(p) | {"new_best": p.new_best} if p else None
