"""Generate a Para Para routine for a song.

Choreography model
------------------
Real Para Para routines are built from a shared vocabulary in fixed patterns
(EKS-D "How to Dance", see ``routine/library.py``): knocks come as right then
left, chops as right-right-left-left, flick + claps finish a phrase and are
often paired in opposite directions, and a routine's phrases **repeat** so
the crowd can learn them. The generator therefore works with **units**, not
single blocks:

- a unit is a short, idiomatic sequence of blocks lasting 4 counts (half a
  bar) or 8 counts (a whole bar, placed on a bar start);
- a **phrase** (32 counts) is a sequence of units ending with a flick + clap
  unit; candidates are drawn at random (weighted by difficulty) and kept
  only if they pass every phrase rule of :func:`parapara.routine.model.phrase_issues`
  (all five block kinds, 32 counts, bar alignment) and avoid repeating the
  same unit three times in a row;
- a **routine** is a few distinct phrases arranged in a repeating pattern
  (e.g. ``A A B B A A B B ...``) over the song's danceable length.

Difficulty (:data:`SETTINGS`):

- **easy**: 4-count movements, mirrored sweep pairs, 2 distinct phrases,
  each played twice in a row;
- **normal**: adds vertical / diagonal sweeps, short chop pairs and paired
  flick + claps; 3 distinct phrases;
- **hard**: favours 2-count blocks (quick sweeps, short rolls, single chops)
  and variety; 4 distinct phrases, cycled.

Generation is **deterministic**: the random seed is derived from the song id
and difficulty, so regenerating gives the same routine.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import (
    COUNTS_PER_BAR,
    COUNTS_PER_PHRASE,
    Block,
    Difficulty,
    Phrase,
    Routine,
    phrase_issues,
)

MAX_ATTEMPTS = 2000


@dataclass(frozen=True)
class Unit:
    name: str
    block_ids: tuple[str, ...]
    finisher: bool = False  # flick + clap: only at the end of a phrase

    @property
    def blocks(self) -> tuple[Block, ...]:
        return tuple(BLOCK_LIBRARY[b] for b in self.block_ids)

    @property
    def counts(self) -> int:
        return sum(b.duration for b in self.blocks)


UNITS: dict[str, Unit] = {u.name: u for u in (
    # 8-count units (start on a bar)
    Unit("sweep_side_pair", ("sweep_to_right_4", "sweep_to_left_4")),
    Unit("sweep_vertical_pair", ("sweep_down_4", "sweep_up_4")),
    Unit("sweep_diagonal_pair", ("sweep_diagonal_down_to_right_4", "sweep_diagonal_up_to_left_4")),
    Unit("chop_rrll", ("chop_right", "chop_right", "chop_left", "chop_left")),
    Unit("knock_double", ("knock_right", "knock_left", "knock_right", "knock_left")),
    Unit("flick_clap_pair", ("flick_clap_to_right", "flick_clap_to_left"), finisher=True),
    # 4-count units (half a bar)
    Unit("sweep_right", ("sweep_to_right_4",)),
    Unit("sweep_left", ("sweep_to_left_4",)),
    Unit("sweep_down", ("sweep_down_4",)),
    Unit("sweep_up", ("sweep_up_4",)),
    Unit("sweep_quick", ("sweep_to_right_2", "sweep_to_left_2")),
    Unit("knock_pair", ("knock_right", "knock_left")),
    Unit("chop_rl", ("chop_right", "chop_left")),
    Unit("wrist_roll", ("wrist_roll_4",)),
    Unit("roll_and_sweep", ("wrist_roll_2", "sweep_to_right_2")),
    Unit("chop_and_roll", ("chop_right", "wrist_roll_2")),
    Unit("flick_clap", ("flick_clap_to_right",), finisher=True),
)}  # fmt: skip


@dataclass(frozen=True)
class DifficultySettings:
    weights: dict[str, float]  # unit name -> relative weight (absent = not used)
    distinct_phrases: int
    pattern: str  # how distinct phrases repeat: "pairs" (AABB...) or "cycle" (ABCD...)


SETTINGS: dict[Difficulty, DifficultySettings] = {
    Difficulty.EASY: DifficultySettings(
        {"sweep_side_pair": 3, "sweep_vertical_pair": 1, "chop_rrll": 3, "knock_double": 1,
         "knock_pair": 3, "wrist_roll": 3, "sweep_right": 1, "sweep_left": 1, "flick_clap": 1},
        distinct_phrases=2, pattern="pairs"),
    Difficulty.NORMAL: DifficultySettings(
        {"sweep_side_pair": 2, "sweep_vertical_pair": 2, "sweep_diagonal_pair": 1, "chop_rrll": 2,
         "knock_double": 1, "knock_pair": 3, "chop_rl": 2, "wrist_roll": 2, "sweep_down": 1,
         "sweep_up": 1, "sweep_quick": 1, "flick_clap": 2, "flick_clap_pair": 1},
        distinct_phrases=3, pattern="pairs"),
    Difficulty.HARD: DifficultySettings(
        {"sweep_diagonal_pair": 2, "sweep_vertical_pair": 1, "chop_rrll": 1, "knock_pair": 2,
         "chop_rl": 3, "sweep_quick": 3, "roll_and_sweep": 3, "chop_and_roll": 3, "wrist_roll": 1,
         "sweep_right": 1, "sweep_left": 1, "flick_clap": 1, "flick_clap_pair": 2},
        distinct_phrases=4, pattern="cycle"),
}  # fmt: skip


class GenerationError(RuntimeError):
    pass


def _seed(song_id: str, difficulty: Difficulty, salt: int = 0) -> int:
    return zlib.crc32(f"{song_id}:{difficulty.value}:{salt}".encode())


def _draw_phrase(rng: random.Random, settings: DifficultySettings) -> tuple[Unit, ...] | None:
    """One candidate phrase: units filling 32 counts, ending with a finisher."""
    finishers = [UNITS[n] for n in settings.weights if UNITS[n].finisher]
    movers = [UNITS[n] for n in settings.weights if not UNITS[n].finisher]
    finisher = rng.choices(finishers, [settings.weights[u.name] for u in finishers])[0]
    units: list[Unit] = []
    used = 0
    while used < COUNTS_PER_PHRASE - finisher.counts:
        remaining = COUNTS_PER_PHRASE - finisher.counts - used
        # 8-count units only on a bar start and when they fit.
        fits = [u for u in movers if u.counts <= remaining and (u.counts < COUNTS_PER_BAR or used % COUNTS_PER_BAR == 0)]
        if not fits:
            return None
        units.append(rng.choices(fits, [settings.weights[u.name] for u in fits])[0])
        used += units[-1].counts
    units.append(finisher)
    return tuple(units)


def _acceptable(units: tuple[Unit, ...]) -> bool:
    blocks = [b for u in units for b in u.blocks]
    if phrase_issues(blocks):
        return False
    names = [u.name for u in units]
    return not any(names[i] == names[i + 1] == names[i + 2] for i in range(len(names) - 2))


def generate_phrases(song_id: str, difficulty: Difficulty, count: int) -> list[Phrase]:
    """``count`` distinct valid phrases for the song (deterministic)."""
    settings = SETTINGS[difficulty]
    rng = random.Random(_seed(song_id, difficulty))
    phrases: list[Phrase] = []
    seen: set[tuple[str, ...]] = set()
    for _ in range(MAX_ATTEMPTS):
        units = _draw_phrase(rng, settings)
        if units is None or not _acceptable(units):
            continue
        key = tuple(b for u in units for b in u.block_ids)
        if key in seen:
            continue
        seen.add(key)
        phrases.append(Phrase(tuple(BLOCK_LIBRARY[b] for b in key)))
        if len(phrases) == count:
            return phrases
    raise GenerationError(f"could not generate {count} distinct {difficulty} phrases")


def arrange(distinct: list[Phrase], n_phrases: int, pattern: str) -> tuple[Phrase, ...]:
    """Lay out ``n_phrases`` phrases: "pairs" = A A B B C C A A ...; "cycle" = A B C D A ..."""
    if pattern == "pairs":
        order = [distinct[(i // 2) % len(distinct)] for i in range(n_phrases)]
    else:
        order = [distinct[i % len(distinct)] for i in range(n_phrases)]
    return tuple(order)


def generate_routine(song_id: str, n_phrases: int, difficulty: Difficulty = Difficulty.NORMAL,
                     routine_id: str | None = None) -> Routine:  # fmt: skip
    """A routine of ``n_phrases`` phrases for ``song_id`` (see module docstring)."""
    if difficulty is Difficulty.CUSTOM:
        raise ValueError("custom routines are written by players, not generated")
    if n_phrases < 1:
        raise ValueError("a routine needs at least one phrase")
    settings = SETTINGS[difficulty]
    distinct = generate_phrases(song_id, difficulty, min(settings.distinct_phrases, n_phrases))
    return Routine(
        id=routine_id or f"{song_id}-{difficulty.value}",
        song_id=song_id,
        difficulty=difficulty,
        phrases=arrange(distinct, n_phrases, settings.pattern),
    )
