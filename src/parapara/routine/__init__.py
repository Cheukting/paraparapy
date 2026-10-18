"""Routine model: building blocks as timed keyframe poses, bars/phrases, validation.

Pure Python (no NumPy, cv2 or pygame), so it is safe to import in both the
capture and the game process.

- :mod:`.geometry`      -- body-frame vectors and helpers
- :mod:`.model`         -- Keyframe / Block / Phrase / Routine and the rules
- :mod:`.library`       -- draft definitions of the five building blocks
- :mod:`.serialization` -- JSON load/save (routines reference blocks by id)
"""

from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import (
    COUNTS_PER_BAR,
    COUNTS_PER_PHRASE,
    ArmPose,
    Block,
    BlockKind,
    Difficulty,
    HandShape,
    Keyframe,
    Phrase,
    Routine,
    RoutineError,
    TimedKeyframe,
)
from parapara.routine.serialization import load_routine, save_routine

__all__ = [
    "BLOCK_LIBRARY",
    "COUNTS_PER_BAR",
    "COUNTS_PER_PHRASE",
    "ArmPose",
    "Block",
    "BlockKind",
    "Difficulty",
    "HandShape",
    "Keyframe",
    "Phrase",
    "Routine",
    "RoutineError",
    "TimedKeyframe",
    "load_routine",
    "save_routine",
]
