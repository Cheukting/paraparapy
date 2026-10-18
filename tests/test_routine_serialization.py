"""Tests for routine JSON load/save, including the shipped sample routine."""

import json
from pathlib import Path

import pytest

from parapara.routine.model import BlockKind, Difficulty, RoutineError
from parapara.routine.serialization import (
    FORMAT_VERSION,
    load_routine,
    routine_from_dict,
    routine_to_dict,
    save_routine,
)

# A hand-written routine (the Step 1 sample), kept as a known-valid test fixture.
SAMPLE = Path(__file__).resolve().parent / "fixtures" / "sample-routine.json"


def _sample_data() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_sample_routine_is_valid() -> None:
    routine = load_routine(SAMPLE)
    assert routine.id == "demo-normal"
    assert routine.difficulty is Difficulty.NORMAL
    assert len(routine.phrases) == 2
    for phrase in routine.phrases:
        assert {b.kind for b in phrase.blocks} == set(BlockKind)


def test_round_trip(tmp_path: Path) -> None:
    routine = load_routine(SAMPLE)
    out = tmp_path / "copy.json"
    save_routine(routine, out)
    assert load_routine(out) == routine
    assert routine_to_dict(routine) == _sample_data()


def test_unsupported_version() -> None:
    data = _sample_data() | {"format_version": FORMAT_VERSION + 1}
    with pytest.raises(RoutineError, match="format_version"):
        routine_from_dict(data)


def test_missing_fields() -> None:
    with pytest.raises(RoutineError, match="song_id, difficulty"):
        routine_from_dict({"format_version": FORMAT_VERSION, "id": "x", "phrases": []})


def test_all_problems_reported_together() -> None:
    data = _sample_data()
    data["difficulty"] = "impossible"
    data["phrases"][0][0] = "moonwalk"  # unknown block
    data["phrases"][1] = data["phrases"][1][:-2]  # drop both flick + claps
    with pytest.raises(RoutineError) as err:
        routine_from_dict(data)
    text = " | ".join(err.value.issues)
    assert "unknown difficulty" in text
    assert "phrase 1: unknown block id(s) ['moonwalk']" in text
    assert "phrase 2: phrase must end with flick + clap" in text


def test_unknown_block_suggests_the_closest_name() -> None:
    data = _sample_data()
    data["phrases"][0][0] = "sweep_to_rihgt_4"
    with pytest.raises(RoutineError, match="did you mean 'sweep_to_right_4'"):
        routine_from_dict(data)


def test_custom_library_is_used() -> None:
    with pytest.raises(RoutineError, match="unknown block"):
        routine_from_dict(_sample_data(), library={})
