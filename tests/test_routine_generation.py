"""Tests for the routine generator, routine storage and `parapara-routines`."""

import json
from dataclasses import replace

import pytest

from parapara.audio.demo_song import demo_song_metadata
from parapara.audio.synth import render_demo_song
from parapara.generator.routine_generator import (
    SETTINGS,
    UNITS,
    GenerationError,
    arrange,
    generate_phrases,
    generate_routine,
)
from parapara.routine import cli
from parapara.routine.model import (
    COUNTS_PER_BAR,
    BlockKind,
    Difficulty,
    RoutineError,
    phrase_issues,
)
from parapara.routine.serialization import load_routine, routine_to_dict
from parapara.routine.store import (
    END_MARGIN_COUNTS,
    check_routine_file,
    ensure_generated,
    fit_issues,
    install_routine,
    max_phrases,
    routines_for,
)

DIFFICULTIES = (Difficulty.EASY, Difficulty.NORMAL, Difficulty.HARD)


@pytest.fixture
def song(tmp_path):
    """A song (the generated test track's metadata) in a temporary folder."""
    s = replace(
        demo_song_metadata(render_demo_song(phrases=4)),
        id="song-x",
        directory=tmp_path / "song-x",
    )
    s.directory.mkdir()
    s.save()
    return s


# --- generator -----------------------------------------------------------------


def test_units_are_well_formed() -> None:
    for unit in UNITS.values():
        assert unit.counts in (2, 4, 8), unit.name
        assert unit.finisher == (unit.blocks[-1].kind is BlockKind.FLICK_CLAP), (
            unit.name
        )
    for settings in SETTINGS.values():
        assert set(settings.weights) <= set(UNITS)
        assert any(UNITS[n].finisher for n in settings.weights)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_generated_routines_follow_the_phrase_rules(difficulty) -> None:
    routine = generate_routine("song-x", 8, difficulty)
    assert len(routine.phrases) == 8 and routine.difficulty is difficulty
    distinct = {tuple(b.id for b in p.blocks) for p in routine.phrases}
    assert len(distinct) == SETTINGS[difficulty].distinct_phrases
    for phrase in routine.phrases:
        assert phrase_issues(phrase.blocks) == []
        assert phrase.blocks[-1].kind is BlockKind.FLICK_CLAP


def test_choreography_conventions() -> None:
    """Knocks: right then left; chops: right before left within a unit; the
    phrase ends on a flick + clap (EKS-D guide conventions)."""
    for difficulty in DIFFICULTIES:
        for phrase in generate_routine("song-y", 4, difficulty).phrases:
            ids = [b.id for b in phrase.blocks]
            for i, block_id in enumerate(ids):
                if block_id == "knock_right":
                    assert ids[i + 1] == "knock_left"
                if block_id == "knock_left":
                    assert ids[i - 1] == "knock_right"


def test_repetition_patterns() -> None:
    easy = generate_routine("song-x", 6, Difficulty.EASY).phrases
    assert (
        easy[0] == easy[1] and easy[2] == easy[3] and easy[4] == easy[0]
    )  # A A B B A A
    hard = generate_routine("song-x", 6, Difficulty.HARD).phrases
    assert hard[0] != hard[1] and hard[4] == hard[0]  # A B C D A B
    assert arrange(["A", "B"], 5, "pairs") == ("A", "A", "B", "B", "A")


def test_difficulty_changes_the_movement() -> None:
    def short_share(d: Difficulty) -> float:
        blocks = [b for p in generate_routine("song-z", 8, d).phrases for b in p.blocks]
        return sum(b.duration == 2 for b in blocks) / len(blocks)

    assert (
        short_share(Difficulty.EASY)
        < short_share(Difficulty.NORMAL)
        < short_share(Difficulty.HARD)
    )


def test_generation_is_deterministic_per_song() -> None:
    a = generate_routine("song-x", 4, Difficulty.NORMAL)
    assert a == generate_routine("song-x", 4, Difficulty.NORMAL)
    assert a.phrases != generate_routine("another-song", 4, Difficulty.NORMAL).phrases


def test_generator_input_errors(monkeypatch) -> None:
    with pytest.raises(ValueError):
        generate_routine("s", 0)
    with pytest.raises(ValueError):
        generate_routine("s", 2, Difficulty.CUSTOM)
    monkeypatch.setattr("parapara.generator.routine_generator.MAX_ATTEMPTS", 1)
    with pytest.raises(GenerationError):
        generate_phrases("s", Difficulty.HARD, 4)


def test_eight_count_units_start_on_a_bar() -> None:
    for phrase in generate_routine("song-q", 8, Difficulty.NORMAL).phrases:
        start = 0
        for block in phrase.blocks:
            assert (
                start // COUNTS_PER_BAR
                == (start + block.duration - 1) // COUNTS_PER_BAR
            )
            start += block.duration


# --- store -------------------------------------------------------------------


def test_max_phrases_and_fit(song) -> None:
    # 8 intro + 4 * 32 + 8 outro beats; routine starts after the intro.
    assert (
        max_phrases(song)
        == (song.duration_beats - song.routine_start_beat - END_MARGIN_COUNTS) // 32
        == 4
    )
    assert fit_issues(generate_routine(song.id, 4), song) == []
    assert "only 4 fit" in fit_issues(generate_routine(song.id, 5), song)[0]
    assert "not 'song-x'" in fit_issues(generate_routine("other", 1), song)[0]


def test_ensure_generated_writes_missing_routines_only(song, tmp_path) -> None:
    written = ensure_generated(song)
    assert sorted(p.stem for p in written) == ["easy", "hard", "normal"]
    assert ensure_generated(song) == []  # already present
    assert len(ensure_generated(song, force=True)) == 3
    routines = routines_for(song)
    assert set(routines) == {"song-x-easy", "song-x-normal", "song-x-hard"}
    assert all(len(r.phrases) == 4 for r in routines.values())


def test_install_a_players_routine(song, tmp_path) -> None:
    mine = tmp_path / "my routine.json"
    mine.write_text(
        json.dumps(routine_to_dict(generate_routine("whatever", 2, Difficulty.EASY)))
    )
    target = install_routine(mine, song)  # song id is taken from the target song
    routine = load_routine(target)
    assert target.name == "my-routine.json" and routine.difficulty is Difficulty.CUSTOM
    assert routine.id == "song-x-my-routine" and routine.song_id == "song-x"
    assert (
        install_routine(mine, song, name="easy").name == "my-easy.json"
    )  # never overwrite generated


def test_install_rejects_invalid_routines(song, tmp_path) -> None:
    bad = tmp_path / "bad.json"
    data = routine_to_dict(generate_routine("song-x", 6))  # too long for the song
    data["phrases"][0][0] = "sweep_to_rihgt_4"
    bad.write_text(json.dumps(data))
    with pytest.raises(RoutineError) as err:
        install_routine(bad, song)
    assert "did you mean" in " ".join(err.value.issues)
    data["phrases"][0][0] = "sweep_to_right_4"
    bad.write_text(json.dumps(data))
    with pytest.raises(RoutineError, match="only 4 fit"):
        install_routine(bad, song)
    assert not (song.directory / "routines").exists()  # nothing installed
    (tmp_path / "broken.json").write_text("{not json")
    assert "cannot read" in check_routine_file(tmp_path / "broken.json", song)[1][0]


# --- CLI ----------------------------------------------------------------------


def test_cli_workflow(song, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_all_songs", lambda: {song.id: song})
    cli.main(["generate"])
    assert "wrote easy, normal, hard" in capsys.readouterr().out
    template = tmp_path / "mine.json"
    cli.main(
        [
            "template",
            "--song",
            "song-x",
            "--difficulty",
            "hard",
            "--phrases",
            "2",
            "-o",
            str(template),
        ]
    )
    assert json.loads(template.read_text())["difficulty"] == "custom"
    cli.main(["check", str(template), "--song", "song-x"])
    assert "OK, 2 phrases" in capsys.readouterr().out
    cli.main(["add", str(template), "--song", "song-x", "--name", "Mine"])
    cli.main(["list"])
    out = capsys.readouterr().out
    assert "song-x-mine" in out and "custom" in out
    cli.main(["blocks"])
    assert "chop_right" in capsys.readouterr().out
    data = json.loads(template.read_text())
    data["phrases"][0] = data["phrases"][0][:-1]
    template.write_text(json.dumps(data))
    with pytest.raises(SystemExit) as failed:
        cli.main(["check", str(template), "--song", "song-x"])
    assert failed.value.code == 1 and "must last 32 counts" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["add", str(template), "--song", "nope"])


def test_import_generates_routines(tmp_path) -> None:
    from parapara.audio import importer
    from parapara.audio.synth import write_wav

    track = render_demo_song(phrases=3)
    audio = tmp_path / "t.wav"
    write_wav(audio, track.samples, track.sample_rate)
    song, _ = importer.import_song(
        audio, importer.SongInfo("T"), library=tmp_path / "lib"
    )
    assert set(routines_for(song)) == {
        f"{song.id}-{d}" for d in ("easy", "normal", "hard")
    }
    assert not (tmp_path / "lib2").exists()
    song2, _ = importer.import_song(
        audio, importer.SongInfo("U"), library=tmp_path / "lib2", routines=False
    )
    assert not (song2.directory / "routines").exists()
