"""Tests for the demo dancer: animator motion (pure) and rendering / preview
app (pygame, in subprocesses, see conftest.py)."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from parapara.animation.animator import (
    CLAP,
    REST,
    SNAP_WINDOW_BEATS,
    Animator,
    block_title,
)
from parapara.generator.routine_generator import generate_routine
from parapara.routine.geometry import angle_between_deg
from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import BlockKind, Difficulty, Phrase, Routine
from parapara.routine.serialization import load_routine

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample-routine.json"


def _routines():
    yield load_routine(FIXTURE)
    for d in (Difficulty.EASY, Difficulty.NORMAL, Difficulty.HARD):
        yield generate_routine("anim-test", 2, d)


def _routine(*block_ids: str) -> Routine:
    """A one-phrase routine starting with the given blocks (rest filled validly)."""
    base = [b.id for b in load_routine(FIXTURE).phrases[0].blocks]
    return Routine(id="t", song_id="s", difficulty=Difficulty.NORMAL,
                   phrases=(Phrase(tuple(BLOCK_LIBRARY[b] for b in block_ids or base)),))  # fmt: skip


@pytest.mark.parametrize("routine", list(_routines()), ids=lambda r: r.id)
def test_dances_exactly_what_is_scored(routine) -> None:
    """On every keyframe's beat, each scored arm is exactly on its target."""
    animator = Animator(routine)
    for tk in routine.timeline():
        pose = animator.pose_at(tk.beat)
        for side in ("right", "left"):
            target = getattr(tk.keyframe, side)
            if target is None:
                continue
            arm = getattr(pose, side)
            assert angle_between_deg(arm.upper, target.upper) < 1e-3, (tk.block.id, tk.beat, side)
            assert angle_between_deg(arm.fore, target.fore) < 1e-3, (tk.block.id, tk.beat, side)
            if target.hand is not None:
                assert angle_between_deg(arm.hand, target.hand) < 1e-3


@pytest.mark.parametrize("routine", list(_routines()), ids=lambda r: r.id)
def test_motion_is_smooth(routine) -> None:
    """No jumps: at 160 BPM and 120 frames/s no arm segment turns more than 20°
    between frames (the fastest snaps, a near-180° arm move within 0.4 beat,
    reach about 15°/frame); the hand may reach 30° (a flick is a whip)."""
    animator = Animator(routine)
    step = 160 / 60 / 120  # beats per frame
    beats = np.arange(-4, routine.total_counts + 2, step)
    poses = [animator.pose_at(b) for b in beats]
    for side in ("right", "left"):
        for seg in ("upper", "fore", "hand"):
            dirs = [getattr(getattr(p, side), seg) for p in poses]
            worst = max(angle_between_deg(a, b) for a, b in zip(dirs, dirs[1:]))
            assert worst < (30 if seg == "hand" else 20), (side, seg, worst)
    xs = [p.body_x for p in poses]
    assert max(abs(a - b) for a, b in zip(xs, xs[1:])) < 0.05


def test_snap_holds_then_lands_on_the_beat() -> None:
    chop = next(tk for tk in load_routine(FIXTURE).timeline() if tk.block.kind is BlockKind.CHOP)
    animator = Animator(load_routine(FIXTURE))
    held = [animator.pose_at(chop.beat + d).right.fore for d in (0.02, 0.3, 1 - SNAP_WINDOW_BEATS - 0.01)]
    assert max(angle_between_deg(held[0], h) for h in held) < 1e-4  # holding still
    moving = animator.pose_at(chop.beat + 1 - SNAP_WINDOW_BEATS / 2).right.fore
    assert angle_between_deg(held[0], moving) > 5  # then snapping to the next target


def test_sweeps_glide_continuously() -> None:
    animator = Animator(_routine())  # the fixture starts with sweep_to_right_4
    forearms = [animator.pose_at(b).right.fore for b in np.linspace(0.1, 2.9, 15)]
    steps = [angle_between_deg(a, b) for a, b in zip(forearms, forearms[1:])]
    assert min(steps) > 1  # never stops inside the sweep


def test_wrist_roll_rolls_but_is_exact_on_beats() -> None:
    routine = load_routine(FIXTURE)
    roll = next(tk for tk in routine.timeline() if tk.block.kind is BlockKind.WRIST_ROLL)
    animator = Animator(routine)
    on_beat = animator.pose_at(roll.beat).right.fore
    mid = animator.pose_at(roll.beat + 0.5).right.fore
    assert angle_between_deg(on_beat, roll.keyframe.right.fore) < 1e-3
    assert angle_between_deg(on_beat, mid) > 10  # the hands circle between beats
    quarter = animator.pose_at(roll.beat + 0.25)
    up_r = quarter.right.fore[1] - roll.keyframe.right.fore[1]
    up_l = quarter.left.fore[1] - roll.keyframe.left.fore[1]
    assert up_r * up_l < 0  # opposite phase: one hand over, one under


def test_knocks_pulse_between_beats() -> None:
    routine = load_routine(FIXTURE)
    knock = next(tk for tk in routine.timeline() if tk.block.kind is BlockKind.KNOCK)
    animator = Animator(routine)
    hit = animator.pose_at(knock.beat).right.fore
    pulled = animator.pose_at(knock.beat + 0.5).right.fore
    assert pulled[2] < hit[2] - 0.2  # fist pulled back towards the body


def test_clap_and_rest_arms() -> None:
    routine = load_routine(FIXTURE)
    animator = Animator(routine)
    clap = next(tk for tk in routine.timeline() if tk.keyframe.clap)
    pose = animator.pose_at(clap.beat)
    assert pose.clap_flash == pytest.approx(1.0)
    assert angle_between_deg(pose.right.fore, CLAP.fore) < 1e-3
    assert angle_between_deg(pose.left.fore, CLAP.mirrored().fore) < 1e-3
    chop = next(tk for tk in routine.timeline() if tk.block.id == "chop_right")
    left = animator.pose_at(chop.beat).left  # the chop is right-handed: left arm rests
    assert angle_between_deg(left.fore, REST.mirrored().fore) < 30


def test_two_step_and_bounce() -> None:
    sway = [Animator.two_step(b)[0] for b in np.arange(0, 8, 0.05)]
    assert min(sway) == pytest.approx(-max(sway), abs=1e-6) and max(sway) > 0.1  # centred sway
    for beat in range(8):  # feet land (stop moving) on every beat
        before, after = Animator.two_step(beat - 1e-4), Animator.two_step(beat + 1e-4)
        assert before[1][0] == pytest.approx(after[1][0], abs=1e-3) and before[1][1] < 1e-3
    animator = Animator(load_routine(FIXTURE))
    assert animator.pose_at(3.0).bounce > animator.pose_at(3.5).bounce  # dip on the beat


def test_counts_blocks_and_preview() -> None:
    routine = load_routine(FIXTURE)
    animator = Animator(routine)
    intro = animator.pose_at(-6.0)
    assert (intro.count, intro.bar, intro.phrase, intro.block) == (0, 0, 0, None)
    assert intro.next_block is not None and intro.next_block_beat == 0.0  # previewed a bar ahead
    assert animator.pose_at(-9.0).next_block is None  # more than a bar away: not yet
    p = animator.pose_at(37.5)
    assert (p.count, p.bar, p.phrase) == (6, 1, 2)
    assert p.block is routine.phrases[1].blocks[2]
    end = animator.pose_at(routine.total_counts + 3)
    assert end.count == 0 and end.block is None and end.next_block is None
    assert angle_between_deg(end.right.fore, REST.fore) < 1e-3  # relaxed after the routine


@pytest.mark.parametrize("block_id", list(BLOCK_LIBRARY))
def test_for_blocks_loops_any_single_move_exactly(block_id) -> None:
    """Animator.for_blocks (the "Learn the moves" screen) dances a block
    repeated back to back, without phrase rules, exactly on every keyframe."""
    block = BLOCK_LIBRARY[block_id]
    animator = Animator.for_blocks([block] * 3)
    assert animator.routine is None and animator.end_beat == 3 * block.duration
    for repeat in range(3):
        start = repeat * block.duration
        assert animator.block_at(start + 0.5).block is block
        for kf in block.keyframes:
            pose = animator.pose_at(start + kf.beat)
            for side in ("right", "left"):
                target = getattr(kf, side)
                if target is not None:
                    assert angle_between_deg(getattr(pose, side).fore, target.fore) < 1e-3


def test_block_titles() -> None:
    assert block_title(BLOCK_LIBRARY["chop_right"]) == "Chop (right)"
    assert block_title(BLOCK_LIBRARY["sweep_to_left_4"]) == "Sweep to left (4)"
    assert block_title(BLOCK_LIBRARY["flick_clap_to_right"]) == "Flick clap to right"


# --- rendering & preview app (pygame in subprocesses) ---------------------------------

RENDER = textwrap.dedent(
    """
    import sys, pygame
    from parapara.animation.animator import Animator
    from parapara.game.dancer_view import draw_dancer, static_pose, RIGHT_COLOUR, LEFT_COLOUR
    from parapara.routine.serialization import load_routine
    from parapara.routine.library import BLOCK_LIBRARY
    pygame.init()
    from pathlib import Path
    animator = Animator(load_routine(Path(sys.argv[1])))
    for mirrored in (True, False):
        s = pygame.Surface((300, 400)); s.fill((0, 0, 0))
        draw_dancer(s, animator.pose_at(0.0), (150, 120), 50, mirrored)
        right = sum(tuple(s.get_at((x, y)))[:3] == RIGHT_COLOUR for x in range(150, 300) for y in range(400))
        left = sum(tuple(s.get_at((x, y)))[:3] == RIGHT_COLOUR for x in range(0, 150) for y in range(400))
        print(mirrored, right > left)
    draw_dancer(s, static_pose(BLOCK_LIBRARY["flick_clap_to_right"].keyframes[-1]), (150, 120), 50)
    print("ok")
    """
)


def _env(**extra):
    return os.environ | {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                         "PYGAME_HIDE_SUPPORT_PROMPT": "1", **extra}  # fmt: skip


def test_dancer_is_mirrored_like_the_guide_figure() -> None:
    """Mirrored (default): the dancer's right (orange) is on the screen's right."""
    out = subprocess.run([sys.executable, "-c", RENDER, str(FIXTURE)], capture_output=True, text=True,
                         env=_env(), check=True).stdout.split()  # fmt: skip
    assert out == ["True", "True", "False", "False", "ok"]


def test_preview_app_runs_headless(tmp_path) -> None:
    from parapara.audio import importer
    from parapara.audio.synth import render_demo_song, write_wav

    track = render_demo_song()
    write_wav(tmp_path / "t.wav", track.samples, track.sample_rate)
    song, _ = importer.import_song(tmp_path / "t.wav", importer.SongInfo("Test"), library=tmp_path / "lib")
    script = "from parapara.game import preview; preview.run([%r, '--routine', 'hard', '--clicks', '--auto', '40'])" % song.id
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=90,
                            cwd=tmp_path, env=_env(PARAPARA_SONGS_DIR=str(tmp_path / "lib")))  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "SDL_" not in result.stderr
    assert json.loads((song.directory / "routines" / "hard.json").read_text())["difficulty"] == "hard"
