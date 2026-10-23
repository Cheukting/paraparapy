"""Watch the demo dancer perform a routine to its song: ``uv run parapara-preview [song] [--routine R]``.

No camera: the song plays and the demo dancer (:mod:`parapara.animation`,
drawn by :mod:`parapara.game.dancer_view`) dances the routine on the beat,
mirrored so you can follow along. The side panel shows the count, bar and
phrase, the move being danced, and a **next move** card one bar ahead with a
small preview of its first pose (the lesson from the recordings: show what
is coming before it is needed). This panel becomes the demo half of the
Step 6 game screen.

Use it to learn a routine before dancing it, to check generated or
hand-written routines, and to check a song's beat map by eye and ear (does
the dancer hit the beat? with ``C`` the metronome clicks are mixed in).

Keys::

    Space          pause / resume        Left / Right  back / forward one bar
    R              next routine           C            metronome clicks on/off
    M              mirror on/off          Esc / Q      quit
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402  (must follow the env var above)

from parapara.animation.animator import Animator  # noqa: E402
from parapara.audio.beatgrid import BeatGrid, render_click_track  # noqa: E402
from parapara.audio.importer import load_all_songs  # noqa: E402
from parapara.audio.song import Song  # noqa: E402
from parapara.game.beatcheck import Player, load_audio  # noqa: E402
from parapara.game.dancer_view import block_title, draw_dancer, static_pose  # noqa: E402
from parapara.game.theme import ACCENT, BG, CARD, DIM, PANEL, TEXT  # noqa: E402
from parapara.routine.model import COUNTS_PER_BAR, Routine  # noqa: E402
from parapara.routine.store import routines_for  # noqa: E402

WIDTH, HEIGHT = 1000, 620
STAGE_WIDTH = 600


def _text(screen, text, pos, size=24, colour=TEXT) -> int:
    rendered = pygame.font.Font(None, size).render(text, True, colour)
    screen.blit(rendered, pos)
    return pos[1] + rendered.get_height() + 4


def draw_frame(screen: pygame.Surface, song: Song, routine: Routine, animator: Animator, song_time: float,
               mirrored: bool, clicks: bool, paused: bool) -> None:  # fmt: skip
    """One frame of the preview (also used by the Step 6 game screen)."""
    beat = song.beat_at(song_time) - song.routine_start_beat
    pose = animator.pose_at(beat)
    screen.fill(BG)
    stage = screen.subsurface((0, 0, STAGE_WIDTH, HEIGHT))
    # Raised hands reach ~1.4 shoulder widths above the shoulders, feet ~3.2 below.
    draw_dancer(stage, pose, (STAGE_WIDTH // 2, round(HEIGHT * 0.25)), HEIGHT / 6.4, mirrored)

    panel = screen.subsurface((STAGE_WIDTH, 0, WIDTH - STAGE_WIDTH, HEIGHT))
    panel.fill(PANEL)
    y = _text(panel, song.title, (20, 16), 30, ACCENT)
    y = _text(panel, f"{song.artist}  ·  {song.bpm:.0f} BPM", (20, y), 20, DIM)
    name = routine.id.removeprefix(song.id + "-")
    y = _text(panel, f"routine: {name} · {routine.difficulty.value} · {len(routine.phrases)} phrases",
              (20, y), 20, DIM)  # fmt: skip
    if pose.count:
        flash = beat - np.floor(beat) < 0.12
        _text(panel, str(pose.count), (40, y + 10), 120, ACCENT if flash else TEXT)
        _text(panel, f"bar {pose.bar}   phrase {pose.phrase}", (150, y + 40), 26, DIM)
    else:
        label = "intro: get ready" if beat < 0 else "finished!"
        _text(panel, label, (40, y + 40), 34, DIM)
    y += 130
    if pose.block is not None:
        y = _text(panel, "now: " + block_title(pose.block), (20, y), 28)
    if pose.next_block is not None and pose.next_block_beat is not None:
        counts = int(np.ceil(pose.next_block_beat - beat))
        card = panel.subsurface((20, y + 10, panel.get_width() - 40, 260))
        card.fill(CARD)
        _text(card, f"next: {block_title(pose.next_block)}", (12, 8), 26, ACCENT)
        _text(card, f"in {counts} count{'s' if counts != 1 else ''}", (12, 34), 22, DIM)
        draw_dancer(card, static_pose(pose.next_block.keyframes[0]), (card.get_width() // 2, 110),
                    36, mirrored)  # fmt: skip
    footer = (f"{'PAUSED  ' if paused else ''}clicks {'on' if clicks else 'off'}  mirror {'on' if mirrored else 'off'}"
              "   [Space] pause  [Left/Right] bar  [R] routine  [C] clicks  [M] mirror  [Esc] quit")  # fmt: skip
    _text(screen, footer, (16, HEIGHT - 26), 19, DIM)


def run(argv: list[str] | None = None) -> None:
    songs = load_all_songs()
    parser = argparse.ArgumentParser(description="Watch the demo dancer perform a routine")
    parser.add_argument("song", nargs="?", help=f"song id (default: first); known: {', '.join(sorted(songs))}")
    parser.add_argument("--routine", help="routine id or difficulty (easy/normal/hard/custom)")
    parser.add_argument("--clicks", action="store_true", help="start with metronome clicks")
    parser.add_argument("--auto", type=int, default=0, help=argparse.SUPPRESS)  # tests: run N frames
    args = parser.parse_args(argv)
    if not songs:
        parser.error("no songs: run `uv run parapara-fetch-demo-songs` or import one")
    song = songs.get(args.song or sorted(songs)[0])
    if song is None:
        parser.error(f"unknown song {args.song!r}")
    routines = sorted(routines_for(song).values(), key=lambda r: (r.difficulty.value, r.id))
    if not routines:
        parser.error(f"{song.id} has no routines: run `uv run parapara-routines generate {song.id}`")
    index = 0
    if args.routine:
        matches = [i for i, r in enumerate(routines) if args.routine in (r.id, r.difficulty.value)]
        if not matches:
            parser.error(f"no routine {args.routine!r}; have: {', '.join(r.id for r in routines)}")
        index = matches[0]

    samples, sr = load_audio(song.audio_path)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"ParaParaPy – preview: {song.title}")
    player = Player(sr)
    if player.sr != sr:
        samples = np.interp(np.arange(int(len(samples) / sr * player.sr)) / player.sr,
                            np.arange(len(samples)) / sr, samples).astype(np.float32)  # fmt: skip
        sr = player.sr
    grid = BeatGrid(song.bpm, song.first_beat_s, song.routine_start_beat)
    plain, clicked = samples, None
    clicks, mirrored = args.clicks, True

    def mix() -> np.ndarray:
        nonlocal clicked
        if clicks and clicked is None:
            clicked = render_click_track(samples, sr, grid)
        return clicked if clicks else plain

    animator = Animator(routines[index])
    player.play(mix(), 0.0)
    duration = len(samples) / sr
    clock = pygame.time.Clock()
    frame, running = 0, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif key == pygame.K_SPACE:
                    player.pause() if not player.paused else player.play(mix(), player.position())
                elif key in (pygame.K_LEFT, pygame.K_RIGHT):
                    step = COUNTS_PER_BAR * song.beat_s * (1 if key == pygame.K_RIGHT else -1)
                    target = min(max(0.0, player.position() + step), duration)
                    player.play(mix(), target) if not player.paused else (player.play(mix(), target), player.pause())
                elif key == pygame.K_r:
                    index = (index + 1) % len(routines)
                    animator = Animator(routines[index])
                elif key == pygame.K_c:
                    clicks = not clicks
                    if not player.paused:
                        player.play(mix(), player.position())
                elif key == pygame.K_m:
                    mirrored = not mirrored
        pos = player.position()
        if pos >= duration and not player.paused:
            player.play(mix(), 0.0)  # loop
        draw_frame(screen, song, routines[index], animator, pos, mirrored, clicks, player.paused)
        pygame.display.flip()
        clock.tick(60)
        frame += 1
        if args.auto and frame >= args.auto:
            running = False
    pygame.quit()


def main(argv: list[str] | None = None) -> None:
    """Console entry point (returns ``None``)."""
    run(argv)


if __name__ == "__main__":
    main()
