"""The game: ``uv run parapara-play`` (roadmap Step 6).

Flow (one capture process for the whole session)::

    MENU ──▶ BODY calibration ──▶ LATENCY calibration ──▶ PLAY ──▶ RESULTS ──▶ MENU
      │      (skipped when a        (skipped when a
      │       saved one is kept)     saved one is kept)
      └──▶ LEARN the moves ──▶ MENU

- **Menu**: songs (built-in demo tracks and your imports) with their routines;
  unverified beat maps are flagged. ``B`` / ``L`` redo the body / latency
  calibration even if saved ones exist; ``T`` opens "Learn the moves".
- **Learn the moves** (for new players): the demo dancer loops one move at a
  time, slowed down (50 / 75 / 100 % of the song's tempo) with metronome
  clicks, next to its name and description; the mirrored camera view lets the
  player try it. Nothing is scored.
- **Body calibration**: the position check (:mod:`parapara.pose.framing`)
  must pass, then a neutral pose and a T-pose are held for a few seconds each
  (:mod:`parapara.pose.calibration`); saved in the user data folder.
- **Latency calibration**: clap on each of 16 metronome clicks; the median
  delay, measured through the same feature pipeline as the scoring
  (:mod:`parapara.scoring.latency`), shifts the scoring schedule. It is saved
  and only needs redoing when the camera, speakers/headphones or room change.
- **Play**: the demo dancer on the left (mirrored), your mirrored camera view
  with skeleton on the right (side by side, as decided). Everything the
  player needs while dancing sits around the dancer, where they look (first
  test run, 2026-10-23): the grade pop-up above its head, the next move as a
  small looping stick figure at its top right (pictures, not text: a move name
  is hard to read from the dancing spot) and the count below its feet. Score,
  combo and progress are in the panel; a
  warning when you leave the picture. Driven by
  :class:`parapara.scoring.session.PlaySession` with the audio playback time as
  the clock. ``Esc`` ends the song early.
- **Results**: score, accuracy, grade counts, max combo, posture and flow. The
  summary goes to ``logs/game-*.json`` and the play is saved as a recording in
  ``recordings/game-*/play.npz`` for ``parapara-score`` / ``parapara-analyze``.

Clap timing uses vision for now; clap detection by sound (PLAN.md §5.1) can
replace it later.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402  (must follow the env var above)

from parapara import settings  # noqa: E402
from parapara.animation.animator import Animator, DancerPose  # noqa: E402
from parapara.audio.beatgrid import BeatGrid, render_click_track  # noqa: E402
from parapara.audio.importer import load_all_songs  # noqa: E402
from parapara.audio.song import Song  # noqa: E402
from parapara.diagnostics.recorder import write_log  # noqa: E402
from parapara.game.beatcheck import Player, load_for_player  # noqa: E402
from parapara.game.dancer_view import block_title, draw_dancer  # noqa: E402
from parapara.game.debug_view import ERROR_COLOUR, LiveFeed  # noqa: E402
from parapara.game.recorder import draw_framing_box  # noqa: E402
from parapara.ipc.client import DEFAULT_WORKER, CaptureClient  # noqa: E402
from parapara.ipc.messages import CaptureConfig  # noqa: E402
from parapara.pose.calibration import CalibrationError, calibrate  # noqa: E402
from parapara.pose.features import FeatureExtractor  # noqa: E402
from parapara.pose.framing import FramingMonitor  # noqa: E402
from parapara.pose.landmarks import PoseSample  # noqa: E402
from parapara.pose.recording import Recording  # noqa: E402
from parapara.routine.library import BLOCK_LIBRARY  # noqa: E402
from parapara.routine.model import Block, BlockKind, Routine  # noqa: E402
from parapara.routine.store import routines_for  # noqa: E402
from parapara.scoring.judge import FlowJudgment, Grade  # noqa: E402
from parapara.scoring.latency import LatencyResult, estimate_latency  # noqa: E402
from parapara.scoring.session import PlaySession  # noqa: E402

WIDTH, HEIGHT = 1240, 720
STAGE_W = 600  # demo dancer on the left; camera (640x480) + score panel on the right
DANCER_X, DANCER_Y, DANCER_SCALE = 210, 190, HEIGHT / 7.6  # mid-shoulder point; px per shoulder width
NEXT_BOX = pygame.Rect(392, 8, 204, 300)  # "next move" picture, top right of the dancer
LEARN_SPEEDS = (0.5, 0.75, 1.0)  # fractions of the song's tempo
LEARN_REPEATS = 64  # a learnt block is looped this many times before the animation restarts
BG, PANEL, TEXT, DIM, ACCENT = (18, 18, 28), (30, 30, 44), (235, 235, 245), (120, 120, 140), (255, 220, 80)
OK, WARN = (90, 220, 120), (255, 150, 60)
GRADE_COLOUR = {Grade.PERFECT: (255, 220, 80), Grade.GREAT: (90, 220, 120), Grade.GOOD: (110, 170, 255),
                Grade.HOLD: (90, 220, 120), Grade.MISS: (255, 90, 90)}  # fmt: skip
POPUP_S = 0.7
CALIBRATION_HOLD_S, CALIBRATION_COUNTDOWN_S = 3.0, 3.0
LATENCY_BPM, LATENCY_COUNT_IN, LATENCY_CLAPS = 100.0, 4, 16


def learn_order() -> list[Block]:
    """The library's moves for "Learn the moves", simplest kinds first
    (knock, chop, wrist roll, sweep, flick + clap); library order within a kind."""
    kinds = [BlockKind.KNOCK, BlockKind.CHOP, BlockKind.WRIST_ROLL, BlockKind.SWEEP, BlockKind.FLICK_CLAP]
    return sorted(BLOCK_LIBRARY.values(), key=lambda b: kinds.index(b.kind))


class Scene(Enum):
    MENU = auto()
    BODY = auto()
    LATENCY = auto()
    PLAY = auto()
    RESULTS = auto()
    LEARN = auto()


class UI:
    """Text size, readable from the dancing spot (1.5-2 m away).

    Every font size in this module is multiplied by ``UI.scale``; ``+`` / ``-``
    change it during the game and the choice is saved (``settings.save_ui``).
    The first test run showed desktop-sized text was unreadable from there.
    """

    scale = 2.0
    MIN, MAX, STEP = 1.0, 2.6, 0.2


def font(size: float) -> pygame.font.Font:
    return pygame.font.Font(None, max(8, round(size * UI.scale)))


def text(surface, s, pos, size=24, colour=TEXT, centre=False, right=False) -> int:
    """Draw ``s`` (scaled by ``UI.scale``, word-wrapped to the surface width);
    returns the y below it."""
    f = font(size)
    x, y = pos
    max_w = surface.get_width() - (16 if centre else x + 16)
    lines, line = [], ""
    for word in s.split(" "):
        trial = f"{line} {word}".strip()
        if line and f.size(trial)[0] > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    for ln in lines:
        r = f.render(ln, True, colour)
        lx = x - r.get_width() // 2 if centre else (x - r.get_width() if right else x)
        surface.blit(r, (lx, y))
        y += r.get_height() + 2
    return y + 4


@dataclasses.dataclass
class Menu:
    songs: list[Song]
    routines: dict[str, list[Routine]]
    song_index: int = 0
    routine_index: int = 0
    redo_body: bool = False
    redo_latency: bool = False

    @property
    def song(self) -> Song:
        return self.songs[self.song_index]

    @property
    def routine(self) -> Routine | None:
        options = self.routines[self.song.id]
        return options[self.routine_index % len(options)] if options else None


class Game:
    """The game's state machine and drawing (see module docstring)."""

    def __init__(self, args: argparse.Namespace, capture: CaptureClient, config: CaptureConfig) -> None:
        self.args = args
        self.capture = capture
        self.config = config
        # Fullscreen, scaled from the WIDTH x HEIGHT layout (letterboxed);
        # F toggles a window. The dummy video driver (tests) gets a plain window.
        headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
        flags = 0 if headless or args.window else pygame.SCALED | pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
        pygame.display.set_caption("parapara")
        UI.scale = float(args.text_scale or settings.load_ui().get("text_scale", UI.scale))
        self.feed = LiveFeed(config)
        self.framing = FramingMonitor()
        self.player = Player(44100)
        songs = load_all_songs()
        routines = {s.id: sorted(routines_for(s).values(), key=lambda r: (["easy", "normal", "hard", "custom"].index(r.difficulty.value), r.id))
                    for s in songs.values()}  # fmt: skip
        self.menu = Menu([songs[k] for k in sorted(songs)], routines)
        self.calibration = settings.load_calibration()
        saved = settings.load_latency()
        self.latency_ms: float | None = args.latency_ms if args.latency_ms is not None else (
            saved["latency_ms"] if saved else None)  # fmt: skip
        self.scene = Scene.MENU
        self.message = ""
        self.mirrored = True
        self.learn_blocks = learn_order()
        self.learn_index, self.learn_speed = 0, 0
        self._reset_scene_state()

    def _reset_scene_state(self) -> None:
        self.t0 = time.monotonic()
        self.step = 0  # sub-step inside a scene
        self.collected: list[PoseSample] = []
        self.session: PlaySession | None = None
        self.animator: Animator | None = None
        self.popups: list[tuple[float, Grade, int]] = []
        self.latency_result: LatencyResult | None = None
        self.click_ns: list[int] = []
        self.results: dict | None = None
        self.calibration_issues: list[str] = []

    def go(self, scene: Scene) -> None:
        self.player.pause()  # stop clicks / music from the scene being left
        self.scene = scene
        self._reset_scene_state()
        if scene is Scene.LATENCY:
            self.latency_extractor = FeatureExtractor(calibration=self.calibration)
        if scene is Scene.LEARN:
            self.start_learn()

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        clock = pygame.time.Clock()
        frames = 0
        running = True
        if self.args.auto:  # tests: straight to play, no calibration
            self.latency_ms = self.latency_ms or 0.0
            self.start_play()
        scripted = {}
        for item in (self.args.keys or "").split():
            frame, name = item.split(":")
            scripted.setdefault(int(frame), []).append(pygame.key.key_code(name))
        while running:
            keys = list(scripted.get(frames, []))  # tests: scripted key presses
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    keys.append(event.key)
            samples = self.feed.update(self.capture)
            for sample in samples:
                self.framing.update(sample)
            running = self.handle(keys, samples) and running
            self.draw()
            pygame.display.flip()
            self.feed.after_flip()
            clock.tick(60)
            frames += 1
            if self.args.shots and frames % 40 == 0:  # reviews: periodic screenshots
                pygame.image.save(self.screen, f"{self.args.shots}/{frames:05d}-{self.scene.name.lower()}.png")
            shot = self.args.screenshot
            if shot and self.scene is Scene.PLAY and self.player.position() > self.args.auto_seconds * 0.8:
                pygame.image.save(self.screen, shot)
                self.args.screenshot = None
            if self.args.auto and self.scene is Scene.RESULTS and frames > 5:
                if self.args.results_screenshot:
                    pygame.image.save(self.screen, self.args.results_screenshot)
                running = False
        self.player.pause()

    def handle(self, keys: list[int], samples: list[PoseSample]) -> bool:
        if pygame.K_m in keys:
            self.mirrored = not self.mirrored
        if pygame.K_f in keys:
            pygame.display.toggle_fullscreen()
        bigger = {pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS} & set(keys)
        smaller = {pygame.K_MINUS, pygame.K_KP_MINUS} & set(keys)
        if bigger or smaller:
            UI.scale = round(min(UI.MAX, max(UI.MIN, UI.scale + (UI.STEP if bigger else -UI.STEP))), 2)
            settings.save_ui(text_scale=UI.scale)
        if self.scene is Scene.MENU:
            return self.handle_menu(keys)
        if pygame.K_ESCAPE in keys and self.scene in (Scene.BODY, Scene.LATENCY, Scene.RESULTS, Scene.LEARN):
            self.go(Scene.MENU)
            return True
        {Scene.BODY: self.handle_body, Scene.LATENCY: self.handle_latency, Scene.PLAY: self.handle_play,
         Scene.RESULTS: self.handle_results, Scene.LEARN: self.handle_learn}[self.scene](keys, samples)  # fmt: skip
        return True

    # -- menu ----------------------------------------------------------------

    def handle_menu(self, keys: list[int]) -> bool:
        m = self.menu
        if pygame.K_ESCAPE in keys or pygame.K_q in keys:
            return False
        if not m.songs:
            return True
        if pygame.K_DOWN in keys:
            m.song_index, m.routine_index = (m.song_index + 1) % len(m.songs), 0
        if pygame.K_UP in keys:
            m.song_index, m.routine_index = (m.song_index - 1) % len(m.songs), 0
        if pygame.K_RIGHT in keys:
            m.routine_index += 1
        if pygame.K_LEFT in keys:
            m.routine_index -= 1
        if pygame.K_b in keys:
            m.redo_body = not m.redo_body
        if pygame.K_l in keys:
            m.redo_latency = not m.redo_latency
        if pygame.K_t in keys:
            self.go(Scene.LEARN)
            return True
        if (pygame.K_RETURN in keys or pygame.K_SPACE in keys) and m.routine is not None:
            self.next_after(Scene.MENU)
        return True

    def next_after(self, scene: Scene) -> None:
        """The scene that follows ``scene``, skipping calibrations that are kept."""
        m = self.menu
        if scene is Scene.MENU and (self.calibration is None or m.redo_body):
            return self.go(Scene.BODY)
        if scene in (Scene.MENU, Scene.BODY) and (self.latency_ms is None or m.redo_latency):
            return self.go(Scene.LATENCY)
        m.redo_body = m.redo_latency = False
        self.start_play()

    # -- body calibration -----------------------------------------------------

    def handle_body(self, keys: list[int], samples: list[PoseSample]) -> None:
        # step 0/2: wait for a stable good position; 1/3: countdown + hold neutral / T-pose; 4: result
        if self.step in (0, 2) and self.framing.stable:
            self.step += 1
            self.t0, self.collected = time.monotonic(), []
        elif self.step in (1, 3):
            elapsed = time.monotonic() - self.t0
            countdown, hold = (0.4, 0.5) if self.args.fast else (CALIBRATION_COUNTDOWN_S, CALIBRATION_HOLD_S)
            if elapsed > countdown:
                self.collected += samples
            if elapsed > countdown + hold:
                if self.step == 1:
                    self.neutral = Recording.from_samples(self.collected, {"kind": "static"})
                    self.step = 2
                else:
                    t_pose = Recording.from_samples(self.collected, {"kind": "static"})
                    try:
                        self.calibration = calibrate(self.neutral, t_pose)
                        settings.save_calibration(self.calibration)
                        self.calibration_issues = []
                    except CalibrationError as err:
                        self.calibration_issues = err.issues
                    self.step = 4
        elif self.step == 4:
            if self.calibration_issues and pygame.K_r in keys:
                self.go(Scene.BODY)
            elif not self.calibration_issues and (pygame.K_RETURN in keys or pygame.K_SPACE in keys):
                self.next_after(Scene.BODY)

    # -- latency calibration ------------------------------------------------------

    def handle_latency(self, keys: list[int], samples: list[PoseSample]) -> None:
        for sample in samples:
            features = self.latency_extractor.update(sample)
            if self.step == 1 and features is not None:
                self.collected.append(features)
        if self.step == 0 and (pygame.K_SPACE in keys or pygame.K_RETURN in keys):
            bpm, count_in, claps = (600.0, 2, 8) if self.args.fast else (LATENCY_BPM, LATENCY_COUNT_IN, LATENCY_CLAPS)
            grid = BeatGrid(bpm, 0.5, 0)
            beats = count_in + claps
            silence = np.zeros(int((0.5 + (beats + 1) * grid.beat_s) * self.player.sr), np.float32)
            self.player.play(render_click_track(silence, self.player.sr, grid, music_gain=0.0, click_gain=0.9), 0.0)
            start = self.player.start_ns
            self.click_ns = [start + int(grid.time_of(b) * 1e9) for b in range(count_in, beats)]
            self.latency_beat_ns = grid.beat_s * 1e9
            self.collected, self.step = [], 1
        elif self.step == 1 and time.monotonic_ns() > self.click_ns[-1] + 1e9:
            self.latency_result = estimate_latency(self.collected, self.click_ns, self.latency_beat_ns)
            self.step = 2
        elif self.step == 2:
            r = self.latency_result
            if pygame.K_r in keys:
                self.go(Scene.LATENCY)
            elif r is not None and r.ok and (pygame.K_RETURN in keys or pygame.K_SPACE in keys):
                self.latency_ms = r.latency_ms
                settings.save_latency(r.latency_ms, r.spread_ms, r.claps_found)
                self.next_after(Scene.LATENCY)
            elif pygame.K_s in keys:  # skip: keep the previous value (or none)
                self.latency_ms = self.latency_ms or 0.0
                self.next_after(Scene.LATENCY)

    # -- play ------------------------------------------------------------------

    def start_play(self) -> None:
        self.go(Scene.PLAY)
        song, routine = self.menu.song, self.menu.routine
        assert routine is not None
        self.song_samples = load_for_player(self.player, song.audio_path)
        self.player.play(self.song_samples, 0.0)
        self.session = PlaySession(song, routine, self.player.start_ns,
                                   int((self.latency_ms or 0.0) * 1e6), self.calibration)  # fmt: skip
        self.animator = Animator(routine)
        end_beat = song.routine_start_beat + routine.total_counts + 4
        self.play_end_s = min(song.beat_time_s(end_beat), len(self.song_samples) / self.player.sr)
        if self.args.auto:
            self.play_end_s = min(self.play_end_s, self.args.auto_seconds)

    def handle_play(self, keys: list[int], samples: list[PoseSample]) -> None:
        assert self.session is not None
        now = time.monotonic()
        for sample in samples:
            for event in self.session.add(sample):
                if not isinstance(event.judgment, FlowJudgment):
                    self.popups.append((now, event.judgment.grade, event.points))
        self.popups = [p for p in self.popups if now - p[0] < POPUP_S]
        if pygame.K_ESCAPE in keys or self.player.position() >= self.play_end_s:
            self.finish_play()

    def finish_play(self) -> None:
        assert self.session is not None
        self.session.finish(now_ns=time.monotonic_ns())
        self.player.pause()
        session = self.session
        self.results = session.summary()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rec_dir = Path("recordings") / f"game-{stamp}"
        session.recording().save(rec_dir / "play.npz")
        path = write_log("game", {"result": self.results, "recording": str(rec_dir / "play.npz")})
        self.message = f"saved {path}"
        self.scene = Scene.RESULTS

    def handle_results(self, keys: list[int], samples: list[PoseSample]) -> None:
        if pygame.K_RETURN in keys or pygame.K_SPACE in keys:
            self.go(Scene.MENU)

    # -- learn the moves -----------------------------------------------------------

    @property
    def learn_bpm(self) -> float:
        bpm = self.menu.song.bpm if self.menu.songs else 150.0
        return bpm * LEARN_SPEEDS[self.learn_speed]

    def start_learn(self) -> None:
        """Loop the selected move with metronome clicks at the learning tempo.

        The audio clock drives the animation (as in play), so the moves land
        on the clicks. Beat 0 is 0.5 s into the click track; the track lasts
        LEARN_REPEATS bars and restarts when it ends.
        """
        block = self.learn_blocks[self.learn_index]
        self.animator = Animator.for_blocks([block] * LEARN_REPEATS)
        self.learn_grid = BeatGrid(self.learn_bpm, 0.5, 0)
        seconds = 0.5 + self.animator.end_beat * self.learn_grid.beat_s
        silence = np.zeros(int(seconds * self.player.sr), np.float32)
        self.learn_track = render_click_track(silence, self.player.sr, self.learn_grid, music_gain=0.0, click_gain=0.5)
        self.player.play(self.learn_track, 0.0)

    def handle_learn(self, keys: list[int], samples: list[PoseSample]) -> None:
        step = (pygame.K_RIGHT in keys) - (pygame.K_LEFT in keys)
        faster = (pygame.K_UP in keys) - (pygame.K_DOWN in keys)
        if step or faster:
            self.learn_index = (self.learn_index + step) % len(self.learn_blocks)
            self.learn_speed = min(len(LEARN_SPEEDS) - 1, max(0, self.learn_speed + faster))
            self.start_learn()
        elif self.player.position() >= len(self.learn_track) / self.player.sr:
            self.start_learn()  # loop

    # -- drawing -----------------------------------------------------------------
    # Sizes below are *base* sizes, multiplied by UI.scale (default 2.0) in
    # text(). Screens show few, short lines: they are read from 1.5-2 m away.

    def draw(self) -> None:
        self.screen.fill(BG)
        camera = self.screen.subsurface((STAGE_W, 0, self.config.width, self.config.height))
        self.feed.draw(camera, self.mirrored)
        failure = self.feed.failure(self.capture)
        if failure:
            text(camera, "CAMERA PROBLEM", (20, 20), 30, ERROR_COLOUR)
            text(camera, failure, (20, 70), 18, ERROR_COLOUR)
        elif self.feed.ready_at is None:
            text(camera, "starting camera ...", (camera.get_width() // 2, 200), 28, DIM, centre=True)
        stage = self.screen.subsurface((0, 0, STAGE_W, HEIGHT))
        panel = self.screen.subsurface((STAGE_W, self.config.height, WIDTH - STAGE_W, HEIGHT - self.config.height))
        panel.fill(PANEL)
        getattr(self, f"draw_{self.scene.name.lower()}")(stage, camera, panel)

    def draw_menu(self, stage, camera, panel) -> None:
        y = text(stage, "parapara", (30, 18), 48, ACCENT)
        m = self.menu
        if not m.songs:
            text(stage, "No songs: run parapara-fetch-demo-songs", (30, y + 20), 22, WARN)
            return
        y += 10
        for i, song in enumerate(m.songs):
            selected = i == m.song_index
            colour = ACCENT if selected else (TEXT if song.beat_map.verified else DIM)
            y = text(stage, f"{'> ' if selected else '  '}{song.title}", (24, y), 24 if selected else 20, colour)
            if selected:
                r = m.routine
                label = f"< {r.difficulty.value} >  {len(r.phrases)} phrases" if r else "no routines"
                y = text(stage, f"     {label}   {song.bpm:.0f} BPM", (24, y), 20, TEXT)
                if not song.beat_map.verified:
                    y = text(stage, "     beat map not checked", (24, y), 18, WARN)
            y += 4
        text(stage, "New? T learn the moves", (24, HEIGHT - 110), 18, ACCENT)
        text(stage, "Up/Down song   Left/Right level   Enter play", (24, HEIGHT - 70), 16, DIM)
        text(stage, "+/- text size   F fullscreen   Esc quit", (24, HEIGHT - 40), 16, DIM)
        cal = "OK" if self.calibration else "needed"
        lat = f"{self.latency_ms:.0f} ms" if self.latency_ms is not None else "needed"
        y = text(panel, f"body: {cal}" + ("  -> redo" if m.redo_body else "  (B redo)" if self.calibration else ""),
                 (20, 16), 22)  # fmt: skip
        text(panel, f"latency: {lat}" + ("  -> redo" if m.redo_latency else "  (L redo)" if self.latency_ms is not None else ""),
             (20, y), 22)  # fmt: skip

    def draw_body(self, stage, camera, panel) -> None:
        titles = {0: "Stand in position", 1: "Neutral pose", 2: "Stand in position", 3: "T-pose", 4: "Calibration"}
        y = text(stage, titles[self.step], (24, 20), 34, ACCENT)
        latest = self.framing.latest
        if self.step in (0, 2):
            draw_framing_box(camera, latest, self.mirrored)
            if latest is not None and latest.ok:
                text(stage, "OK - hold still", (24, y + 20), 28, OK)
            else:
                for issue in (latest.issues if latest else ())[:1]:
                    text(stage, issue.message, (24, y + 20), 28, WARN)
        elif self.step in (1, 3):
            how = "Arms down, relaxed" if self.step == 1 else "Arms straight out to the sides"
            text(stage, how, (24, y + 10), 26)
            elapsed = time.monotonic() - self.t0
            countdown = 0.4 if self.args.fast else CALIBRATION_COUNTDOWN_S
            label = str(int(np.ceil(countdown - elapsed))) if elapsed < countdown else "HOLD"
            text(stage, label, (STAGE_W // 2, 300), 110, ACCENT, centre=True)
        elif self.calibration_issues:
            y = text(stage, "Try again:", (24, y + 10), 28, WARN)
            text(stage, self.calibration_issues[0], (24, y), 22, WARN)
            text(stage, "R retry   Esc menu", (24, HEIGHT - 50), 20, DIM)
        else:
            text(stage, "Done!", (24, y + 20), 40, OK)
            text(stage, "Enter continue", (24, HEIGHT - 50), 22, DIM)

    def draw_latency(self, stage, camera, panel) -> None:
        y = text(stage, "Clap to the beat", (24, 20), 34, ACCENT)
        if self.step == 0:
            y = text(stage, f"Listen to {LATENCY_COUNT_IN} clicks, then clap on the next {LATENCY_CLAPS}.",
                     (24, y + 10), 24)  # fmt: skip
            y = text(stage, "Hands in view.", (24, y), 24)
            text(stage, "Only needed once: it is saved. Redo it (L in the menu) after changing camera, "
                 "speakers or headphones.", (24, y + 20), 18, DIM)  # fmt: skip
            text(stage, "Space start", (24, HEIGHT - 50), 24, ACCENT)
        elif self.step == 1:
            now = time.monotonic_ns()
            done = sum(c <= now for c in self.click_ns)
            text(stage, "listen ..." if done == 0 else "CLAP!", (STAGE_W // 2, 230), 64, ACCENT, centre=True)
            if done:
                text(stage, f"{done} / {len(self.click_ns)}", (STAGE_W // 2, 340), 40, TEXT, centre=True)
        else:
            r = self.latency_result
            if r and r.ok:
                text(stage, f"{r.latency_ms:.0f} ms", (STAGE_W // 2, 220), 72, OK, centre=True)
                text(stage, f"{r.claps_found} claps heard", (STAGE_W // 2, 340), 22, DIM, centre=True)
                text(stage, "Enter accept   R retry", (24, HEIGHT - 50), 22, DIM)
            else:
                y = text(stage, "Try again:", (24, y + 10), 28, WARN)
                text(stage, r.reason if r else "no result", (24, y), 22, WARN)
                text(stage, "R retry   S skip", (24, HEIGHT - 50), 22, DIM)

    def draw_play(self, stage, camera, panel) -> None:
        session, song = self.session, self.menu.song
        assert session is not None and self.animator is not None
        pos = self.player.position()
        beat = song.beat_at(pos) - song.routine_start_beat
        pose = self.animator.pose_at(beat)
        draw_dancer(stage, pose, (DANCER_X, DANCER_Y), DANCER_SCALE, self.mirrored)
        self.draw_count(stage, pose.count, beat)
        if pose.next_block is not None and pose.next_block_beat is not None:
            # The next move, danced in a loop by a small figure in sync with
            # the count it will be danced on (blocks start on a multiple of
            # their length, so beat % duration is the position inside it).
            preview = self.animator.pose_at(pose.next_block_beat + beat % pose.next_block.duration)
            self.draw_next(stage, preview, int(np.ceil(pose.next_block_beat - beat)))
        # Grade pop-up just above the demo dancer's head: players watch the
        # dancer, not their camera view (first test run, 2026-10-23). Drawn
        # after the dancer so a raised arm never hides it.
        now = time.monotonic()
        for t, grade, points in self.popups[-1:]:
            label = grade.value.upper() + ("!" if grade in (Grade.PERFECT, Grade.GREAT) else "")
            rise = round(12 * (1 - (now - t) / POPUP_S))
            text(stage, label, (DANCER_X, 18 - rise), 56, GRADE_COLOUR[grade], centre=True)
        warning = session.framing_warning
        if warning is not None:
            text(camera, warning.issues[0].message, (camera.get_width() // 2, camera.get_height() - 120), 28,
                 WARN, centre=True)  # fmt: skip
        k = session.keeper
        text(panel, f"{k.score:,}", (20, 6), 56, ACCENT)
        text(panel, f"combo {k.combo}", (20, 6 + round(56 * UI.scale)), 30)
        progress = min(1.0, pos / self.play_end_s) if self.play_end_s else 0.0
        bar = pygame.Rect(20, panel.get_height() - 22, panel.get_width() - 40, 12)
        pygame.draw.rect(panel, DIM, bar, 1)
        pygame.draw.rect(panel, ACCENT, (bar.x, bar.y, round(bar.w * progress), bar.h))

    def draw_count(self, stage, count: int, beat: float) -> None:
        """The count (1-8) below the dancer's feet, flashing on the beat."""
        on_beat = count and beat - np.floor(beat) < 0.12
        label = str(count) if count else ("ready" if beat < 0 else "")
        text(stage, label, (DANCER_X, HEIGHT - round(64 * UI.scale) - 16), 56 if count else 34,
             ACCENT if on_beat else TEXT, centre=True)  # fmt: skip

    def draw_next(self, stage, pose: DancerPose, counts: int) -> None:
        """Small "next move" picture at the dancer's top right: a framed stick
        figure dancing the move, with how many counts until it starts.

        Only the upper body is shown (moves are arms only), so the figure can
        be drawn large: the drawing is clipped to a window from above raised
        hands down to just below hanging hands.
        """
        box = stage.subsurface(NEXT_BOX)
        box.fill(PANEL)
        pygame.draw.rect(box, ACCENT, box.get_rect(), 3)
        text(box, "next", (box.get_width() // 2, 4), 16, ACCENT, centre=True)
        scale = 52  # px per shoulder width
        window = box.subsurface((3, 40, box.get_width() - 6, round(3.6 * scale)))
        draw_dancer(window, pose, (window.get_width() // 2, round(1.7 * scale)), scale, self.mirrored)
        text(box, f"in {counts}", (box.get_width() // 2, box.get_height() - round(22 * UI.scale) - 4), 20,
             centre=True)  # fmt: skip

    def draw_learn(self, stage, camera, panel) -> None:
        block = self.learn_blocks[self.learn_index]
        assert self.animator is not None
        beat = self.learn_grid.beat_at(self.player.position()) % self.animator.end_beat
        pose = self.animator.pose_at(beat)
        draw_dancer(stage, pose, (DANCER_X, DANCER_Y + 30), DANCER_SCALE, self.mirrored)
        text(stage, block_title(block), (16, 8), 30, ACCENT)
        self.draw_count(stage, pose.count, beat)
        y = text(panel, block.description, (20, 12), 18)
        y = text(panel, f"move {self.learn_index + 1} / {len(self.learn_blocks)}   "
                        f"speed {LEARN_SPEEDS[self.learn_speed]:.0%}", (20, y), 16, DIM)  # fmt: skip
        text(panel, "Left/Right move   Up/Down speed   Esc menu", (20, panel.get_height() - round(18 * UI.scale) - 10),
             14, DIM)  # fmt: skip

    def draw_results(self, stage, camera, panel) -> None:
        r = self.results or {}
        y = text(stage, "Results", (24, 18), 40, ACCENT)
        y = text(stage, f"{r.get('score', 0):,}", (24, y), 56)
        y = text(stage, f"accuracy {r.get('accuracy_pct', 0):.0f}%", (24, y + 6), 28)
        y = text(stage, f"best combo {r.get('max_combo', 0)}", (24, y), 28)
        g = r.get("grades", {})
        y = text(stage, f"perfect {g.get('perfect', 0)}  great {g.get('great', 0)}  good {g.get('good', 0)}",
                 (24, y + 6), 20, DIM)  # fmt: skip
        y = text(stage, f"hold {g.get('hold', 0)}  miss {g.get('miss', 0)}", (24, y), 20, DIM)
        if r.get("ended_early"):
            text(stage, "ended early", (24, y), 18, DIM)
        text(stage, "Enter menu", (24, HEIGHT - 50), 22, DIM)
        text(panel, f"posture {r.get('posture_pct', 0):.0f}%   latency {r.get('latency_ms', 0):.0f} ms",
             (20, 16), 20)  # fmt: skip
        text(panel, self.message, (20, panel.get_height() - 40), 12, DIM)


def run(argv: list[str] | None = None, *, worker: str = DEFAULT_WORKER) -> None:
    parser = argparse.ArgumentParser(description="Para Para dance game")
    parser.add_argument("--latency-ms", type=float, help="override the latency calibration")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", choices=("lite", "full"), default="full")
    parser.add_argument("--window", action="store_true", help="run in a window instead of fullscreen")
    parser.add_argument("--text-scale", type=float, help="text size (default: saved, or 2.0; +/- in game)")
    parser.add_argument("--auto", action="store_true", help=argparse.SUPPRESS)  # tests: play straight away
    parser.add_argument("--auto-seconds", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--screenshot", help=argparse.SUPPRESS)  # save one play frame (reviews)
    parser.add_argument("--fast", action="store_true", help=argparse.SUPPRESS)  # tests: short calibrations
    parser.add_argument("--shots", help=argparse.SUPPRESS)  # reviews: screenshot folder
    parser.add_argument("--keys", help=argparse.SUPPRESS)  # tests: "frame:key ..." scripted presses
    parser.add_argument("--results-screenshot", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = CaptureConfig(camera_index=args.camera, pose_model=args.model)
    pygame.init()
    try:
        with CaptureClient(config, worker=worker) as capture:
            Game(args, capture, config).run()
    finally:
        pygame.quit()


def main(argv: list[str] | None = None) -> None:
    """Console entry point (returns ``None``)."""
    run(argv)


if __name__ == "__main__":
    main()
