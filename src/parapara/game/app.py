"""The ParaParaPy game: ``uv run parapara-play`` (roadmap Step 6).

Flow (one capture process for the whole session)::

    MENU ──▶ BODY calibration ──▶ LATENCY calibration ──▶ PLAY ──▶ RESULTS ──▶ MENU
      │      (first song after       (first song after
      │       launch, or B redo)      launch, or L redo)
      └──▶ LEARN the moves ──▶ MENU

- **Menu**: songs (built-in demo tracks and your imports) with their routines;
  unverified beat maps are flagged. ``B`` / ``L`` redo the body / latency
  calibration even if saved ones exist; ``T`` opens "Learn the moves".
- **Learn the moves** (for new players): the demo dancer loops one move at a
  time, slowed down (50 / 75 / 100 % of the song's tempo) with metronome
  clicks, next to its name and description; the mirrored camera view lets the
  player try it. Nothing is scored.
- **Calibration once per launch** (third test run): both calibrations run
  before the first song after the game starts and are kept for the next
  songs of the session. ``S`` on the latency screen reuses the last saved
  latency; ``--reuse-calibration`` skips both when saved ones exist.
- **Body calibration**: the position check (:mod:`parapara.pose.framing`)
  must pass, then a neutral pose and a T-pose (each shown by a stick figure)
  are held for a few seconds each
  (:mod:`parapara.pose.calibration`); saved in the user data folder.
- **Latency calibration**: clap on each of 8 metronome clicks; the median
  delay, measured through the same feature pipeline as the scoring
  (:mod:`parapara.scoring.latency`), shifts the scoring schedule.
- **Play**: the demo dancer on the left (mirrored), your mirrored camera view
  with skeleton on the right (side by side, as decided). Everything the
  player needs while dancing sits around the dancer, where they look (first
  test run, 2026-10-23): the grade pop-up above its head, the next move as a
  still stick figure at its top right (pictures, not text: a move name is
  hard to read from the dancing spot; still, because an animation was
  distracting) with a dot sliding into a ring when it starts, and the count below its
  feet. The music starts as soon as a song is chosen; a **4-count count-in**
  (4, 3, 2, 1 above the dancer) leads into the first move (``--count-in``).
  Score, combo and
  progress are in the panel; a
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
import sys
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402  (must follow the env var above)

from parapara import __version__, scores, settings  # noqa: E402
from parapara.animation.animator import Animator, DancerPose  # noqa: E402
from parapara.audio.beatgrid import BeatGrid, render_click_track  # noqa: E402
from parapara.audio.importer import load_all_songs  # noqa: E402
from parapara.audio.song import Song  # noqa: E402
from parapara.diagnostics.live_stats import LowLightMonitor  # noqa: E402
from parapara.diagnostics.recorder import write_log  # noqa: E402
from parapara.game.beatcheck import Player, load_for_player  # noqa: E402
from parapara.game.branding import NAME, set_window_icon, wordmark  # noqa: E402
from parapara.game.dancer_view import REST, block_title, draw_dancer, static_pose  # noqa: E402
from parapara.game.debug_view import LiveFeed  # noqa: E402
from parapara.game.notice_screens import draw_document, privacy_gate  # noqa: E402
from parapara.game.recorder import draw_framing_box  # noqa: E402
from parapara.game.theme import (  # noqa: E402
    ACCENT,
    BG,
    DIM,
    ERROR,
    GOOD,
    GREAT,
    HOLD,
    MISS,
    OK,
    PANEL,
    PERFECT,
    TEXT,
    WARN,
)
from parapara.game.ui import (  # noqa: E402  (Drawn, layout_issues re-exported for tests)
    UI,
    Drawn,  # noqa: F401
    adjust_scale,
    change_scale,
    display_flags,
    font,
    footer,
    layout_issues,
    load_scale,
    register_box,
    text,
    text_height,
    toggle_fullscreen,
)
from parapara.ipc.client import DEFAULT_WORKER, CaptureClient  # noqa: E402
from parapara.ipc.messages import CaptureConfig  # noqa: E402
from parapara.notices import credits_entries, privacy_entries, song_credit  # noqa: E402
from parapara.pose.calibration import CalibrationError, calibrate  # noqa: E402
from parapara.pose.features import FeatureExtractor, PoseFeatures  # noqa: E402
from parapara.pose.framing import FramingMonitor  # noqa: E402
from parapara.pose.landmarks import PoseSample  # noqa: E402
from parapara.pose.recording import Recording  # noqa: E402
from parapara.routine.library import BLOCK_LIBRARY  # noqa: E402
from parapara.routine.model import ArmPose, Block, BlockKind, Keyframe, Routine  # noqa: E402
from parapara.routine.store import routines_for  # noqa: E402
from parapara.scoring.judge import FlowJudgment, Grade  # noqa: E402
from parapara.scoring.latency import LatencyResult, estimate_latency  # noqa: E402
from parapara.scoring.session import PlaySession  # noqa: E402

WIDTH, HEIGHT = 1240, 720
STAGE_W = 600  # demo dancer on the left; camera (640x480) + score panel on the right
DANCER_X, DANCER_Y, DANCER_SCALE = (
    210,
    190,
    HEIGHT / 7.6,
)  # mid-shoulder point; px per shoulder width
NEXT_BOX = pygame.Rect(392, 8, 204, 300)  # "next move" picture, top right of the dancer
LEARN_SPEEDS = (0.5, 0.75, 1.0)  # fractions of the song's tempo
COUNT_IN_DEFAULT, COUNT_IN_CHOICES = (
    4,
    (0, 2, 4, 8),
)  # counts before the first move (settings)
LEARN_REPEATS = (
    64  # a learnt block is looped this many times before the animation restarts
)
GRADE_COLOUR = {Grade.PERFECT: PERFECT, Grade.GREAT: GREAT, Grade.GOOD: GOOD,
                Grade.HOLD: HOLD, Grade.MISS: MISS}  # fmt: skip
POPUP_S = 0.7
CALIBRATION_HOLD_S, CALIBRATION_COUNTDOWN_S = 3.0, 3.0
# 8 claps (fourth test run: 16 felt too long); estimate_latency needs MIN_CLAPS (6) of them.
LATENCY_BPM, LATENCY_COUNT_IN, LATENCY_CLAPS = 100.0, 4, 8
NEXT_LEAD_BEATS = 4  # the "next" box's timing dot travels during the last 4 counts
NEXT_HIT_BEATS = (
    0.35  # after a move starts, its picture stays with the ring lit for this long
)


# Stick-figure demonstrations of the body calibration poses.
CALIBRATION_POSES = {
    "neutral": Keyframe(beat=0.0, right=REST, left=REST.mirrored()),
    "t_pose": Keyframe(
        beat=0.0,
        right=ArmPose(upper=(1, 0, 0), fore=(1, 0, 0)),
        left=ArmPose(upper=(-1, 0, 0), fore=(-1, 0, 0)),
    ),  # fmt: skip
}


def learn_order() -> list[Block]:
    """The library's moves for "Learn the moves", simplest kinds first
    (knock, chop, wrist roll, sweep, flick + clap); library order within a kind."""
    kinds = [
        BlockKind.KNOCK,
        BlockKind.CHOP,
        BlockKind.WRIST_ROLL,
        BlockKind.SWEEP,
        BlockKind.FLICK_CLAP,
    ]
    return sorted(BLOCK_LIBRARY.values(), key=lambda b: kinds.index(b.kind))


class Scene(Enum):
    MENU = auto()
    BODY = auto()
    LATENCY = auto()
    PLAY = auto()
    RESULTS = auto()
    LEARN = auto()
    CREDITS = auto()
    PRIVACY = auto()
    SCORES = auto()
    SETTINGS = auto()


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

    def __init__(
        self, args: argparse.Namespace, capture: CaptureClient, config: CaptureConfig
    ) -> None:
        self.args = args
        self.capture = capture
        self.config = config
        # Fullscreen, scaled from the WIDTH x HEIGHT layout (letterboxed);
        # F toggles a window. The dummy video driver (tests) gets a plain window.
        set_window_icon()
        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT), display_flags(args.window)
        )
        pygame.display.set_caption(NAME)
        load_scale(args.text_scale)
        self.feed = LiveFeed(config)
        self.framing = FramingMonitor()
        self.player = Player(44100)
        songs = load_all_songs()
        routines = {s.id: sorted(routines_for(s).values(), key=lambda r: (["easy", "normal", "hard", "custom"].index(r.difficulty.value), r.id))
                    for s in songs.values()}  # fmt: skip
        self.menu = Menu([songs[k] for k in sorted(songs)], routines)
        # Calibrate at every launch (the player, their spot and the light
        # change between sessions), then keep both for the rest of the session
        # (third test run, 2026-10-23). The saved values are only a fallback:
        # S on the latency screen reuses the last one, and --reuse-calibration
        # (tests, development) skips calibrating when saved ones exist.
        saved = settings.load_latency()
        self.saved_latency_ms: float | None = saved["latency_ms"] if saved else None
        reuse = args.reuse_calibration
        self.calibration = settings.load_calibration() if reuse else None
        self.latency_ms: float | None = args.latency_ms if args.latency_ms is not None else (
            self.saved_latency_ms if reuse else None)  # fmt: skip
        self.scene = Scene.MENU
        self.message = ""
        # Settings kept between launches (ui.json; Settings screen, O in the menu).
        prefs = settings.load_ui()
        self.mirrored = bool(prefs.get("mirror", True))
        self.count_in = (
            args.count_in
            if args.count_in is not None
            else int(prefs.get("count_in", COUNT_IN_DEFAULT))
        )
        self.setting_index = 0
        self.learn_blocks = learn_order()
        self.learn_index, self.learn_speed = 0, 0
        self.lead_in_s = 0.0
        self.low_light = LowLightMonitor()  # "more light helps" hint (camera fps)
        self.camera_hint_top = (
            0  # top of the hint on the camera view (warnings go above it)
        )
        self.placement: scores.Placement | None = (
            None  # the last play's place on the score board
        )
        self.layout_reported: set[str] = set()
        self._reset_scene_state()

    def _reset_scene_state(self) -> None:
        self.t0 = time.monotonic()
        self.step = 0  # sub-step inside a scene
        self.collected = []
        self.latency_collected = []
        self.session: PlaySession | None = None
        self.animator: Animator | None = None
        self.popups: list[tuple[float, Grade, int]] = []
        self.latency_result: LatencyResult | None = None
        self.click_ns: list[int] = []
        self.count_in_ns: list[int] = []
        self.results: dict | None = None
        self.calibration_issues: list[str] = []
        self.doc_scroll, self.doc_more = 0, False  # Credits / Privacy screens
        self.play_fps: list[float] = []
        self.play_low = 0

    def go(self, scene: Scene) -> None:
        self.player.pause()  # stop clicks / music from the scene being left
        self.scene = scene
        self._reset_scene_state()
        if scene is Scene.LATENCY:
            self.latency_extractor = FeatureExtractor(calibration=self.calibration)
            self.latency_collected = []
        if scene is Scene.LEARN:
            self.start_learn()

    # -- main loop ---------------------------------------------------------

    def run(self, start_frame: int = 0) -> None:
        """The main loop; ``start_frame`` continues the frame count of the
        privacy gate (scripted test keys use one counter for both)."""
        clock = pygame.time.Clock()
        frames = start_frame
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
                self.low_light.add(sample.captured_ns, sample.frame_index)
            if self.scene is Scene.PLAY:  # camera rate during the song, for the log
                self.play_fps.append(self.low_light.fps)
                self.play_low += self.low_light.active
            running = self.handle(keys, samples) and running
            self.draw()
            pygame.display.flip()
            self.feed.after_flip()
            clock.tick(60)
            frames += 1
            if self.args.shots and frames % 40 == 0:  # reviews: periodic screenshots
                pygame.image.save(
                    self.screen,
                    f"{self.args.shots}/{frames:05d}-{self.scene.name.lower()}.png",
                )
            shot = self.args.screenshot
            if (
                shot
                and self.scene is Scene.PLAY
                and self.player.position() > self.args.auto_seconds * 0.8
            ):
                pygame.image.save(self.screen, shot)
                self.args.screenshot = None
            if self.args.auto and self.scene is Scene.RESULTS and frames > 5:
                if self.args.results_screenshot:
                    pygame.image.save(self.screen, self.args.results_screenshot)
                running = False
        self.player.pause()

    def handle(self, keys: list[int], samples: list[PoseSample]) -> bool:
        if pygame.K_m in keys:
            self.set_mirrored(not self.mirrored)
        if pygame.K_f in keys:
            self.set_fullscreen()
        adjust_scale(keys)
        if self.scene is Scene.MENU:
            return self.handle_menu(keys)
        if pygame.K_ESCAPE in keys and self.scene in (
            Scene.BODY,
            Scene.LATENCY,
            Scene.RESULTS,
            Scene.LEARN,
        ):
            self.go(Scene.MENU)
            return True
        if self.scene is Scene.SETTINGS:
            self.handle_settings(keys)
            return True
        if self.scene in (Scene.CREDITS, Scene.PRIVACY, Scene.SCORES):
            if self.scene is Scene.SCORES and (
                pygame.K_LEFT in keys or pygame.K_RIGHT in keys
            ):
                self.menu.routine_index += (
                    1 if pygame.K_RIGHT in keys else -1
                )  # another level
                self.doc_scroll = 0
            self.handle_document(keys)
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
        if pygame.K_c in keys or pygame.K_p in keys:
            self.go(Scene.CREDITS if pygame.K_c in keys else Scene.PRIVACY)
            return True
        if pygame.K_s in keys:
            self.go(Scene.SCORES)
            return True
        if pygame.K_o in keys:
            self.go(Scene.SETTINGS)
            return True
        if (
            pygame.K_RETURN in keys or pygame.K_SPACE in keys
        ) and m.routine is not None:
            self.next_after(Scene.MENU)
        return True

    def next_after(self, scene: Scene) -> None:
        """The scene that follows ``scene``, skipping calibrations that are kept."""
        m = self.menu
        if scene is Scene.MENU and (self.calibration is None or m.redo_body):
            return self.go(Scene.BODY)
        if scene in (Scene.MENU, Scene.BODY) and (
            self.latency_ms is None or m.redo_latency
        ):
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
            countdown, hold = (
                (0.4, 0.5)
                if self.args.fast
                else (CALIBRATION_COUNTDOWN_S, CALIBRATION_HOLD_S)
            )
            if elapsed > countdown:
                self.collected += samples
            if elapsed > countdown + hold:
                if self.step == 1:
                    self.neutral = Recording.from_samples(
                        self.collected, {"kind": "static"}
                    )
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
            elif not self.calibration_issues and (
                pygame.K_RETURN in keys or pygame.K_SPACE in keys
            ):
                self.next_after(Scene.BODY)

    # -- latency calibration ------------------------------------------------------

    def handle_latency(self, keys: list[int], samples: list[PoseSample]) -> None:
        for sample in samples:
            features = self.latency_extractor.update(sample)
            if self.step == 1 and features is not None:
                self.latency_collected.append(features)
        if (
            self.step == 0 and pygame.K_s in keys
        ):  # skip: reuse the last saved value (or none)
            self.latency_ms = (
                self.latency_ms
                if self.latency_ms is not None
                else (self.saved_latency_ms or 0.0)
            )
            self.next_after(Scene.LATENCY)
        elif self.step == 0 and (pygame.K_SPACE in keys or pygame.K_RETURN in keys):
            bpm, count_in, claps = (
                (600.0, 2, 8)
                if self.args.fast
                else (LATENCY_BPM, LATENCY_COUNT_IN, LATENCY_CLAPS)
            )
            grid = BeatGrid(bpm, 0.5, 0)
            beats = count_in + claps
            silence = np.zeros(
                int((0.5 + (beats + 1) * grid.beat_s) * self.player.sr), np.float32
            )
            self.player.play(
                render_click_track(
                    silence, self.player.sr, grid, music_gain=0.0, click_gain=0.9
                ),
                0.0,
            )
            start = self.player.start_ns
            self.click_ns = [
                start + int(grid.time_of(b) * 1e9) for b in range(count_in, beats)
            ]
            self.count_in_ns = [
                start + int(grid.time_of(b) * 1e9) for b in range(count_in)
            ]
            self.latency_beat_ns = grid.beat_s * 1e9
            self.latency_collected = []
            self.step = 1
        elif self.step == 1 and time.monotonic_ns() > self.click_ns[-1] + 1e9:
            self.latency_result = estimate_latency(
                self.latency_collected, self.click_ns, self.latency_beat_ns
            )
            self.step = 2
        elif self.step == 2:
            r = self.latency_result
            if pygame.K_r in keys:
                self.go(Scene.LATENCY)
            elif (
                r is not None
                and r.ok
                and (pygame.K_RETURN in keys or pygame.K_SPACE in keys)
            ):
                self.latency_ms = r.latency_ms
                settings.save_latency(r.latency_ms, r.spread_ms, r.claps_found)
                self.next_after(Scene.LATENCY)
            elif pygame.K_s in keys:  # skip: reuse the last saved value (or none)
                self.latency_ms = (
                    self.latency_ms
                    if self.latency_ms is not None
                    else (self.saved_latency_ms or 0.0)
                )
                self.next_after(Scene.LATENCY)

    # -- play ------------------------------------------------------------------

    def start_play(self) -> None:
        """Play the song (straight after the song is chosen) and score the routine.

        **Count-in** (fourth test run): the music starts at once and the last
        ``--count-in`` counts (4) before the first move are counted down on
        screen (4, 3, 2, 1); no clicks. Only a song whose intro is shorter
        than that gets silence in front (``self.lead_in_s``), so positions on
        the player are song time + ``lead_in_s``.
        """
        self.go(Scene.PLAY)
        song, routine = self.menu.song, self.menu.routine
        assert routine is not None
        samples = load_for_player(self.player, song.audio_path)
        sr = self.player.sr
        first_move_s = song.beat_time_s(song.routine_start_beat)
        self.lead_in_s = max(0.0, self.count_in * song.beat_s - first_move_s)
        track = np.concatenate(
            [np.zeros(round(self.lead_in_s * sr), np.float32), samples]
        )
        self.player.play(track, 0.0)
        song_start_ns = self.player.start_ns + round(self.lead_in_s * 1e9)
        self.session = PlaySession(song, routine, song_start_ns,
                                   int((self.latency_ms or 0.0) * 1e6), self.calibration)  # fmt: skip
        self.animator = Animator(routine)
        end_beat = song.routine_start_beat + routine.total_counts + 4
        self.play_end_s = self.lead_in_s + min(
            song.beat_time_s(end_beat), len(samples) / sr
        )
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
        fps = [f for f in self.play_fps if f > 0]
        self.results["camera_fps"] = round(float(np.median(fps)), 1) if fps else None
        self.results["low_light_pct"] = (
            round(100 * self.play_low / len(self.play_fps), 1) if self.play_fps else 0.0
        )
        self.placement = scores.record(self.results)  # None if ended early (not ranked)
        self.results["placement"] = scores.placement_dict(self.placement)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rec_dir = Path("recordings") / f"game-{stamp}"
        session.recording().save(rec_dir / "play.npz")
        path = write_log(
            "game", {"result": self.results, "recording": str(rec_dir / "play.npz")}
        )
        self.message = f"saved {path}"
        self.scene = Scene.RESULTS

    def handle_results(self, keys: list[int], samples: list[PoseSample]) -> None:
        if pygame.K_RETURN in keys or pygame.K_SPACE in keys:
            self.go(Scene.MENU)

    # -- credits & privacy ------------------------------------------------------------

    def handle_document(self, keys: list[int]) -> None:
        """Credits / Privacy: [Up/Down] scroll, [Esc]/[Enter] back to the menu."""
        if pygame.K_ESCAPE in keys or pygame.K_RETURN in keys or pygame.K_SPACE in keys:
            self.go(Scene.MENU)
        elif pygame.K_DOWN in keys and self.doc_more:
            self.doc_scroll += 1
        elif pygame.K_UP in keys:
            self.doc_scroll = max(0, self.doc_scroll - 1)

    def draw_credits(self, stage, camera, panel) -> None:
        self.doc_more = draw_document(
            stage, credits_entries(self.menu.songs), self.doc_scroll, "[Esc] back"
        )

    def draw_privacy(self, stage, camera, panel) -> None:
        self.doc_more = draw_document(
            stage, privacy_entries(), self.doc_scroll, "[Esc] back"
        )

    # -- settings ---------------------------------------------------------------------

    def set_mirrored(self, on: bool) -> None:
        self.mirrored = on
        settings.save_ui(mirror=on)

    def set_fullscreen(self) -> None:
        if toggle_fullscreen():
            settings.save_ui(fullscreen=bool(pygame.display.is_fullscreen()))

    def setting_rows(self) -> list[tuple[str, str, str]]:
        """``(label, value, explanation)`` of each setting, in screen order."""
        full = pygame.display.is_fullscreen()
        return [
            ("Text size", f"{UI.scale:.1f}", "Make it readable from where you dance (also + / -)."),
            ("Mirror view", "on" if self.mirrored else "off",
             "On: the dancer and your camera view move like a mirror (also M)."),
            ("Count-in", f"{self.count_in} counts" if self.count_in else "none",
             "Counted down above the dancer before the first move."),
            ("Fullscreen", "on" if full else "off", "Off: play in a window (also F)."),
        ]  # fmt: skip

    def handle_settings(self, keys: list[int]) -> None:
        """[Up/Down] choose, [Left/Right] change (saved at once), [Esc]/[Enter] back."""
        if pygame.K_ESCAPE in keys or pygame.K_RETURN in keys:
            self.go(Scene.MENU)
            return
        rows = len(self.setting_rows())
        self.setting_index = (
            self.setting_index + (pygame.K_DOWN in keys) - (pygame.K_UP in keys)
        ) % rows
        step = (pygame.K_RIGHT in keys) - (pygame.K_LEFT in keys)
        if not step:
            return
        if self.setting_index == 0:
            change_scale(step)
        elif self.setting_index == 1:
            self.set_mirrored(not self.mirrored)
        elif self.setting_index == 2:
            i = (
                COUNT_IN_CHOICES.index(self.count_in)
                if self.count_in in COUNT_IN_CHOICES
                else 2
            )
            self.count_in = COUNT_IN_CHOICES[
                min(len(COUNT_IN_CHOICES) - 1, max(0, i + step))
            ]
            settings.save_ui(count_in=self.count_in)
        else:
            self.set_fullscreen()

    def draw_settings(self, stage, camera, panel) -> None:
        y = text(stage, "Settings", (24, 16), 34, ACCENT)
        bottom = footer(
            stage, [("[Up/Down] choose   [Left/Right] change   [Esc] back", 16, DIM)]
        )
        for i, (label, value, help_text) in enumerate(self.setting_rows()):
            selected = i == self.setting_index
            box_h = (text_height(stage, label, 22, 36) + text_height(stage, f"<  {value}  >", 24, 36)
                     + (text_height(stage, help_text, 16, 36) if selected else 0) + 8)  # fmt: skip
            if y + 8 + box_h > bottom:
                break
            rect = pygame.Rect(12, y + 8, STAGE_W - 24, box_h)
            if selected:
                pygame.draw.rect(stage, PANEL, rect)
                pygame.draw.rect(stage, ACCENT, rect, 3)
                register_box(stage, rect, "[setting]")
            inner = stage.subsurface(rect.inflate(-8, -8))
            ly = text(inner, label, (20, 0), 22, TEXT)
            ly = text(
                inner,
                f"<  {value}  >" if selected else value,
                (20, ly),
                24,
                ACCENT if selected else DIM,
            )
            if selected:
                text(inner, help_text, (20, ly), 16, DIM)
            y = rect.bottom
        footer(panel, [(f"{NAME} {__version__}", 14, DIM)], x=20)

    # -- score board ----------------------------------------------------------------

    @staticmethod
    def board_line(rank: int, entry: dict) -> str:
        played = datetime.fromisoformat(entry["played"]).strftime("%d %b %H:%M")
        return f"{rank}.   {entry['score']:,}   {entry['accuracy_pct']:.0f}%   {played}"

    def draw_scores(self, stage, camera, panel) -> None:
        """The score board of the selected song: Left/Right switch level."""
        song, routine = self.menu.song, self.menu.routine
        entries = [("title", "Score board"), ("heading", song.title)]
        if routine is None:
            entries.append(("small", "no routines"))
        else:
            entries.append(("text", f"< {routine.difficulty.value} >"))
            board = scores.top(song.id, routine.id, 10)
            entries += [
                ("text", self.board_line(i + 1, e)) for i, e in enumerate(board)
            ] or [("small", "No plays yet: finish a song to get on the board.")]
        self.doc_more = draw_document(
            stage, entries, self.doc_scroll, "[Left/Right] level   [Esc] back"
        )

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
        self.learn_track = render_click_track(
            silence, self.player.sr, self.learn_grid, music_gain=0.0, click_gain=0.5
        )
        self.player.play(self.learn_track, 0.0)

    def handle_learn(self, keys: list[int], samples: list[PoseSample]) -> None:
        step = (pygame.K_RIGHT in keys) - (pygame.K_LEFT in keys)
        faster = (pygame.K_UP in keys) - (pygame.K_DOWN in keys)
        if step or faster:
            self.learn_index = (self.learn_index + step) % len(self.learn_blocks)
            self.learn_speed = min(
                len(LEARN_SPEEDS) - 1, max(0, self.learn_speed + faster)
            )
            self.start_learn()
        elif self.player.position() >= len(self.learn_track) / self.player.sr:
            self.start_learn()  # loop

    # -- drawing -----------------------------------------------------------------
    # Sizes below are *base* sizes, multiplied by UI.scale (default 2.0) in
    # text(). Screens show few, short lines: they are read from 1.5-2 m away.
    # Layout rules (checked by --check-layout at every text size): text flows
    # down from the top using the y each text() call returns, key hints are
    # stacked up from the bottom with footer(), and big fixed-position labels
    # move down when the text above them needs the room.

    def draw(self) -> None:
        UI.drawn = [] if self.args.check_layout else None
        self.screen.fill(BG)
        camera = self.screen.subsurface(
            (STAGE_W, 0, self.config.width, self.config.height)
        )
        self.feed.draw(camera, self.mirrored)
        failure = self.feed.failure(self.capture)
        if failure:
            y = text(camera, "CAMERA PROBLEM", (20, 20), 30, ERROR)
            text(camera, failure, (20, y), 18, ERROR)
        elif self.feed.ready_at is None:
            text(
                camera,
                "starting camera ...",
                (camera.get_width() // 2, 200),
                28,
                DIM,
                centre=True,
            )
        stage = self.screen.subsurface((0, 0, STAGE_W, HEIGHT))
        panel = self.screen.subsurface(
            (STAGE_W, self.config.height, WIDTH - STAGE_W, HEIGHT - self.config.height)
        )
        panel.fill(PANEL)
        # Low-light hint at the bottom of the camera view (other warnings go above it).
        self.camera_hint_top = camera.get_height()
        if self.low_light.active and not failure and self.feed.ready_at is not None:
            hint = f"Camera {self.low_light.fps:.0f} fps: more light helps"
            x = max(8, camera.get_width() // 2 - font(20).size(hint)[0] // 2)
            h = text_height(camera, hint, 20, x)
            band = pygame.Surface((camera.get_width(), h + 12), pygame.SRCALPHA)
            band.fill((0, 0, 0, 160))  # readable on a bright camera picture
            camera.blit(band, (0, camera.get_height() - h - 16))
            self.camera_hint_top = footer(camera, [(hint, 20, WARN)], x=x)
        getattr(self, f"draw_{self.scene.name.lower()}")(stage, camera, panel)
        if UI.drawn is not None:
            for issue in layout_issues(UI.drawn):
                if issue not in self.layout_reported:
                    self.layout_reported.add(issue)
                    print(
                        f"LAYOUT ({self.scene.name.lower()}, text size {UI.scale}): {issue}",
                        file=sys.stderr,
                    )

    def draw_menu(self, stage, camera, panel) -> None:
        y = self.draw_title(stage)
        m = self.menu
        if not m.songs:
            text(
                stage, "No songs: run parapara-fetch-demo-songs", (30, y + 20), 22, WARN
            )
            return
        bottom = footer(stage, [("New? [T] learn the moves", 18, ACCENT),
                                ("[Up/Down] song   [Left/Right] level   [Enter] play", 16, DIM),
                                ("[S] scores   [O] settings   [C] credits   [P] privacy   [Esc] quit", 16, DIM)])  # fmt: skip
        # Song list: the selected song is framed with its level and tempo
        # (third test run: it blended into the other titles). If the list is
        # taller than the room above the key hints, it scrolls to keep the
        # selected song in view.
        inner_w = STAGE_W - 2 * 12
        box_surface = pygame.Surface((inner_w, HEIGHT))  # for measuring only

        def selected_lines(song: Song) -> list[tuple[str, float, tuple[int, int, int]]]:
            r = m.routine
            lines: list[tuple[str, float, tuple[int, int, int]]] = [(song.title, 24, ACCENT)]
            licence = song_credit(song)[1][1].split(",")[
                0
            ]  # "CC BY 4.0" / "your own music"
            if (
                song.artist
            ):  # artist and licence right where the song is chosen, as CC BY asks
                lines.append((f"by {song.artist}   {licence}", 16, DIM))
            if r is not None:
                lines.append((f"< {r.difficulty.value} >   {len(r.phrases)} phrases   {song.bpm:.0f} BPM", 20, TEXT))
                top = scores.best(song.id, r.id)
                if top:
                    lines.append(
                        (f"best {top['score']:,}   {top['accuracy_pct']:.0f}%", 16, DIM)
                    )
            else:
                lines.append(("no routines", 20, TEXT))
            if not song.beat_map.verified:
                lines.append(("beat map not checked", 18, WARN))
            return lines

        def height(i: int) -> int:
            if i == m.song_index:
                return 16 + sum(
                    text_height(box_surface, s, size, 12)
                    for s, size, _ in selected_lines(m.songs[i])
                )
            return text_height(stage, m.songs[i].title, 20) + 2

        y += 10
        heights = [height(i) for i in range(len(m.songs))]
        first = 0
        while (
            first < m.song_index
            and y + sum(heights[first : m.song_index + 1]) > bottom - 8
        ):
            first += 1
        for i in range(first, len(m.songs)):
            if y + heights[i] > bottom - 8:
                break
            song = m.songs[i]
            if i == m.song_index:
                rect = pygame.Rect(12, y, inner_w, heights[i] - 4)
                pygame.draw.rect(stage, PANEL, rect)
                pygame.draw.rect(stage, ACCENT, rect, 3)
                register_box(stage, rect, "[selected song]")
                inner = stage.subsurface(rect.inflate(-6, -6))
                ly = 5
                for s, size, colour in selected_lines(song):
                    ly = text(inner, s, (9, ly), size, colour)
            else:
                text(
                    stage,
                    song.title,
                    (24, y),
                    20,
                    TEXT if song.beat_map.verified else DIM,
                )
            y += heights[i]

        def status(name: str, value: str | None, redo: bool, key: str) -> str:
            if value is None:
                return f"{name}: before the first song"
            return f"{name}: {value}" + ("  -> redo" if redo else f"  ({key} redo)")

        footer(panel, [(f"{NAME} {__version__}", 14, DIM)], x=20)
        y = text(
            panel,
            status("body", "OK" if self.calibration else None, m.redo_body, "B"),
            (20, 16),
            22,
        )
        lat = f"{self.latency_ms:.0f} ms" if self.latency_ms is not None else None
        text(panel, status("latency", lat, m.redo_latency, "L"), (20, y), 22)

    def draw_title(self, stage) -> int:
        """The ParaParaPy logo (or its name, if the logo file is missing) at the
        top of the menu; returns the y below it."""
        logo = wordmark(round(44 * UI.scale))
        if logo is None:
            return text(stage, NAME, (30, 18), 48, ACCENT)
        if (
            logo.get_width() > STAGE_W - 48
        ):  # very large text sizes: keep it on the stage
            logo = pygame.transform.smoothscale(
                logo,
                (
                    STAGE_W - 48,
                    round(logo.get_height() * (STAGE_W - 48) / logo.get_width()),
                ),
            )
        rect = stage.blit(logo, (20, 10))
        register_box(stage, rect, "[logo]")
        return rect.bottom + 4

    def draw_body(self, stage, camera, panel) -> None:
        titles = {
            0: "Stand in position",
            1: "Neutral pose",
            2: "Stand in position",
            3: "T-pose",
            4: "Calibration",
        }
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
            how = (
                "Arms down, relaxed"
                if self.step == 1
                else "Arms straight out to the sides"
            )
            y = text(stage, how, (24, y + 10), 26)
            elapsed = time.monotonic() - self.t0
            countdown = 0.4 if self.args.fast else CALIBRATION_COUNTDOWN_S
            label = (
                str(int(np.ceil(countdown - elapsed)))
                if elapsed < countdown
                else "HOLD"
            )
            # The pose to hold, shown by a stick figure (fifth test run) between
            # the instructions and the countdown at the bottom; fitted like the
            # Learn screen's dancer (head 0.8 shoulder widths above the
            # shoulders, feet 3.3 below).
            label_top = HEIGHT - 10 - text_height(stage, "HOLD", 80)
            text(stage, label, (STAGE_W // 2, label_top), 80, ACCENT, centre=True)
            scale = min(DANCER_SCALE, (label_top - y - 20) / 4.1)
            pose = static_pose(
                CALIBRATION_POSES["neutral" if self.step == 1 else "t_pose"]
            )
            draw_dancer(
                stage,
                pose,
                (STAGE_W // 2, round(y + 10 + 0.8 * scale)),
                scale,
                self.mirrored,
            )
        elif self.calibration_issues:
            bottom = footer(stage, [("[R] retry   [Esc] menu", 20, DIM)])
            y = text(stage, "Try again:", (24, y + 10), 28, WARN)
            if y < bottom:
                text(stage, self.calibration_issues[0], (24, y), 22, WARN)
        else:
            footer(stage, [("[Enter] continue", 22, DIM)])
            text(stage, "Done!", (24, y + 20), 40, OK)

    def draw_latency(self, stage, camera, panel) -> None:
        y = text(stage, "Clap to the beat", (24, 20), 34, ACCENT)
        last = (
            f" (last: {self.saved_latency_ms:.0f} ms)"
            if self.saved_latency_ms is not None
            else ""
        )
        if self.step == 0:
            footer(stage, [("[Space] start", 24, ACCENT), (f"[S] skip{last}", 18, DIM)])
            y = text(stage, f"Listen to {LATENCY_COUNT_IN} clicks, then clap on the next {LATENCY_CLAPS}.",
                     (24, y + 10), 24)  # fmt: skip
            y = text(stage, "Hands in view.", (24, y), 24)
            text(
                stage,
                "Once per game start: the next songs use it too.",
                (24, y + 16),
                18,
                DIM,
            )
        elif self.step == 1:
            # 4, 3, 2, 1 on the count-in clicks (sixth test run), then CLAP!
            # with the number of claps so far.
            now = time.monotonic_ns()
            done = sum(c <= now for c in self.click_ns)
            heard = sum(c <= now for c in self.count_in_ns)
            label = (
                "CLAP!"
                if done
                else (str(len(self.count_in_ns) - heard + 1) if heard else "ready")
            )
            beat_ns = self.latency_beat_ns
            on_click = any(
                0 <= now - c < 0.3 * beat_ns for c in self.count_in_ns + self.click_ns
            )
            y = text(stage, label, (STAGE_W // 2, max(y + 40, 230)), 64 if done or not heard else 90,
                     ACCENT if on_click or done else TEXT, centre=True)  # fmt: skip
            if done:
                text(
                    stage,
                    f"{done} / {len(self.click_ns)}",
                    (STAGE_W // 2, y + 20),
                    40,
                    TEXT,
                    centre=True,
                )
        else:
            r = self.latency_result
            if r and r.ok:
                footer(stage, [("[Enter] accept   [R] retry", 22, DIM)])
                y = text(
                    stage,
                    f"{r.latency_ms:.0f} ms",
                    (STAGE_W // 2, max(y + 40, 220)),
                    72,
                    OK,
                    centre=True,
                )
                text(
                    stage,
                    f"{r.claps_found} claps heard",
                    (STAGE_W // 2, y + 10),
                    22,
                    DIM,
                    centre=True,
                )
            else:
                bottom = footer(stage, [(f"[R] retry   [S] skip{last}", 22, DIM)])
                y = text(stage, "Try again:", (24, y + 10), 28, WARN)
                if y < bottom:
                    text(stage, r.reason if r else "no result", (24, y), 22, WARN)

    def draw_play(self, stage, camera, panel) -> None:
        session, song = self.session, self.menu.song
        assert session is not None and self.animator is not None
        pos = self.player.position() - self.lead_in_s  # song time
        beat = song.beat_at(pos) - song.routine_start_beat
        pose = self.animator.pose_at(beat)
        draw_dancer(stage, pose, (DANCER_X, DANCER_Y), DANCER_SCALE, self.mirrored)
        counting_in = -self.count_in <= beat < 0
        self.draw_count(stage, pose.count, beat, idle="" if counting_in else "ready")
        # The next move as a still picture of its first pose (a looping
        # animation was too distracting: third test run). Just after a move
        # starts, its picture stays a moment with the ring lit (the dot has
        # arrived), then the box moves on to the following move.
        current = self.animator.block_at(beat)
        if current is not None and beat - current.start < NEXT_HIT_BEATS:
            self.draw_next(
                stage, static_pose(current.block.keyframes[0]), current.start - beat
            )
        elif pose.next_block is not None and pose.next_block_beat is not None:
            self.draw_next(
                stage,
                static_pose(pose.next_block.keyframes[0]),
                pose.next_block_beat - beat,
            )
        # Above the dancer's head (players watch the dancer, not their camera
        # view): the count-in, then the grade pop-ups. The area ends at the
        # "next" box, so long words shrink rather than run under it.
        top = stage.subsurface((0, 0, NEXT_BOX.left - 8, DANCER_Y - 40))
        if counting_in:  # 4, 3, 2, 1 into the first move, flashing on each count
            on_beat = beat - np.floor(beat) < 0.25
            text(
                top,
                str(int(np.ceil(-beat))),
                (DANCER_X, 0),
                64,
                ACCENT if on_beat else TEXT,
                centre=True,
            )
        now = time.monotonic()
        for t, grade, points in self.popups[-1:]:
            label = grade.value.upper() + (
                "!" if grade in (Grade.PERFECT, Grade.GREAT) else ""
            )
            rise = round(12 * (1 - (now - t) / POPUP_S))
            text(
                top, label, (DANCER_X, 18 - rise), 56, GRADE_COLOUR[grade], centre=True
            )
        warning = session.framing_warning
        if warning is not None:  # above the low-light hint, if that is shown
            message = warning.issues[0].message
            y = min(
                camera.get_height() - 120,
                self.camera_hint_top
                - 8
                - text_height(camera, message, 28, camera.get_width() // 2),
            )
            text(camera, message, (camera.get_width() // 2, y), 28, WARN, centre=True)
        k = session.keeper
        y = text(panel, f"{k.score:,}", (20, 6), 56, ACCENT)
        text(panel, f"combo {k.combo}", (20, y), 30)
        progress = (
            min(1.0, max(0.0, pos) / (self.play_end_s - self.lead_in_s))
            if self.play_end_s > self.lead_in_s
            else 0.0
        )
        bar = pygame.Rect(20, panel.get_height() - 22, panel.get_width() - 40, 12)
        pygame.draw.rect(panel, DIM, bar, 1)
        pygame.draw.rect(panel, ACCENT, (bar.x, bar.y, round(bar.w * progress), bar.h))

    def draw_count(
        self,
        stage,
        count: int,
        beat: float,
        top: int | None = None,
        idle: str = "ready",
    ) -> int:
        """The count (1-8) below the dancer's feet, flashing on the beat
        (``idle`` before the routine); returns its top y."""
        on_beat = count and beat - np.floor(beat) < 0.12
        label = str(count) if count else (idle if beat < 0 else "")
        size = 56 if count else 34
        y = HEIGHT - 10 - text_height(stage, "8", 56) if top is None else top
        text(
            stage, label, (DANCER_X, y), size, ACCENT if on_beat else TEXT, centre=True
        )
        return y

    def draw_next(self, stage, pose: DancerPose, beats_to_go: float) -> None:
        """Small "next move" picture at the dancer's top right: a framed still
        stick figure in the move's first pose, and a moving dot for its timing.

        **Timing dot** (sixth test run: counting dots were hard to read): a
        ring on the left of a track under the picture is the move's start. In
        the last NEXT_LEAD_BEATS counts a dot slides at constant speed from the
        right end of the track into the ring, arriving exactly when the move
        starts (``beats_to_go`` = 0); further away it waits, dimmed, at the
        right end. At 0 or just after it (``beats_to_go`` <= 0, see
        draw_play) the ring lights up: "now".

        Only the upper body is shown (moves are arms only), so the figure can
        be drawn large: the drawing is clipped to a window from above raised
        hands down to just below hanging hands.
        """
        box = stage.subsurface(NEXT_BOX)
        box.fill(PANEL)
        pygame.draw.rect(box, ACCENT, box.get_rect(), 3)
        register_box(stage, NEXT_BOX, "[next box]")
        y = text(box, "next", (box.get_width() // 2, 4), 16, ACCENT, centre=True)
        ring_r, dot_r = 19, 13
        track_y = box.get_height() - ring_r - 12
        label_y = track_y - ring_r - 6
        scale = 52  # px per shoulder width
        window_h = max(10, min(round(3.6 * scale), label_y - y))
        window = box.subsurface((3, y, box.get_width() - 6, window_h))
        draw_dancer(
            window,
            pose,
            (window.get_width() // 2, round(1.7 * scale)),
            scale,
            self.mirrored,
        )
        ring_x, end_x = 14 + ring_r, box.get_width() - 14 - dot_r
        pygame.draw.line(box, DIM, (ring_x, track_y), (end_x, track_y), 3)
        if beats_to_go <= 0:  # the move starts now
            pygame.draw.circle(box, ACCENT, (ring_x, track_y), ring_r)
            return
        pygame.draw.circle(box, ACCENT, (ring_x, track_y), ring_r, 4)
        u = min(
            1.0, beats_to_go / NEXT_LEAD_BEATS
        )  # 1 = waiting at the right end, 0 = in the ring
        colour = ACCENT if beats_to_go <= NEXT_LEAD_BEATS else DIM
        pygame.draw.circle(
            box, colour, (round(ring_x + (end_x - ring_x) * u), track_y), dot_r
        )

    def draw_learn(self, stage, camera, panel) -> None:
        block = self.learn_blocks[self.learn_index]
        assert self.animator is not None
        beat = self.learn_grid.beat_at(self.player.position()) % self.animator.end_beat
        pose = self.animator.pose_at(beat)
        top = text(stage, block_title(block), (16, 8), 30, ACCENT)
        count_top = self.draw_count(stage, pose.count, beat)
        # Fit the dancer between the title and the count: from raised hands
        # (1.7 shoulder widths above the shoulders) to the feet (3.3 below).
        scale = min(DANCER_SCALE, (count_top - top - 10) / 5.0)
        draw_dancer(
            stage, pose, (DANCER_X, round(top + 1.7 * scale)), scale, self.mirrored
        )
        bottom = footer(
            panel, [("[Left/Right] move   [Up/Down] speed   [Esc] menu", 14, DIM)], x=20
        )
        y = text(panel, f"move {self.learn_index + 1} / {len(self.learn_blocks)}   "
                        f"speed {LEARN_SPEEDS[self.learn_speed]:.0%}", (20, 10), 16, DIM)  # fmt: skip
        description = panel.subsurface((0, y, panel.get_width(), max(1, bottom - y)))
        text(description, block.description, (20, 0), 18)

    def draw_results(self, stage, camera, panel) -> None:
        r = self.results or {}
        footer(stage, [("[Enter] menu", 22, DIM)])
        y = text(stage, "Results", (24, 18), 40, ACCENT)
        y = text(stage, f"{r.get('score', 0):,}", (24, y), 56)
        y = text(stage, f"accuracy {r.get('accuracy_pct', 0):.0f}%", (24, y + 6), 28)
        y = text(stage, f"best combo {r.get('max_combo', 0)}", (24, y), 28)
        g = r.get("grades", {})
        y = text(stage, f"perfect {g.get('perfect', 0)}  great {g.get('great', 0)}  good {g.get('good', 0)}",
                 (24, y + 6), 20, DIM)  # fmt: skip
        y = text(
            stage, f"hold {g.get('hold', 0)}  miss {g.get('miss', 0)}", (24, y), 20, DIM
        )
        p = self.placement
        if r.get("ended_early"):
            text(stage, "ended early: not on the score board", (24, y + 6), 18, DIM)
        elif p is not None and p.new_best:
            text(stage, "New best!", (24, y + 6), 32, OK)
        elif p is not None:
            text(
                stage, f"#{p.rank} of {p.entries} on this level", (24, y + 6), 24, TEXT
            )
        # Panel: posture / latency, then this level's board (yours highlighted).
        bottom = footer(panel, [(self.message, 12, DIM)], x=20)
        y = text(panel, f"posture {r.get('posture_pct', 0):.0f}%   latency {r.get('latency_ms', 0):.0f} ms",
                 (20, 10), 20)  # fmt: skip
        routine = self.menu.routine
        if routine is None or p is None:
            return
        y = text(
            panel, f"Score board ({routine.difficulty.value})", (20, y + 4), 18, ACCENT
        )
        for i, entry in enumerate(scores.top(self.menu.song.id, routine.id, 5)):
            line = self.board_line(i + 1, entry)
            if y + text_height(panel, line, 16, 20) > bottom:
                break
            y = text(panel, line, (20, y), 16, ACCENT if i + 1 == p.rank else TEXT)


def run(argv: list[str] | None = None, *, worker: str = DEFAULT_WORKER) -> None:
    parser = argparse.ArgumentParser(description="Para Para dance game")
    parser.add_argument(
        "--latency-ms", type=float, help="override the latency calibration"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", choices=("lite", "full"), default="full")
    parser.add_argument(
        "--window", action="store_true", help="run in a window instead of fullscreen"
    )
    parser.add_argument(
        "--text-scale",
        type=float,
        help="text size (default: saved, or 2.0; +/- in game)",
    )
    parser.add_argument(
        "--count-in",
        type=int,
        help="counts shown (4, 3, 2, 1) before the first move "
        "(default: the saved setting, or 4)",
    )
    parser.add_argument("--reuse-calibration", action="store_true",
                        help="use the saved calibrations instead of calibrating at launch")  # fmt: skip
    parser.add_argument(
        "--check-layout", action="store_true", help=argparse.SUPPRESS
    )  # tests: report overlapping text
    parser.add_argument(
        "--auto", action="store_true", help=argparse.SUPPRESS
    )  # tests: play straight away
    parser.add_argument(
        "--auto-seconds", type=float, default=3.0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--screenshot", help=argparse.SUPPRESS
    )  # save one play frame (reviews)
    parser.add_argument(
        "--fast", action="store_true", help=argparse.SUPPRESS
    )  # tests: short calibrations
    parser.add_argument("--shots", help=argparse.SUPPRESS)  # reviews: screenshot folder
    parser.add_argument(
        "--keys", help=argparse.SUPPRESS
    )  # tests: "frame:key ..." scripted presses
    parser.add_argument("--results-screenshot", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # Fullscreen unless --window or the saved setting says window (Settings, F).
    args.window = args.window or not settings.load_ui().get("fullscreen", True)
    config = CaptureConfig(camera_index=args.camera, pose_model=args.model)
    pygame.init()
    try:
        # The privacy notice comes before the camera (and MediaPipe) starts.
        load_scale(args.text_scale)
        accepted, frames = privacy_gate(args.window, args.keys, args.check_layout)
        if not accepted:
            return
        with CaptureClient(config, worker=worker) as capture:
            Game(args, capture, config).run(start_frame=frames)
    finally:
        pygame.quit()


def main(argv: list[str] | None = None) -> None:
    """Console entry point (returns ``None``)."""
    run(argv)


if __name__ == "__main__":
    main()
