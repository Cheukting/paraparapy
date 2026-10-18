"""The game: ``uv run parapara-play`` (roadmap Step 6).

Flow (one capture process for the whole session)::

    MENU ──▶ BODY calibration ──▶ LATENCY calibration ──▶ PLAY ──▶ RESULTS ──▶ MENU
             (skipped when a        (skipped when a
              saved one is kept)     saved one is kept)

- **Menu**: songs (built-in demo tracks and your imports) with their routines;
  unverified beat maps are flagged. ``B`` / ``L`` redo the body / latency
  calibration even if saved ones exist.
- **Body calibration**: the position check (:mod:`parapara.pose.framing`)
  must pass, then a neutral pose and a T-pose are held for a few seconds each
  (:mod:`parapara.pose.calibration`); saved in the user data folder.
- **Latency calibration**: clap on each of 16 metronome clicks; the median
  delay, measured through the same feature pipeline as the scoring
  (:mod:`parapara.scoring.latency`), shifts the scoring schedule.
- **Play**: the demo dancer on the left (mirrored, with the count and a
  next-move card), your mirrored camera view with skeleton on the right
  (side by side, as decided), grade pop-ups, score, combo and progress; a
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
from parapara.animation.animator import Animator  # noqa: E402
from parapara.audio.beatgrid import BeatGrid, render_click_track  # noqa: E402
from parapara.audio.importer import load_all_songs  # noqa: E402
from parapara.audio.song import Song  # noqa: E402
from parapara.diagnostics.recorder import write_log  # noqa: E402
from parapara.game.beatcheck import Player, load_for_player  # noqa: E402
from parapara.game.dancer_view import block_title, draw_dancer, static_pose  # noqa: E402
from parapara.game.debug_view import ERROR_COLOUR, LiveFeed, draw_hud  # noqa: E402
from parapara.game.recorder import draw_framing_box  # noqa: E402
from parapara.ipc.client import DEFAULT_WORKER, CaptureClient  # noqa: E402
from parapara.ipc.messages import CaptureConfig  # noqa: E402
from parapara.pose.calibration import CalibrationError, calibrate  # noqa: E402
from parapara.pose.features import FeatureExtractor  # noqa: E402
from parapara.pose.framing import FramingMonitor  # noqa: E402
from parapara.pose.landmarks import PoseSample  # noqa: E402
from parapara.pose.recording import Recording  # noqa: E402
from parapara.routine.model import Routine  # noqa: E402
from parapara.routine.store import routines_for  # noqa: E402
from parapara.scoring.judge import FlowJudgment, Grade  # noqa: E402
from parapara.scoring.latency import LatencyResult, estimate_latency  # noqa: E402
from parapara.scoring.session import PlaySession  # noqa: E402

WIDTH, HEIGHT = 1240, 720
STAGE_W = 600  # demo dancer on the left; camera (640x480) + score panel on the right
BG, PANEL, TEXT, DIM, ACCENT = (18, 18, 28), (30, 30, 44), (235, 235, 245), (120, 120, 140), (255, 220, 80)
OK, WARN = (90, 220, 120), (255, 150, 60)
GRADE_COLOUR = {Grade.PERFECT: (255, 220, 80), Grade.GREAT: (90, 220, 120), Grade.GOOD: (110, 170, 255),
                Grade.HOLD: (90, 220, 120), Grade.MISS: (255, 90, 90)}  # fmt: skip
POPUP_S = 0.7
CALIBRATION_HOLD_S, CALIBRATION_COUNTDOWN_S = 3.0, 3.0
LATENCY_BPM, LATENCY_COUNT_IN, LATENCY_CLAPS = 100.0, 4, 16


class Scene(Enum):
    MENU = auto()
    BODY = auto()
    LATENCY = auto()
    PLAY = auto()
    RESULTS = auto()


def text(surface, s, pos, size=24, colour=TEXT, centre=False) -> int:
    r = pygame.font.Font(None, size).render(s, True, colour)
    x, y = pos
    if centre:
        x -= r.get_width() // 2
    surface.blit(r, (x, y))
    return y + r.get_height() + 4


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
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("parapara")
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
        self.scene = scene
        self._reset_scene_state()
        if scene is Scene.LATENCY:
            self.latency_extractor = FeatureExtractor(calibration=self.calibration)

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
        if self.scene is Scene.MENU:
            return self.handle_menu(keys)
        if pygame.K_ESCAPE in keys and self.scene in (Scene.BODY, Scene.LATENCY, Scene.RESULTS):
            self.go(Scene.MENU)
            return True
        {Scene.BODY: self.handle_body, Scene.LATENCY: self.handle_latency, Scene.PLAY: self.handle_play,
         Scene.RESULTS: self.handle_results}[self.scene](keys, samples)  # fmt: skip
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

    # -- drawing -----------------------------------------------------------------

    def draw(self) -> None:
        self.screen.fill(BG)
        camera = self.screen.subsurface((STAGE_W, 0, self.config.width, self.config.height))
        self.feed.draw(camera, self.mirrored)
        failure = self.feed.failure(self.capture)
        if failure:
            draw_hud(camera, ["CAPTURE ERROR", failure], ERROR_COLOUR)
        stage = self.screen.subsurface((0, 0, STAGE_W, HEIGHT))
        panel = self.screen.subsurface((STAGE_W, self.config.height, WIDTH - STAGE_W, HEIGHT - self.config.height))
        panel.fill(PANEL)
        getattr(self, f"draw_{self.scene.name.lower()}")(stage, camera, panel)

    def draw_menu(self, stage, camera, panel) -> None:
        y = text(stage, "parapara", (30, 24), 64, ACCENT)
        y = text(stage, "Up/Down song   Left/Right routine   Enter play   M mirror   Esc quit", (30, y), 20, DIM)
        m = self.menu
        if not m.songs:
            text(stage, "No songs: run `uv run parapara-fetch-demo-songs`", (30, y + 20), 24, WARN)
            return
        y += 16
        for i, song in enumerate(m.songs):
            selected = i == m.song_index
            flag = "" if song.beat_map.verified else "  (beat map unverified)"
            y = text(stage, f"{'▶ ' if selected else '  '}{song.title} · {song.bpm:.0f} BPM{flag}", (30, y),
                     28 if selected else 24, ACCENT if selected else TEXT)  # fmt: skip
            if selected:
                r = m.routine
                label = (f"routine: {r.id.removeprefix(song.id + '-')} ({r.difficulty.value}, "
                         f"{len(r.phrases)} phrases)") if r else "no routines: run `parapara-routines generate`"  # fmt: skip
                y = text(stage, "      " + label, (30, y), 22, DIM)
        cal = "saved" if self.calibration else "needed"
        lat = f"{self.latency_ms:.0f} ms" if self.latency_ms is not None else "needed"
        y = text(panel, f"body calibration: {cal}{'  (redo: B)' if self.calibration else ''}"
                        f"{'  -> will redo' if m.redo_body else ''}", (20, 20), 24)  # fmt: skip
        text(panel, f"latency: {lat}{'  (redo: L)' if self.latency_ms is not None else ''}"
                    f"{'  -> will redo' if m.redo_latency else ''}", (20, y), 24)  # fmt: skip

    def draw_body(self, stage, camera, panel) -> None:
        titles = {0: "Position check", 1: "Neutral pose", 2: "Position check", 3: "T-pose", 4: "Calibration"}
        y = text(stage, titles[self.step], (30, 30), 48, ACCENT)
        latest = self.framing.latest
        if self.step in (0, 2):
            draw_framing_box(camera, latest, self.mirrored)
            for issue in (latest.issues if latest else ())[:2]:
                y = text(stage, issue.message, (30, y + 10), 26, WARN)
            if latest is not None and latest.ok:
                text(stage, "Position OK, hold still ...", (30, y + 10), 26, OK)
        elif self.step in (1, 3):
            pose = "Stand relaxed, arms down." if self.step == 1 else "Arms straight out, level with shoulders."
            y = text(stage, pose, (30, y + 10), 26)
            elapsed = time.monotonic() - self.t0
            label = str(int(np.ceil(CALIBRATION_COUNTDOWN_S - elapsed))) if elapsed < CALIBRATION_COUNTDOWN_S else "HOLD"
            text(stage, label, (STAGE_W // 2, 300), 120, ACCENT, centre=True)
        else:
            if self.calibration_issues:
                y = text(stage, "Calibration failed:", (30, y + 10), 28, WARN)
                for issue in self.calibration_issues:
                    y = text(stage, issue, (30, y), 22, WARN)
                text(stage, "R retry   Esc menu", (30, y + 20), 24, DIM)
            else:
                c = self.calibration
                y = text(stage, "Done!", (30, y + 10), 32, OK)
                text(stage, f"shoulders {c.shoulder_width_m:.2f} m · lean {c.neutral_lean_deg:.0f}° · Enter continue",
                     (30, y), 22, DIM)  # fmt: skip

    def draw_latency(self, stage, camera, panel) -> None:
        y = text(stage, "Clap to the beat", (30, 30), 48, ACCENT)
        if self.step == 0:
            y = text(stage, f"After {LATENCY_COUNT_IN} count-in clicks, clap on each of the next {LATENCY_CLAPS}.",
                     (30, y + 10), 24)  # fmt: skip
            text(stage, "Keep your hands in view. Space start   Esc menu", (30, y), 22, DIM)
        elif self.step == 1:
            now = time.monotonic_ns()
            done = sum(c <= now for c in self.click_ns)
            text(stage, "listen ..." if done == 0 else f"clap! {done}/{len(self.click_ns)}", (STAGE_W // 2, 300), 72,
                 ACCENT, centre=True)  # fmt: skip
        else:
            r = self.latency_result
            if r and r.ok:
                y = text(stage, f"latency {r.latency_ms:.0f} ms (spread {r.spread_ms:.0f} ms, {r.claps_found} claps)",
                         (30, y + 10), 28, OK)  # fmt: skip
                text(stage, "Enter accept   R retry", (30, y), 22, DIM)
            else:
                y = text(stage, r.reason if r else "no result", (30, y + 10), 26, WARN)
                text(stage, "R retry   S skip (keep previous)", (30, y), 22, DIM)

    def draw_play(self, stage, camera, panel) -> None:
        session, song = self.session, self.menu.song
        assert session is not None and self.animator is not None
        pos = self.player.position()
        beat = song.beat_at(pos) - song.routine_start_beat
        pose = self.animator.pose_at(beat)
        draw_dancer(stage, pose, (STAGE_W // 2, 170), HEIGHT / 7.2, self.mirrored)
        text(stage, str(pose.count) if pose.count else ("get ready" if beat < 0 else ""), (24, 14), 72 if pose.count else 36,
             ACCENT if pose.count and beat - np.floor(beat) < 0.12 else TEXT)  # fmt: skip
        if pose.block is not None:
            title = pygame.font.Font(None, 28).render(block_title(pose.block), True, TEXT)
            stage.blit(title, (STAGE_W - 24 - title.get_width(), 26))
        if pose.next_block is not None and pose.next_block_beat is not None:
            card = stage.subsurface((20, HEIGHT - 150, 260, 130))
            card.fill(PANEL)
            text(card, f"next: {block_title(pose.next_block)}", (10, 8), 22, ACCENT)
            text(card, f"in {int(np.ceil(pose.next_block_beat - beat))}", (10, 30), 20, DIM)
            draw_dancer(card, static_pose(pose.next_block.keyframes[0]), (190, 30), 15, self.mirrored)
        # grade pop-ups over the camera view
        now = time.monotonic()
        for t, grade, points in self.popups[-1:]:
            fade = 1 - (now - t) / POPUP_S
            label = grade.value.upper() + ("!" if grade in (Grade.PERFECT, Grade.GREAT) else "")
            text(camera, label, (camera.get_width() // 2, 40 - round(10 * fade)), 72, GRADE_COLOUR[grade], centre=True)
            if points:
                text(camera, f"+{points}", (camera.get_width() // 2, 100), 32, TEXT, centre=True)
        warning = session.framing_warning
        if warning is not None:
            draw_hud(camera, ["WARNING: " + warning.issues[0].message], WARN)
        k = session.keeper
        text(panel, f"{k.score:,}", (20, 14), 72, ACCENT)
        text(panel, f"combo {k.combo}   x{k.multiplier:.1f}", (20, 80), 32)
        g = k.grades
        text(panel, f"perfect {g[Grade.PERFECT]}  great {g[Grade.GREAT]}  good {g[Grade.GOOD]}  "
                    f"hold {g[Grade.HOLD]}  miss {g[Grade.MISS]}", (20, 120), 22, DIM)  # fmt: skip
        progress = min(1.0, pos / self.play_end_s) if self.play_end_s else 0.0
        bar = pygame.Rect(20, panel.get_height() - 40, panel.get_width() - 40, 12)
        pygame.draw.rect(panel, DIM, bar, 1)
        pygame.draw.rect(panel, ACCENT, (bar.x, bar.y, round(bar.w * progress), bar.h))
        text(panel, f"{song.title} · Esc end", (20, panel.get_height() - 24), 18, DIM)

    def draw_results(self, stage, camera, panel) -> None:
        r = self.results or {}
        y = text(stage, "Results", (30, 30), 56, ACCENT)
        y = text(stage, f"{r.get('score', 0):,} points", (30, y + 10), 48)
        if r.get("ended_early"):
            y = text(stage, f"ended early: {r.get('keyframes', 0)} moves judged", (30, y), 22, DIM)
        rows = [f"accuracy {r.get('accuracy_pct', 0):.1f}%", f"max combo {r.get('max_combo', 0)}",
                "  ".join(f"{k} {v}" for k, v in r.get("grades", {}).items()),
                f"posture {r.get('posture_pct', 0):.0f}%" + (f"   flow {r['flow_pct']:.0f}%" if r.get("flow_pct") is not None else ""),
                f"latency {r.get('latency_ms', 0):.0f} ms" + ("" if r.get("calibrated") else "  (not body-calibrated)")]  # fmt: skip
        for row in rows:
            y = text(stage, row, (30, y + 6), 26)
        text(stage, "Enter menu   Esc menu", (30, y + 20), 22, DIM)
        text(panel, self.message, (20, 20), 18, DIM)


def run(argv: list[str] | None = None, *, worker: str = DEFAULT_WORKER) -> None:
    parser = argparse.ArgumentParser(description="Para Para dance game")
    parser.add_argument("--latency-ms", type=float, help="override the latency calibration")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", choices=("lite", "full"), default="full")
    parser.add_argument("--auto", action="store_true", help=argparse.SUPPRESS)  # tests: play straight away
    parser.add_argument("--auto-seconds", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--screenshot", help=argparse.SUPPRESS)  # save one play frame (reviews)
    parser.add_argument("--fast", action="store_true", help=argparse.SUPPRESS)  # tests: short calibrations
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
