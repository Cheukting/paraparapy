"""Guided clip recorder: ``uv run parapara-record``.

Walks the player through the recording script in
:mod:`parapara.diagnostics.clips` with the same live camera view as the
debug window, plus a guide panel on the right showing:

- the clip title and instructions,
- a stick figure of the **target pose** (mirrored, like the camera view, so
  the player copies it like a mirror; unscored arms are dimmed). Before a
  move clip starts it **plays the move in a loop** at the clip's tempo as a
  preview; while recording it stays on the beat (``--lookahead`` can make it
  run ahead instead),
- a big countdown / count-in / beat counter, and a progress bar.

Block clips play a metronome (accented on count 1) so the moves are danced
on the beat, both during the preview loop (``--quiet-preview`` to silence it)
and while recording; the scheduled beat times are saved with the clip.

Flow per clip: preview (watch the move) → ``SPACE`` start → countdown /
count-in → recording → review
(``ENTER`` keep, ``R`` redo, ``S`` skip). ``Esc`` quits at any time; kept
clips are already saved.

Output: ``recordings/<YYYYmmdd-HHMMSS>/<clip id>.npz`` (see
:mod:`parapara.pose.recording`) and a ``session.json`` manifest. Only
landmarks are saved, never camera images. ``recordings/`` is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402  (must follow the env var above)

from parapara.diagnostics.clips import CLIP_IDS, CLIPS, Clip, ClipTimeline, Phase  # noqa: E402
from parapara.diagnostics.recorder import environment  # noqa: E402
from parapara.game.debug_view import (  # noqa: E402
    ERROR_COLOUR,
    LEFT_COLOUR,
    RIGHT_COLOUR,
    TEXT_COLOUR,
    LiveFeed,
    draw_hud,
)
from parapara.game.overlay import GuideArm, guide_figure  # noqa: E402
from parapara.ipc.client import DEFAULT_WORKER, CaptureClient  # noqa: E402
from parapara.ipc.messages import CaptureConfig  # noqa: E402
from parapara.pose.landmarks import PoseSample  # noqa: E402
from parapara.pose.recording import Recording  # noqa: E402
from parapara.routine.model import ArmPose  # noqa: E402

PANEL_WIDTH = 340
PANEL_BG = (28, 28, 40)
DIM = (90, 90, 110)
ACCENT = (255, 220, 80)
DEFAULT_OUT = Path("recordings")


# --- metronome ---------------------------------------------------------------


class Metronome:
    """Short synthesized clicks played through ``pygame.mixer``.

    Degrades to silent if no audio device is available (the visual beat
    counter still works). Click timing follows the display loop (60 Hz), so a
    click may be up to ~17 ms late, plus the audio output latency; the beat
    times saved with a clip are the *scheduled* times.
    """

    def __init__(self) -> None:
        self._sounds: dict[bool, Any] = {}
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=256)
            freq, _, channels = pygame.mixer.get_init()
            for accent in (False, True):
                self._sounds[accent] = pygame.sndarray.make_sound(
                    self._click(freq, channels, 1760.0 if accent else 1175.0)
                )
        except pygame.error as err:
            print(f"metronome disabled (no audio): {err}")

    @staticmethod
    def _click(freq: int, channels: int, pitch: float) -> np.ndarray:
        """A 30 ms decaying sine burst, shaped for the mixer's channel count
        (the mixer may open a different count than requested)."""
        t = np.arange(int(freq * 0.03)) / freq
        wave = (np.sin(2 * np.pi * pitch * t) * np.exp(-t * 120) * 20000).astype(np.int16)
        return wave if channels == 1 else np.repeat(wave[:, None], channels, axis=1).copy()

    def click(self, accent: bool) -> None:
        if accent in self._sounds:
            self._sounds[accent].play()


# --- recorder state ----------------------------------------------------------


class Stage(Enum):
    READY = auto()  # waiting for SPACE
    RUNNING = auto()  # countdown / count-in / recording
    REVIEW = auto()  # ENTER keep, R redo, S skip
    FINISHED = auto()  # all clips done


@dataclass
class ClipRun:
    """One attempt at a clip."""

    timeline: ClipTimeline
    started_ns: int  # may lie slightly in the future: see preview_started
    samples: list[PoseSample] = field(default_factory=list)
    next_beat: int = 0
    # For block clips started from the preview loop: the count-in begins on
    # the preview's next beat, so the metronome keeps one unbroken rhythm.
    # Until then the preview keeps playing; this is its start time.
    preview_started: float | None = None
    preview_t: float = -1.0

    def elapsed(self) -> float:
        return (time.monotonic_ns() - self.started_ns) / 1e9

    def window_ns(self) -> tuple[int, int]:
        tl = self.timeline
        return (self.started_ns + int(tl.recording_start * 1e9),
                self.started_ns + int(tl.recording_end * 1e9))  # fmt: skip

    def add(self, samples: list[PoseSample]) -> None:
        start, end = self.window_ns()
        self.samples += [s for s in samples if start <= s.captured_ns < end]

    def recording(self, bpm: float, camera: dict) -> Recording:
        tl = self.timeline
        clip = tl.clip
        beat_times = [self.started_ns + int(t * 1e9) for t, _ in tl.beats()]
        return Recording.from_samples(
            self.samples,
            {
                "clip_id": clip.id,
                "title": clip.title,
                "kind": clip.kind,
                "block_id": clip.block_id,
                "bars": clip.bars if clip.kind == "block" else None,
                "bpm": bpm if clip.kind == "block" else None,
                "count_in_beats": tl.count_in_beats if clip.kind == "block" else None,
                "lookahead_counts": tl.lookahead_counts if clip.kind == "block" else None,
                "beat_times_ns": beat_times,
                "recording_window_ns": list(self.window_ns()),
                "camera": camera,
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            },
        )


# --- drawing -------------------------------------------------------------------


def _text(surface: pygame.Surface, text: str, pos: tuple[int, int], size: int = 22,
          colour=TEXT_COLOUR) -> int:  # fmt: skip
    """Blit text, word-wrapped to the surface width; returns the y below it."""
    font = pygame.font.Font(None, size)
    x, y = pos
    max_width = surface.get_width() - x - 8
    line = ""
    for word in text.split(" "):
        candidate = f"{line} {word}".strip()
        if line and font.size(candidate)[0] > max_width:
            rendered = font.render(line, True, colour)
            surface.blit(rendered, (x, y))
            y += rendered.get_height() + 4
            line = word
        else:
            line = candidate
    rendered = font.render(line, True, colour)
    surface.blit(rendered, (x, y))
    return y + rendered.get_height() + 4


# Display-only clap pose: the clap keyframe has no arm targets (any clap
# counts), but the guide should still *show* hands meeting in front of the chest.
_CLAP_DISPLAY = ArmPose(upper=(0.35, -0.75, 0.6), fore=(-0.75, 0.55, 0.4))


def _draw_guide(surface: pygame.Surface, phase: Phase | None, guide, centre, scale) -> None:
    keyframe = phase.guide if phase is not None and phase.guide is not None else guide
    if keyframe is None:
        return
    right, left = keyframe.right, keyframe.left
    if keyframe.clap and right is None and left is None:
        right, left = _CLAP_DISPLAY, _CLAP_DISPLAY.mirrored()
    fig = guide_figure(right, left, centre, scale)
    pygame.draw.circle(surface, TEXT_COLOUR, fig.head, fig.head_radius, 2)
    pygame.draw.line(surface, TEXT_COLOUR, fig.neck, fig.hip, 3)
    pygame.draw.line(surface, TEXT_COLOUR, fig.left.shoulder, fig.right.shoulder, 3)

    def arm(a: GuideArm, colour: tuple[int, int, int]) -> None:
        c = colour if a.scored else DIM
        for p, q, w in ((a.shoulder, a.elbow, 6), (a.elbow, a.wrist, 5), (a.wrist, a.hand, 3)):
            pygame.draw.line(surface, c, p, q, w)
        pygame.draw.circle(surface, c, a.wrist, 5)

    arm(fig.left, LEFT_COLOUR)
    arm(fig.right, RIGHT_COLOUR)
    if keyframe.clap:
        _text(surface, "CLAP!", (centre[0] + round(0.7 * scale), centre[1] + round(0.5 * scale)), 36, ACCENT)


def draw_panel(surface: pygame.Surface, *, clip: Clip | None, index: int, total: int,
               stage: Stage, phase: Phase | None, review: list[str], bpm: float) -> None:  # fmt: skip
    surface.fill(PANEL_BG)
    w, h = surface.get_size()
    if clip is None:
        y = _text(surface, "All clips done!", (16, 16), 34, ACCENT)
        y = _text(surface, "Recordings saved.", (16, y + 8))
        _text(surface, "[Esc] / [Enter] quit", (16, h - 30), 20, DIM)
        return
    y = _text(surface, f"Clip {index + 1}/{total}", (16, 12), 20, DIM)
    y = _text(surface, clip.title, (16, y), 32, ACCENT)
    for line in clip.instructions:
        y = _text(surface, line, (16, y), 21)
    if clip.kind == "block":
        y = _text(surface, f"{clip.bars} bars at {bpm:.0f} BPM", (16, y), 20, DIM)

    # Room above the shoulders for an arm raised straight up (~2 shoulder widths).
    _draw_guide(surface, phase, clip.guide, (w // 2, y + 135), 58)

    big = ""
    if stage is Stage.READY and phase is not None and phase.name == "preview":
        caption = "preview: watch the move" if clip.kind == "block" else "get into this pose"
        _text(surface, caption, (16, h - 190), 24, ACCENT)
        if phase.count:
            _text(surface, phase.label, (w // 2 - 22, h - 150), 72, ACCENT)
    if stage is Stage.RUNNING and phase is not None:
        big = phase.label
        # Only when the guide runs ahead (--lookahead > 0) does it need a caption.
        if clip.kind == "block" and phase.name == "recording" and phase.guide_count != phase.count:
            _text(surface, f"next: hit on count {phase.guide_count}", (16, h - 190), 24, ACCENT)
        if phase.name == "recording" and clip.kind == "block":
            _text(surface, f"bar {phase.bar}", (16, h - 128), 22, DIM)
    if big:
        colour = (255, 80, 80) if big == "REC" else ACCENT
        _text(surface, big, (w // 2 - 22, h - 150), 72, colour)
    if stage is Stage.RUNNING and phase is not None and phase.name == "recording":
        pygame.draw.rect(surface, DIM, (16, h - 64, w - 32, 10), 1)
        pygame.draw.rect(surface, (255, 80, 80), (16, h - 64, round((w - 32) * phase.progress), 10))

    footer = {
        Stage.READY: "[Space] start   [S] skip   [Esc] quit",
        Stage.RUNNING: "recording ... [Esc] quit",
        Stage.REVIEW: "[Enter] keep   [R] redo   [S] skip",
    }.get(stage, "")
    if stage is Stage.REVIEW:
        yy = h - 150
        for line in review:
            yy = _text(surface, line, (16, yy), 22)
    _text(surface, footer, (16, h - 30), 20, DIM)


# --- app -----------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guided recording of calibration and move clips")
    parser.add_argument("--bpm", type=float, default=120.0, help="metronome tempo for move clips")
    parser.add_argument("--clips", default=",".join(c.id for c in CLIPS),
                        help=f"comma-separated, from: {', '.join(CLIP_IDS)}")  # fmt: skip
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="recordings directory")
    parser.add_argument("--countdown", type=float, default=3.0, help="seconds before a static pose")
    parser.add_argument("--static-seconds", type=float, default=4.0, help="seconds per static pose")
    parser.add_argument("--lookahead", type=float, default=0.0,
                        help="counts the guide runs ahead while recording (default 0: on the beat)")  # fmt: skip
    parser.add_argument("--quiet-preview", action="store_true",
                        help="no metronome clicks while previewing a move")  # fmt: skip
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", choices=("lite", "full"), default="full")
    parser.add_argument("--no-mirror", action="store_true", help="start unmirrored")
    parser.add_argument("--auto", action="store_true", help=argparse.SUPPRESS)  # tests: no keys
    args = parser.parse_args(argv)
    unknown = [c for c in args.clips.split(",") if c and c not in CLIP_IDS]
    if unknown:
        parser.error(f"unknown clip(s): {unknown}")
    return args


def run(argv: list[str] | None = None, *, worker: str = DEFAULT_WORKER) -> Path:
    """Run the guided recorder; returns the session directory.

    ``worker`` and the hidden ``--auto`` flag (start and keep every clip
    without key presses) exist for the automated headless test.
    """
    args = _parse_args(argv)
    clips = [CLIP_IDS[c] for c in args.clips.split(",") if c]
    config = CaptureConfig(camera_index=args.camera, width=args.width, height=args.height,
                           fps=args.fps, pose_model=args.model)  # fmt: skip
    session_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest: dict[str, Any] = {"created": datetime.now().isoformat(timespec="seconds"),
                                "bpm": args.bpm, "config": vars(args) | {"out": str(args.out)},
                                "environment": environment(), "clips": {}}  # fmt: skip

    def save_manifest() -> None:
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "session.json").write_text(json.dumps(manifest, indent=2) + "\n")

    pygame.init()
    screen = pygame.display.set_mode((config.width + PANEL_WIDTH, max(config.height, 560)))
    pygame.display.set_caption("parapara – guided recorder")
    clock = pygame.time.Clock()
    metronome = Metronome()
    feed = LiveFeed(config)
    mirrored = not args.no_mirror

    index = 0
    stage = Stage.READY
    run: ClipRun | None = None
    review: list[str] = []
    running = True
    preview_started: float | None = None  # time.monotonic() when READY began
    preview_t = -1.0  # preview time of the previous frame (for click timing)

    with CaptureClient(config, worker=worker) as capture:
        while running:
            clip = clips[index] if index < len(clips) else None
            if clip is None:
                stage = Stage.FINISHED
            keys: set[int] = set()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    keys.add(event.key)
            if pygame.K_ESCAPE in keys or pygame.K_q in keys:
                running = False
            if pygame.K_m in keys:
                mirrored = not mirrored

            samples = feed.update(capture)
            failure = feed.failure(capture)
            ready = feed.ready_at is not None and failure is None

            # --- state machine ---
            if stage is Stage.READY and ready and (pygame.K_SPACE in keys or args.auto):
                assert clip is not None
                timeline = ClipTimeline(clip, bpm=args.bpm, countdown_s=args.countdown,
                                        static_seconds=args.static_seconds,
                                        lookahead_counts=args.lookahead)  # fmt: skip
                start_ns, lead_in = time.monotonic_ns(), None
                if clip.kind == "block" and preview_started is not None:
                    # Start the count-in on the preview's next beat.
                    now = time.monotonic()
                    next_beat_t = timeline.next_preview_beat(now - preview_started)
                    start_ns += int((preview_started + next_beat_t - now) * 1e9)
                    lead_in = preview_started
                run = ClipRun(timeline, start_ns, preview_started=lead_in, preview_t=preview_t)
                stage = Stage.RUNNING
            elif stage is Stage.READY and pygame.K_s in keys:
                index += 1
            elif stage is Stage.REVIEW and run is not None:
                if pygame.K_RETURN in keys or pygame.K_KP_ENTER in keys or args.auto:
                    recording = run.recording(args.bpm, feed.camera_details)
                    path = session_dir / f"{run.timeline.clip.id}.npz"
                    recording.save(path)
                    manifest["clips"][run.timeline.clip.id] = {
                        "file": path.name, "samples": len(recording),
                        "person_found_pct": round(recording.person_found_pct, 1),
                        "duration_s": round(recording.duration_s, 2),
                    }  # fmt: skip
                    save_manifest()
                    print(f"saved {path} ({len(recording)} samples)")
                    index, stage, run = index + 1, Stage.READY, None
                elif pygame.K_r in keys:
                    stage, run = Stage.READY, None
                elif pygame.K_s in keys:
                    index, stage, run = index + 1, Stage.READY, None
            elif stage is Stage.FINISHED and (pygame.K_RETURN in keys or args.auto):
                running = False

            phase: Phase | None = None
            if stage is Stage.READY and clip is not None:
                # Preview loop: restarts whenever a clip becomes ready again.
                if preview_started is None:
                    preview_started, preview_t = time.monotonic(), -1.0
                preview = ClipTimeline(clip, bpm=args.bpm)
                now_t = time.monotonic() - preview_started
                phase = preview.preview_at(now_t)
                if not args.quiet_preview:
                    for accent in preview.preview_clicks(preview_t, now_t):
                        metronome.click(accent)
                preview_t = now_t
            else:
                preview_started = None
            if stage is Stage.RUNNING and run is not None and run.elapsed() < 0:
                # Waiting for the next preview beat: keep the preview going.
                assert run.preview_started is not None
                now_t = time.monotonic() - run.preview_started
                phase = run.timeline.preview_at(now_t)
                if not args.quiet_preview:
                    for accent in run.timeline.preview_clicks(run.preview_t, now_t):
                        metronome.click(accent)
                run.preview_t = now_t
            elif stage is Stage.RUNNING and run is not None:
                run.add(samples)
                t = run.elapsed()
                beats = run.timeline.beats()
                while run.next_beat < len(beats) and beats[run.next_beat][0] <= t:
                    metronome.click(beats[run.next_beat][1])
                    run.next_beat += 1
                phase = run.timeline.phase_at(t)
                if phase.name == "done":
                    rec = run.recording(args.bpm, feed.camera_details)
                    review = [f"{len(rec)} samples, {rec.duration_s:.1f} s",
                              f"person found {rec.person_found_pct:.0f}%"]  # fmt: skip
                    stage = Stage.REVIEW

            # --- draw ---
            feed.draw(screen, mirrored)
            if failure:
                draw_hud(screen, ["CAPTURE ERROR", failure], ERROR_COLOUR)
            elif not ready:
                draw_hud(screen, [f"{feed.status.kind}: {feed.status.message}"])
            else:
                draw_hud(screen, [feed.camera_info, *feed.stats.hud_lines()[:2]])
            panel = screen.subsurface((config.width, 0, PANEL_WIDTH, screen.get_height()))
            draw_panel(panel, clip=clip, index=index, total=len(clips), stage=stage,
                       phase=phase, review=review, bpm=args.bpm)  # fmt: skip
            pygame.display.flip()
            feed.after_flip()
            clock.tick(60)

    pygame.quit()
    if manifest["clips"]:
        save_manifest()
        print(f"session saved to {session_dir}")
    else:
        print("no clips kept; nothing saved")
    return session_dir


def main(argv: list[str] | None = None) -> None:
    """Console entry point. Returns ``None`` on purpose: console scripts call
    ``sys.exit(main())``, which would print any returned value and exit with
    status 1. Use :func:`run` to get the result programmatically."""
    run(argv)


if __name__ == "__main__":
    main()
