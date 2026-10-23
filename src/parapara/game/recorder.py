"""Guided clip recorder: ``uv run parapara-record``.

Walks the player through the recording script in
:mod:`parapara.diagnostics.clips`. Layout like the game (fullscreen, large
text from :mod:`parapara.game.ui`, ``+`` / ``-`` text size, ``F`` window):

- left, where the player looks: the clip title and instructions, a big stick
  figure of the **target pose** (mirrored, like the camera view, so the player
  copies it like a mirror; unscored arms are dimmed), one status sentence
  ("Watch the move", "Step back", "Recording - dance!", "Saved", "Again:
  ...") and the big countdown / count-in / beat count. Before a move clip
  starts the figure **plays the move in a loop** at the clip's tempo; while
  recording it stays on the beat (``--lookahead`` can make it run ahead);
- right: the live camera view with the framing box, and an info panel.

Block clips play a metronome (accented on count 1) so the moves are danced
on the beat, both during the preview loop (``--quiet-preview`` to silence it)
and while recording; the scheduled beat times are saved with the clip. The
count-in is one full 8-count bar (8, 7, ... 1) starting on count 1 of the
preview's next bar, so preview, count-in and recording keep one rhythm.

**Hands-free** (default; the player stands back and cannot reach the
keyboard): each clip waits until the position check passes (hips in the
picture, room to stretch, centred, facing the camera:
:mod:`parapara.pose.framing`) and the preview has run (one bar of the move,
or 3 s of a static pose), then starts by itself. After recording, a quality
check (person seen >= 90% of the time, hips and shoulders in the picture for
>= 90% of it) keeps the clip and moves on, or says why and redoes it, up to
:data:`MAX_ATTEMPTS` times (the last attempt is then kept, marked
``quality_ok: false``). ``S`` skips a clip, ``R`` redoes the last one, ``Esc``
quits; kept clips are already saved. ``--manual`` restores the key-driven
flow (``SPACE`` start, ``ENTER`` keep, ``R`` redo, ``S`` skip).

Output: ``recordings/<YYYYmmdd-HHMMSS>/<clip id>.npz`` (see
:mod:`parapara.pose.recording`) and a ``session.json`` manifest (with each
clip's attempts and quality). Only landmarks are saved, never camera images.
``recordings/`` is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from parapara.game.branding import NAME, set_window_icon  # noqa: E402
from parapara.game.debug_view import ERROR_COLOUR, LEFT_COLOUR, RIGHT_COLOUR, TEXT_COLOUR, LiveFeed  # noqa: E402
from parapara.game.overlay import GuideArm, guide_figure, to_screen  # noqa: E402
from parapara.ipc.client import DEFAULT_WORKER, CaptureClient  # noqa: E402
from parapara.ipc.messages import CaptureConfig  # noqa: E402
from parapara.pose.framing import FramingMonitor, FramingResult  # noqa: E402
from parapara.pose.landmarks import PoseSample  # noqa: E402
from parapara.pose.recording import Recording  # noqa: E402
from parapara.game.ui import (  # noqa: E402
    UI,
    adjust_scale,
    display_flags,
    footer,
    layout_issues,
    load_scale,
    text,
    text_height,
)
from parapara.routine.model import COUNTS_PER_BAR, ArmPose  # noqa: E402

PANEL_BG = (28, 28, 40)
DIM = (90, 90, 110)
ACCENT = (255, 220, 80)
DEFAULT_OUT = Path("recordings")
OK_COLOUR = (90, 220, 120)
WARN_COLOUR = (255, 150, 60)


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
    READY = auto()  # preview; starts by itself (hands-free) or on SPACE (--manual)
    RUNNING = auto()  # countdown / count-in / recording
    REVIEW = auto()  # result shown; hands-free: kept or redone automatically
    FINISHED = auto()  # all clips done


# Hands-free flow (fourth recording request: the player stands back and cannot
# reach the keyboard). Every step advances by itself; keys remain shortcuts.
PREVIEW_BARS = 1  # a block clip's move is previewed for at least this many 8-count bars
STATIC_PREVIEW_S = 3.0  # a static pose is shown at least this long before its countdown
RESULT_S = 2.5  # how long a result ("Saved", "Again: ...") stays up before moving on
MAX_ATTEMPTS = 3  # a clip failing the quality check is redone up to this many times
MIN_PERSON_PCT = 90.0  # quality: % of samples with a person found
MAX_OUT_OF_FRAME = 0.1  # quality: share of recording frames with hips/shoulders/person missing


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
    # Framing during the recording window (quality check): display frames
    # seen, and how many had the person / shoulders / hips out of the picture.
    framed_frames: int = 0
    out_of_frame: int = 0

    def elapsed(self) -> float:
        return (time.monotonic_ns() - self.started_ns) / 1e9

    def window_ns(self) -> tuple[int, int]:
        tl = self.timeline
        return (self.started_ns + int(tl.recording_start * 1e9),
                self.started_ns + int(tl.recording_end * 1e9))  # fmt: skip

    def add(self, samples: list[PoseSample]) -> None:
        start, end = self.window_ns()
        self.samples += [s for s in samples if start <= s.captured_ns < end]

    def note_framing(self, framing: FramingResult | None) -> None:
        """Count one display frame of the recording window for the quality check."""
        self.framed_frames += 1
        if framing is None or framing.codes & {"no_person", "no_shoulders", "no_hips"}:
            self.out_of_frame += 1

    def quality_issue(self, recording: Recording) -> str | None:
        """Why this attempt should be redone, or ``None`` if it is good."""
        if len(recording) == 0:
            return "no camera samples"
        if recording.person_found_pct < MIN_PERSON_PCT:
            return f"you were seen only {recording.person_found_pct:.0f}% of the time"
        if self.framed_frames and self.out_of_frame / self.framed_frames > MAX_OUT_OF_FRAME:
            return "hips or shoulders left the picture"
        return None

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
# Layout like the game (fixed 1240x720, fullscreen-scaled): the guide on the
# left where the player looks, the camera top right, a status panel below it.
# Text uses parapara.game.ui (sizes x UI.scale, readable from the dancing spot).

WIDTH, HEIGHT = 1240, 720
STAGE_W = 600

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
    w = max(3, round(scale * 0.1))  # line width grows with the figure
    pygame.draw.circle(surface, TEXT_COLOUR, fig.head, fig.head_radius, w)
    pygame.draw.line(surface, TEXT_COLOUR, fig.neck, fig.hip, w)
    pygame.draw.line(surface, TEXT_COLOUR, fig.left.shoulder, fig.right.shoulder, w)

    def arm(a: GuideArm, colour: tuple[int, int, int]) -> None:
        c = colour if a.scored else DIM
        for p, q, lw in ((a.shoulder, a.elbow, w + 2), (a.elbow, a.wrist, w + 1), (a.wrist, a.hand, w)):
            pygame.draw.line(surface, c, p, q, lw)
        pygame.draw.circle(surface, c, a.wrist, w + 2)

    arm(fig.left, LEFT_COLOUR)
    arm(fig.right, RIGHT_COLOUR)
    if keyframe.clap:
        text(surface, "CLAP!", (centre[0] + round(0.8 * scale), centre[1] + round(0.4 * scale)), 30, ACCENT)


def draw_framing_box(surface: pygame.Surface, framing: FramingResult | None, mirrored: bool) -> None:
    """Outline the picture area the player needs (arm span, head room, hips)."""
    if framing is None or framing.needed_box is None:
        return
    x0, y0, x1, y1 = framing.needed_box
    corners = to_screen([(x0, y0), (x1, y1)], surface.get_size(), mirrored)
    (ax, ay), (bx, by) = corners
    rect = pygame.Rect(min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay))
    pygame.draw.rect(surface, OK_COLOUR if framing.ok else WARN_COLOUR, rect, 2)


def status_line(*, clip: Clip | None, stage: Stage, phase: Phase | None, framing: FramingResult | None,
                framed: bool, result: tuple[str, tuple[int, int, int]] | None,
                manual: bool) -> tuple[str, tuple[int, int, int]]:  # fmt: skip
    """The one sentence telling the player what is happening / what to do."""
    if clip is None:
        return ("All done - thank you!", OK_COLOUR)
    if stage is Stage.REVIEW and result is not None:
        return result
    if stage is Stage.READY:
        if not framed:
            issue = framing.issues[0].message if framing is not None and framing.issues else "Step into the picture"
            return (issue, WARN_COLOUR)
        if manual:
            return ("Press Space to start", ACCENT)
        return ("Watch the move" if clip.kind == "block" else "Get into this pose", ACCENT)
    if stage is Stage.RUNNING and phase is not None:
        if phase.name == "preview":
            return ("Get ready ...", ACCENT)
        if phase.name in ("countdown", "count_in"):
            return ("Get ready ...", ACCENT)
        if framing is not None and framing.codes & {"no_person", "no_shoulders", "no_hips"}:
            return (framing.issues[0].message, WARN_COLOUR)
        return ("Recording - dance!" if clip.kind == "block" else "Recording - hold still", (255, 90, 90))
    return ("", TEXT_COLOUR)


def draw_stage(surface: pygame.Surface, *, clip: Clip | None, index: int, total: int, stage: Stage,
               phase: Phase | None, status: tuple[str, tuple[int, int, int]]) -> None:  # fmt: skip
    """The guide side: clip, instructions, a big guide figure, the status
    sentence and the big count."""
    surface.fill(PANEL_BG)
    if clip is None:
        text(surface, "All clips done!", (24, 30), 40, ACCENT)
        text(surface, status[0], (24, 140), 26, status[1])
        return
    y = text(surface, f"Clip {index + 1} of {total}", (24, 10), 16, DIM)
    y = text(surface, clip.title, (24, y), 34, ACCENT)
    y = text(surface, " ".join(clip.instructions), (24, y), 18)
    # Bottom up: the big count / label, then the status sentence above it.
    big = ""
    if phase is not None and (stage is Stage.RUNNING or phase.count):
        big = phase.label
    label_h = text_height(surface, "8", 64, STAGE_W // 2)
    label_top = HEIGHT - 10 - label_h
    if big:
        colour = (255, 80, 80) if big == "REC" else ACCENT
        text(surface, big, (STAGE_W // 2, label_top), 64, colour, centre=True)
    status_top = label_top - text_height(surface, status[0], 26, 24) - 4
    text(surface, status[0], (24, status_top), 26, status[1])
    if stage is Stage.RUNNING and phase is not None and phase.name == "recording":
        pygame.draw.rect(surface, DIM, (24, status_top - 18, STAGE_W - 48, 10), 1)
        pygame.draw.rect(surface, (255, 80, 80), (24, status_top - 18, round((STAGE_W - 48) * phase.progress), 10))
    # The guide figure fills the room between the instructions and the status:
    # from a raised hand (~2.3 shoulder widths above the shoulders) to the hips
    # and hanging hands (~1.6 below).
    room = status_top - 24 - y
    scale = max(10.0, min(110.0, room / 4.0))
    _draw_guide(surface, phase, clip.guide, (STAGE_W // 2, round(y + 2.3 * scale)), scale)


def draw_info(surface: pygame.Surface, *, clip: Clip | None, bpm: float, manual: bool, attempt: int) -> None:
    """The panel under the camera: tempo, attempt, and the few keys."""
    surface.fill(PANEL_BG)
    y = 12
    if clip is not None and clip.kind == "block":
        y = text(surface, f"{clip.bars} bars at {bpm:.0f} BPM", (20, y), 18)
    if attempt > 1:
        y = text(surface, f"attempt {attempt} of {MAX_ATTEMPTS}", (20, y), 18, WARN_COLOUR)
    keys = ("Space start   Enter keep   R redo   S skip   Esc quit" if manual
            else "Hands-free: nothing to press.   S skip   Esc quit")  # fmt: skip
    footer(surface, [(keys, 14, DIM), ("+/- text size   F fullscreen   M mirror", 14, DIM)], x=20)


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
    parser.add_argument("--no-framing-check", action="store_true",
                        help="allow starting clips without a good position")  # fmt: skip
    parser.add_argument("--manual", action="store_true",
                        help="keys instead of hands-free: Space start, Enter keep, R redo, S skip")  # fmt: skip
    parser.add_argument("--window", action="store_true", help="run in a window instead of fullscreen")
    parser.add_argument("--text-scale", type=float, help="text size (default: saved, or 2.0; +/- while recording)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", choices=("lite", "full"), default="full")
    parser.add_argument("--no-mirror", action="store_true", help="start unmirrored")
    parser.add_argument("--auto", action="store_true", help=argparse.SUPPRESS)  # tests: start at once, keep all
    parser.add_argument("--check-layout", action="store_true", help=argparse.SUPPRESS)  # tests
    parser.add_argument("--shots", help=argparse.SUPPRESS)  # reviews: screenshot folder
    args = parser.parse_args(argv)
    unknown = [c for c in args.clips.split(",") if c and c not in CLIP_IDS]
    if unknown:
        parser.error(f"unknown clip(s): {unknown}")
    return args


def run(argv: list[str] | None = None, *, worker: str = DEFAULT_WORKER) -> Path:
    """Run the guided recorder; returns the session directory.

    Hands-free by default: each clip starts once the player is in position
    and has watched the preview, is checked after recording (person seen,
    hips and shoulders in the picture), then kept and followed by the next
    clip, or redone (up to MAX_ATTEMPTS; the last attempt is kept and marked
    ``quality_ok: false``). ``--manual`` restores the key-driven flow.

    ``worker`` and the hidden ``--auto`` flag (start at once and keep every
    clip) exist for the automated headless test.
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
    set_window_icon()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), display_flags(args.window))
    pygame.display.set_caption(f"{NAME} – guided recorder")
    load_scale(args.text_scale)
    clock = pygame.time.Clock()
    metronome = Metronome()
    feed = LiveFeed(config)
    framing = FramingMonitor()
    mirrored = not args.no_mirror
    manual = args.manual

    index = 0
    stage = Stage.READY
    run: ClipRun | None = None
    attempt = 1
    result: tuple[str, tuple[int, int, int]] | None = None
    result_at = 0.0  # time.monotonic() when the result was shown
    pending: tuple[Recording, bool] | None = None  # REVIEW: the attempt and whether to keep it
    running = True
    preview_started: float | None = None  # time.monotonic() when READY began
    preview_t = -1.0  # preview time of the previous frame (for click timing)
    reported: set[str] = set()
    frames = 0

    def keep(recording: Recording, quality_ok: bool) -> None:
        path = session_dir / f"{recording.metadata['clip_id']}.npz"
        recording.save(path)
        manifest["clips"][recording.metadata["clip_id"]] = {
            "file": path.name, "samples": len(recording),
            "person_found_pct": round(recording.person_found_pct, 1),
            "duration_s": round(recording.duration_s, 2), "attempts": attempt, "quality_ok": quality_ok,
        }  # fmt: skip
        save_manifest()
        print(f"saved {path} ({len(recording)} samples)")

    with CaptureClient(config, worker=worker) as capture:
        while running:
            clip = clips[index] if index < len(clips) else None
            if clip is None and stage is not Stage.FINISHED:
                stage, result_at = Stage.FINISHED, time.monotonic()
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
            if pygame.K_f in keys:
                pygame.display.toggle_fullscreen()
            adjust_scale(keys)

            samples = feed.update(capture)
            for sample in samples:
                framing.update(sample)
            failure = feed.failure(capture)
            ready = feed.ready_at is not None and failure is None
            # Hands-free, every clip waits for a good position (the hips were
            # out of the picture in a whole earlier session); --manual keeps
            # the old rule (only static clips wait).
            needs_framing = clip is not None and (clip.requires_framing or not manual)
            framed = clip is None or not needs_framing or args.no_framing_check or framing.stable
            now = time.monotonic()
            # A block clip starts half a beat before the end of its preview
            # bar(s), so the count-in begins right at the next bar (not a bar later).
            beat_s = 60.0 / args.bpm
            previewed = preview_started is not None and clip is not None and (now - preview_started) >= (
                (PREVIEW_BARS * COUNTS_PER_BAR - 0.5) * beat_s if clip.kind == "block" else STATIC_PREVIEW_S)
            start = pygame.K_SPACE in keys or args.auto or (not manual and previewed)

            # --- state machine ---
            if stage is Stage.READY and ready and framed and start:
                assert clip is not None
                timeline = ClipTimeline(clip, bpm=args.bpm, countdown_s=args.countdown,
                                        static_seconds=args.static_seconds,
                                        lookahead_counts=args.lookahead)  # fmt: skip
                start_ns, lead_in = time.monotonic_ns(), None
                if clip.kind == "block" and preview_started is not None:
                    # Start the count-in on count 1 of the preview's next bar.
                    next_bar_t = timeline.next_preview_bar(now - preview_started)
                    start_ns += int((preview_started + next_bar_t - now) * 1e9)
                    lead_in = preview_started
                run = ClipRun(timeline, start_ns, preview_started=lead_in, preview_t=preview_t)
                stage = Stage.RUNNING
            elif stage is Stage.READY and pygame.K_s in keys:
                index, attempt = index + 1, 1
            elif stage is Stage.REVIEW and pending is not None:
                recording, good = pending
                if manual:
                    if pygame.K_RETURN in keys or pygame.K_KP_ENTER in keys or args.auto:
                        keep(recording, good)
                        index, attempt, stage, run, pending = index + 1, 1, Stage.READY, None, None
                    elif pygame.K_r in keys:
                        attempt, stage, run, pending = attempt + 1, Stage.READY, None, None
                    elif pygame.K_s in keys:
                        index, attempt, stage, run, pending = index + 1, 1, Stage.READY, None, None
                elif pygame.K_s in keys:
                    index, attempt, stage, run, pending = index + 1, 1, Stage.READY, None, None
                elif pygame.K_r in keys or (now - result_at >= RESULT_S or args.auto):
                    redo = pygame.K_r in keys or (not good and not args.auto and attempt < MAX_ATTEMPTS)
                    if redo:
                        attempt, stage, run, pending = attempt + 1, Stage.READY, None, None
                    else:
                        keep(recording, good)
                        index, attempt, stage, run, pending = index + 1, 1, Stage.READY, None, None
            elif stage is Stage.FINISHED and (pygame.K_RETURN in keys or args.auto
                                              or (not manual and now - result_at >= 5.0)):  # fmt: skip
                running = False

            phase: Phase | None = None
            if stage is Stage.READY and clip is not None:
                # Preview loop: restarts whenever a clip becomes ready again.
                if preview_started is None:
                    preview_started, preview_t = now, -1.0
                preview = ClipTimeline(clip, bpm=args.bpm)
                now_t = now - preview_started
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
                now_t = now - run.preview_started
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
                if phase.name == "recording":
                    run.note_framing(framing.latest)
                if phase.name == "done":
                    rec = run.recording(args.bpm, feed.camera_details)
                    issue = run.quality_issue(rec)
                    if issue is None:
                        result = ("Saved - next clip coming" if not manual else "Good - Enter keeps it", OK_COLOUR)
                    elif attempt < MAX_ATTEMPTS and not manual:
                        result = (f"Again: {issue}", WARN_COLOUR)
                    else:
                        result = (f"Kept, but {issue}", WARN_COLOUR)
                    pending, result_at, stage = (rec, issue is None), now, Stage.REVIEW

            # --- draw ---
            UI.drawn = [] if args.check_layout else None
            screen.fill(PANEL_BG)
            camera = screen.subsurface((STAGE_W, 0, config.width, config.height))
            feed.draw(camera, mirrored)
            if failure:
                text(camera, "CAMERA PROBLEM", (20, 20), 30, ERROR_COLOUR)
            elif not ready:
                text(camera, "starting camera ...", (camera.get_width() // 2, camera.get_height() // 3), 24, DIM,
                     centre=True)  # fmt: skip
            elif stage is Stage.READY:
                draw_framing_box(camera, framing.latest, mirrored)
            status = status_line(clip=clip, stage=stage, phase=phase, framing=framing.latest, framed=framed,
                                 result=result, manual=manual)  # fmt: skip
            draw_stage(screen.subsurface((0, 0, STAGE_W, HEIGHT)), clip=clip, index=index, total=len(clips),
                       stage=stage, phase=phase, status=status)  # fmt: skip
            info = screen.subsurface((STAGE_W, config.height, WIDTH - STAGE_W, HEIGHT - config.height))
            draw_info(info, clip=clip, bpm=args.bpm, manual=manual, attempt=attempt)
            if UI.drawn is not None:
                for issue in layout_issues(UI.drawn):
                    if issue not in reported:
                        reported.add(issue)
                        print(f"LAYOUT ({stage.name.lower()}, text size {UI.scale}): {issue}", file=sys.stderr)
            pygame.display.flip()
            feed.after_flip()
            clock.tick(60)
            frames += 1
            if args.shots and frames % 30 == 0:  # reviews: periodic screenshots
                pygame.image.save(screen, f"{args.shots}/{frames:05d}-{stage.name.lower()}.png")

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
