"""Live capture debug window (roadmap Step 2).

Run with ``uv run parapara-capture-debug`` (``--help`` for options).

Shows the camera picture with the tracked upper-body skeleton, and
a HUD with tracking rate, per-model inference time and latency. It is the
tool for the real-dancer test run: checking that tracking keeps up with fast
moves, and that left/right are what we think they are.

Colour code: **cyan = dancer's left**, **orange = dancer's right**, so a
left/right mix-up is obvious at a glance. Keys: ``M`` toggles the mirrored
(selfie) view, ``Esc``/``Q`` quits.

On exit, a summary of the whole session (rates, timings, latency, detection
rates) is written to ``logs/capture-debug-*.json`` (``--no-log`` to skip).
For a side-by-side comparison of configurations use
``uv run parapara-capture-bench`` instead.

Architecture: this process imports ``pygame`` only. The camera and MediaPipe
run in the capture process started by
:class:`~parapara.ipc.client.CaptureClient`; frames arrive through shared
memory and landmarks through a queue.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402  (must follow the env var above)

from parapara.diagnostics.live_stats import CaptureStats  # noqa: E402
from parapara.diagnostics.recorder import DEFAULT_LOG_DIR, RunRecorder, write_log  # noqa: E402
from parapara.game.overlay import to_screen, visible  # noqa: E402
from parapara.game.theme import (  # noqa: E402
    BG,
    CENTRE_COLOUR,
    ERROR,
    LEFT_COLOUR,
    RIGHT_COLOUR,
    TEXT_COLOUR,
)
from parapara.ipc.client import DEFAULT_WORKER, CaptureClient  # noqa: E402
from parapara.ipc.messages import CaptureConfig, CaptureStatus, StatusKind  # noqa: E402
from parapara.pose.landmarks import (  # noqa: E402
    UPPER_BODY_CONNECTIONS,
    PoseLandmark,
    PoseSample,
)

ERROR_COLOUR = ERROR  # public name kept for the capture-error colour

_LEFT_POINTS = {p for p in PoseLandmark if p.name.startswith("LEFT_")}
_RIGHT_POINTS = {p for p in PoseLandmark if p.name.startswith("RIGHT_")}


def _side_colour(a: int, b: int) -> tuple[int, int, int]:
    if a in _LEFT_POINTS and b in _LEFT_POINTS:
        return LEFT_COLOUR
    if a in _RIGHT_POINTS and b in _RIGHT_POINTS:
        return RIGHT_COLOUR
    return CENTRE_COLOUR


def draw_pose(surface: pygame.Surface, sample: PoseSample, mirrored: bool) -> None:
    """Draw the upper-body skeleton (including the pose model's hand points)."""
    size = surface.get_size()
    if sample.pose_image is not None:
        pose = sample.pose_image
        points = to_screen(pose, size, mirrored)
        for a, b in UPPER_BODY_CONNECTIONS:
            if visible(pose, a) and visible(pose, b):
                pygame.draw.line(surface, _side_colour(a, b), points[a], points[b], 4)
        font = pygame.font.Font(None, 28)
        for index, label, colour in (
            (PoseLandmark.LEFT_WRIST, "L", LEFT_COLOUR),
            (PoseLandmark.RIGHT_WRIST, "R", RIGHT_COLOUR),
        ):
            if visible(pose, index):
                x, y = points[index]
                pygame.draw.circle(surface, colour, (x, y), 7)
                surface.blit(font.render(label, True, TEXT_COLOUR), (x + 9, y - 9))


def draw_hud(surface: pygame.Surface, lines: list[str], colour=TEXT_COLOUR) -> None:
    font = pygame.font.Font(None, 22)
    rendered = [font.render(line, True, colour) for line in lines]
    if not rendered:
        return
    width = max(r.get_width() for r in rendered) + 16
    height = sum(r.get_height() + 2 for r in rendered) + 12
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    surface.blit(panel, (8, 8))
    y = 14
    for r in rendered:
        surface.blit(r, (16, y))
        y += r.get_height() + 2


def _parse_args(argv: list[str] | None) -> tuple[CaptureConfig, bool, Path | None]:
    parser = argparse.ArgumentParser(description="Para Para capture debug view")
    parser.add_argument(
        "--camera", type=int, default=0, help="camera index (default 0)"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="requested camera fps")
    parser.add_argument(
        "--model", choices=("lite", "full"), default="full", help="pose model"
    )
    parser.add_argument("--no-mirror", action="store_true", help="start unmirrored")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="where the session summary is written")  # fmt: skip
    parser.add_argument(
        "--no-log", action="store_true", help="don't write a session summary"
    )
    args = parser.parse_args(argv)
    config = CaptureConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        pose_model=args.model,
    )
    return config, not args.no_mirror, None if args.no_log else args.log_dir


@dataclasses.dataclass
class SessionResult:
    """Outcome of one :func:`run_session`."""

    summary: dict
    camera: dict
    last_status: CaptureStatus
    quit_requested: bool  # the user closed the window or pressed Esc/Q
    mirrored: bool  # final mirror setting, carried over to the next session

    @property
    def error(self) -> str | None:
        if self.last_status.kind is StatusKind.ERROR:
            return self.last_status.message or "capture error"
        return self.summary.get("error")


def poll_quit_and_mirror(mirrored: bool) -> tuple[bool, bool]:
    """Handle window events; returns ``(quit_requested, mirrored)``."""
    quit_requested = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
        ):
            quit_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
            mirrored = not mirrored
    return quit_requested, mirrored


class LiveFeed:
    """Camera picture + skeleton + HUD statistics for one capture process.

    Shared by the debug view, the windowed benchmark and the guided recorder.
    Call :meth:`update` once per display frame (it drains the capture queue
    and fetches the newest frame), :meth:`draw` to paint the camera area, and
    :meth:`after_flip` right after ``pygame.display.flip()`` to measure
    camera → screen latency.
    """

    def __init__(self, config: CaptureConfig, warmup_s: float = 2.0) -> None:
        self.config = config
        self.recorder = RunRecorder(warmup_s=warmup_s)
        self.stats = CaptureStats.new()
        self.status = CaptureStatus(StatusKind.STARTING, "starting capture process")
        self.camera_details: dict = {}
        self.camera_info = ""
        self.ready_at: float | None = None  # time.monotonic() of READY
        self._recent: deque[PoseSample] = deque(maxlen=16)  # to match samples to frames
        self._frame_surface: pygame.Surface | None = None
        self._shown_index = -1
        self._shown_captured_ns = 0
        self._new_frame = False

    def update(self, capture: CaptureClient) -> list[PoseSample]:
        """Process new messages and frames; returns the new samples."""
        samples = []
        for message in capture.poll():
            if isinstance(message, PoseSample):
                self.stats.add_sample(message)
                self.recorder.add(message)
                self._recent.append(message)
                samples.append(message)
            else:
                self.status = message
                if message.kind is StatusKind.READY:
                    self.ready_at = time.monotonic()
                    d = self.camera_details = dict(message.details)
                    self.camera_info = (
                        f"camera {d.get('camera_width')}x{d.get('camera_height')} "
                        f"@ {float(d.get('camera_fps', 0)):.0f} fps"
                    )
                elif message.kind is StatusKind.ERROR:
                    print(message.details.get("traceback", message.message))
        snapshot = capture.latest_frame(newer_than=self._shown_index)
        self._new_frame = snapshot is not None
        if snapshot is not None:
            self._frame_surface = pygame.image.frombuffer(
                snapshot.frame.tobytes(), (self.config.width, self.config.height), "RGB"
            )
            self._shown_index, self._shown_captured_ns = (
                snapshot.frame_index,
                snapshot.captured_ns,
            )
        return samples

    def draw(self, screen: pygame.Surface, mirrored: bool) -> None:
        """Paint the camera picture and skeleton into the top-left
        ``width × height`` area of ``screen``."""
        area = screen.subsurface((0, 0, self.config.width, self.config.height))
        area.fill(BG)
        if self._frame_surface is None:
            return
        frame = self._frame_surface
        area.blit(
            pygame.transform.flip(frame, True, False) if mirrored else frame, (0, 0)
        )
        # The sample for exactly this frame, if we still have it.
        sample = next(
            (s for s in reversed(self._recent) if s.frame_index == self._shown_index),
            None,
        )
        if sample is not None:
            draw_pose(area, sample, mirrored)

    def after_flip(self) -> None:
        if self._new_frame and self._shown_captured_ns:
            latency_ms = (time.monotonic_ns() - self._shown_captured_ns) / 1e6
            self.stats.display_latency_ms.add(latency_ms)
            self.recorder.add_display_latency(latency_ms)

    def failure(self, capture: CaptureClient) -> str | None:
        """A readable message if capture failed, else ``None``."""
        if self.status.kind is StatusKind.ERROR:
            return self.status.message or "capture error"
        if not capture.is_alive and self.status.kind is not StatusKind.STOPPED:
            return f"capture process exited (code {capture.exitcode})"
        return None

    def hud_lines(self, mirrored: bool) -> list[str]:
        return [f"{self.status.kind}: {self.status.message}", self.camera_info,
                *self.stats.hud_lines(),
                f"mirror {'on' if mirrored else 'off'}   [M] mirror  [Esc] quit"]  # fmt: skip


def run_session(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    config: CaptureConfig,
    *,
    worker: str = DEFAULT_WORKER,
    mirrored: bool = True,
    duration_s: float | None = None,
    warmup_s: float = 2.0,
    banner: tuple[str, ...] = (),
    stop_on_error: bool = False,
    max_frames: int | None = None,
) -> SessionResult:
    """Run one capture process and show its live view until it ends.

    Shared by the debug view (runs until the user quits) and the windowed
    benchmark (runs each configuration for a fixed time).

    Args:
        duration_s: if set, end the session this many seconds after the
            warm-up that follows ``READY``; ``None`` runs until quit.
        warmup_s: seconds excluded from the recorded summary.
        banner: extra HUD lines shown first (e.g. the benchmark step).
        stop_on_error: end the session on a capture error instead of
            showing it until the user quits.
        max_frames: end after this many display frames once the capture is
            ``READY`` (automated tests). A worker that never becomes ready is
            stopped after five times as many frames, so a broken spawn cannot
            hang an automated run during process start-up.
    """
    feed = LiveFeed(config, warmup_s=warmup_s)
    quit_requested = False
    with CaptureClient(config, worker=worker) as capture:
        drawn = 0
        ready_frames = 0
        startup_grace = max_frames * 5 if max_frames is not None else None
        while True:
            quit_requested, mirrored = poll_quit_and_mirror(mirrored)
            if quit_requested:
                break
            feed.update(capture)
            feed.draw(screen, mirrored)

            # Timed sessions: phase line and end condition.
            phase = ""
            if duration_s is not None:
                if feed.ready_at is None:
                    phase = "starting ..."
                else:
                    elapsed = time.monotonic() - feed.ready_at
                    if elapsed < warmup_s:
                        phase = (
                            f"warming up ... keep dancing ({warmup_s - elapsed:.0f} s)"
                        )
                    else:
                        phase = f"recording: keep dancing! {max(0.0, warmup_s + duration_s - elapsed):.0f} s left"
                    if elapsed >= warmup_s + duration_s:
                        break

            failure = feed.failure(capture)
            if failure:
                draw_hud(screen, [*banner, "CAPTURE ERROR", failure], ERROR_COLOUR)
            else:
                draw_hud(
                    screen,
                    [
                        line
                        for line in (*banner, phase, *feed.hud_lines(mirrored))
                        if line
                    ],
                )
            pygame.display.flip()
            feed.after_flip()
            clock.tick(60)
            drawn += 1
            ready_frames += feed.ready_at is not None
            if stop_on_error and failure:
                if feed.status.kind is not StatusKind.ERROR:
                    feed.status = CaptureStatus(StatusKind.ERROR, failure)
                break
            if (
                max_frames is not None
                and startup_grace is not None
                and (ready_frames >= max_frames or drawn >= startup_grace)
            ):
                break
            elif (
                max_frames is not None
                and startup_grace is None
                and ready_frames >= max_frames
            ):
                break

    return SessionResult(
        feed.recorder.summary(),
        feed.camera_details,
        feed.status,
        quit_requested,
        mirrored,
    )


def main(
    argv: list[str] | None = None,
    *,
    worker: str = DEFAULT_WORKER,
    max_frames: int | None = None,
) -> None:
    """Run the debug window until the user quits.

    ``worker`` and ``max_frames`` exist for the automated smoke test, which
    runs the loop headless against a fake capture worker for a fixed number
    of display frames.
    """
    config, mirrored, log_dir = _parse_args(argv)
    pygame.init()
    screen = pygame.display.set_mode((config.width, config.height))
    pygame.display.set_caption("ParaParaPy – capture debug")
    result = run_session(
        screen,
        pygame.time.Clock(),
        config,
        worker=worker,
        mirrored=mirrored,
        max_frames=max_frames,
    )
    pygame.quit()
    if log_dir is not None:
        path = write_log(
            "capture-debug",
            {
                "config": dataclasses.asdict(config),
                "camera": result.camera,
                "last_status": {
                    "kind": str(result.last_status.kind),
                    "message": result.last_status.message,
                },
                "summary": result.summary,
            },  # fmt: skip
            log_dir,
        )
        print(f"session summary written to {path}")


if __name__ == "__main__":
    main()
