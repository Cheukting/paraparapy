"""Windowed capture benchmark: ``uv run parapara-capture-bench``.

Runs several capture configurations back to back (default ``full`` and
``lite`` pose models), showing the **same live view as the debug
window** for each one (camera picture, skeleton, HUD), plus the benchmark step
and a countdown, so the player gets feedback while dancing.

Each configuration gets its own capture process (the model choice is fixed
when the worker starts). Between configurations a short "next up" screen
gives the camera time to close and reopen. ``Esc``/``Q`` stops the benchmark
early; results collected so far are still logged.

Results are printed as a table and written to ``logs/capture-bench-*.json``,
in the same format as the headless :mod:`parapara.diagnostics.bench`.
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402  (must follow the env var above)

from parapara.diagnostics.bench import build_parser, finish, parse_configs  # noqa: E402
from parapara.game.debug_view import (  # noqa: E402
    TEXT_COLOUR,
    draw_hud,
    poll_quit_and_mirror,
    run_session,
)
from parapara.ipc.client import DEFAULT_WORKER  # noqa: E402


def _between_configs(
    screen: pygame.Surface, clock: pygame.time.Clock, seconds: float, lines: list[str], mirrored: bool
) -> tuple[bool, bool]:
    """Show a "next up" screen for ``seconds``; returns ``(quit, mirrored)``."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        quit_requested, mirrored = poll_quit_and_mirror(mirrored)
        if quit_requested:
            return True, mirrored
        screen.fill((20, 20, 30))
        draw_hud(screen, [*lines, f"starting in {end - time.monotonic():.0f} s"], TEXT_COLOUR)
        pygame.display.flip()
        clock.tick(30)
    return False, mirrored


def run(
    argv: list[str] | None = None,
    *,
    worker: str = DEFAULT_WORKER,
    max_frames: int | None = None,
) -> Path:
    """Run the windowed benchmark and return the log path.

    ``worker`` and ``max_frames`` (per configuration) exist for the
    automated headless test with a fake capture worker.
    """
    parser = build_parser("Benchmark capture configurations with a live view")
    parser.add_argument("--no-mirror", action="store_true", help="start unmirrored")
    args = parser.parse_args(argv)
    configs = parse_configs(parser, args)
    mirrored = not args.no_mirror

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("parapara – capture benchmark")
    clock = pygame.time.Clock()

    results: dict[str, dict[str, Any]] = {}
    total = len(configs)
    for i, (name, config) in enumerate(configs):
        step = f"benchmark {i + 1}/{total}: {name}"
        if i:
            quit_requested, mirrored = _between_configs(
                screen, clock, args.pause, [f"done: {configs[i - 1][0]}", f"next: {step}"], mirrored
            )
            if quit_requested:
                break
        print(f"[{i + 1}/{total}] {name}: keep dancing for ~{args.warmup + args.seconds:.0f} s ...")
        session = run_session(
            screen,
            clock,
            config,
            worker=worker,
            mirrored=mirrored,
            duration_s=args.seconds,
            warmup_s=args.warmup,
            banner=(step,),
            stop_on_error=True,
            max_frames=max_frames,
        )
        mirrored = session.mirrored
        entry: dict[str, Any] = {"config": dataclasses.asdict(config)}
        if session.error:
            entry["error"] = session.error
        else:
            entry |= {"camera": session.camera, "summary": session.summary}
        if session.quit_requested:
            entry["note"] = "stopped early by the user"
        results[name] = entry
        if session.quit_requested:
            break

    pygame.quit()
    return finish(args, results)


def main(argv: list[str] | None = None) -> None:
    """Console entry point. Returns ``None`` on purpose: console scripts call
    ``sys.exit(main())``, which would print any returned value and exit with
    status 1. Use :func:`run` to get the result programmatically."""
    run(argv)


if __name__ == "__main__":
    main()
