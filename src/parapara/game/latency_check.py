"""Latency calibration test: ``uv run parapara-latency-test``.

(Module named ``latency_check``, not ``latency_test``: pytest would collect a
``*_test.py`` file as tests.)

A separate, hands-free tool (the game itself is unchanged) to find out how
consistent the "clap to the beat" latency calibration is, and whether 8 claps
are enough. Three launches of the game measured 261, 96 and 122 ms with 8
claps (16 claps had given 225 ms), 2026-10-23.

It runs the **same measurement as the game**: the same click track
(:func:`parapara.audio.beatgrid.render_click_track` through the game's audio
player), the same capture pipeline and features, the same estimator
(:func:`parapara.scoring.latency.estimate_latency`), at the game's tempo
(100 BPM) with a 4-click count-in. Rounds alternate between clap counts
(``--claps 8,16``, ``--rounds 6``: 8, 16, 8, 16, 8, 16).

Flow (no keys needed; the player stands back): position check (hips in the
picture, as in the game's calibration) -> per round: "Round 2 of 6: 16 claps"
-> 4, 3, 2, 1 on the count-in clicks -> CLAP! with a counter -> the result for
a few seconds -> next round -> a summary, then it closes by itself. ``Esc``
quits early (finished rounds are still logged).

Log: ``logs/latency-test-<date-time>.json`` with, per round, the requested
claps, the click times, the result as the game would compute it (on the
unsmoothed timing copy, like the judge) **and** on the smoothed wrist distance
for comparison, every clap's offset from its click, and the wrist-distance
series itself (both copies) so the analysis can be redone offline.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from parapara import settings  # noqa: E402
from parapara.audio.beatgrid import BeatGrid, render_click_track  # noqa: E402
from parapara.diagnostics.recorder import write_log  # noqa: E402
from parapara.game.beatcheck import Player  # noqa: E402
from parapara.game.branding import NAME, set_window_icon  # noqa: E402
from parapara.game.debug_view import LiveFeed  # noqa: E402
from parapara.game.notice_screens import privacy_gate  # noqa: E402
from parapara.game.recorder import draw_framing_box  # noqa: E402
from parapara.game.theme import ACCENT, BG, DIM, ERROR, OK, PANEL, TEXT, WARN  # noqa: E402
from parapara.game.ui import (  # noqa: E402
    UI,
    adjust_scale,
    display_flags,
    footer,
    layout_issues,
    load_scale,
    text,
)
from parapara.ipc.client import DEFAULT_WORKER, CaptureClient  # noqa: E402
from parapara.ipc.messages import CaptureConfig  # noqa: E402
from parapara.pose.features import FeatureExtractor, PoseFeatures  # noqa: E402
from parapara.pose.framing import FramingMonitor  # noqa: E402
from parapara.scoring.latency import LatencyResult, estimate_latency  # noqa: E402

WIDTH, HEIGHT, STAGE_W = 1240, 720, 600  # same layout as the game
GAME_BPM, COUNT_IN = 100.0, 4  # the game's latency calibration (parapara.game.app)
INTRO_S, RESULT_S, SUMMARY_S = 2.5, 3.5, 8.0


class Step(Enum):
    POSITION = auto()  # waiting for a good position
    INTRO = auto()  # "Round k of n: N claps"
    CLICKS = auto()  # count-in + claps
    RESULT = auto()  # this round's result
    SUMMARY = auto()  # all rounds, then close


@dataclass
class Round:
    """One calibration round and what it measured."""

    number: int
    claps: int
    start_ns: int = 0
    click_ns: list[int] = field(default_factory=list)
    count_in_ns: list[int] = field(default_factory=list)
    features: list[PoseFeatures] = field(default_factory=list)
    result: LatencyResult | None = None  # as the game computes it (timing copy)
    smoothed: LatencyResult | None = None  # on the smoothed wrist distance

    def evaluate(self, beat_ns: float) -> None:
        self.result = estimate_latency(self.features, self.click_ns, beat_ns)
        # Without the timing copy, for_timing() falls back to the smoothed values.
        plain = [dataclasses.replace(f, timing_right=None, timing_left=None, timing_wrist_distance=None)
                 for f in self.features]  # fmt: skip
        self.smoothed = estimate_latency(plain, self.click_ns, beat_ns)

    def to_log(self) -> dict[str, Any]:
        rel = lambda t: round((t - self.start_ns) / 1e6, 1)  # noqa: E731  (ms since the round's audio start)
        return {
            "round": self.number, "claps_requested": self.claps,
            "click_ms": [rel(t) for t in self.click_ns], "count_in_ms": [rel(t) for t in self.count_in_ns],
            "result": dataclasses.asdict(self.result) if self.result else None,
            "result_smoothed": dataclasses.asdict(self.smoothed) if self.smoothed else None,
            "series": {"t_ms": [rel(f.t_ns) for f in self.features],
                       "wrist_distance": [round(f.wrist_distance, 3) for f in self.features],
                       "wrist_distance_timing": [round(f.for_timing().wrist_distance, 3) for f in self.features]},
        }  # fmt: skip


def summarise(rounds: list[Round]) -> dict[str, Any]:
    """Per clap count: the accepted latencies, their mean and range."""
    out: dict[str, Any] = {}
    for claps in sorted({r.claps for r in rounds}):
        values = [
            r.result.latency_ms
            for r in rounds
            if r.claps == claps and r.result and r.result.ok
        ]
        out[str(claps)] = {
            "rounds": sum(r.claps == claps for r in rounds), "accepted": len(values), "latency_ms": values,
            "mean_ms": round(float(np.mean(values)), 1) if values else None,
            "range_ms": round(max(values) - min(values), 1) if len(values) > 1 else None,
        }  # fmt: skip
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hands-free test of the clap latency calibration"
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument(
        "--claps", default="8,16", help="clap counts, used in turn (default 8,16)"
    )
    parser.add_argument(
        "--bpm", type=float, default=GAME_BPM, help="click tempo (the game uses 100)"
    )
    parser.add_argument(
        "--window", action="store_true", help="run in a window instead of fullscreen"
    )
    parser.add_argument(
        "--text-scale", type=float, help="text size (default: saved, or 2.0)"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", choices=("lite", "full"), default="full")
    parser.add_argument(
        "--fast", action="store_true", help=argparse.SUPPRESS
    )  # tests: short pauses
    parser.add_argument(
        "--check-layout", action="store_true", help=argparse.SUPPRESS
    )  # tests
    args = parser.parse_args(argv)
    args.clap_counts = [int(c) for c in args.claps.split(",") if c]
    return args


def run(
    argv: list[str] | None = None, *, worker: str = DEFAULT_WORKER
) -> dict[str, Any]:
    """Run the test; returns the logged payload (also written to ``logs/``)."""
    args = _parse_args(argv)
    config = CaptureConfig(camera_index=args.camera, pose_model=args.model)
    intro_s, result_s, summary_s = (
        (0.3, 0.3, 0.3) if args.fast else (INTRO_S, RESULT_S, SUMMARY_S)
    )
    pygame.init()
    load_scale(args.text_scale)
    accepted, _ = privacy_gate(
        args.window, None, args.check_layout
    )  # before the camera starts
    if not accepted:
        pygame.quit()
        return {}
    set_window_icon()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), display_flags(args.window))
    pygame.display.set_caption(f"{NAME} – latency test")
    clock, player = pygame.time.Clock(), Player(44100)
    feed, framing = LiveFeed(config), FramingMonitor()
    extractor = FeatureExtractor(calibration=settings.load_calibration())
    grid = BeatGrid(args.bpm, 0.5, 0)
    beat_ns = grid.beat_s * 1e9
    rounds: list[Round] = []
    step, step_at, current = Step.POSITION, time.monotonic(), None
    reported: set[str] = set()
    running = True

    def start_round() -> Round:
        r = Round(
            len(rounds) + 1, args.clap_counts[len(rounds) % len(args.clap_counts)]
        )
        beats = COUNT_IN + r.claps
        silence = np.zeros(
            int((0.5 + (beats + 1) * grid.beat_s) * player.sr), np.float32
        )
        player.play(
            render_click_track(
                silence, player.sr, grid, music_gain=0.0, click_gain=0.9
            ),
            0.0,
        )
        r.start_ns = player.start_ns
        r.count_in_ns = [
            r.start_ns + int(grid.time_of(b) * 1e9) for b in range(COUNT_IN)
        ]
        r.click_ns = [
            r.start_ns + int(grid.time_of(b) * 1e9) for b in range(COUNT_IN, beats)
        ]
        return r

    with CaptureClient(config, worker=worker) as capture:
        while running:
            keys = set()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    keys.add(event.key)
            if pygame.K_ESCAPE in keys or pygame.K_q in keys:
                running = False
            if pygame.K_f in keys:
                pygame.display.toggle_fullscreen()
            adjust_scale(keys)
            samples = feed.update(capture)
            for sample in samples:
                framing.update(sample)
                features = extractor.update(sample)
                if step is Step.CLICKS and current is not None and features is not None:
                    current.features.append(features)
            now, now_ns = time.monotonic(), time.monotonic_ns()

            # --- steps (all advance by themselves) ---
            if step is Step.POSITION and feed.ready_at is not None and framing.stable:
                step, step_at = Step.INTRO, now
            elif step is Step.INTRO and now - step_at >= intro_s:
                current = start_round()
                step = Step.CLICKS
            elif (
                step is Step.CLICKS
                and current is not None
                and now_ns > current.click_ns[-1] + 1e9
            ):
                current.evaluate(beat_ns)
                rounds.append(current)
                step, step_at = Step.RESULT, now
            elif step is Step.RESULT and now - step_at >= result_s:
                step, step_at = (
                    (Step.INTRO if len(rounds) < args.rounds else Step.SUMMARY),
                    now,
                )
            elif step is Step.SUMMARY and now - step_at >= summary_s:
                running = False

            # --- draw ---
            UI.drawn = [] if args.check_layout else None
            screen.fill(BG)
            camera = screen.subsurface((STAGE_W, 0, config.width, config.height))
            feed.draw(camera, True)
            failure = feed.failure(capture)
            if failure:
                text(camera, "CAMERA PROBLEM", (20, 20), 30, ERROR)
            elif step is Step.POSITION:
                draw_framing_box(camera, framing.latest, True)
            stage = screen.subsurface((0, 0, STAGE_W, HEIGHT))
            panel = screen.subsurface(
                (STAGE_W, config.height, WIDTH - STAGE_W, HEIGHT - config.height)
            )
            panel.fill(PANEL)
            draw_stage(stage, step, current, rounds, args, framing, now_ns, beat_ns)
            draw_panel(panel, rounds)
            if UI.drawn is not None:
                for issue in layout_issues(UI.drawn):
                    if issue not in reported:
                        reported.add(issue)
                        print(
                            f"LAYOUT ({step.name.lower()}, text size {UI.scale}): {issue}",
                            file=sys.stderr,
                        )
            pygame.display.flip()
            feed.after_flip()
            clock.tick(60)
    player.pause()
    pygame.quit()
    payload = {"bpm": args.bpm, "count_in": COUNT_IN, "clap_counts": args.clap_counts,
               "calibrated": extractor.calibration is not None, "summary": summarise(rounds),
               "rounds": [r.to_log() for r in rounds]}  # fmt: skip
    if rounds:
        print(f"log written to {write_log('latency-test', payload)}")
    return payload


def draw_stage(stage, step: Step, current: Round | None, rounds: list[Round], args, framing, now_ns: int,
               beat_ns: float) -> None:  # fmt: skip
    """Left side: what is happening, with one big label."""
    y = text(stage, "Latency test", (24, 20), 34, ACCENT)
    total = args.rounds
    big, colour, line = "", ACCENT, ""
    if step is Step.POSITION:
        latest = framing.latest
        line = (
            latest.issues[0].message
            if latest is not None and latest.issues
            else "Stand in position"
        )
        colour = WARN if latest is not None and latest.issues else ACCENT
    elif step is Step.INTRO:
        n = len(rounds) + 1
        line, big = (
            f"Round {n} of {total}",
            f"{args.clap_counts[(n - 1) % len(args.clap_counts)]} claps",
        )
    elif step is Step.CLICKS and current is not None:
        heard = sum(c <= now_ns for c in current.count_in_ns)
        done = sum(c <= now_ns for c in current.click_ns)
        line = f"Round {current.number} of {total}: clap on every click"
        big = "CLAP!" if done else (str(COUNT_IN - heard + 1) if heard else "ready")
        on_click = any(
            0 <= now_ns - c < 0.3 * beat_ns
            for c in current.count_in_ns + current.click_ns
        )
        colour = ACCENT if on_click or done else TEXT
        if done:
            text(
                stage,
                f"{done} / {current.claps}",
                (STAGE_W // 2, 430),
                40,
                TEXT,
                centre=True,
            )
    elif step is Step.RESULT and rounds:
        r = rounds[-1].result
        line = f"Round {rounds[-1].number} of {total} done"
        big, colour = (
            (f"{r.latency_ms:.0f} ms", OK) if r and r.ok else ("again next round", WARN)
        )
        if r and not r.ok:
            text(stage, r.reason, (24, 430), 20, WARN)
    elif step is Step.SUMMARY:
        line, big, colour = "All rounds done - thank you!", "Done", OK
    y = text(stage, line, (24, y + 10), 24, colour if step is Step.POSITION else TEXT)
    if big:
        text(
            stage,
            big,
            (STAGE_W // 2, max(y + 40, 230)),
            72 if len(big) < 8 else 44,
            colour,
            centre=True,
        )
    footer(stage, [("Hands-free: nothing to press.   [Esc] quit", 16, DIM)])


def draw_panel(panel, rounds: list[Round]) -> None:
    """Under the camera: the results so far (one line per round)."""
    y = 10
    for r in rounds[-5:]:
        res = r.result
        value = (
            f"{res.latency_ms:.0f} ms ({res.claps_found} claps)"
            if res and res.ok
            else "not accepted"
        )
        y = text(
            panel,
            f"round {r.number}, {r.claps} claps: {value}",
            (20, y),
            16,
            TEXT if res and res.ok else DIM,
        )


def main(argv: list[str] | None = None) -> None:
    """Console entry point (returns ``None``; see :func:`run`)."""
    run(argv)


if __name__ == "__main__":
    main()
