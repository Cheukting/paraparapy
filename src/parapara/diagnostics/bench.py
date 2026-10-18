"""Capture benchmark: run several capture configurations back to back and
log their performance.

This module is the *headless* benchmark (``python -m
parapara.diagnostics.bench``); ``uv run parapara-capture-bench`` runs the
windowed version in :mod:`parapara.game.capture_bench`, which shows the same
live view as the debug window. Dance in front of the camera while it runs. For each configuration it starts the
capture process, waits for ``READY``, records ``--seconds`` of samples (after
a warm-up), stops, and moves on. The results are printed as a table and
written to ``logs/capture-bench-*.json`` (same format for both versions).

Default configurations compare the two pose models (``full``, the game
default, and ``lite``). The hand model was removed after the first benchmark
(2026-10-18) showed it halved the tracking rate.

Comparing ``camera_fps`` with ``tracking_fps`` separates a camera-limited
frame rate (e.g. a webcam lowering its rate in dim light) from an
inference-limited one.

No window is opened, so this process needs neither ``pygame`` nor ``cv2``.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any

from parapara.diagnostics.recorder import DEFAULT_LOG_DIR, RunRecorder, write_log
from parapara.ipc.client import DEFAULT_WORKER, CaptureClient
from parapara.ipc.messages import CaptureConfig, CaptureStatus, StatusKind
from parapara.pose.landmarks import PoseSample

CONFIGS: dict[str, dict[str, Any]] = {
    "full": {"pose_model": "full"},
    "lite": {"pose_model": "lite"},
}
DEFAULT_CONFIGS = ("full", "lite")


class BenchError(RuntimeError):
    pass


def run_config(
    config: CaptureConfig,
    seconds: float,
    warmup_s: float,
    worker: str = DEFAULT_WORKER,
    ready_timeout: float = 120.0,
) -> dict[str, Any]:
    """Run one configuration and return its summary (plus camera details)."""
    recorder = RunRecorder(warmup_s=warmup_s)
    ready: CaptureStatus | None = None
    with CaptureClient(config, worker=worker) as capture:
        ready_deadline = time.monotonic() + ready_timeout
        end: float | None = None  # set when READY arrives
        # One loop for all messages: samples that arrive in the same batch as
        # READY must be recorded, not discarded.
        while end is None or time.monotonic() < end:
            for message in capture.poll():
                if isinstance(message, PoseSample):
                    recorder.add(message)
                elif message.kind is StatusKind.ERROR:
                    raise BenchError(message.message)
                elif message.kind is StatusKind.READY:
                    ready = message
                    end = time.monotonic() + warmup_s + seconds
                else:
                    print(f"  {message.kind}: {message.message}")
            if end is None and (not capture.is_alive or time.monotonic() > ready_deadline):
                raise BenchError("capture process did not become ready")
            time.sleep(0.005)
    assert ready is not None
    return {"camera": ready.details, "summary": recorder.summary()}


def format_table(results: dict[str, dict[str, Any]]) -> str:
    header = (
        f"{'config':<12} {'track fps':>9} {'cam fps':>8} {'skip %':>7} "
        f"{'pose ms':>8} {'cam→lm p50':>11} {'person %':>9}"
    )
    lines = [header, "-" * len(header)]
    for name, result in results.items():
        s = result.get("summary", {})
        if "error" in result or "error" in s:
            lines.append(f"{name:<12} ERROR: {result.get('error') or s.get('error')}")
            continue
        lines.append(
            f"{name:<12} {s['tracking_fps']:>9} {s['camera_fps']:>8} {s['skipped_pct']:>7} "
            f"{s['pose_ms']['mean']:>8} "
            f"{s['camera_to_landmarks_ms']['p50']:>11} {s['person_found_pct']:>9}"
        )
    return "\n".join(lines)


def build_parser(description: str = "Benchmark capture configurations") -> argparse.ArgumentParser:
    """Command-line options shared by the headless and the windowed benchmark."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seconds", type=float, default=15.0, help="recorded seconds per config")
    parser.add_argument("--warmup", type=float, default=2.0, help="ignored seconds at the start")
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                        help=f"comma-separated, from: {', '.join(CONFIGS)}")  # fmt: skip
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pause", type=float, default=3.0, help="seconds between configs")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser


def parse_configs(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[tuple[str, CaptureConfig]]:
    """``--configs`` → ``[(name, CaptureConfig), ...]`` (errors exit via ``parser``)."""
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        parser.error(f"unknown config(s): {unknown}")
    base = CaptureConfig(camera_index=args.camera, width=args.width, height=args.height, fps=args.fps)
    return [(n, dataclasses.replace(base, **CONFIGS[n])) for n in names]


def finish(args: argparse.Namespace, results: dict[str, dict[str, Any]]) -> Path:
    """Print the comparison table and write the benchmark log."""
    print()
    print(format_table(results))
    path = write_log(
        "capture-bench",
        {"seconds": args.seconds, "warmup_s": args.warmup, "results": results},
        args.log_dir,
    )
    print(f"\nlog written to {path}")
    return path


def main(argv: list[str] | None = None, *, worker: str = DEFAULT_WORKER) -> Path:
    """Headless benchmark (``python -m parapara.diagnostics.bench``).

    ``uv run parapara-capture-bench`` runs the windowed version in
    :mod:`parapara.game.capture_bench`, which shows the live camera view.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configs = parse_configs(parser, args)
    results: dict[str, dict[str, Any]] = {}
    for i, (name, config) in enumerate(configs):
        if i:
            time.sleep(args.pause)  # let the camera close before reopening it
        print(f"[{i + 1}/{len(configs)}] {name}: keep dancing for ~{args.warmup + args.seconds:.0f} s ...")
        try:
            results[name] = {"config": dataclasses.asdict(config),
                             **run_config(config, args.seconds, args.warmup, worker)}  # fmt: skip
        except BenchError as err:
            results[name] = {"config": dataclasses.asdict(config), "error": str(err)}
    return finish(args, results)


if __name__ == "__main__":
    main()
