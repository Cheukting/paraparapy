"""Record a capture run and summarize it for a log file.

The live HUD shows rolling values that are hard to compare by eye; a
:class:`RunRecorder` instead keeps one small row per sample for the whole
run and computes a summary at the end (rates, mean/median/p95 timings,
detection rates). Summaries are written as JSON under ``logs/`` (git-ignored)
so performance runs can be compared over time and read by tools.

Warm-up: MediaPipe's first inferences and the camera's auto-exposure are
slower than steady state, so samples captured during the first
``warmup_s`` seconds are excluded from the summary.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from parapara.pose.landmarks import PoseSample

DEFAULT_LOG_DIR = Path("logs")


@dataclass
class RunRecorder:
    """Collects per-sample measurements and summarizes them.

    Args:
        warmup_s: seconds at the start of the run to ignore.
    """

    warmup_s: float = 2.0
    _rows: list[tuple[float, ...]] = field(default_factory=list)
    _display_latency_ms: list[float] = field(default_factory=list)
    _first_captured_ns: int | None = None

    def add(self, sample: PoseSample) -> None:
        if self._first_captured_ns is None:
            self._first_captured_ns = sample.captured_ns
        if (sample.captured_ns - self._first_captured_ns) / 1e9 < self.warmup_s:
            return
        self._rows.append(
            (
                sample.captured_ns,
                sample.processed_ns,
                sample.frame_index,
                sample.pose_ms,
                sample.processing_ms,
                sample.has_pose,
                sample.dropped_samples,
            )
        )

    def add_display_latency(self, latency_ms: float) -> None:
        """Camera → screen latency measured by a viewer (optional)."""
        self._display_latency_ms.append(latency_ms)

    @property
    def sample_count(self) -> int:
        return len(self._rows)

    def summary(self) -> dict[str, Any]:
        """Summary statistics of the recorded (post-warm-up) samples.

        Rates:
            ``tracking_fps``: processed samples per second.
            ``camera_fps``: frames the camera delivered per second, from the
            frame-index range (see ``live_stats.CameraRateMeter``).
            ``skipped_pct``: camera frames that were never processed.
        """
        if len(self._rows) < 2:
            return {
                "samples": len(self._rows),
                "error": "not enough samples after warm-up",
            }
        a = np.array(self._rows, dtype=np.float64)
        captured, processed, index = a[:, 0], a[:, 1], a[:, 2]
        duration_s = (processed[-1] - processed[0]) / 1e9
        captured_span_s = (captured[-1] - captured[0]) / 1e9
        delivered = index[-1] - index[0]
        summary: dict[str, Any] = {
            "samples": len(a),
            "duration_s": round(duration_s, 2),
            "tracking_fps": _r((len(a) - 1) / duration_s if duration_s > 0 else 0.0),
            "camera_fps": _r(
                delivered / captured_span_s if captured_span_s > 0 else 0.0
            ),
            "skipped_pct": _r(
                100 * (1 - (len(a) - 1) / delivered) if delivered > 0 else 0.0
            ),
            "pose_ms": _stats(a[:, 3]),
            "camera_to_landmarks_ms": _stats(a[:, 4]),
            "person_found_pct": _r(100 * a[:, 5].mean()),
            "dropped_samples": int(a[-1, 6]),
        }
        if self._display_latency_ms:
            summary["camera_to_screen_ms"] = _stats(np.array(self._display_latency_ms))
        return summary


def _r(value: float) -> float:
    return round(float(value), 1)


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": _r(values.mean()),
        "p50": _r(np.percentile(values, 50)),
        "p95": _r(np.percentile(values, 95)),
        "max": _r(values.max()),
    }


def environment() -> dict[str, str]:
    """Machine and library versions, recorded with every log.

    Versions are read from package metadata so that ``mediapipe`` (and its
    bundled ``cv2``) is never imported by this process.
    """

    def version(dist: str) -> str:
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return "not installed"

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "mediapipe": version("mediapipe"),
        "opencv-contrib-python": version("opencv-contrib-python"),
    }


def write_log(
    kind: str, payload: dict[str, Any], log_dir: Path = DEFAULT_LOG_DIR
) -> Path:
    """Write ``payload`` (plus timestamp and environment) to
    ``<log_dir>/<kind>-YYYYmmdd-HHMMSS.json`` and return the path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    path = log_dir / f"{kind}-{now:%Y%m%d-%H%M%S}.json"
    record = {"kind": kind, "created": now.isoformat(timespec="seconds"),
              "environment": environment(), **payload}  # fmt: skip
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
