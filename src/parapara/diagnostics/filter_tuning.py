"""Tune the smoothing filters on recordings: ``uv run parapara-tune-filters``.

Repeats the tuning done for :data:`parapara.pose.features.LANDMARK_FILTER`
and :data:`~parapara.pose.features.TORSO_FILTER` on any recording session
(e.g. after changing camera or with a new dancer), following the One Euro
authors' procedure: jitter is measured while holding still, lag while moving
fast.

- **Landmark filter grid** (``min_cutoff`` × ``beta``):
  - *jitter*: median frame-to-frame change of the forearm directions in
    the static clips (``neutral``, ``t_pose``, ``right_arm_up``,
    ``right_arm_forward``);
  - *lag*: time shift that best aligns the filtered with the raw wrist
    trajectories in the block clips (1 ms resolution after interpolation).
- **Torso filter grid**: mean block-clip fit error (as in
  ``parapara-analyze``) of the full feature pipeline for each torso setting.

Results on ``recordings/20261018-144829`` chose the current defaults:
landmarks ``min_cutoff 0.5, beta 10`` (jitter 1.19° → 0.47° per frame, lag
~28 ms); torso ``min_cutoff 0.5, beta 0.5`` (mean error 19.8° → 19.6°).

Pure NumPy (no cv2/mediapipe/pygame).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from parapara.diagnostics.pose_analysis import (
    STATIC_CLIPS,
    analyze_block,
    latest_session,
)
from parapara.pose.body_frame import arm_directions
from parapara.pose.features import (
    LANDMARK_FILTER,
    FeatureExtractor,
    FilterParams,
    as_arm_directions,
    extract_recording,
)
from parapara.pose.filters import filter_series
from parapara.pose.landmarks import PoseLandmark as P
from parapara.pose.recording import Recording

LANDMARK_GRID = {
    "min_cutoff": (0.5, 1.0, 2.0, 3.0, 5.0),
    "beta": (0.0, 3.0, 10.0, 30.0),
}
TORSO_GRID = (
    (50.0, 0.0),
    (2.0, 0.0),
    (0.5, 0.0),
    (0.5, 0.5),
    (0.2, 0.0),
)  # (min_cutoff, beta)


def forearm_jitter_deg(world: np.ndarray) -> float:
    """Median frame-to-frame change of both forearm directions, degrees."""
    steps = []
    for d in arm_directions(world).values():
        dots = (d.fore[1:] * d.fore[:-1]).sum(-1)
        steps.append(np.degrees(np.arccos(np.clip(dots, -1, 1))))
    return float(np.nanmedian(np.concatenate(steps)))


def trajectory_lag_ms(rec: Recording, filtered: np.ndarray, max_ms: int = 250) -> float:
    """Median (over wrists and x/y) shift aligning filtered with raw positions."""
    t = (rec.captured_ns - rec.captured_ns[0]) / 1e6
    grid = np.arange(0, t[-1], 1.0)
    lags = []
    for idx in (P.RIGHT_WRIST, P.LEFT_WRIST):
        for axis in (0, 1):
            raw = np.interp(grid, t, rec.pose_world[:, idx, axis])
            filt = np.interp(grid, t, filtered[:, idx, axis])
            errors = [
                np.mean((filt[s:] - raw[: len(raw) - s]) ** 2) for s in range(max_ms)
            ]
            lags.append(int(np.argmin(errors)))
    return float(np.median(lags))


def tune(session: Path) -> dict:
    static = [
        Recording.load(session / f"{c}.npz")
        for c in STATIC_CLIPS
        if (session / f"{c}.npz").exists()
    ]
    blocks = [Recording.load(f) for f in sorted(session.glob("block_*.npz"))]
    report: dict = {"session": str(session), "landmarks": [], "torso": []}
    if static:
        report["raw_jitter_deg"] = round(
            np.mean([forearm_jitter_deg(r.pose_world) for r in static]), 2
        )
    for mc, beta in itertools.product(
        LANDMARK_GRID["min_cutoff"], LANDMARK_GRID["beta"]
    ):
        params = {"min_cutoff": mc, "beta": beta}
        row = {**params}
        if static:
            row["jitter_deg"] = round(np.mean([forearm_jitter_deg(
                np.concatenate([filter_series(r.pose_world[:, :, :3], r.captured_ns, **params),
                                r.pose_world[:, :, 3:]], axis=-1)) for r in static]), 2)  # fmt: skip
        if blocks:
            row["lag_ms"] = float(np.median([trajectory_lag_ms(
                r, filter_series(r.pose_world[:, :, :3], r.captured_ns, **params)) for r in blocks]))  # fmt: skip
        report["landmarks"].append(row)
    for mc, beta in TORSO_GRID:
        errors = [analyze_block(r, arms=as_arm_directions(extract_recording(
            r, FeatureExtractor(landmark_filter=LANDMARK_FILTER, torso_filter=FilterParams(mc, beta))
        ))).mean_error_deg for r in blocks]  # fmt: skip
        report["torso"].append({"min_cutoff": mc, "beta": beta,
                                "mean_error_deg": round(float(np.mean(errors)), 1) if errors else None})  # fmt: skip
    return report


def format_tuning(report: dict) -> str:
    lines = [f"session {report['session']}", "",
             f"landmark filter (raw jitter {report.get('raw_jitter_deg', '?')}°/frame):"]  # fmt: skip
    for row in report["landmarks"]:
        lines.append(f"  min_cutoff {row['min_cutoff']:<4} beta {row['beta']:<5} "
                     f"jitter {row.get('jitter_deg', '-')}°  lag {row.get('lag_ms', '-')} ms")  # fmt: skip
    lines += ["", "torso filter (mean block fit error, full pipeline):"]
    for row in report["torso"]:
        lines.append(
            f"  min_cutoff {row['min_cutoff']:<4} beta {row['beta']:<4} error {row['mean_error_deg']}°"
        )
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Tune the landmark and torso smoothing filters"
    )
    parser.add_argument(
        "session", nargs="?", type=Path, help="default: latest in recordings/"
    )
    args = parser.parse_args(argv)
    report = tune(args.session or latest_session())
    print(format_tuning(report))
    return report


def main(argv: list[str] | None = None) -> None:
    """Console entry point (returns ``None``; see ``pose_analysis.main``)."""
    run(argv)


if __name__ == "__main__":
    main()
