"""Compare recorded clips with the block library: ``uv run parapara-analyze``.

Use this to **recalibrate the draft poses** whenever new recordings exist
(e.g. from a different dancer): it reports, for every keyframe, the target
direction from :mod:`parapara.routine.library` next to the direction actually
danced, so ``library.py`` can be updated from data. The report is printed and
written to ``<session>/analysis.json``.

What it measures
----------------
Static clips (``neutral``, ``t_pose``, ``right_arm_up``, ``right_arm_forward``):

- median arm directions in the body frame (axis checks),
- wrist jitter (standard deviation in metres per MediaPipe axis),
- shoulder width and upper-arm / forearm lengths (calibration values),
- torso lean towards the camera (camera tilt).

Block clips (one building block danced to the metronome):

- **best lag**: the time shift between the metronome beats and the movement
  that best matches the targets (reaction time + system latency). A block
  repeats every ``duration`` counts, so shifting by a whole repeat looks
  identical: the lag is only searched within one repeat, starting at
  ``LAG_SEARCH_START_S``. If the match hardly depends on the lag (e.g. a knock
  hits the same pose twice), the lag is reported as not measurable (``None``)
  and the fits use lag 0 (they barely depend on it then);
- per keyframe and arm: mean danced direction at the keyframe times, and its
  angle to the target, both in 3-D and in the **front plane** (``x``/``y``
  only). Depth (``z``) is MediaPipe's least reliable axis, so a large 3-D
  error with a small front-plane error usually means depth noise rather than
  a wrong pose.

Calibration (``calibration`` in the report): if the session has
``neutral.npz`` and ``t_pose.npz``, the player calibration
(:mod:`parapara.pose.calibration`) is computed and saved as
``<session>/calibration.json``; if it is rejected, the reasons are reported.

Framing (``framing`` in the report): for every clip, the share of samples in
which the player was well placed (:mod:`parapara.pose.framing`) and which
issues occurred, e.g. hips out of the picture. Clips with poor framing make
unreliable references and should be re-recorded. (``turned`` is expected
while dancing sweeps and flicks.)

Body movement (``torso_motion`` and ``frame_comparison`` in the report):
dancers turn, lean and step while dancing, so arm directions are measured
relative to the body. For every block clip the report gives the torso's
turning / side tilt / lean (mean ± std) and the hip sway in the image, plus
the mean error under each body-frame variant of
:data:`parapara.pose.body_frame.FRAME_METHODS` and under the full real-time
pipeline (``torso+filters``: :class:`parapara.pose.features.FeatureExtractor`),
to check that the defaults are still the best choice for new recordings.

Pure NumPy (no cv2/mediapipe/pygame).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from parapara.pose.body_frame import (
    FRAME_METHODS,
    ArmDirections,
    angle_deg,
    arm_directions,
    body_frames,
    torso_motion,
)
from collections import Counter

from parapara.pose.framing import check_framing
from parapara.pose.landmarks import PoseLandmark as P
from parapara.pose.recording import Recording
from parapara.routine.library import BLOCK_LIBRARY
from parapara.routine.model import Block

DEFAULT_RECORDINGS = Path("recordings")
LAG_SEARCH_START_S = -0.3
LAG_STEP_S = 0.02
FLAT_ERROR_DEG = 2.0  # error range below which the lag counts as not measurable
STATIC_CLIPS = ("neutral", "t_pose", "right_arm_up", "right_arm_forward")


def _r(v: Any, digits: int = 2) -> Any:
    """Round floats / arrays for readable JSON."""
    if isinstance(v, np.ndarray | list | tuple):
        return [_r(x, digits) for x in v]
    return round(float(v), digits)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _front(v: np.ndarray) -> np.ndarray:
    """Project direction(s) onto the body's front plane (drop z) and normalize."""
    v = np.array(v, dtype=np.float64)
    v[..., 2] = 0.0
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# --- static clips -------------------------------------------------------------


@dataclass
class StaticReport:
    clip_id: str
    samples: int
    arms: dict[str, dict[str, list[float]]]  # side -> segment -> median direction
    wrist_jitter_m: dict[str, list[float]]  # side -> std per MediaPipe axis
    shoulder_width_m: float
    upper_arm_ratio: float  # upper arm length / shoulder width
    forearm_ratio: float
    torso_lean_deg: float  # + = leaning towards the camera


def analyze_static(rec: Recording) -> StaticReport:
    world = rec.pose_world[rec.has_pose]
    arms = arm_directions(world)
    w = world[:, :, :3].astype(np.float64)

    def length(a: int, b: int) -> float:
        return float(np.nanmedian(np.linalg.norm(w[:, a] - w[:, b], axis=-1)))

    shoulder = length(P.RIGHT_SHOULDER, P.LEFT_SHOULDER)
    upper = np.mean([length(P.RIGHT_SHOULDER, P.RIGHT_ELBOW), length(P.LEFT_SHOULDER, P.LEFT_ELBOW)])
    fore = np.mean([length(P.RIGHT_ELBOW, P.RIGHT_WRIST), length(P.LEFT_ELBOW, P.LEFT_WRIST)])
    up = np.nanmedian(body_frames(world).up, axis=0)  # torso up, MediaPipe coords
    # Lean towards the camera: up tilting towards -z (MediaPipe +z = away).
    lean = float(np.degrees(np.arctan2(-up[2], -up[1])))
    return StaticReport(
        clip_id=rec.metadata.get("clip_id", "?"),
        samples=len(world),
        arms={side: {seg: _r(np.nanmedian(getattr(d, seg), axis=0))
                     for seg in ("upper", "fore", "hand")} for side, d in arms.items()},  # fmt: skip
        wrist_jitter_m={
            "right": _r(np.nanstd(w[:, P.RIGHT_WRIST], axis=0), 3),
            "left": _r(np.nanstd(w[:, P.LEFT_WRIST], axis=0), 3),
        },
        shoulder_width_m=_r(shoulder, 3),
        upper_arm_ratio=_r(upper / shoulder),
        forearm_ratio=_r(fore / shoulder),
        torso_lean_deg=_r(lean, 1),
    )


# --- block clips --------------------------------------------------------------


@dataclass
class KeyframeFit:
    beat: float
    side: str
    instances: int
    target_upper: list[float]
    seen_upper: list[float]
    error_upper_deg: float
    front_error_upper_deg: float
    target_fore: list[float]
    seen_fore: list[float]
    error_fore_deg: float
    front_error_fore_deg: float


@dataclass
class BlockReport:
    clip_id: str
    block_id: str
    bpm: float
    best_lag_ms: float | None  # None: not measurable (see module docstring)
    mean_error_deg: float  # at the best lag
    mean_error_at_zero_lag_deg: float
    keyframes: list[KeyframeFit] = field(default_factory=list)


def _keyframe_events(block: Block, n_counts: int, beat_s: float) -> list[tuple[float, int]]:
    """``(time_s, keyframe_index)`` for the block repeated over ``n_counts``."""
    return [
        (start * beat_s + kf.beat * beat_s, k)
        for start in range(0, n_counts, block.duration)
        for k, kf in enumerate(block.keyframes)
    ]


def analyze_block(
    rec: Recording,
    block: Block | None = None,
    lags_s: np.ndarray | None = None,
    max_gap_s: float = 0.06,
    frame_method: str = "torso",
    arms: dict[str, ArmDirections] | None = None,
) -> BlockReport:
    """Fit a block clip against its library block (see module docstring).

    Args:
        block: block to compare with; defaults to the clip's ``block_id``.
        lags_s: candidate lags (movement after the beat is positive);
            defaults to one block repeat starting at ``LAG_SEARCH_START_S``.
        frame_method: body-frame variant (see ``body_frame.FRAME_METHODS``).
        arms: precomputed arm directions per sample (e.g. from the smoothed
            feature pipeline, ``features.as_arm_directions``); overrides
            ``frame_method``.
        max_gap_s: a keyframe instance is skipped if no sample lies within
            this distance of its (lagged) time.
    """
    m = rec.metadata
    block = block or BLOCK_LIBRARY[m["block_id"]]
    beat_s = 60.0 / m["bpm"]
    beats = np.asarray(m["beat_times_ns"], dtype=np.int64)[m["count_in_beats"]:]
    t = (rec.captured_ns - beats[0]) / 1e9  # seconds since the first recorded beat
    if arms is None:
        arms = arm_directions(rec.pose_world, body_frames(rec.pose_world, method=frame_method))
    events = _keyframe_events(block, len(beats), beat_s)
    if lags_s is None:
        period = block.duration * beat_s
        lags_s = np.round(np.arange(LAG_SEARCH_START_S, LAG_SEARCH_START_S + period, LAG_STEP_S), 3)

    def sample_at(time_s: float) -> int | None:
        i = int(np.argmin(np.abs(t - time_s)))
        return i if abs(t[i] - time_s) <= max_gap_s else None

    def mean_error(lag: float) -> float:
        errors = []
        for time_s, k in events:
            i = sample_at(time_s + lag)
            if i is None:
                continue
            kf = block.keyframes[k]
            for side in ("right", "left"):
                target = getattr(kf, side)
                d = arms[side]
                if target is None or not d.valid[i]:
                    continue
                errors += [float(angle_deg(d.upper[i:i + 1], target.upper)[0]),
                           float(angle_deg(d.fore[i:i + 1], target.fore)[0])]  # fmt: skip
        return float(np.mean(errors)) if errors else float("inf")

    scores = np.array([mean_error(lag) for lag in lags_s])
    finite = scores[np.isfinite(scores)]
    measurable = finite.size > 0 and float(finite.max() - finite.min()) >= FLAT_ERROR_DEG
    best = float(lags_s[int(np.argmin(scores))]) if measurable else 0.0

    fits = []
    for k, kf in enumerate(block.keyframes):
        idx = [i for time_s, kk in events if kk == k and (i := sample_at(time_s + best)) is not None]
        for side in ("right", "left"):
            target = getattr(kf, side)
            d = arms[side]
            rows = [i for i in idx if d.valid[i]]
            if target is None or not rows:
                continue
            seen_u, seen_f = _unit(d.upper[rows].mean(0)), _unit(d.fore[rows].mean(0))
            tu, tf = np.array(target.upper), np.array(target.fore)
            fits.append(KeyframeFit(
                beat=kf.beat, side=side, instances=len(rows),
                target_upper=_r(tu), seen_upper=_r(seen_u),
                error_upper_deg=_r(angle_deg(seen_u, tu), 0),
                front_error_upper_deg=_r(angle_deg(_front(seen_u), _front(tu)), 0),
                target_fore=_r(tf), seen_fore=_r(seen_f),
                error_fore_deg=_r(angle_deg(seen_f, tf), 0),
                front_error_fore_deg=_r(angle_deg(_front(seen_f), _front(tf)), 0),
            ))  # fmt: skip
    return BlockReport(
        clip_id=m.get("clip_id", "?"),
        block_id=block.id,
        bpm=float(m["bpm"]),
        best_lag_ms=_r(best * 1000, 0) if measurable else None,
        mean_error_deg=_r(mean_error(best), 1),
        mean_error_at_zero_lag_deg=_r(mean_error(0.0), 1),
        keyframes=fits,
    )


def body_movement(rec: Recording) -> dict[str, Any]:
    """Torso motion statistics and the frame-variant comparison for a block clip."""
    world = rec.pose_world[rec.has_pose]
    motion = {k: {"mean": _r(v.mean(), 1), "std": _r(v.std(), 1)}
              for k, v in torso_motion(world).items()}  # fmt: skip
    hips_x = rec.pose_image[rec.has_pose][:, [P.RIGHT_HIP, P.LEFT_HIP], 0].mean(1)
    motion["hip_sway_pct_of_width"] = _r(100 * hips_x.std(), 1)
    comparison = {m: analyze_block(rec, frame_method=m).mean_error_deg for m in FRAME_METHODS}
    # The real-time pipeline: filtered landmarks + adaptive torso smoothing.
    from parapara.pose.features import as_arm_directions, extract_recording

    smoothed = as_arm_directions(extract_recording(rec))
    comparison["torso+filters"] = analyze_block(rec, arms=smoothed).mean_error_deg
    return {"torso_motion": motion, "frame_comparison": comparison}


def framing_summary(rec: Recording) -> dict[str, Any]:
    results = [check_framing(s) for s in rec.samples()]
    if not results:
        return {"ok_pct": 0.0, "issues": {}}
    counts = Counter(code for r in results for code in r.codes)
    return {
        "ok_pct": _r(100 * sum(r.ok for r in results) / len(results), 1),
        "issues": {code: _r(100 * n / len(results), 1) for code, n in counts.most_common()},
    }


# --- session -----------------------------------------------------------------


def latest_session(root: Path = DEFAULT_RECORDINGS) -> Path:
    sessions = sorted(p for p in root.iterdir() if (p / "session.json").exists())
    if not sessions:
        raise FileNotFoundError(f"no recording sessions in {root}")
    return sessions[-1]


def analyze_session(session: Path) -> dict[str, Any]:
    static, blocks, framing = {}, {}, {}
    for path in sorted(session.glob("*.npz")):
        rec = Recording.load(path)
        framing[path.stem] = framing_summary(rec)
        if rec.metadata.get("kind") == "static" and rec.has_pose.any():
            static[path.stem] = asdict(analyze_static(rec))
        elif rec.metadata.get("kind") == "block" and rec.has_pose.any():
            blocks[path.stem] = asdict(analyze_block(rec)) | body_movement(rec)
    calibration: dict[str, Any] = {}
    if (session / "neutral.npz").exists() and (session / "t_pose.npz").exists():
        from parapara.pose.calibration import CalibrationError, calibrate_session

        try:
            cal = calibrate_session(session)
            cal.save(session / "calibration.json")
            calibration = asdict(cal) | {"saved_to": str(session / "calibration.json")}
        except CalibrationError as err:
            calibration = {"rejected": err.issues}
    return {"session": str(session), "calibration": calibration, "framing": framing,
            "static": static, "blocks": blocks}  # fmt: skip


def format_report(report: dict[str, Any]) -> str:
    lines = [f"session {report['session']}", ""]
    cal = report.get("calibration") or {}
    if "rejected" in cal:
        lines += ["calibration REJECTED: " + "; ".join(cal["rejected"]), ""]
    elif cal:
        lines += [f"calibration: shoulders {cal['shoulder_width_m']} m, upper/fore "
                  f"{cal['upper_arm_ratio']}/{cal['forearm_ratio']}, neutral lean "
                  f"{cal['neutral_lean_deg']}°, head {cal['neutral_head_height']} "
                  f"-> {cal['saved_to']}", ""]  # fmt: skip
    lines.append("framing (share of samples well placed; issues in %):")
    for clip, f in report.get("framing", {}).items():
        issues = ", ".join(f"{k} {v}%" for k, v in f["issues"].items()) or "-"
        lines.append(f"  {clip:28s} ok {f['ok_pct']:5.1f}%   {issues}")
    lines.append("")
    for clip, s in report["static"].items():
        r = s["arms"]["right"]
        lines.append(f"{clip:18s} R upper {r['upper']}  fore {r['fore']}   "
                     f"lean {s['torso_lean_deg']}°  shoulders {s['shoulder_width_m']} m  "
                     f"upper/fore ratio {s['upper_arm_ratio']}/{s['forearm_ratio']}")  # fmt: skip
    for clip, b in report["blocks"].items():
        lag = "not measurable" if b["best_lag_ms"] is None else f"{b['best_lag_ms']:+.0f} ms"
        lines += ["", f"== {b['block_id']}: best lag {lag}, mean error "
                      f"{b['mean_error_deg']}° (at lag 0: {b['mean_error_at_zero_lag_deg']}°)"]  # fmt: skip
        if "torso_motion" in b:
            t = b["torso_motion"]
            lines.append("   torso: " + ", ".join(
                f"{k} {v['mean']:+.1f}±{v['std']:.1f}°" for k, v in t.items() if isinstance(v, dict)
            ) + f", hip sway ±{t['hip_sway_pct_of_width']}% of width")  # fmt: skip
            lines.append("   error by body frame: " + " | ".join(
                f"{m} {e}°" for m, e in b["frame_comparison"].items()))  # fmt: skip
        for f in b["keyframes"]:
            lines.append(
                f"   beat {f['beat']:<4} {f['side']:5s} upper {f['target_upper']} -> seen "
                f"{f['seen_upper']} ({f['error_upper_deg']:.0f}°, front {f['front_error_upper_deg']:.0f}°)"
                f" | fore {f['target_fore']} -> seen {f['seen_fore']} "
                f"({f['error_fore_deg']:.0f}°, front {f['front_error_fore_deg']:.0f}°)"
            )
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Compare recorded clips with the block library")
    parser.add_argument("session", nargs="?", type=Path,
                        help="recording session directory (default: latest in recordings/)")  # fmt: skip
    args = parser.parse_args(argv)
    session = args.session or latest_session()
    report = analyze_session(session)
    print(format_report(report))
    out = session / "analysis.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nreport written to {out}")
    return report


def main(argv: list[str] | None = None) -> None:
    """Console entry point. Returns ``None`` on purpose: console scripts call
    ``sys.exit(main())``, which would print any returned value and exit with
    status 1. Use :func:`run` to get the result programmatically."""
    run(argv)


if __name__ == "__main__":
    main()
