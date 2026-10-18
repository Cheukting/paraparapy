"""Score recorded block clips with the real scoring engine: ``uv run parapara-score``.

Runs every ``block_*.npz`` clip of a recording session through the same
pipeline the game will use (:class:`~parapara.pose.features.FeatureExtractor`
→ :class:`~parapara.scoring.judge.Judge` →
:class:`~parapara.scoring.scorekeeper.ScoreKeeper`), with the clip's
recorded beat times as the schedule. Useful to check scoring changes against
real dancing, and to compare players or sessions.

Latency: ``--latency-ms`` applies one fixed offset (as the game will, from the
clap-to-the-beat calibration); ``--auto-latency`` uses each clip's best lag
from ``parapara-analyze`` instead, to see how well the poses match once
timing is taken out of the picture.

Uses ``<session>/calibration.json`` if present (written by
``parapara-analyze``). Writes ``<session>/score.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parapara.diagnostics.pose_analysis import analyze_block, latest_session
from parapara.pose.calibration import Calibration
from parapara.pose.features import FeatureExtractor, extract_recording
from parapara.pose.recording import Recording
from parapara.routine.library import BLOCK_LIBRARY
from parapara.scoring.judge import judge_all
from parapara.scoring.schedule import schedule_block_clip
from parapara.scoring.scorekeeper import ScoreKeeper


def score_clip(rec: Recording, calibration: Calibration | None, latency_ms: float) -> dict[str, Any]:
    m = rec.metadata
    schedule = schedule_block_clip(BLOCK_LIBRARY[m["block_id"]], m["beat_times_ns"][m["count_in_beats"]:],
                                   int(latency_ms * 1e6))  # fmt: skip
    features = extract_recording(rec, FeatureExtractor(calibration=calibration))
    keeper = ScoreKeeper(calibrated=calibration is not None)
    for judgment in judge_all(schedule, features):
        keeper.add(judgment)
    for f in features:
        keeper.add_posture(f)
    return {"block_id": m["block_id"], "latency_ms": latency_ms, **keeper.summary()}


def score_session(session: Path, latency_ms: float = 0.0, auto_latency: bool = False,
                  calibration: Calibration | None = None) -> dict[str, Any]:  # fmt: skip
    if calibration is None and (session / "calibration.json").exists():
        calibration = Calibration.load(session / "calibration.json")
    clips = {}
    for path in sorted(session.glob("block_*.npz")):
        rec = Recording.load(path)
        lag = latency_ms
        if auto_latency:
            lag = analyze_block(rec).best_lag_ms or 0.0
        clips[path.stem] = score_clip(rec, calibration, lag)
    return {"session": str(session), "calibrated": calibration is not None, "clips": clips}


def format_scores(report: dict[str, Any]) -> str:
    lines = [f"session {report['session']} (calibrated: {report['calibrated']})", "",
             f"{'block':22s} {'latency':>8} {'accuracy':>9} {'score':>7}  grades (P/G/Gd/H/M)"
             f"   combo  flow  posture"]  # fmt: skip
    for c in report["clips"].values():
        g = c["grades"]
        flow = "-" if c["flow_pct"] is None else f"{c['flow_pct']:.0f}%"
        lines.append(f"{c['block_id']:22s} {c['latency_ms']:6.0f}ms {c['accuracy_pct']:8.1f}% {c['score']:7d}  "
                     f"{g['perfect']}/{g['great']}/{g['good']}/{g['hold']}/{g['miss']}"
                     f"   {c['max_combo']:5d}  {flow:>4}  {c['posture_pct']:5.1f}%")  # fmt: skip
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Score recorded block clips")
    parser.add_argument("session", nargs="?", type=Path, help="default: latest in recordings/")
    parser.add_argument("--latency-ms", type=float, default=0.0, help="fixed latency offset")
    parser.add_argument("--auto-latency", action="store_true", help="use each clip's analysed lag")
    args = parser.parse_args(argv)
    session = args.session or latest_session()
    report = score_session(session, args.latency_ms, args.auto_latency)
    print(format_scores(report))
    (session / "score.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nreport written to {session / 'score.json'}")
    return report


def main(argv: list[str] | None = None) -> None:
    """Console entry point (returns ``None``; see ``pose_analysis.main``)."""
    run(argv)


if __name__ == "__main__":
    main()
