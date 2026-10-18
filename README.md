# parapara

A desktop Para Para dance game: a webcam tracks the player's upper body and
scores each move against a choreographed routine. See [PLAN.md](PLAN.md) for
the design and roadmap, and [CHANGELOG.md](CHANGELOG.md) for progress.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```sh
uv sync
uv run pytest
```

## Capture debug view

```sh
uv run parapara-capture-debug            # --help for options (--model lite, ...)
```

On first run it downloads ~15 MB of MediaPipe pose models to
`~/Library/Caches/parapara/models`. macOS will ask for camera access for your
terminal or IDE. Cyan = dancer's left, orange = dancer's right; `M` toggles
the mirrored view, `Esc` quits.

## Capture benchmark

```sh
uv run parapara-capture-bench            # ~1 min: dance in front of the camera
```

Shows the live camera view while it runs the full and lite pose models, then
writes a comparison to
`logs/capture-bench-*.json`. `Esc` stops early. A version without a window:
`uv run python -m parapara.diagnostics.bench`.

## Guided recorder

```sh
uv run parapara-record                   # --bpm 100 for a slower metronome
```

Walks you through calibration poses and one clip per building block, with a
guide figure and a metronome. Clips are saved (landmarks only, no images) to
`recordings/<date-time>/`.

## Pose analysis / recalibration

```sh
uv run parapara-analyze                  # latest session in recordings/, or pass a path
```

Compares the recorded clips with the block library (danced vs. target arm
directions, best lag, calibration values) and writes `analysis.json` into the
session folder. Use it to recalibrate `src/parapara/routine/library.py`. If
the session has neutral-pose and T-pose clips it also saves the player
calibration (`calibration.json`).

```sh
uv run parapara-tune-filters             # re-tune the smoothing filters on a session
```

## Scoring recorded clips

```sh
uv run parapara-score                    # latest session; --latency-ms N or --auto-latency
```

Runs the block clips through the real scoring engine (grades, accuracy,
combo, flow, posture) and writes `score.json` into the session folder.
