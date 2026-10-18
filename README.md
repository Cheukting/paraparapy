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
