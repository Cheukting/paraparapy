"""Smoke tests for the Step 0 environment (PLAN.md §8).

These fail early and loudly if a dependency upgrade changes the APIs or
versions the design relies on. Each library is checked in a fresh
interpreter, so ``cv2`` and ``pygame`` are never loaded into the same
process (see test_import_boundary.py).
"""

from __future__ import annotations

import subprocess
import sys

import parapara


def _run(code: str) -> str:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_python_version() -> None:
    # torch / onnxruntime lack 3.15 wheels; the project targets 3.13+.
    assert sys.version_info >= (3, 13)


def test_entry_point_exists() -> None:
    assert callable(parapara.main)


def test_mediapipe_tasks_api_available() -> None:
    """The capture layer uses the Tasks API (PoseLandmarker / HandLandmarker)
    in VIDEO or LIVE_STREAM mode; legacy ``mp.solutions`` is not used."""
    out = _run(
        "from mediapipe.tasks.python import vision; "
        "print(vision.PoseLandmarker.__name__, vision.HandLandmarker.__name__, "
        "vision.RunningMode.LIVE_STREAM.name)"
    )
    assert out == "PoseLandmarker HandLandmarker LIVE_STREAM"


def test_pygame_is_community_edition() -> None:
    # The original ``pygame`` has no wheels for Python >= 3.13. ``IS_CE`` is an
    # int flag; the env var silences pygame's stdout banner on import.
    out = _run(
        "import os; os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'; "
        "import pygame; print(bool(getattr(pygame, 'IS_CE', False)))"
    )
    assert out == "True"
