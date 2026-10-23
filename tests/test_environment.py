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
    """The capture layer uses the Tasks API (PoseLandmarker) in VIDEO mode;
    legacy ``mp.solutions`` is not used."""
    out = _run(
        "from mediapipe.tasks.python import vision; "
        "print(vision.PoseLandmarker.__name__, vision.RunningMode.VIDEO.name)"
    )
    assert out == "PoseLandmarker VIDEO"


def test_pygame_is_community_edition() -> None:
    # The original ``pygame`` has no wheels for Python >= 3.13. ``IS_CE`` is an
    # int flag; the env var silences pygame's stdout banner on import.
    out = _run(
        "import os; os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'; "
        "import pygame; print(bool(getattr(pygame, 'IS_CE', False)))"
    )
    assert out == "True"


def test_console_entry_points_return_none() -> None:
    """Console scripts run ``sys.exit(main())``: a non-None return value is
    printed and turns into exit status 1, so every entry point must return
    ``None`` (regression: several returned their log/report path)."""
    import ast
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    scripts = tomllib.loads((root / "pyproject.toml").read_text())["project"]["scripts"]
    for name, target in scripts.items():
        module, func = target.split(":")
        source = (root / "src" / Path(*module.split("."))).with_suffix(".py")
        if not source.exists():
            source = root / "src" / Path(*module.split(".")) / "__init__.py"
        tree = ast.parse(source.read_text())
        (fn,) = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func
        ]
        returns = ast.unparse(fn.returns) if fn.returns else None
        assert returns == "None", f"{name}: {target} is annotated to return {returns}"
        assert all(
            r.value is None for r in ast.walk(fn) if isinstance(r, ast.Return)
        ), name
