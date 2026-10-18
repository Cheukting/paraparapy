"""Enforce the capture/game process boundary described in PLAN.md §4.

Why this exists
---------------
``opencv-contrib-python`` 5.x (pulled in by ``mediapipe``) and ``pygame-ce``
each ship their own bundled ``libSDL2`` dylib. Importing both into a single
process makes macOS report duplicate Objective-C classes and warns of
"spurious casting failures and mysterious crashes". The design therefore runs
camera capture + MediaPipe in a separate OS process from the pygame game loop.

Rules
-----
- ``parapara.capture`` may import ``cv2`` / ``mediapipe``, never ``pygame``.
- ``parapara.game`` may import ``pygame``, never ``cv2`` / ``mediapipe``.
- Every other subpackage imports none of them, so it is safe on both sides.

Two complementary checks:

1. **Static (AST)**: scans every ``.py`` file, including imports nested inside
   functions, which a runtime import check would miss until the function runs.
2. **Runtime**: imports each subpackage in a *fresh* interpreter and inspects
   ``sys.modules``, which catches transitive imports via other modules.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "parapara"

CAPTURE_LIBS = frozenset({"cv2", "mediapipe"})
GAME_LIBS = frozenset({"pygame"})

# Forbidden top-level modules for each subpackage. The top-level package
# itself (``__init__.py`` directly under parapara/) is keyed as "".
FORBIDDEN: dict[str, frozenset[str]] = {
    "capture": GAME_LIBS,
    "game": CAPTURE_LIBS,
}
DEFAULT_FORBIDDEN = CAPTURE_LIBS | GAME_LIBS

SUBPACKAGES = sorted(
    p.name for p in PACKAGE_ROOT.iterdir() if (p / "__init__.py").is_file()
)


def _forbidden_for(subpackage: str) -> frozenset[str]:
    return FORBIDDEN.get(subpackage, DEFAULT_FORBIDDEN)


def _imported_top_level_modules(source: str) -> set[str]:
    """Return top-level names of every absolute import in ``source``."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_subpackages_discovered() -> None:
    """Guard against the test silently checking nothing (e.g. a moved package)."""
    assert {"capture", "game", "pose", "scoring"} <= set(SUBPACKAGES)


@pytest.mark.parametrize("py_file", sorted(PACKAGE_ROOT.rglob("*.py")), ids=str)
def test_static_import_boundary(py_file: Path) -> None:
    relative = py_file.relative_to(PACKAGE_ROOT)
    subpackage = relative.parts[0] if len(relative.parts) > 1 else ""
    imported = _imported_top_level_modules(py_file.read_text(encoding="utf-8"))
    violations = imported & _forbidden_for(subpackage)
    assert not violations, f"{relative} imports forbidden modules: {sorted(violations)}"


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_runtime_import_boundary(subpackage: str) -> None:
    probe = (
        "import json, sys, importlib; "
        f"importlib.import_module('parapara.{subpackage}'); "
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(result.stdout))
    violations = loaded & _forbidden_for(subpackage)
    assert not violations, (
        f"importing parapara.{subpackage} loads forbidden modules: {sorted(violations)}"
    )


def test_ast_helper_detects_nested_imports() -> None:
    """The static check must see imports hidden inside functions."""
    source = "def f():\n    import cv2\n    from pygame import mixer\n"
    assert _imported_top_level_modules(source) == {"cv2", "pygame"}
