"""Per-machine player settings kept between sessions.

Stored as JSON in the user data folder (next to the user song library):
``$PARAPARA_DATA_DIR``, else ``~/Library/Application Support/parapara`` on
macOS, else ``$XDG_DATA_HOME/parapara`` (``~/.local/share``).

- ``latency.json``: the last accepted latency calibration (depends on this
  machine's speakers/camera and the player, so it is offered for reuse);
- ``calibration.json``: the last player calibration (neutral pose + T-pose);
- ``ui.json``: display preferences (text size, chosen from the dancing spot).

Pure Python (no cv2/mediapipe/pygame).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from parapara.pose.calibration import Calibration


def data_dir() -> Path:
    if env := os.environ.get("PARAPARA_DATA_DIR"):
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "parapara"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "parapara"


def _path(name: str) -> Path:
    return data_dir() / name


def save_latency(latency_ms: float, spread_ms: float, claps: int) -> None:
    _path("latency.json").parent.mkdir(parents=True, exist_ok=True)
    _path("latency.json").write_text(json.dumps({
        "latency_ms": latency_ms, "spread_ms": spread_ms, "claps": claps,
        "measured": datetime.now().isoformat(timespec="seconds"),
    }, indent=2) + "\n")  # fmt: skip


def load_latency() -> dict[str, Any] | None:
    path = _path("latency.json")
    return json.loads(path.read_text()) if path.exists() else None


def save_calibration(calibration: Calibration) -> None:
    calibration.save(_path("calibration.json"))


def load_calibration() -> Calibration | None:
    path = _path("calibration.json")
    try:
        return Calibration.load(path) if path.exists() else None
    except (ValueError, KeyError, TypeError):
        return None  # an old or broken file: calibrate again


def save_ui(**prefs: Any) -> None:
    current = load_ui()
    _path("ui.json").parent.mkdir(parents=True, exist_ok=True)
    _path("ui.json").write_text(json.dumps(current | prefs, indent=2) + "\n")


def load_ui() -> dict[str, Any]:
    path = _path("ui.json")
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return {}
