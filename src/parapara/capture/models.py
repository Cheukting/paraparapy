"""MediaPipe model bundles: where to get them, and verified local caching.

Sources (MediaPipe docs, "Models" sections, checked 2026-10-18):

- https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/index#models
- https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index#models

The official URLs point at a ``latest`` path, so the content could change
without notice. Each bundle is therefore **pinned by SHA-256** (computed from
the files downloaded on 2026-10-18; the pose bundle's ``Last-Modified`` header
was 2023-04-28). A mismatch fails loudly instead of silently running a
different model that the scoring thresholds were not tuned for. To upgrade a
model deliberately, download it, update the hash and size here, and re-test.

Licensing: the model cards are linked from the pages above. Their terms must
be checked before the game is distributed (PLAN.md §10).

This module uses only the standard library (no ``cv2``/``mediapipe``).
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_BASE = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    filename: str
    url: str
    sha256: str
    size: int


MODELS: dict[str, ModelSpec] = {
    "pose_lite": ModelSpec(
        "pose_landmarker_lite.task",
        f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
        5_777_746,
    ),
    "pose_full": ModelSpec(
        "pose_landmarker_full.task",
        f"{_BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
        9_398_198,
    ),
    "hand": ModelSpec(
        "hand_landmarker.task",
        f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        7_819_105,
    ),
}


class ModelDownloadError(RuntimeError):
    pass


def default_model_dir() -> Path:
    """Model cache directory.

    ``$PARAPARA_MODEL_DIR`` if set; otherwise ``~/Library/Caches/parapara/models``
    on macOS and ``$XDG_CACHE_HOME/parapara/models`` (default ``~/.cache``)
    elsewhere. Models are kept out of the git repository (~23 MB).
    """
    if env := os.environ.get("PARAPARA_MODEL_DIR"):
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "parapara" / "models"
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "parapara" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(
    spec: ModelSpec,
    model_dir: Path | None = None,
    on_download: Callable[[ModelSpec], None] | None = None,
    timeout: float = 60.0,
) -> Path:
    """Return the path of a verified local copy of ``spec``, downloading it
    if it is missing or corrupt.

    The download goes to a temporary file in the same directory and is
    renamed into place only after the hash matches, so an interrupted or
    tampered download never leaves a bad file under the final name.

    Args:
        on_download: called before a download starts (e.g. to report
            progress to the game as "downloading models").

    Raises:
        ModelDownloadError: on network failure or hash/size mismatch.
    """
    directory = model_dir or default_model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec.filename

    if target.exists() and target.stat().st_size == spec.size and _sha256(target) == spec.sha256:
        return target

    if on_download is not None:
        on_download(spec)
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{spec.filename}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(spec.url, timeout=timeout) as resp:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
        actual = _sha256(tmp)
        if actual != spec.sha256:
            raise ModelDownloadError(
                f"{spec.filename}: SHA-256 mismatch (expected {spec.sha256}, got {actual}). "
                "The upstream 'latest' model may have changed; see parapara/capture/models.py."
            )
        os.replace(tmp, target)
    except OSError as err:  # includes urllib.error.URLError
        raise ModelDownloadError(f"could not download {spec.url}: {err}") from err
    finally:
        tmp.unlink(missing_ok=True)
    return target
