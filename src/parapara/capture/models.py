"""MediaPipe model bundles: where to get them, and verified local caching.

Sources (MediaPipe docs, "Models" sections, checked 2026-10-18):

- https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/index#models

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
import socket
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
}


class ModelDownloadError(RuntimeError):
    """Raised when a model cannot be downloaded or verified.

    The error message is shown directly to the user in the game UI, so it
    should be actionable. When the network is unavailable and no cached model
    exists, the message explains the situation and what to do next.
    """


def default_model_dir() -> Path:
    """Model cache directory.

    ``$PARAPARA_MODEL_DIR`` if set; otherwise ``~/Library/Caches/parapara/models``
    on macOS and ``$XDG_CACHE_HOME/parapara/models`` (default ``~/.cache``)
    elsewhere. Models are kept out of the git repository (~15 MB).
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


def _is_network_available(timeout: float = 5.0) -> bool:
    """Check whether the network is reachable by probing a well-known host.

    This is a best-effort check used only to improve the error message when
    a model download fails. A ``True`` result does not guarantee that the
    model URL is reachable, and a ``False`` result does not guarantee that
    no network path exists (e.g. behind a proxy).
    """
    try:
        socket.create_connection(("storage.googleapis.com", 443), timeout=timeout)
        return True
    except OSError:
        return False


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
        The error message is user-facing and actionable.
    """
    directory = model_dir or default_model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec.filename

    if (
        target.exists()
        and target.stat().st_size == spec.size
        and _sha256(target) == spec.sha256
    ):
        return target

    if on_download is not None:
        on_download(spec)
    fd, tmp_name = tempfile.mkstemp(
        dir=directory, prefix=f".{spec.filename}.", suffix=".part"
    )
    tmp = Path(tmp_name)
    try:
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(spec.url, timeout=timeout) as resp,
        ):
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
        actual = _sha256(tmp)
        if actual != spec.sha256:
            raise ModelDownloadError(
                f"{spec.filename}: SHA-256 mismatch (expected {spec.sha256}, got {actual}). "
                "The upstream 'latest' model may have changed; see parapara/capture/models.py."
            )
        os.replace(tmp, target)
    except ModelDownloadError:
        # Re-raise unchanged (e.g. hash mismatch)
        raise
    except OSError as err:
        # Build a user-friendly message that explains the situation and what
        # to do next. The message is displayed in the game UI as a camera
        # problem, so it must be actionable without technical jargon.
        original_err = err
        if not _is_network_available():
            raise ModelDownloadError(
                f"Cannot download pose model '{spec.filename}' — no internet connection.\n\n"
                "ParaParaPy needs an internet connection on the first run to download "
                "the pose detection models (~15 MB). Once downloaded, they are cached "
                "and the game works offline.\n\n"
                "Please connect to the internet and try again."
            ) from original_err
        else:
            raise ModelDownloadError(
                f"Cannot download pose model '{spec.filename}': {err}\n\n"
                "Check your internet connection and try again. If the problem persists, "
                f"you can download the model manually from:\n{spec.url}"
            ) from original_err
    finally:
        tmp.unlink(missing_ok=True)
    return target
