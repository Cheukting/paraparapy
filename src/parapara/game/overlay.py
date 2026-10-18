"""Pure helpers for drawing landmark overlays.

HUD statistics live in :mod:`parapara.diagnostics.live_stats`.

Kept free of ``pygame`` so they can be unit-tested in the main pytest process
(which also imports ``cv2`` for the capture tests; the two must never be
loaded together, see the package docstring).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

VISIBILITY_THRESHOLD = 0.5


def to_screen(
    points: npt.ArrayLike, size: tuple[int, int], mirrored: bool
) -> list[tuple[int, int]]:
    """Normalized image landmarks (``x, y`` in ``[0, 1]``) → pixel positions.

    ``mirrored`` flips x, matching a horizontally flipped camera picture, so
    that the overlay stays on top of the body in a "selfie" view.
    """
    xy = np.asarray(points, dtype=np.float64)[:, :2].copy()
    if mirrored:
        xy[:, 0] = 1.0 - xy[:, 0]
    width, height = size
    xy *= (width, height)
    return [(round(x), round(y)) for x, y in xy]


def visible(pose: npt.NDArray[np.float32], index: int, threshold: float = VISIBILITY_THRESHOLD) -> bool:
    """Whether a pose landmark is confidently visible (NaN counts as not)."""
    v = float(pose[index, 3])
    return v == v and v >= threshold  # v == v is False for NaN
