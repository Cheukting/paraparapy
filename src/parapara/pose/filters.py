"""One Euro filter: low-jitter, low-lag smoothing of landmark streams.

Algorithm: Casiez, Roussel & Vogel, "1€ Filter: A Simple Speed-based Low-pass
Filter for Noisy Input in Interactive Systems", CHI 2012
(https://gery.casiez.net/1euro/). Implemented here from the published
algorithm, vectorized so one filter smooths a whole ``(33, 3)`` landmark
array per sample. (The authors' reference code is BSD-3-Clause; none of it is
copied.)

How it works: an exponential low-pass filter whose cutoff frequency adapts
to the signal's speed::

    alpha(fc) = 1 / (1 + tau / Te),  tau = 1 / (2π fc),  Te = sample period
    dx        = (x - x̂_prev) / Te            # speed vs. previous *filtered* value
    dx̂        = lowpass(dx, alpha(d_cutoff))
    fc        = min_cutoff + beta · |dx̂|
    x̂         = lowpass(x, alpha(fc))

Holding still → low cutoff → strong smoothing (no jitter). Moving fast →
higher cutoff → little smoothing (little lag). Tuning (from the authors):
set ``beta = 0``, lower ``min_cutoff`` until jitter at rest is acceptable,
then raise ``beta`` until fast movements no longer lag.

Details:

- The first sample (and the first after a gap) passes through unchanged.
- Timestamps are nanoseconds (``time.monotonic_ns()``), so irregular frame
  timing (skipped camera frames) is handled exactly.
- Elements that are NaN (landmark missing / no person) reset just those
  elements; they restart from the next valid value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _alpha(cutoff_hz: npt.ArrayLike, dt_s: float) -> npt.NDArray[np.float64]:
    tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff_hz, dtype=np.float64))
    return 1.0 / (1.0 + tau / dt_s)


@dataclass
class OneEuroFilter:
    """Streaming One Euro filter over arrays of any (fixed) shape.

    Args:
        min_cutoff: cutoff frequency at rest, Hz (lower = smoother, laggier).
        beta: speed coefficient (higher = less lag when moving fast), in
            Hz per (signal unit / s).
        d_cutoff: cutoff for smoothing the speed estimate, Hz.
    """

    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        if self.min_cutoff <= 0 or self.d_cutoff <= 0 or self.beta < 0:
            raise ValueError("min_cutoff and d_cutoff must be > 0, beta >= 0")
        self.reset()

    def reset(self) -> None:
        self._x: npt.NDArray[np.float64] | None = None  # last filtered value
        self._dx: npt.NDArray[np.float64] | None = None  # last filtered speed
        self._t_ns: int | None = None

    def __call__(self, x: npt.ArrayLike, t_ns: int) -> npt.NDArray[np.float64]:
        """Filter one sample taken at ``t_ns``; returns the smoothed array."""
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t_ns is None or x.shape != self._x.shape:
            return self._restart(x, t_ns)
        dt = (t_ns - self._t_ns) / 1e9
        if dt <= 0:  # repeated / out-of-order timestamp: keep the last output
            return self._x.copy()

        # Elements that were missing restart from their new value.
        fresh = np.isnan(self._x) & ~np.isnan(x)
        prev = np.where(fresh, x, self._x)
        prev_dx = np.where(fresh | np.isnan(self._dx), 0.0, self._dx)

        dx = (x - prev) / dt
        edx = prev_dx + _alpha(self.d_cutoff, dt) * (dx - prev_dx)
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        out = prev + _alpha(cutoff, dt) * (x - prev)

        self._x, self._dx, self._t_ns = out, edx, t_ns  # NaN inputs stay NaN
        return out.copy()

    def _restart(
        self, x: npt.NDArray[np.float64], t_ns: int
    ) -> npt.NDArray[np.float64]:
        self._x, self._dx, self._t_ns = x.copy(), np.zeros_like(x), t_ns
        return x.copy()


def filter_series(
    values: npt.ArrayLike, times_ns: npt.ArrayLike, **params: float
) -> npt.NDArray[np.float64]:
    """Filter a whole recorded series offline (``values[i]`` at ``times_ns[i]``)."""
    f = OneEuroFilter(**params)
    return np.stack(
        [
            f(v, int(t))
            for v, t in zip(np.asarray(values), np.asarray(times_ns), strict=True)
        ]
    )
