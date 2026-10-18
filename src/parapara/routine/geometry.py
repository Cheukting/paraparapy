"""Tiny 3-D vector helpers for authoring and comparing keyframe directions.

Body frame (PLAN.md §3) -- dancer's own anatomy:

- ``+x``: towards the dancer's **right** hand
- ``+y``: up
- ``+z``: forward, out of the dancer's chest (towards the camera)

Handedness: physically this (right, up, forward) frame is **left-handed**
(right × up points *backwards*), so when the frame is built from landmarks,
forward must be computed as ``up × right``. Measured on 2026-10-18
recordings, MediaPipe's world landmarks use the opposite direction on every
axis: ``+x`` = dancer's left, ``+y`` = down, ``+z`` = away from the camera,
which is a physically right-handed frame.

Vectors are plain ``tuple[float, float, float]`` so that keyframes stay
immutable, hashable and trivially JSON-serializable. Scoring code that needs
speed converts them to NumPy arrays in bulk; this module only deals with a
handful of hand-authored vectors and so stays dependency-free.
"""

from __future__ import annotations

import math

type Vec3 = tuple[float, float, float]

# Named directions in the body frame, for readable keyframe authoring.
RIGHT: Vec3 = (1.0, 0.0, 0.0)
LEFT: Vec3 = (-1.0, 0.0, 0.0)
UP: Vec3 = (0.0, 1.0, 0.0)
DOWN: Vec3 = (0.0, -1.0, 0.0)
FORWARD: Vec3 = (0.0, 0.0, 1.0)

_EPSILON = 1e-9


def unit(x: float, y: float, z: float) -> Vec3:
    """Return ``(x, y, z)`` scaled to length 1.

    Lets authors write intuitive un-normalized directions such as
    ``unit(1, 1, 0)`` for "45° up and out to the right".

    Raises:
        ValueError: for a (near-)zero vector, which has no direction.
    """
    length = math.sqrt(x * x + y * y + z * z)
    if length < _EPSILON:
        raise ValueError("cannot normalize a zero-length vector")
    if abs(length - 1.0) < _EPSILON:
        # Already unit length: return unchanged. Re-dividing would perturb the
        # last bits, so e.g. mirror(mirror(pose)) would no longer equal pose
        # exactly, breaking equality-based symmetry checks.
        return (float(x), float(y), float(z))
    return (x / length, y / length, z / length)


def normalize(v: Vec3) -> Vec3:
    """Return ``v`` scaled to length 1 (see :func:`unit`)."""
    return unit(*v)


def mirror_x(v: Vec3) -> Vec3:
    """Reflect ``v`` through the body's sagittal plane (negate ``x``).

    Combined with swapping the left/right arms, this turns a right-handed
    move into the equivalent left-handed one.
    """
    return (-v[0], v[1], v[2])


def slerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    """Spherical interpolation between directions ``a`` (t=0) and ``b`` (t=1).

    Used to author sweeps: the arm direction turns at a constant angular
    speed along the shortest arc between the start and end directions.
    ``a`` and ``b`` must not be opposite (the shortest arc is undefined).
    """
    a_n, b_n = normalize(a), normalize(b)
    omega = math.radians(angle_between_deg(a_n, b_n))
    if omega < _EPSILON:
        return a_n
    if math.pi - omega < 1e-6:
        raise ValueError("cannot slerp between opposite directions")
    s = math.sin(omega)
    wa, wb = math.sin((1 - t) * omega) / s, math.sin(t * omega) / s
    return normalize(tuple(wa * p + wb * q for p, q in zip(a_n, b_n, strict=True)))  # type: ignore[arg-type]


def angle_between_deg(a: Vec3, b: Vec3) -> float:
    """Angle in degrees between two non-zero vectors, in ``[0, 180]``."""
    a_n, b_n = normalize(a), normalize(b)
    dot = sum(p * q for p, q in zip(a_n, b_n, strict=True))
    # Clamp: rounding can push |dot| marginally above 1, outside acos' domain.
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))
