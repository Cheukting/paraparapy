"""The latest camera frame, shared between processes without copying it
through a pipe.

Why shared memory
-----------------
A 640×480 RGB frame is ~0.9 MB. Pickling 30 of those per second through a
``multiprocessing.Queue`` adds latency and CPU load, and the game only ever
needs the *newest* frame. So the capture process overwrites a single slot in
``multiprocessing.shared_memory`` and the game reads it when it draws.

Layout: a small header (``frame_index``, ``captured_ns`` as int64) followed by
one ``(height, width, 3)`` uint8 RGB frame.

Consistency: a ``multiprocessing.Lock`` guards each write and read, so a
reader never sees half of one frame and half of the next. Copying a frame
takes well under a millisecond, so contention is negligible.

Ownership: the game process *creates* the buffer (and unlinks it on close);
the capture process *attaches* with ``track=False`` (Python ≥ 3.13) so its
resource tracker does not delete the segment when the worker exits.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Self

import numpy as np
import numpy.typing as npt

_HEADER_FIELDS = 2  # frame_index, captured_ns
_HEADER_BYTES = _HEADER_FIELDS * np.dtype(np.int64).itemsize
_NO_FRAME = -1


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    """A copy of the frame that was in the buffer at read time."""

    frame_index: int
    captured_ns: int
    frame: npt.NDArray[np.uint8]  # (height, width, 3) RGB


class SharedFrameBuffer:
    """Single-slot, lock-protected frame buffer in shared memory.

    Use :meth:`create` in the owning (game) process and :meth:`attach` in the
    capture process; pass the ``name``, ``shape`` and the *same* lock object.
    """

    def __init__(
        self,
        shm: shared_memory.SharedMemory,
        shape: tuple[int, int, int],
        lock: Any,
        owner: bool,
    ) -> None:
        self._shm = shm
        self._lock = lock
        self._owner = owner
        self.shape = shape
        self._header = np.ndarray((_HEADER_FIELDS,), dtype=np.int64, buffer=shm.buf)
        self._frame = np.ndarray(
            shape, dtype=np.uint8, buffer=shm.buf, offset=_HEADER_BYTES
        )

    @staticmethod
    def nbytes(shape: tuple[int, int, int]) -> int:
        return _HEADER_BYTES + int(np.prod(shape))

    @classmethod
    def create(cls, shape: tuple[int, int, int], lock: Any) -> Self:
        """Allocate a new buffer (owner side). Starts with "no frame yet"."""
        shm = shared_memory.SharedMemory(create=True, size=cls.nbytes(shape))
        buffer = cls(shm, shape, lock, owner=True)
        buffer._header[:] = (_NO_FRAME, 0)
        return buffer

    @classmethod
    def attach(cls, name: str, shape: tuple[int, int, int], lock: Any) -> Self:
        """Attach to an existing buffer (capture side)."""
        shm = shared_memory.SharedMemory(name=name, track=False)
        if shm.size < cls.nbytes(shape):
            shm.close()
            raise ValueError(
                f"shared memory {name!r} is too small for frames of shape {shape}"
            )
        return cls(shm, shape, lock, owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    def write(
        self, frame: npt.NDArray[np.uint8], frame_index: int, captured_ns: int
    ) -> None:
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError(
                f"expected uint8 frame of shape {self.shape}, got {frame.dtype} {frame.shape}"
            )
        with self._lock:
            self._frame[...] = frame
            self._header[:] = (frame_index, captured_ns)

    def read(self, newer_than: int = _NO_FRAME) -> FrameSnapshot | None:
        """Copy out the current frame.

        Returns ``None`` if no frame has been written yet, or if the stored
        frame's index is not greater than ``newer_than`` (lets the caller skip
        re-uploading a frame it already has).
        """
        with self._lock:
            frame_index, captured_ns = (int(v) for v in self._header)
            if frame_index == _NO_FRAME or frame_index <= newer_than:
                return None
            frame = self._frame.copy()
        return FrameSnapshot(frame_index, captured_ns, frame)

    def close(self) -> None:
        """Detach; the owner also frees the shared memory segment."""
        # Drop NumPy views first: SharedMemory.close() fails while they exist.
        del self._header, self._frame
        self._shm.close()
        if self._owner:
            self._shm.unlink()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
