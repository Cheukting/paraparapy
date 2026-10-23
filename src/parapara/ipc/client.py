"""Game-side handle for the capture process.

:class:`CaptureClient` starts the capture worker in a separate OS process,
owns the shared frame buffer and the result queue, and gives the game loop
three non-blocking calls: :meth:`~CaptureClient.poll` (new samples and status
messages), :meth:`~CaptureClient.latest_frame` and
:meth:`~CaptureClient.stop`.

Keeping ``cv2`` out of the game process
---------------------------------------
``multiprocessing`` pickles the ``Process`` target by reference, which would
make the *parent* import the worker module (and with it ``cv2``). Instead the
target is the tiny :func:`_bootstrap` trampoline defined here, and the worker
is named by a ``"module:function"`` string that is only imported inside the
child process. The same mechanism lets tests substitute a fake worker.

The ``spawn`` start method is used on every platform: it is the macOS
default, and a fresh interpreter avoids inheriting pygame/SDL state that
``fork`` would copy.

Worker contract
---------------
The target is called with keyword arguments ``config`` (CaptureConfig),
``results`` (Queue), ``stop_event`` (Event), ``frame_buffer_name`` (str),
``frame_shape`` (tuple) and ``frame_lock`` (Lock). It must put
:class:`~parapara.pose.landmarks.PoseSample` and
:class:`~parapara.ipc.messages.CaptureStatus` objects on ``results`` and
return soon after ``stop_event`` is set.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import queue
from typing import Any, Self

from parapara.ipc.frame_buffer import FrameSnapshot, SharedFrameBuffer
from parapara.ipc.messages import CaptureConfig, CaptureStatus
from parapara.pose.landmarks import PoseSample

DEFAULT_WORKER = "parapara.capture.worker:run"

type CaptureMessage = PoseSample | CaptureStatus


def _bootstrap(target: str, kwargs: dict[str, Any]) -> None:
    """Child-process entry point: import ``module:function`` and call it."""
    module_name, _, func_name = target.partition(":")
    func = getattr(importlib.import_module(module_name), func_name)
    func(**kwargs)


class CaptureClient:
    """Starts, talks to, and stops the capture worker process.

    Usage::

        with CaptureClient(CaptureConfig()) as capture:
            while running:
                for message in capture.poll():
                    ...
                snapshot = capture.latest_frame(newer_than=shown_index)

    Args:
        config: capture settings passed to the worker.
        worker: ``"module:function"`` of the worker to run.
        queue_size: maximum queued messages. The worker drops samples rather
            than blocking when the game falls behind; stale poses are useless
            for a rhythm game anyway.
    """

    def __init__(
        self, config: CaptureConfig, worker: str = DEFAULT_WORKER, queue_size: int = 64
    ) -> None:
        self.config = config
        self._worker = worker
        self._queue_size = queue_size
        self._ctx = mp.get_context("spawn")
        self._process: mp.process.BaseProcess | None = None
        self._results: Any = None
        self._stop_event: Any = None
        self._frames: SharedFrameBuffer | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("capture already started")
        lock = self._ctx.Lock()
        self._frames = SharedFrameBuffer.create(self.config.frame_shape, lock)
        self._results = self._ctx.Queue(maxsize=self._queue_size)
        self._stop_event = self._ctx.Event()
        kwargs = {
            "config": self.config,
            "results": self._results,
            "stop_event": self._stop_event,
            "frame_buffer_name": self._frames.name,
            "frame_shape": self.config.frame_shape,
            "frame_lock": lock,
        }
        self._process = self._ctx.Process(
            target=_bootstrap,
            args=(self._worker, kwargs),
            name="parapara-capture",
            daemon=True,
        )
        self._process.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Ask the worker to finish, wait for it, then release resources.

        A worker that does not exit within ``timeout`` seconds is terminated.
        Safe to call more than once.
        """
        if self._process is None:
            return
        self._stop_event.set()
        self._process.join(timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(1.0)
        # Don't block interpreter exit on undelivered queue items.
        self._results.cancel_join_thread()
        self._results.close()
        assert self._frames is not None
        self._frames.close()
        # Clear every handle so poll()/latest_frame() after stop() return
        # "nothing" instead of touching a closed queue or freed memory.
        self._process = None
        self._frames = None
        self._results = None
        self._stop_event = None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def exitcode(self) -> int | None:
        return None if self._process is None else self._process.exitcode

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- per-frame calls (non-blocking) --------------------------------------

    def poll(self, max_messages: int = 256) -> list[CaptureMessage]:
        """Return all messages received since the last call, oldest first."""
        messages: list[CaptureMessage] = []
        if self._results is None:
            return messages
        for _ in range(max_messages):
            try:
                messages.append(self._results.get_nowait())
            except queue.Empty:
                break
        return messages

    def latest_frame(self, newer_than: int = -1) -> FrameSnapshot | None:
        """Copy of the newest camera frame, or ``None`` if nothing newer."""
        if self._frames is None:
            return None
        return self._frames.read(newer_than)
