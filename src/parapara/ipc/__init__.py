"""Inter-process plumbing between the game process and the capture process.

Pure Python + NumPy + ``multiprocessing``: must never import ``cv2``,
``mediapipe`` or ``pygame`` (enforced by tests/test_import_boundary.py), so
both processes can use it.

- :mod:`.messages`     -- capture configuration and status messages
- :mod:`.frame_buffer` -- latest camera frame in shared memory
- :mod:`.client`       -- game-side handle that starts and talks to the worker
"""
