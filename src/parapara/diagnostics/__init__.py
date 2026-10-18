"""Capture performance diagnostics: live HUD statistics, run recording and
the capture benchmark (``uv run parapara-capture-bench``).

Pure Python + NumPy + ``multiprocessing``: never imports ``cv2``,
``mediapipe`` or ``pygame`` (enforced by tests/test_import_boundary.py).
"""
