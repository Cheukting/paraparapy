"""Webcam capture and MediaPipe pose/hand landmark detection.

Runs in a dedicated OS process. This is the ONLY subpackage allowed to import
``cv2`` and ``mediapipe``, and it must never import ``pygame`` (duplicate
bundled libSDL2, see the package docstring). Implemented in roadmap Step 2.
"""
