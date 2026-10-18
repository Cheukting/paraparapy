"""pygame-ce screens (menu, calibration, play, results) and the main loop.

The ONLY subpackage allowed to import ``pygame``; must never import ``cv2`` or
``mediapipe`` (duplicate bundled libSDL2, see the package docstring). Roadmap Step 6.
"""
