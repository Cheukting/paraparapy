"""Para Para dance game: webcam pose tracking and rhythm scoring.

Package layout (see PLAN.md §4 for the full design):

- ``routine``   -- building blocks as timed keyframes; routine model and validation
- ``generator`` -- routine generator with difficulty levels
- ``capture``   -- camera + MediaPipe wrapper; runs in its own OS process
- ``pose``      -- landmark normalization, mirroring, feature extraction
- ``scoring``   -- keyframe matching, timing grades, posture/smoothness metrics
- ``audio``     -- song library, beat maps, playback clock
- ``game``      -- pygame-ce screens and the main loop

Process boundary
----------------
``opencv-contrib-python`` (a MediaPipe dependency) and ``pygame-ce`` each bundle
their own ``libSDL2``. Loading both into one process causes duplicate
Objective-C class warnings on macOS and possible crashes, so:

- only ``capture`` may import ``cv2`` / ``mediapipe``;
- only ``game`` may import ``pygame``;
- every other subpackage is pure Python + NumPy and is safe on either side.

``tests/test_import_boundary.py`` enforces these rules.
"""


def main() -> None:
    """Console entry point (``parapara``). The game loop arrives in Step 6."""
    print("parapara: game not implemented yet (see PLAN.md roadmap).")
