"""PyInstaller runtime hook for ParaParaPy's multiprocessing start method.

The capture worker is included as a hidden import in ``paraparapy.spec`` and
is imported by the spawned child through ``parapara.ipc.client``. Do not
pre-import it here: that would load OpenCV and MediaPipe in the game process
before the UI appears.
"""

import multiprocessing

# Set the start method to 'spawn' for consistency
# This is important for macOS where fork can cause issues with pygame/SDL
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Already set, ignore
    pass
