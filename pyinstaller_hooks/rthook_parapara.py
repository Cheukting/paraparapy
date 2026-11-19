"""PyInstaller runtime hook for ParaParaPy.

This hook runs early in the frozen app's lifecycle to ensure that the capture
worker module can be imported when the child process spawns. In a frozen app,
the worker is not a separate file on disk — it's part of the bundled archive,
so the bootstrap mechanism in ``parapara.ipc.client`` needs the module to be
findable via ``importlib.import_module``.

PyInstaller handles this automatically for most cases, but this hook adds
explicit support for the worker module and sets up the multiprocessing
context correctly.
"""

import multiprocessing
import sys


def ensure_worker_importable():
    """Ensure the capture worker module is importable in frozen apps.

    In a frozen PyInstaller app, the worker process is spawned via
    ``multiprocessing.Process`` with the ``spawn`` start method. The bootstrap
    function in ``parapara.ipc.client._bootstrap`` imports the worker module
    by name, so it must be in sys.modules or findable via importlib.

    PyInstaller should have already bundled the module, but this hook
    pre-imports it to ensure it's available.
    """
    try:
        import parapara.capture.worker  # noqa: F401
    except ImportError as e:
        # Log the error but don't fail — the worker may not be needed
        # (e.g., running CLI tools instead of the game)
        print(f"Warning: Could not pre-import capture worker: {e}", file=sys.stderr)


# Set the start method to 'spawn' for consistency
# This is important for macOS where fork can cause issues with pygame/SDL
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Already set, ignore
    pass

# Pre-import the worker module to ensure it's available
ensure_worker_importable()
