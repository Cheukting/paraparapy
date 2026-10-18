"""Session-wide guard for the cv2/pygame process boundary.

The test suite imports ``cv2``/``mediapipe`` in-process (capture tests) but
must never also import ``pygame`` in the same process: both bundle their own
``libSDL2`` (see tests/test_import_boundary.py). Tests that need pygame run it
in a subprocess. This hook fails the run if that rule is ever broken.
"""

import sys

import pytest


@pytest.fixture(autouse=True, scope="session")
def _never_load_cv2_and_pygame_together():
    yield
    both = {"cv2", "pygame"} <= sys.modules.keys()
    assert not both, (
        "cv2 and pygame were both imported into the pytest process; "
        "run pygame tests in a subprocess instead"
    )
