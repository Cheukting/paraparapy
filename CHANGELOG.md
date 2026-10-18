# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `.gitignore` (Python caches, `.venv`, `.idea`, `.DS_Store`); initialized git repository.
- **Step 0 – project setup.**
  - `parapara` package skeleton with subpackages `routine`, `generator`,
    `capture`, `pose`, `scoring`, `audio`, `game`; each `__init__` documents its
    responsibility and roadmap step.
  - Runtime dependencies: `mediapipe` 1.1.1 (brings `opencv-contrib-python`
    5.0.0.93, `sounddevice`), `numpy` 2.5.4, `pygame-ce` 2.5.8.
    Dev dependency: `pytest` 9.1.1.
  - `tests/test_import_boundary.py`: static (AST) and runtime checks that only
    `capture` imports `cv2`/`mediapipe` and only `game` imports `pygame`.
  - `tests/test_environment.py`: smoke tests for the Python version, MediaPipe
    Tasks API (`PoseLandmarker`, `HandLandmarker`, `RunningMode.LIVE_STREAM`)
    and pygame-ce.
- `CHANGELOG.md`.

### Changed
- Renamed the package `pythonproject31` → `parapara` (entry point `parapara`);
  removed the PyCharm sample `src/main.py`.
- Lowered required Python from 3.15 to 3.13 (`.python-version`,
  `pyproject.toml`): `torch` and `onnxruntime` do not yet publish 3.15 wheels.
- Rewrote `PLAN.md`: desktop app; routines restructured into 8-count bars /
  32-count phrases (the original 8-count-with-all-five-blocks requirement was
  impossible); measurable quality requirements; architecture, scoring design,
  tech stack and roadmap.
- `PLAN.md` decisions: footwork is a visual guide only and scoring uses the
  upper body only (accessibility, seated play); single player for now, with a
  leaderboard and multiplayer as future work.
- `PLAN.md` architecture: capture runs in a separate OS process, because
  `opencv-contrib-python` 5.x and `pygame-ce` both bundle `libSDL2` and conflict
  when loaded into one process on macOS.
