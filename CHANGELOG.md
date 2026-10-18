# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `parapara.diagnostics` (pure; no cv2/mediapipe/pygame):
  - `live_stats`: HUD statistics, moved from `game.overlay`, plus the new
    `CameraRateMeter` (camera delivery rate and share of skipped frames, to
    tell a camera-limited frame rate from an inference-limited one).
  - `recorder`: `RunRecorder` whole-run summaries (rates, mean/p50/p95
    timings, detection rates, warm-up excluded) and `write_log` to
    `logs/*.json` with machine and library versions.
  - `bench`: headless benchmark (`python -m parapara.diagnostics.bench`) that
    runs several capture configurations back to back (default `full+hands`,
    `full`, `lite+hands`) and logs a comparison.
- `parapara.game.capture_bench`: `uv run parapara-capture-bench` runs the same
  benchmark **with the live camera view** (skeleton, HUD, step and countdown)
  so the dancer gets feedback; `Esc` stops early and still logs.
- `debug_view.run_session`: the live-view loop, factored out so the debug view
  and the windowed benchmark share it.
- Debug view writes a session summary to `logs/capture-debug-*.json` on exit
  (`--no-log`, `--log-dir`).
- `logs/` added to `.gitignore`. Tests: `test_diagnostics.py`.
- `PLAN.md`: first human test results: left/right labels confirmed, tracking
  at 16–18 fps (target ~30).
- **Step 2 – pose capture prototype.**
  - `parapara.capture` (capture process only): `models` (pinned SHA-256
    MediaPipe bundles, verified download to `~/Library/Caches/parapara/models`),
    `camera` (newest-frame grabber thread), `detector` (Pose + Hand
    Landmarker, Tasks API, VIDEO mode, strictly increasing timestamps),
    `worker` (process entry point and testable `capture_loop`).
  - `parapara.ipc` (both processes): `CaptureClient` (spawned worker started
    by module name so the game never imports cv2), `SharedFrameBuffer`
    (lock-protected shared-memory frame slot), `CaptureConfig` /
    `CaptureStatus` messages.
  - `parapara.pose.landmarks`: `PoseSample`, `HandObservation`, landmark
    indices and connections, and hand-to-wrist left/right assignment.
  - `parapara.game.debug_view` + `overlay`: live debug window
    (`uv run parapara-capture-debug`) with the upper-body skeleton, hands,
    tracking FPS, inference times, latency and dropped samples.
  - Tests: `test_pose_landmarks.py`, `test_ipc.py` (real spawned processes
    with fake workers), `test_capture.py` (including a real-MediaPipe test),
    `test_game_overlay.py` (headless pygame in subprocesses), plus a
    `conftest.py` guard that cv2 and pygame never load into one process.
- `PLAN.md`: §4.2 capture pipeline, step 4b real-dancer test run, open items.
- **Step 1 – routine model** (`parapara.routine`).
  - `geometry`: body-frame vectors (`+x` dancer's right, `+y` up, `+z`
    forward), `unit`, `mirror_x`, `angle_between_deg`. `normalize` leaves
    unit vectors bit-for-bit unchanged so that mirroring twice is exact.
  - `model`: `ArmPose` (upper-arm / forearm / optional hand directions + hand
    shape), `Keyframe`, `Block`, `Phrase`, `Routine`, `TimedKeyframe`. Every
    object validates on construction and reports all issues together in a
    `RoutineError`. Phrase rules: all five kinds, ends with Flick + Clap,
    32 counts, no block crosses a bar line, blocks start on a multiple of their
    length. `Routine.timeline()` yields every keyframe with absolute beat,
    bar and count.
  - `library`: **draft** definitions of 18 built-in blocks, checked against
    the EKS-D "How to Dance" guide (kirakira.org): Sweep ×10 (horizontal,
    vertical, diagonal), Knock ×2 (same hand twice, upper arm slightly
    forward), Chop ×2 (45° up, then a salute fold in front of the chest with
    the flat hand level), Wrist Roll ×2 (not in the reference, unverified),
    Flick + Clap ×2 (one arm travelling across the body; the clap is scored
    only as a clap, regardless of side). Left variants are generated
    by mirroring. `geometry.slerp` interpolates sweep directions.
  - `serialization`: JSON format v1, with routines referencing blocks by id.
  - `assets/routines/demo-normal.json`: hand-written two-phrase sample routine.
  - Tests: `test_routine_geometry.py`, `test_routine_model.py`,
    `test_routine_library.py`, `test_routine_serialization.py`.
- `PLAN.md` §1: move descriptions, footwork counts and right-hand lead
  updated from the EKS-D reference; pairing patterns noted for the generator.
  §7: the demo is shown mirrored.
- `PLAN.md` §3: arm targets are direction vectors; body frame defined; player
  calibration step (neutral pose + T-pose) added. §2: fixed the
  self-contradictory bar-alignment rule.
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

### Removed
- **Hand tracking** (MediaPipe Hand Landmarker), decided after the first
  capture benchmark (2026-10-18): it cost ~55 ms per frame (tracking fell from
  23 to 12 fps, camera→screen latency from 41 to 114 ms) and found each hand
  only ~20% of the time. Removed the hand model download, `HandObservation`,
  `assign_hands_to_wrists`, `CaptureConfig.detect_hands`, the `--no-hands`
  option, hand drawing and the `+hands` benchmark configurations. The debug
  view now draws the pose model's own hand points (index/pinky/thumb).

### Changed
- Claps will be detected **by sound**, confirmed by the wrists meeting
  (`PLAN.md` §5.1). `HandShape` is now descriptive only (demo animation), and
  `ArmPose.hand` is scored from the pose model's hand points.
- Capture benchmark defaults to the `full` and `lite` pose models.
- `PLAN.md`: benchmark results recorded, hand-landmark column removed from the
  block table, tech stack updated, open items re-sorted.
- `tests/test_import_boundary.py` runtime check now imports every module of
  each subpackage, not just its `__init__`.
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
