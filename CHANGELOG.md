# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Step 5 – music library & beat maps** (`parapara.audio`).
  - `synth`: procedurally generated "Eurobeat-like" demo track (150 BPM,
    intro / 64-count main section / outro; kick, off-beat hats, claps on 2 and
    4, octave bass, saw-lead arpeggios), deterministic, exact beat map.
    `uv run parapara-make-demo-song` writes `assets/songs/demo/demo.wav`
    (git-ignored, ~3 MB) and `song.json` (committed).
  - `song`: `song.json` format v1 with mandatory licence and source, beat
    grid (`bpm`, `first_beat_s`, `routine_start_beat`), beat-map status;
    validated loading, warnings for unverified maps and tempos outside
    140-160 BPM; `Song.routine_clock` gives the scoring `BeatClock`.
  - `beatmap` / `uv run parapara-beatmap <song dir>`: librosa 1.0 beat
    tracking on the **kick band** (the full spectrum locked onto Eurobeat's
    off-beat hats/bass: ~150 ms errors), constant-tempo least-squares fit
    using the **mean** beat spacing (frame quantization biased the median),
    detection-lag correction. On the demo track: 150.007 BPM (true 150),
    first beat 501.7 ms (true 500), drift 4 ms.
  - `beatgrid` + `uv run parapara-beatcheck [song dir]`: guided by-ear check
    with sample-accurate clicks mixed into the audio, a kick-energy strip
    under the beat grid, nudge keys, and save-as-verified.
  - librosa added in a `tools` dependency group (installed by default for
    development; not needed to play).
  - Tests: `test_audio.py`.
- **Step 4 – scoring engine** (`parapara.scoring`).
  - `schedule`: `BeatClock` (tempo, first beat, latency offset),
    `schedule_routine`, `schedule_block_clip`; repeated poses are marked as
    holds.
  - `matching`: front-weighted angular error (depth counts half), pose score
    from block tolerances, clap score from wrist distance.
  - `judge`: streaming `Judge` emitting a judgment as soon as each window
    closes: best match in the window with sub-sample time refinement,
    Perfect / Great / Good / Hold / Miss, claps (sound onset + vision gate, or
    vision only), and a "keep moving" flow check for flowing blocks.
  - `scorekeeper`: points × timing factor × combo multiplier, flow and
    posture bonuses, accuracy, grade counts.
  - `uv run parapara-score [session] [--latency-ms N | --auto-latency]`
    scores recorded clips with the real pipeline and writes `score.json`.
  - `PoseFeatures` gained body-relative wrist positions.
  - Tests: `test_scoring.py` (synthetic on-beat, late and still dancers;
    clap onsets; streaming; combo); the synthetic dancer helper moved to
    `tests/pose_synth.py`.
- `PLAN.md` §5 rewritten to match the implemented scoring, with the findings
  from scoring the recordings.
- **Step 3 – pose features.**
  - `parapara.pose.filters.OneEuroFilter`: vectorized One Euro filter
    (Casiez et al., CHI 2012), with ns timestamps and per-element reset on
    missing landmarks.
  - `parapara.pose.calibration`: per-player calibration from the neutral pose
    and T-pose: shoulder width, arm segment ratios, neutral torso "up" (camera
    angle), neutral lean and head height. Validated, with the arm checks in
    the front plane (depth bias made hanging forearms read ~40° forward) and
    framing required. JSON save/load.
  - `parapara.pose.features.FeatureExtractor`: streaming pipeline (filtered
    landmarks → body frame → adaptive torso smoothing → arm directions,
    wrist distance in shoulder widths, lean / side tilt / shoulder tilt / head
    height, relative to the calibration).
  - Filter settings tuned on `recordings/20261018-144829`: landmarks
    `min_cutoff 0.5, beta 10` (forearm jitter at rest 1.19° → 0.47° per
    frame, ~28 ms lag); torso `min_cutoff 0.5, beta 0.5` (a fixed slow torso
    filter lagged sweep turns: 18° → 26–32° error).
  - `uv run parapara-tune-filters [session]` repeats the tuning on new
    recordings; `parapara-analyze` now also computes and saves
    `calibration.json` and reports the full pipeline (`torso+filters`) in the
    frame comparison.
  - Tests: `test_filters.py`, `test_features_calibration.py`.
- **Position (framing) check** (`parapara.pose.framing`): torso and hips in
  the picture, room to stretch the arms sideways and upwards (1.3 shoulder
  widths, measured in the T-pose), not too far, centred, facing the camera.
  It gives instructions in the player's own left/right ("Move a little to
  your LEFT", "Step back until your hips are in the picture"); thresholds
  come from the 2026-10-18 recordings. `FramingMonitor` debounces it.
  - Recorder: the check is shown before every clip with a guide rectangle
    on the camera view; calibration and axis clips only start once the
    position has been OK for ~0.4 s (`--no-framing-check`); move clips warn
    while recording if the hips leave the picture.
  - `parapara-analyze` reports framing per clip (share of samples OK,
    issues). It revealed that the hips were out of the picture throughout
    the second recording session.
  - Tests: `test_framing.py`; `pose_synth.image_pose`; the streaming fake
    worker now sends a well-framed person.
- Body movement analysis: `body_frame.FRAME_METHODS` (`torso` default,
  `camera`, `yaw`, `hips`, `blend`) and `torso_motion` (turn / side tilt /
  lean). `parapara-analyze` now reports torso motion and hip sway per block
  clip and compares the mean error under every frame variant, so the choice
  of body frame can be re-checked on new recordings.
- Recorder **move preview**: while waiting for Space, the guide figure plays
  the move in a loop at the clip's tempo with the count shown
  (`ClipTimeline.preview_at`). During recording the guide stays **on the
  beat**: the look-ahead introduced earlier now defaults to 0 (`--lookahead`
  still enables it).
- The preview loop is **audible**: metronome clicks on every beat, accented
  on count 1 (`--quiet-preview` to silence), with the count shown large.
  Pressing Space starts the count-in on the preview's next beat, so the
  rhythm continues unbroken into the recording.
- `parapara.pose.body_frame`: MediaPipe world landmarks → per-sample body
  frame (right / up / forward rebuilt from the torso every sample, with a
  vertical fallback when the hips are hidden) and arm segment directions
  (upper arm, forearm, hand) with validity masks.
- `uv run parapara-analyze [session]` (`parapara.diagnostics.pose_analysis`):
  compares recorded clips with the block library, reporting the best lag
  (searched within one block repeat; "not measurable" when the pose barely
  changes), per-keyframe danced vs. target directions in 3-D and in the
  front plane, and calibration values from static clips. Writes
  `<session>/analysis.json`. This is the tool for recalibrating the poses.
- Recorder look-ahead: the guide shows the next pose one count ahead
  (`--lookahead`), labelled "next: hit on count N"; saved in clip metadata.
- Tests: `test_body_frame_and_analysis.py` with synthetic skeletons
  (`tests/pose_synth.py`), plus a check that console entry points return
  `None`.

### Fixed
- Console entry points `parapara-capture-bench`, `parapara-record`,
  `parapara-analyze` and the headless benchmark returned a value from
  `main()`, which `sys.exit(main())` printed and turned into exit status 1.
  `main()` now returns `None`; `run()` returns the result.
- `geometry.py` wrongly described the body frame as right-handed; it is
  physically left-handed (forward = up × right).
- **Step 3a – guided recorder** (`uv run parapara-record`): live camera view
  plus a guide panel (instructions, mirrored stick figure of the target pose,
  countdown / count-in / beat counter, progress bar) that walks the player
  through calibration poses (neutral, T-pose), axis checks (right arm up /
  forward) and one clip per building block danced to a metronome. Each clip
  can be kept, redone or skipped.
  - `parapara.pose.recording`: `Recording` (`.npz`, landmarks only, never
    images; loads without pickle) with beat times and metadata.
  - `parapara.diagnostics.clips`: the recording script and `ClipTimeline`
    (phases, metronome beats, current target keyframe).
  - `parapara.game.overlay.guide_figure`: front-view projection of target
    poses for the guide figure.
  - `debug_view.LiveFeed`: the shared camera + skeleton + HUD view, now used by
    the debug view, the benchmark and the recorder.
  - `recordings/` added to `.gitignore`. Tests: `test_recording.py`
    (including a headless recorder run with a streaming fake worker).
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
- **Block library calibrated against the first real-dancer recording**
  (`recordings/20261018-144829`): chop salute with the elbow at shoulder
  height; chop out keeps 45° with a 40° tolerance; knock upper arm and forearm
  tilted forward; flick forearm level across the chest with the elbow
  carrying it; sweeps ±45° with the crossing arm's upper arm pointing forward;
  wrist roll holds one position per count (rolling is invisible to body
  landmarks). Mean angle error per block fell from 30-50° to 9-29°.
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
