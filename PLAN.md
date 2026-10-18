# Plan: Para Para Dance Game (Desktop)

A desktop rhythm game. The player dances Para Para in front of a webcam, the game
tracks their upper body in real time, and scores each move against a
choreographed routine synced to Eurobeat music.

---

## 1. Background: what is Para Para?

Para Para (パラパラ) is a synchronized, choreographed Japanese dance style where
participants perform fixed hand and arm movements in unison to fast-paced
electronic music (mainly Eurobeat, also Techpara / Trapara).

- **Upper-body focus**: arm, hand and upper-body gestures; footwork is a minimal
  side-to-side two-step.
- **Song-specific choreography**: every song has its own fixed sequence built
  from a shared vocabulary of moves.
- **Strict 4/4 timing**, counted in repeating 8-count bars.

### 1.1 Footwork: the two-step (the "engine")

Feet shoulder-width apart, knees loose. "Step right [1], step together [2],
step left [3], step together [4], step right [5]" — one step per beat. This is the timing
reference for all arm moves. **In-game it is shown as a visual guide only and
is never scored.** Scoring is based purely on the upper body, so the game is
accessible to players who cannot (or prefer not to) step — including seated
players — and only the upper body needs to be in camera frame.

### 1.2 Arm & hand building blocks

Reference: EKS-D "How to Dance", <https://kirakira.org/XD/learn/how-to-dance/>
(checked 2026-10-18). Moves "with rare exception, begin on the right-hand side".

| Block | Description | Duration | Character |
|---|---|---|---|
| **Sweep** | Arms sweep from one point to another: horizontal, vertical, or diagonal across the body; one arm or both; soft wrists | 2 or 4 counts | Flowing |
| **Knock** | "Banging on an imaginary door": fist, thumb towards the body; upper arm horizontal, forearm up (right angle). Same hand on beats 1 & 2, then mirrored (right 1–2, left 3–4) | 2 counts per hand | Sharp |
| **Chop** | Arm reaches 45° up and out to the side (beat 1), then folds in front of the chest, flat hand level (not pointing down), like a salute (beat 2). Often right 1–2, 3–4, then left 5–6, 7–8 | 2 counts per chop | Sharp |
| **Wrist Roll** | Hands close to chest at shoulder height, wrists rolling over each other (*not in the reference; unverified*) | 2 or 4 counts | Flowing |
| **Flick + Clap** | "nami + nami + nami + pon!": one arm crosses the body and does a backhanded wrist flick just above shoulder height (1), again in front (2), again on the opposite side (3), then a clap where the third flick ended (4). **Any clap scores, regardless of side or arm position.** Often paired in opposite directions | 4 counts | Phrase finisher |

Pairing patterns (knock right then left; chops right-right-left-left;
flick + clap pairs in opposite directions) are choreography rules for the
routine generator (Step 7), not part of the individual blocks.

---

## 2. Routine requirements (revised)

The original requirements asked for an 8-count routine containing all five
blocks, which is impossible (Flick + Clap alone is 4 counts, Chop is 4 counts
for both arms). Revised structure:

- **Count** = one beat. **Bar** = 8 counts. **Phrase** = 4 bars = 32 counts.
- A **routine** is a sequence of phrases covering the playable part of a song.
- Every **phrase** must:
  1. contain at least one of each of the five building blocks;
  2. **end with Flick + Clap** (the clap lands on the last count of the phrase);
  3. fill exactly 32 counts;
  4. never let a block cross a bar boundary, and start every block on a
     multiple of its own length (4-count blocks on count 1 or 5, 2-count
     blocks on odd counts).
- Tempo: songs are **140–160 BPM**, 4/4 time.

The vague quality requirements become measurable scoring components:

| Original requirement | Measured as |
|---|---|
| "Smooth, flowing motion" | Low normalized **jerk** of wrist trajectories during Sweep / Wrist Roll |
| "Strong, confident posture" | Torso near vertical, shoulders level, head up (angles from pose landmarks) |
| "Clear, audible clap" | **Audible clap**: a sharp sound burst from the microphone in the timing window on the phrase's final count, confirmed by the wrists meeting (§5.1) |

---

## 3. Core design decision: one routine representation for everything

Each building block is defined as **timed keyframe poses**: target arm/hand
configurations at specific counts (e.g. Chop-right = "right arm extended at 45°
on count 1; right wrist at left collarbone on count 2").

Arm targets are **unit direction vectors** (decided 2026-10-18), per arm:
upper arm (shoulder → elbow), forearm (elbow → wrist) and optionally hand
(wrist → knuckles, for flicks, measured with the pose model's coarse
index/pinky points), plus a hand shape (open / fist / any) that is used only
for the demo animation and **never scored** (hands are not tracked).
Direction vectors ignore limb length, so targets are independent of the
player's size and distance from the camera.

**Body frame** (dancer's own anatomy; physically left-handed, see `geometry.py`): origin at mid-shoulder,
`+x` = towards the dancer's **right** hand, `+y` = up, `+z` = forward (out of
the chest, towards the camera). "Right arm" always means the dancer's own
right arm; mirroring (for the player copying an on-screen demo, and for
generating left/right variants of a block) swaps the arms and negates `x`.

**Player calibration** (before each session): first a **position check**
(`parapara.pose.framing`) guides the player until the whole torso, the hips
and the room to stretch the arms are in the picture and they face the camera
("Step back", "Move a little to your LEFT", ...); then the player holds a
neutral pose and a T-pose for a few seconds. This measures shoulder width and segment
lengths (to turn direction targets into positions for collarbone/clap checks
and for drawing the guide figure) and the camera's tilt (to align the camera
frame with the body frame).

The five block definitions are **first drafts** written from the textual
descriptions; they will be refined from recordings of a real dancer.

This single representation drives:

1. **Routine generation** (grammar of blocks with durations and constraints),
2. **Scoring** (player pose compared against keyframes on each beat),
3. **Demo animation** (avatar animated procedurally from the same keyframes),

so the scoring always matches the demo exactly, and new routines need no motion
capture.

Routines are stored as data files (JSON/TOML) and validated on load.

---

## 4. Architecture

```
            ┌────────────── capture process/thread ─────────────┐
 webcam ──► │ OpenCV frame grab → MediaPipe Pose → filter        │──► landmark queue
            └────────────────────────────────────────────────────┘          │
                                                                            ▼
 song + beat map ──► audio clock ──► game loop (pygame-ce) ◄── scoring engine
                                       │  render: camera view, skeleton,
                                       │  upcoming-move cues, avatar, HUD
                                       ▼
                                  results screen
```

- **Capture runs in a separate OS process** (`multiprocessing`, spawn) so slow
  inference never stalls the game loop. Each landmark sample carries its
  capture timestamp. Landmarks go over a `multiprocessing.Queue`; camera
  frames for display go through `multiprocessing.shared_memory`.
- **Hard rule: the game process never imports `cv2` or `mediapipe`, and the
  capture process never imports `pygame`.** `opencv-contrib-python` 5.x and
  `pygame-ce` each bundle their own `libSDL2` dylib; loading both in one
  process makes macOS warn of duplicate Objective-C classes and possible
  crashes (verified 2026-10-18; the headless OpenCV wheel bundles SDL2 too).
  A test enforces this import boundary.
- **The audio playback position is the master clock**, not wall time.
- **Latency calibration**: a "clap to the beat" screen measures the combined
  camera + inference + audio-output delay; the offset is saved per machine and
  subtracted when judging timing.

### 4.1 Package layout

```
src/parapara/
  routine/     # block definitions, keyframes, routine model, validation, (de)serialization
  generator/   # routine generator with difficulty levels
  capture/     # capture process only: camera, MediaPipe, model download (cv2/mediapipe allowed)
  ipc/         # shared by both processes: CaptureClient, shared frame buffer, messages
  pose/        # landmark types, One Euro filter, normalization, mirroring, features
  scoring/     # keyframe matching, timing judgment, DTW, quality metrics
  audio/       # song library, beat maps, playback clock
  game/        # pygame-ce screens: menu, calibration, play, results, capture debug view
tests/         # pytest; uses recorded landmark sequences as fixtures
assets/        # music, beat maps, routines, avatar renders
```

### 4.2 Capture pipeline (implemented in Step 2)

- **Camera**: OpenCV `VideoCapture` (AVFoundation on macOS) read on a
  background thread that keeps only the **newest** frame, so slow inference
  skips frames instead of falling behind real time.
- **Detection**: MediaPipe Tasks `PoseLandmarker` (one person) in `VIDEO`
  running mode (synchronous, so every result belongs to a known frame).
  Timestamps are forced to be strictly increasing, as `VIDEO` mode requires.
- **No hand tracking** (decided 2026-10-18 after the capture benchmark): the
  `HandLandmarker` took ~55 ms per frame, halving tracking to ~12 fps and
  quadrupling latency, and found each hand only ~20% of the time at dancing
  distance. The pose model's index/pinky/thumb points give hand direction.
- **Models** (`pose_landmarker_full`/`_lite`) are
  downloaded on first use to `~/Library/Caches/parapara/models` and pinned by
  SHA-256, because the official URLs point at a moving `latest` path.
- **IPC**: samples (`PoseSample`) and status messages go through a bounded
  queue (the worker drops samples rather than block); the frame goes through
  a lock-protected shared-memory slot, published after detection under the
  same frame index as its sample.
- **Timing**: all timestamps are `time.monotonic_ns()` (system-wide on macOS),
  so the game can measure camera→landmarks and camera→screen latency.
- **Debug view**: `uv run parapara-capture-debug` shows the camera with the
  skeleton (cyan = dancer's left, orange = dancer's right), tracking rate,
  per-model inference time, latency and dropped samples.

---

## 5. Scoring design

Implemented in Step 4 (`parapara.scoring`); based on the 2026-10-18 recordings.

Feature pipeline per sample (Step 3, `pose.features`): One Euro filtered
landmarks → body frame from the shoulders/torso (adaptively smoothed) → arm
segment directions, wrist positions and distance, posture; relative to the
player calibration. Features always use the dancer's anatomical left/right;
mirroring is only a display concern.

Per keyframe (`scoring.judge`), within a window around its expected time
(`scoring.schedule`: routine + beat clock + **latency offset**):

- **Pose accuracy**: angular error of upper arm, forearm (and hand when set),
  **front-weighted** (depth counts half: it is MediaPipe's noisiest axis).
  Full credit within the block's tolerance (25° sharp, 35° flowing, 40° chop
  reach), linear to zero at twice the tolerance.
- **Timing**: the best-matching sample in the window, refined between
  samples (~23 fps camera); offset graded:

  | Grade | Window |
  |---|---|
  | Perfect | ±60 ms |
  | Great | ±120 ms |
  | Good | ±190 ms, capped at 0.45 beat (no overlap at fast tempos) |
  | Hold | repeated pose (e.g. second knock): judged on pose only |
  | Miss | outside the window, or pose score < 0.3 |

- **Claps**: hands together (wrist distance); timing from the clap sound when
  available (§5.1), otherwise from the moment the wrists are closest.
- **Flowing blocks** (sweeps, wrist roll): an extra "keep moving" score: the
  share of the block with the wrists moving faster than 0.6 shoulder widths/s
  (held poses stay below, the recorded wrist roll above). This replaces the
  originally planned DTW comparison, which the keyframe matching already
  covers; DTW remains an option if flowing moves need finer judgment.

Score (`scoring.scorekeeper`): Σ 100 × pose score × timing factor (Perfect 1,
Great 0.8, Good 0.5, Hold 1) × combo multiplier (+0.1 per 10-hit combo, max
×2), + flow bonus (up to 50 per flowing block), + posture bonus (up to 10% of
the maximum keyframe points: torso lean / side tilt within 15° of neutral,
head not dropped). Accuracy (mean keyframe points) is reported separately.
Only upper-body landmarks are used; **footwork and hand shape are never
scored**.

Findings when scoring the recordings (`parapara-score`):

- **Latency compensation is essential**: a sweep danced 0.7 s late scored 45%
  uncompensated vs. 81% compensated, so the clap-to-the-beat latency
  calibration (§4) is a must before play. The latency must be measured through
  the same pipeline (it then also absorbs the ~28 ms landmark-filter lag).
- Flick + Clap scores only ~30% with its current poses (recalibration
  deferred, §10).
- The posture rule marked 30% of a sweep as poor posture (the torso tilts with
  the sweep); to be tuned on more recordings.

### 5.1 Clap detection (sound + vision)

Decided 2026-10-18: the clap that ends each phrase is detected **by sound**.

- **Microphone onset detection**: a clap is a short, broadband burst, so the
  microphone stream (via `sounddevice`) is watched for sharp energy onsets
  (spectral flux in the high band). Audio gives far better timing than the
  ~23 fps camera (a few ms instead of ±20 ms), so the clap's *timing grade*
  comes from the sound.
- **Problem: the music claps too.** Eurobeat typically has a clap/snare on the
  beat, played through the speakers into the microphone at exactly the moment
  the player should clap; sound alone would score the song's own clap as a
  perfect hit. Mitigations, in order:
  1. **Vision gate**: an onset only counts if the pose wrists are close
     together (relative to shoulder width) at that moment; no hand model needed.
  2. **Reference rejection**: the game knows exactly what it is playing. During
     calibration it measures the speaker→microphone delay and gain; onsets
     explained by the played signal (same time, similar strength) are
     discarded, and only *excess* onset energy counts.
  3. **Headphones**: recommended in the settings; they remove the problem.
- **Calibration** reuses the "clap to the beat" screen (§4): it measures the
  audio input latency as well as the camera latency.
- Implemented with the audio module (roadmap Steps 5–6); latest
  `sounddevice` version and docs to be checked then.

---

## 6. Music

- **Sources**: royalty-free tracks from Free Music Archive / Jamendo (search
  "90s High-Energy Synth", "Fast J-Pop Dance", "Upbeat Synthwave", 140–160 BPM),
  or AI-generated tracks (prompt: "155 BPM, 1990s Japanese Arcade Eurobeat, fast
  synth brass melody, high-energy driving bassline, instrumental, aggressive
  electronic dance music").
- **Licensing**: record the license of every track in the library metadata.
  CC-BY-NC forbids commercial use; CC-BY-ND may forbid edits. AI-generated
  music rights depend on the generator's terms and subscription tier.
- **Beat map** per song: BPM, first-beat offset, start/end of the playable
  section. Detected with `librosa`, then **manually verified** (a 30 ms drift
  breaks scoring). Stored next to the audio file.

---

## 7. Demo animation

- Avatar: a VRM model made in **VRoid Studio** (or from VRoid Hub, respecting
  each model's license conditions).
- Animation generated **procedurally from routine keyframes** in **Blender**
  (VRM add-on, scripted via `bpy`), rendered offline to video per routine.
- The demo is shown **mirrored** (like the "REVERSE"/"MIRROR" practice
  videos in the dance community): when the character moves its right arm, it
  appears on the right of the screen, so the player copies it like a mirror
  with their own right arm.
- Early fallback: a 2D stick figure drawn in pygame-ce directly from keyframes,
  so the game is playable before the 3D pipeline exists.
- Recording a human dancer / video-to-animation AI is optional, only for
  validating that keyframes look natural.

---

## 8. Tech stack (versions checked 2026-10-18)

| Purpose | Library / tool | Notes |
|---|---|---|
| Python | **3.13** | torch / onnxruntime have no 3.15 wheels yet |
| Pose tracking | `mediapipe` 1.1.1 (Tasks API: PoseLandmarker) | macOS wheel is arm64-only; no hand tracking (§4.2) |
| Camera / image ops | `opencv-contrib-python` (pulled in by mediapipe) | don't also install `opencv-python` |
| Numerics | `numpy` 2.x | |
| Beat detection | `librosa` 1.0.0 | offline tool only, not needed at runtime |
| Game window, rendering, audio | `pygame-ce` 2.5.8 | original `pygame` has no wheels for Python ≥ 3.13 |
| Microphone (clap detection), precise audio timing | `sounddevice` | already a mediapipe dependency; declare explicitly when used |
| Avatar animation | VRoid Studio, Blender + VRM add-on | offline pipeline |
| Testing | `pytest` | scoring tested on recorded landmark fixtures |
| Avoid | `ultralytics` (YOLO pose) | AGPL-3.0 licence |

Camera guidance for players: front-facing, ~1.5–2 m away, upper body in
frame (standing or seated), even lighting; 60 fps camera recommended (flicks blur at 30 fps).

---

## 9. Implementation roadmap

Each step ends with working, tested code and a CHANGELOG entry.

| Step | Goal | Deliverable |
|---|---|---|
| **0. Project setup** | Python 3.13, package layout, dependencies, pytest | `uv sync` + `pytest` pass |
| **1. Routine model** | Blocks as keyframes, bar/phrase structure, validation rules from §2, JSON load/save | Validated hand-written sample routine |
| **2. Pose capture prototype** | Webcam + MediaPipe Pose, skeleton overlay, FPS + latency readout, capture benchmark | Live debug window, `parapara-capture-bench` |
| **3a. Guided recorder** | Record calibration poses, axis checks and each building block (landmarks only) | `parapara-record`, first recordings |
| **3. Pose features** | One Euro filter, body frame, calibration, feature extraction; axes verified on the 3a recordings | Unit-tested feature functions |
| **4. Scoring engine** | Keyframe matching, timing grades, holds, claps, flow, posture, combo | `parapara.scoring`, `parapara-score` on recordings |
| **4b. Real-dancer test run** | Record a real dancer with the debug view; check tracking keeps up, left/right labels, and refine the draft block poses against the recordings | Recorded fixtures + revised `library.py` |
| **5. Music library & beat maps** | Collect tracks, license metadata, beat-map tool | ≥ 1 verified song |
| **6. Game loop** | pygame-ce screens, audio clock, calibration, move cues, HUD, results; stick-figure demo | First playable song |
| **7. Routine generator** | Grammar-based generator, Easy / Normal / Hard | 3 routines per song |
| **8. Avatar demo** | VRM + Blender procedural animation, rendered demo videos | Demo video per routine |
| **9. Polish** | Menus, high scores, settings, packaging | Distributable app |

Difficulty levels (step 7) vary move density (4-count vs 2-count blocks),
block variety, and left/right asymmetry; tempo is fixed by the song.

---

## 10. Decisions & future work

Decided (2026-10-18):
- Desktop app, Python 3.13, package name `parapara`.
- Footwork is a visual guide only; scoring uses upper-body posture only
  (accessibility, seated play supported).
- Single player only for now.

Open items (to resolve before the step named):
- **Step 4b deferred** (project owner, 2026-10-18): continue with Step 5
  and revisit the real-dancer test run later, together with:
- **Deferred recording & recalibration** (postponed by the project owner on
  2026-10-18; do before Step 4b at the latest): re-record the six block
  clips with the position check passing (hips in the picture), run
  `parapara-analyze`, and recalibrate `library.py`, especially Flick + Clap,
  whose second take did not fit (lag alias / different interpretation).
  Session 2 (`recordings/20261018-151657`) has the hips out of frame and
  should not be used for calibration.
- Real-dancer test run (4b): verify the draft poses (wrist roll, one-armed
  sweeps, chop salute, clap).
- Camera rate: the webcam delivers ~23 fps although 30 is requested
  (typical auto-exposure behaviour in dim light). Retest in brighter light;
  target ~30 fps (≈ 13 frames per beat at 155 BPM).
- Clap detection (5–6): implement §5.1 and test with speakers *and*
  headphones on Eurobeat tracks that contain claps/snares.
- Distribution (9): check the MediaPipe model card licences, and tell
  players that the MediaPipe Tasks API sends usage metrics to Google
  (images are processed on-device).

Findings from the first recording session (2026-10-18, `parapara-analyze`):
- MediaPipe world axes: +x = dancer's left, +y = down, +z = away from the
  camera (opposite to the body frame on every axis).
- Depth is the weakest axis: wrist jitter 4–17 mm in x/y but 12–59 mm in z,
  and directions read biased towards the camera. Scoring (Step 4) should
  weight the front plane (x/y) more than depth.
- The dancer trailed the beat by 200–700 ms because the guide showed poses
  only *on* their count. The recorder now **previews the whole move before
  recording** and keeps the guide on the beat while recording; **the game
  (Step 6) must likewise show players what is coming before they need it**
  (e.g. a preview of each new block before it starts).
- Lag is only identifiable within one block repeat (a pose shifted by a whole
  repeat looks identical), and not at all for moves that hold one pose.
- Calibration values: shoulder width ~0.35 m, upper arm / forearm ≈ 0.6–0.7 ×
  shoulder width, torso lean towards the camera 4–12° (camera angle).
- Second session (2026-10-18, with the move preview): lag fell to
  140–240 ms for chop, sweeps and wrist roll. Dancers move their torso a lot:
  shoulders turn by ±26° in sweeps and flicks, the torso tilts sideways by up
  to ±9°, and the hips sway by up to ±9% of the image width when stepping.
- Framing: in the second session the **hips were out of the picture in
  every clip** (hip landmarks at 100–107% of the image height), so the body
  frame relied on guessed hip positions. A position check now gates
  calibration and is shown before every clip; the game must keep checking
  it during play (Step 6).
- Body frame choice (compared with `parapara-analyze`): arm directions must
  be relative to the **shoulders**. Camera-relative errors are 31° vs 18° on
  sweeps; a hip-based frame fails because shoulders twist relative to the
  hips. The default `torso` frame is best or near-best on 10 of 12 clips. Its
  weakness is chops (raising an arm also lifts that shoulder, ~3–5° extra
  error). Step 3 will smooth the torso frame more slowly than the arms and
  apply per-player calibration offsets.

- Step 3 results (2026-10-18): filtering makes features ~60% steadier
  frame to frame (fewer flickering scores) but barely changes the *mean* fit
  to the targets (19.8° → 19.6°), which is dominated by how moves are danced
  and by depth bias. So Step 4 should rely on tolerances, front-plane
  weighting and time windows rather than expect filtering to close the gap.

Resolved:
- Left/right (2026-10-18, human test): MediaPipe pose labels are the
  dancer's anatomical sides on an unmirrored webcam image (left = cyan in the
  debug view). Tracking visibly keeps up with the movements.
- Tracking rate (2026-10-18, `parapara-capture-bench`, MacBook webcam,
  640×480): full pose model + hand model **12.4 fps** (hands 53 ms/frame,
  47% of frames skipped, 114 ms camera→screen); full pose model alone
  **23.1 fps = camera rate**, 0% skipped, pose 25 ms, **41 ms camera→screen**;
  the lite model brings no benefit. Decision: full pose model, **no hand
  tracking**; claps by sound (§5.1).

Future (out of scope for the current roadmap):
- Leaderboard (local first, online later).
- Multiplayer (several dancers in frame, or networked).
