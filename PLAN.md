# Plan: Para Para Dance Game (Desktop)

A desktop rhythm game. The player dances Para Para in front of a webcam, the game
tracks their body and hands in real time, and scores each move against a
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

Feet shoulder-width apart, knees loose. Step right, touch left foot to right;
step left, touch right foot to left — one step per beat. This is the timing
reference for all arm moves. **In-game it is shown as a visual guide only and
is never scored.** Scoring is based purely on the upper body, so the game is
accessible to players who cannot (or prefer not to) step — including seated
players — and only the upper body needs to be in camera frame.

### 1.2 Arm & hand building blocks

| Block | Description | Duration | Character | Needs hand landmarks? |
|---|---|---|---|---|
| **Sweep** | Both arms move parallel across the chest from one side to the other; soft wrists | 2 or 4 counts | Flowing | No |
| **Knock** | Closed fists snapped out to sharp right-angle positions (forward / up), in pairs | 2 counts (1 per punch) | Sharp | Yes (fist) |
| **Chop** | One arm extends outward at 45°, then snaps in to rest near the opposite collarbone | 2 counts per arm (1 out, 1 in) | Sharp | No |
| **Wrist Roll** | Hands close to chest at shoulder height, wrists rolling over each other | 2 or 4 counts | Flowing | Optional |
| **Flick + Clap** | Three backhanded outward wrist flicks (counts 1–3), sharp clap on count 4 | 4 counts | Phrase finisher | Optional |

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
  3. fill exactly 32 counts; blocks must not cross a bar boundary unless the
     block is 4 counts long and starts on count 1 or 5 of a bar.
- Tempo: songs are **140–160 BPM**, 4/4 time.

The vague quality requirements become measurable scoring components:

| Original requirement | Measured as |
|---|---|
| "Smooth, flowing motion" | Low normalized **jerk** of wrist trajectories during Sweep / Wrist Roll |
| "Strong, confident posture" | Torso near vertical, shoulders level, head up (angles from pose landmarks) |
| "Clear, audible clap" | **Visual clap**: wrists/palms meet within the timing window on the phrase's final count (microphone detection is unreliable while music plays) |

---

## 3. Core design decision: one routine representation for everything

Each building block is defined as **timed keyframe poses**: target arm/hand
configurations at specific counts (e.g. Chop-right = "right arm extended at 45°
on count 1; right wrist at left collarbone on count 2"). Keyframes are
expressed in a **normalized, body-relative coordinate system** (origin at
mid-shoulder, unit = shoulder width), so they are independent of the player's
size and distance from the camera.

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
 webcam ──► │ OpenCV frame grab → MediaPipe Pose + Hand → filter │──► landmark queue
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

### 4.1 Proposed package layout

```
src/parapara/
  routine/     # block definitions, keyframes, routine model, validation, (de)serialization
  generator/   # routine generator with difficulty levels
  capture/     # camera + MediaPipe wrapper, One Euro filter
  pose/        # normalization, mirroring, feature extraction (angles, directions)
  scoring/     # keyframe matching, timing judgment, DTW, quality metrics
  audio/       # song library, beat maps, playback clock
  game/        # pygame-ce screens: menu, calibration, play, results
tests/         # pytest; uses recorded landmark sequences as fixtures
assets/        # music, beat maps, routines, avatar renders
```

---

## 5. Scoring design

Pipeline for every landmark sample:

1. **Smooth** with a One Euro filter (low jitter, low lag).
2. **Normalize**: translate to mid-shoulder origin, scale by shoulder width.
3. **Mirror** horizontally (player faces the screen and copies the demo like a
   mirror).
4. **Extract features**: elbow and shoulder angles, upper/lower-arm direction
   vectors, wrist positions, fist/open-hand state, wrist distance (for claps).

Per keyframe (a target pose on a given count):

- Search a window around the beat time for the best-matching sample.
- **Pose accuracy** (0–1): weighted similarity of features to the target.
- **Timing** from the offset of the best sample:

  | Grade | Window (tunable; ~155 BPM beat ≈ 387 ms) |
  |---|---|
  | Perfect | ±60 ms |
  | Great | ±120 ms |
  | Good | ±190 ms |
  | Miss | otherwise, or pose accuracy below threshold |

- Flowing blocks (Sweep, Wrist Roll) are additionally compared over their whole
  duration with **DTW** against the reference trajectory, plus a smoothness
  (jerk) score.

Final score = sum of keyframe scores × combo multiplier + posture bonus.
Only upper-body landmarks (shoulders, elbows, wrists, hands, head, and hips
for torso angle when visible) are used; **footwork is never scored**. No model training is required initially; a learned classifier
is a possible later improvement once recorded gameplay data exists.

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
- Early fallback: a 2D stick figure drawn in pygame-ce directly from keyframes,
  so the game is playable before the 3D pipeline exists.
- Recording a human dancer / video-to-animation AI is optional, only for
  validating that keyframes look natural.

---

## 8. Tech stack (versions checked 2026-10-18)

| Purpose | Library / tool | Notes |
|---|---|---|
| Python | **3.13** | torch / onnxruntime have no 3.15 wheels yet |
| Pose + hand tracking | `mediapipe` 1.1.1 (Tasks API: PoseLandmarker, HandLandmarker) | macOS wheel is arm64-only; verify 1.x API against current docs |
| Camera / image ops | `opencv-contrib-python` (pulled in by mediapipe) | don't also install `opencv-python` |
| Numerics | `numpy` 2.x | |
| Beat detection | `librosa` 1.0.0 | offline tool only, not needed at runtime |
| Game window, rendering, audio | `pygame-ce` 2.5.8 | original `pygame` has no wheels for Python ≥ 3.13 |
| Precise audio timing (if needed) | `sounddevice` | already a mediapipe dependency |
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
| **2. Pose capture prototype** | Webcam + MediaPipe Pose/Hand, skeleton overlay, FPS + latency readout | Live debug window |
| **3. Pose features** | One Euro filter, normalization, mirroring, feature extraction | Unit-tested feature functions |
| **4. Scoring engine** | Keyframe matching, timing grades, DTW, posture/smoothness | Scores computed on recorded fixtures |
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

Future (out of scope for the current roadmap):
- Leaderboard (local first, online later).
- Multiplayer (several dancers in frame, or networked).
