# Contributing to ParaParaPy

This guide explains how the code base is organised, how the pieces fit
together and why they are built the way they are, so you can find your way
around and change things safely. For what the game is and how to play it,
see the [README](README.md) and the [player guide](docs/how-to-play.md); for
the design history, decisions and roadmap, see [PLAN.md](PLAN.md); for what
changed when, see [CHANGELOG.md](CHANGELOG.md).

**Contents**

1. [Getting started](#1-getting-started)
2. [Ground rules](#2-ground-rules)
3. [The big picture](#3-the-big-picture)
4. [Package by package](#4-package-by-package)
5. [Data on disk](#5-data-on-disk)
6. [Tools and commands](#6-tools-and-commands)
7. [Testing](#7-testing)
8. [Common tasks](#8-common-tasks)
9. [Glossary](#9-glossary)

---

## 1. Getting started

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.13, a webcam for
playing (not for the tests), macOS or Linux.

```sh
uv sync                    # installs the game plus the dev (pytest) and tools (librosa) groups
uv run pytest              # ~450 tests, ~5 min, headless (no camera, no window, no sound)
uv run parapara-fetch-demo-songs   # the four CC BY demo songs (git-ignored)
uv run parapara-play       # the game
```

The first run of anything that uses the camera downloads the MediaPipe pose
model into `~/Library/Caches/parapara/models` (`$XDG_CACHE_HOME/parapara/models`
on Linux, or `$PARAPARA_MODEL_DIR`), checks its hash and reuses it afterwards.

Names: the project is **ParaParaPy** (distribution `paraparapy`). The Python
package, the commands (`parapara-*`) and the user data folder keep the short
name `parapara`, so saved settings and scripts keep working.

## 2. Ground rules

These come from [AGENTS.md](AGENTS.md) and from how the project has been run:

- **Document the design** in the code: every module starts with a docstring
  that explains *what* it does and *why* it is built that way (measurements,
  rejected alternatives, dates of decisions). Keep it current when you
  change the code.
- **Every change comes with tests** (see §7), and with an entry in
  [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]` (Keep a Changelog
  format: Added / Changed / Fixed).
- **Check the latest version and documentation** of an open-source library
  before using it (PLAN.md §8 lists the versions checked).
- **Never bypass a check** (a test threshold, the layout check, the import
  boundary guard) without a good, written justification and the project
  owner's agreement.
- **Commits** only after the project owner has reviewed the change.
- **Releases**: bump `version` in `pyproject.toml` (shown in the game via
  `parapara.__version__`), move the `[Unreleased]` entries of the CHANGELOG
  into a new `## [x.y.z] - YYYY-MM-DD` section with a short summary, and
  leave `[Unreleased]` empty. 0.1.0 (2026-10-23) is the first release.
- **Measure, don't guess**: scoring and timing constants come from
  recordings of a real dancer and from the synthetic "perfect player" (§7).
  When you change one, record the before/after numbers in the docstring and
  the CHANGELOG.
- **Readable from the dancing spot**: players stand 1.5-2 m away. Use the
  large-text helpers (`parapara.game.ui`) for anything a player reads, put
  what they need while dancing around the demo dancer, and prefer pictures
  to text (see the play-test findings in PLAN.md §10).

## 3. The big picture

### Two processes

```
 ┌───────────── capture process (spawned) ─────────────┐        ┌──────────────── game process ────────────────────────┐
 │ camera.py  newest webcam frame (OpenCV)             │ frames │ ipc/client.py   CaptureClient: poll(), latest_frame() │
 │ detector.py MediaPipe Pose Landmarker (VIDEO mode)  │ ─────▶ │ pose/           filter → body frame → PoseFeatures    │
 │ worker.py  loop: frame → landmarks → PoseSample     │ samples│ scoring/        schedule → judge → score              │
 │            (cv2 / mediapipe live only here)         │ ─────▶ │ game/           pygame-ce screens, audio, drawing     │
 └─────────────────────────────────────────────────────┘        └───────────────────────────────────────────────────────┘
          shared memory (ipc/frame_buffer.py) for the picture, a multiprocessing Queue for samples and status
```

`opencv` (a MediaPipe dependency) and `pygame-ce` each bundle their own SDL2
library; loading both into one process causes duplicate-class warnings and
crashes on macOS. So:

- only `parapara.capture` may import `cv2` / `mediapipe`, only
  `parapara.game` may import `pygame`, and every other subpackage is pure
  Python + NumPy (usable on either side);
- the capture worker runs in a **spawned** process; `ipc/client.py` starts it
  through a trampoline and a `"module:function"` string, so the game process
  never imports the worker module (and tests can swap in a fake worker);
- `tests/test_import_boundary.py` checks the rule statically and at run time,
  and `tests/conftest.py` fails the test run if `cv2` and `pygame` ever end
  up in the same pytest process.

### One frame's journey

1. The **worker** grabs the newest frame (dropping stale ones: latency beats
   completeness), runs MediaPipe and puts a `PoseSample` (33 world and image
   landmarks, capture timestamp, frame index) on the queue; the frame goes
   into shared memory for display.
2. The **game** polls the queue each display frame (60 Hz). `LiveFeed` draws
   the newest frame with the skeleton; `FramingMonitor` checks the player's
   position.
3. `FeatureExtractor` smooths the landmarks (One Euro filter), builds the
   **body frame** from the torso, and turns them into `PoseFeatures`: arm
   segment directions relative to the body, wrist distance, posture angles,
   plus an **unsmoothed timing copy** of the arms (§4, `pose`).
4. `PlaySession` feeds the features to the `Judge`, which compares them with
   the routine's **schedule** (each target pose with its due time on the
   song's beat clock, shifted by the measured latency) and emits a
   `Judgment` per keyframe once its timing window has passed.
5. `ScoreKeeper` turns judgments into points, combo and accuracy; the game
   shows the grade above the demo dancer.

### Routines, songs and the beat clock

A **routine** is pure choreography in counts (no seconds): phrases of 32
counts built from five kinds of **blocks** (sweep, knock, chop, wrist roll,
flick + clap), each block a list of **keyframes** (target arm directions at a
count). A **song** carries a constant-tempo beat map (`bpm`, `first_beat_s`,
`routine_start_beat`). Combining them gives the **schedule** (what is due
when) for the judge and the **animator** (what the demo dancer does at any
moment). The demo dancer dances exactly what is scored.

## 4. Package by package

All paths are under `src/parapara/`. Each module's docstring has the details;
this is the map.

### `routine/`: what to dance

| Module | What it does |
|---|---|
| `geometry.py` | Tiny 3-D vector helpers. Body frame: +x the dancer's right, +y up, +z forward. |
| `model.py` | `ArmPose` → `Keyframe` → `Block` → `Phrase` (32 counts) → `Routine`. Objects validate themselves (phrase rules: all five kinds, ends with flick + clap, bar alignment); `*_issues()` return problems instead of raising. Tolerances: 25° sharp, 35° flowing. |
| `library.py` | The building blocks (`BLOCK_LIBRARY`), authored for the right arm and mirrored. Targets are calibrated against recordings of a real dancer; the docstring records every recalibration with numbers. |
| `serialization.py`, `store.py` | Routines as JSON next to their song (`<song>/routines/`), loading, checking and installing players' own routines. |
| `cli.py` | `parapara-routines generate / list / blocks / template / check / add`. |

### `generator/`: routines for any song

`routine_generator.py` builds routines from idiomatic **units** (knock right
then left, chop pairs, flick + clap finishers ...) into phrases that pass the
phrase rules, then repeats phrases in a pattern (`A A B B ...`) over the
song's danceable length. Difficulty (easy / normal / hard) changes move
density, variety and the number of distinct phrases. Seeded, so the same
song always gets the same routines.

### `audio/`: songs and beat maps

| Module | What it does |
|---|---|
| `song.py` | `song.json` format and validation; `beat_time_s`, `routine_clock()`. Licence fields optional (players' own music). |
| `beatmap.py` | Automatic beat map with librosa on the kick-drum band, a constant-tempo fit (Eurobeat keeps a steady tempo) and a small detection-lag correction. |
| `beatgrid.py` | Editable beat grid and **click tracks** (clicks mixed into the samples: sample-accurate, unlike clicks fired from the 60 Hz loop). |
| `suitability.py`, `importer.py` | Checks a track is danceable (tempo, beat clarity, drift) and imports it into the user library with generated routines. |
| `fetch_demo_songs.py`, `find_songs.py` | The CC BY demo tracks (`assets/songs/catalog.json`) and a Jamendo search. |
| `synth.py`, `demo_song.py` | A generated "Eurobeat-like" track: a **test fixture only**, never for dancing. |

Beat maps must be **verified by ear** (`parapara-beatcheck`, guide in
`docs/verify-beat-maps.md`): a 30 ms error ruins the scoring.

### `capture/` and `ipc/`: the camera side

| Module | What it does |
|---|---|
| `capture/camera.py` | Webcam reader on its own thread that always hands out the newest frame. |
| `capture/detector.py` | MediaPipe Pose Landmarker (Tasks API, VIDEO mode), converted to plain NumPy. No hand tracking (decided: claps are judged from the wrists). |
| `capture/models.py` | Model download, hash check and cache. |
| `capture/worker.py` | The capture loop and its status messages. |
| `ipc/client.py`, `frame_buffer.py`, `messages.py` | Process start-up (spawn), shared-memory frame buffer, `CaptureConfig` / `CaptureStatus`. |

### `pose/`: from landmarks to features

| Module | What it does |
|---|---|
| `landmarks.py` | `PoseSample` and landmark indices (shared by both processes). |
| `filters.py` | One Euro filter (low jitter at rest, low lag when moving). |
| `body_frame.py` | MediaPipe world axes (+x dancer's left, +y down, +z away) → the dancer's body frame from the shoulders and hips; arm directions. Arms are measured **relative to the torso** because dancers turn and lean. |
| `calibration.py` | Neutral pose + T-pose → shoulder width, segment lengths, neutral lean and "up"; validated, saved per machine. |
| `features.py` | `FeatureExtractor` → `PoseFeatures`. Two copies of the arms: **smoothed** (judges *how good* a pose is) and **unsmoothed** (`for_timing()`, judges *when* it was hit; the smoothed error kept creeping down 40-130 ms after the arm stopped). |
| `framing.py` | Is the player well placed? (hips in the picture, room to stretch, centred, facing the camera), with player-facing messages. |
| `recording.py` | Save / load landmark clips (`.npz`; landmarks only, never images). |

### `scoring/`: grading

| Module | What it does |
|---|---|
| `schedule.py` | `BeatClock` (beats ↔ monotonic ns, latency offset) and the schedule of keyframes with due times; marks "hold" keyframes (same pose again). |
| `matching.py` | Pose error: angle between target and measured arm segments, **front-plane weighted** (depth is MediaPipe's weakest axis: `DEPTH_WEIGHT` 0.5); pose score from error and tolerance; clap gate from the wrist distance. |
| `judge.py` | Streaming judge. Per keyframe, within its timing window: best pose sample → pose score; hit time from the timing copy (held pose: start of the near-minimum stretch; passed-through pose: refined minimum) → Perfect ≤ 60 ms, Great ≤ 120 ms, Good ≤ 190 ms (capped at 0.45 beat), else Miss. Holds graded on pose only; claps by wrists (or sound onsets later); flowing blocks also get a "keeps moving" check. |
| `scorekeeper.py` | Points (100 × pose score × timing factor × combo multiplier: +0.1 per 10 non-misses, max ×2), flow and posture bonus, accuracy. |
| `latency.py` | "Clap to the beat": median clap offset from the clicks, in the same signal the judge times hits on; accepted with ≥ 6 claps and spread ≤ 60 ms (limit from real clapping). |
| `session.py` | `PlaySession`: one play (features → judge → score keeper), framing warnings, the play as a recording, ending early. |

### `animation/`: the demo dancer

`animator.py` is pure: routine + beat → `DancerPose`. Exactly on the target
on every scored beat; in between, motion styles (snap, glide, roll, knock
pulse, flick whip, clap), a two-step and a bounce. `Animator.for_blocks()`
animates single moves (Learn the moves). Drawing is in `game/dancer_view.py`.

### `game/`: everything the player sees (pygame-ce)

| Module | What it does |
|---|---|
| `app.py` | `parapara-play`: a scene state machine (menu, body / latency calibration, play, results, learn, credits, privacy, scores, settings). Calibrates once per launch, plays the song through `beatcheck.Player` (the audio clock drives the beat), keeps everything the player needs around the demo dancer. |
| `ui.py` | Shared large text: `UI.scale` (+ / -, `change_scale()`, saved), `toggle_fullscreen()`, word wrap that never runs off the surface, `footer()` for bottom-up key hints, fullscreen flags, and the **layout check** (`--check-layout`: records every text rectangle and reports overlaps or cut-off text). |
| `dancer_view.py`, `overlay.py` | The stick-figure dancer, still poses, the recorder's guide figure. |
| `notice_screens.py` | Scrollable documents (credits, privacy) and the **privacy gate** shown before the camera starts for the first time. |
| `branding.py` | Logo → menu wordmark and window icon. |
| `recorder.py` | `parapara-record`: hands-free guided recording of calibration and move clips with a quality check. |
| `latency_check.py` | `parapara-latency-test`: repeated latency calibrations, logged (named `latency_check` because pytest collects `*_test.py`). |
| `beatcheck.py`, `preview.py` | Beat-map check by ear; watching a routine. |
| `debug_view.py`, `capture_bench.py` | Live capture view (`LiveFeed`, also used by the other apps) and the capture benchmark. |

### Top-level modules

| Module | What it does |
|---|---|
| `settings.py` | Per-machine data folder (`$PARAPARA_DATA_DIR`, else `~/Library/Application Support/parapara` or `$XDG_DATA_HOME/parapara`): calibration, latency, UI preferences. |
| `scores.py` | The local score board. |
| `notices.py` | The privacy notice and credits texts (pure, unit-tested; facts and their sources in the docstring). |

### `diagnostics/`: measuring

`live_stats.py` (rate meters, `LowLightMonitor`), `recorder.py`
(`write_log`), `clips.py` (the recording script and its timeline),
`pose_analysis.py` (`parapara-analyze`: recorded clips vs. the block library,
best lag, per-keyframe "seen" directions), `score_clips.py`
(`parapara-score`), `filter_tuning.py`, `bench.py`.

## 5. Data on disk

| Where | What | In git? |
|---|---|---|
| `assets/songs/catalog.json` | the vetted CC BY demo songs (sources, measurements) | yes |
| `assets/songs/<id>/` | fetched demo songs: audio, `song.json`, `routines/*.json` | no (fetched) |
| `assets/branding/` | the logo and its derived wordmark / icon | yes |
| data folder (see `settings.py`) | `calibration.json`, `latency.json`, `ui.json` (settings: text size, mirror, count-in, fullscreen; privacy notice accepted), `scores.json`, `songs/` (players' imports) | no |
| `recordings/` | recorder sessions (`<date-time>/*.npz`, `session.json`, `analysis.json`) and plays (`game-*/play.npz`) | no |
| `logs/` | `game-*.json` (result of each play), `latency-test-*.json`, capture benchmarks | no |

Recordings hold landmarks only, never camera images.

## 6. Tools and commands

| Command | Use |
|---|---|
| `parapara-play` (or `parapara`) | the game |
| `parapara-record` | record calibration and move clips (hands-free) |
| `parapara-analyze [session]` | compare clips with the block library; recalibration data |
| `parapara-score [session]` | score clips with the real engine (`--latency-ms`, `--auto-latency`) |
| `parapara-tune-filters` | re-tune the smoothing filters on a session |
| `parapara-latency-test` | consistency of the clap latency calibration |
| `parapara-capture-debug`, `parapara-capture-bench` | camera / tracking speed and latency |
| `parapara-fetch-demo-songs`, `parapara-find-songs`, `parapara-import-song` | songs |
| `parapara-beatmap`, `parapara-beatcheck` | detect and verify beat maps |
| `parapara-routines`, `parapara-preview` | generate / check / add routines; watch one |

Hidden options used by tests and reviews (not for players): `--auto`,
`--auto-seconds`, `--fast`, `--keys "frame:key ..."` (scripted key presses),
`--shots DIR` (periodic screenshots), `--check-layout`, `--results-screenshot`.

## 7. Testing

```sh
uv run pytest                          # everything
uv run pytest tests/test_scoring.py    # one file
uv run pytest -k layout                # by name
```

How the suite is built:

- **Pure logic is tested in-process** (routine model, generator, filters,
  body frame, features, judge, scorekeeper, scores, notices, ...).
- **Anything with pygame runs in a subprocess** with the dummy SDL video and
  audio drivers (`SDL_VIDEODRIVER=dummy`), because of the cv2 / pygame rule
  (§3). `tests/test_game_app.py::_run_game` is the pattern: it imports a
  generated song into a temporary library, sets `PARAPARA_DATA_DIR` /
  `PARAPARA_SONGS_DIR` to temporary folders, accepts the privacy notice and
  drives the game with `--fast --keys "10:return 300:escape ..."`.
- **Fake cameras** (`tests/fake_capture_worker.py`): `stream` (a well-placed
  person at ~50 fps), `slow_stream` (~14 fps, dim light), `perfect_player`
  (dances a routine). Pass them as `worker="fake_capture_worker:stream"`.
- **Synthetic poses** (`tests/pose_synth.py`): MediaPipe landmarks built from
  body-frame arm directions (the inverse of `body_frame.py`), danced blocks,
  and `routine_play_samples()`: the **perfect player**, the demo dancer's own
  motion seen by a camera. It is the ground truth for timing: it must score
  100% with hits within ±40 ms (`test_play_session.py`).
- **Layout**: game, recorder and tool screens are run with `--check-layout`
  at text sizes 1.0, 2.0 and 2.6; any overlapping or cut-off text fails.
- pytest only collects `tests/` (`[tool.pytest] testpaths`).

Tests never touch the real data folder or the camera. Real-world checks are
done by the project owner with the guided tools (`parapara-record`,
`parapara-latency-test`), which write logs for analysis instead of asking for
numbers.

## 8. Common tasks

**Change a move's target pose.** Record clips (`parapara-record --bpm 150`),
run `parapara-analyze` and `parapara-score --latency-ms <your latency>`, move
the keyframe in `routine/library.py` towards the "seen" direction (fully in
x / y, partly in z: depth reads biased towards the camera), re-score, and
record the numbers in the library docstring. Judge sweep *timing* from game
plays, not recorder clips (the recorder's guide snaps between keyframes).

**Add a building block.** Define it in `library.py` (right arm; add the
mirrored variant), add it to `BLOCK_LIBRARY`, teach the generator a unit that
uses it (`generator/routine_generator.py`), check the animator's motion style
(`animation/animator.py`), and add tests in `test_routine_library.py` and
`test_animation.py` (the dancer must be exact on every keyframe).

**Add a song.** `parapara-import-song track.mp3 --title ... --artist ...`
(beat map, suitability check and routines are generated), then verify the
beat map by ear (`docs/verify-beat-maps.md`). For a new demo song, add it to
`assets/songs/catalog.json` with its licence and attribution, and to
`CREDITS.md`.

**Add or change a game screen.** Add a `Scene`, a `handle_*` and a `draw_*`
method in `game/app.py`; use `ui.text()` / `footer()` for text and
`register_box()` for framed boxes so the layout check covers them; add a
`--keys` flow test with `--check-layout` at sizes 1.0 and 2.6.

**Change a scoring constant.** Re-run the perfect player and the saved plays
and sessions (the approach in `judge.py` / `features.py` docstrings), keep
the perfect player at 100%, and document before/after numbers.

**Change the privacy notice.** Edit `notices.py`, check the facts against
the sources in its docstring, and bump `PRIVACY_VERSION` if the substance
changes (everyone sees it once more).

## 9. Glossary

| Term | Meaning |
|---|---|
| count, bar, phrase | one beat; 8 counts; 32 counts (4 bars) |
| block | one building-block move (sweep, knock, chop, wrist roll, flick + clap) |
| keyframe | a target pose (arm directions) at a count inside a block |
| body frame | coordinates relative to the dancer's torso: +x their right, +y up, +z forward |
| front-weighted error | pose error with the depth component scaled down |
| hold | a keyframe repeating the previous pose; graded on pose only |
| latency | total delay from the music to the judged movement (speakers, camera, model, player); measured by clapping |
| timing copy | the unsmoothed arm directions used to time hits |
| perfect player | the demo dancer's motion as synthetic camera input; the timing ground truth |
| framing | the player's position in the camera picture |
