# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **CI**: GitHub Actions workflow (`.github/workflows/ci.yml`) with jobs for running all tests, ruff lint and format checks, pyrefly type checking, and dependency audit with `pip-audit`.
- **Git hooks**: Added `prek.toml` for [prek](https://github.com/j178/prek) to run the same checks locally before commits (tests, ruff, pyrefly, pip-audit).
- **Tooling config**: Added ruff and pyrefly configuration in `pyproject.toml` and dev dependencies for ruff, pyrefly, prek, and pip-audit.

### Changed
- **Git hooks**: the `pytest` prek hook now uses `pytest-testmon`, so it runs
  only the tests affected by the staged changes instead of the full suite.
  The first run (or a run after `.testmondata` is removed) still collects
  coverage for every test; later runs are much faster. The cache is ignored
  by git (`.testmondata`).
- **Key hints in the UI**: player keys are now shown in brackets (e.g.
  `[Esc]`, `[Space]`, `[Up/Down]`) across the game, recorder, latency test,
  preview, beat check and document screens.
- **Brand colours in the UI**: every game screen now uses the five-colour
  palette in `assets/branding/PALETTE.md` (vibrant magenta, bright gold,
  midnight plum, electric violet, cool off-white). The new
  `parapara.game.theme` module is the single source for surfaces, text,
  accents, status colours, dancer/guide colours and grades; panels, muted
  text and status shades are explicit mixes of palette colours. A pure test
  (`tests/test_theme.py`) checks the constants against the palette file and
  the WCAG contrast of text/status shades on the plum background.
- Latency calibration accepts a clap spread up to **60 ms** (was 40 ms). The
  project owner's `parapara-latency-test` run (six rounds, 8 and 16 claps)
  had spreads of 18-70 ms from claps drifting within a round; 40 ms rejected
  4 of 6 rounds although every median was within 25 ms of the true ~166 ms.
  The game keeps 8 claps (16 were not more consistent). Tests: ±160 ms
  scatter still rejected, ±120 ms accepted.
- `PLAN.md`: clap detection by sound moved to the future plans (not for the
  first release; claps stay vision-only).

### Fixed
- **Dependency audit hook**: the `pip-audit` prek hook used the nonexistent
  `uv export --no-headers` flag, so the export always failed and the hook fell
  back to auditing the venv. It now uses `--no-header`, and `certifi` was
  upgraded to 2026.10.23 to clear PYSEC-2026-4320.
- `tests/test_game_overlay.py::test_debug_view_main_loop_runs_headless_with_fake_worker`
  was flaky: it stopped the debug loop after 40 display frames, but the
  spawned fake worker sometimes needed longer than 40 frames at 60 fps
  (~0.7 s) to report READY, so the log had no camera details. `max_frames`
  now counts frames after READY (with a five-times start-up grace so a
  broken worker cannot hang an automated run).

## [0.1.0] - 2026-10-23

The first playable version of ParaParaPy: pose capture with MediaPipe in its
own process, body-relative pose features, the scoring engine, songs with
verified beat maps and generated routines, the smooth stick-figure demo
dancer, the game (calibration, play, results, score board, learn the moves,
settings, credits, privacy notice), the hands-free recorder and the
analysis tools. Moves checked against a real dancer (Step 4b); six play
tests by the project owner shaped the screens. Not packaged yet (PLAN.md
§10).

### Added
- **Settings screen** (`O` in the menu): text size, mirror view, count-in
  (0 / 2 / 4 / 8 counts) and fullscreen; Up/Down choose, Left/Right change,
  saved at once in `ui.json` and kept between launches (mirror, count-in and
  fullscreen used to reset; `M` / `F` now save too). Command-line options
  (`--count-in`, `--window`, `--text-scale`) still override the saved
  values. Test: changing and saving, no overlapping text at sizes 1.0 / 2.6.
- **Version 0.1.0** shown in the menu and on the settings screen
  (`parapara.__version__`); this release section.
- `CONTRIBUTING.md`: a guide to the code base: getting started, ground
  rules, the two-process architecture and a frame's journey, every package
  and module, data on disk, commands, how the tests are built (subprocess
  pygame tests, fake cameras, the perfect player, the layout check), common
  tasks (recalibrating a move, adding a block, a song or a screen) and a
  glossary. Linked from the README.
- **Low-light hint** (`LowLightMonitor`): "Camera N fps: more light helps"
  on a dark band at the bottom of the camera view while the camera delivers
  under 25 fps for 3 s (webcams slow down in dim light; ~23 fps was seen
  against the requested 30); other camera warnings move above it. Each
  play's log records `camera_fps` and `low_light_pct`. Tests: the monitor's
  hold / recovery, a ~14 fps fake camera in the game.
- **Score board** (`parapara.scores`, `scores.json` in the data folder): the
  best 20 finished plays per song and level (plays ended early are not
  ranked). The results screen says "New best!" or "#2 of 5 on this level"
  and lists the board with your play highlighted; the selected song shows
  its best score; `S` in the menu opens the score board (Left/Right level).
  The placement is logged with each play. Tests: ranking, levels, cap,
  broken file; the game flow and layout at text sizes 1.0 and 2.6.
- **Player guide**: `docs/how-to-play.md` (setup, first start, menu, learning
  the moves, calibration, playing, grades, score board, troubleshooting),
  linked from the README.
- **Credits** (`C` in the menu of `parapara-play`): demo songs with title,
  artist, CC BY licence and source; MediaPipe and its pose models (Apache
  2.0), pygame-ce, NumPy, OpenCV; the dance-move reference; the logo. The
  selected song's box shows its artist and licence. `CREDITS.md` has the
  full attributions with links.
- **Privacy & licence notice** (`parapara.notices`,
  `parapara.game.notice_screens`): shown full-screen before the camera starts
  for the first time in the game, the recorder and the latency test (`Enter`
  accepts and is remembered in `ui.json` per notice version, `Esc` quits
  without starting the camera), and from the menu (`P`). It says camera
  images are processed on the computer and never saved or sent, that
  MediaPipe sends Google usage statistics (no images; cannot be switched
  off; facts checked against Google's notice of 2026-06-05), what the game
  saves locally, and the licences. Long texts scroll (Up/Down) and keep
  headings with their first line.
- Tests: notice and credits wording, the first-start gate (Esc quits, Enter
  accepts and is remembered), Credits / Privacy screens without overlapping
  text at text sizes 1.0 and 2.6. Tests use a temporary data folder.
- **`parapara-latency-test`**: a separate, hands-free test of the latency
  calibration (the game is unchanged): position check, then rounds
  alternating 8 and 16 claps (`--rounds`, `--claps`) with the game's click
  track, tempo, count-in and estimator, a summary, closes by itself. Logs
  per round the click times, the result as the game computes it and on the
  smoothed signal, each clap's offset and the wrist-distance series.
  Test: headless run and log format.
- Demo songs: the beat maps of 80s Motivational Chiptune, Onion March and Goa
  Trolls accepted by the project owner (marked verified locally; the
  verification is not yet carried by `catalog.json`, see PLAN.md §10).
- **ParaParaPy**: the project's name, with the project owner's logo
  (`assets/branding/paraparapy-logo.png`). `parapara.game.branding` derives
  the menu title (the wordmark, cropped) and the window / dock icon (the
  first "P"'s flower, cut to a circle); `python -m parapara.game.branding`
  writes them as PNGs (the wordmark heads the README). Window titles say
  ParaParaPy. The distribution is `paraparapy`; the import package, the
  commands and the user data folder keep the name `parapara`, so saved
  calibrations and scripts keep working. `parapara` now starts the game.
  Test: wordmark crop and round icon.
- **Hands-free recorder** (`parapara-record`; the player stands back and
  cannot reach the keyboard): each clip starts by itself once the position
  check passes and the preview has run, is quality-checked after recording
  (person seen >= 90%, hips and shoulders in the picture >= 90% of the time)
  and kept, or redone with the reason shown (up to 3 attempts; the last is
  kept and marked `quality_ok: false` in `session.json`, with `attempts`).
  The session closes by itself at the end. `--manual` keeps the key flow.
  Block clips: one **8-count count-in bar** (8, 7, ... 1; was 4 beats) that
  starts on count 1 of the preview's next bar, so preview, count-in and
  recording keep one unbroken 8-count rhythm (`ClipTimeline.next_preview_bar`
  replaces `next_preview_beat`).
  Fullscreen with the game's large text and layout (guide figure left,
  camera right), `+` / `-`, `F`, `--window`, `--text-scale`. Tests: the
  hands-free run, the quality check, no overlapping text at sizes 1.0/2.6.
- `parapara.game.ui`: the game's text helpers (text size, word wrap,
  footer, layout check, fullscreen flags) shared by the game and the recorder.
- `PLAN.md` §10: pre-release checklist (must do / should do / after
  shipping) agreed after the sixth test run.
- `parapara-play` third test-run feedback:
  - **4-bar count-in** (8-count bars) with metronome clicks before the first
    move, counted above the dancer ("3 bars", "last bar!"). Short intros get
    silence in front so the count-in always fits; `--countdown-bars N`
    changes it (0 = none).
  - **Layout check**: `text()` records what it draws and `layout_issues()`
    reports overlapping or cut-off text (`--check-layout`, hidden). Tests run
    every screen at text sizes 1.0, 2.0 and 2.6 with a long song list and a
    long title, and check that the checker itself catches overlaps.
  - `--reuse-calibration`: use the saved calibrations instead of calibrating
    at launch (tests, development).
- **Learn the moves** (`T` in the `parapara-play` menu, for new players): the
  demo dancer loops one move at a time with metronome clicks at 50 / 75 /
  100 % of the selected song's tempo (Up/Down), with its name and
  description; Left/Right go through all 18 moves, simplest kinds first. The
  mirrored camera view lets you try it; nothing is scored.
- `Animator.for_blocks(blocks)`: animate any block sequence without the
  phrase rules (used by Learn the moves).
- Tests: every library move loops exactly (`test_animation.py`), the Learn
  the moves flow (`test_game_app.py`).

### Changed
- Text keeps runs of spaces inside a line (column gaps such as
  "1.   43,826   79%"; they had collapsed to one space).
- pytest only collects `tests/` (`[tool.pytest] testpaths`): it had also
  imported `src/` modules named `*_test.py`, which loaded pygame into the
  test process next to cv2 (caught by the conftest guard).
- Latency calibration reads the claps from the same unsmoothed timing copy
  the judge times hits on (it still used the smoothed wrist distance, ~28 ms
  later, after the timing change). Test helper updated.
- **Hit timing on the unsmoothed arms** (`PoseFeatures.for_timing`, used by
  the judge for keyframes and claps; poses are still scored on the smoothed
  arms). Through the landmark smoothing a pose's error kept creeping down
  40-130 ms after the arm had stopped, so held poses were graded late: the
  demo dancer as a perfect player now scores 100% (was 91-94%) with hits
  within ±16 ms; the day's game plays 75.6% -> 76.2%. Tests: perfect player
  timed within ±40 ms; the timing copy has no smoothing lag.
- **Chop reach 25°** (was 45°): as danced at game speed in every recording;
  with the timing fix it no longer costs points.
- **Step 4b, block poses checked against a real dancer** (two hands-free
  sessions at 150 BPM on 2026-10-23 and the day's 6 game plays):
  - Flick + Clap recalibrated: the front and side flicks' forearm targets
    now point forward and slightly up, as danced in both sessions (score at
    120 ms latency 39% / 26% -> 59% / 52%; arm error 18° -> 9-11°).
  - Wrist roll marked as checked (13.6-14.6° error in three sessions).
  - Chop keeps its 45° reach: a 25° target fit the recordings better but the
    judge then timed the reach later (-5 points); on hold until the
    held-pose timing fix (PLAN.md §10).
  - Sweeps unchanged: on time in the game plays; the recorder's lag comes
    from its snapping guide figure (kept as it is: the recorder works fine).
  - Test: the flick's recalibrated forearm directions.
- `parapara-play` sixth test-run feedback:
  - Latency calibration: the 4 count-in clicks are counted down on screen
    (4, 3, 2, 1, flashing on each click), then CLAP!.
  - "next" box: the four counting dots are replaced by a **timing dot** that
    slides at constant speed along a track into a ring on the left during
    the last 4 counts, arriving exactly when the move starts; the ring then
    lights up for a moment before the box shows the following move.
- `parapara-play` fifth test-run feedback: the body calibration shows the
  pose to hold (neutral pose, T-pose) as a stick figure between the
  instructions and the countdown.
- `parapara-play` fourth test-run feedback:
  - **Latency calibration: 8 claps** instead of 16 (still needs 6 to be
    accepted; test with 8 synthetic claps).
  - **The music starts right after the song is chosen**: the 4-bar count-in
    with clicks and silence is replaced by a visual **4-count count-in**
    (4, 3, 2, 1 above the dancer, flashing on each count). Silence is added
    only if a song's intro is shorter than that. `--countdown-bars` is now
    `--count-in` (counts, default 4).
  - "next" box: **dots instead of "in N"**: four rings that fill in the last
    four counts before the move and go out one per count.
- `parapara-play` third test-run feedback:
  - **Calibrate at every launch**, not between songs of the same session
    (body and latency); `S` on the latency screen reuses the last saved
    latency.
  - The "next" box shows a **still picture** of the move's first pose (the
    looping animation was distracting).
  - The **selected song is framed** with its level, tempo and beat-map note;
    the song list scrolls when it is taller than the screen.
  - **Text placement**: text flows from the top, key hints stack from the
    bottom, big labels move down when needed, too-wide words shrink, and
    the grade pop-up stays left of the "next" box. Fixed the overlaps found
    by the new layout check (e.g. score and combo at large text sizes).
- **Game screens readable from the dancing spot** (first test run, 2026-10-23:
  the text was too small to read from 1.5-2 m away).
  - `parapara-play` opens **fullscreen** (`pygame.SCALED`, the 1240x720
    layout scaled to the display); `F` toggles a window, `--window` starts in
    one.
  - All text is scaled by a **text size** (default 2.0, range 1.0-2.6):
    `+` / `-` change it in the game, `--text-scale` sets it; the choice is
    saved in `ui.json` next to the calibrations (`settings.save_ui` /
    `load_ui`).
  - Screens redrawn with fewer, larger lines: big count, move card, grade
    pop-up, CLAP prompt, warnings as large text; long messages word-wrap.
  - The grade pop-up (Perfect / Great / Good / Hold / Miss) moved from the
    camera view to just above the demo dancer's head, where players look
    while dancing.
  - The **next move is a picture**, not text: a small framed stick figure at
    the dancer's top right dances the coming move in a loop (upper body only,
    so it can be drawn large), with "in N" counts. It replaces the text card
    at the bottom, which was hard to read while dancing.
  - The count moved below the dancer's feet; the dancer moved left to make
    room for the picture.
  - The latency screen says the calibration is needed only once (saved;
    redo with L after changing camera, speakers or headphones).
  - Tests: text-size keys and saving (`test_game_app.py`), `ui.json` round
    trip (`test_play_session.py`).

### Added
- `docs/verify-beat-maps.md`: step-by-step guide to verifying a song's beat
  map by ear (linked from the README, PLAN.md, `parapara-beatcheck` and the
  unverified-song warning).
- **Step 6 – the game** (`uv run parapara-play`).
  - `scoring.session.PlaySession`: features → judge (schedule on the song's
    beat clock shifted by the measured latency) → score, framing warnings,
    the play kept as a recording; ending early drops moves not yet due.
  - `scoring.latency.estimate_latency`: "clap to the beat" calibration
    through the scoring's own feature pipeline (median clap offset, spread
    check).
  - `parapara.settings`: latency and body calibration saved per machine
    (`$PARAPARA_DATA_DIR` or `~/Library/Application Support/parapara`).
  - `game.app`: menu (songs, routines, unverified beat maps flagged; B / L
    redo calibrations), body calibration (position check, neutral pose,
    T-pose), latency calibration (16 clicks), play (demo dancer and mirrored
    camera side by side, next-move card, grade pop-ups, score, combo,
    progress, out-of-picture warning), results (also `logs/game-*.json` and
    `recordings/game-*/play.npz`).
  - Tests: `test_play_session.py`, `test_game_app.py` (headless flows with
    a streaming fake camera and scripted keys).

### Fixed
- Judge timing for **held poses**: the hit is when the pose was reached
  (start of the near-minimum stretch lasting ≥ 0.4 beat), not the most
  exact sample of the hold, which graded hit-and-hold dancing ~100 ms late
  (found with the demo dancer as a perfect player).
- `parapara-beatcheck`: skip keys `,` / `.` (one phrase back / forward) and
  End (last 15 s), to check a beat map late in a song.
- `PLAN.md` §7: demo dancer reviewed (on beat, looks good); flick + clap
  animation to be made clearer later.
- **Step 5c – demo dancer.**
  - `parapara.animation.animator.Animator`: routine + beat → `DancerPose`,
    exact on every scored keyframe; motion styles snap / glide / roll /
    knock pulse / flick whip / clap / rest; centred two-step and beat bounce;
    current and next block (one bar ahead).
  - `parapara.game.dancer_view`: smooth stick-figure renderer (mirrored,
    knees by IK, depth-sized wrists), `static_pose` for previews.
  - `uv run parapara-preview [song] [--routine R] [--clicks]`: watch the
    dancer perform a routine to its song, with count/bar/phrase, current move
    and a next-move card; pause, seek by bar, switch routine, clicks, mirror.
  - Tests: `test_animation.py` (exact on scored beats, smoothness at
    160 BPM / 120 fps, snap holds, glides, roll, knocks, clap, rest arms,
    two-step, preview; headless rendering and app).
- `PLAN.md` future work: an interface to record a routine with the camera;
  a demo song with vocals (none of the demo tracks has vocals; "Lets Go To A
  Rave" confirmed instrumental).
- **Step 5b – routines for every song** (roadmap Step 7 brought forward).
  - `parapara.generator.routine_generator`: choreography-unit generator
    (knock R→L, chops RRLL, mirrored sweep pairs, wrist rolls, flick + clap
    finishers), easy / normal / hard settings (unit weights, 2-4 distinct
    phrases, AABB or ABCD repetition), deterministic per song.
  - `parapara.routine.store`: routines stored next to their song
    (`<song>/routines/*.json`), fit check against the song length,
    `ensure_generated`, `install_routine` for players' own routines.
  - `uv run parapara-routines generate|list|blocks|template|check|add`.
  - Fetching and importing songs now generate their routines.
  - `Difficulty.CUSTOM` for player-written routines; unknown move names in a
    routine file get a "did you mean ...?" hint.
  - Tests: `test_routine_generation.py`.
- **Step 5 additions – song import and demo catalog.**
  - `audio.suitability`: Para Para suitability check (tempo 135–165 BPM,
    steady tempo, beats on a constant grid, beat clarity from kick on- vs.
    off-beat, length, declared licence; NC/ND/SA warnings). Thresholds from
    measurements of the demo track and ~50 CC BY tracks.
  - The generated track is now a **machine-testing fixture only**, not for
    dancing: `assets/songs/demo/` and the `parapara-make-demo-song` command
    were removed; the demo routine now targets the CC BY track "Lets Go To A
    Rave" (`defrini-lets-go-to-a-rave`); `parapara-beatcheck` needs a song
    folder argument.
  - Demo catalog: "broke" (Shane Ivers) replaced by "Goa Trolls"
    (Soulbringer, CC BY 4.0, 145 BPM, beat clarity 4.3) after a player found
    "broke" hard to dance to; the beat-clarity warning threshold rose from
    1.5 to 1.8. No CC BY vocal dance track passed the check yet (open item).
  - `uv run parapara-fetch-demo-songs`: downloads the four catalog tracks
    into `assets/songs/<id>/` (git-ignored; `--from-dir` for a shared drive)
    and imports them with licence and attribution.
  - **Licences are optional** for user songs (personal, local use):
    `song.json` no longer requires `license`/`source`/`artist` (missing
    licence = "unspecified (personal use)"), the import no longer requires
    them, and the suitability check no longer rejects on licence. Libraries
    skip songs whose audio file is missing (`skip_missing_audio`).
  - `uv run parapara-import-song <file> ...` (`audio.importer`): checks,
    rejects unsuitable tracks with reasons, copies accepted ones into the user
    song library outside git (`$PARAPARA_SONGS_DIR` or
    `~/Library/Application Support/parapara/songs`) with an unverified beat
    map; `--catalog <id>` fills the metadata. `load_all_songs` merges the
    built-in and user libraries.
  - `assets/songs/catalog.json`: four vetted CC BY 4.0 demo candidates from
    Free Music Archive (metadata, attribution, measured tempo; no audio in
    git), plus rejected examples and the search notes.
  - `JAMENDO_CLIENT_ID=... uv run parapara-find-songs` (`audio.find_songs`):
    Jamendo API search for plain CC BY, downloadable, high-tempo dance tracks,
    with the suitability check on each.
  - `.gitignore`: MP3/OGG/FLAC files and Jamendo search results.
  - Tests: `test_song_import.py`.
- **Step 5 – music library & beat maps** (`parapara.audio`).
  - `synth`: procedurally generated "Eurobeat-like" demo track (150 BPM,
    intro / 64-count main section / outro; kick, off-beat hats, claps on 2 and
    4, octave bass, saw-lead arpeggios), deterministic, exact beat map.
    **For machine testing only** (`audio.demo_song.make_demo_song` renders it
    into temporary test folders); it is not in the game's song library.
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
- Removed `assets/routines/`: routines live only next to their song. The
  hand-written Step 1 sample moved to `tests/fixtures/sample-routine.json`
  (test data).
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
