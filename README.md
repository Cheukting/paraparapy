# parapara

A desktop Para Para dance game: a webcam tracks the player's upper body and
scores each move against a choreographed routine. See [PLAN.md](PLAN.md) for
the design and roadmap, and [CHANGELOG.md](CHANGELOG.md) for progress.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```sh
uv sync
uv run pytest
```

## Capture debug view

```sh
uv run parapara-capture-debug            # --help for options (--model lite, ...)
```

On first run it downloads ~15 MB of MediaPipe pose models to
`~/Library/Caches/parapara/models`. macOS will ask for camera access for your
terminal or IDE. Cyan = dancer's left, orange = dancer's right; `M` toggles
the mirrored view, `Esc` quits.

## Capture benchmark

```sh
uv run parapara-capture-bench            # ~1 min: dance in front of the camera
```

Shows the live camera view while it runs the full and lite pose models, then
writes a comparison to
`logs/capture-bench-*.json`. `Esc` stops early. A version without a window:
`uv run python -m parapara.diagnostics.bench`.

## Guided recorder

```sh
uv run parapara-record                   # --bpm 100 for a slower metronome
```

Walks you through calibration poses and one clip per building block, with a
guide figure and a metronome. Clips are saved (landmarks only, no images) to
`recordings/<date-time>/`.

## Pose analysis / recalibration

```sh
uv run parapara-analyze                  # latest session in recordings/, or pass a path
```

Compares the recorded clips with the block library (danced vs. target arm
directions, best lag, calibration values) and writes `analysis.json` into the
session folder. Use it to recalibrate `src/parapara/routine/library.py`. If
the session has neutral-pose and T-pose clips it also saves the player
calibration (`calibration.json`).

```sh
uv run parapara-tune-filters             # re-tune the smoothing filters on a session
```

## Routines

```sh
uv run parapara-routines generate        # easy/normal/hard routines for every song missing them
uv run parapara-routines list            # songs and their routines
uv run parapara-routines blocks          # the moves you can use
uv run parapara-routines template --song <id> -o mine.json   # start your own routine
uv run parapara-routines check mine.json --song <id>         # validate it
uv run parapara-routines add mine.json --song <id>           # install it
```

Routines are generated automatically when a song is fetched or imported, and
stored next to the song (`<song folder>/routines/`). A routine is a list of
32-count phrases, each a list of move names; every phrase needs all five
kinds of move and ends with a flick + clap.

## Scoring recorded clips

```sh
uv run parapara-score                    # latest session; --latency-ms N or --auto-latency
```

Runs the block clips through the real scoring engine (grades, accuracy,
combo, flow, posture) and writes `score.json` into the session folder.

## Songs

```sh
uv run parapara-fetch-demo-songs         # the 4 CC BY demo tracks -> assets/songs/ (git-ignored)
uv run parapara-beatmap assets/songs/<id> --audio track.ogg   # detect a new track's beat map
uv run parapara-beatcheck assets/songs/<id>                   # verify it by ear, then save
```

Every song folder has a `song.json` with its licence, source and beat map.

### Adding your own songs

```sh
uv run parapara-fetch-demo-songs                # the 4 CC BY demo tracks -> assets/songs/ (git-ignored)
uv run parapara-import-song track.mp3           # your own song; --title/--artist optional
```

Your own songs are for personal, local play, so no licence information is
needed. The track is checked first (about 140–160 BPM, steady tempo, clear
beat, long enough) and rejected with reasons if unsuitable. Accepted
songs go to your own library (`~/Library/Application Support/parapara/songs`,
or `$PARAPARA_SONGS_DIR`), never into git. Verify the beat map with
`parapara-beatcheck` afterwards. To search Jamendo for CC BY candidates, get a
free developer id at https://devportal.jamendo.com and run
`JAMENDO_CLIENT_ID=... uv run parapara-find-songs`.
