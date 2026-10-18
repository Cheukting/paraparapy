# Verifying a song's beat map

Every song has a **beat map** in its `song.json`: the tempo (`bpm`), the time
of the first beat (`first_beat_s`) and the beat on which the routine's count 1
falls (`routine_start_beat`). The scoring judges every move against these
times, so an error of only ~30 ms is already noticeable, and a tempo off by
0.1 BPM drifts by ~120 ms over three minutes.

Imported and fetched songs get an **automatic** beat map (`parapara-beatmap`:
librosa on the kick-drum band plus a constant-tempo fit). It is usually close,
but it is marked **unverified** until a person has listened to it. The game
flags unverified songs in the menu.

Verify each song once, before playing it seriously. It takes about 5 minutes.

## What you need

- The song in the library (`uv run parapara-routines list` shows all songs).
- Headphones or speakers.

## Steps

```sh
uv run parapara-beatcheck assets/songs/<song id>      # built-in demo tracks
uv run parapara-beatcheck "<your song library>/<song id>"   # your imports
```

The app plays the song with **metronome clicks mixed in** (a higher click on
count 1 of every 8-count), a big **count 1-8** that flashes on each beat, and a
strip with the song's **kick-drum energy** (blue) under the **beat grid**
(lines; orange = count 1; white = now).

### 1. The start: are the clicks on the kicks?

During the first 30 s, listen: the clicks should sit exactly on the kick drum.
In the strip, every line should touch the **left edge** (attack) of a blue
spike.

| You notice | Press |
|---|---|
| clicks slightly **late** | **←** (first beat 5 ms earlier; **Shift+←** 1 ms) |
| clicks slightly **early** | **→** |

Differences of 5-10 ms can be left alone.

### 2. The middle and the end: do they stay on?

Press **`.`** a few times (forward one phrase = 32 counts each), then **End**
(the last 15 s).

| You notice | Press |
|---|---|
| clicks **drift later** towards the end | **↑** (tempo +0.01 BPM; **Shift+↑** +0.1) |
| clicks **drift earlier** | **↓** |

After changing the tempo, press **Home** and check the start again (a tempo
change shifts all later beats).

### 3. Where is count 1?

Press **Home**. Listen for where the song's musical phrases start, where a
melody or section begins (often with a crash cymbal or a new instrument): that
is a natural "1", and it should be an **orange** line.

- Orange lines a few beats off: **`[`** (count 1 one beat earlier) or **`]`**
  (one beat later) until they line up.
- Long quiet intro: press **`]`** repeatedly to start the routine where the
  dancing should begin (the count goes from "intro" to "1").

The panel shows "routine starts on beat N".

### 4. Save

Press **S** (writes the values and marks the beat map **verified**), then
**Esc**.

Other keys: **Space** pause / resume, **`,`** back one phrase.

### 5. Only if you changed the routine start (step 3)

The song's routines were sized for the old start. Regenerate them and watch one
to confirm the dancer's count 1 lands on the music's phrase starts:

```sh
uv run parapara-routines generate --force <song id>
uv run parapara-preview <song id> --routine easy --clicks
```

Your own routines (`custom`) are not regenerated; check them with
`uv run parapara-routines check <file> --song <song id>`.

## Quick check without the beat-check app

`uv run parapara-preview <song id> --clicks` plays the song with clicks while
the demo dancer dances. If the dancer visibly hits the beat all the way through,
the tempo and first beat are fine. This does not mark the song verified and
does not check where count 1 is.

## History

- 2026-10-18: "Lets Go To A Rave" verified by ear by the project owner (no
  changes: 140.001 BPM, first beat 5.18 s, routine start on beat 8).
