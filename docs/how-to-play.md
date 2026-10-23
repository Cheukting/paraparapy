# How to play ParaParaPy

ParaParaPy is a Para Para dance game: a stick-figure dancer shows the moves,
your webcam watches your **upper body**, and every move is graded on how
close and how on-time it was. Footwork (the side-to-side two-step) is shown
as a guide only and never scored, so you can also play seated.

```sh
uv run parapara-play        # or: uv run parapara
```

## 1. Set up

- **Camera**: about chest height, facing you.
- **Distance**: stand about 2-2.5 m back. Your **head, both arms stretched
  out and your hips** must be in the picture: the game checks this and tells
  you to step back, closer or to the side.
- **Light**: a bright room. In dim light webcams deliver fewer pictures per
  second, which makes the timing coarser; the game shows
  "Camera N fps: more light helps" when that happens.
- **Sound**: speakers or headphones, either works (the clap calibration
  measures the delay of your setup).
- **Text size**: press `+` / `-` from where you stand until you can read the
  screen comfortably; it is remembered. `F` switches between fullscreen and
  a window.

## 2. The first start

The game first shows a **privacy and licence notice**: your camera picture is
processed on this computer and never saved or sent; Google's MediaPipe pose
tracking sends Google usage statistics (no images); each song you play is
kept as body positions (no images) in `recordings/`. Press `Enter` to accept
and start the camera, or `Esc` to quit. `P` in the menu shows it again.

## 3. The menu

| Key | |
|---|---|
| Up / Down | choose a song (the selected one is framed, with its artist, licence and your best score) |
| Left / Right | choose the level: easy, normal, hard (and your own routines) |
| Enter | play |
| T | **learn the moves** (recommended before your first song) |
| S | score board of the selected song |
| C / P | credits / privacy and licences |
| B / L | redo the body / clap calibration before the next song |
| + / - , F, M | text size, fullscreen, mirror the camera view |
| Esc | quit |

## 4. Learn the moves (`T`)

The dancer shows one move at a time with metronome clicks, slowed down:
Left/Right for the next move, Up/Down for the speed (50 / 75 / 100 %). Your
camera view is next to it so you can copy it. The moves:

- **Knock**: a fist knocks twice on an imaginary door (upper arm out to the
  side, forearm up).
- **Chop**: the arm reaches up and out to the side, then folds in front of
  the chest like a salute.
- **Wrist roll**: hands in front of the chest, wrists rolling over each other.
- **Sweep**: both forearms sweep together from one side to the other, or up
  and down.
- **Flick + Clap**: the hand flicks across the body, in front, then to the
  side, and you clap on count 4. Any clap counts.

Orange is the dancer's right arm, cyan the left; the dancer is mirrored, so
copy it like a mirror.

## 5. Calibration (once per game start)

Before your first song after starting the game:

1. **Body**: stand in position, then hold the **neutral pose** (arms down,
   relaxed) and the **T-pose** (arms straight out) for a few seconds each; a
   stick figure shows each pose. `R` retries.
2. **Clap to the beat**: press `Space`, listen to 4 clicks (4, 3, 2, 1 on the
   screen), then clap on each of the next 8 clicks. The result is the delay
   of your camera, screen and speakers. `Enter` accepts, `R` retries.
   **Don't skip it** (`S`): without it the grades come out late and unfair.

The songs after that use the same calibration. Redo it (`B` / `L`) if you
move the camera, change speakers or headphones, or someone else plays.

## 6. Playing

- The **music starts** straight away; **4, 3, 2, 1** above the dancer counts
  you into the first move.
- **Dance with the dancer** on the left (your camera view is on the right).
  The **count** (1-8) is under its feet.
- **Next move**: the small framed picture at the dancer's top right shows the
  coming move. A **dot slides into the ring** on its left; the move starts
  when the dot reaches the ring.
- **Grades** pop up above the dancer's head:
  - **PERFECT** / **GREAT** / **GOOD**: the pose was right, hit within
    60 / 120 / ~190 ms of the beat;
  - **HOLD**: the same pose held on the next count (e.g. the second knock);
  - **MISS**: the pose was too far off, or not seen.
- Consecutive hits build a **combo**; good posture adds a bonus.
- If you leave the picture, a warning tells you how to get back in.
- `Esc` ends the song early (an early end is not put on the score board).

## 7. Results and the score board

After the song: score, accuracy, best combo and the grade counts. A finished
song goes on the **score board** of that song and level ("New best!" or
"#2 of 5 on this level"), shown on the results screen; `S` in the menu shows
the whole board (Left/Right for the other levels). Scores are kept on this
computer only.

## 8. Troubleshooting

| You see | Try |
|---|---|
| "CAMERA PROBLEM" | close other apps using the camera; on macOS allow camera access for your terminal |
| "Step back ..." / "Move a little to your LEFT." | follow it until your hips and both stretched arms fit in the picture |
| "Camera N fps: more light helps" | turn on more light or face a window |
| Grades always late or early | redo the clap calibration (`L` in the menu) and clap exactly on the clicks |
| Text too small / too big | `+` / `-` |
| Calibration "Try again" | hold the pose still, arms fully out for the T-pose, whole body in the picture |

Credits for the music and software: `C` in the menu, or [CREDITS.md](../CREDITS.md).
