"""Verify a song's beat map by ear and eye: ``uv run parapara-beatcheck [song dir]``.

Plays the song with metronome clicks mixed into the audio at the current
beat grid (sample-accurate, see :mod:`parapara.audio.beatgrid`) and shows:

- a big **count 1-8** that flashes on every beat, with bar and phrase number;
- a scrolling strip of the song's **kick-drum energy** with the grid's beat
  lines on top (count 1 of every 8-count highlighted), from 1 s in the past
  to 2 s ahead: in a good beat map every line sits on the start of a kick;
- the current tempo, first-beat offset, routine start and the detected drift.

Keys::

    ← / →          first beat −/+ 5 ms   (Shift: 1 ms)
    ↓ / ↑          tempo −/+ 0.01 BPM    (Shift: 0.1)
    [ / ]          routine start (count 1) −/+ 1 beat
    , / .          back / forward one phrase (32 counts)
    Space          pause / resume        Home  restart     End  last 15 s
    S              save as verified      Esc   quit

Saving writes ``bpm``, ``first_beat_s``, ``routine_start_beat`` into
``song.json`` and marks the beat map ``verified`` (method ``manual`` if
anything was changed). It edits the JSON directly, so a song whose licence
is still missing can be beat-checked too.

Audio: WAV is read with the standard library; other formats (OGG, FLAC, MP3)
need ``soundfile``, installed with the tools dependency group.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402  (must follow the env var above)

from parapara.audio.beatgrid import COUNTS_PER_BAR, BeatGrid, low_band_envelope, render_click_track  # noqa: E402
from parapara.audio.synth import read_wav  # noqa: E402

WIDTH, HEIGHT = 900, 520
STRIP_PAST_S, STRIP_AHEAD_S = 1.0, 2.0
ENVELOPE_HOP_S = 0.005
BG, TEXT, DIM = (20, 20, 30), (235, 235, 235), (110, 110, 130)
ACCENT, ENERGY, LINE, BAR_LINE, NOW = (255, 220, 80), (70, 140, 220), (200, 200, 200), (255, 120, 80), (255, 255, 255)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() == ".wav":
        try:
            return read_wav(path)
        except ValueError:
            pass  # not 16-bit PCM: fall back to soundfile
    import soundfile  # tools dependency

    data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


class Player:
    """Plays a mono float buffer from any position (re-mixable on the fly)."""

    def __init__(self, sr: int) -> None:
        pygame.mixer.init(frequency=sr, size=-16, channels=1, buffer=512)
        self.sr, _, self.channels = pygame.mixer.get_init()
        self._sound: pygame.mixer.Sound | None = None
        self._start_pos = 0.0
        self._started_at: float | None = None  # None = paused

    def position(self) -> float:
        if self._started_at is None:
            return self._start_pos
        return self._start_pos + time.monotonic() - self._started_at

    def play(self, mix: np.ndarray, pos: float) -> None:
        if self._sound is not None:
            self._sound.stop()
        start = int(max(0.0, pos) * self.sr)
        data = (mix[start:] * 32767).astype(np.int16)
        if self.channels > 1:
            data = np.repeat(data[:, None], self.channels, axis=1)
        self._sound = pygame.sndarray.make_sound(np.ascontiguousarray(data)) if len(data) else None
        if self._sound is not None:
            self._sound.play()
        self._start_pos, self._started_at = max(0.0, pos), time.monotonic()

    def pause(self) -> None:
        self._start_pos = self.position()
        self._started_at = None
        if self._sound is not None:
            self._sound.stop()

    @property
    def paused(self) -> bool:
        return self._started_at is None


def _text(screen, text, pos, size=24, colour=TEXT):
    screen.blit(pygame.font.Font(None, size).render(text, True, colour), pos)


def draw(screen: pygame.Surface, grid: BeatGrid, env: np.ndarray, pos: float, duration: float,
         info: dict, dirty: bool, paused: bool) -> None:  # fmt: skip
    screen.fill(BG)
    beat = grid.beat_at(pos)
    current = int(np.floor(beat))
    rel = current - grid.routine_start_beat
    count = rel % COUNTS_PER_BAR + 1
    flash = 0 <= beat - current < 0.15
    colour = ACCENT if flash and grid.is_bar_start(current) else (TEXT if flash else DIM)
    _text(screen, str(count) if current >= 0 else "-", (WIDTH // 2 - 30, 40), 160, colour)
    where = "intro" if rel < 0 else f"bar {rel // COUNTS_PER_BAR + 1}  phrase {rel // 32 + 1}"
    _text(screen, where, (WIDTH // 2 - 60, 170), 28, DIM)

    # Kick energy with the beat grid on top.
    top, h = 230, 140
    t0, t1 = pos - STRIP_PAST_S, pos + STRIP_AHEAD_S

    def x_of(t: float) -> int:
        return round((t - t0) / (t1 - t0) * WIDTH)

    i0, i1 = max(0, int(t0 / ENVELOPE_HOP_S)), min(len(env), int(t1 / ENVELOPE_HOP_S))
    for i in range(i0, i1):
        x, v = x_of(i * ENVELOPE_HOP_S), float(env[i])
        pygame.draw.line(screen, ENERGY, (x, top + h), (x, top + h - round(v * h)))
    for b in grid.beats_between(max(0.0, t0), t1):
        x = x_of(grid.time_of(b))
        pygame.draw.line(screen, BAR_LINE if grid.is_bar_start(b) else LINE, (x, top - 10), (x, top + h), 2)
    pygame.draw.line(screen, NOW, (x_of(pos), top - 18), (x_of(pos), top + h + 8), 1)
    _text(screen, "kick energy (blue) vs. beat grid (lines; orange = count 1)", (16, top + h + 12), 20, DIM)

    lines = [
        f"tempo {grid.bpm:.3f} BPM    first beat {grid.first_beat_s * 1000:.1f} ms    "
        f"routine starts on beat {grid.routine_start_beat}",
        f"detected drift {info.get('drift_ms', '?')} ms    method {info.get('method')}    "
        f"verified {info.get('verified')}{'   (UNSAVED CHANGES)' if dirty else ''}",
        f"{pos:6.1f} / {duration:.1f} s{'   PAUSED' if paused else ''}",
        # Key names spelled out: pygame's default font has no arrow glyphs.
        "Left/Right offset 5 ms (Shift 1)   Down/Up tempo 0.01 (Shift 0.1)   [ ] count 1   "
        ", . phrase back/fwd   End last 15 s   Space pause   Home restart   S save   Esc quit",
    ]
    for k, line in enumerate(lines):
        _text(screen, line, (16, HEIGHT - 110 + k * 24), 21, TEXT if k < 3 else DIM)


def save(song_dir: Path, grid: BeatGrid, changed: bool) -> Path:
    path = song_dir / "song.json"
    data = json.loads(path.read_text())
    beat_map = data.get("beat_map", {})
    note = f"verified by ear {datetime.now():%Y-%m-%d %H:%M}"
    data |= asdict(grid) | {"beat_map": beat_map | {
        "method": "manual" if changed else beat_map.get("method", "manual"),
        "verified": True,
        "notes": (beat_map.get("notes", "") + "; " + note).strip("; "),
    }}  # fmt: skip
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def run(argv: list[str] | None = None) -> BeatGrid:
    parser = argparse.ArgumentParser(description="Verify a song's beat map by ear")
    parser.add_argument("song_dir", type=Path, help="e.g. assets/songs/defrini-lets-go-to-a-rave")
    parser.add_argument("--auto", action="store_true", help=argparse.SUPPRESS)  # tests: scripted keys
    args = parser.parse_args(argv)
    data = json.loads((args.song_dir / "song.json").read_text())
    grid = loaded = BeatGrid(float(data["bpm"]), float(data["first_beat_s"]), int(data["routine_start_beat"]))
    samples, sr = load_audio(args.song_dir / data["audio"])
    duration = len(samples) / sr

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"parapara – beat check: {data.get('title', args.song_dir.name)}")
    player = Player(sr)
    if player.sr != sr:  # the mixer could not open the file's rate: resample (linear)
        samples = np.interp(np.arange(int(duration * player.sr)) / player.sr,
                            np.arange(len(samples)) / sr, samples).astype(np.float32)  # fmt: skip
        sr = player.sr
    env = low_band_envelope(samples, sr, ENVELOPE_HOP_S)
    mix = render_click_track(samples, sr, grid)
    player.play(mix, 0.0)
    clock = pygame.time.Clock()
    script = [pygame.K_PERIOD, pygame.K_END, pygame.K_COMMA, pygame.K_RIGHT, pygame.K_UP, pygame.K_s,
              pygame.K_ESCAPE] if args.auto else []  # fmt: skip
    frame, running = 0, True

    while running:
        keys: list[tuple[int, bool]] = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys.append((event.key, bool(event.mod & pygame.KMOD_SHIFT)))
        if script and frame % 5 == 4:
            keys.append((script.pop(0), False))
        before = grid
        for key, shift in keys:
            if key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif key == pygame.K_LEFT:
                grid = grid.nudge_offset_ms(-1 if shift else -5)
            elif key == pygame.K_RIGHT:
                grid = grid.nudge_offset_ms(1 if shift else 5)
            elif key == pygame.K_UP:
                grid = grid.nudge_bpm(0.1 if shift else 0.01)
            elif key == pygame.K_DOWN:
                grid = grid.nudge_bpm(-0.1 if shift else -0.01)
            elif key == pygame.K_LEFTBRACKET:
                grid = grid.shift_downbeat(-1)
            elif key == pygame.K_RIGHTBRACKET:
                grid = grid.shift_downbeat(1)
            elif key == pygame.K_SPACE:
                player.pause() if not player.paused else player.play(mix, player.position())
            elif key == pygame.K_HOME:
                player.play(mix, 0.0)
            elif key in (pygame.K_COMMA, pygame.K_PERIOD, pygame.K_END):
                # Jump around to check the beat map late in the song too.
                if key == pygame.K_END:
                    target = duration - 15.0
                else:
                    target = player.position() + (32 if key == pygame.K_PERIOD else -32) * grid.beat_s
                target = min(max(0.0, target), duration - 0.5)
                was_paused = player.paused
                player.play(mix, target)
                if was_paused:
                    player.pause()
            elif key == pygame.K_s:
                path = save(args.song_dir, grid, grid != loaded)
                data = json.loads(path.read_text())
                loaded = grid
                print(f"saved {path}")
        if grid != before:
            mix = render_click_track(samples, sr, grid)
            if not player.paused:
                player.play(mix, player.position())
        pos = player.position()
        if pos >= duration and not player.paused:
            player.play(mix, 0.0)  # loop
        draw(screen, grid, env, pos, duration, data.get("beat_map", {}), grid != loaded, player.paused)
        pygame.display.flip()
        clock.tick(60)
        frame += 1
    pygame.quit()
    return grid


def main(argv: list[str] | None = None) -> None:
    """Console entry point (returns ``None``; see ``pose_analysis.main``)."""
    run(argv)


if __name__ == "__main__":
    main()
