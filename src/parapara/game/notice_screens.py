"""Screens for long texts (privacy notice, credits) and the first-start gate.

- :func:`draw_document`: :mod:`parapara.notices` entries in large text,
  scrollable (Up/Down) when they do not fit, with key hints at the bottom.
  Used by the game's Credits and Privacy screens.
- :func:`privacy_gate`: before an app starts the camera (MediaPipe) for the
  first time, it shows the privacy & licence notice full-screen. ``Enter``
  accepts (remembered in ``ui.json`` as ``privacy_notice``, see
  :data:`parapara.notices.PRIVACY_VERSION`), ``Esc`` quits without starting
  the camera. The notice is shown again when its version changes.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from parapara import settings  # noqa: E402
from parapara.game.branding import NAME, set_window_icon, wordmark  # noqa: E402
from parapara.game.ui import UI, display_flags, footer, layout_issues, register_box, text, text_height  # noqa: E402
from parapara.notices import PRIVACY_VERSION, Entry, privacy_entries  # noqa: E402

BG, TEXT, DIM, ACCENT = (18, 18, 28), (235, 235, 245), (120, 120, 140), (255, 220, 80)
STYLES = {"title": (34, ACCENT), "heading": (24, ACCENT), "text": (20, TEXT), "small": (16, DIM)}
WIDTH, HEIGHT = 1240, 720


def draw_document(surface: pygame.Surface, entries: list[Entry], scroll: int,
                  keys: str, x: int = 24, top: int = 16) -> bool:  # fmt: skip
    """Draw ``entries[scroll:]`` from ``top`` down to the key hints; returns
    whether entries were left out at the bottom (then "Down: more" is shown)."""
    hint_h = text_height(surface, keys, 16, x) + text_height(surface, "Down: more", 16, x) + 10
    bottom = surface.get_height() - hint_h
    y, more = top, False
    shown = entries[scroll:]
    for i, (style, s) in enumerate(shown):
        size, colour = STYLES[style]
        gap = 10 if style == "heading" and i else 0
        need = gap + text_height(surface, s, size, x)
        if style == "heading" and i + 1 < len(shown):  # keep a heading with its first line
            nxt_style, nxt = shown[i + 1]
            need += text_height(surface, nxt, STYLES[nxt_style][0], x)
        if y + need > bottom:
            more = True
            break
        y = text(surface, s, (x, y + gap), size, colour)
    lines = [(keys, 16, DIM)]
    if more or scroll:
        lines.insert(0, ("Down: more" if more else "Up: back to the top", 16, ACCENT))
    footer(surface, lines, x=x)
    return more


def privacy_gate(window: bool = False, keys_spec: str | None = None,
                 check_layout: bool = False) -> tuple[bool, int]:  # fmt: skip
    """Show the privacy notice unless this version was accepted already.

    Returns ``(accepted, frames)``: ``accepted`` is False if the player quit;
    ``frames`` is how many display frames the gate took (tests script keys on
    one frame counter across the gate and the game: ``--keys "frame:key"``).
    """
    if settings.load_ui().get("privacy_notice") == PRIVACY_VERSION:
        return True, 0
    set_window_icon()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), display_flags(window))
    pygame.display.set_caption(NAME)
    scripted: dict[int, list[int]] = {}
    for item in (keys_spec or "").split():
        frame, name = item.split(":")
        scripted.setdefault(int(frame), []).append(pygame.key.key_code(name))
    entries, scroll, frames = privacy_entries(), 0, 0
    clock, reported = pygame.time.Clock(), set()
    while True:
        keys = list(scripted.get(frames, []))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, frames
            if event.type == pygame.KEYDOWN:
                keys.append(event.key)
        frames += 1
        if pygame.K_ESCAPE in keys or pygame.K_q in keys:
            return False, frames
        if pygame.K_RETURN in keys or pygame.K_KP_ENTER in keys or pygame.K_SPACE in keys:
            settings.save_ui(privacy_notice=PRIVACY_VERSION)
            return True, frames
        UI.drawn = [] if check_layout else None
        screen.fill(BG)
        column = screen.subsurface((170, 0, 900, HEIGHT))  # a readable line length
        logo = wordmark(round(30 * UI.scale))
        top = 16
        if logo is not None:
            rect = column.blit(logo, (24, 12))
            register_box(column, rect, "[logo]")
            top = rect.bottom + 8
        more = draw_document(column, entries, scroll, "Enter: OK, start the camera    Esc: quit", top=top)
        if pygame.K_DOWN in keys and more:
            scroll += 1
        if pygame.K_UP in keys:
            scroll = max(0, scroll - 1)
        if UI.drawn is not None:
            for issue in layout_issues(UI.drawn):
                if issue not in reported:
                    reported.add(issue)
                    print(f"LAYOUT (privacy gate, text size {UI.scale}): {issue}", file=sys.stderr)
        pygame.display.flip()
        clock.tick(60)
