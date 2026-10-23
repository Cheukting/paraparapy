"""Large, readable text for the game screens, shared by ``parapara-play``
and ``parapara-record`` (both are used from the dancing spot, 1.5-2 m away).

- :class:`UI`: the text size (``UI.scale``, changed with ``+`` / ``-`` and
  saved in ``ui.json`` via :mod:`parapara.settings`) and the layout check.
- :func:`text`: word-wrapped text that never runs off its surface;
  :func:`footer`: lines stacked up from the bottom (key hints).
- Layout check: while ``UI.drawn`` is a list, :func:`text` and
  :func:`register_box` record what they draw and :func:`layout_issues`
  reports overlapping or cut-off text (``--check-layout``; run by the tests
  on every screen at several text sizes).
"""

from __future__ import annotations

import dataclasses
import functools
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from parapara import settings  # noqa: E402

TEXT = (235, 235, 245)


class UI:
    """Text size and layout checking.

    - **Text size**, readable from the dancing spot (1.5-2 m away): every font
      size in this module is multiplied by ``UI.scale``; ``+`` / ``-`` change it
      during the game and the choice is saved (``settings.save_ui``). The first
      test run showed desktop-sized text was unreadable from there.
    - **Layout check** (third test run: some texts overlapped): while
      ``UI.drawn`` is a list, :func:`text` and :func:`register_box` record
      what they draw (absolute rectangles) and :func:`layout_issues` reports
      overlapping or cut-off text. ``parapara-play --check-layout`` prints
      the issues; the tests run every screen at several text sizes with it.
    """

    scale = 2.0
    MIN, MAX, STEP = 1.0, 2.6, 0.2
    drawn: list[Drawn] | None = None


@dataclasses.dataclass(frozen=True)
class Drawn:
    """One recorded element: a line of text or a box (absolute screen pixels)."""

    rect: pygame.Rect
    label: str
    bounds: pygame.Rect  # the surface it was drawn on
    box: bool = False


@functools.lru_cache(maxsize=64)
def _font(px: int) -> pygame.font.Font:
    return pygame.font.Font(None, px)


def font(size: float) -> pygame.font.Font:
    return _font(max(8, round(size * UI.scale)))


def _wrap(f: pygame.font.Font, s: str, max_w: int) -> list[str]:
    lines, line = [], ""
    for word in s.split(" "):
        trial = f"{line} {word}".strip()
        if line and f.size(trial)[0] > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line]


def text(surface, s, pos, size=24, colour=TEXT, centre=False, right=False, draw=True) -> int:
    """Draw ``s`` at ``size`` x ``UI.scale``; returns the y below it.

    Long text is word-wrapped to the room left on ``surface`` (for centred
    text: to both sides of ``x``); a single word that is still too wide is
    drawn smaller, so text never runs off the surface. ``draw=False`` only
    measures (returns the y below without drawing).
    """
    x, y = pos
    w = surface.get_width()
    max_w = max(16, (2 * min(x, w - x) - 8) if centre else (x - 8 if right else w - x - 8))
    px = max(8, round(size * UI.scale))
    while True:
        f = _font(px)
        lines = _wrap(f, s, max_w)
        if px <= 8 or all(f.size(ln)[0] <= max_w for ln in lines):
            break
        px -= 2
    ox, oy = surface.get_abs_offset()
    for ln in lines:
        lw, lh = f.size(ln)
        lx = x - lw // 2 if centre else (x - lw if right else x)
        if draw:
            surface.blit(f.render(ln, True, colour), (lx, y))
            if UI.drawn is not None and ln.strip():
                UI.drawn.append(Drawn(pygame.Rect(ox + lx, oy + y, lw, lh), ln.strip(),
                                      pygame.Rect((ox, oy), surface.get_size())))  # fmt: skip
        y += lh + 2
    return y + 4


def text_height(surface, s, size, width_from_x: int = 24) -> int:
    """Height ``text`` would take at ``(width_from_x, 0)`` on ``surface``."""
    return text(surface, s, (width_from_x, 0), size, draw=False)


def footer(surface, lines: list[tuple[str, float, tuple[int, int, int]]], x: int = 24, margin: int = 10) -> int:
    """Draw ``(text, size, colour)`` lines stacked up from the bottom of
    ``surface`` (key hints); returns the top y, i.e. the room left above."""
    y = surface.get_height() - margin
    for s, size, colour in reversed(lines):
        y -= text_height(surface, s, size, x)
        text(surface, s, (x, y), size, colour)
    return y


def register_box(surface, rect: pygame.Rect, label: str) -> None:
    """Record a framed box (text inside it is fine, text crossing its edge is not)."""
    if UI.drawn is not None:
        ox, oy = surface.get_abs_offset()
        UI.drawn.append(Drawn(rect.move(ox, oy), label, pygame.Rect((ox, oy), surface.get_size()), box=True))


def layout_issues(drawn: list[Drawn]) -> list[str]:
    """Overlapping text, text crossing a box edge, and text cut off by the
    edge of the surface it was drawn on."""
    issues = []
    for d in drawn:
        if not d.box and not d.bounds.contains(d.rect):
            issues.append(f"cut off: {d.label!r}")
    for i, a in enumerate(drawn):
        for b in drawn[i + 1:]:
            if a.box and b.box or not a.rect.colliderect(b.rect):
                continue
            if a.box or b.box:
                box, other = (a, b) if a.box else (b, a)
                if not box.rect.contains(other.rect):
                    issues.append(f"crosses {box.label}: {other.label!r}")
            else:
                issues.append(f"overlap: {a.label!r} / {b.label!r}")
    return issues


def adjust_scale(keys: list[int] | set[int]) -> None:
    """``+`` / ``-`` change the text size (clamped, saved for next time)."""
    bigger = {pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS} & set(keys)
    smaller = {pygame.K_MINUS, pygame.K_KP_MINUS} & set(keys)
    if bigger or smaller:
        UI.scale = round(min(UI.MAX, max(UI.MIN, UI.scale + (UI.STEP if bigger else -UI.STEP))), 2)
        settings.save_ui(text_scale=UI.scale)


def load_scale(override: float | None) -> None:
    """Text size from the command line, else the saved one, else the default."""
    UI.scale = float(override or settings.load_ui().get("text_scale", UI.scale))


def display_flags(window: bool) -> int:
    """Fullscreen, scaled from the fixed layout (letterboxed), unless a window
    is asked for or the dummy video driver is used (tests)."""
    headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    return 0 if headless or window else pygame.SCALED | pygame.FULLSCREEN
