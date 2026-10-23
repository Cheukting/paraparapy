"""ParaParaPy branding: the logo as the menu title and the window icon.

The source is the project owner's logo, ``assets/branding/paraparapy-logo.png``
(2000x2000, transparent background, wordmark "ParaParaPy" in pink and yellow
with a yellow flower in each "P"). Everything else is derived from it:

- :func:`wordmark`: the logo cropped to its artwork (the transparent margin
  removed) and scaled to a height: the title on the menu screen;
- :func:`icon`: the flower of the first "P" on its pink, cut to a circle
  (the letters' outlines touch, so no crop of one letter is clean): the window
  / dock icon of ``parapara-play`` and ``parapara-record``.

``uv run python -m parapara.game.branding`` writes both as PNG files next to
the logo (``paraparapy-wordmark.png`` for the README, ``paraparapy-icon.png``
for packaging). If the logo file is missing, the functions return ``None`` and
the game falls back to a text title.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

# Found from the source tree (works from any working directory with the
# editable install used today; packaging will bundle it: PLAN.md §10).
BRANDING_DIR = Path(__file__).resolve().parents[3] / "assets" / "branding"
LOGO = BRANDING_DIR / "paraparapy-logo.png"
NAME = "ParaParaPy"
# The flower inside the first "P", in pixels of the 2000x2000 logo file.
FLOWER_CENTRE, FLOWER_HALF = (214, 881), 70


@functools.cache
def _logo(path: Path = LOGO) -> pygame.Surface | None:
    if not path.exists():
        return None
    image = pygame.image.load(str(path))
    return image.convert_alpha() if pygame.display.get_surface() is not None else image


def wordmark(height: int, path: Path = LOGO) -> pygame.Surface | None:
    """The logo's artwork (transparent margin cropped), ``height`` pixels high."""
    logo = _logo(path)
    if logo is None:
        return None
    art = logo.subsurface(logo.get_bounding_rect(min_alpha=1))
    width = round(art.get_width() * height / art.get_height())
    return pygame.transform.smoothscale(art, (width, height))


def icon(size: int = 256, path: Path = LOGO) -> pygame.Surface | None:
    """The first "P"'s flower on its pink background, cut to a circle."""
    logo = _logo(path)
    if logo is None:
        return None
    (cx, cy), half = FLOWER_CENTRE, FLOWER_HALF
    crop = logo.subsurface(pygame.Rect(cx - half, cy - half, 2 * half, 2 * half))
    out = (
        pygame.transform.smoothscale(crop, (size, size)).convert_alpha()
        if pygame.display.get_surface()
        else pygame.transform.smoothscale(crop, (size, size))
    )
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    mask.fill((255, 255, 255, 0))
    pygame.draw.circle(mask, (255, 255, 255, 255), (size // 2, size // 2), size // 2)
    out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)  # keep only the disc
    return out


def set_window_icon() -> None:
    """Use :func:`icon` as the window / dock icon (call before ``set_mode``)."""
    image = icon(256)
    if image is not None:
        pygame.display.set_icon(image)


def main() -> None:
    """Write the derived wordmark and icon PNGs next to the logo."""
    pygame.init()
    pygame.display.set_mode((1, 1), pygame.HIDDEN)
    mark, ico = wordmark(240), icon(512)
    if mark is None or ico is None:
        raise SystemExit(f"logo not found: {LOGO}")
    pygame.image.save(mark, str(BRANDING_DIR / "paraparapy-wordmark.png"))
    pygame.image.save(ico, str(BRANDING_DIR / "paraparapy-icon.png"))
    print(
        f"wrote {BRANDING_DIR / 'paraparapy-wordmark.png'} and {BRANDING_DIR / 'paraparapy-icon.png'}"
    )
    pygame.quit()


if __name__ == "__main__":
    main()
