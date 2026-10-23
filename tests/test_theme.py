"""The game theme is derived from the five-colour brand palette.

These tests are pure (no pygame) so they can run in the pytest process next
to the cv2 tests. They guard the two promises of ``game/theme.py``: every
brand colour matches ``assets/branding/PALETTE.md``, and the derived text /
status shades stay readable on the plum background.
"""

import re
from pathlib import Path

from parapara.game import theme

PALETTE = Path(__file__).resolve().parents[1] / "assets" / "branding" / "PALETTE.md"
BRAND = {
    (216, 11, 128): "#D80B80",
    (253, 223, 44): "#FDDF2C",
    (43, 15, 84): "#2B0F54",
    (142, 22, 170): "#8E16AA",
    (250, 250, 252): "#FAFAFC",
}


def _contrast(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """WCAG 2.1 contrast ratio (1 = identical, 21 = black on white)."""
    def luminance(colour: tuple[int, int, int]) -> float:
        channels = []
        for value in colour:
            srgb = value / 255
            channels.append(srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4)
        red, green, blue = channels
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_brand_colours_match_palette_file() -> None:
    documented = {int(hexcode[1:], 16) for hexcode in re.findall(r"#[0-9A-Fa-f]{6}", PALETTE.read_text())}
    expected = {int(hexcode[1:], 16) for hexcode in BRAND.values()}
    assert documented == expected

    for colour, hexcode in BRAND.items():
        assert colour in {theme.MAGENTA, theme.GOLD, theme.PLUM, theme.VIOLET, theme.OFFWHITE}, hexcode


def test_mix_blends_and_clamps() -> None:
    assert theme.mix(theme.PLUM, theme.VIOLET, 0) == theme.PLUM
    assert theme.mix(theme.PLUM, theme.VIOLET, 1) == theme.VIOLET
    assert theme.mix(theme.PLUM, theme.VIOLET, -1) == theme.PLUM
    assert theme.mix(theme.PLUM, theme.VIOLET, 2) == theme.VIOLET
    assert theme.mix((0, 100, 200), (200, 0, 40), 0.5) == (100, 50, 120)


def test_text_and_status_shades_are_readable_on_the_plum_ground() -> None:
    # Body text: WCAG AAA for normal text; hints and large status text: AA.
    assert _contrast(theme.TEXT, theme.BG) >= 7.0
    for name in ("DIM", "WARN", "ERROR", "PERFECT", "GREAT", "GOOD", "HOLD", "MISS", "ACCENT"):
        assert _contrast(getattr(theme, name), theme.BG) >= 3.0, name
