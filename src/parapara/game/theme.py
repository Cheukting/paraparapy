"""The game's colour theme, derived from the brand palette.

The five source colours are documented in ``assets/branding/PALETTE.md``:

===========  =========  =====================================================
Name         Hex        Role in the brand
===========  =========  =====================================================
``MAGENTA``  ``#D80B80``  Vibrant magenta: the dominant brand hue.
``GOLD``     ``#FDDF2C``  Bright gold: the high-energy highlight / action pop.
``PLUM``     ``#2B0F54``  Midnight plum: the deep, readable ground colour.
``VIOLET``   ``#8E16AA``  Electric violet: the mid-tone bridge.
``OFFWHITE`` ``#FAFAFC``  Cool off-white: the clean light surface / text.
===========  =========  =====================================================

The palette has only five colours, so the theme derives the extra shades it
needs by mixing two palette colours with :func:`mix`. This keeps every value
"on brand" while still giving panels, dimmed text and status messages enough
separation to be readable (accessibility note: status text is always a light
shade on the plum ground; magenta/gold are not used as small text on gold or
off-white).

Semantic names used by the screens
----------------------------------

* ``BG`` / ``PANEL`` / ``CARD``: the plum ground, and two progressively
  lighter plum/violet surfaces for panels and cards.
* ``TEXT`` / ``DIM``: off-white body text and a plum-tinted muted variant
  for hints.
* ``PRIMARY`` / ``ACCENT``: magenta for brand identity (logo, focus) and gold
  for key actions/highlights.
* ``OK`` / ``WARN`` / ``ERROR`` / ``RECORD``: status colours. Gold reads as
  "good", light violet as "attention" and magenta as "problem/recording";
  this is the closest mapping to the five-colour palette, and the shapes and
  wording of the messages still carry the meaning (colour is never the only
  cue).
* ``LEFT`` / ``RIGHT`` / ``BODY`` / ``FLASH``: the dancer and guide figure.
  The dancer's left is gold and right is magenta, matching the two brand
  highlights; the body is off-white and flashes are gold.
* ``PERFECT`` / ``GREAT`` / ``GOOD`` / ``HOLD`` / ``MISS``: grade colours,
  from gold through the two purples to magenta.

Design rule for new UI: use these names, not raw tuples, and prefer a
palette colour or a :func:`mix` of two. Keep ``PALETTE.md`` as the single
source of truth; if the palette changes, update the constants below together
with that file.
"""

from __future__ import annotations

Colour = tuple[int, int, int]

# --- the five brand colours (assets/branding/PALETTE.md) ---------------------
MAGENTA: Colour = (216, 11, 128)  # #D80B80
GOLD: Colour = (253, 223, 44)  # #FDDF2C
PLUM: Colour = (43, 15, 84)  # #2B0F54
VIOLET: Colour = (142, 22, 170)  # #8E16AA
OFFWHITE: Colour = (250, 250, 252)  # #FAFAFC


def mix(a: Colour, b: Colour, t: float) -> Colour:
    """Blend ``a`` towards ``b`` by ``t`` (0 = ``a``, 1 = ``b``).

    The derived shades in this module are all built with this helper so they
    stay recognisably within the palette.
    """
    t = min(1.0, max(0.0, t))
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


# --- surfaces and text -------------------------------------------------------
BG = PLUM
PANEL = mix(PLUM, VIOLET, 0.16)  # score panel, menu box, next-move box
CARD = mix(PLUM, VIOLET, 0.28)  # small cards inside a panel
TEXT = OFFWHITE
DIM = mix(OFFWHITE, PLUM, 0.58)  # still readable on plum, clearly secondary

# --- brand accents -----------------------------------------------------------
PRIMARY = MAGENTA
ACCENT = GOLD

# --- status colours ----------------------------------------------------------
OK = GOLD
WARN = mix(VIOLET, OFFWHITE, 0.35)  # light violet: readable, distinct from gold
ERROR = MAGENTA
RECORD = MAGENTA

# --- dancer / guide figure ---------------------------------------------------
LEFT_COLOUR = GOLD
RIGHT_COLOUR = MAGENTA
BODY_COLOUR = OFFWHITE
FLASH_COLOUR = GOLD
CENTRE_COLOUR = OFFWHITE
TEXT_COLOUR = OFFWHITE

# --- grades ------------------------------------------------------------------
PERFECT = GOLD
GREAT = mix(VIOLET, OFFWHITE, 0.40)  # readable grade colours on the plum ground
GOOD = mix(VIOLET, OFFWHITE, 0.20)
HOLD = mix(GOLD, OFFWHITE, 0.35)
MISS = MAGENTA

# --- beat-check plot (parapara-beatcheck) ------------------------------------
ENERGY = VIOLET
LINE = DIM
BAR_LINE = MAGENTA
NOW = OFFWHITE
