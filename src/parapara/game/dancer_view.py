"""Draw the demo dancer (a clean stick figure) and the "next move" card.

Graphics are deliberately simple (decided 2026-10-18): what matters is the
smooth motion from :mod:`parapara.animation.animator`. The figure has thick
limbs with round joints, a head, a torso and legs whose knees bend with the
bounce and the two-step (two-link inverse kinematics from hip to foot).

Front view: body-frame ``z`` is dropped; arms further back are drawn first so
an arm reaching forward stays in front, and the wrist dot grows as the hand
comes towards the viewer (so forward knocks and the chop salute read). Colours follow the debug view:
**orange = dancer's right, cyan = dancer's left**.

Mirroring: with ``mirrored=True`` (the default, PLAN.md §7) the dancer's
right is drawn on the **right of the screen**, so the player copies it like a
mirror, the same convention as the recorder's guide figure.
"""

from __future__ import annotations

import math

import pygame

from parapara.animation.animator import CLAP, REST, ArmState, DancerPose, block_title  # noqa: F401 (re-export)
from parapara.game.theme import BODY_COLOUR, FLASH_COLOUR, LEFT_COLOUR, RIGHT_COLOUR
from parapara.routine.model import ArmPose, Keyframe

# Proportions in shoulder widths.
SHOULDER_HALF, HIP_HALF = 0.5, 0.28
TORSO = 1.3
UPPER_ARM, FOREARM, HAND = 0.72, 0.68, 0.22
THIGH, SHIN = 0.95, 0.95
NECK, HEAD_RADIUS = 0.18, 0.3


def _knee(
    hip: tuple[float, float], foot: tuple[float, float], outward: float
) -> tuple[float, float]:
    """Knee position for a two-link leg (thigh, shin), bending ``outward``."""
    dx, dy = foot[0] - hip[0], foot[1] - hip[1]
    d = min(math.hypot(dx, dy), THIGH + SHIN - 1e-6)
    mx, my = hip[0] + dx / 2, hip[1] + dy / 2
    h = math.sqrt(max(0.0, THIGH**2 - (d / 2) ** 2))
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length  # perpendicular to the leg
    if px * outward < 0:
        px, py = -px, -py
    return mx + px * h, my + py * h


def draw_dancer(surface: pygame.Surface, pose: DancerPose, centre: tuple[int, int], scale: float,
                mirrored: bool = True) -> None:  # fmt: skip
    """Draw ``pose`` with the mid-shoulder point near ``centre`` (pixels),
    ``scale`` pixels per shoulder width."""
    sign = 1 if mirrored else -1
    cx, cy = centre
    width = max(3, round(scale * 0.13))

    def px(x: float, y: float) -> tuple[int, int]:
        return round(cx + sign * (x + pose.body_x) * scale), round(
            cy - (y - pose.bounce) * scale
        )

    def line(a, b, colour, w=width) -> None:
        pygame.draw.line(surface, colour, a, b, w)
        pygame.draw.circle(surface, colour, a, w // 2)
        pygame.draw.circle(surface, colour, b, w // 2)

    # Legs: feet on the ground (not bouncing), hips dip with the bounce.
    ground = -(TORSO + THIGH + SHIN) + pose.bounce
    for foot, hip_x, colour in ((pose.right_foot, HIP_HALF, RIGHT_COLOUR),
                                (pose.left_foot, -HIP_HALF, LEFT_COLOUR)):  # fmt: skip
        hip = (hip_x, -TORSO)
        foot_pt = (foot[0] - pose.body_x, ground + foot[1])
        knee = _knee(hip, foot_pt, outward=hip_x)
        line(px(*hip), px(*knee), colour)
        line(px(*knee), px(*foot_pt), colour)
        fx, fy = px(*foot_pt)
        pygame.draw.line(
            surface,
            colour,
            (fx, fy),
            (fx + sign * round(0.12 * scale) * (1 if hip_x > 0 else -1), fy),
            width,
        )

    # Torso, shoulders, hips, head.
    line(px(0, 0), px(0, -TORSO), BODY_COLOUR)
    line(px(-SHOULDER_HALF, 0), px(SHOULDER_HALF, 0), BODY_COLOUR)
    line(px(-HIP_HALF, -TORSO), px(HIP_HALF, -TORSO), BODY_COLOUR)
    line(px(0, 0), px(0, NECK), BODY_COLOUR)
    hx, hy = px(0, NECK + HEAD_RADIUS)
    pygame.draw.circle(
        surface, BODY_COLOUR, (hx, hy), round(HEAD_RADIUS * scale), width
    )

    # Arms, back to front.
    arms = []
    for state, side_x, colour in (
        (pose.right, SHOULDER_HALF, RIGHT_COLOUR),
        (pose.left, -SHOULDER_HALF, LEFT_COLOUR),
    ):
        s = (side_x, 0.0)
        e = (s[0] + state.upper[0] * UPPER_ARM, s[1] + state.upper[1] * UPPER_ARM)
        w = (e[0] + state.fore[0] * FOREARM, e[1] + state.fore[1] * FOREARM)
        h = (w[0] + state.hand[0] * HAND, w[1] + state.hand[1] * HAND)
        depth = (
            state.upper[2] * UPPER_ARM + state.fore[2] * FOREARM
        )  # wrist z, shoulder widths
        arms.append((depth, s, e, w, h, colour))
    for depth, s, e, w, h, colour in sorted(arms, key=lambda a: a[0]):
        line(px(*s), px(*e), colour)
        line(px(*e), px(*w), colour)
        line(px(*w), px(*h), colour, max(2, width - 2))
        pygame.draw.circle(
            surface, colour, px(*w), round(width * (0.8 + 0.6 * max(-0.5, depth)))
        )

    if pose.clap_flash > 0:  # a burst where the hands meet
        wrists = [px(a[3][0], a[3][1]) for a in arms]
        mid = ((wrists[0][0] + wrists[1][0]) // 2, (wrists[0][1] + wrists[1][1]) // 2)
        radius = round(scale * (0.15 + 0.35 * pose.clap_flash))
        pygame.draw.circle(surface, FLASH_COLOUR, mid, radius, max(2, width // 2))


def static_pose(keyframe: Keyframe) -> DancerPose:
    """A still pose of one keyframe (for previews), standing at rest."""

    def arm(pose: ArmPose | None, side: str) -> ArmState:
        base = pose or (CLAP if keyframe.clap else REST)
        if pose is None and side == "left":
            base = base.mirrored()
        hand = base.hand if base.hand is not None else base.fore
        return ArmState(base.upper, base.fore, hand)

    return DancerPose(0.0, arm(keyframe.right, "right"), arm(keyframe.left, "left"),
                      1.0 if keyframe.clap else 0.0, 0.0, 0.0, (0.1, 0.0), (-0.1, 0.0),
                      0, 0, 0, None, None, None)  # fmt: skip
