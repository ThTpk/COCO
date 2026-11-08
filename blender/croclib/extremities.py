"""Hands and feet.

These are built as explicit geometry rather than swept off the limb skeleton.
Fingers and toes are thin next to everything around them, and every operation
that makes a torso look good -- smoothing, thickening to a round section,
blurring away facets -- eats them. Giving them their own generator keeps them
out of reach of all that.

Both are built for the character's left side only; the build mirrors across X.
"""

from __future__ import annotations

import math

from mathutils import Vector

from . import meshutil as mu
from .proportions import L

# The hand hangs off the end of the arm; keep the two in step.
HAND_X = L.shoulder_x * 1.24
HAND_Y = -0.030


def _digit(base: Vector, direction: Vector, length: float, r0: float, r1: float,
           curl: Vector, segments: int = 5, sides: int = 8):
    """A tapered finger or toe that curls as it goes, ending in a claw.

    Returns (flesh, claw) so the two can take different materials.
    """
    direction = Vector(direction).normalized()
    curl = Vector(curl)
    pts, radii = [], []
    for i in range(segments + 1):
        t = i / segments
        p = base + direction * (length * t) + curl * (t * t * length)
        pts.append(p)
        radii.append((mu.lerp(r0, r1, t), mu.lerp(r0, r1, t) * 0.92))
    flesh = mu.tube(pts, radii, sides, 0.82, cap_start=True, cap_end=True)

    tip_dir = (pts[-1] - pts[-2]).normalized()
    claw_len = length * 0.42
    claw_base = pts[-1] - tip_dir * (r1 * 0.5)
    # Claws hook downward and a touch inward, as on the reference hand.
    hook = Vector((0.0, 0.0, -1.0)) * claw_len * 0.38
    claw_tip = pts[-1] + tip_dir * claw_len + hook
    claw = mu.cone(claw_base, claw_tip, r1 * 0.92, sides)
    return flesh, claw


# --------------------------------------------------------------------------
# hand
# --------------------------------------------------------------------------

def hand():
    """Left hand, hanging knuckles-forward with the fingers loosely spread.

    Returns (flesh, claws).
    """
    wrist_z = L.wrist
    cx, cy = HAND_X, HAND_Y

    # Palm: a slab that widens from the wrist out to the knuckle line.
    palm_stations = [
        (wrist_z + 0.012, 0.072, 0.042, cy),
        (wrist_z - 0.022, 0.086, 0.046, cy - 0.004),
        (wrist_z - 0.062, 0.099, 0.046, cy - 0.012),
        (wrist_z - 0.100, 0.100, 0.040, cy - 0.024),
    ]
    rings = [
        mu.ring((cx, yy, zz), rx, ry, 16, 0.60,
                normal=(0.0, 0.0, -1.0), up=(0.0, -1.0, 0.0))
        for zz, rx, ry, yy in palm_stations
    ]
    flesh = [mu.loft(rings, cap_start=True, cap_end=True)]
    claws = []

    knuckle_z = palm_stations[-1][0]
    knuckle_y = palm_stations[-1][3]
    # (x offset from palm centre, length, base radius, outward splay)
    fingers = [
        (-0.066, 0.118, 0.0265, -0.014),   # index, nearest the body
        (-0.023, 0.128, 0.0275, -0.004),
        (0.023, 0.120, 0.0265, 0.006),
        (0.064, 0.100, 0.0235, 0.020),     # little finger
    ]
    for dx, length, r0, splay in fingers:
        base = Vector((cx + dx, knuckle_y, knuckle_z + 0.006))
        direction = Vector((splay, -0.26, -1.0))
        curl = Vector((splay * 0.4, -0.30, 0.06))
        f, c = _digit(base, direction, length, r0, r0 * 0.62, curl)
        flesh.append(f)
        claws.append(c)

    # Thumb: off the inner edge of the palm, swung forward and down.
    thumb_base = Vector((cx - 0.092, knuckle_y + 0.030, wrist_z - 0.052))
    f, c = _digit(thumb_base, Vector((-0.42, -0.62, -0.68)), 0.100, 0.030, 0.018,
                  Vector((0.06, -0.14, -0.06)))
    flesh.append(f)
    claws.append(c)

    return mu.merge(*flesh), mu.merge(*claws)


# --------------------------------------------------------------------------
# foot
# --------------------------------------------------------------------------

# (y, top z, half width) along the left foot, heel behind (+y) to ball ahead (-y)
FOOT = [
    (0.126, 0.112, 0.080),
    (0.064, 0.186, 0.100),
    (0.000, 0.212, 0.112),
    (-0.076, 0.150, 0.121),
    (-0.150, 0.104, 0.127),
    (-0.205, 0.076, 0.122),
]


def foot():
    """Left foot: a broad plantigrade slab with four clawed toes.

    Returns (flesh, claws).
    """
    cx = L.stance_x
    rings = []
    for y, top, half_w in FOOT:
        rings.append(
            mu.ring((cx, y, top * 0.5), half_w, top * 0.5, 18, 0.55,
                    normal=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0))
        )
    flesh = [mu.loft(rings, cap_start=True, cap_end=True)]

    # Ankle column, blending the leg into the top of the foot.
    # Matched to the leg's own ankle radius at the top. A fatter mid ring read
    # as the cuff of a boot sitting on the foot rather than as an ankle.
    ankle = [
        (L.ankle + 0.055, L.ankle_half * 0.98, L.ankle_depth * 0.98, 0.014),
        (L.ankle - 0.030, L.ankle_half * 1.08, L.ankle_depth * 1.06, 0.006),
        (0.132, 0.100, 0.086, -0.002),
    ]
    arings = [
        mu.ring((cx, yy, zz), rx, ry, 18, 0.70,
                normal=(0.0, 0.0, -1.0), up=(0.0, -1.0, 0.0))
        for zz, rx, ry, yy in ankle
    ]
    flesh.append(mu.loft(arings, cap_start=True, cap_end=True))

    claws = []
    # (x offset, length, base radius, splay)
    toes = [
        (-0.086, 0.142, 0.034, -0.028),   # big toe, on the inside
        (-0.029, 0.152, 0.032, -0.008),
        (0.029, 0.140, 0.030, 0.014),
        (0.082, 0.114, 0.026, 0.038),
    ]
    for dx, length, r0, splay in toes:
        base = Vector((cx + dx, -0.196, 0.058))
        direction = Vector((splay, -1.0, -0.10))
        curl = Vector((splay * 0.3, 0.0, -0.045))
        f, c = _digit(base, direction, length, r0, r0 * 0.70, curl, segments=4)
        flesh.append(f)
        claws.append(c)

    return mu.merge(*flesh), mu.merge(*claws)
