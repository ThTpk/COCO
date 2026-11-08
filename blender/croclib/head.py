"""The crocodilian skull.

The head is the whole likeness, so it is not swept along a spine like the
limbs. It is lofted from an explicit side-view profile instead -- each station
gives the top line, the bottom line and the half width at one depth -- because
that is exactly how the reference sheet reads: a long boxy snout under a brow
that steps up sharply, and a jaw that tucks just inside it.
"""

from __future__ import annotations

import math

from mathutils import Vector

from . import meshutil as mu
from .proportions import L

RING = 20

# (y, top z, bottom z, half width, superellipse power)
# y runs from behind the skull (+) to the snout tip (-), since the character
# faces -Y.
UPPER = [
    (0.118, 2.096, 1.998, 0.098, 0.78),   # occiput
    (0.080, 2.140, 1.980, 0.142, 0.62),
    (0.020, 2.152, 1.967, 0.158, 0.56),   # brow ridge, widest point
    (-0.045, 2.139, 1.963, 0.146, 0.54),  # eye line
    (-0.075, 2.126, 1.961, 0.130, 0.52),  # the step down onto the snout
    (-0.110, 2.104, 1.960, 0.114, 0.50),
    (-0.180, 2.078, 1.962, 0.100, 0.48),
    (-0.255, 2.057, 1.966, 0.090, 0.48),
    (-0.325, 2.041, 1.972, 0.078, 0.52),
    (-0.363, 2.029, 1.981, 0.057, 0.72),  # rounded tip
]

JAW = [
    (0.086, 1.977, 1.929, 0.126, 0.80),   # hinge
    (0.020, 1.973, 1.907, 0.137, 0.64),
    (-0.060, 1.969, 1.905, 0.123, 0.58),
    (-0.140, 1.967, 1.913, 0.105, 0.55),
    (-0.220, 1.969, 1.923, 0.091, 0.55),
    (-0.300, 1.973, 1.935, 0.077, 0.58),
    (-0.352, 1.979, 1.950, 0.056, 0.72),
]


def _loft_profile(profile, cap_start=True, cap_end=True):
    rings = []
    for y, top, bottom, half_w, power in profile:
        centre = (0.0, y, (top + bottom) * 0.5)
        rings.append(
            mu.ring(centre, half_w, (top - bottom) * 0.5, RING, power,
                    normal=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0))
        )
    return mu.loft(rings, cap_start, cap_end)


def cranium():
    return _loft_profile(UPPER)


def jaw():
    return _loft_profile(JAW)


def _interp(profile, y):
    """Top, bottom and half width of a profile at an arbitrary depth."""
    for (y0, t0, b0, w0, _), (y1, t1, b1, w1, _) in zip(profile, profile[1:]):
        if y1 <= y <= y0:
            f = 0.0 if abs(y0 - y1) < 1e-9 else (y0 - y) / (y0 - y1)
            return (mu.lerp(t0, t1, f), mu.lerp(b0, b1, f), mu.lerp(w0, w1, f))
    return (profile[-1][1], profile[-1][2], profile[-1][3])


def teeth():
    """Two rows top and bottom, biggest at the canines and tapering forward.

    They are placed off the jaw profile rather than at fixed coordinates, so
    they keep following the gum line whenever the skull is retuned.
    """
    parts = []
    count = 13
    for i in range(count):
        f = i / (count - 1)
        y = mu.lerp(-0.010, -0.340, f)
        # Two canines sit proud of the rest, as on the sheet.
        scale = 1.0 - 0.45 * f
        if 0.10 < f < 0.26:
            scale *= 1.55
        if 0.46 < f < 0.60:
            scale *= 1.25

        top, bottom, half_w = _interp(UPPER, y)
        length = 0.040 * scale
        for sx in (-1.0, 1.0):
            x = sx * half_w * 0.88
            base = Vector((x, y, bottom + 0.004))
            tip = Vector((x * 0.97, y - 0.004, bottom - length))
            parts.append(mu.cone(base, tip, 0.0135 * scale, 6))

        jt, jb, jw = _interp(JAW, y)
        jlen = 0.032 * scale
        for sx in (-1.0, 1.0):
            x = sx * jw * 0.84
            base = Vector((x, y, jt - 0.004))
            tip = Vector((x * 0.97, y - 0.003, jt + jlen))
            parts.append(mu.cone(base, tip, 0.0115 * scale, 6))
    return mu.merge(*parts)


def eyes():
    """Eyeball plus the hooded ridge over it that gives the glare."""
    balls, hoods = [], []
    for sx in (-1.0, 1.0):
        centre = Vector((sx * 0.112, -0.052, 2.108))
        balls.append(_sphere(centre, 0.030, 12, 8))

        # A wedge of brow pulled forward and down over the outer corner.
        back = centre + Vector((sx * 0.018, 0.052, 0.018))
        front = centre + Vector((sx * -0.004, -0.050, -0.004))
        hoods.append(mu.cone(back, front, 0.040, 10, radius2=0.012))
    return mu.merge(*balls), mu.merge(*hoods)


def crest():
    """Rows of spikes over the skull and down the nape.

    Three rows: a tall centre line and a shorter one either side, which is what
    reads as a fan from the front and as a sawtooth from the side.
    """
    spine = mu.catmull_rom([
        Vector((0.0, -0.030, 2.150)),
        Vector((0.0, 0.030, 2.158)),
        Vector((0.0, 0.090, 2.140)),
        Vector((0.0, 0.126, 2.086)),
        Vector((0.0, 0.132, 2.020)),
        Vector((0.0, 0.118, 1.955)),
        Vector((0.0, 0.098, 1.900)),
    ], 4)

    parts = []
    n = len(spine)
    for row, (x_off, scale) in enumerate(((0.0, 1.0), (0.044, 0.74), (-0.044, 0.74))):
        for i in range(1, n - 1, 2):
            t = i / (n - 1)
            p = spine[i] + Vector((x_off, 0.0, 0.0))
            tangent = (spine[i + 1] - spine[i - 1]).normalized()
            out = Vector((0.0, 0.0, 1.0)) if t < 0.35 else Vector((0.0, 1.0, 0.25)).normalized()
            out = (out - tangent * out.dot(tangent)).normalized()
            grow = (1.0 - 0.55 * t) * scale
            height = 0.052 * grow
            base_r = 0.019 * grow
            # Side rows sit slightly lower, following the skull's curve.
            sink = 0.010 if row else 0.0
            base = p - out * sink
            tip = base + out * height - tangent * height * 0.45
            parts.append(mu.cone(base, tip, base_r, 6))
    return mu.merge(*parts)


def nostrils():
    bumps = []
    for sx in (-1.0, 1.0):
        bumps.append(_sphere(Vector((sx * 0.024, -0.336, 2.026)), 0.016, 8, 6))
    return mu.merge(*bumps)


def _sphere(centre: Vector, radius: float, seg: int, rings_n: int):
    verts = []
    for r in range(rings_n + 1):
        phi = math.pi * r / rings_n
        z = math.cos(phi) * radius
        rad = math.sin(phi) * radius
        for s in range(seg):
            a = 2.0 * math.pi * s / seg
            verts.append(centre + Vector((math.cos(a) * rad, math.sin(a) * rad, z)))
    faces = []
    for r in range(rings_n):
        for s in range(seg):
            s2 = (s + 1) % seg
            a = r * seg + s
            b = r * seg + s2
            c = (r + 1) * seg + s2
            d = (r + 1) * seg + s
            faces.append((a, b, c, d))
    return verts, faces
