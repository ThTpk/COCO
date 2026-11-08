"""Torso, neck, arms, legs and tail.

Only a little of this ends up on screen -- the shirt covers the torso and the
jeans cover the legs -- so the budget goes where skin actually shows: the
forearms, the tail and its scutes. The covered parts still have to be the right
volume, because the clothing is built by offsetting outward from them.
"""

from __future__ import annotations

import math

from mathutils import Vector

from . import meshutil as mu
from .proportions import L, POWER_LIMB, POWER_TAIL, POWER_TORSO

RING = 24


def torso():
    """Hip to neck as a stack of measured cross-sections.

    The keel at the chest and the taper at the waist are both in the table; the
    only thing added here is the shoulder shelf, which no single front-view
    width can give you because the deltoids read as part of the arm.
    """
    stations = [
        # (z,                 half-width,            half-depth,       lean)
        (L.crotch,            L.hip_half * 0.88,     L.hip_depth * 0.92, 0.00),
        (L.hip,               L.hip_half,            L.hip_depth,       0.00),
        (L.belt,              L.waist_half,          L.waist_depth,     0.02),
        ((L.belt + L.chest) * 0.5, L.chest_half * 0.94, L.chest_depth * 1.02, 0.05),
        (L.chest,             L.chest_half,          L.chest_depth,     0.06),
        (L.armpit,            L.chest_half * 1.06,   L.chest_depth * 1.00, 0.04),
        (L.shoulder,          L.shoulder_half * 0.76, L.chest_depth * 0.92, 0.00),
        (L.shoulder + 0.045,  L.shoulder_half * 0.58, L.chest_depth * 0.78, -0.03),
        (L.neck_base,         L.neck_half * 1.55,    L.neck_depth * 1.35, -0.06),
    ]
    rings = [
        mu.ring((0.0, 0.0, zz), rx, ry, RING, POWER_TORSO, shear_y=lean)
        for zz, rx, ry, lean in stations
    ]
    return mu.loft(rings, cap_start=True, cap_end=True)


def neck():
    """Short and thick, tipped back so the skull sits over the shoulders.

    It has to stop below the jaw hinge: carried up to the chin it surfaced
    through the lower jaw as a collar.
    """
    top = L.chin + (L.jaw_hinge - L.chin) * 0.10
    stations = [
        (L.neck_base - 0.05, L.neck_half * 1.60, L.neck_depth * 1.40),
        (L.neck_base + 0.04, L.neck_half * 1.18, L.neck_depth * 1.22),
        (top, L.neck_half * 1.00, L.neck_depth * 1.10),
    ]
    rings = [mu.ring((0.0, 0.0, zz), rx, ry, RING, 0.80) for zz, rx, ry in stations]
    return mu.loft(rings, cap_start=True, cap_end=True)


# --------------------------------------------------------------------------
# arm
# --------------------------------------------------------------------------

def arm_path() -> list[Vector]:
    """Shoulder to wrist for the character's left arm (+X), hanging slightly out.

    The chain starts *inside* the ribcage and below the shoulder line. Started
    at the joint itself, the swept tube's end cap -- a disc tilted to face
    along the arm -- stood proud of the torso as a pair of fins above the
    shoulders. Burying the first two stations hides the cap and lets the
    deltoid read as a crown over the joint instead of a spur off it.
    """
    sx = L.shoulder_x
    return [
        Vector((sx * 0.45, 0.012, L.shoulder - 0.030)),   # buried in the chest
        Vector((sx * 0.88, 0.006, L.shoulder - 0.085)),   # deltoid crown
        Vector((sx * 1.08, 0.000, (L.shoulder + L.elbow) * 0.5)),
        Vector((sx * 1.19, 0.014, L.elbow)),
        Vector((sx * 1.26, -0.008, (L.elbow + L.wrist) * 0.5)),
        Vector((sx * 1.29, -0.030, L.wrist)),
    ]


def arm_radii() -> list[tuple[float, float]]:
    ua, el, fa, wr = L.upper_arm_r, L.elbow_r, L.forearm_r, L.wrist_r
    return [
        (ua * 0.80, ua * 0.84),   # inside the chest, kept small so it hides
        (ua * 1.22, ua * 1.24),   # deltoid crown
        (ua * 0.98, ua * 1.00),
        (el, el * 1.06),
        (fa, fa * 0.92),          # forearm belly, flattened front-to-back
        (wr, wr * 0.86),
    ]


def arm():
    pts = mu.catmull_rom(arm_path(), 5)
    radii_ctrl = arm_radii()
    radii = _resample_radii(radii_ctrl, len(pts))
    return mu.tube(pts, radii, RING, POWER_LIMB, cap_start=True, cap_end=True)


def forearm_spikes():
    """The row of scutes along the outside of the forearm, elbow to wrist."""
    pts = mu.catmull_rom(arm_path(), 5)
    parts = []
    n = len(pts)
    for i in range(n):
        t = i / (n - 1)
        if t < 0.52 or t > 0.96:
            continue
        p = pts[i]
        # Fade in after the elbow and out again before the bracer.
        grow = mu.smoothstep((t - 0.52) / 0.18) * (1.0 - mu.smoothstep((t - 0.80) / 0.16))
        if grow < 0.05:
            continue
        length = mu.lerp(0.02, 0.105, grow)
        base_r = mu.lerp(0.012, 0.030, grow)
        tangent = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
        out = Vector((1.0, 0.25, 0.0))
        out = (out - tangent * out.dot(tangent)).normalized()
        tip = p + out * (L.forearm_r * 0.95 + length) + tangent * length * 0.55
        parts.append(mu.cone(p + out * L.forearm_r * 0.55, tip, base_r, 8))
    return mu.merge(*parts) if parts else ([], [])


# --------------------------------------------------------------------------
# leg
# --------------------------------------------------------------------------

def leg_path() -> list[Vector]:
    """Hip to ankle for the left leg, standing with the feet apart."""
    return [
        Vector((L.hip_x, 0.005, L.hip + 0.030)),
        Vector((L.hip_x * 1.04, 0.012, (L.hip + L.knee) * 0.5)),
        Vector((L.stance_x * 1.06, 0.004, L.knee)),
        Vector((L.stance_x * 1.02, 0.030, L.calf)),
        Vector((L.stance_x, 0.012, L.ankle)),
    ]


def leg_radii() -> list[tuple[float, float]]:
    return [
        (L.thigh_half * 1.08, L.thigh_depth * 1.12),
        (L.thigh_half * 0.94, L.thigh_depth * 0.98),
        (L.knee_half * 1.10, L.knee_depth * 1.18),
        (L.knee_half * 1.02, L.knee_depth * 1.22),   # calf belly
        (L.ankle_half, L.ankle_depth),
    ]


def leg():
    pts = mu.catmull_rom(leg_path(), 5)
    radii = _resample_radii(leg_radii(), len(pts))
    return mu.tube(pts, radii, RING, POWER_LIMB, cap_start=True, cap_end=True)


# --------------------------------------------------------------------------
# tail
# --------------------------------------------------------------------------

def tail_path() -> list[Vector]:
    """Out of the lower back, down and round to the character's right.

    Read off the side and back sheets: it leaves the hips almost horizontally,
    dips to its lowest point behind the right heel, then sweeps forward with
    the last third lifting slightly off the ground.
    """
    return [
        Vector((0.00, L.hip_depth * 0.55, L.hip + 0.02)),
        Vector((0.00, L.hip_depth * 1.35, L.hip - 0.09)),
        Vector((-0.09, 0.46, 0.84)),
        Vector((-0.26, 0.56, 0.56)),
        Vector((-0.44, 0.49, 0.33)),
        Vector((-0.58, 0.27, 0.19)),
        Vector((-0.63, 0.00, 0.13)),
        Vector((-0.60, -0.26, 0.11)),
    ]


def _tail_curve(samples_per_span: int = 7):
    return mu.catmull_rom(tail_path(), samples_per_span)


def tail():
    pts = _tail_curve()
    n = len(pts)
    radii = []
    for i in range(n):
        t = i / (n - 1)
        # Thick at the root, then a long even taper to a point.
        r = L.tail_root_r * (1.0 - t) ** 1.35
        r = max(r, 0.008)
        radii.append((r * 0.92, r * 1.08))   # taller than wide, like a real tail
    return mu.tube(pts, radii, 16, POWER_TAIL, cap_start=True, cap_end=False)


def tail_scutes():
    """The sawtooth ridge along the top of the tail."""
    pts = _tail_curve()
    n = len(pts)
    parts = []
    for i in range(2, n - 2):
        t = i / (n - 1)
        r = max(L.tail_root_r * (1.0 - t) ** 1.35, 0.008)
        grow = mu.smoothstep(t / 0.18) * (1.0 - mu.smoothstep((t - 0.82) / 0.18))
        if grow < 0.06:
            continue
        p = pts[i]
        tangent = (pts[i + 1] - pts[i - 1]).normalized()
        up = Vector((0.0, 0.0, 1.0))
        up = (up - tangent * up.dot(tangent))
        if up.length < 1e-5:
            up = Vector((0.0, 1.0, 0.0))
        up.normalize()
        height = r * mu.lerp(0.45, 1.05, grow)
        tip = p + up * (r * 1.02 + height) - tangent * height * 0.35
        parts.append(mu.cone(p + up * r * 0.45, tip, r * 0.46, 6))
    return mu.merge(*parts) if parts else ([], [])


# --------------------------------------------------------------------------

def _resample_radii(ctrl: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    """Spread control radii over a denser resampled path, smoothly."""
    out = []
    span = len(ctrl) - 1
    for i in range(count):
        f = i / (count - 1) * span
        k = min(int(f), span - 1)
        local = mu.smoothstep(f - k)
        rx = mu.lerp(ctrl[k][0], ctrl[k + 1][0], local)
        ry = mu.lerp(ctrl[k][1], ctrl[k + 1][1], local)
        out.append((rx, ry))
    return out
