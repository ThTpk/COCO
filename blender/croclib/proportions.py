"""Where every part of the character sits, in metres.

The vertical landmarks and the widths come from `blender/measure_refs.py`,
which reads them off the reference silhouettes as fractions of body height; the
depths come from the side sheet. They are frozen here as a plain table rather
than read live, so that a tweak to one number is a readable diff and the build
does not depend on the image set being present.

Axes: +X is the character's left, -Y is the direction he faces (Blender's front
view looks from -Y), +Z is up. Z = 0 is the ground, under the soles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HEIGHT = 2.20  # crest to sole


def z(t: float) -> float:
    """Height in metres from a measured fraction-of-body-height from the top."""
    return round(HEIGHT * (1.0 - t), 4)


def m(f: float) -> float:
    """A measured fraction of body height, in metres."""
    return round(HEIGHT * f, 4)


@dataclass(frozen=True)
class Landmarks:
    # --- vertical stations, top of crest downward -------------------------
    crest_top: float = z(0.000)
    skull_top: float = z(0.022)
    brow: float = z(0.045)
    eye: float = z(0.058)
    jaw_hinge: float = z(0.088)
    mouth: float = z(0.108)
    chin: float = z(0.125)
    neck_base: float = z(0.148)
    shoulder: float = z(0.180)
    armpit: float = z(0.250)
    chest: float = z(0.265)
    elbow: float = z(0.345)
    belt: float = z(0.415)
    wrist: float = z(0.448)
    hip: float = z(0.472)
    crotch: float = z(0.520)
    finger_tip: float = z(0.535)
    knee: float = z(0.680)
    calf: float = z(0.760)
    ankle: float = z(0.895)
    instep: float = z(0.940)
    sole: float = z(1.000)

    # --- half-widths across X --------------------------------------------
    skull_half: float = m(0.072)
    snout_half: float = m(0.046)
    jaw_half: float = m(0.062)
    neck_half: float = m(0.060)
    shoulder_half: float = m(0.188)
    chest_half: float = m(0.142)
    waist_half: float = m(0.125)
    hip_half: float = m(0.138)
    thigh_half: float = m(0.061)
    knee_half: float = m(0.050)
    ankle_half: float = m(0.040)

    # --- depths along Y ---------------------------------------------------
    skull_depth: float = m(0.082)
    neck_depth: float = m(0.068)
    chest_depth: float = m(0.106)
    waist_depth: float = m(0.098)
    hip_depth: float = m(0.100)
    thigh_depth: float = m(0.070)
    knee_depth: float = m(0.056)
    ankle_depth: float = m(0.046)

    # --- head -------------------------------------------------------------
    head_back_y: float = m(0.052)    # behind the head's origin
    snout_tip_y: float = -m(0.165)   # ahead of it (-Y is forward)
    snout_drop: float = m(0.028)     # tip sits this far below the brow line

    # --- arm --------------------------------------------------------------
    shoulder_x: float = m(0.140)
    upper_arm_r: float = m(0.052)
    elbow_r: float = m(0.046)
    forearm_r: float = m(0.047)
    wrist_r: float = m(0.033)
    hand_len: float = m(0.098)
    hand_half: float = m(0.046)
    hand_thick: float = m(0.022)

    # --- leg --------------------------------------------------------------
    hip_x: float = m(0.080)
    foot_len: float = m(0.195)
    foot_half: float = m(0.058)
    heel_y: float = m(0.052)

    # --- tail -------------------------------------------------------------
    tail_root_r: float = m(0.072)
    tail_len: float = m(0.760)

    # --- stance -----------------------------------------------------------
    stance_x: float = m(0.105)       # ankle offset from centreline
    arm_drop_x: float = m(0.030)     # how far the hands hang out from the hip


L = Landmarks()


# Cross-section "boxiness" for the superellipse sweeps. Lower is squarer.
POWER_LIMB = 0.86
POWER_TORSO = 0.72
POWER_SNOUT = 0.55
POWER_TAIL = 0.78


@dataclass
class Palette:
    """Flat colours lifted from the reference sheet, linear-light."""
    scale_back: tuple = (0.620, 0.230, 0.048, 1.0)
    scale_belly: tuple = (0.760, 0.430, 0.170, 1.0)
    shirt: tuple = (0.055, 0.090, 0.165, 1.0)
    denim: tuple = (0.165, 0.265, 0.395, 1.0)
    leather: tuple = (0.105, 0.058, 0.032, 1.0)
    metal: tuple = (0.420, 0.430, 0.440, 1.0)
    gem: tuple = (0.022, 0.330, 0.115, 1.0)
    bone: tuple = (0.780, 0.760, 0.700, 1.0)
    claw: tuple = (0.280, 0.285, 0.290, 1.0)
    eye: tuple = (0.520, 0.045, 0.030, 1.0)
    mouth: tuple = (0.180, 0.055, 0.060, 1.0)


P = Palette()
