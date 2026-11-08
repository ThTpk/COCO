"""Measure the character's proportions from the reference turnaround.

Run under Blender's Python (it bundles numpy and can decode the jpegs):

    blender -b --python blender/measure_refs.py

The reference art puts a saturated orange/blue character on a desaturated grey
backdrop, so saturation alone separates figure from ground -- no chroma key, no
background plate needed. Everything downstream is a fraction of body height, so
the absolute pixel scale never matters.

Writes blender/measurements.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

try:
    import bpy
except ImportError:  # pragma: no cover - only for linting outside Blender
    bpy = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "input", "albedo")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements.json")

# Downsample before any labelling: proportions survive it and a 4.3 Mpx
# union-find in pure python does not.
SHRINK = 4


def load_rgb(path: str) -> np.ndarray:
    """Return the image as float RGB, row 0 = top of picture."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    # Blender hands back bottom-up RGBA; flip to picture order.
    return buf.reshape(h, w, 4)[::-1, :, :3]


def shrink(rgb: np.ndarray, factor: int = SHRINK) -> np.ndarray:
    h, w = rgb.shape[:2]
    h -= h % factor
    w -= w % factor
    small = rgb[:h, :w].reshape(h // factor, factor, w // factor, factor, 3)
    return small.mean(axis=(1, 3))


def saturation(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def largest_blob(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest 4-connected region.

    Iterative dilation under the mask: far cheaper to write than union-find and
    the figure is one contiguous blob, so it converges in a few dozen passes.
    """
    h, w = mask.shape
    # Seed from the densest row's centre -- that is the torso, never a speck.
    rows = mask.sum(axis=1)
    seed_y = int(np.argmax(rows))
    cols = np.flatnonzero(mask[seed_y])
    seed_x = int(cols[len(cols) // 2])

    grown = np.zeros_like(mask)
    grown[seed_y, seed_x] = True
    while True:
        nxt = grown.copy()
        nxt[1:, :] |= grown[:-1, :]
        nxt[:-1, :] |= grown[1:, :]
        nxt[:, 1:] |= grown[:, :-1]
        nxt[:, :-1] |= grown[:, 1:]
        nxt &= mask
        if nxt.sum() == grown.sum():
            return nxt
        grown = nxt


def chroma(rgb: np.ndarray) -> np.ndarray:
    """Colour with brightness divided out: (r, g) of r+g+b = 1."""
    total = np.maximum(rgb.sum(axis=2, keepdims=True), 1e-6)
    return (rgb / total)[..., :2]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Close interior holes by flooding the background in from the border."""
    outside = np.zeros_like(mask)
    free = ~mask
    outside[0, :] = free[0, :]
    outside[-1, :] = free[-1, :]
    outside[:, 0] = free[:, 0]
    outside[:, -1] = free[:, -1]
    while True:
        nxt = outside.copy()
        nxt[1:, :] |= outside[:-1, :]
        nxt[:-1, :] |= outside[1:, :]
        nxt[:, 1:] |= outside[:, :-1]
        nxt[:, :-1] |= outside[:, 1:]
        nxt &= free
        if nxt.sum() == outside.sum():
            return ~nxt
        outside = nxt


def figure_mask(path: str, thr: float = 0.035, sat_floor: float = 0.022) -> np.ndarray:
    """Figure as whatever differs in *colour* from the backdrop on its own row.

    Three traps in this reference set, each of which defeats the obvious rule:

    - A fixed saturation cut cannot serve both the sheets that stand him in a
      plain grey void and the ones that put him in a dungeon whose warm stone
      is more saturated than his denim.
    - Measuring the backdrop per sheet does not help either, because the stone
      is not one colour -- there are pillars in the margins and bare wall in
      the middle. So the backdrop is modelled *per row*, from the median of the
      two outer margins, which the figure never reaches.
    - Plain RGB distance from that backdrop then catches the painted vignette,
      a soft halo of pure brightness falloff hugging the figure. Dividing
      brightness out before comparing drops the halo and keeps the character,
      who differs from grey stone in hue, not just in level.

    The saturation floor is the backstop: a near-grey pixel is backdrop no
    matter what its chromaticity rounds to. It has to stay low, though -- the
    steel bracers and bone claws are grey too, and a floor strict enough to
    reject stone also cut the wrists, which dropped both hands off the figure
    entirely when only the largest blob was kept.
    """
    rgb = shrink(load_rgb(path))
    h, w = rgb.shape[:2]

    edge = max(2, int(w * 0.06))
    chr_ = chroma(rgb)
    margin = np.concatenate([chr_[:, :edge, :], chr_[:, -edge:, :]], axis=1)
    bg = np.median(margin, axis=1)[:, None, :]  # (h, 1, 2)

    dist = np.sqrt(((chr_ - bg) ** 2).sum(axis=2))
    mask = (dist > thr) & (saturation(rgb) > sat_floor)
    # Drop the caption bar some sheets carry along the bottom edge.
    mask[int(h * 0.97):, :] = False
    return fill_holes(largest_blob(mask))


def row_extent(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: leftmost column, rightmost column, filled pixel count."""
    h, w = mask.shape
    idx = np.arange(w)[None, :]
    any_row = mask.any(axis=1)
    left = np.where(any_row, np.where(mask, idx, w).min(axis=1), -1)
    right = np.where(any_row, np.where(mask, idx, -1).max(axis=1), -1)
    return left, right, mask.sum(axis=1)


def describe(name: str, path: str) -> dict:
    mask = figure_mask(path)
    h, w = mask.shape
    left, right, count = row_extent(mask)
    rows = np.flatnonzero(count > 0)
    top, bottom = int(rows[0]), int(rows[-1])
    height = bottom - top + 1

    # Normalised profile: 0 at the top of the head, 1 at the soles.
    steps = 200
    prof = []
    for i in range(steps):
        y = top + int(round(i * (height - 1) / (steps - 1)))
        if count[y] == 0:
            prof.append(None)
            continue
        prof.append(
            {
                "t": round(i / (steps - 1), 4),
                "left": round((left[y] - w / 2) / height, 4),
                "right": round((right[y] - w / 2) / height, 4),
                "fill": round(float(count[y]) / height, 4),
            }
        )

    cols = np.flatnonzero(mask.any(axis=0))
    return {
        "source": os.path.basename(path),
        "pixel_height": int(height),
        "pixel_width": int(cols[-1] - cols[0] + 1),
        "centre_x": round(float((cols[0] + cols[-1]) / 2 - w / 2) / height, 4),
        "profile": prof,
    }


def main() -> None:
    views = {
        "front": "albedo_front.jpeg",
        "back": "albedo_back.jpeg",
        "left": "albedo_left.jpeg",
        "right": "albedo_right.jpeg",
    }
    data = {}
    for key, fname in views.items():
        path = os.path.join(REF, fname)
        if not os.path.exists(path):
            print(f"skip {fname}: missing")
            continue
        data[key] = describe(key, path)
        d = data[key]
        print(f"{key:6s} h={d['pixel_height']} w={d['pixel_width']} cx={d['centre_x']}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
