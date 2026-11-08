"""Dump the figure mask for each reference so the threshold can be eyeballed."""

import os
import sys

import numpy as np
import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_refs as mr  # noqa: E402

OUTDIR = os.path.join(mr.ROOT, "output", "debug")
os.makedirs(OUTDIR, exist_ok=True)


def save_gray(arr: np.ndarray, path: str) -> None:
    h, w = arr.shape
    img = bpy.data.images.new(os.path.basename(path), width=w, height=h, alpha=False)
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = arr
    rgba[..., 3] = 1.0
    img.pixels.foreach_set(rgba[::-1].ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


for fname in sorted(os.listdir(mr.REF)):
    if not fname.endswith((".jpeg", ".jpg", ".png")):
        continue
    if "hand" in fname:
        continue
    path = os.path.join(mr.REF, fname)
    rgb = mr.shrink(mr.load_rgb(path))
    sat = mr.saturation(rgb)
    val = rgb.max(axis=2)
    stem = os.path.splitext(fname)[0]
    print(
        f"{stem:22s} sat med={np.median(sat):.3f} p90={np.percentile(sat, 90):.3f} "
        f"val med={np.median(val):.3f} p10={np.percentile(val, 10):.3f}"
    )
    save_gray(sat, os.path.join(OUTDIR, f"{stem}_sat.png"))
    save_gray(mr.figure_mask(path).astype(np.float32), os.path.join(OUTDIR, f"{stem}_mask.png"))
print("wrote", OUTDIR)
