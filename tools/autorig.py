"""ใส่โครงกระดูกและอนิเมชันเดินให้ .glb ที่มีแต่เนื้อ ไม่มี rig

ทำสามขั้นในไฟล์เดียว:

  1. voxelize  แปลง mesh เป็นกริด occupancy ตัน ด้วยกฎ even-odd
  2. ประเมินข้อต่อ  อ่านตำแหน่งข้อต่อจาก silhouette ของกริดนั้น
  3. เรียก Blender แบบ headless (tools/blender_rig.py) มาสร้าง armature
     ผูก skinning weight แล้วอบ clip ลงไฟล์

ชื่อกระดูกใช้แบบ Mixamo เพื่อให้ retarget อนิเมชันจากที่อื่นเข้ามาได้

สมมติฐานเรื่องท่าและแกน: ตัวละครยืนตรง หันหน้าไปทาง +Z, +Y คือขึ้น
และด้านซ้ายของตัวละครคือ +X (ระบบขวามือ: left = up x forward)
ถ้าโมเดลใหม่หันคนละทาง ต้องหมุนก่อนส่งเข้าเครื่องมือนี้

ใช้:
    python tools/autorig.py output/mesh.glb output/mesh_rigged.glb
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glbmesh import first_triangle_mesh, read_glb  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

BONE_PARENTS: dict[str, str | None] = {
    "Hips": None,
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Head": "Neck",
    "HeadTop_End": "Head",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
}


def say(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. voxelize
# ---------------------------------------------------------------------------


def voxelize(verts: np.ndarray, faces: np.ndarray, res_y: int = 192):
    """คืนกริด occupancy ตัน (ny, nx, nz) พร้อม transform กลับเป็นเมตร

    ยิงรังสีขนานแกน Z ผ่านกึ่งกลางของทุกคอลัมน์ (ix, iy) เก็บทุกจุดที่รังสี
    ตัดผิว เรียงตาม z แล้วเติมเนื้อระหว่างคู่ที่ 0-1, 2-3, ... (กฎ even-odd)
    ไม่ต้องสนทิศของ normal และทนต่อ mesh ที่ normal กลับด้านบางหน้า
    """
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    pad = (hi - lo).max() * 0.02
    lo, hi = lo - pad, hi + pad
    step = (hi[1] - lo[1]) / res_y
    nx = int(np.ceil((hi[0] - lo[0]) / step))
    ny = res_y
    nz = int(np.ceil((hi[2] - lo[2]) / step))

    tri = verts[faces]
    gx = (tri[:, :, 0] - lo[0]) / step - 0.5
    gy = (tri[:, :, 1] - lo[1]) / step - 0.5

    x0 = np.ceil(gx.min(axis=1)).astype(np.int64)
    x1 = np.floor(gx.max(axis=1)).astype(np.int64)
    y0 = np.ceil(gy.min(axis=1)).astype(np.int64)
    y1 = np.floor(gy.max(axis=1)).astype(np.int64)
    span = max(int((x1 - x0).max()), int((y1 - y0).max())) + 1

    hit_x: list[np.ndarray] = []
    hit_y: list[np.ndarray] = []
    hit_z: list[np.ndarray] = []
    for dy in range(span):
        for dx in range(span):
            ix = x0 + dx
            iy = y0 + dy
            ok = (ix <= x1) & (iy <= y1) & (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)
            sel = np.flatnonzero(ok)
            if sel.size == 0:
                continue
            px = ix[sel].astype(np.float64)
            py = iy[sel].astype(np.float64)
            ax, bx, cx_ = gx[sel, 0], gx[sel, 1], gx[sel, 2]
            ay, by, cy_ = gy[sel, 0], gy[sel, 1], gy[sel, 2]
            det = (by - cy_) * (ax - cx_) + (cx_ - bx) * (ay - cy_)
            good = np.abs(det) > 1e-12
            if not good.any():
                continue
            sel = sel[good]
            px, py, det = px[good], py[good], det[good]
            ax, bx, cx_ = ax[good], bx[good], cx_[good]
            ay, by, cy_ = ay[good], by[good], cy_[good]
            w0 = ((by - cy_) * (px - cx_) + (cx_ - bx) * (py - cy_)) / det
            w1 = ((cy_ - ay) * (px - cx_) + (ax - cx_) * (py - cy_)) / det
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            sel = sel[inside]
            z = (w0[inside] * tri[sel, 0, 2] + w1[inside] * tri[sel, 1, 2]
                 + w2[inside] * tri[sel, 2, 2])
            hit_x.append(ix[sel].astype(np.int64))
            hit_y.append(iy[sel].astype(np.int64))
            hit_z.append(z)

    ix = np.concatenate(hit_x)
    iy = np.concatenate(hit_y)
    z = np.concatenate(hit_z)
    key = iy * nx + ix
    order = np.lexsort((z, key))
    key, z = key[order], z[order]

    solid = np.zeros((ny, nx, nz), dtype=bool)
    bounds = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(key)]))
    leaky = 0
    for s, e in zip(starts, ends):
        n = e - s
        if n < 2:
            continue
        if n % 2:
            # จุดตัดเป็นเลขคี่แปลว่ารังสีเฉียดขอบผิวพอดีหรือผิวมีรู
            # ทิ้งจุดสุดท้ายไป ดีกว่าเติมเนื้อยาวไปจนสุดกริด
            leaky += 1
            n -= 1
        cy, cx_ = divmod(int(key[s]), nx)
        zi = (z[s:s + n] - lo[2]) / step - 0.5
        for j in range(0, n, 2):
            a = max(int(np.ceil(zi[j])), 0)
            b = min(int(np.floor(zi[j + 1])), nz - 1)
            if b >= a:
                solid[cy, cx_, a:b + 1] = True

    say(f"voxelize: {nx}x{ny}x{nz} step {step:.4f} m, "
        f"เนื้อ {solid.mean():.1%}, คอลัมน์ที่ผิวรั่ว {leaky}/{len(starts)}")
    return solid, lo, float(step)


# ---------------------------------------------------------------------------
# 2. ประเมินข้อต่อ
# ---------------------------------------------------------------------------


def _runs(row: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """ช่วงที่ติดกันของ True ในแถวเดียว คืนเป็น [start, end) """
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[cuts + 1]))
    ends = np.concatenate((idx[cuts], [idx[-1]])) + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


class Body:
    """ตัววัดสัดส่วนจากกริด occupancy"""

    def __init__(self, solid: np.ndarray, lo: np.ndarray, step: float):
        self.step = step
        self.lo = lo
        self.ny, self.nx, self.nz = solid.shape

        rows = np.flatnonzero(solid.any(axis=(1, 2)))
        if rows.size == 0:
            raise SystemExit("กริดว่าง ไม่มีเนื้อให้วิเคราะห์")
        self.y_foot, self.y_head = int(rows[0]), int(rows[-1])
        self.H = self.y_head - self.y_foot
        if self.H < 48:
            raise SystemExit(f"ตัวละครสูงแค่ {self.H} voxel — เพิ่ม --res")

        # แกนกลางลำตัว วัดจากช่วงอกซึ่งไม่มีหางและขามารบกวน
        chest = solid[self.at(0.62):self.at(0.80)]
        xs = np.flatnonzero(chest.any(axis=(0, 2)))
        self.cx = float(xs.mean())

        self.tail_mask, self.z_cut = self._find_tail(solid)
        self.body = solid & ~self.tail_mask
        self.front = self.body.any(axis=2)          # (ny, nx)
        self.solid = solid

    # -- ตัวช่วยพิกัด ------------------------------------------------------
    def at(self, frac: float) -> int:
        return int(round(self.y_foot + frac * self.H))

    def frac(self, iy: float) -> float:
        return (iy - self.y_foot) / self.H

    def world(self, ix: float, iy: float, iz: float) -> list[float]:
        return [float(self.lo[0] + (ix + 0.5) * self.step),
                float(self.lo[1] + (iy + 0.5) * self.step),
                float(self.lo[2] + (iz + 0.5) * self.step)]

    # -- หาง ---------------------------------------------------------------
    def _find_tail(self, solid: np.ndarray) -> tuple[np.ndarray, int]:
        """แยกหางออกจากลำตัวด้วยระนาบหลังของอก

        หางของตัวละครนี้ทอดไปทาง -Z แล้วลาดลง ส่วนขากับเท้าอยู่ในช่วง z ของ
        ลำตัวทั้งหมด ระนาบ "หลังของอก" จึงตัดสองอย่างนี้ออกจากกันได้ตรง ๆ
        โดยไม่ต้องไล่หาส่วนประกอบที่เชื่อมกัน
        """
        chest = solid[self.at(0.62):self.at(0.84)]
        zs = np.flatnonzero(chest.any(axis=(0, 1)))
        z_back = int(zs[0])
        z_cut = z_back - max(1, int(0.01 * self.H))

        mask = np.zeros_like(solid)
        hip_top = self.at(0.62)
        mask[:hip_top, :, :z_cut] = solid[:hip_top, :, :z_cut]
        return mask, z_cut

    def tail_path(self, n: int) -> np.ndarray | None:
        """แกนกลางหาง ไล่จากโคนไปปลาย คืน (n+1, 3) ในพิกัด voxel (ix, iy, iz)"""
        cells = np.argwhere(self.tail_mask)        # (k, 3) -> iy, ix, iz
        if cells.shape[0] < 50:
            return None
        iz = cells[:, 2]
        z_hi, z_lo = int(iz.max()), int(iz.min())
        if z_hi - z_lo < 6:
            return None

        # หางยาวไปตามแกน Z เป็นหลัก จึงหั่นเป็นชั้นตาม z แล้วเอาจุดกึ่งกลาง
        edges = np.linspace(z_hi, z_lo, n + 2)
        pts = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (iz <= a) & (iz >= b)
            if sel.sum() < 5:
                continue
            c = cells[sel]
            pts.append([c[:, 1].mean(), c[:, 0].mean(), c[:, 2].mean()])
        if len(pts) < 3:
            return None
        return np.array(pts, dtype=float)

    # -- ลำตัว -------------------------------------------------------------
    def z_center(self, iy: float, ix: float, half: int = 2) -> float:
        """กึ่งกลางความลึกของเนื้อตรงจุด (iy, ix)"""
        y = int(np.clip(round(iy), 0, self.ny - 1))
        x = int(np.clip(round(ix), 0, self.nx - 1))
        x0, x1 = max(0, x - half), min(self.nx, x + half + 1)
        col = self.body[y, x0:x1].any(axis=0)
        zs = np.flatnonzero(col)
        return float(zs.mean()) if zs.size else self.nz / 2


def estimate(b: Body, tail_bones: int = 5) -> tuple[dict[str, list[float]], dict]:
    joints: dict[str, np.ndarray] = {}
    notes: list[str] = []
    min_run = max(2, b.nx // 60)

    # --- หว่างขา ----------------------------------------------------------
    #
    # ไล่ *ขึ้น* จากข้อเท้า ไม่ใช่ลงมาจากเอว เพราะการไล่ลงจะหยุดที่ช่องว่างแรก
    # ที่เจอ ซึ่งมักเป็นรูเล็ก ๆ จากผิวที่รั่วตอน voxelize แล้วได้หว่างขาสูงเกิน
    # จริง (วัดได้ 40% ขณะที่ของจริงอยู่ 38%) การไล่ขึ้นแล้วบังคับว่าต้องมี
    # ช่องว่างติดกันหลายแถวและสองข้างต้องหนาพอ ทำให้รูเดี่ยว ๆ ไม่ผ่าน
    leg_w = max(2, int(0.05 * b.H))

    def split_row(y: int) -> bool:
        runs = _runs(b.front[y], min_run)
        for lhs, rhs in zip(runs[:-1], runs[1:]):
            if lhs[1] <= b.cx <= rhs[0] and min(lhs[1] - lhs[0], rhs[1] - rhs[0]) >= leg_w:
                return True
        return False

    crotch = None
    for y in range(b.at(0.62), b.at(0.12), -1):
        if all(split_row(k) for k in (y, y - 1, y - 2)):
            crotch = y + 1
            break
    if crotch is None:
        crotch = b.at(0.45)
        notes.append("หาหว่างขาไม่เจอ ใช้ 45% ของความสูงแทน")
    crotch = int(np.clip(crotch, b.at(0.28), b.at(0.58)))
    say(f"หว่างขาอยู่ที่ {b.frac(crotch):.0%} ของความสูง")

    # --- ขา ---------------------------------------------------------------
    ankle_y = b.y_foot + 0.05 * b.H
    knee_y = 0.5 * (crotch + ankle_y)
    leg_x: dict[str, float] = {}
    for side, sign in (("Left", +1), ("Right", -1)):
        xs = []
        for y in range(int(ankle_y), crotch):
            runs = _runs(b.front[y], min_run)
            pick = [r for r in runs if (0.5 * (r[0] + r[1]) - b.cx) * sign > 0]
            if pick:
                r = max(pick, key=lambda r: r[1] - r[0])
                xs.append(0.5 * (r[0] + r[1]))
        leg_x[side] = float(np.median(xs)) if xs else b.cx + sign * 0.08 * b.H

    for side in ("Left", "Right"):
        lx = leg_x[side]
        hip_x = b.cx + 0.55 * (lx - b.cx)
        joints[f"{side}UpLeg"] = np.array([hip_x, crotch, b.z_center(crotch, hip_x)])
        joints[f"{side}Leg"] = np.array([lx, knee_y, b.z_center(knee_y, lx)])
        joints[f"{side}Foot"] = np.array([lx, ankle_y, b.z_center(ankle_y, lx)])
        # ปลายเท้ายื่นไปทาง +Z ซึ่งเป็นทิศที่ตัวละครหันหน้า
        row = b.body[int(np.clip(round(b.y_foot + 0.02 * b.H), 0, b.ny - 1))]
        x = int(np.clip(round(lx), 0, b.nx - 1))
        zs = np.flatnonzero(row[max(0, x - 3):x + 4].any(axis=0))
        toe_z = float(zs[-1]) if zs.size else b.z_center(ankle_y, lx)
        joints[f"{side}ToeBase"] = np.array([lx, b.y_foot + 0.02 * b.H, toe_z])

    # --- คอ ---------------------------------------------------------------
    #
    # กฎ "คอคือจุดที่ลำตัวแคบที่สุด" ใช้กับหัวจระเข้ไม่ได้ ปากยาวยื่นไปทาง +Z
    # ภาพด้านหน้าจึงเห็นหัวแคบลงเรื่อย ๆ จนถึงปลายปาก จุดแคบที่สุดเลยไปตกที่
    # ปลายปากทุกครั้ง ใช้ "แถวที่ความกว้างลดเร็วที่สุด" แทน ซึ่งคือรอยต่อ
    # ระหว่างหัวกับไหล่จริง ๆ
    span = max(3, int(0.03 * b.H))
    width = b.front.sum(axis=1).astype(float)
    drop = [(width[y + span] - width[y - span], y)
            for y in range(b.at(0.78), b.at(0.95))]
    neck_y = min(drop)[1] if drop else b.at(0.88)

    # ไหล่อยู่ใต้คอลงมา ไม่ใช่แถวที่ silhouette กว้างที่สุด เพราะท่านี้แขน
    # ปล่อยลงข้างลำตัว แถวที่กว้างที่สุดจึงเป็นระดับข้อศอก ไม่ใช่หัวไหล่
    shoulder_y = int(neck_y - 0.06 * b.H)
    say(f"คออยู่ที่ {b.frac(neck_y):.0%} ไหล่ {b.frac(shoulder_y):.0%} ของความสูง")

    # --- แกนลำตัว ---------------------------------------------------------
    hips_y = crotch + 0.10 * b.H
    spine_rows = {
        "Hips": hips_y,
        "Spine": hips_y + 0.28 * (shoulder_y - hips_y),
        "Spine1": hips_y + 0.58 * (shoulder_y - hips_y),
        "Spine2": hips_y + 0.88 * (shoulder_y - hips_y),
        "Neck": neck_y,
    }
    for name, y in spine_rows.items():
        joints[name] = np.array([b.cx, y, b.z_center(y, b.cx)])

    head_y = 0.5 * (neck_y + b.y_head)
    joints["Head"] = np.array([b.cx, head_y, b.z_center(head_y, b.cx)])
    joints["HeadTop_End"] = np.array([b.cx, b.y_head - 0.01 * b.H,
                                      b.z_center(b.y_head - 0.03 * b.H, b.cx)])

    # --- แขน: ไล่ตามก้อนเนื้อที่แยกจากลำตัวลงมาจากไหล่ --------------------
    #
    # "ก้อนนอกสุดที่อยู่ฝั่งนี้" อย่างเดียวไม่พอ: แถวที่แขนยังติดลำตัวมีก้อนเดียว
    # คือลำตัว+แขน ซึ่งจุดกึ่งกลางเกือบตรงแกนกลางตัว ถ้ารับมาเป็นแขนจะได้ข้อไหล่
    # ไปกองอยู่กลางอก (วัดได้ x=58 ขณะที่แขนจริงอยู่ที่ x=90) จึงรับก้อนเป็นแขน
    # เฉพาะเมื่อขอบด้านในของมันพ้นลำตัวออกมาแล้วจริง ๆ
    apart = 0.10 * b.H
    for side, sign in (("Left", +1), ("Right", -1)):
        path: list[tuple[float, float]] = []
        free: list[int] = []            # index ในเส้นทางที่แขนแยกจากลำตัวแล้ว
        for y in range(shoulder_y, b.at(0.20), -1):
            runs = _runs(b.front[y], min_run)
            if not runs:
                continue
            if sign > 0:
                outer = [r for r in runs if r[0] - b.cx > apart]
            else:
                outer = [r for r in runs if b.cx - r[1] > apart]
            if outer:
                r = outer[-1] if sign > 0 else outer[0]
                path.append((0.5 * (r[0] + r[1]), y))
                free.append(len(path) - 1)
            else:
                # ยังติดลำตัว ใช้ขอบนอกสุดของ silhouette เป็นแนวแขน
                edge = runs[-1][1] if sign > 0 else runs[0][0]
                path.append((b.cx + 0.86 * (edge - b.cx), y))

        # ปลายแขนคือแถวแยกตัวแถวสุดท้าย — ต่ำกว่านั้นขอบ silhouette เป็นสะโพก
        # ไม่ใช่มือ ถ้าไล่ต่อจะได้ข้อมือไปอยู่กลางตัว (x=78 แทนที่จะเป็น 91)
        #
        # ไม่ตัดที่แถวแรกที่กลับไปติดลำตัว เพราะแขนแตะสะโพกเป็นช่วง ๆ ระหว่างทาง
        # อยู่แล้ว ตัดแบบนั้นจะได้แขนยาวแค่ 15 voxel จากของจริง 80
        seen = set(free)
        tip = None
        for i in reversed(free):
            if sum(1 for k in range(i - 4, i + 1) if k in seen) >= 3:
                tip = i
                break
        arr = np.array(path[:tip + 1], float) if tip else np.empty((0, 2))

        if arr.shape[0] < 8:
            notes.append(f"ไล่ตามแขนฝั่ง {side} ไม่สำเร็จ ใช้สัดส่วนมาตรฐานแทน")
            outer_x = b.cx + sign * 0.16 * b.H
            arr = np.array([[outer_x, shoulder_y - t * 0.34 * b.H]
                            for t in np.linspace(0, 1, 12)])

        def along(frac: float, arr=arr) -> np.ndarray:
            k = int(np.clip(round(frac * (arr.shape[0] - 1)), 0, arr.shape[0] - 1))
            x, y = arr[k]
            return np.array([x, y, b.z_center(y, x)])

        sh = along(0.05)
        joints[f"{side}Shoulder"] = np.array(
            [b.cx + 0.30 * (sh[0] - b.cx), shoulder_y, b.z_center(shoulder_y, b.cx)])
        joints[f"{side}Arm"] = sh
        joints[f"{side}ForeArm"] = along(0.45)
        joints[f"{side}Hand"] = along(0.92)

    parents = dict(BONE_PARENTS)

    # --- หาง ---------------------------------------------------------------
    path = b.tail_path(tail_bones)
    if path is not None:
        for i, p in enumerate(path):
            joints[f"Tail{i + 1}"] = p
            parents[f"Tail{i + 1}"] = "Hips" if i == 0 else f"Tail{i}"
        say(f"หาง {len(path)} ข้อ ยาว "
            f"{np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / b.H:.0%} ของความสูงตัว")
    else:
        notes.append("ไม่พบหาง จึงไม่ได้ใส่กระดูกหาง")

    world = {name: b.world(*p) for name, p in joints.items()}
    for n in notes:
        say(f"  หมายเหตุ: {n}")
    return world, {"joints": world, "parents": parents, "notes": notes}


def n_tail_bones(parents: dict) -> int:
    return sum(1 for k in parents if k.startswith("Tail"))


# ---------------------------------------------------------------------------
# 3. อนิเมชัน — ทุก channel คือคลื่นไซน์หนึ่งลูกบนกระดูกหนึ่งชิ้น
# ---------------------------------------------------------------------------


def _ch(bone, axis="X", amp=0.0, phase=0.0, offset=0.0, harmonic=1.0,
        mode="sin", kind="rot") -> dict:
    return {"bone": bone, "axis": axis, "amp": amp, "phase": phase,
            "offset": offset, "harmonic": harmonic, "mode": mode, "kind": kind}


def tail_wave(n, axis="Z", amp_first=4.0, amp_last=16.0, phase_step=0.09,
              phase0=0.0, harmonic=1.0) -> list[dict]:
    """คลื่นวิ่งไล่จากโคนหางไปปลายหาง

    หัวใจคือ phase_step: แต่ละข้อเริ่มขยับช้ากว่าข้อก่อนหน้าเล็กน้อย หางจึง
    สะบัดเป็นคลื่น ไม่ใช่แกว่งทั้งท่อนพร้อมกันเหมือนไม้แข็ง
    """
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        out.append(_ch(f"Tail{i + 1}", axis,
                       amp=amp_first + (amp_last - amp_first) * t,
                       phase=phase0 + phase_step * i, harmonic=harmonic))
    return out


def clips(n_tail: int, fps: int = 24) -> dict:
    idle = [
        _ch("Spine", "X", amp=0.8), _ch("Spine1", "X", amp=1.2, phase=0.05),
        _ch("Spine2", "X", amp=0.9, phase=0.10), _ch("Neck", "X", amp=-0.8, phase=0.15),
        _ch("Head", "Y", amp=1.2, harmonic=0.5, phase=0.25),
        _ch("LeftArm", "Y", amp=1.5, phase=0.3), _ch("RightArm", "Y", amp=-1.5, phase=0.3),
    ] + tail_wave(n_tail, amp_first=2.0, amp_last=9.0)

    walk = [
        # ขา: ค่าลบ = เหวี่ยงไปข้างหน้า สองข้างต่างเฟสกันครึ่งลูป
        _ch("RightUpLeg", "X", amp=-26, phase=0.00),
        _ch("LeftUpLeg", "X", amp=-26, phase=0.50),
        # เข่าใช้ halfsin เพราะงอได้ทางเดียว
        _ch("RightLeg", "X", amp=40, phase=0.55, mode="halfsin"),
        _ch("LeftLeg", "X", amp=40, phase=0.05, mode="halfsin"),
        _ch("RightFoot", "X", amp=-10, phase=0.30),
        _ch("LeftFoot", "X", amp=-10, phase=0.80),
        # แขนแกว่งสวนกับขาข้างเดียวกัน
        _ch("RightArm", "X", amp=18, phase=0.00),
        _ch("LeftArm", "X", amp=18, phase=0.50),
        _ch("RightForeArm", "X", amp=10, offset=8, phase=0.10),
        _ch("LeftForeArm", "X", amp=10, offset=8, phase=0.60),
        # ลำตัวบิดสวนกับสะโพก
        _ch("Spine1", "Z", amp=4), _ch("Spine2", "Z", amp=3, phase=0.05),
        _ch("Hips", "Z", amp=-5),
        # ย่อ-ยืดสองครั้งต่อหนึ่งรอบ ตามจังหวะก้าว
        _ch("Hips", "Z", amp=0.035, harmonic=2, phase=0.25, kind="loc"),
    ] + tail_wave(n_tail, amp_first=3.0, amp_last=14.0, phase_step=0.10)

    out = [{"name": "Idle", "frames": 64, "channels": idle},
           {"name": "Walk", "frames": 32, "channels": walk}]
    if n_tail:
        out.append({"name": "TailSwing", "frames": 40,
                    "channels": tail_wave(n_tail, amp_first=8.0, amp_last=28.0,
                                          phase_step=0.12)
                    + [_ch("Hips", "Z", amp=2.0, phase=0.5)]})
    return {"fps": fps, "clips": out}


# ---------------------------------------------------------------------------
# 4. เรียก Blender
# ---------------------------------------------------------------------------


def find_blender(explicit: str | None) -> Path:
    import os

    cands = [explicit, os.environ.get("BLENDER_EXE")]
    cands += sorted(
        (str(p) for p in Path(r"C:\Program Files\Blender Foundation").glob("*/blender.exe")),
        reverse=True)
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    raise SystemExit("หา blender.exe ไม่เจอ — ตั้ง BLENDER_EXE ให้ชี้ไปที่ไฟล์นั้น")


def run_blender(exe: Path, mesh: Path, joints: Path, anim: Path, out: Path,
                timeout: int = 1800) -> None:
    script = Path(__file__).resolve().parent / "blender_rig.py"
    cmd = [str(exe), "--background", "--factory-startup", "--python", str(script), "--",
           "--mesh", str(mesh), "--joints", str(joints), "--out", str(out),
           "--anim", str(anim), "--draco", "0"]
    say(f"เรียก Blender: {exe.name}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")
    for line in (proc.stdout or "").splitlines():
        if line.startswith("[rig]"):
            say("  " + line[5:].strip())

    # Blender คืน exit 0 ได้แม้สคริปต์ traceback กลางคัน จึงต้องให้สคริปต์
    # ประกาศความสำเร็จของตัวเอง ไม่งั้นจะได้ไฟล์เก่าค้างมาแล้วนึกว่าสำเร็จ
    if proc.returncode != 0 or not out.exists() or "[rig] เสร็จสมบูรณ์" not in (proc.stdout or ""):
        tail = "\n".join((proc.stdout or "").splitlines()[-30:])
        err = "\n".join((proc.stderr or "").splitlines()[-15:])
        raise SystemExit(f"Blender ไม่สำเร็จ\n--- stdout ---\n{tail}\n--- stderr ---\n{err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--res", type=int, default=192, help="ความสูงของกริด voxel")
    ap.add_argument("--tail-bones", type=int, default=5)
    ap.add_argument("--blender", default=None)
    ap.add_argument("--work", type=Path, default=None,
                    help="โฟลเดอร์เก็บ joints/anim json (ค่าเริ่มต้น: ข้าง dst)")
    args = ap.parse_args()

    gltf, blob = read_glb(args.src)
    verts, faces = first_triangle_mesh(gltf, blob)
    say(f"{args.src.name}: {len(verts):,} จุดยอด {len(faces):,} สามเหลี่ยม")

    solid, lo, step = voxelize(verts, faces, args.res)
    body = Body(solid, lo, step)
    say(f"สูง {body.H * step:.2f} m ({body.H} voxel)")
    world, payload = estimate(body, args.tail_bones)

    work = args.work or args.dst.parent
    work.mkdir(parents=True, exist_ok=True)
    jp = work / (args.dst.stem + "_joints.json")
    apath = work / (args.dst.stem + "_anim.json")
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    apath.write_text(json.dumps(clips(n_tail_bones(payload["parents"])),
                                indent=2, ensure_ascii=False), encoding="utf-8")
    say(f"ข้อต่อ {len(world)} จุด -> {jp.name}")

    run_blender(find_blender(args.blender), args.src.resolve(), jp.resolve(),
                apath.resolve(), args.dst.resolve())
    say(f"เสร็จ -> {args.dst} ({args.dst.stat().st_size / 1048576:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
