"""ขั้นที่ 5: ประเมินตำแหน่งข้อต่อจากรูปทรงที่ได้ เพื่อเอาไปวางโครงกระดูก

วิเคราะห์จาก silhouette ด้านหน้าของกริด voxel (ฉาย solid ลงแกน Z) ซึ่งอ่าน
ตำแหน่งข้อต่อได้ตรงไปตรงมากว่าการไปงมใน mesh

ชื่อกระดูกใช้ตามแบบ Mixamo (`Hips`, `LeftUpLeg`, `Spine2`, ...) ตั้งใจเลือก
แบบนี้เพราะทำให้ retarget อนิเมชันจาก Mixamo หรือ AMASS เข้ามาได้ง่ายขึ้นมาก

ข้อควรรู้: นี่คือการ *ประเมิน* จากเรขาคณิต ไม่ใช่การตรวจจับท่าด้วยโมเดล AI
มันใช้ได้ดีกับท่า A-pose/T-pose ที่ยืนตรงและแขนไม่ทับลำตัว ถ้าต้องการ
คุณภาพระดับ production การส่ง mesh ไป auto-rig ที่ Mixamo ยังให้ skinning
weight ที่ดีกว่า (ใช้ `--rig none` เพื่อ export ไปทางนั้น)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import PipelineConfig
from .visual_hull import HullGrid

log = logging.getLogger(__name__)

# ลำดับชั้นของกระดูก: ชื่อลูก -> ชื่อพ่อ (None = ราก)
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


@dataclass
class Skeleton:
    """ตำแหน่งข้อต่อในพิกัด voxel index (ix, iy, iz)"""

    joints: dict[str, np.ndarray]
    # ลำดับชั้นของโครงนี้โดยเฉพาะ — ไม่ใช่ BONE_PARENTS เสมอไป เพราะจำนวน
    # กระดูกหางขึ้นกับว่าตรวจเจอหางหรือไม่ และตั้งไว้กี่ข้อ
    parents: dict[str, str | None] = field(default_factory=lambda: dict(BONE_PARENTS))
    notes: list[str] = field(default_factory=list)

    def to_world(self, verts_vox: np.ndarray, verts_world: np.ndarray) -> dict[str, list[float]]:
        """แปลงข้อต่อไปหน่วยเมตร โดยอนุมาน transform จาก vertex ก่อน/หลังแปลง

        ใช้วิธีนี้แทนการคิดสูตรซ้ำ เพื่อให้ข้อต่อกับ mesh ใช้ transform
        ตัวเดียวกันเป๊ะ ๆ ไม่มีทางหลุดกัน
        """
        out: dict[str, list[float]] = {}
        scale = np.empty(3)
        offset = np.empty(3)
        for ax in range(3):
            lo_v, hi_v = verts_vox[:, ax].min(), verts_vox[:, ax].max()
            lo_w, hi_w = verts_world[:, ax].min(), verts_world[:, ax].max()
            span = hi_v - lo_v
            scale[ax] = (hi_w - lo_w) / span if span > 1e-9 else 1.0
            offset[ax] = lo_w - lo_v * scale[ax]

        for name, p in self.joints.items():
            out[name] = (np.asarray(p, float) * scale + offset).tolist()
        return out


# ---------------------------------------------------------------------------
# เครื่องมือวิเคราะห์ silhouette
# ---------------------------------------------------------------------------


def _runs(row: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """หาช่วงต่อเนื่องของ True ในแถวเดียว คืน [(start, end), ...] แบบรวมปลาย"""
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_len]


def _resample_path(path: np.ndarray, n_points: int) -> np.ndarray:
    """สุ่มจุดใหม่บนเส้นทางให้ห่างเท่ากันตามความยาวจริงของเส้น

    ไม่ใช่ห่างเท่ากันตามดัชนีแถว เพราะช่วงที่หางโค้งมากจะมีจุดหนาแน่นกว่า
    ทำให้กระดูกยาวไม่เท่ากันและข้อต่อไปกระจุกตรงโคน
    """
    deltas = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(deltas)])
    if s[-1] <= 1e-6:
        return np.repeat(path[:1], n_points, axis=0)
    want = np.linspace(0.0, s[-1], n_points)
    return np.stack([np.interp(want, s, path[:, k]) for k in range(3)], axis=1)


def tail_joints(
    tail_path: np.ndarray, n_bones: int
) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
    """สร้างโซ่กระดูกหางจากแกนกลางหาง

    ได้ Tail1..TailN บวกข้อปลาย TailEnd หนึ่งข้อ ข้อปลายมีไว้บอกทิศของกระดูก
    ชิ้นสุดท้าย เหมือนที่ HeadTop_End ทำให้กระดูก Head — ถ้าไม่มี ตัวสร้าง
    armature จะต้องเดาทิศเอาเอง
    """
    pts = _resample_path(tail_path, n_bones + 1)
    names = [f"Tail{i + 1}" for i in range(n_bones)] + ["TailEnd"]

    joints = {name: pts[i].astype(float) for i, name in enumerate(names)}
    parents: dict[str, str | None] = {"Tail1": "Hips"}
    for a, b in zip(names, names[1:]):
        parents[b] = a
    return joints, parents


def _center_x(front: np.ndarray, y_foot: int, H: int, min_run: int) -> float:
    """ตำแหน่งกลางลำตัวในแกน X หาจากจุดกึ่งกลางระหว่างเท้าสองข้าง

    วัดที่ระดับข้อเท้า แล้วเอาสองท่อนที่หนาที่สุด (= เท้าซ้ายขวา) มาหาจุดกึ่งกลาง

    ใช้ centroid ของช่วงล่างตรง ๆ ไม่ได้ เพราะส่วนที่ยื่นอย่างปลายหางจะลาก
    ค่ากลางไปข้างหนึ่งจนไปตกทับขา แล้วการตรวจหว่างขาจะพังทั้งหมด
    """
    for frac in (0.08, 0.14, 0.20):
        y = y_foot + int(frac * H)
        if y >= front.shape[0]:
            continue
        runs = _runs(front[y], min_run)
        if len(runs) >= 2:
            two = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:2]
            return float(np.mean([0.5 * (a + b) for a, b in two]))
        if len(runs) == 1:
            return 0.5 * (runs[0][0] + runs[0][1])

    band = front[y_foot : y_foot + max(1, int(0.2 * H))]
    if not band.any():
        return front.shape[1] / 2.0
    return float(np.where(band)[1].mean())


def _has_central_gap(row: np.ndarray, center_x: float, min_len: int) -> bool:
    """แถวนี้มีช่องว่างที่คร่อมแกนกลางลำตัวหรือไม่ (= ขาแยกกันแล้ว)

    ดูเฉพาะช่องว่าง *ภายใน* silhouette คือระหว่างขอบซ้ายสุดกับขวาสุด
    และต้องเป็นช่องที่ครอบ center_x เพื่อไม่ให้ช่องระหว่างมือกับลำตัวมานับด้วย
    """
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return False
    lo, hi = int(idx[0]), int(idx[-1])
    if not lo < center_x < hi:
        return False
    for g0, g1 in _runs(~row[lo : hi + 1], min_len):
        if (g0 + lo) <= center_x <= (g1 + lo):
            return True
    return False


def _central_run(row: np.ndarray, center_x: float, min_len: int) -> tuple[int, int] | None:
    """ท่อนของ silhouette ที่คร่อมแกนกลางลำตัวในแถวนี้ (= ลำตัว ไม่ใช่แขนหรือหาง)"""
    for a, b in _runs(row, min_len):
        if a <= center_x <= b:
            return a, b
    return None


def _widest_central_row(
    front: np.ndarray, center_x: float, y_lo: int, y_hi: int, min_len: int
) -> int:
    """แถวที่ลำตัวผายกว้างที่สุดในช่วงที่ระบุ — ใช้หาระดับสะโพก

    วัดเฉพาะท่อนที่คร่อมแกนกลาง เพื่อไม่ให้แขนที่กางออกมาปนเข้ามาในความกว้าง
    """
    best_y, best_w = y_lo, -1
    for y in range(max(y_lo, 0), min(y_hi, front.shape[0] - 1) + 1):
        run = _central_run(front[y], center_x, min_len)
        if run is None:
            continue
        w = run[1] - run[0]
        if w > best_w:
            best_y, best_w = y, w
    return best_y


def _depth_center_profile(solid: np.ndarray, center_x: float) -> np.ndarray:
    """แกนกลางความลึก (Z) ของลำตัวในแต่ละแถว

    วัดเฉพาะแถบแคบ ๆ รอบแกนกลางลำตัว เพื่อไม่ให้แขนที่กางออกหรือหางที่ยื่น
    ไปด้านหลังมาดึงค่า
    """
    ny, nx, nz = solid.shape
    w = max(1, nx // 20)
    lo = int(np.clip(center_x - w, 0, nx - 1))
    hi = int(np.clip(center_x + w, 0, nx - 1)) + 1

    out = np.full(ny, np.nan, np.float32)
    for y in range(ny):
        idx = np.flatnonzero(solid[y, lo:hi].any(axis=0))
        if idx.size:
            out[y] = 0.5 * (idx[0] + idx[-1])

    good = np.flatnonzero(~np.isnan(out))
    if good.size:
        out = np.interp(np.arange(ny), good, out[good])
    else:
        out[:] = nz / 2.0
    return ndimage.uniform_filter1d(out, size=5, mode="nearest")


def _neck_from_depth_shift(
    solid: np.ndarray, center_x: float, y_lo: int, y_hi: int
) -> tuple[int, float]:
    """หาคอจากจุดที่แกนความลึกเลื่อนเร็วที่สุด

    หัวของสัตว์ที่มีปากยื่น (และของมนุษย์ในระดับที่น้อยกว่า) อยู่ค่อนไปทาง
    ด้านหน้าของลำตัว แกนกลางความลึกจึงกระโดดไปข้างหน้าอย่างรวดเร็วตรงรอยต่อคอ
    สัญญาณนี้ใช้ได้แม้ silhouette ด้านหน้าจะไม่มีรอยคอดที่คอเลย

    คืน (แถว, ขนาดการเลื่อนต่อแถว)
    """
    zc = _depth_center_profile(solid, center_x)
    grad = np.abs(np.gradient(zc))
    lo = max(y_lo, 1)
    hi = min(y_hi, len(grad) - 2)
    if hi <= lo:
        return (lo + hi) // 2, 0.0
    band = grad[lo : hi + 1]
    y = lo + int(np.argmax(band))
    return y, float(band.max())


def _z_center(solid: np.ndarray, iy: float, ix: float) -> float:
    """ความลึก (แกน Z) ที่กึ่งกลางเนื้อ ณ ตำแหน่ง (iy, ix) ที่ระบุ"""
    ny, nx, nz = solid.shape
    y = int(np.clip(round(iy), 0, ny - 1))
    x = int(np.clip(round(ix), 0, nx - 1))
    idx = np.flatnonzero(solid[y, x])
    if idx.size == 0:
        # ไม่มีเนื้อตรงนั้นพอดี — ขยายการค้นหาไปทั้งแถว
        idx = np.flatnonzero(solid[y].any(axis=0))
        if idx.size == 0:
            return nz / 2.0
    return 0.5 * float(idx[0] + idx[-1])


# ---------------------------------------------------------------------------
# ประเมินข้อต่อ
# ---------------------------------------------------------------------------


def estimate(grid: HullGrid, cfg: PipelineConfig | None = None) -> Skeleton:
    """อ่านตำแหน่งข้อต่อจากกริด occupancy"""
    solid = grid.occupancy > 0.5
    ny, nx, nz = solid.shape
    front = solid.any(axis=2)                       # (ny, nx) silhouette ด้านหน้า

    rows_with_body = np.flatnonzero(front.any(axis=1))
    if rows_with_body.size == 0:
        raise ValueError("กริดว่าง ไม่มีเนื้อให้วิเคราะห์")
    y_foot, y_head = int(rows_with_body[0]), int(rows_with_body[-1])
    H = y_head - y_foot
    if H < 32:
        raise ValueError(f"ตัวละครสูงแค่ {H} voxel — เพิ่ม voxel_res ก่อน")

    widths = front.sum(axis=1).astype(float)
    min_run = max(2, nx // 60)
    notes: list[str] = []

    def at(frac: float) -> int:
        """แปลงสัดส่วนความสูง (0 = ฝ่าเท้า, 1 = ยอดหัว) เป็น index แถว"""
        return int(round(y_foot + frac * H))

    # --- หว่างขา ----------------------------------------------------------
    #
    # นับเฉพาะช่องว่างที่ *คร่อมแกนกลางลำตัว* เท่านั้น
    # การนับแค่ "แถวนี้มีกี่ท่อน" ใช้ไม่ได้ เพราะท่า A-pose มีช่องว่างระหว่าง
    # มือกับสะโพกอยู่แล้วที่ระดับเอว ซึ่งอยู่เหนือหว่างขาจริงขึ้นไปมาก
    # ถ้าไม่กรอง จะได้หว่างขาสูงเกินจริงราว 10% ของความสูงตัว
    body_cx_probe = _center_x(front, y_foot, H, min_run)

    split_y = None
    for y in range(at(0.60), y_foot, -1):
        if _has_central_gap(front[y], body_cx_probe, min_run):
            split_y = y + 1
            break

    # จุดที่ silhouette แยก กับระดับข้อสะโพกจริง เป็นคนละอย่างกัน
    #
    # ตัวละครที่ใส่กางเกงหลวมหรือต้นขาชิดกัน ภาพจะยังเห็นเป็นก้อนเดียวต่อลงมา
    # อีกไกลกว่าจะเห็นช่องว่าง ถ้าเอาจุดแยกมาเป็นข้อสะโพกตรง ๆ กระดูกต้นขา
    # จะสั้นกุดและข้อสะโพกจะต่ำลงไปอยู่กลางต้นขา
    #
    # จึงรับค่าที่วัดได้เฉพาะเมื่ออยู่ในช่วงที่เป็นไปได้ตามสัดส่วนร่างกาย
    # นอกช่วงนั้นถือว่าวัดไม่ได้ แล้วหันไปใช้ตำแหน่งที่สะโพกผายกว้างที่สุดแทน
    split_frac = (split_y - y_foot) / H if split_y is not None else None
    if split_frac is not None and 0.38 <= split_frac <= 0.58:
        crotch_y = split_y
    else:
        hip_y = _widest_central_row(front, body_cx_probe, at(0.30), at(0.62), min_run)
        crotch_y = int(np.clip(hip_y - 0.05 * H, at(0.40), at(0.54)))
        if split_frac is None:
            notes.append("หาจุดที่ขาแยกไม่เจอเลย — ใช้ตำแหน่งที่สะโพกผายกว้างสุดแทน")
        else:
            notes.append(
                f"ขาเริ่มแยกที่ {split_frac:.0%} ของความสูง ซึ่งต่ำกว่าข้อสะโพกตามสัดส่วนร่างกาย "
                "(ต้นขาชิดกันหรือใส่กางเกงหลวม) — วางข้อสะโพกจากตำแหน่งที่สะโพกผายกว้างสุดแทน"
            )
    log.info("หว่างขาอยู่ที่ %.0f%% ของความสูง", 100 * (crotch_y - y_foot) / H)

    # --- ขาซ้าย/ขวา: ดูที่ระดับกลางน่อง ซึ่งขาแยกกันชัด -------------------
    probe_y = int(np.clip((crotch_y + y_foot) / 2, y_foot + 1, crotch_y - 1))
    leg_runs = _runs(front[probe_y], min_run)
    if len(leg_runs) >= 2:
        # เอาสองท่อนที่ *หนาที่สุด* ไม่ใช่ซ้ายสุดกับขวาสุด
        # ที่ระดับน่อง หางมักพาดลงมาเป็นท่อนที่สามและอยู่ริมสุด
        # ถ้าหยิบตามตำแหน่งจะได้หางมาเป็นขาข้างหนึ่ง
        two = sorted(leg_runs, key=lambda r: r[1] - r[0], reverse=True)[:2]
        two.sort(key=lambda r: r[0])
        right_leg_x = 0.5 * (two[0][0] + two[0][1])    # ix น้อย = -X = ขาขวา
        left_leg_x = 0.5 * (two[1][0] + two[1][1])
    else:
        body = _runs(front[probe_y], min_run)
        c0, c1 = (body[0] if body else (nx * 0.4, nx * 0.6))
        center = 0.5 * (c0 + c1)
        half = 0.25 * (c1 - c0)
        right_leg_x, left_leg_x = center - half, center + half
        notes.append("ขาแยกไม่ชัดที่ระดับกลางน่อง — เดาตำแหน่งขาจากความกว้างลำตัว")

    # --- ไหล่: แถวที่ลำตัวผายกว้างที่สุดในช่วงบน ---------------------------
    #
    # หาไหล่ก่อนคอ เพราะไหล่เป็นจุดที่เด่นชัดกว่ามาก ส่วนคอนั้นบางตัวละคร
    # ไม่มีรอยคอดเลย (คอสัตว์เลื้อยคลาน คอหนา เกราะคอ) การไล่หาแถวที่แคบสุด
    # ในช่วงกว้าง ๆ จึงมักไปเจอยอดหัวแทนคอ
    shoulder_y = _widest_central_row(front, body_cx_probe, at(0.64), at(0.88), min_run)

    sh_run = _central_run(front[shoulder_y], body_cx_probe, min_run)
    s0, s1 = sh_run if sh_run else (nx * 0.35, nx * 0.65)

    # --- คอ: แถวที่แคบที่สุดเหนือไหล่ ภายในกรอบสัดส่วนที่เป็นไปได้ ---------
    #
    # ต้องมีกรอบกำกับ เพราะบางตัวละครไม่มีรอยคอดที่คอเลย (คอสัตว์เลื้อยคลาน
    # คอหนา เกราะคอ หัวที่แคบกว่าคอ) พอไม่มีจุดต่ำสุดจริง การไล่หาจะวิ่ง
    # ไปชนขอบบนของช่วงค้นหาเสมอ แล้วได้หัวที่สูงแค่ไม่กี่เปอร์เซ็นต์ของตัว
    #
    # คอของมนุษย์อยู่ราว 82–87% ของความสูง เราจึงค้นในกรอบนั้นบวกลบเล็กน้อย
    neck_lo = max(shoulder_y + max(2, int(0.02 * H)), at(0.76))
    neck_hi = min(y_head - max(2, int(0.06 * H)), at(0.90))
    neck_y = at(0.85)
    if neck_hi > neck_lo:
        best_w = np.inf
        for y in range(neck_lo, neck_hi + 1):
            run = _central_run(front[y], body_cx_probe, min_run)
            if run is None:
                continue
            w = run[1] - run[0]
            if w < best_w:
                best_w, neck_y = w, y
        if neck_y >= neck_hi - 1:
            # ความกว้างใช้ไม่ได้ (ไม่มีรอยคอด) หันไปใช้จุดที่แกนความลึกเลื่อนเร็วสุด
            alt_y, strength = _neck_from_depth_shift(solid, body_cx_probe,
                                                     neck_lo, neck_hi)
            body_depth = max(float(np.ptp(np.where(solid.any(axis=(0, 1)))[0])), 1.0)
            if strength > 0.01 * body_depth and alt_y < neck_hi - 1:
                notes.append(
                    f"ไม่มีรอยคอดที่คอใน silhouette ด้านหน้า — หาคอจากจุดที่แกนความลึก"
                    f"เลื่อนไปข้างหน้าเร็วที่สุดแทน ได้ที่ {(alt_y - y_foot) / H:.0%} ของความสูง"
                )
                neck_y = alt_y
            else:
                notes.append(
                    "ลำตัวแคบลงเรื่อย ๆ จนถึงยอดหัว และแกนความลึกก็ไม่เลื่อนชัด — "
                    "วางคอที่ขอบบนของช่วงที่เป็นไปได้ (90% ของความสูง)"
                )
    else:
        notes.append("ช่วงระหว่างไหล่กับยอดหัวแคบเกินกว่าจะหาคอได้ — ใช้ 85% ของความสูง")

    # --- ปรับไหล่อีกครั้งเมื่อรู้ตำแหน่งคอที่เชื่อถือได้แล้ว ------------------
    #
    # รอบแรกเราหาไหล่จากช่วงกว้าง ๆ ซึ่งมักไปเจอจุดที่อกหรือปีกหลังผายกว้างสุด
    # ซึ่งต่ำกว่าข้อไหล่จริงพอสมควร ข้อไหล่อยู่ใต้คอลงมาไม่กี่เปอร์เซ็นต์เสมอ
    # พอได้คอแล้วจึงจำกัดช่วงค้นหาใหม่ให้เกาะกับคอ
    sh_lo = max(at(0.58), neck_y - int(0.16 * H))
    sh_hi = max(sh_lo + 1, neck_y - int(0.02 * H))
    shoulder_y = _widest_central_row(front, body_cx_probe, sh_lo, sh_hi, min_run)
    sh_run = _central_run(front[shoulder_y], body_cx_probe, min_run)
    if sh_run:
        s0, s1 = sh_run

    head_frac = (y_head - neck_y) / H
    if head_frac < 0.08:
        notes.append(
            f"หัวสูงแค่ {head_frac:.0%} ของตัว ซึ่งน้อยผิดปกติ — "
            "คออาจถูกวางสูงเกินไป กระดูก Head จะสั้นกว่าที่ควร"
        )
    shoulder_center = 0.5 * (s0 + s1)
    shoulder_half = 0.5 * (s1 - s0)
    # ข้อต่อไหล่อยู่ลึกเข้ามาจากขอบ silhouette ไม่ได้อยู่ที่ขอบพอดี
    left_sh_x = shoulder_center + 0.62 * shoulder_half
    right_sh_x = shoulder_center - 0.62 * shoulder_half

    # --- ปลายมือ: จุดที่ยื่นออกด้านข้างสุดในช่วงที่แขนกวาดถึง -------------
    arm_lo, arm_hi = at(0.35), at(0.92)
    band = front[arm_lo:arm_hi]
    ys, xs = np.where(band)
    if xs.size:
        li = int(np.argmax(xs))
        ri = int(np.argmin(xs))
        left_hand = np.array([xs[li], arm_lo + ys[li]], float)
        right_hand = np.array([xs[ri], arm_lo + ys[ri]], float)
    else:
        left_hand = np.array([s1, at(0.5)], float)
        right_hand = np.array([s0, at(0.5)], float)
        notes.append("หาปลายมือไม่เจอ — เดาจากความกว้างไหล่")

    span = abs(left_hand[0] - right_hand[0])
    if span < 1.1 * (s1 - s0):
        notes.append(
            f"ช่วงแขน ({span:.0f} voxel) แคบกว่าไหล่มาก — ท่าทางอาจไม่ใช่ A-pose "
            "หรือแขนทับลำตัวอยู่ ทำให้ตำแหน่งข้อศอก/ข้อมือคลาดเคลื่อน"
        )

    # --- ประกอบข้อต่อ -----------------------------------------------------
    def J(ix: float, iy: float, iz: float | None = None) -> np.ndarray:
        z = _z_center(solid, iy, ix) if iz is None else iz
        return np.array([ix, iy, z], float)

    body_cx = shoulder_center
    hips_y = crotch_y + 0.04 * H

    joints: dict[str, np.ndarray] = {
        "Hips": J(body_cx, hips_y),
        "Spine": J(body_cx, hips_y + 0.07 * H),
        "Spine1": J(body_cx, hips_y + 0.15 * H),
        "Spine2": J(body_cx, shoulder_y),
        "Neck": J(body_cx, neck_y),
        "Head": J(body_cx, neck_y + 0.03 * H),
        "HeadTop_End": J(body_cx, y_head - 0.01 * H),
    }

    for side, sh_x, hand in (("Left", left_sh_x, left_hand),
                             ("Right", right_sh_x, right_hand)):
        sh = J(sh_x, shoulder_y)
        hx, hy = hand
        # ข้อมืออยู่ลึกเข้ามาจากปลายสุดของ silhouette เล็กน้อย (ปลายสุดคือนิ้ว)
        wrist = J(sh_x + 0.92 * (hx - sh_x), shoulder_y + 0.92 * (hy - shoulder_y))
        elbow = J(0.5 * (sh[0] + wrist[0]), 0.5 * (sh[1] + wrist[1]))
        joints[f"{side}Shoulder"] = J(body_cx + 0.35 * (sh_x - body_cx), shoulder_y)
        joints[f"{side}Arm"] = sh
        joints[f"{side}ForeArm"] = elbow
        joints[f"{side}Hand"] = wrist

    for side, leg_x in (("Left", left_leg_x), ("Right", right_leg_x)):
        hip_x = body_cx + 0.55 * (leg_x - body_cx)
        ankle_y = y_foot + 0.04 * H
        knee_y = 0.5 * (crotch_y + ankle_y)
        joints[f"{side}UpLeg"] = J(hip_x, crotch_y)
        joints[f"{side}Leg"] = J(leg_x, knee_y)
        joints[f"{side}Foot"] = J(leg_x, ankle_y)
        # ปลายเท้ายื่นไปทาง +Z (ทิศที่ตัวละครหันหน้า)
        toe_z = _z_center(solid, ankle_y, leg_x)
        foot_col = np.flatnonzero(solid[int(np.clip(round(ankle_y), 0, ny - 1))].any(axis=0))
        toe_z = float(foot_col[-1]) if foot_col.size else toe_z
        joints[f"{side}ToeBase"] = np.array([leg_x, y_foot + 0.01 * H, toe_z], float)

    missing = set(BONE_PARENTS) - set(joints)
    if missing:
        raise RuntimeError(f"ประเมินข้อต่อเหล่านี้ไม่ได้: {sorted(missing)}")

    parents = dict(BONE_PARENTS)

    # --- หาง: ต่อโซ่กระดูกเข้ากับ Hips ถ้าขั้น carve หาแกนกลางหางมาให้ ------
    n_bones = int(cfg.tail_bones) if cfg is not None else 0
    if grid.tail_path is not None and n_bones > 0:
        tj, tp = tail_joints(grid.tail_path, n_bones)
        joints.update(tj)
        parents.update(tp)
        length = float(np.linalg.norm(np.diff(grid.tail_path, axis=0), axis=1).sum())
        log.info("เพิ่มกระดูกหาง %d ข้อ (+ข้อปลาย) ยาวรวม %.0f voxel", n_bones, length)
        if length < 0.15 * H:
            notes.append(
                f"หางสั้นผิดปกติ (ยาว {length / H:.0%} ของความสูงตัว) — "
                "ภาพแต่ละมุมอาจวางหางไว้คนละตำแหน่ง ทำให้เหลือเนื้อหางน้อย"
            )
    elif n_bones > 0 and cfg is not None and cfg.keep_tail:
        notes.append(
            "ไม่พบหางในภาพ จึงไม่ได้ใส่กระดูกหาง — "
            "ถ้าตัวละครมีหางจริง ลองลด --tail-min-rows หรือตรวจว่ามุมข้าง"
            "แยกหางออกจากลำตัวได้หรือไม่"
        )

    for n in notes:
        log.warning("ประเมินข้อต่อ: %s", n)
    log.info("ประเมินข้อต่อครบ %d จุด / กระดูก %d ชิ้น", len(joints), len(parents))
    return Skeleton(joints=joints, parents=parents, notes=notes)


# ---------------------------------------------------------------------------
# ส่งออกให้ Blender
# ---------------------------------------------------------------------------


def write_json(
    skeleton: Skeleton,
    verts_vox: np.ndarray,
    verts_world: np.ndarray,
    path: Path,
) -> Path:
    """เขียนข้อต่อ (หน่วยเมตร) + ลำดับชั้นกระดูก ให้สคริปต์ฝั่ง Blender อ่าน"""
    payload = {
        "joints": skeleton.to_world(verts_vox, verts_world),
        "parents": skeleton.parents,
        "notes": skeleton.notes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("เขียนข้อต่อลง %s", path)
    return path


def draw_debug(grid: HullGrid, skeleton: Skeleton, path: Path) -> None:
    """วาดโครงกระดูกทับ silhouette ด้านหน้า เพื่อตรวจด้วยตา"""
    import cv2

    solid = grid.occupancy > 0.5
    front = solid.any(axis=2)
    ny, nx = front.shape

    scale = max(1, 700 // max(ny, 1))
    img = np.zeros((ny, nx, 3), np.uint8)
    img[front] = (55, 55, 60)

    def px(p: np.ndarray) -> tuple[int, int]:
        return int(round(p[0])), int(round(ny - 1 - p[1]))  # กลับแกน Y ให้หัวอยู่บน

    for child, parent in skeleton.parents.items():
        if parent is None:
            continue
        cv2.line(img, px(skeleton.joints[parent]), px(skeleton.joints[child]),
                 (90, 220, 255), 1, cv2.LINE_AA)
    for name, p in skeleton.joints.items():
        cv2.circle(img, px(p), 2, (80, 255, 140), -1)

    img = cv2.resize(img, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img[:, :, ::-1])
    log.info("เขียนภาพตรวจโครงกระดูก: %s", path)
