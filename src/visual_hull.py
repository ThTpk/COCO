"""ขั้นที่ 2: สร้างกริด occupancy จาก silhouette ของทุกมุม (visual hull / space carving)

หลักการ: voxel หนึ่งจะ "ทึบ" ก็เมื่อมันตกอยู่ใน silhouette ของ *ทุก* มุมที่มี
ถ้ามุมใดมุมหนึ่งมองทะลุตำแหน่งนั้นไปเห็นพื้นหลัง แปลว่าตำแหน่งนั้นต้องว่าง

ระบบพิกัด (ตรงกับ glTF / three.js):
    +X = ขวาของภาพมุม front       (ความกว้างลำตัว)
    +Y = ขึ้น                      (ความสูง)
    +Z = ทิศที่ตัวละครหันหน้าไป    (ความหนาลำตัว)

ข้อจำกัดที่ต้องรู้
-----------------
หน้าตัดแนวนอนของ visual hull จาก 3 มุมตั้งฉาก คือ *สี่เหลี่ยม* ที่ได้จากการ
ตัดกันของช่วง X (จากมุมหน้า) กับช่วง Z (จากมุมข้าง) ผลคือรูปทรงจะออกเหลี่ยม
โดยธรรมชาติ ไม่ใช่ทรงกระบอกแบบแขนขาจริง

เราชดเชยด้วยสองอย่าง: เบลอ occupancy ก่อน marching cubes (`hull_blur`)
และ Taubin smoothing บน mesh ทีหลัง ซึ่งลบเหลี่ยมได้เกือบหมด
แต่ส่วนเว้าที่ไม่มีมุมไหนมองเห็น (เช่น ซอกใต้คาง ร่องกลางหลัง) จะยังตื้นกว่าจริง
นั่นเป็นข้อจำกัดเชิงหลักการของวิธีนี้ ไม่ใช่บั๊ก
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import PipelineConfig
from .preprocess import PreparedView, mask_bbox

log = logging.getLogger(__name__)


@dataclass
class HullGrid:
    """กริด occupancy พร้อมข้อมูลที่ต้องใช้แปลงกลับไปเป็นพิกัดโลก/พิกเซล"""

    occupancy: np.ndarray      # (ny, nx, nz) float32 ในช่วง 0..1
    voxel_px: float            # ขนาด voxel คิดเป็นพิกเซลบน canvas
    origin_px: tuple[float, float, float]   # (row_bottom, col_x0, col_z0) บน canvas
    canvas: int
    # แกนกลางของหาง ถ้าตรวจเจอ: จุดเรียงจากโคนไปปลาย ในพิกัด voxel (ix, iy, iz)
    tail_path: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        ny, nx, nz = self.occupancy.shape
        return ny, nx, nz

    @property
    def filled_ratio(self) -> float:
        return float((self.occupancy > 0.5).mean())

    def voxel_to_canvas(self, iy: np.ndarray, ix: np.ndarray, iz: np.ndarray):
        """แปลง index ของ voxel กลับเป็นพิกัดพิกเซลบน canvas

        คืน (row, col_x, col_z) — row ใช้กับทุกมุม, col_x ใช้กับ front/back,
        col_z ใช้กับมุมข้าง
        """
        row_bottom, col_x0, col_z0 = self.origin_px
        row = row_bottom - (iy + 0.5) * self.voxel_px
        col_x = col_x0 + (ix + 0.5) * self.voxel_px
        col_z = col_z0 + (iz + 0.5) * self.voxel_px
        return row, col_x, col_z


def _axis_extent(views: dict[str, PreparedView], axis: str) -> tuple[int, int]:
    """หาช่วงคอลัมน์บน canvas ที่ตัวละครกินพื้นที่ สำหรับแกนที่ระบุ

    รวมทุกมุมที่วัดแกนนั้นได้ แล้วเอา union เพื่อไม่ให้ตัดส่วนใดของตัวละครหลุด
    """
    lo, hi = None, None
    for v in views.values():
        if v.spec.axis_u != axis:
            continue
        _, _, x0, x1 = mask_bbox(v.mask)
        lo = x0 if lo is None else min(lo, x0)
        hi = x1 if hi is None else max(hi, x1)
    if lo is None:
        raise ValueError(f"ไม่มีมุมไหนให้ข้อมูลแกน {axis} ได้เลย")
    return lo, hi


def _row_extent(views: dict[str, PreparedView]) -> tuple[int, int]:
    """ช่วงแถวบน canvas — ทุกมุมควรตรงกันอยู่แล้วหลัง normalize"""
    los, his = [], []
    for v in views.values():
        y0, y1, _, _ = mask_bbox(v.mask)
        los.append(y0)
        his.append(y1)
    if max(los) - min(los) > 2 or max(his) - min(his) > 2:
        log.warning("ช่วงแถวของแต่ละมุมไม่ตรงกันเป๊ะ (top %s, bottom %s) "
                    "— ขั้น normalize อาจมีปัญหา", los, his)
    return min(los), max(his)


def _runs(row: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """ช่วงต่อเนื่องของ True ในแถวเดียว คืน [(start, end), ...] แบบรวมปลาย"""
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_len]


def _remove_spikes(
    fb: np.ndarray, sd: np.ndarray, cfg: PipelineConfig
) -> tuple[np.ndarray, np.ndarray]:
    """ลบส่วนยื่นที่บางกว่า `spike_open` ออกจาก silhouette ทั้งสองแกน

    ทำไมต้องลบ: เรามีแค่ภาพหน้ากับข้าง ไม่มีภาพมุมบน จึงไม่มีทางรู้ว่าหนาม
    อยู่ตรงไหนรอบตัว ส่วนยื่นบาง ๆ อันเดียวจึงถูกยืดครอบหน้าตัดทั้งวง
    กลายเป็นสันวงแหวนรอบคอรอบหัว แทนที่จะเป็นหนามแหลมอันเดียวบนสันหลัง

    ที่แย่กว่านั้นคือหนามทำให้ silhouette แตกเป็นหลายท่อนสลับไปมาระหว่างแถว
    (1 ท่อน -> 4 ท่อน -> 1 ท่อน ภายในไม่กี่แถว) ซึ่งทำให้รูปทรงกระโดด
    เป็นขั้นบันไดตามแนวตั้ง

    ยอมเสียหนามไป แลกกับผิวที่เรียบและอ่านออกว่าเป็นหัวจริง ๆ
    ใครอยากได้หนามคืนให้ตั้ง --spike-open 0 แล้วไปปั้นเพิ่มใน Blender
    """
    k = int(cfg.spike_open)
    if k <= 0:
        return _smooth_silhouette(fb, cfg), _smooth_silhouette(sd, cfg)

    # ลบหนามเฉพาะ silhouette หน้า/หลัง เท่านั้น
    #
    # สองแกนมีปัญหาคนละแบบ และแก้คนละวิธี
    #   หน้า/หลัง : หนามทำให้ silhouette แตกเป็นหลายท่อนสลับไปมาระหว่างแถว
    #               (1 -> 4 -> 1 ท่อนในไม่กี่แถว) รูปทรงจึงกระโดดเป็นขั้นบันได
    #               ไม่มีวิธีอื่นนอกจากลบทิ้ง
    #   ข้าง      : หนามยืดช่วงความลึกของแถว ซึ่ง median filter ใน
    #               `_body_depth_profile` กรองออกให้แล้ว และ `_add_crest`
    #               เอาส่วนที่กรองออกไปนั้นมาปั้นเป็นสันแคบคืนให้
    #
    # ถ้าลบหนามจาก silhouette ข้างด้วย จะไม่เหลืออะไรให้ `_add_crest` คืน
    st = np.ones((2 * k + 1, 2 * k + 1), bool)
    opened = ndimage.binary_opening(fb, structure=st)
    if opened.sum() > 0.6 * fb.sum():
        log.info("ลบหนามออกจาก silhouette หน้า/หลัง %.1f%% (กันรูปทรงเป็นขั้นบันได)",
                 100 * (1 - opened.sum() / max(fb.sum(), 1)))
        fb = opened
    else:
        log.warning("ข้าม opening ของ silhouette หน้า/หลัง — จะกินเนื้อหายเกินครึ่ง "
                    "(ลด --spike-open ลง)")

    return _smooth_silhouette(fb, cfg), _smooth_silhouette(sd, cfg)


def _smooth_silhouette(slab: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """เกลาขอบ silhouette โดยเน้นแนวตั้ง

    เส้นขอบตัวละครที่วาดด้วยมือหรือสร้างด้วย AI จะสั่นไปมาระดับ 1-2 พิกเซล
    ระหว่างแถวที่ติดกัน ตามปกติตาคนมองไม่เห็น แต่เราปั้นหน้าตัดทีละแถว
    อย่างอิสระ ความสั่นนั้นจึงกลายเป็นลอนถี่แนวนอนพาดทั่วตัวในโมเดล 3D

    เกลาแนวตั้งแรงกว่าแนวนอน เพราะเราต้องการให้ความกว้างเปลี่ยนอย่างต่อเนื่อง
    ตามความสูง โดยไม่ไปทำให้ silhouette อ้วนหรือผอมผิดรูป
    """
    s = float(cfg.silhouette_smooth)
    if s <= 0:
        return slab
    blurred = ndimage.gaussian_filter(slab.astype(np.float32), sigma=(s, s * 0.4))
    return blurred >= 0.5


def _body_depth_profile(
    sd: np.ndarray, cfg: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ไล่หาช่วงความลึกของ *ลำตัว* ในแต่ละแถว โดยไต่จากหัวลงมาหาเท้า

    ทำไมต้องไต่ แทนที่จะเลือกก้อนที่ใหญ่ที่สุด: ส่วนที่ยื่นออกมาอย่างหาง
    กวาดตัวในแนวนอน ที่บางระดับความสูงมันจึง *กว้างกว่า* ลำตัวเสียอีก
    ถ้าเลือกด้วยขนาด เราจะไปได้ตำแหน่งของหางมาแทนขา แล้ววางขาผิดที่จนหลุดจากตัว

    แถวบนสุด (หัว) แทบไม่มีทางกำกวม เราจึงเริ่มจากตรงนั้นแล้วไล่ลงมา
    โดยแต่ละแถวเลือกก้อนที่ศูนย์กลางใกล้แถวก่อนหน้าที่สุด ซึ่งเป็นการ
    ติดตามลำตัวตามความต่อเนื่องทางกายภาพ

    ขอบหน้า/ขอบหลังถูกกรองด้วย median แยกกันคนละเส้น เพราะหนามบนสันหลังเป็น
    ค่าผิดปกติแบบ *กระโดด* ของขอบหลัง ไม่ใช่สัญญาณรบกวนแบบเฉลี่ยได้
    gaussian จะเกลี่ยหนามเข้าไปในรูปทรงจนเกิดสันวงแหวนรอบคอ ส่วน median
    ตัดค่าที่โดดออกไปทิ้งทั้งก้อนแล้วคงขอบจริงไว้คมชัด ซึ่งตรงกับที่เราต้องการ

    คืน (cz, rz, valid, ขอบหลังดิบก่อนกรองหนาม) ความยาวเท่าจำนวนแถว
    """
    ny = sd.shape[0]

    z_front = np.zeros(ny, np.float32)
    z_back = np.zeros(ny, np.float32)
    valid = np.zeros(ny, bool)

    prev_c: float | None = None
    # ไล่จากแถวบนสุด (index สูง = หัว) ลงมาหาเท้า
    for y in range(ny - 1, -1, -1):
        runs = _runs(sd[y], 1)
        if not runs:
            continue

        if prev_c is None or not cfg.side_main_component_only:
            if cfg.side_main_component_only:
                z0, z1 = max(runs, key=lambda r: r[1] - r[0])
            else:
                z0 = min(r[0] for r in runs)
                z1 = max(r[1] for r in runs)
        else:
            # เลือกก้อนที่ "ครอบ" ศูนย์กลางเดิมก่อน ถ้าไม่มีค่อยเอาก้อนที่ใกล้สุด
            covering = [r for r in runs if r[0] <= prev_c <= r[1]]
            pool = covering or runs
            z0, z1 = min(pool, key=lambda r: abs(0.5 * (r[0] + r[1]) - prev_c))

        z_back[y] = z0          # ขอบด้านหลัง (ฝั่งที่มีหนามสันหลัง)
        z_front[y] = z1         # ขอบด้านหน้า (ฝั่งอก/จมูก)
        valid[y] = True
        prev_c = 0.5 * (z0 + z1)

    # เก็บขอบหลังดิบไว้ก่อนกรอง ส่วนต่างระหว่างดิบกับที่กรองแล้วคือ *หนาม*
    # ซึ่งเราจะเอาไปปั้นเป็นสันแคบที่แนวกลางหลังทีหลัง แทนที่จะทิ้งไปเฉย ๆ
    z_back_raw = z_back.copy()

    w = int(cfg.depth_median)
    if w >= 3 and valid.any():
        idx = np.flatnonzero(valid)
        for arr in (z_back, z_front):
            # กรองเฉพาะแถวที่มีข้อมูล เพื่อไม่ให้ศูนย์จากแถวว่างดึงค่าเพี้ยน
            arr[idx] = ndimage.median_filter(arr[idx], size=w | 1, mode="nearest")

    cz = 0.5 * (z_back + z_front)
    rz = np.maximum(0.5 * (z_front - z_back), 0.5)
    return cz, rz, valid, z_back_raw


def _add_crest(
    occ: np.ndarray,
    z_back_raw: np.ndarray,
    cz: np.ndarray,
    rz: np.ndarray,
    valid: np.ndarray,
    body_cx: float,
    rx_head: float,
    cfg: PipelineConfig,
) -> int:
    """คืนหนามสันหลังเป็นสันแคบที่แนวกลางหลัง

    หนามคือส่วนยื่นแคบ ๆ บนแนวกลางหลัง ถ้าปล่อยให้มันไปยืดช่วง Z ของทั้งแถว
    ความลึกนั้นจะถูกครอบไปทั่วหน้าตัด กลายเป็นสันวงแหวนรอบคอรอบตัว แทนที่จะ
    เป็นหนามอันเดียวบนสันหลัง เพราะเราไม่มีภาพมุมบนมาบอกว่ามันอยู่ตรงไหนรอบตัว

    median filter ในขั้นก่อนหน้าตัดหนามออกจากโปรไฟล์ความลึกของลำตัวไปแล้ว
    ตรงนี้เอาส่วนที่ถูกตัดกลับมาปั้นใหม่ให้แคบ ซึ่งเป็นรูปแบบเดียวกับที่ใช้กับ
    หาง (ยื่นหลัง) และปาก (ยื่นหน้า) — ยื่นแคบ ๆ ก็ปั้นเป็นก้อนแคบ ๆ
    """
    ny, nx, nz = occ.shape
    xs = np.arange(nx, dtype=np.float32)
    zs = np.arange(nz, dtype=np.float32)
    rx = max(rx_head * cfg.crest_width, 1.0)
    rows = 0

    for y in np.flatnonzero(valid):
        z_core = float(cz[y] - rz[y])          # ขอบหลังของลำตัวหลังกรองหนามออก
        z_tip = float(z_back_raw[y])           # ปลายหนามจริง
        if z_core - z_tip < cfg.crest_min:
            continue
        c = 0.5 * (z_core + z_tip)
        r = 0.5 * (z_core - z_tip)
        dx = ((xs - body_cx) / rx) ** 2
        dz = ((zs - c) / max(r, 0.5)) ** 2
        occ[y] |= (dx[:, None] + dz[None, :]) <= 1.0
        rows += 1

    if rows:
        log.info("คืนหนามสันหลังเป็นสันแคบ %d แถว กว้าง %.0f voxel", rows, 2 * rx)
    return rows


def _body_center_x(fb: np.ndarray, min_run: int) -> float:
    """แกนกลางลำตัวในแกน X หาจากช่วงล่างซึ่งมีแต่ขา

    ช่วงล่างไม่โดนแขนที่กางออกหรือหางที่ยื่นไปข้างหลังมาดึง จึงเป็นจุดอ้างอิง
    ที่นิ่งที่สุด และเป็นตำแหน่งที่ใช้วางหางได้ด้วย เพราะหางอยู่แนวกลางลำตัว
    """
    ny, nx = fb.shape
    lower = fb[: max(1, ny // 5)]
    return float(np.where(lower)[1].mean()) if lower.any() else nx / 2.0


def _body_width_profile(fb: np.ndarray, min_run: int) -> np.ndarray:
    """ครึ่งความกว้างของ *ลำตัว* ในแต่ละแถว (ไม่นับแขนที่แยกออกไป)

    ใช้เป็นตัวหารเวลาคิดความลึกของแต่ละชิ้น ต้องเป็นค่าที่เปลี่ยนอย่างราบรื่น
    ตามความสูง ถ้าใช้ค่าสูงสุดของแถวตรง ๆ มันจะกระโดดทุกครั้งที่แขนแยกออก
    จากลำตัว แล้วผิวจะเป็นขั้นบันได
    """
    ny, nx = fb.shape
    cx = _body_center_x(fb, min_run)

    out = np.zeros(ny, np.float32)
    for y in range(ny):
        runs = _runs(fb[y], min_run)
        if not runs:
            continue
        central = next((r for r in runs if r[0] <= cx <= r[1]), None)
        if central is None:
            # ไม่มีท่อนคร่อมแกนกลาง (เช่น แถวที่เห็นแต่มือสองข้าง) เอาท่อนที่หนาสุด
            central = max(runs, key=lambda r: r[1] - r[0])
        out[y] = max(0.5 * (central[1] - central[0]), 0.5)

    # เติมแถวที่ว่างด้วยค่าเพื่อนบ้าน เพื่อให้ profile ต่อเนื่อง
    nz_idx = np.flatnonzero(out > 0)
    if nz_idx.size:
        out = np.interp(np.arange(ny), nz_idx, out[nz_idx])
    else:
        out[:] = 1.0
    return out


def _limb_runs(
    fb: np.ndarray, min_run: int, cfg: PipelineConfig
) -> tuple[list[list[tuple[int, int]] | None], float]:
    """สำหรับแต่ละแถว: ท่อนของลำตัว/ขา โดย *ตัดหางที่เชื่อมติดมาออกแล้ว*

    ไล่ติดตามขาจาก *เท้าขึ้นมา* ทีละแถว เท้าเป็นจุดเริ่มที่ไม่กำกวม เพราะ
    แถวล่างสุดมีแต่เท้าสองข้าง จากนั้นแต่ละแถวจับคู่ท่อนกับขาที่ติดตามอยู่
    ด้วยความต่อเนื่องของตำแหน่ง

    ต้องใช้การติดตาม เพราะดูแค่แถวเดียวแยกหางออกจากขาไม่ได้เลย: มุมหน้าไม่มี
    ข้อมูลความลึก และบ่อยครั้งหางกับขา *เชื่อมเป็นท่อนเดียวกัน* ใน silhouette
    การกรองทั้งท่อนจึงทำอะไรไม่ได้ ต้องตัดท่อนให้เหลือเฉพาะความกว้างของขา

    ความกว้างของขาเอามาจากแถวล่าง ๆ ที่ขายังไม่ชนหาง แล้วพาค่านั้นไต่ขึ้นไป
    แถวไหนที่ท่อนอ้วนผิดปกติเทียบกับที่จำไว้ ส่วนเกินคือหาง

    คืน (ท่อนลำตัว/ขาต่อแถว, แกนกลางลำตัว)
    """
    ny, nx = fb.shape
    out: list[list[tuple[int, int]] | None] = [None] * ny
    max_jump = max(3.0, 0.05 * nx)
    clip = float(cfg.leg_clip_factor)

    rows = [y for y in range(ny) if fb[y].any()]
    if not rows:
        return out, nx / 2.0
    y_low = rows[0]

    # เริ่มจากแถวแรกที่เห็น *เท้าคู่* ไม่ใช่แถวแรกที่มีสองท่อนเฉย ๆ
    #
    # เท้าสองข้างมักแตะพื้นคนละระดับ แถวล่างสุดจึงมีเท้าเดียว และที่แย่กว่านั้น
    # ปลายหางที่แตะพื้นก็เป็นท่อนหนึ่งเหมือนกัน ถ้าหยิบ "สองท่อนที่กว้างที่สุด"
    # มาดื้อ ๆ อาจได้ขาข้างหนึ่งคู่กับเศษหาง แล้วแกนกลางลำตัวจะเพี้ยนไปไกล
    # จนตัวติดตามทั้งสองไปเกาะขาข้างเดียวกัน แล้วขาอีกข้างหายทั้งท่อน
    #
    # เท้าคู่สังเกตได้จากการที่สองท่อนกว้างใกล้เคียงกัน
    candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for y in range(y_low, min(ny, y_low + int(0.30 * ny))):
        rs = _runs(fb[y], min_run)
        if len(rs) < 2:
            continue
        two = sorted(rs, key=lambda r: r[1] - r[0], reverse=True)[:2]
        w = [q[1] - q[0] for q in two]
        if min(w) < 0.5 * max(w):
            continue        # กว้างต่างกันมาก น่าจะเป็นขาคู่กับเศษหาง
        two.sort(key=lambda r: r[0])
        candidates.append((y, two))

    if candidates:
        # แกนกลางลำตัว = ค่ากลางของจุดกึ่งกลางเท้าคู่ จากหลายแถว
        # ใช้ median เพราะแถวล่างสุดบางแถวหยิบ "นิ้วเท้าสองนิ้วของเท้าเดียวกัน"
        # มาเป็นคู่ได้ ค่ากลางจะกรองแถวแบบนั้นทิ้งไปเอง
        body_cx = float(np.median(
            [0.25 * (t[0][0] + t[0][1] + t[1][0] + t[1][1]) for _, t in candidates]))

        # แถวเริ่มต้นต้องเป็นแถวที่สองท่อนอยู่ *คนละฝั่ง* ของแกนกลางจริง ๆ
        # ไม่งั้นตัวติดตามทั้งสองจะออกตัวบนขาข้างเดียวกัน แล้วขาอีกข้างหายทั้งท่อน
        straddle = [(y, t) for y, t in candidates
                    if 0.5 * (t[0][0] + t[0][1]) < body_cx < 0.5 * (t[1][0] + t[1][1])]
        y_start, start = straddle[0] if straddle else candidates[0]
        if not straddle:
            log.warning("หาแถวที่เห็นเท้าคนละข้างไม่เจอ — การแยกขาออกจากหาง"
                        "อาจไม่แม่น")
    else:
        y_start = y_low
        start = sorted(_runs(fb[y_low], min_run),
                       key=lambda r: r[1] - r[0], reverse=True)[:2]
        if not start:
            return out, nx / 2.0
        body_cx = float(np.mean([0.5 * (a + b) for a, b in start]))
        log.warning("หาเท้าคู่ไม่เจอในช่วงล่างของตัว — การแยกขาออกจากหาง"
                    "อาจไม่แม่น")

    def sweep(ys, tracks: list[list[float]]) -> None:
        for y in ys:
            runs = _runs(fb[y], min_run)
            if not runs:
                continue

            chosen: list[tuple[int, int]] = []
            used: list[tuple[int, int]] = []
            new_tracks: list[list[float]] = []

            for c, w in tracks:
                inside = [r for r in runs if r[0] <= c <= r[1]]
                r = inside[0] if inside else min(
                    runs, key=lambda q: abs(0.5 * (q[0] + q[1]) - c))
                if not inside and abs(0.5 * (r[0] + r[1]) - c) > max_jump:
                    continue    # ขาข้างนี้หลุดการติดตาม

                chosen.append(r)
                used.append(r)
                new_tracks.append([0.5 * (r[0] + r[1]), float(r[1] - r[0])])

            for r in runs:                    # ท่อนที่คร่อมแกนกลางต้องมีเสมอ
                if r[0] <= body_cx <= r[1] and r not in used:
                    chosen.append(r)
                    used.append(r)

            # ยังไม่ครบสองขา: รับท่อนที่หนาพอ ๆ กันเข้ามาเป็นขาอีกข้าง
            if len(chosen) < 2:
                widest = max((q[1] - q[0] for q in chosen), default=0)
                extra = [q for q in runs
                         if q not in used and q[1] - q[0] >= 0.5 * widest]
                if extra:
                    r = max(extra, key=lambda q: q[1] - q[0])
                    chosen.append(r)
                    new_tracks.append([0.5 * (r[0] + r[1]), float(r[1] - r[0])])

            if not chosen:
                chosen.append(max(runs, key=lambda q: q[1] - q[0]))

            # --- ตัดหางที่เชื่อมติดขา โดยใช้ขาอีกข้างเป็นไม้บรรทัด -----------
            #
            # ขาซ้ายกับขาขวาหนาพอ ๆ กันที่ความสูงเดียวกันเสมอ ถ้าข้างหนึ่งอ้วน
            # ผิดปกติ ส่วนเกินคือหางที่เชื่อมติดมา
            #
            # ใช้ความสมมาตรในแถวเดียวกัน ไม่ใช่ความกว้างที่จำมาจากแถวล่าง
            # เพราะต้นขากว้างกว่าข้อเท้ามาก การจำข้ามแถวจะกลายเป็นลูปป้อนกลับ
            # ที่ตัดแคบลงเรื่อย ๆ จนกินขาหายไปทั้งข้าง
            if len(chosen) == 2 and not any(a <= body_cx <= b for a, b in chosen):
                wide, narrow = sorted(chosen, key=lambda q: q[1] - q[0], reverse=True)
                ww, nw = wide[1] - wide[0], narrow[1] - narrow[0]
                if nw > 0 and ww > clip * nw:
                    keep = nw * 1.15
                    # หางยื่นออกด้านนอก ตัดจากฝั่งที่ไกลแกนกลางลำตัว
                    if 0.5 * (wide[0] + wide[1]) > body_cx:
                        trimmed = (wide[0], int(round(wide[0] + keep)))
                    else:
                        trimmed = (int(round(wide[1] - keep)), wide[1])
                    chosen = [trimmed if q == wide else q for q in chosen]
                    new_tracks = [[0.5 * (a + b), float(b - a)] for a, b in chosen]

            out[y] = chosen
            tracks = new_tracks or [[0.5 * (a + b), float(b - a)] for a, b in chosen]

    t0 = [[0.5 * (a + b), float(b - a)] for a, b in start]
    sweep(range(y_start, ny), [list(x) for x in t0])
    sweep(range(y_start - 1, y_low - 1, -1), [list(x) for x in t0])
    return out, body_cx


def _limit_aspect(
    rz: np.ndarray,
    cz: np.ndarray,
    rx_body: np.ndarray,
    valid: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """จำกัดไม่ให้หน้าตัดของ *ลำตัว* ลึกเกินกว่าความกว้างของตัวเองมากนัก

    ปัญหาที่แก้: ส่วนที่ยื่นไปข้างหน้าอย่างปากสัตว์ ห้อยลงมาคลุมถึงระดับคอ
    mask ด้านข้างที่แถวนั้นจึงกินตั้งแต่ท้ายทอยถึงปลายปาก พอเอาความลึกนั้นไป
    ครอบความกว้างของบ่าทั้งแถว จะได้แผ่นแบนคาดรอบตัวเหมือนปีกหมวก

    ตัวหัวเองต้องได้รับการยกเว้น เพราะหัวที่มีปากยื่น *ควร* ลึกกว่ากว้างจริง ๆ
    เราจึงไล่ระดับการจำกัดตามความกว้างของแถว แถวที่แคบเท่าหัวไม่ถูกแตะเลย

    คืน (rz ใหม่, cz ใหม่, ความกว้างหัว, ขอบหน้าเดิมก่อนถูกหั่น)
    """
    z_front_orig = (cz + rz).astype(np.float32)
    idx = np.flatnonzero(valid & (rx_body > 0))

    if cfg.max_aspect <= 0 or idx.size < 8:
        fallback = float(rx_body[idx].min()) if idx.size else 1.0
        return rz, cz, max(fallback, 1.0), z_front_orig

    # ความกว้างของ "หัว" = ค่าต่ำสุดในช่วงบนของตัว ซึ่งก็คือช่วงคอ/หัว
    top = idx[idx >= idx.min() + int(0.72 * (idx.max() - idx.min()))]
    rx_head = float(np.percentile(rx_body[top], 25)) if top.size \
        else float(rx_body[idx].min())
    rx_head = max(rx_head, 1.0)

    span = max(cfg.aspect_blend - 1.0, 0.1) * rx_head
    t = np.clip((rx_body - rx_head) / span, 0.0, 1.0)   # 0 = หัว, 1 = ลำตัว

    # จำกัดเฉพาะ *เหนือ* แถวที่ลำตัวกว้างที่สุดขึ้นไป ซึ่งก็คือช่วงคอกับหัว
    #
    # ถ้าไม่จำกัดขอบเขต กฎนี้จะไปโดนขาด้วย เพราะขาก็ลึกกว่ากว้างเหมือนกัน
    # แล้วขั้นคืนส่วนที่ยื่นหน้าจะไปงอก "ปาก" ออกมาที่หน้าแข้ง
    y_widest = int(idx[np.argmax(rx_body[idx])])
    t[:y_widest] = 0.0

    capped = np.minimum(rz, cfg.max_aspect * np.maximum(rx_body, 0.5))
    new_rz = (1.0 - t) * rz + t * capped

    # หดเข้าจากด้านหน้า โดยตรึงขอบหลังไว้ที่เดิม
    new_cz = (cz - rz) + new_rz

    trimmed = int((rz - new_rz > 0.5).sum())
    if trimmed:
        log.info("จำกัดความลึกลำตัวตามความกว้าง %d แถว", trimmed)
    return (new_rz.astype(np.float32), new_cz.astype(np.float32),
            rx_head, z_front_orig)


def _add_snout(
    occ: np.ndarray,
    z_front_orig: np.ndarray,
    cz: np.ndarray,
    rz: np.ndarray,
    valid: np.ndarray,
    body_cx: float,
    rx_head: float,
    cfg: PipelineConfig,
) -> int:
    """คืนส่วนที่ยื่นไปข้างหน้า (ปาก/ขากรรไกร) กลับมาเป็นก้อนแคบของตัวเอง

    `_limit_aspect` หั่นความลึกของลำตัวลงเพื่อลบแผ่นแบนรอบคอ แต่ความลึกที่
    หั่นออกไปนั้นคือ *ปาก* ซึ่งเป็นของจริง ถ้าไม่คืนมา ตัวละครจะหน้าสั้นกุด

    ความต่างอยู่ที่ความกว้าง: คอกว้างเท่าบ่า แต่ปากแคบเท่าหัว เราจึงคืนมันมา
    เป็นก้อนแยกที่กว้างเท่าหัวเท่านั้น
    """
    ny, nx, nz = occ.shape
    xs = np.arange(nx, dtype=np.float32)
    zs = np.arange(nz, dtype=np.float32)
    rx = max(rx_head * cfg.snout_width, 1.0)
    rows = 0

    for y in np.flatnonzero(valid):
        z_cap = float(cz[y] + rz[y])
        z_out = float(z_front_orig[y])
        if z_out - z_cap < 1.5:
            continue
        c = 0.5 * (z_cap + z_out)
        r = 0.5 * (z_out - z_cap)
        dx = ((xs - body_cx) / rx) ** 2
        dz = ((zs - c) / max(r, 0.5)) ** 2
        occ[y] |= (dx[:, None] + dz[None, :]) <= 1.0
        rows += 1

    if rows:
        log.info("คืนส่วนที่ยื่นหน้า (ปาก) เป็นก้อนแคบ %d แถว กว้าง %.0f voxel",
                 rows, 2 * rx)
    return rows


def _trace_to_voxels(trace, grid: HullGrid, ny: int, nx: int) -> np.ndarray:
    """แปลงเส้นทางหางจากพิกัดพิกเซลบน canvas เป็นพิกัด voxel

    คืนอาเรย์ (N, 3) ของ (ix, iy, ครึ่งความกว้างเป็น voxel) เรียงจากโคนไปปลาย
    """
    row_bottom, col_x0, _ = grid.origin_px
    vp = grid.voxel_px

    iy = (row_bottom - trace.rows) / vp - 0.5
    ix = (trace.cx - col_x0) / vp - 0.5
    hw = trace.half_w / vp

    ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    iy, ix, hw = iy[ok], ix[ok], hw[ok]

    # เรียงจากโคน (iy มาก) ไปปลาย (iy น้อย) และเก็บแถวละจุดเดียว
    order = np.argsort(-iy)
    iy, ix, hw = iy[order], ix[order], hw[order]
    keep = np.concatenate(([True], np.diff(np.round(iy)) != 0))
    return np.stack([ix[keep], iy[keep], hw[keep]], axis=1)


def _add_tail_tube(
    occ: np.ndarray,
    tail_vox: np.ndarray,
    cz: np.ndarray,
    rz: np.ndarray,
    valid: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    """ปั้นหางเป็นท่อกลมตามเส้นทางที่ลากจากภาพด้านหลัง

    ภาพด้านหลังบอกตำแหน่งแกน X กับความกว้างมาครบ แต่บอกความลึกไม่ได้เลย
    เราจึงวางหางไว้ที่ *ผิวด้านหลังของลำตัว* ในแต่ละแถว ซึ่งเป็นที่ที่หางพาด
    อยู่จริงเมื่อมองจากด้านหลัง และสมมติว่าหน้าตัดกลม เพราะหางสัตว์หนาพอ ๆ กัน
    ทุกทิศ การเดาว่ากลมจึงสมเหตุผลกว่าการเดาความลึกจากมุมอื่นที่วางหางคนละที่
    """
    ny, nx, nz = occ.shape
    xs = np.arange(nx, dtype=np.float32)
    zs = np.arange(nz, dtype=np.float32)
    before = int(occ.sum())
    path = []

    for ix, iy, hw in tail_vox:
        y = int(round(iy))
        if not (0 <= y < ny) or not valid[y]:
            continue
        r = max(float(hw) * cfg.tail_thickness, 1.0)
        # +Z คือทิศหน้า ผิวหลังจึงอยู่ที่ cz - rz และการดันหางออกไปด้านหลัง
        # คือการ *ลบ* ค่าลงไปอีก ไม่ใช่บวก
        czt = float(cz[y] - rz[y]) - r * cfg.tail_z_bias
        dx = ((xs - float(ix)) / r) ** 2
        dz = ((zs - czt) / r) ** 2
        occ[y] |= (dx[:, None] + dz[None, :]) <= 1.0
        path.append((float(ix), float(y), czt))

    if not path:
        return None
    log.info("ปั้นหางตามภาพด้านหลัง %d แถว (+%.1f%% ของปริมาตรตัว)",
             len(path), 100 * (occ.sum() - before) / max(before, 1))
    return np.array(path, np.float32)


def _fill_elliptical(
    fb: np.ndarray, sd: np.ndarray, cfg: PipelineConfig,
    tail_rows: np.ndarray | None = None,
) -> np.ndarray:
    """สร้างหน้าตัดเป็นวงรีแทนสี่เหลี่ยม

    ทำทีละแถวความสูง:
      * แยก mask ด้านหน้าเป็นชิ้น ๆ ตามแกน X (ลำตัว แขนซ้าย แขนขวา ขาสองข้าง)
      * อ่านช่วงความลึกจาก mask ด้านข้าง โดยเลือกเฉพาะก้อนหลัก (ตัดหางทิ้ง)
      * ให้แต่ละชิ้นมีความลึกตามสัดส่วนความกว้างของตัวเอง ยกกำลัง depth_falloff
        แขนที่แคบจึงได้หน้าตัดเกือบกลม ไม่ใช่แผ่นแบนกว้างเท่าลำตัว

    นี่ไม่ใช่ visual hull ตามนิยามอีกต่อไป (silhouette ด้านข้างจะไม่เป๊ะ 100%)
    แต่ให้รูปทรงที่เหมือนสิ่งมีชีวิตมากกว่ามาก ซึ่งเป็นสิ่งที่ต้องการจริง ๆ
    """
    ny, nx = fb.shape
    nz = sd.shape[1]
    occ = np.zeros((ny, nx, nz), bool)

    xs = np.arange(nx, dtype=np.float32)
    zs = np.arange(nz, dtype=np.float32)
    p = float(cfg.cross_section_power)
    min_run = max(1, nx // 80)
    dropped_rows = 0

    cz_profile, rz_profile, valid, z_back_raw = _body_depth_profile(sd, cfg)
    rx_body = _body_width_profile(fb, min_run)

    # เกลาโปรไฟล์ทั้งสามตามแนวตั้งก่อนใช้
    #
    # ถ้าไม่เกลา ผิวจะออกมาเป็นวงแหวนขั้นบันได เพราะแต่ละแถวคำนวณอิสระกัน
    # พอ silhouette เปลี่ยนจำนวนท่อน (แขนแยกออกจากลำตัว หนามโผล่) ค่าตัวหาร
    # กระโดดทันที ความลึกของทุกชิ้นในแถวนั้นก็กระโดดตาม เกิดเป็นรอยต่อคม ๆ
    sm = float(cfg.profile_smooth)
    if sm > 0:
        cz_profile = ndimage.gaussian_filter1d(cz_profile, sm, mode="nearest")
        rz_profile = ndimage.gaussian_filter1d(rz_profile, sm, mode="nearest")
        rx_body = ndimage.gaussian_filter1d(rx_body, sm, mode="nearest")

    rz_profile, cz_profile, rx_head, z_front_orig = _limit_aspect(
        rz_profile, cz_profile, rx_body, valid, cfg)

    # หาหางก่อนปั้นลำตัว เพราะลำตัวต้องรู้ว่าท่อนไหนคือหางเพื่อจะได้ข้ามไป
    limb_runs, body_cx = _limb_runs(fb, min_run, cfg)
    pruned_count = 0
    if tail_rows is None:
        tail_rows = np.zeros(ny, bool)

    for y in range(ny):
        x_runs = _runs(fb[y], min_run)
        if not x_runs or not valid[y]:
            continue
        if len(_runs(sd[y], 1)) > 1:
            dropped_rows += 1

        # ในช่วงที่มีหาง ท่อนที่ไม่ใช่ลำตัว/ขา คือเนื้อหาง ปล่อยให้ขั้นปั้นหาง
        # จัดการที่ความลึกที่ถูกต้อง ไม่งั้นจะได้หางสองอันคนละความลึก
        #
        # ปลอดภัยเพราะช่วงที่มีหางอยู่ต่ำกว่าระดับมือเสมอ ท่อนนอกลำตัว/ขา
        # ในช่วงนั้นจึงไม่ใช่มือแน่นอน และยังกันอีกชั้นด้วยการไม่ตัดท่อนที่
        # หนาใกล้เคียงขา เผื่อการติดตามขาหลุดไปบางแถว
        if tail_rows[y] and limb_runs[y]:
            trimmed = sum((b - a) for a, b in x_runs) -                       sum((b - a) for a, b in limb_runs[y])
            if trimmed > 0:
                pruned_count += 1
            x_runs = limb_runs[y]

        cz = float(cz_profile[y])
        rz_full = float(rz_profile[y])
        rx_max = max(float(rx_body[y]), 0.5)

        plane = occ[y]
        for a, b in x_runs:
            cx = 0.5 * (a + b)
            rx = max(0.5 * (b - a), 0.5)
            # ตรึงอัตราส่วนไว้ ไม่ให้ชิ้นที่กว้างกว่าลำตัว (เช่นช่วงไหล่กางแขน)
            # ได้ความลึกเกินความหนาจริงของตัว และไม่ให้ชิ้นที่บางมากแบนจนหาย
            ratio = float(np.clip(rx / rx_max, 0.15, 1.0))
            rz = rz_full * ratio ** cfg.depth_falloff
            dx = np.abs((xs - cx) / rx) ** p
            dz = np.abs((zs - cz) / max(rz, 0.5)) ** p
            plane |= (dx[:, None] + dz[None, :]) <= 1.0

    # --- ปาก: คืนส่วนที่ยื่นหน้าซึ่งถูกหั่นออกไป เป็นก้อนแคบของตัวเอง --------
    if cfg.max_aspect > 0 and cfg.snout_width > 0:
        _add_snout(occ, z_front_orig, cz_profile, rz_profile, valid,
                   body_cx, rx_head, cfg)

    # --- หนามสันหลัง: คืนส่วนที่ median filter ตัดไป เป็นสันแคบแนวกลาง -------
    if cfg.crest_width > 0:
        _add_crest(occ, z_back_raw, cz_profile, rz_profile, valid,
                   body_cx, rx_head, cfg)

    if pruned_count:
        log.info("ตัดเนื้อหางออกจากขั้นปั้นลำตัว %d แถว (ไปปั้นใหม่ที่ความลึกจริงแทน)", pruned_count)
    if dropped_rows:
        log.info("ตัดส่วนที่ยื่นออกจากลำตัวใน mask ด้านข้าง %d แถว "
                 "(หาง/เป้/ปีก จะไม่ไปพองความหนาลำตัว)", dropped_rows)
    return occ, cz_profile, rz_profile, valid


def carve(views: dict[str, PreparedView], cfg: PipelineConfig) -> HullGrid:
    """ตัดกริด voxel ด้วย silhouette ของทุกมุม

    voxel เป็นลูกบาศก์จริง (ขนาดเท่ากันทั้งสามแกน) กริดจึงไม่ใช่ทรงลูกบาศก์
    แต่พอดีกับสัดส่วนตัวละคร — ได้ความละเอียดต่อหน่วยความจำดีกว่ากริด N^3 มาก
    """
    S = cfg.canvas
    row_top, row_bottom = _row_extent(views)
    x0, x1 = _axis_extent(views, "x")
    z0, z1 = _axis_extent(views, "z")

    pad = 2.0  # เผื่อขอบ 2 พิกเซล กัน mesh ไปติดผิวกริด
    ext_y = (row_bottom - row_top) + 1 + 2 * pad
    ext_x = (x1 - x0) + 1 + 2 * pad
    ext_z = (z1 - z0) + 1 + 2 * pad

    # voxel เป็นลูกบาศก์: ใช้ความสูงเป็นตัวกำหนดความละเอียด
    voxel_px = ext_y / cfg.voxel_res
    ny = cfg.voxel_res
    nx = max(4, int(np.ceil(ext_x / voxel_px)))
    nz = max(4, int(np.ceil(ext_z / voxel_px)))

    origin = (row_bottom + pad, x0 - pad, z0 - pad)
    log.info("กริด %d x %d x %d voxel (%.2f px/voxel, %.1f M voxel)",
             ny, nx, nz, voxel_px, ny * nx * nz / 1e6)

    grid = HullGrid(np.zeros((ny, nx, nz), np.float32), voxel_px, origin, S)

    iy = np.arange(ny)
    ix = np.arange(nx)
    iz = np.arange(nz)
    rows, cols_x, cols_z = grid.voxel_to_canvas(iy, ix, iz)

    # ปัดเป็น index พิกเซลและตรึงไว้ในขอบ canvas
    ri = np.clip(np.rint(rows).astype(np.int32), 0, S - 1)
    xi = np.clip(np.rint(cols_x).astype(np.int32), 0, S - 1)
    zi = np.clip(np.rint(cols_z).astype(np.int32), 0, S - 1)

    # รวม mask ของแต่ละแกนเป็นแผ่นเดียว: fb บอกช่วง X ต่อแถว, sd บอกช่วง Z ต่อแถว
    fb = np.ones((ny, nx), bool)
    sd = np.ones((ny, nz), bool)
    used: list[str] = []

    for name, v in views.items():
        if v.spec.axis_u == "x":
            fb &= v.mask[ri[:, None], xi[None, :]]
        else:
            sd &= v.mask[ri[:, None], zi[None, :]]
        used.append(name)

    fb, sd = _remove_spikes(fb, sd, cfg)

    if cfg.carve_mode == "ellipse":
        # ลากหางจากภาพด้านหลังก่อน ลำตัวต้องรู้ว่าแถวไหนมีเนื้อหางปนอยู่
        trace = None
        if cfg.keep_tail and "back" in views:
            from . import tailtrace
            trace = tailtrace.trace(views["back"], cfg)

        tail_rows = np.zeros(ny, bool)
        tail_vox = None
        if trace is not None:
            tail_vox = _trace_to_voxels(trace, grid, ny, nx)
            tail_rows[tail_vox[:, 1].astype(int)] = True

        occ, czp, rzp, valid = _fill_elliptical(fb, sd, cfg, tail_rows)

        if tail_vox is not None:
            grid.tail_path = _add_tail_tube(occ, tail_vox, czp, rzp, valid, cfg)
    else:
        # โหมด hull ตามนิยาม ใช้ช่วง Z ทั้งหมดอยู่แล้ว หางจึงติดมาเอง
        occ = fb[:, :, None] & sd[:, None, :]

    if not occ.any():
        raise RuntimeError(
            "ไม่มี voxel ทึบเหลือเลยหลัง carve\n"
            "สาเหตุที่พบบ่อย: มุมข้างหันหน้าผิดทาง (ลอง --flip-side) "
            "หรือ mask ของมุมใดมุมหนึ่งผิด (ดู output/debug/01_mask_*.png)"
        )

    log.info("carve ด้วย %d มุม (%s) -> voxel ทึบ %.2f%%",
             len(used), ", ".join(used), 100 * occ.mean())

    field = occ.astype(np.float32)
    if cfg.hull_blur > 0:
        # เบลอเพื่อให้ isosurface โค้งมน ลบเหลี่ยมที่เกิดจากการตัดกันของสี่เหลี่ยม
        field = ndimage.gaussian_filter(field, sigma=cfg.hull_blur, mode="constant")

    # ปิดผิวกริดให้เป็น 0 เพื่อให้ marching cubes ได้ mesh ปิดสนิททุกด้าน
    field[0, :, :] = field[-1, :, :] = 0
    field[:, 0, :] = field[:, -1, :] = 0
    field[:, :, 0] = field[:, :, -1] = 0

    grid.occupancy = field
    log.info("occupancy หลังเบลอ: ทึบ %.2f%%", 100 * grid.filled_ratio)
    return grid


def carve_report(grid: HullGrid, views: dict[str, PreparedView]) -> list[str]:
    """ตรวจคุณภาพของ hull แบบหยาบ ๆ เพื่อเตือนก่อนไปขั้นถัดไป"""
    notes: list[str] = []
    solid = grid.occupancy > 0.5

    # จำนวนก้อนที่แยกจากกัน — ควรมีก้อนเดียว
    labels, n = ndimage.label(solid)
    if n > 1:
        sizes = np.bincount(labels.ravel())[1:]
        frac = sizes.max() / sizes.sum()
        # เศษ voxel หลุดไม่กี่ตัวเป็นเรื่องปกติของ marching cubes และถูกตัดทิ้ง
        # ในขั้น clean อยู่แล้ว เตือนเฉพาะตอนที่มีชิ้นส่วนจริงหลุดออกไป
        if frac < 0.99:
            notes.append(
                f"hull แตกเป็น {n} ก้อน (ก้อนใหญ่สุดคิดเป็น {frac:.1%}) — "
                "มักเกิดจากแขนขาขาดหายใน mask บางมุม"
            )

    if grid.filled_ratio < 0.02:
        notes.append(f"hull บางผิดปกติ (ทึบแค่ {grid.filled_ratio:.2%} ของกริด) — "
                     "ตรวจว่ามุมข้างหันถูกทางหรือไม่")

    # เทียบความหนา/ความกว้างสูงสุด ถ้าหนากว่ากว้างมากคือมุมสลับกัน
    ny, nx, nz = grid.shape
    width = solid.any(axis=2).sum(axis=1).max()
    depth = solid.any(axis=1).sum(axis=1).max()
    if depth > width * 1.6:
        notes.append(
            f"ความหนาลำตัว ({depth} voxel) มากกว่าความกว้าง ({width} voxel) อย่างผิดสังเกต "
            "— มุม front กับ side อาจสลับไฟล์กัน"
        )
    return notes
