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
from dataclasses import dataclass, replace

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
    # รัศมีหางที่แต่ละจุดบน tail_path (หน่วย voxel) — ใช้ปั้นหางเป็น mesh จริง
    # ตอนที่ใช้โมเดลฐาน ซึ่งไม่ได้ผ่าน marching cubes จาก occupancy
    tail_radius: np.ndarray | None = None

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
    fb: np.ndarray, sd: np.ndarray, cfg: PipelineConfig,
    keep: np.ndarray | None = None, front: np.ndarray | None = None,
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
    # ในเขตมือ/เท้า ใช้ silhouette ของ *มุมหน้ามุมเดียว* แทนที่จะใช้ผลตัดกัน
    #
    # นิ้วมือกับนิ้วเท้าหันเข้าหากล้องมุมหน้า ร่องระหว่างนิ้วจึงเห็นชัดที่สุด
    # ในมุมนั้น ส่วนมุมหลังเห็นหลังมือ/ส้นเท้าซึ่งร่องตื้นกว่าและเหลื่อมกัน
    # ไม่กี่ voxel พอ AND สองมุม ร่องก็ถูกถมหมด แล้วมือออกมาเป็นนวม
    #
    # นี่คือการยอมละทิ้งนิยาม visual hull เฉพาะในเขตเล็ก ๆ นี้ เพื่อให้ได้
    # รูปทรงที่ตรงกับตัวละครจริงมากกว่า ซึ่งเป็นการแลกแบบเดียวกับที่โหมด
    # ellipse ทำอยู่แล้วทั้งตัว
    if keep is not None and front is not None and keep.any():
        opened_gaps = int((front & keep & ~fb).sum())
        fb = np.where(keep, front, fb)
        if opened_gaps:
            log.info("เขตมือ/เท้า: ใช้ silhouette มุมหน้าแทนผลตัดกัน "
                     "(เปิดร่องระหว่างนิ้วคืน %d voxel)", opened_gaps)

    raw = fb.copy()
    k = int(cfg.spike_open)
    if k <= 0:
        return _smooth_silhouette(fb, cfg, keep), _smooth_silhouette(sd, cfg)

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

    # คืน silhouette ดิบในเขตมือ/เท้า
    #
    # opening ขนาด 7x7 ลบทุกอย่างที่บางกว่า 7 voxel ซึ่งก็คือ *นิ้วทุกนิ้ว*
    # (นิ้วหนาราว 4 voxel ที่ความละเอียดนี้) กฎนี้ถูกสำหรับหนามสันหลังที่เรา
    # ตั้งใจลบ แต่ผิดสำหรับมือ เพราะนิ้วไม่ใช่สัญญาณรบกวน มันคือตัวชิ้นงาน
    if keep is not None and keep.any():
        restored = int((raw & keep & ~fb).sum())
        fb = fb | (raw & keep)
        if restored:
            log.info("คืน silhouette ดิบในเขตมือ/เท้า %d voxel (นิ้วที่ opening กินไป)",
                     restored)

    return _smooth_silhouette(fb, cfg, keep), _smooth_silhouette(sd, cfg)


def _smooth_silhouette(
    slab: np.ndarray, cfg: PipelineConfig, keep: np.ndarray | None = None,
) -> np.ndarray:
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
    field = slab.astype(np.float32)
    blurred = ndimage.gaussian_filter(field, sigma=(s, s * 0.4))

    # ร่องระหว่างนิ้วกว้างแค่ 2-3 voxel การเบลอตามแนวตั้งเต็มที่จะถมมันเต็ม
    # ในเขตมือ/เท้าจึงเบลออ่อนกว่ามาก
    #
    # ต้องผสมกันแบบ *ไล่ระดับ* ไม่ใช่สลับทันทีที่ขอบเขต เพราะการสลับทันที
    # ทำให้ขอบ silhouette กระโดดหนึ่งถึงสอง voxel ตรงรอยต่อพอดี แล้วเกิด
    # เป็นชั้นแบน ๆ ยื่นรอบข้อมือกับข้อเท้าเหมือนบัวปูน
    if keep is not None and keep.any():
        weak = ndimage.gaussian_filter(field, sigma=(s * 0.25, s * 0.1))
        w = ndimage.gaussian_filter(keep.astype(np.float32), sigma=(2.0 * s, s))
        w = np.clip(w / max(float(w.max()), 1e-6), 0.0, 1.0)
        blurred = (1.0 - w) * blurred + w * weak
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
    z_back_body: np.ndarray,
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
    zone = np.zeros((ny, nx), bool)

    lo = max(0, int(np.floor(body_cx - rx)))
    hi = min(nx, int(np.ceil(body_cx + rx)) + 1)
    for y in np.flatnonzero(valid):
        z_core = float(z_back_body[y])         # ขอบหลังของลำตัวหลังกรองหนามออก
        z_tip = float(z_back_raw[y])           # ปลายหนามจริง
        if z_core - z_tip < cfg.crest_min:
            continue
        c = 0.5 * (z_core + z_tip)
        r = 0.5 * (z_core - z_tip)
        dx = ((xs - body_cx) / rx) ** 2
        dz = ((zs - c) / max(r, 0.5)) ** 2
        occ[y] |= (dx[:, None] + dz[None, :]) <= 1.0
        zone[y, lo:hi] = True
        rows += 1

    if rows:
        log.info("คืนหนามสันหลังเป็นสันแคบ %d แถว กว้าง %.0f voxel", rows, 2 * rx)
    return rows, zone


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


def _find_chin_row(
    z_front: np.ndarray, valid: np.ndarray, cfg: PipelineConfig
) -> int:
    """แถวที่ *หัวจบและลำตัวเริ่ม* = แนวคาง

    สัญญาณที่ใช้คือขอบหน้าของ silhouette ด้านข้าง: ไล่จากยอดหัวลงมา ขอบหน้า
    จะค่อย ๆ ยื่นออกไปตามแนวปากจนสุดที่ปลายปาก แล้ว *ถอยกลับฮวบเดียว*
    ตรงใต้คาง เพราะใต้คางคืออก ซึ่งอยู่ถอยไปข้างหลังกว่าปลายปากมาก
    (วัดได้ว่าถอย 12 ซม. ภายในแถวเดียว) จุดนั้นคือแนวคาง

    ทำไมไม่ใช้ความกว้างจากมุมหน้า: ตัวละครจำนวนมาก (รวมตัวนี้) ไม่มีคอให้เห็น
    จากด้านหน้าเลย ไหล่ชนกรามพอดี ความกว้างจึงค่อย ๆ โตขึ้นเรื่อย ๆ
    ไม่มีจุดหักให้จับ ส่วนมุมข้างมีแนวคางชัดเสมอ

    คืน index ของแถวแรกที่ *ไม่ใช่หัว* (-1 ถ้าหาไม่เจอ)
    """
    idx = np.flatnonzero(valid)
    if idx.size < 16:
        return -1
    y_lo, y_hi = int(idx.min()), int(idx.max())
    span = y_hi - y_lo

    # มองหาเฉพาะช่วงบนของตัว แนวคางไม่เคยอยู่ต่ำกว่านี้
    lo = y_hi - int(cfg.chin_search * span)
    band = [y for y in range(lo, y_hi) if valid[y] and valid[y + 1]]
    if len(band) < 4:
        return -1

    drop = {y: float(z_front[y + 1] - z_front[y]) for y in band}
    y_best = max(drop, key=lambda y: drop[y])
    best = drop[y_best]

    # ต้องถอยมากพอจึงจะเชื่อว่าเป็นคางจริง ไม่ใช่ขอบที่สั่น
    scale = float(np.percentile(z_front[idx], 90) - np.percentile(z_front[idx], 10))
    if best < cfg.chin_min_drop * max(scale, 1.0):
        return -1
    return y_best


def _limit_aspect(
    rz: np.ndarray,
    cz: np.ndarray,
    rx_body: np.ndarray,
    valid: np.ndarray,
    cfg: PipelineConfig,
    y_chin: int = -1,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """จำกัดไม่ให้หน้าตัดของ *ลำตัว* ลึกเกินกว่าความกว้างของตัวเองมากนัก

    ปัญหาที่แก้: ส่วนที่ยื่นไปข้างหน้าอย่างปากสัตว์ ห้อยลงมาคลุมถึงระดับคอ
    mask ด้านข้างที่แถวนั้นจึงกินตั้งแต่ท้ายทอยถึงปลายปาก พอเอาความลึกนั้นไป
    ครอบความกว้างของบ่าทั้งแถว จะได้แผ่นแบนคาดรอบตัวเหมือนปีกหมวก

    แบ่งตัวเป็นสามช่วงด้วย *แนวคาง* ที่อ่านจากมุมข้าง:

      เหนือคาง (หัว) : ไม่หั่นเลยโดยค่าเริ่มต้น หัวสัตว์ปากยาว *ควร* ลึก
                       กว่ากว้างสองเท่าจริง ๆ การไปหั่นแล้วเอามาแปะคืนเป็น
                       ก้อนปากทำให้เกิดจานแบนคาดกลางหัวกับก้อนเห็ดบนกะโหลก
                       (เปิดได้ด้วย --max-aspect ถ้าตัวละครคอเล็กจนเกิดปีกหมวก)
      ใต้คางถึงเหนือเท้า : หั่นด้วย `body_aspect` จาก *ด้านหลัง* เพราะส่วนเกิน
                       คือหางที่เชื่อมติดมาในมุมข้าง
      เท้า           : ไม่หั่น นิ้วเท้ายื่นไปหน้าทำให้เท้าลึกกว่ากว้างเสมอ

    เคยไล่ระดับการหั่นตามความกว้างของแถวแทนการแบ่งช่วง ซึ่งพังสองทาง: ขาที่
    แคบพอ ๆ กับหัวหลุดจากกฎ (หางเลยไปพองต้นขา) และกลางหัวโดนหั่นเป็นหย่อม ๆ
    เพราะความกว้างหัวไล่ขึ้นเรื่อย ๆ จนข้ามเกณฑ์ตรงกลางพอดี

    คืน (rz ใหม่, cz ใหม่, ความกว้างหัว, ขอบหน้าเดิม, ขอบหลังก่อนหั่น, แถบหัว)
    """
    z_front_orig = (cz + rz).astype(np.float32)
    z_back_before = (cz - rz).astype(np.float32)
    idx = np.flatnonzero(valid & (rx_body > 0))
    head_rows = np.zeros(rz.shape, bool)

    if idx.size < 8:
        fallback = float(rx_body[idx].min()) if idx.size else 1.0
        return (rz, cz, max(fallback, 1.0), z_front_orig, z_back_before, head_rows)

    y_lo, y_hi = int(idx.min()), int(idx.max())
    span_rows = max(y_hi - y_lo, 1)

    if y_chin < 0:
        y_chin = y_hi - int(cfg.head_band * span_rows)
        log.warning("หาแนวคางจากมุมข้างไม่เจอ — ใช้สัดส่วนตายตัวแทน "
                    "(หัว = %.0f%% บนสุด)", 100 * cfg.head_band)
    head_rows[y_chin + 1:] = True
    log.info("แนวคางอยู่ที่ %.1f%% ของความสูง (หัวสูง %.1f%% ของตัว)",
             100 * (y_chin - y_lo) / span_rows,
             100 * (y_hi - y_chin) / span_rows)

    foot_rows = np.zeros(rz.shape, bool)
    foot_rows[: y_lo + int(cfg.foot_band * span_rows) + 1] = True

    # ความกว้างของหัว = ค่ากลางของแถบหัว ใช้เป็นไม้บรรทัดของสันหนามกับปาก
    head_idx = np.flatnonzero(head_rows & valid & (rx_body > 0))
    rx_head = float(np.median(rx_body[head_idx])) if head_idx.size else 1.0
    rx_head = max(rx_head, 1.0)

    new_rz, new_cz = rz.copy(), cz.copy()

    # --- เพดานของหัว (ปิดไว้โดยค่าเริ่มต้น) --------------------------------
    trimmed_head = 0
    if cfg.max_aspect > 0:
        cap_head = cfg.max_aspect * np.maximum(rx_body, 0.5)
        hit = head_rows & valid & (new_rz > cap_head)
        if hit.any():
            # หดเข้าจากด้านหน้า ตรึงขอบหลัง — ส่วนที่หั่นออกคือ "ปาก"
            z_back_now = new_cz - new_rz
            new_rz = np.where(hit, cap_head, new_rz)
            new_cz = np.where(hit, z_back_now + new_rz, new_cz)
            trimmed_head = int(hit.sum())

    # --- เพดานของลำตัวกับขา: หั่นจากด้านหลัง (ส่วนเกินคือหาง) ----------------
    body_rows = valid & ~head_rows & ~foot_rows
    if cfg.body_aspect > 0 and body_rows.any():
        z_front_now = new_cz + new_rz
        cap_body = cfg.body_aspect * np.maximum(rx_body, 0.5)
        hit = body_rows & (new_rz > cap_body)
        if hit.any():
            new_rz = np.where(hit, cap_body, new_rz)
            new_cz = np.where(hit, z_front_now - new_rz, new_cz)
            log.info("หั่นความลึกช่วงลำตัว/ขาที่เกิน %.2f เท่าของความกว้าง %d แถว "
                     "(ส่วนเกินคือหางที่เชื่อมติดมาในมุมข้าง)",
                     cfg.body_aspect, int(hit.sum()))

    if trimmed_head:
        log.info("จำกัดความลึกช่วงหัวตามความกว้าง %d แถว", trimmed_head)
    return (new_rz.astype(np.float32), new_cz.astype(np.float32),
            rx_head, z_front_orig, z_back_before, head_rows)


def _add_snout(
    occ: np.ndarray,
    z_front_orig: np.ndarray,
    cz: np.ndarray,
    rz: np.ndarray,
    valid: np.ndarray,
    body_cx: float,
    rx_head: float,
    rx_body: np.ndarray,
    cfg: PipelineConfig,
) -> int:
    """คืนส่วนที่ยื่นไปข้างหน้า (ปาก/ขากรรไกร) กลับมาเป็นก้อนแคบของตัวเอง

    `_limit_aspect` หั่นความลึกของลำตัวลงเพื่อลบแผ่นแบนรอบคอ แต่ความลึกที่
    หั่นออกไปนั้นคือ *ปาก* ซึ่งเป็นของจริง ถ้าไม่คืนมา ตัวละครจะหน้าสั้นกุด

    ความต่างอยู่ที่ความกว้าง: คอกว้างเท่าบ่า แต่ปากแคบเท่าหัว เราจึงคืนมันมา
    เป็นก้อนแยกที่กว้างเท่าหัวเท่านั้น

    ความกว้างต้องคิดรายแถว ไม่ใช่ค่าคงที่ค่าเดียว: ปากของสัตว์เรียวลงไปทาง
    ปลาย ถ้าใช้ความกว้างเดียวทุกแถวจะได้ *จานแบน* แปะหน้าหัว แทนที่จะเป็น
    ขากรรไกรที่สอบเข้า และแถวไหนที่ลำตัวแคบกว่าหัว ปากก็ต้องไม่กว้างเกิน
    ลำตัวแถวนั้น ไม่งั้นจะมีเนื้อโผล่ออกนอก silhouette ด้านหน้า
    """
    ny, nx, nz = occ.shape
    xs = np.arange(nx, dtype=np.float32)
    zs = np.arange(nz, dtype=np.float32)
    rows = 0
    rx_used: list[float] = []

    for y in np.flatnonzero(valid):
        z_cap = float(cz[y] + rz[y])
        z_out = float(z_front_orig[y])
        if z_out - z_cap < 1.5:
            continue
        c = 0.5 * (z_cap + z_out)
        r = 0.5 * (z_out - z_cap)
        # ยิ่งยื่นไกล ยิ่งแคบ — ปลายปากเรียวกว่าโคน
        taper = 1.0 / (1.0 + 0.6 * r / max(rx_head, 1.0))
        rx = max(min(rx_head * cfg.snout_width * taper,
                     float(rx_body[y])), 1.0)
        dx = ((xs - body_cx) / rx) ** 2
        dz = ((zs - c) / max(r, 0.5)) ** 2
        occ[y] |= (dx[:, None] + dz[None, :]) <= 1.0
        rows += 1
        rx_used.append(rx)

    if rows:
        log.info("คืนส่วนที่ยื่นหน้า (ปาก) เป็นก้อนเรียว %d แถว กว้าง %.0f-%.0f voxel",
                 rows, 2 * min(rx_used), 2 * max(rx_used))
    return rows


def _carve_neck(
    occ: np.ndarray,
    cz: np.ndarray,
    rz: np.ndarray,
    valid: np.ndarray,
    y_chin: int,
    rx_head: float,
    body_cx: float,
    cfg: PipelineConfig,
) -> int:
    """เซาะคอออกจากก้อนไหล่ โดยไม่แตะ silhouette ของมุมไหนเลย

    ทำไมต้องปั้นเอง: ตัวละครที่ไหล่ใหญ่ชนกราม *ไม่มีคอให้เห็นในมุมไหนเลย*
    มุมหน้าวัดความกว้างทีละ 1% ของความสูงแล้วโตขึ้นทางเดียวตลอด ไม่มีรอยคอด
    มุมข้างก็เห็นแต่ขอบนอกของบ่า คอถูกบังอยู่ *ข้างใน* ทั้งสองมุม
    visual hull จึงเติมเนื้อเต็มช่องระหว่างคอกับยอดบ่า ได้หัวที่กลืนเป็น
    ทรงกรวยต่อกับไหล่เหมือนใส่ฮู้ด

    กุญแจคือ: คอเป็นรายละเอียด *บนผิว* ไม่ใช่เรื่องของเส้นขอบ
      * เส้นขอบแกน X ของแถวนั้นมาจาก "ปลายบ่า" ไม่ใช่คอ
      * เส้นขอบแกน Z ของแถวนั้นมาจาก "คอ/อก" ซึ่งเราเก็บไว้เต็ม
    การลดความลึกเฉพาะช่วงที่อยู่ *ข้าง ๆ คอ* จึงไม่ไปแตะเงาของมุมไหนเลย
    แต่ทำให้ซอกระหว่างคอกับบ่าโผล่ออกมา ซึ่งเป็นสิ่งที่ตาคนใช้อ่านว่า
    "หัวตั้งอยู่บนคอ" จริง ๆ

    รูปทรงที่ได้: แกนกลางเป็นลำคอความลึกเต็ม สองข้างเป็นสันบ่าที่บางตรง
    ใต้คางแล้วค่อยหนาขึ้นจนเต็มเมื่อลงไปถึงระดับไหล่ ตรงกับกล้ามเนื้อสะบัก
    ที่ลาดลงจากคอไปหาหัวไหล่

    ขั้นนี้ *ตัดอย่างเดียว ไม่เพิ่มเนื้อ* จึงไม่มีทางทำให้เกินภาพต้นทางได้
    """
    if not cfg.neck_carve or y_chin <= 0:
        return 0

    ny, nx, nz = occ.shape
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return 0
    head_h = max(int(idx.max()) - y_chin, 1)
    n_rows = max(2, int(cfg.neck_len * head_h))

    xs = np.arange(nx, dtype=np.float32)
    zs = np.arange(nz, dtype=np.float32)
    p = float(cfg.cross_section_power)
    neck_r = max(cfg.neck_width * rx_head, 1.0)
    removed = 0
    rows = 0

    for i in range(n_rows):
        y = y_chin - i
        if y < 0 or not valid[y]:
            continue
        plane = occ[y]
        on = np.flatnonzero(plane.any(axis=1))
        if on.size < 4:
            continue
        a, b = int(on[0]), int(on[-1])
        rx_row = max(0.5 * (b - a), 1.0)
        if rx_row <= neck_r * 1.05:
            continue              # แถวนี้แคบเท่าคออยู่แล้ว ไม่มีอะไรให้เซาะ

        cx_row = 0.5 * (a + b)
        cz_row = float(cz[y])
        rz_full = max(float(rz[y]), 0.5)
        # t: 0 ที่ใต้คาง -> 1 ที่ระดับไหล่เต็ม
        t = i / max(n_rows - 1, 1)
        s = cfg.neck_shoulder_min + (1.0 - cfg.neck_shoulder_min) * t

        dxn = np.abs((xs - body_cx) / neck_r) ** p
        dzn = np.abs((zs - cz_row) / rz_full) ** p
        keep = (dxn[:, None] + dzn[None, :]) <= 1.0          # ลำคอ

        dxs = np.abs((xs - cx_row) / rx_row) ** p
        dzs = np.abs((zs - cz_row) / max(rz_full * s, 0.5)) ** p
        keep |= (dxs[:, None] + dzs[None, :]) <= 1.0         # สันบ่า

        before = int(plane.sum())
        plane &= keep
        removed += before - int(plane.sum())
        rows += 1

    if rows:
        log.info("เซาะคอ %d แถว (คอกว้าง %.0f voxel, บ่าบางสุด %.0f%% ของความลึก) "
                 "-%d voxel", rows, 2 * neck_r, 100 * cfg.neck_shoulder_min, removed)
    return removed


def _trace_to_voxels(trace, grid: HullGrid, ny: int, nx: int) -> np.ndarray:
    """แปลงเส้นทางหางจากพิกัดพิกเซลบน canvas เป็นพิกัด voxel

    คืนอาเรย์ (N, 3) ของ (ix, iy, รัศมีเป็น voxel) เรียงจากโคนไปปลาย โดย
    *สุ่มใหม่ตามระยะทางจริงบนเส้น* ไม่ใช่แถวละจุด

    เดิมเก็บแถวละจุดเดียว ซึ่งใช้ได้เฉพาะหางที่ห้อยลงตรง ๆ พอหางสะบัดจน
    เกือบนอน หลายสิบพิกเซลของความยาวหางจะตกอยู่ในแถวเดียวกัน แล้วเหลือจุด
    เดียว หางที่ปั้นได้จึงขาดเป็นช่วง ๆ การสุ่มตามระยะทางให้จุดถี่สม่ำเสมอ
    ตลอดเส้นไม่ว่าหางจะเอียงแค่ไหน
    """
    row_bottom, col_x0, _ = grid.origin_px
    vp = grid.voxel_px

    iy = (row_bottom - trace.rows) / vp - 0.5
    ix = (trace.cx - col_x0) / vp - 0.5
    hw = trace.half_w / vp

    ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    iy, ix, hw = iy[ok], ix[ok], hw[ok]
    if iy.size < 2:
        return np.stack([ix, iy, hw], axis=1)

    # เรียงจากโคน (iy มาก) ไปปลาย (iy น้อย)
    order = np.argsort(-iy)
    iy, ix, hw = iy[order], ix[order], hw[order]

    seg = np.hypot(np.diff(ix), np.diff(iy))
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s[-1])
    if total < 1e-6:
        return np.stack([ix, iy, hw], axis=1)

    n_samp = max(int(np.ceil(total * 2.0)) + 1, len(iy))
    su = np.linspace(0.0, total, n_samp)
    return np.stack([np.interp(su, s, ix),
                     np.interp(su, s, iy),
                     np.interp(su, s, hw)], axis=1)


def _add_tail_tube(
    occ: np.ndarray,
    tail_vox: np.ndarray,
    cz: np.ndarray,
    rz: np.ndarray,
    valid: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    """ปั้นหางเป็นท่อกลมตามเส้นทางที่ลากจากภาพด้านหลัง

    ภาพด้านหลังบอกตำแหน่งแกน X กับความหนามาครบ แต่บอกความลึกไม่ได้เลย
    เราจึงวางหางไว้ที่ *ผิวด้านหลังของลำตัว* ในแต่ละแถว ซึ่งเป็นที่ที่หางพาด
    อยู่จริงเมื่อมองจากด้านหลัง และสมมติว่าหน้าตัดกลม เพราะหางสัตว์หนาพอ ๆ กัน
    ทุกทิศ การเดาว่ากลมจึงสมเหตุผลกว่าการเดาความลึกจากมุมอื่นที่วางหางคนละที่

    วาดเป็น *ทรงกลมเรียงตามเส้น* ไม่ใช่จานรายแถว เพราะจานรายแถวจะขาดเป็น
    ช่วงทันทีที่หางเอียงจนเกือบนอน (หนึ่งแถวครอบความยาวหางหลายสิบ voxel)
    """
    ny, nx, nz = occ.shape
    before = int(occ.sum())
    path = []
    last_row = -999

    for ix, iy, hw in tail_vox:
        y = int(round(iy))
        if not (0 <= y < ny) or not valid[y]:
            continue
        r = max(float(hw) * cfg.tail_thickness, 1.0)
        # +Z คือทิศหน้า ผิวหลังจึงอยู่ที่ cz - rz และการดันหางออกไปด้านหลัง
        # คือการ *ลบ* ค่าลงไปอีก ไม่ใช่บวก
        czt = float(cz[y] - rz[y]) - r * cfg.tail_z_bias

        y0, y1 = max(0, int(np.floor(iy - r))), min(ny, int(np.ceil(iy + r)) + 1)
        x0, x1 = max(0, int(np.floor(ix - r))), min(nx, int(np.ceil(ix + r)) + 1)
        z0, z1 = max(0, int(np.floor(czt - r))), min(nz, int(np.ceil(czt + r)) + 1)
        if y0 >= y1 or x0 >= x1 or z0 >= z1:
            continue
        gy = (np.arange(y0, y1, dtype=np.float32) - iy) ** 2
        gx = (np.arange(x0, x1, dtype=np.float32) - ix) ** 2
        gz = (np.arange(z0, z1, dtype=np.float32) - czt) ** 2
        ball = (gy[:, None, None] + gx[None, :, None] + gz[None, None, :]) <= r * r
        occ[y0:y1, x0:x1, z0:z1] |= ball

        # เก็บเส้นกระดูกแถวละจุด พอสำหรับวางโซ่กระดูกหาง
        if y != last_row:
            path.append((float(ix), float(y), czt, r))
            last_row = y

    if not path:
        return None, None
    log.info("ปั้นหางตามภาพด้านหลัง %d ลูกกลม / %d แถว (+%.1f%% ของปริมาตรตัว)",
             len(tail_vox), len(path), 100 * (occ.sum() - before) / max(before, 1))
    arr = np.array(path, np.float32)
    return arr[:, :3], arr[:, 3]


def _fill_elliptical(
    fb: np.ndarray, sd: np.ndarray, cfg: PipelineConfig,
    tail_rows: np.ndarray | None = None,
    tail_span: np.ndarray | None = None,
    ext=None,
    sole_plans: list | None = None,
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

    # หาแนวคาง *ก่อน* เกลาโปรไฟล์
    #
    # สัญญาณของคางคือขอบหน้าถอยกลับฮวบเดียวภายในแถวเดียว พอเกลาด้วย gaussian
    # sigma 4 แถว การถอย 32 voxel จะกลายเป็นความชัน 3 voxel ต่อแถว ซึ่งจมอยู่
    # ในระดับสัญญาณรบกวนจนหาไม่เจอ ต้องอ่านจากค่าดิบเท่านั้น
    y_chin = _find_chin_row(cz_profile + rz_profile, valid, cfg)

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

    (rz_profile, cz_profile, rx_head, z_front_orig,
     z_back_pre_aspect, head_rows) = _limit_aspect(
        rz_profile, cz_profile, rx_body, valid, cfg, y_chin)

    # หาหางก่อนปั้นลำตัว เพราะลำตัวต้องรู้ว่าท่อนไหนคือหางเพื่อจะได้ข้ามไป
    limb_runs, body_cx = _limb_runs(fb, min_run, cfg)
    pruned_count = 0
    if tail_rows is None:
        tail_rows = np.zeros(ny, bool)
    if tail_span is None:
        tail_span = np.zeros((ny, 2), np.float32)

    for y in range(ny):
        x_runs = _runs(fb[y], min_run)
        if not x_runs or not valid[y]:
            continue
        if len(_runs(sd[y], 1)) > 1:
            dropped_rows += 1

        # ในช่วงที่มีหาง ตัดออกเฉพาะท่อนที่ *ทับตำแหน่งหางจริง* เท่านั้น
        #
        # เคยเขียนว่า "ท่อนที่ไม่ใช่ลำตัว/ขา คือหาง" แล้วแทนทั้งแถวด้วยท่อน
        # ลำตัว/ขา ซึ่งใช้ได้ตอนที่ช่วงหางอยู่ต่ำกว่าระดับมือเสมอ
        # พอเปลี่ยนมาลากหางจากภาพหลัง ช่วงหางขยายขึ้นไปคลุมระดับมือด้วย
        # แขนกับมือจึงถูกทิ้งหมด เหลือแต่ตอ
        #
        # ตอนนี้รู้ตำแหน่งกับความกว้างของหางรายแถวแล้ว จึงเทียบได้ตรง ๆ ว่า
        # ท่อนไหนคือหาง ท่อนที่อยู่ไกลออกไป (แขน มือ) ไม่ถูกแตะเลย
        if tail_rows[y] and limb_runs[y]:
            core = limb_runs[y]
            t0, t1 = tail_span[y]
            kept = list(core)
            for r in x_runs:
                if any(r[0] <= c[0] and c[1] <= r[1] for c in core):
                    continue              # ท่อนลำตัว/ขา มีตัวที่เล็มแล้วแทนอยู่
                if r[0] <= t1 and t0 <= r[1]:
                    pruned_count += 1
                    continue              # ทับหาง ปล่อยให้ขั้นปั้นหางจัดการ
                kept.append(r)            # ไกลจากหาง = แขน/มือ เก็บไว้
            x_runs = sorted(kept)

        cz = float(cz_profile[y])
        rz_full = float(rz_profile[y])
        rx_max = max(float(rx_body[y]), 0.5)

        # จับท่อนในเขตมือ/เท้าเข้ากลุ่มตามอวัยวะ (มือซ้าย / มือขวา / เท้าหนึ่งข้าง)
        #
        # ต้องจัดกลุ่ม เพราะค่าที่ต้องใช้คือขนาดของ *อวัยวะทั้งชิ้น*:
        #   - ความหนาฝ่ามือคิดจากความกว้างของมือทั้งมือ ไม่ใช่ของนิ้วกลุ่มเดียว
        #   - "นิ้ว" คือท่อนที่แคบเทียบกับมือข้างเดียวกัน ไม่ใช่เทียบกับลำตัว
        #     หรือเทียบกับมืออีกข้างที่อยู่คนละฟากของแถวเดียวกัน
        zone_group: dict[tuple[int, int], tuple[int, int]] = {}
        if ext is not None and (ext.foot_rows[y] or ext.hand_rows[y]):
            zrow = ext.zone[y]
            gap_max = max(4, nx // 25)
            groups: list[list[tuple[int, int]]] = []
            for r in x_runs:
                if not zrow[int(round(0.5 * (r[0] + r[1])))]:
                    continue
                if groups and r[0] - groups[-1][-1][1] <= gap_max:
                    groups[-1].append(r)
                else:
                    groups.append([r])
            for grp in groups:
                span = max(grp[-1][1] - grp[0][0], 1)
                widest = max((b - a for a, b in grp), default=1) or 1
                for r in grp:
                    zone_group[r] = (span, widest)

        plane = occ[y]
        for a, b in x_runs:
            cx = 0.5 * (a + b)
            rx = max(0.5 * (b - a), 0.5)
            # ตรึงอัตราส่วนไว้ ไม่ให้ชิ้นที่กว้างกว่าลำตัว (เช่นช่วงไหล่กางแขน)
            # ได้ความลึกเกินความหนาจริงของตัว และไม่ให้ชิ้นที่บางมากแบนจนหาย
            ratio = float(np.clip(rx / rx_max, 0.15, 1.0))
            rz = rz_full * ratio ** cfg.depth_falloff
            cz_run = cz

            grp = zone_group.get((a, b))
            if grp is not None:
                span, widest = grp
                narrow = (b - a) <= cfg.digit_ratio * widest
                # ความลึกของมือต้องมาจากสัดส่วนของมือเอง ไม่ใช่ของสะโพก
                #
                # ไม่มีมุมไหนวัดความหนาของมือได้เลย: มุมหน้าเห็นหลังมือ
                # มุมข้างเห็นมือซ้อนอยู่กับสะโพกแยกไม่ออก กฎ depth_falloff
                # จึงไปหยิบความลึกของ *สะโพก* มาใช้ มือเลยหนา 14 ซม.
                # กลายเป็นก้อนนวมที่ร่องนิ้วจมหายไปหมด
                #
                # ฝ่ามือหนาราว 0.4 เท่าของความกว้างตัวเอง ซึ่งเป็นสัดส่วนที่
                # จริงกับมือแทบทุกแบบ ใช้สัดส่วนของตัวเองจึงแม่นกว่าการไป
                # ยืมความลึกจากส่วนอื่นของร่างกาย
                if narrow:
                    # นิ้ว: หน้าตัดกลม ไม่ใช่ครีบแบนที่ลึกเท่าลำตัว
                    rz = min(rz, rx * cfg.digit_round)
                elif ext.hand_rows[y]:
                    rz = min(rz, 0.5 * span * cfg.palm_round)
                if narrow:
                    # นิ้วเท้าชี้ไปข้างหน้า ต้องวางชิดขอบหน้าของฝ่าเท้า
                    cz_run = cz + float(ext.z_push[y]) * max(rz_full - rz, 0.0)

            dx = np.abs((xs - cx) / rx) ** p
            dz = np.abs((zs - cz_run) / max(rz, 0.5)) ** p
            plane |= (dx[:, None] + dz[None, :]) <= 1.0

    # --- ปาก: คืนส่วนที่ยื่นหน้าซึ่งถูกหั่นออกไป เป็นก้อนแคบของตัวเอง --------
    #
    # จำกัดไว้เฉพาะแถบหัวเท่านั้น ไม่งั้นทุกแถวที่โดนหั่นความลึก (รวมช่วงคอ
    # กับไหล่) จะได้ก้อนกว้างเท่าหัวแปะเพิ่มเข้าไป กลายเป็นจานแบนคาดรอบคอ
    if cfg.max_aspect > 0 and cfg.snout_width > 0:
        _add_snout(occ, z_front_orig, cz_profile, rz_profile,
                   valid & head_rows, body_cx, rx_head, rx_body, cfg)

    crest_zone = np.zeros_like(fb)

    # --- หนามสันหลัง: คืนส่วนที่ median filter ตัดไป เป็นสันแคบแนวกลาง -------
    #
    # เทียบกับขอบหลัง *ก่อน* เพดานลำตัวจะหั่นหาง ไม่งั้นช่วงที่หางถูกตัดออก
    # จะถูกนับเป็น "หนาม" แล้วงอกสันพาดหลังยาวลงไปถึงขา
    if cfg.crest_width > 0:
        _, crest_zone = _add_crest(occ, z_back_raw, z_back_pre_aspect, valid,
                                   body_cx, rx_head, cfg)

    # --- คอ: เซาะซอกระหว่างคอกับบ่า -----------------------------------------
    #
    # ต้องทำ *หลัง* คืนสันหนาม ไม่งั้นสันที่พาดลงท้ายทอยจะถูกเซาะหายไปด้วย
    _carve_neck(occ, cz_profile, rz_profile, valid, y_chin, rx_head,
                body_cx, cfg)

    # --- เท้า: ใช้ผังฝ่าเท้าถ้ามี ไม่งั้นเซาะร่องเอาจาก silhouette มุมหน้า ---
    if ext is not None:
        from . import extremities as _ext
        if sole_plans:
            _ext.apply_sole_plan(occ, ext, sole_plans, cz_profile, rz_profile,
                                 valid, cfg)
        elif cfg.toe_grooves:
            _ext.carve_toe_grooves(occ, ext, cz_profile, rz_profile, valid, cfg)

    if pruned_count:
        log.info("ตัดเนื้อหางออกจากขั้นปั้นลำตัว %d แถว (ไปปั้นใหม่ที่ความลึกจริงแทน)", pruned_count)
    if dropped_rows:
        log.info("ตัดส่วนที่ยื่นออกจากลำตัวใน mask ด้านข้าง %d แถว "
                 "(หาง/เป้/ปีก จะไม่ไปพองความหนาลำตัว)", dropped_rows)
    return occ, cz_profile, rz_profile, valid, crest_zone


def _blend_detail_blur(
    field: np.ndarray, blurred: np.ndarray, zone: np.ndarray,
    cfg: PipelineConfig, what: str = "มือ/เท้า",
) -> np.ndarray:
    """เบลอเขตรายละเอียดเบากว่าที่อื่น แล้วไล่ระดับเข้าหากันไม่ให้มีรอยต่อ

    sigma 2.6 voxel ลบอะไรก็ตามที่บางกว่าราว 5 voxel ซึ่งก็คือนิ้วทั้งมือ
    และหนามบนสันหลังทั้งแถว แต่ลดทั้งตัวไม่ได้ เพราะลำตัวที่ปั้นจากหน้าตัด
    ทีละแถวจะกลับไปเป็นผิววงแหวนขั้นบันไดทันที (ลองลดเป็น 1.2 แล้วเห็นชัด)
    เบลอสองค่าแล้วผสมกันตามเขตจึงได้ทั้งสองอย่าง

    `zone` เป็น mask (ny, nx) ของพื้นที่ที่ต้องการให้คม
    ทำเฉพาะแถบแถวที่เกี่ยวข้อง เพื่อไม่ต้องถือ volume ขนาดเต็มไว้สองชุด
    """
    rows_on = zone.any(axis=1)
    if not rows_on.any():
        return blurred
    pad = max(4, int(2 * cfg.hull_blur))
    idx = np.flatnonzero(rows_on)
    y0 = max(0, int(idx[0]) - pad)
    y1 = min(field.shape[0], int(idx[-1]) + 1 + pad)

    sub = ndimage.gaussian_filter(field[y0:y1], sigma=float(cfg.detail_blur),
                                  mode="constant")

    # น้ำหนัก: 1 ในเขต แล้วไล่ลงเป็น 0 ภายในระยะ ~hull_blur
    w = ndimage.gaussian_filter(zone.astype(np.float32),
                                sigma=float(cfg.hull_blur), mode="constant")
    w = np.clip(w / max(float(w.max()), 1e-6), 0.0, 1.0)[y0:y1, :, None]

    blurred[y0:y1] = (1.0 - w) * blurred[y0:y1] + w * sub
    log.info("เบลอเขต%sด้วย sigma %.1f (ลำตัวใช้ %.1f) ครอบแถว %d-%d",
             what, cfg.detail_blur, cfg.hull_blur, y0, y1 - 1)
    return blurred


#: ความละเอียดอ้างอิงที่ค่าตั้งต้นทั้งหมดใน PipelineConfig ถูกจูนมา
REF_VOXEL_RES = 320

#: ค่าที่มีหน่วยเป็น "จำนวน voxel" จึงต้องสเกลตามความละเอียด
_LENGTH_FIELDS = ("hull_blur", "silhouette_smooth", "profile_smooth",
                  "depth_median", "spike_open", "crest_min", "detail_blur",
                  "detail_dilate")


def _scaled_cfg(cfg: PipelineConfig) -> PipelineConfig:
    """สเกลค่าที่มีหน่วยเป็น voxel ให้ได้ผลทางกายภาพเท่าเดิมทุกความละเอียด

    ค่าอย่าง hull_blur=2.6 หรือ spike_open=3 ไม่ได้แปลว่า "2.6 มิลลิเมตร"
    แต่แปลว่า "2.6 voxel" พอเพิ่ม --voxel-res จาก 320 เป็น 448 voxel หนึ่งลูก
    เล็กลง 1.4 เท่า การเบลอเท่าเดิมจึงกลายเป็นการเบลอที่ *อ่อนลง 1.4 เท่า*
    ในหน่วยเซนติเมตร ผลคือเพิ่มความละเอียดแล้วผิวกลับเป็นขั้นบันไดหนักกว่าเดิม
    ซึ่งตรงข้ามกับที่ควรเป็น

    สเกลตรงนี้ทำให้ --voxel-res เป็นปุ่มที่ทำสิ่งเดียวตามชื่อ: เพิ่มความละเอียด
    ของการสุ่ม โดยไม่ไปเปลี่ยนรูปทรงที่ตั้งใจไว้
    """
    k = cfg.voxel_res / REF_VOXEL_RES
    if abs(k - 1.0) < 1e-6:
        return cfg
    changes: dict[str, object] = {}
    for name in _LENGTH_FIELDS:
        value = getattr(cfg, name)
        changes[name] = (max(0, int(round(value * k))) if isinstance(value, int)
                         else float(value) * k)
    log.info("สเกลค่าหน่วย voxel ด้วย %.2f เท่า (ความละเอียด %d เทียบฐาน %d)",
             k, cfg.voxel_res, REF_VOXEL_RES)
    return replace(cfg, **changes)


def carve(views: dict[str, PreparedView], cfg: PipelineConfig,
          sole_plans: list | None = None) -> HullGrid:
    """ตัดกริด voxel ด้วย silhouette ของทุกมุม

    voxel เป็นลูกบาศก์จริง (ขนาดเท่ากันทั้งสามแกน) กริดจึงไม่ใช่ทรงลูกบาศก์
    แต่พอดีกับสัดส่วนตัวละคร — ได้ความละเอียดต่อหน่วยความจำดีกว่ากริด N^3 มาก
    """
    cfg = _scaled_cfg(cfg)
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
    fb_front = None
    used: list[str] = []

    for name, v in views.items():
        if v.spec.axis_u == "x":
            slab = v.mask[ri[:, None], xi[None, :]]
            fb &= slab
            if name == "front":
                fb_front = slab
        else:
            sd &= v.mask[ri[:, None], zi[None, :]]
        used.append(name)

    # หาเขตมือ/เท้า *ก่อน* ลบหนาม เพราะหลังลบแล้วนิ้วไม่เหลือให้หา
    ext = None
    if cfg.carve_mode == "ellipse" and (cfg.detail_hands or cfg.detail_feet):
        from . import extremities
        ext = extremities.detect(fb, cfg, fb_front)
        if not ext.any:
            ext = None

    fb, sd = _remove_spikes(fb, sd, cfg,
                            keep=None if ext is None else ext.zone,
                            front=fb_front)

    if cfg.carve_mode == "ellipse":
        # ลากหางจากภาพด้านหลังก่อน ลำตัวต้องรู้ว่าแถวไหนมีเนื้อหางปนอยู่
        trace = None
        if cfg.keep_tail and "back" in views:
            from . import tailtrace
            trace = tailtrace.trace(views["back"], cfg)

        tail_rows = np.zeros(ny, bool)
        tail_span = np.zeros((ny, 2), np.float32)
        tail_vox = None
        if trace is not None:
            tail_vox = _trace_to_voxels(trace, grid, ny, nx)
            # หนึ่งแถวมีได้หลายจุด (ช่วงที่หางเกือบนอน) ช่วงที่ต้องตัดออกจาก
            # ลำตัวจึงเป็น *ยูเนียน* ของทุกจุดในแถวนั้น ไม่ใช่จุดสุดท้าย
            lo = np.full(ny, np.inf, np.float32)
            hi = np.full(ny, -np.inf, np.float32)
            for tx, ty, thw in tail_vox:
                r = int(round(ty))
                if not 0 <= r < ny:
                    continue
                margin = thw * cfg.tail_prune_pad
                lo[r] = min(lo[r], tx - margin)
                hi[r] = max(hi[r], tx + margin)
            tail_rows = np.isfinite(lo)
            tail_span = np.stack([np.where(tail_rows, lo, 0.0),
                                  np.where(tail_rows, hi, 0.0)], axis=1)

        occ, czp, rzp, valid, crest_zone = _fill_elliptical(
            fb, sd, cfg, tail_rows, tail_span, ext, sole_plans)

        if tail_vox is not None:
            grid.tail_path, grid.tail_radius = _add_tail_tube(
                occ, tail_vox, czp, rzp, valid, cfg)
    else:
        # โหมด hull ตามนิยาม ใช้ช่วง Z ทั้งหมดอยู่แล้ว หางจึงติดมาเอง
        occ = fb[:, :, None] & sd[:, None, :]
        crest_zone = np.zeros_like(fb)

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
        blurred = ndimage.gaussian_filter(field, sigma=cfg.hull_blur, mode="constant")
        if cfg.detail_blur < cfg.hull_blur:
            # เขตที่ต้องคม: มือ/เท้า (นิ้ว) และสันหนามกลางหลัง
            #
            # หนามสูงราว 6 voxel หนา 4 voxel การเบลอ sigma 3.6 ลบมันเกลี้ยง
            # ทั้งที่ `_add_crest` เพิ่งปั้นคืนมาให้ ผลคือหัวเรียบเป็นกรวย
            # ไม่มีสันเลย ทั้งที่ภาพต้นทางมีสันหนามเป็นเอกลักษณ์
            zone = crest_zone if cfg.crest_sharp else np.zeros_like(crest_zone)
            if ext is not None:
                zone = zone | ext.zone
            blurred = _blend_detail_blur(field, blurred, zone, cfg,
                                         "มือ/เท้า/สันหนาม")
        field = blurred

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
