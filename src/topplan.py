"""ดึง *ผังมุมบน* (รูปตัวละครมองจากข้างบน) เพื่อบอกความลึกของหาง

ทำไมต้องมีภาพนี้
----------------
``_add_tail_tube`` ใน :mod:`src.visual_hull` ต้องวางหางในพิกัดสามมิติ แต่
ภาพด้านหลังที่มันใช้บอกได้แค่ ``(X, Y)`` กับความหนา — **ไม่มี Z เลย** โค้ด
จึงต้องเดาเอาว่าหางแปะอยู่ที่ผิวด้านหลังของลำตัวทุกแถว ซึ่งใช้ได้กับหางที่
ห้อยตรง ๆ แต่ผิดทันทีที่หางสะบัดออกไปข้างหน้าหรือพาดขวางขา

มุมข้างก็ช่วยไม่ได้ เพราะเงามุมข้างให้ช่วง Z ได้ *แถวละค่าเดียว* แล้วเอาไป
ใช้กับทุกตำแหน่ง X ในแถวนั้น ที่ความสูงเดียวกันขาอยู่ X หนึ่ง หางอยู่ X
อีกที่หนึ่ง มุมข้างแยกสองอย่างนี้ไม่ออก (ความลึกของหางจึงรั่วไปพองขา)

ภาพมุมบนมองเห็น **ระนาบ X-Z** ตรง ๆ หางที่สะบัดไปทางไหน ลึกเท่าไร อยู่ในภาพ
นี้ครบ เป็นระนาบเดียวกับที่ :mod:`src.soleplan` ใช้กับฝ่าเท้า ต่างกันที่
มองจากข้างบนแทนข้างล่าง

วิธีประกอบกับภาพด้านหลัง
------------------------
ภาพด้านหลังให้ ``(X, Y)`` ของหาง ภาพมุมบนให้ ``(X, Z)`` ทั้งสองเส้นเรียงจาก
*โคนไปปลาย* เหมือนกัน จึงจับคู่กันด้วย **ความยาวตามเส้น** (สัดส่วน 0..1 จาก
โคน) แล้วได้ ``(X, Y, Z)`` ครบ — ไม่ต้องจับคู่ด้วยค่า X ซึ่งพังเมื่อหางวน
กลับทางเดิม

ข้อจำกัดที่ยอมรับ
-----------------
เหมือนภาพใต้ฝ่าเท้า: ภาพมุมบนเป็น perspective แรง (ไหล่ใหญ่เว่อร์เพราะใกล้
กล้อง) จึงใช้เป็น *รูปร่างเชิงสัดส่วน* เท่านั้น ความลึกที่อ่านได้ถูกปรับ
สเกลให้พอดีกับความลึกลำตัวที่วัดจากมุมข้าง ซึ่งแม่นอยู่แล้ว
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import IMAGE_EXTS, PipelineConfig

log = logging.getLogger(__name__)

#: ชื่อไฟล์ที่ถือว่าเป็นภาพมองจากข้างบน
TOP_ALIASES = ("top", "up", "above", "overhead", "มุมบน", "ข้างบน")


@dataclass
class TopPlan:
    """ผังมุมบนหนึ่งใบ พร้อมเส้นทางหางในระนาบ X-Z

    ``tail_x`` / ``tail_z`` เรียงจากโคนหาง (0) ไปปลายหาง (n-1) หน่วยเป็น
    *สัดส่วนของครึ่งความลึกลำตัว* โดยมีจุดกำเนิดอยู่ที่กึ่งกลางลำตัว:

    * ``tail_z = 0``  คือกึ่งกลางลำตัวตามแกนลึก
    * ``tail_z = -1`` คือผิวด้านหลังของลำตัว (ทิศ -Z)
    * ``tail_z = +1`` คือผิวด้านหน้าของลำตัว (ทิศ +Z)

    ``tail_x`` ใช้หน่วยเดียวกันแต่เทียบกับครึ่ง *ความกว้าง* ลำตัว
    """

    mask: np.ndarray            # ผัง X-Z ที่ครอปแล้ว (แถว = Z, +Z อยู่แถวท้าย)
    tail_x: np.ndarray
    tail_z: np.ndarray
    tail_r: np.ndarray          # รัศมีหาง เป็นสัดส่วนของครึ่งความกว้างลำตัว
    source: Path
    #: โปรไฟล์ความลึกตามแกน X — ดู :meth:`depth_at`
    prof_x: np.ndarray = None
    prof_r: np.ndarray = None
    #: ระยะที่หางถอยหลังจากโคน **คิดเป็นเท่าของความลึกลำตัว (จมูกถึงหลัง)**
    #: เป็นหน่วยเดียวที่เทียบข้ามภาพได้ เพราะทั้งสองฝั่งวัดของชิ้นเดียวกัน
    tail_back: np.ndarray = None
    #: รัศมีหางในหน่วยเดียวกัน
    tail_back_r: np.ndarray = None

    def __len__(self) -> int:
        return len(self.tail_x)

    def depth_at(self, x_norm):
        """ครึ่งความลึกเชิงสัมพัทธ์ที่ตำแหน่ง X (หน่วย: ครึ่งความกว้างลำตัว)

        คืนค่าในช่วง 0..1 โดย 1 = ส่วนที่ลึกที่สุดของลำตัว ใช้แทนกฎ
        ``depth_falloff`` ซึ่งเดาว่า "ท่อนแคบ = ตื้น" — จริงกับนิ้วแต่ผิดกับ
        แขน เพราะแขนแคบแต่กลมเกือบเท่าลำตัว มุมบนวัดค่านี้ได้ตรง ๆ
        """
        if self.prof_x is None or len(self.prof_x) < 2:
            return np.ones_like(np.asarray(x_norm, float))
        return np.interp(np.asarray(x_norm, float), self.prof_x, self.prof_r,
                         left=self.prof_r[0], right=self.prof_r[-1])

    def sample(self, t: np.ndarray | float):
        """สุ่มเส้นหางที่ตำแหน่ง ``t`` (0 = โคน, 1 = ปลาย) คืน (x, z, r)"""
        if len(self) < 2:
            return 0.0, 0.0, 0.0
        u = np.linspace(0.0, 1.0, len(self))
        return (np.interp(t, u, self.tail_x),
                np.interp(t, u, self.tail_z),
                np.interp(t, u, self.tail_r))

    def back_at(self, t):
        """ระยะถอยหลังของหางที่ตำแหน่ง t (0..1) เป็นเท่าของความลึกลำตัว"""
        if self.tail_back is None or len(self.tail_back) < 2:
            return np.zeros_like(np.asarray(t, float))
        u = np.linspace(0.0, 1.0, len(self.tail_back))
        return np.interp(t, u, self.tail_back)

    def back_r_at(self, t):
        if self.tail_back_r is None or len(self.tail_back_r) < 2:
            return np.zeros_like(np.asarray(t, float))
        u = np.linspace(0.0, 1.0, len(self.tail_back_r))
        return np.interp(t, u, self.tail_back_r)


def find_top_image(input_dir: Path) -> Path | None:
    """หาไฟล์ภาพมุมบนในโฟลเดอร์ (ดูจาก token ในชื่อไฟล์ ไม่สนตัวพิมพ์)"""
    if not input_dir.is_dir():
        return None
    for p in sorted(input_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem.lower().replace("-", "_").replace(" ", "_")
        tokens = set(stem.split("_"))
        if any(a in tokens or a == stem for a in TOP_ALIASES):
            return p
    return None


def _runs(row: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[brk + 1]))
    ends = np.concatenate((idx[brk], [idx[-1]]))
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= min_len]


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def _tail_side(mask: np.ndarray, cz: float) -> int:
    """หางอยู่ฝั่งไหนของลำตัวในภาพ: +1 = ด้านล่างของภาพ, -1 = ด้านบน

    ตัดสินด้วย "ยาวแต่บาง" — หางยื่นไกลกว่าหัวและแคบกว่ามาก ส่วนหัวสั้นและ
    หนากว่า ใช้อัตราส่วน ความยาว/ความหนา เทียบกันสองฝั่ง
    """
    def score(part: np.ndarray) -> float:
        if not part.any():
            return 0.0
        d = ndimage.distance_transform_edt(part)
        rows = np.flatnonzero(part.any(axis=1))
        thick = float(np.median([d[r].max() for r in rows]))
        return len(rows) / max(thick, 1.0)

    return 1 if score(mask[int(cz):]) >= score(mask[: int(cz)]) else -1


def _trace(mask: np.ndarray, cfg: PipelineConfig
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """ตามหางจากโคนไปปลายในผังมุมบน คืน (rows, cx, half_w) ในพิกเซลของผัง

    ผังถูกพลิกมาแล้วให้หางชี้ลง (แถวมากขึ้น = ไกลจากลำตัวไปทางหาง)
    """
    h, w = mask.shape
    dist = ndimage.distance_transform_edt(mask)
    assert dist is not None
    body_r = float(dist.max())
    if body_r < 4:
        return None

    core = dist > 0.55 * body_r
    if not core.any():
        return None
    cz, cx = ndimage.center_of_mass(core)

    min_run = max(3, int(0.004 * w))
    max_jump = 0.10 * w
    c = float(cx)
    rows, cxs, hws = [], [], []
    for y in range(int(cz), h):
        rr = _runs(mask[y], min_run)
        if not rr:
            continue
        r = min(rr, key=lambda q: abs(0.5 * (q[0] + q[1]) - c))
        if abs(0.5 * (r[0] + r[1]) - c) > max_jump:
            continue
        seg = dist[y, r[0]: r[1] + 1]
        c = float(r[0] + int(np.argmax(seg)))
        rows.append(y)
        cxs.append(c)
        hws.append(float(seg.max()))

    if len(rows) < 12:
        return None

    rows = np.asarray(rows, np.float32)
    cxs = np.asarray(cxs, np.float32)
    hws = np.asarray(hws, np.float32)

    # ตัดช่วงที่ยังเป็นลำตัวออก — เริ่มนับเป็น "หาง" ตั้งแต่จุดที่บางลงกว่า
    # ลำตัวอย่างชัดเจน ไม่งั้นโคนหางจะไปเริ่มกลางหลัง
    thin = np.flatnonzero(hws < 0.45 * body_r)
    if thin.size:
        k = int(thin[0])
        rows, cxs, hws = rows[k:], cxs[k:], hws[k:]
    if len(rows) < 12:
        return None

    # หางเรียวลงทางเดียวจากโคนไปปลาย
    hws = np.minimum.accumulate(hws)

    s = max(float(cfg.tail_path_smooth), 0.0)
    if s > 0 and len(cxs) > 5:
        k = max(3, int(s) | 1)
        cxs = ndimage.uniform_filter1d(cxs, k, mode="nearest")
        hws = ndimage.uniform_filter1d(hws, k, mode="nearest")
    return rows, cxs, hws


def load(input_dir: Path, cfg: PipelineConfig) -> TopPlan | None:
    """อ่านภาพมุมบนแล้วคืนผังพร้อมเส้นทางหาง คืน ``None`` ถ้าใช้ไม่ได้"""
    path = find_top_image(input_dir)
    if path is None:
        return None

    # import ตรงนี้เพื่อไม่ให้เกิด import วน (preprocess ไม่ได้ใช้โมดูลนี้)
    from .preprocess import extract_mask, load_rgba

    log.info("ผังมุมบน <- %s", path.name)
    mask = _largest_blob(extract_mask(load_rgba(path), cfg, "front"))
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        log.warning("ผังมุมบน: แยกตัวละครออกจากพื้นหลังไม่ได้ — ข้ามไป")
        return None
    mask = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    dist = ndimage.distance_transform_edt(mask)
    core = dist > 0.55 * float(dist.max())
    cz, _ = ndimage.center_of_mass(core)
    side = _tail_side(mask, cz)
    work = mask if side == 1 else mask[::-1]

    tr = _trace(work, cfg)
    if tr is None:
        log.warning("ผังมุมบน: หาหางในระนาบ X-Z ไม่เจอ — ข้ามไป")
        return None
    rows, cxs, hws = tr

    # --- แปลงเป็นหน่วยสัดส่วนของลำตัว -------------------------------------
    #
    # ใช้ *ลำตัวหลัก* เป็นไม้บรรทัด ไม่ใช่กรอบทั้งภาพ เพราะกรอบรวมแขนที่กาง
    # ออกกับหางที่สะบัดไปไกล ซึ่งไม่ได้สะท้อนความกว้าง/ลึกของลำตัวเลย
    cys, cxs_core = np.nonzero(core if side == 1 else core[::-1])
    x_mid = 0.5 * (cxs_core.min() + cxs_core.max())
    x_half = max(0.5 * (cxs_core.max() - cxs_core.min()), 1.0)
    z_mid = 0.5 * (cys.min() + cys.max())
    z_half = max(0.5 * (cys.max() - cys.min()), 1.0)

    # แถวที่มากขึ้นในผัง = ไกลออกไปทางหาง = ทิศ -Z (ด้านหลัง) ของตัวละคร
    tail_x = (cxs - x_mid) / x_half
    tail_z = -(rows - z_mid) / z_half
    tail_r = hws / x_half

    # --- โปรไฟล์ความลึกตามแกน X ------------------------------------------
    #
    # ตัดแถวที่เป็นหางออกก่อน ไม่งั้นคอลัมน์ X ที่หางพาดผ่านจะได้ความลึกของหาง
    # มาปนกับของขา ซึ่งเป็นปัญหาเดียวกับที่มุมข้างเจอ
    body = work.copy()
    if len(rows):
        body[int(rows.min()):] = False
    cols = np.flatnonzero(body.any(axis=0))
    prof_x = prof_r = None
    if cols.size >= 8:
        half = np.zeros(cols.size, np.float32)
        for i, c in enumerate(cols):
            zz = np.flatnonzero(body[:, c])
            half[i] = 0.5 * (zz.max() - zz.min() + 1)
        half = ndimage.uniform_filter1d(half, max(3, cols.size // 25 | 1),
                                        mode="nearest")
        prof_x = (cols.astype(np.float32) - x_mid) / x_half
        prof_r = half / max(float(half.max()), 1.0)
        log.info("ผังมุมบน: โปรไฟล์ความลึก %d คอลัมน์ "
                 "(ที่ขอบตัว %.2f, กลางตัว %.2f ของค่าสูงสุด)",
                 cols.size, float(prof_r[0]), float(prof_r[cols.size // 2]))

    # --- ระยะถอยหลังของหาง เทียบกับ "ความลึกลำตัว" ------------------------
    #
    # ใช้ช่วง *จมูกถึงโคนหาง* ในผังเป็นไม้บรรทัด ไม่ใช่ core ที่ threshold
    # มาจากความหนา เพราะ core เป็นแค่สี่เหลี่ยมขนมเปียกปูนเล็ก ๆ กลางลำตัว
    # หน่วยจะเล็กเกินจริงจนระยะหางพองไป 3-4 เท่า (วัดได้ว่าออกมา 4.5 เท่า
    # ของครึ่งลำตัว ซึ่งแปลงเป็น voxel แล้วหลุดออกนอกกริด)
    #
    # ช่วงจมูกถึงโคนหางเป็นของที่ *มุมข้างก็วัดได้* เหมือนกัน สองฝั่งจึงเทียบ
    # กันได้ตรง ๆ โดยไม่ต้องรู้ระยะกล้องหรือ perspective ของภาพมุมบนเลย
    root_row = float(rows[0])
    nose_row = float(np.flatnonzero(work.any(axis=1))[0])
    body_span = max(root_row - nose_row, 1.0)
    tail_back = (rows - root_row) / body_span
    tail_back_r = hws / body_span
    log.info("ผังมุมบน: จมูกถึงโคนหาง %.0f px, หางถอยหลังอีก %.2f เท่าของระยะนั้น",
             body_span, float(tail_back[-1]))

    log.info("ผังมุมบน: หาง %d จุด, ลึก %.2f -> %.2f เท่าของครึ่งลำตัว, "
             "กวาดแกน X %.2f เท่า", len(rows), tail_z[0], tail_z[-1],
             float(np.ptp(tail_x)))
    return TopPlan(mask=work, tail_x=tail_x, tail_z=tail_z, tail_r=tail_r,
                   source=path, prof_x=prof_x, prof_r=prof_r,
                   tail_back=tail_back, tail_back_r=tail_back_r)
