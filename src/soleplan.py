"""ดึง *ผังฝ่าเท้า* (รูปเท้ามองจากใต้ฝ่าเท้า) จากภาพมุมล่าง

ทำไมต้องมีภาพนี้
----------------
มุมหน้า/หลัง/ข้าง บอกตำแหน่งนิ้วเท้าไม่ได้เลย นิ้วเท้าทอดไปข้างหน้าและ
กางออกเป็นรูปพัดอยู่บน *ระนาบ X-Z* ซึ่งเป็นระนาบเดียวที่ไม่มีมุมไหนใน
สามมุมมาตรฐานมองเห็น มุมหน้าเห็นร่องระหว่างนิ้วแค่ 2 ซม. ล่างสุดของเท้า
เท่านั้น (สูงกว่านั้นนิ้วรวมเป็นหลังเท้าก้อนเดียวตามกายวิภาค) เท้าที่ได้
จาก visual hull ล้วน ๆ จึงเป็นก้อนรองเท้าบูทเสมอ ไม่ว่าจะเพิ่มความละเอียด
เท่าไรก็ตาม เพราะข้อมูลไม่มีตั้งแต่ต้น ไม่ใช่เพราะสุ่มหยาบ

ภาพมองจากใต้ฝ่าเท้าให้ระนาบนั้นมาตรง ๆ: จำนวนนิ้ว ตำแหน่ง ความกว้าง
มุมที่กางออก และรูปส้น ครบในภาพเดียว

ข้อจำกัดที่ยอมรับ
-----------------
ภาพแบบนี้เป็นภาพ perspective แรง ๆ (ฝ่าเท้าใหญ่เว่อร์เพราะอยู่ใกล้กล้อง)
เราจึงใช้มันเป็น *รูปร่างเชิงสัดส่วน* เท่านั้น แล้วไปยืดให้พอดีกับความกว้าง
เท้าที่วัดจากมุมหน้า และความยาวเท้าที่วัดจากมุมข้าง ซึ่งสองค่านั้นแม่นอยู่แล้ว
สิ่งที่ภาพนี้ให้คือ "นิ้วอยู่ตรงไหนบนฝ่าเท้า" ซึ่งเป็นสิ่งเดียวที่ขาด
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import IMAGE_EXTS, PipelineConfig

log = logging.getLogger(__name__)

#: ชื่อไฟล์ที่ถือว่าเป็นภาพมองจากใต้ฝ่าเท้า
SOLE_ALIASES = ("down", "sole", "soles", "bottom", "under", "underside",
                "ฝ่าเท้า", "ใต้เท้า")


@dataclass
class FootPlan:
    """ผังเท้าหนึ่งข้างในกรอบมาตรฐาน

    `mask[i, j]` : i เรียงจากส้น (0) ไปปลายนิ้ว (n-1) = ทิศ +Z
                   j เรียงตามแกน X ทิศเดียวกับมุมหน้า
    `side`       : -1 = เท้าที่อยู่ทางซ้ายของภาพมุมหน้า, +1 = ทางขวา
    """

    mask: np.ndarray
    side: int

    @property
    def aspect(self) -> float:
        """ยาว/กว้าง ของผัง"""
        return self.mask.shape[0] / max(self.mask.shape[1], 1)


def find_sole_image(input_dir: Path) -> Path | None:
    for p in sorted(input_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem.lower().replace("-", "_").replace(" ", "_")
        tokens = set(stem.split("_"))
        if any(a in tokens or a == stem for a in SOLE_ALIASES):
            return p
    return None


def _runs(row: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[brk + 1]))
    ends = np.concatenate((idx[brk], [idx[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_len]


def _toe_end_is_top(foot: np.ndarray) -> bool:
    """ปลายนิ้วอยู่ด้านบนของภาพหรือไม่

    ส้นเท้าเป็นแผ่นมนก้อนเดียว ส่วนปลายนิ้วแตกเป็นหลายท่อน (นิ้วกับเล็บ)
    นับจำนวนท่อนต่อแถวที่ปลายทั้งสองข้าง ฝั่งที่แตกท่อนมากกว่าคือฝั่งนิ้ว
    เชื่อถือได้กว่าการดูว่าฝั่งไหนแคบกว่า เพราะส้นของสัตว์หลายแบบก็เรียว
    """
    h, w = foot.shape
    band = max(3, h // 5)
    min_len = max(2, w // 40)
    top = sum(len(_runs(foot[y], min_len)) for y in range(band))
    bot = sum(len(_runs(foot[y], min_len)) for y in range(h - band, h))
    return top >= bot


def extract(path: Path, cfg: PipelineConfig,
            mask_fn=None) -> list[FootPlan]:
    """ดึงผังเท้าซ้าย/ขวาจากภาพมุมล่าง คืนลิสต์ว่างถ้าแยกไม่ได้"""
    from .preprocess import extract_mask, load_rgba

    rgba = load_rgba(path)
    mask = (mask_fn or extract_mask)(rgba, cfg, "sole")
    h, w = mask.shape
    min_run = max(4, w // 25)

    # เส้นแบ่งซ้าย/ขวา: กลางช่องว่างในแถวที่เห็นเท้าสองข้างแยกกันพอดี
    #
    # แถวพวกนี้คือช่วงส้นเท้า ซึ่งอยู่ล่างสุดและไม่มีอะไรอื่นมาปน
    mids = [0.5 * (rr[0][1] + rr[1][0])
            for rr in (_runs(mask[y], min_run) for y in range(h - 1, -1, -1))
            if len(rr) == 2]
    if len(mids) < h // 20:
        log.warning("ภาพมุมล่าง %s: แยกเท้าสองข้างออกจากกันไม่ได้ — ข้ามผังฝ่าเท้า",
                    path.name)
        return []
    split = int(round(float(np.median(mids))))

    plans: list[FootPlan] = []
    for side, sub in ((-1, mask[:, :split]), (1, mask[:, split:])):
        # ความกว้างของ silhouette ฝั่งนี้ในแต่ละแถว
        #
        # เท้าเป็นส่วนที่กว้างที่สุดของฝั่งนี้มาก ๆ (ฝ่าเท้าอยู่ใกล้กล้อง)
        # ส่วนขากางเกงที่ต่อขึ้นไปแคบกว่าราวครึ่งหนึ่ง จึงตัดออกได้ด้วย
        # การไล่ขึ้นจากแถวที่กว้างที่สุดแล้วหยุดเมื่อความกว้างหล่นฮวบ
        # ถ้าไม่ตัด ชิ้นส่วนขาจะติดมาที่ปลายนิ้วแล้วผังเท้าเพี้ยนทั้งใบ
        span = np.zeros(h, np.int32)
        for y in range(h):
            rr = _runs(sub[y], max(4, min_run // 3))
            if rr:
                span[y] = rr[-1][1] - rr[0][0] + 1
        if not span.any():
            continue
        y_peak = int(np.argmax(span))
        y_bottom = int(np.flatnonzero(span)[-1])
        limit = cfg.sole_span_keep * float(span[y_peak])
        y_top = y_peak
        while y_top > 0 and span[y_top - 1] >= limit:
            y_top -= 1

        band = sub[y_top : y_bottom + 1]
        lab, n = ndimage.label(band)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        foot = lab == int(np.argmax(sizes)) + 1
        ys, xs = np.where(foot)
        if ys.size < 500:
            continue
        foot = foot[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

        # ให้แถวแรกเป็นส้น แถวสุดท้ายเป็นปลายนิ้ว (ทิศ +Z)
        if _toe_end_is_top(foot):
            foot = foot[::-1]
        plans.append(FootPlan(np.ascontiguousarray(foot), side))

    if len(plans) != 2:
        log.warning("ภาพมุมล่าง %s: ดึงผังเท้าได้ %d ข้าง (ต้องได้ 2) — ข้ามไป",
                    path.name, len(plans))
        return []

    # เท้าสองข้างเป็นภาพกระจกของกันและกัน ใช้ใบที่สะอาดกว่าแล้วพลิกใส่อีกข้าง
    #
    # ภาพมุมล่างมักมีของอื่นพาดทับฝ่าเท้าข้างใดข้างหนึ่ง (ในภาพนี้คือปลายหาง
    # ที่ทอดมาทับเท้าขวา) ซึ่งเชื่อมติดกับเท้าใน mask จนแยกด้วย morphology
    # ไม่ได้ — opening ที่แรงพอจะตัดหางก็กินนิ้วหายไปหมดเหมือนกัน
    # ของที่พาดทับมีแต่ *เพิ่ม* พื้นที่ ใบที่เล็กกว่าจึงเป็นใบที่สะอาดกว่าเสมอ
    if cfg.sole_mirror:
        areas = [int(p.mask.sum()) for p in plans]
        clean = int(np.argmin(areas))
        if areas[1 - clean] > (1.0 + cfg.sole_mirror_slack) * areas[clean]:
            src = plans[clean]
            other = plans[1 - clean]
            log.info("ผังเท้าฝั่ง%s ใหญ่กว่าอีกข้าง %.0f%% (น่าจะมีของพาดทับ) "
                     "— ใช้ผังอีกข้างพลิกกระจกแทน",
                     "ซ้ายของภาพ" if other.side < 0 else "ขวาของภาพ",
                     100 * (areas[1 - clean] / max(areas[clean], 1) - 1))
            plans[1 - clean] = FootPlan(
                np.ascontiguousarray(src.mask[:, ::-1]), other.side)

    for p in plans:
        n_toes = max(len(_runs(p.mask[y], max(2, p.mask.shape[1] // 40)))
                     for y in range(int(0.75 * p.mask.shape[0]), p.mask.shape[0]))
        log.info("ผังเท้าฝั่ง%s: %d x %d px, ยาว/กว้าง %.2f, นิ้ว %d",
                 "ซ้ายของภาพ" if p.side < 0 else "ขวาของภาพ",
                 p.mask.shape[1], p.mask.shape[0], p.aspect, n_toes)
    return plans


def load(input_dir: Path, cfg: PipelineConfig) -> list[FootPlan]:
    """หาไฟล์ภาพมุมล่างในโฟลเดอร์ input แล้วดึงผังเท้า"""
    if not cfg.use_sole_plan:
        return []
    path = find_sole_image(input_dir)
    if path is None:
        return []
    log.info("ผังฝ่าเท้า <- %s", path.name)
    try:
        return extract(path, cfg)
    except (ValueError, RuntimeError) as exc:
        log.warning("อ่านผังฝ่าเท้าจาก %s ไม่ได้: %s", path.name, exc)
        return []


def resample(plan: FootPlan, nx: int, nz: int) -> np.ndarray:
    """ย่อ/ขยายผังเป็นกริด (nx, nz) สำหรับวางลงใน occupancy

    คืนอาเรย์ (nx, nz) ให้ index แรกเป็นแกน X และที่สองเป็นแกน Z
    ตรงกับรูปแบบของ `occ[y]`
    """
    import cv2

    src = plan.mask.astype(np.float32)          # (z, x)
    dst = cv2.resize(src, (max(nx, 1), max(nz, 1)),
                     interpolation=cv2.INTER_AREA)
    return (dst >= 0.5).T                        # -> (x, z)
