"""ดึง *ผังมือ* (ภาพมือมองจากหลังมือ) มาใช้เซาะร่องระหว่างนิ้ว

ทำไมต้องมีภาพนี้
----------------
มือกินพื้นที่ราว 1/10 ของความสูงตัวละคร ร่องระหว่างนิ้วจึงกว้างแค่ไม่กี่
พิกเซลในภาพมุมหน้า พอย่อลงมาเป็น voxel (ราว 2 px ต่อ voxel) ร่องเหลือ
1-2 voxel แล้วถูกกลบหายไปตอนเบลอ มือที่ได้จาก visual hull จึงเป็น *นวม*
เสมอ ไม่ว่าจะเพิ่มความละเอียดแค่ไหน — ผมกวาดค่าทุกตัวที่เกี่ยวกับเขตมือแล้ว
(``spike_open``, ``detail_blur``, ``digit_ratio``, ``palm_round``,
``detail_hands``) จำนวนก้อนในเขตมือไม่ขยับเลยสักค่าเดียว

นี่เป็นปัญหาเดียวกับนิ้วเท้า ซึ่งโปรเจกต์นี้แก้ไปแล้วด้วย :mod:`src.soleplan`
คือหา *ภาพของอวัยวะนั้นโดยเฉพาะ* มาเป็นผังเชิงสัดส่วน แล้วเอามาตัดเนื้อส่วน
ที่ไม่ใช่มือออก ต่างกันที่เท้าใช้ระนาบ X-Z (มองจากใต้ฝ่าเท้า) ส่วนมือใช้
ระนาบ X-Y (มองจากหลังมือ) เพราะมือห้อยอยู่ข้างลำตัว

ข้อจำกัดที่ยอมรับ
-----------------
เหมือนผังฝ่าเท้า: ใช้เป็น *รูปร่างเชิงสัดส่วน* เท่านั้น ขนาดจริงยังมาจาก
ภาพมุมมาตรฐาน (กว้างเท่าไรจากมุมหน้า ลึกเท่าไรจากกฎความหนาฝ่ามือ) ผังบอก
อย่างเดียวว่า "เนื้อมือกระจายตัวยังไงในระนาบนั้น" ซึ่งเป็นสิ่งเดียวที่ขาด

ผังเป็นมือข้างเดียว อีกข้างใช้ภาพกระจก ซึ่งถูกต้องกับมือเกือบทุกแบบ
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import IMAGE_EXTS, PipelineConfig

log = logging.getLogger(__name__)

#: ชื่อไฟล์ที่ถือว่าเป็นภาพมือ
HAND_ALIASES = ("hand", "hands", "palm", "มือ", "ฝ่ามือ", "หลังมือ")


@dataclass
class HandPlan:
    """ผังมือหนึ่งข้างในกรอบมาตรฐาน

    ``mask[i, j]`` : i เรียงจากข้อมือ (0) ไปปลายนิ้ว (n-1)
                     j เรียงตามแกน X ทิศเดียวกับมุมหน้า
    """

    mask: np.ndarray
    source: Path
    #: -1 = มือข้างซ้ายของภาพมุมหน้า, +1 = ข้างขวา, 0 = ยังไม่ระบุ (ใช้ได้ทั้งคู่)
    side: int = 0

    @property
    def aspect(self) -> float:
        return self.mask.shape[0] / max(self.mask.shape[1], 1)

    def mirrored(self) -> "HandPlan":
        return HandPlan(mask=self.mask[:, ::-1].copy(), source=self.source,
                        side=-self.side)


def find_hand_images(input_dir: Path) -> list[tuple[Path, int]]:
    """หาไฟล์ภาพมือทั้งหมด คืน (path, side) โดย side จากคำว่า left/right

    รับได้ทั้งภาพเดียว (ใช้กระจกเอาอีกข้าง) และสองภาพแยกซ้ายขวา ซึ่งดีกว่า
    เพราะมือสองข้างของตัวละครมักไม่ได้เป็นกระจกกันเป๊ะ
    """
    if not input_dir.is_dir():
        return []
    out: list[tuple[Path, int]] = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem.lower().replace("-", "_").replace(" ", "_")
        tokens = set(stem.split("_"))
        if not any(a in tokens for a in HAND_ALIASES):
            continue
        side = 0
        if tokens & {"left", "l", "ซ้าย"}:
            side = -1
        elif tokens & {"right", "r", "ขวา"}:
            side = 1
        out.append((p, side))
    return out


def _finger_end(mask: np.ndarray) -> int:
    """ปลายไหนคือปลายนิ้ว: ฝั่งที่แตกเป็นหลายก้อน คืน +1 = ท้ายภาพ

    ข้อมือเป็นท่อนตันก้อนเดียวเสมอ ส่วนปลายนิ้วแตกเป็นสี่ห้าก้อน จึงนับ
    จำนวนก้อนเฉลี่ยของสองฝั่งแล้วเทียบกัน เชื่อถือได้กว่าการเดาจากชื่อไฟล์
    หรือสมมติว่าภาพวางหัวตั้งเสมอ
    """
    h = mask.shape[0]
    band = max(2, h // 8)

    def blobs(rows) -> float:
        n = [ndimage.label(mask[r])[1] for r in rows if mask[r].any()]
        return float(np.mean(n)) if n else 0.0

    top = blobs(range(0, band))
    bot = blobs(range(h - band, h))
    return 1 if bot >= top else -1


def _load_one(path: Path, side: int, cfg: PipelineConfig) -> HandPlan | None:
    from .preprocess import extract_mask, load_rgba

    mask = extract_mask(load_rgba(path), cfg, "front")
    lab, n = ndimage.label(mask)
    if n == 0:
        log.warning("ผังมือ: แยกมือออกจากพื้นหลังไม่ได้ — ข้ามไป")
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    mask = lab == int(sizes.argmax())
    ys, xs = np.nonzero(mask)
    mask = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    if _finger_end(mask) < 0:
        mask = mask[::-1]            # พลิกให้ข้อมืออยู่แถว 0 เสมอ

    # จำนวนก้อนมากสุดตามแนวแถว = จำนวนนิ้วที่ผังนี้แยกให้ได้
    h = mask.shape[0]
    fingers = max((ndimage.label(mask[r])[1]
                   for r in range(h // 2, h)), default=0)
    if fingers < 2:
        log.warning("ผังมือ %s แยกนิ้วไม่ได้เลย (มากสุด %d ก้อน) — ข้ามไป",
                    path.name, fingers)
        return None

    log.info("ผังมือ <- %s (%s): %d x %d px, ยาว/กว้าง %.2f, แยกนิ้วได้ %d",
             path.name, {-1: "ซ้าย", 1: "ขวา", 0: "ไม่ระบุข้าง"}[side],
             mask.shape[1], mask.shape[0],
             mask.shape[0] / max(mask.shape[1], 1), fingers)
    return HandPlan(mask=mask, source=path, side=side)


def load(input_dir: Path, cfg: PipelineConfig) -> list[HandPlan]:
    """คืนผังมือที่ใช้ได้ (0, 1 หรือ 2 ใบ) — ใบเดียวจะถูกกระจกให้อีกข้างตอนใช้"""
    out = []
    for path, side in find_hand_images(input_dir):
        plan = _load_one(path, side, cfg)
        if plan is not None:
            out.append(plan)
    if not out:
        return []
    if len(out) == 1 and out[0].side != 0:
        log.info("มีผังมือข้างเดียว — อีกข้างจะใช้ภาพกระจก")
    return out


def hand_spans(ext, nx: int) -> list[tuple[int, int]]:
    """ช่วงแกน X ของมือแต่ละข้าง อ่านจากเขตมือที่ ``extremities`` หามาแล้ว"""
    if ext is None or not ext.hand_rows.any():
        return []
    band = ext.zone & ext.hand_rows[:, None]
    cols = band.any(axis=0)
    if not cols.any():
        return []
    lab, n = ndimage.label(cols)
    out = []
    for i in range(1, n + 1):
        idx = np.flatnonzero(lab == i)
        if idx.size >= max(4, nx // 40):
            out.append((int(idx[0]), int(idx[-1])))
    return out


def _pick(plans: list[HandPlan], side: int) -> np.ndarray:
    """เลือกผังของมือข้างที่ต้องการ ไม่มีก็กระจกจากอีกข้าง"""
    for p in plans:
        if p.side == side:
            return p.mask
    for p in plans:
        if p.side == -side:
            return p.mask[:, ::-1]
    return plans[0].mask if side < 0 else plans[0].mask[:, ::-1]


def apply(occ: np.ndarray, ext, plans, cfg: PipelineConfig) -> int:
    """เซาะร่องระหว่างนิ้วตามผัง — *ตัดอย่างเดียว ไม่เคยเติมเนื้อ*

    ยืดผังให้พอดีกับกรอบของมือแต่ละข้างที่ visual hull หามาได้ (กว้างจาก
    แกน X, ยาวจากช่วงแถวที่เป็นเขตมือ) แล้วลบ voxel ที่ตกนอกผังทิ้ง

    ร่องค่อย ๆ ปิดเมื่อขึ้นไปทางข้อมือ ด้วยการขยายผังตามความสูง เหมือนที่
    ``apply_sole_plan`` ทำกับนิ้วเท้า — นิ้วจริงก็แยกชัดที่ปลายแล้วรวมเป็น
    หลังมือก้อนเดียวที่โคน
    """
    import cv2

    ny, nx, nz = occ.shape
    if not plans:
        return 0
    plan_aspect = float(np.mean([p.aspect for p in plans]))
    rows = np.flatnonzero(ext.hand_rows) if ext is not None else np.array([])
    spans = hand_spans(ext, nx)
    if rows.size < 6 or not spans:
        return 0

    # กรอบของผังต้องเป็น *ตัวมือ* ไม่ใช่ทั้งเขตมือ
    #
    # ext.hand_rows กินตั้งแต่ปลายแขนลงมา ถ้ายืดผังคลุมทั้งช่วงนั้น นิ้วจะไป
    # ตกต่ำกว่าที่ควรและข้อมือกว้าง ๆ ในผังจะไปขยายฝ่ามือ (ลองแล้ว มือผอมลง
    # แต่นิ้วไม่แยก) จึงใช้ *สัดส่วนของผังเอง* เป็นตัวกำหนดความยาว:
    # ความกว้างมืออ่านจาก silhouette ซึ่งแม่น แล้วคูณด้วย ยาว/กว้าง ของผัง
    #
    # ปลายนิ้วอยู่ที่แถวล่างสุดของเขตมือ เพราะมือห้อยลงข้างลำตัว
    y_tip = int(rows.min())
    body_cx = 0.5 * (spans[0][0] + spans[-1][1])
    removed = 0

    for x_lo, x_hi in spans:
        w = x_hi - x_lo + 1
        hh = int(round(plan_aspect * w * cfg.hand_length_scale))
        hh = min(hh, int(rows.max()) - y_tip + 1)
        y_lo, y_hi = y_tip, y_tip + hh - 1
        if w < 6 or hh < 6:
            continue
        side = -1 if 0.5 * (x_lo + x_hi) < body_cx else 1
        pm = _pick(plans, side)
        # ผังเรียงข้อมือ -> ปลายนิ้ว ส่วน occ แถว 0 คือเท้า มือจึงต้องกลับทิศ
        sized = cv2.resize(pm.astype(np.uint8), (w, hh),
                           interpolation=cv2.INTER_NEAREST) > 0
        sized = sized[::-1]          # แถวมากขึ้น = สูงขึ้น = ไปทางข้อมือ

        for i in range(hh):
            y = y_lo + i
            if y >= ny:
                break
            # ปิดร่องทีละนิดเมื่อขึ้นไปทางข้อมือ
            t = i / max(hh - 1, 1)
            r = int(round(cfg.hand_merge_radius * w * t))
            row = sized[i]
            if r > 0:
                row = ndimage.binary_dilation(row, np.ones(2 * r + 1, bool))
            cut = ~row
            if not cut.any():
                continue
            sub = occ[y, x_lo:x_hi + 1]
            hit = sub[cut]
            removed += int(hit.sum())
            sub[cut] = False

    if removed:
        log.info("เซาะร่องนิ้วมือตามผัง %d ข้าง: ปลายนิ้วแถว %d ยาว %d แถว "
                 "(ยาว/กว้างของผัง %.2f) -%d voxel",
                 len(spans), y_tip, hh, plan_aspect, removed)
    return removed
