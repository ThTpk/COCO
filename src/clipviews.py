r"""ประกอบภาพสี่มุมหลักจากคลิป ให้พร้อมส่งเข้า pipeline หลัก

ทำไมต้องประกอบ ไม่ใช่หยิบเฟรมเดียวมาใช้
----------------------------------------
pipeline หลักต้องการภาพ 4 มุมที่ *เห็นตัวครบทั้งตัว* เพราะมันย่อทุกมุมให้สูง
เท่ากันก่อนเฉือน voxel  แต่คลิปกล้องวนรอบตัวมักเข้าใกล้จนตัวล้นเฟรมในบางช่วง
(คลิปตัวอย่าง: มุมหน้าเห็นครบ แต่ทุกมุมตั้งแต่ 30 ถึง 310 องศา ขาหลุดขอบล่าง)
หยิบเฟรมมุมหลังมาใช้ตรง ๆ จะได้ตัวละครครึ่งท่อน แล้ว pipeline จะย่อครึ่งท่อน
นั้นให้สูงเท่ากับมุมหน้าทั้งตัว — ผิดทั้งมาตราส่วนและรูปทรง

เมื่อรู้พารามิเตอร์กล้องของทุกเฟรมแล้ว (ดู ``turntable.py``) เราแปลงทุกเฟรม
ไปอยู่บน canvas กลางใบเดียวกันได้ ซึ่งทำให้สองอย่างเป็นไปได้
  1. ทุกมุมมีมาตราส่วนตรงกันเป๊ะ ไม่ต้องให้ pipeline เดาจากความสูงของ mask
  2. ส่วนที่หลุดเฟรมไป เติมจากเฟรม *มุมใกล้เคียง* ที่เห็นส่วนนั้นได้

ลำดับการเติม เรียงจากน่าเชื่อถือมากไปน้อย
  1. เฟรมที่มุมใกล้ที่สุดที่เห็นจุดนั้น (ต่างกันไม่เกิน span องศา)
  2. ภาพสะท้อนของมุมคู่ตรงข้าม — มุมหลังยืมขาจากมุมหน้าที่พลิกซ้ายขวา
     ใช้ได้เพราะร่างกายสมมาตรซ้ายขวา และเงาของขาเมื่อมองจากหลังแทบจะเป็น
     ภาพสะท้อนของเงาเมื่อมองจากหน้า
ทุกพิกเซลที่เติมมาจากแหล่งที่สองขึ้นไปถูกบันทึกไว้ในภาพ debug ด้วย จะได้รู้ว่า
ส่วนไหนของผลลัพธ์มาจากข้อมูลจริงและส่วนไหนมาจากการอนุมาน
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .turntable import Camera, ClipFrame, MaskStats

log = logging.getLogger(__name__)

# มุมที่ pipeline หลักต้องการ (องศา วัดจากมุมหน้า ทวนไปทาง +X)
CARDINALS: dict[str, float] = {
    "front": 0.0,
    "side_r": 90.0,
    "back": 180.0,
    "side": 270.0,
}

# มุมที่จะเอาภาพสะท้อนมาเติมให้ ถ้าเฟรมมุมใกล้เคียงยังเติมไม่เต็ม
MIRROR_OF: dict[str, str] = {
    "back": "front",
    "side": "side_r",
    "side_r": "side",
    "front": "back",
}


@dataclass
class BuiltView:
    name: str
    rgba: np.ndarray          # (S, S, 4) uint8
    source_index: int         # เฟรมหลักที่ใช้
    borrowed: np.ndarray      # (S, S) bool — พิกเซลที่ยืมมาจากแหล่งอื่น
    sources: list[str]
    known: np.ndarray         # (S, S) bool — พิกเซลที่มีเฟรมไหนสักเฟรมให้ข้อมูล


def _affine(cam: Camera, size: int, margin: float) -> np.ndarray:
    """เมทริกซ์แปลงจากพิกเซลของเฟรม ไปเป็นพิกเซลบน canvas กลาง

    canvas กลางนิยามให้ตรงกับที่ ``preprocess.normalize_view`` สร้าง: ตัวละคร
    สูงเต็ม canvas โดยเว้นขอบบนล่างด้านละ margin ฝ่าเท้าอยู่ที่ขอบล่างของ
    พื้นที่นั้น และแกนหมุนอยู่กึ่งกลางภาพ
    """
    k = size * (1.0 - 2.0 * margin) / cam.scale
    ground_row = size * (1.0 - margin)
    return np.float32([[k, 0, size / 2 - k * cam.cx],
                       [0, k, ground_row - k * cam.cy]])


def _warp(frame: ClipFrame, cam: Camera, size: int, margin: float
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """คืน (rgb บน canvas, mask บน canvas, พื้นที่ที่เฟรมนี้ให้ข้อมูลได้)"""
    M = _affine(cam, size, margin)
    # ย่อใช้ INTER_AREA (ไม่เกิด alias) ขยายใช้ INTER_LINEAR (ไม่เป็นบล็อก)
    interp = cv2.INTER_AREA if M[0, 0] < 1.0 else cv2.INTER_LINEAR
    rgb = cv2.warpAffine(frame.rgb, M, (size, size), flags=interp)
    mask = cv2.warpAffine(frame.mask.astype(np.uint8) * 255, M, (size, size),
                          flags=interp) > 127
    covered = cv2.warpAffine(np.full(frame.mask.shape, 255, np.uint8), M,
                             (size, size), flags=cv2.INTER_NEAREST) > 127
    frame.unload()
    return rgb, mask, covered


def _angle_gap(a: float, b: float) -> float:
    """ระยะเชิงมุมที่สั้นที่สุดระหว่างสองมุม (องศา)"""
    return abs(float(np.degrees(np.angle(np.exp(1j * (a - b))))))


def _pick_base(cams: list[Camera], stats: list[MaskStats], target: float,
               crop_penalty: float = 8.0) -> int:
    """เลือกเฟรมหลักของมุมหนึ่ง

    ยึดมุมเป็นหลัก ไม่ยึดความครบของตัว — เฟรมที่มุมเพี้ยนไป 25 องศาให้ความลึก
    ลำตัวผิดไปทั้งใบ ซึ่งแก้ทีหลังไม่ได้ ส่วนขาที่หลุดเฟรมเติมทีหลังได้
    ค่าปรับเล็ก ๆ มีไว้เลือกเฟรมที่ครบกว่าเมื่อมุมพอ ๆ กันเท่านั้น
    """
    tgt = np.radians(target)
    best, best_cost = 0, 1e9
    for i, (c, s) in enumerate(zip(cams, stats)):
        cost = _angle_gap(c.theta, tgt)
        cost += crop_penalty * (bool(s.touch_top) + bool(s.touch_bottom))
        if cost < best_cost:
            best, best_cost = i, cost
    return best


def build_views(frames: list[ClipFrame], cams: list[Camera],
                stats: list[MaskStats], cfg: PipelineConfig,
                span: float = 40.0) -> dict[str, BuiltView]:
    """ประกอบภาพสี่มุมหลักจากคลิป"""
    S = cfg.canvas
    out: dict[str, BuiltView] = {}

    for name, target in CARDINALS.items():
        tgt = np.radians(target)
        base_i = _pick_base(cams, stats, target)
        order = sorted(range(len(cams)), key=lambda i: _angle_gap(cams[i].theta, tgt))
        order = [i for i in order if _angle_gap(cams[i].theta, tgt) <= span]
        order = [base_i] + [i for i in order if i != base_i]

        rgb = np.zeros((S, S, 3), np.uint8)
        alpha = np.zeros((S, S), bool)
        known = np.zeros((S, S), bool)
        used: list[str] = []
        borrowed = np.zeros((S, S), bool)

        for n, i in enumerate(order):
            w_rgb, w_mask, w_cov = _warp(frames[i], cams[i], S, cfg.margin)
            new = w_cov & ~known
            if not new.any():
                continue
            # เฉพาะตอนเติมจากเฟรมอื่น: ถ้าเพิ่มเนื้อไม่ถึงครึ่งเปอร์เซ็นต์ของภาพ
            # ก็ไม่ต้องบันทึกว่าใช้ เพื่อให้รายงานอ่านง่าย
            rgb[new] = w_rgb[new]
            alpha |= new & w_mask
            if n > 0:
                borrowed |= new
            known |= w_cov
            if n == 0 or (new & w_mask).sum() > 0.0005 * S * S:
                used.append(f"#{cams[i].index}({np.degrees(cams[i].theta) % 360:.0f}°)")
            if known.all():
                break

        out[name] = BuiltView(name, np.dstack([rgb, alpha.astype(np.uint8) * 255]),
                              cams[base_i].index, borrowed, used, known)
        log.info("มุม %-6s: เฟรมหลัก #%d (%.0f°) เติมจาก %d เฟรม — "
                 "ยังว่างอีก %.1f%% ของภาพ",
                 name, cams[base_i].index,
                 np.degrees(cams[base_i].theta) % 360, max(0, len(used) - 1),
                 100 * float((~known).mean()))

    _mirror_fill(out)
    return out


def _mirror_fill(views: dict[str, BuiltView]) -> None:
    """เติมส่วนที่ยังว่างด้วยภาพสะท้อนของมุมคู่ตรงข้าม

    มุมหลังของคลิปนี้ไม่มีเฟรมไหนเห็นขาเลยสักเฟรม (กล้องเข้าใกล้ที่สุดพอดี
    ตอนวนไปด้านหลัง) เฟรมมุมใกล้เคียงที่เห็นขาอยู่ห่างออกไปเป็นร้อยองศา
    ซึ่งไกลเกินกว่าจะเอามาแปะ  เงาขาเมื่อมองจากหลังเกือบเป็นภาพสะท้อนของเงา
    เมื่อมองจากหน้า การยืมจากมุมหน้าที่พลิกซ้ายขวาจึงใกล้ความจริงกว่ามาก

    เป็นการอนุมาน ไม่ใช่ข้อมูลจริง — จึงถูกนับรวมใน borrowed ด้วย
    """
    # เติมมุมที่ขาดน้อยก่อน เพื่อให้มุมที่ขาดมากได้ของที่เติมครบแล้วไปใช้
    order = sorted(views, key=lambda n: float((~views[n].known).mean()))
    for name in order:
        view = views[name]
        known = view.known
        if known.all():
            continue
        src = views.get(MIRROR_OF.get(name, ""))
        if src is None:
            continue
        gap = ~known
        flipped = np.ascontiguousarray(src.rgba[:, ::-1])
        flipped = _align_at_seam(view.rgba[:, :, 3] > 127, flipped, gap)
        view.rgba[gap] = flipped[gap]
        view.borrowed |= gap
        view.known = np.ones_like(known)
        view.sources.append(f"สะท้อนจากมุม {src.name}")
        log.info("มุม %-6s: เติม %.1f%% ของภาพจากภาพสะท้อนของมุม %s",
                 name, 100 * float(gap.mean()), src.name)


def _align_at_seam(host: np.ndarray, src: np.ndarray, gap: np.ndarray,
                   band: int = 14) -> np.ndarray:
    """ขยับ/ย่อภาพที่จะเอามาเติม ให้ต่อกับของเดิมตรงรอยต่อพอดี

    ของที่ยืมมาเป็นคนละเฟรม ซึ่งตัวละครมีสัดส่วนเพี้ยนไปจากเฟรมหลักได้ถึง 10%
    (คลิปที่ AI สร้างไม่ได้คงรูปตลอด) ถ้าแปะดิบ ๆ เงาจะสะดุดเป็นขั้นตรงรอยต่อ
    แล้วรูปทรงที่เฉือนได้จะมีบ่าหรือคอดโผล่กลางน่องแบบที่ไม่มีในตัวละครจริง

    เทียบ "กว้างแค่ไหน อยู่ตรงไหน" ของแถวเหนือรอยต่อ แล้วยืดตามแนวนอน
    ให้เท่ากัน — แนวตั้งไม่ต้องแตะ เพราะทุกมุมถูกวางให้ฝ่าเท้าอยู่แถวเดียวกัน
    ตั้งแต่ตอน warp แล้ว
    """
    rows = np.flatnonzero(gap.any(axis=1))
    if rows.size == 0:
        return src
    seam = int(rows[0])
    lo = max(0, seam - band)
    if seam - lo < 4:
        return src

    def span(mask: np.ndarray) -> tuple[float, float] | None:
        xs = np.flatnonzero(mask[lo:seam].any(axis=0))
        if xs.size < 4:
            return None
        return float(0.5 * (xs[0] + xs[-1])), float(0.5 * (xs[-1] - xs[0]) + 1)

    a = span(host)
    b = span(src[:, :, 3] > 127)
    if a is None or b is None:
        return src
    k = float(np.clip(a[1] / max(b[1], 1.0), 0.8, 1.25))
    dx = a[0] - b[0] * k
    M = np.float32([[k, 0, dx], [0, 1, 0]])
    out = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]),
                         flags=cv2.INTER_LINEAR)
    log.debug("จัดรอยต่อที่แถว %d: ยืดแนวนอน %.3f เลื่อน %.1f px", seam, k, dx)
    return out


def write_views(views: dict[str, BuiltView], out_dir: Path) -> dict[str, Path]:
    """เขียนภาพสี่มุมเป็น PNG มี alpha ให้ pipeline หลักอ่านต่อ"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, v in views.items():
        p = out_dir / f"{name}.png"
        bgra = v.rgba[:, :, [2, 1, 0, 3]]
        ok, buf = cv2.imencode(".png", bgra)
        if not ok:
            raise RuntimeError(f"เขียนภาพมุม {name} ไม่สำเร็จ")
        buf.tofile(str(p))
        paths[name] = p
        log.info("เขียนมุม %-6s -> %s (จาก %s)", name, p, ", ".join(v.sources))
    return paths


def debug_sheet(views: dict[str, BuiltView], path: Path) -> None:
    """ภาพตรวจงาน: สี่มุมเรียงกัน ส่วนที่ยืมมาจากแหล่งอื่นถูกย้อมสี"""
    tiles = []
    for name in ("front", "side_r", "back", "side"):
        v = views[name]
        img = v.rgba[:, :, :3].copy()
        a = v.rgba[:, :, 3] > 127
        img[~a] = (24, 24, 32)
        tint = v.borrowed & a
        img[tint] = (0.45 * img[tint] + np.array([140, 0, 60])).astype(np.uint8)
        img = cv2.resize(img, (420, 420))
        cv2.putText(img, f"{name}  #{v.source_index}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, "pink = borrowed", (10, 410),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 190), 1)
        tiles.append(img)
    sheet = np.hstack(tiles)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", sheet[:, :, ::-1])
    if ok:
        buf.tofile(str(path))
        log.info("เขียนภาพตรวจงานสี่มุม -> %s", path)
