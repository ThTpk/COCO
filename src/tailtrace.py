"""ลากเส้นทางหางจาก *ภาพด้านหลังภาพเดียว*

ทำไมต้องใช้ภาพด้านหลัง: ในภาพนั้นหางห้อยทับกางเกงอยู่ มันจึงอยู่ *ข้างใน*
silhouette ไม่ใช่ที่ขอบ การดู mask อย่างเดียวมองไม่เห็นหางเลย ต้องแยกด้วยสี

ทำไมไม่ใช้ภาพด้านข้าง: ภาพแต่ละมุมวางหางไว้คนละตำแหน่ง การเอามุมข้างมาบอก
ความลึกแล้วมุมหน้าบอกความกว้าง ทำให้ได้หางที่ไม่ตรงกับภาพไหนเลย ใช้มุมเดียว
เป็นแหล่งอ้างอิงจึงได้หางที่ตรงกับภาพนั้นจริง ๆ

ผลลัพธ์คือเส้นทาง (แถว, ตำแหน่งแกน X, ครึ่งความกว้าง) ในพิกัดพิกเซลของ canvas
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from .config import PipelineConfig
from .preprocess import PreparedView

log = logging.getLogger(__name__)


@dataclass
class TailTrace:
    """เส้นทางหางในพิกัดพิกเซลของ canvas เรียงจากโคน (บน) ไปปลาย (ล่าง)"""

    rows: np.ndarray        # แถวบน canvas
    cx: np.ndarray          # ตำแหน่งแกน X
    half_w: np.ndarray      # ครึ่งความกว้าง

    def __len__(self) -> int:
        return len(self.rows)


def _runs(row: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    i = np.flatnonzero(row)
    if i.size == 0:
        return []
    b = np.flatnonzero(np.diff(i) > 1)
    s = np.concatenate(([i[0]], i[b + 1]))
    e = np.concatenate((i[b], [i[-1]]))
    return [(int(a), int(c)) for a, c in zip(s, e) if c - a + 1 >= min_len]


def _colour_clusters(rgb: np.ndarray, band: np.ndarray, k: int = 2) -> list[np.ndarray]:
    """แบ่งพิกเซลในช่วงล่างของตัวออกเป็น k กลุ่มสี

    ใช้ hue เป็นเวกเตอร์บนวงกลม (cos, sin) แทนตัวเลขดิบ เพราะ hue วนรอบ
    สีแดงที่ค่า 179 กับ 1 อยู่ติดกันจริง แต่ตัวเลขห่างกันสุดขอบ
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.float32) * (np.pi / 90.0)   # 0..180 -> 0..2π
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    feat = np.stack([np.cos(h) * s, np.sin(h) * s, s * 0.5], axis=2)

    pts = feat[band].reshape(-1, 3).astype(np.float32)
    if len(pts) < k * 50:
        return []

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, _ = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)

    out = []
    idx = np.flatnonzero(band.ravel())
    for c in range(k):
        m = np.zeros(band.size, bool)
        m[idx[labels.ravel() == c]] = True
        out.append(m.reshape(band.shape))
    return out


def _track(mask: np.ndarray, body_cx: float, y_top: int, y_bot: int,
           min_run: int, max_jump: float) -> TailTrace | None:
    """ไล่ตามท่อนเดียวกันจากโคนลงไปหาปลาย ด้วยความต่อเนื่องของตำแหน่ง

    ความหนาของหางวัดจาก *distance transform* ไม่ใช่ความยาวของ run ในแถวนั้น

    ทำไมถึงสำคัญมาก: หางสะบัดเป็นแนวเฉียงจนเกือบนอน ช่วงที่มันนอน run ใน
    แถวเดียวคือ *ความยาวของหางช่วงนั้น* ไม่ใช่ความหนา ตัวเลขจึงพุ่งจาก 20 px
    ไปเป็น 200 px แล้วหางที่ปั้นออกมากลายเป็นแผ่นชั้นแบน ๆ ยื่นออกจากสะโพก
    distance transform ให้รัศมีที่ตั้งฉากกับแนวหางเสมอ ไม่ว่าหางจะเอียงเท่าไร
    """
    dist = ndimage.distance_transform_edt(mask)
    assert dist is not None

    start = None
    for y in range(y_top, min(y_bot, y_top + (y_bot - y_top) // 3)):
        rr = _runs(mask[y], min_run)
        if rr:
            start = min(rr, key=lambda r: abs(0.5 * (r[0] + r[1]) - body_cx))
            y_start = y
            break
    if start is None:
        return None

    c = 0.5 * (start[0] + start[1])
    rows, cxs, hws = [], [], []
    for y in range(y_start, y_bot + 1):
        rr = _runs(mask[y], min_run)
        if not rr:
            continue
        r = min(rr, key=lambda q: abs(0.5 * (q[0] + q[1]) - c))
        if abs(0.5 * (r[0] + r[1]) - c) > max_jump:
            continue          # กระโดดไกลเกินไป ไม่ใช่ท่อนเดียวกัน
        # จุดกึ่งกลางคือจุดที่ "ลึกที่สุด" ในท่อน ซึ่งอยู่บนแกนกลางหางจริง ๆ
        seg = dist[y, r[0] : r[1] + 1]
        c = float(r[0] + int(np.argmax(seg)))
        rows.append(y)
        cxs.append(c)
        hws.append(float(seg.max()))

    if len(rows) < 8:
        return None
    return TailTrace(np.array(rows), np.array(cxs, float), np.array(hws, float))


def trace(view: PreparedView, cfg: PipelineConfig) -> TailTrace | None:
    """ลากเส้นทางหางจากภาพด้านหลัง คืน None ถ้าแยกหางออกจากเสื้อผ้าไม่ได้"""
    rgb, m = view.rgb, view.mask
    ys = np.flatnonzero(m.any(axis=1))
    if ys.size == 0:
        return None
    y0, y1 = int(ys[0]), int(ys[-1])
    H = y1 - y0
    S = m.shape[0]

    band = m.copy()
    band[: y0 + int(cfg.tail_band_top * H)] = False
    if band.sum() < 500:
        return None

    # แกนกลางลำตัววัดจากช่วงเท้า ซึ่งไม่โดนหางหรือแขนดึง
    foot = m[max(y1 - int(0.08 * H), 0) : y1 + 1]
    body_cx = float(np.where(foot)[1].mean()) if foot.any() else S / 2

    min_run = max(3, S // 180)
    max_jump = cfg.tail_max_jump * S
    kernel_o = np.ones((5, 5), bool)
    kernel_c = np.ones((9, 9), bool)

    # ความกว้างลำตัวในช่วงที่มองหาหาง ใช้เป็นไม้บรรทัดตัดสินว่า "แคบ" แค่ไหน
    body_w = float(np.median([m[y].sum() for y in range(
        int(ys[0] + cfg.tail_band_top * H), y1 + 1) if m[y].any()] or [S]))

    # เสื้อผ้ากินพื้นที่ส่วนใหญ่ของท่อนล่างเสมอ หางเป็นส่วนน้อย
    # จึงลองกลุ่มที่เล็กกว่าก่อน แล้วค่อยถอยไปกลุ่มใหญ่ถ้ากลุ่มเล็กใช้ไม่ได้
    clusters = sorted(_colour_clusters(rgb, band), key=lambda c: int(c.sum()))

    best, best_score = None, -1.0
    for ci, cluster in enumerate(clusters):
        clean = ndimage.binary_closing(
            ndimage.binary_opening(cluster, kernel_o), kernel_c)
        lab, n = ndimage.label(clean)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        blob = lab == int(np.argmax(sizes)) + 1

        tr = _track(blob, body_cx, int(ys[0] + cfg.tail_band_top * H), y1,
                    min_run, max_jump)
        if tr is None:
            continue

        # หางที่ถูกต้องคือท่อนที่ ยาว + แคบเทียบลำตัว + *เรียวลงไปทางปลาย*
        #
        # สองข้อแรกอย่างเดียวไม่พอ: ขากางเกงข้างหนึ่งก็ยาวและแคบพอ ๆ กัน
        # (วัดได้ว่าคะแนนชนะหางด้วยซ้ำ แล้วโมเดลได้ "หาง" ที่เป็นขาซ้ำอีกขา)
        # สิ่งที่หางมีแต่ขาไม่มีคือ *ความเรียว* — หางเล็กลงเรื่อย ๆ จนเป็นปลาย
        # แหลม ส่วนขาจบลงด้วยเท้าซึ่งกว้างกว่าน่อง เพิ่มโบนัสให้ท่อนที่ส่าย
        # ออกข้างด้วย แต่ให้น้ำหนักน้อย เพราะหางที่ห้อยตรง ๆ ก็มีเยอะ
        span = (tr.rows[-1] - tr.rows[0]) / max(H, 1)
        width = float(np.median(tr.half_w) * 2)
        narrow = 1.0 - min(1.0, width / max(0.5 * body_w, 1.0))

        q = max(2, len(tr) // 4)
        w_base = float(np.median(tr.half_w[:q]))
        w_tip = float(np.median(tr.half_w[-q:]))
        taper = float(np.clip(1.0 - w_tip / max(w_base, 1e-6), 0.0, 1.0))

        drift = float(np.abs(np.diff(tr.cx)).sum())
        sweep = float(np.clip(drift / max(tr.rows[-1] - tr.rows[0], 1), 0.0, 1.0))

        score = span * max(narrow, 0.05) * (0.25 + 0.75 * taper) \
            * (0.85 + 0.15 * sweep)
        log.debug("กลุ่มสีขนาด %d px: พาด %.0f%% กว้างกลาง %.0f (ลำตัว %.0f) "
                  "เรียว %.2f ส่าย %.2f คะแนน %.3f",
                  int(cluster.sum()), 100 * span, width, body_w, taper, sweep, score)
        if score > best_score:
            best, best_score = tr, score

    if best is None or best_score < cfg.tail_trace_min_score:
        log.info("แยกหางจากภาพด้านหลังไม่ได้ (คะแนนดีสุด %.3f) — ข้ามการสร้างหาง",
                 max(best_score, 0.0))
        return None

    # หางเรียวลงทางเดียวเสมอ: บังคับให้รัศมีไม่โตขึ้นเมื่อไล่จากโคนไปปลาย
    #
    # ช่วงที่หางลอดใต้ชายเสื้อหรือถูกขาบัง จะวัดได้บางกว่าความจริงมาก
    # (วัดได้ 3 px ทั้งที่ท่อนถัดลงไปหนา 38 px) ถ้าไม่แก้ หางจะคอดเป็นคอขวด
    # ตรงโคนแล้วหลุดออกจากตัว การไล่ค่าสูงสุดจากปลายขึ้นมาเติมช่วงที่ถูกบัง
    # ด้วยความหนาของท่อนที่อยู่ถัดไป ซึ่งเป็นค่าที่ใกล้ความจริงที่สุดที่มี
    best.half_w = np.maximum.accumulate(best.half_w[::-1])[::-1]

    # เกลาเส้นทางตามแนวตั้ง ขอบที่หยักจากการ segment จะกลายเป็นหางเป็นปุ่ม ๆ
    s = max(float(cfg.tail_path_smooth), 0.0)
    if s > 0 and len(best) > 5:
        best.cx = ndimage.gaussian_filter1d(best.cx, s, mode="nearest")
        best.half_w = ndimage.gaussian_filter1d(best.half_w, s, mode="nearest")

    q = max(2, len(best) // 4)
    log.info("ลากหางจากภาพด้านหลังได้ %d แถว: %.0f%% -> %.0f%% ของความสูง, "
             "หนาโคน %.0f -> ปลาย %.0f px",
             len(best), 100 * (y1 - best.rows[0]) / H,
             100 * (y1 - best.rows[-1]) / H,
             2 * float(np.median(best.half_w[:q])),
             2 * float(np.median(best.half_w[-q:])))
    return best
