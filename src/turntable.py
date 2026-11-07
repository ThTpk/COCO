r"""อ่านคลิปกล้องวนรอบตัวละคร แล้วประเมินพารามิเตอร์กล้องของทุกเฟรม

ต่างจาก pipeline หลักตรงไหน
--------------------------
pipeline หลัก (``src/cli.py``) รับภาพนิ่ง 4 มุมที่ตั้งฉากกันพอดี แล้วปั้น
หน้าตัดทีละแถวจากความกว้าง (มุมหน้า/หลัง) กับความลึก (มุมข้าง) — เป็นการเดา
รูปหน้าตัดด้วยวงรี เพราะมีข้อมูลแค่สองแกน

คลิปให้อะไรมากกว่านั้น: กล้องวนรอบตัวจนครบ 360 องศา เราจึงมี silhouette
หลายสิบมุม พอจะ *เฉือน* voxel ตามมุมจริงได้ทั้งวง (visual hull ของจริง)
หน้าตัดจึงออกมาตามรูปตัวละครเอง ไม่ต้องสมมติว่าเป็นวงรี

ราคาที่ต้องจ่ายคือ ต้องรู้ว่าแต่ละเฟรม "กล้องอยู่มุมไหน ห่างแค่ไหน สูงแค่ไหน"
ซึ่งคลิปไม่ได้บอกมา โมดูลนี้คือส่วนที่ประเมินสิ่งนั้น

ระบบพิกัด
---------
ใช้ชุดเดียวกับ pipeline หลัก: +X = ขวาของภาพมุมหน้า, +Y = ขึ้น, +Z = เข้าหา
กล้องมุมหน้า ตัวละครสูง 1.0 หน่วยโลก และฝ่าเท้าอยู่ที่ Y = 0

เฟรมที่ i มีกล้องอยู่มุม theta (0 = มุมหน้า, เพิ่มทวนไปทาง +X) ระยะห่างแปลง
เป็นมาตราส่วน scale (พิกเซลต่อหนึ่งหน่วยโลก) และแกนหมุนของ turntable ตกลงบน
ภาพที่คอลัมน์ cx ส่วนระดับพื้น (Y = 0) ตกที่แถว cy  การฉายจุดโลกลงภาพจึงเป็น

    u = scale * (X*cos(theta) - Z*sin(theta)) + cx
    v = cy - scale * Y

คือสมมติว่าเป็นภาพฉายขนาน (orthographic) ที่มุม theta  กล้องจริงเป็น
perspective แต่ระยะกล้องไกลกว่าตัวละครหลายเท่า ความคลาดจึงอยู่ในระดับไม่กี่
เปอร์เซ็นต์ของความกว้างตัว ซึ่งขั้น carve เผื่อไว้ด้วย tolerance อยู่แล้ว

ทำไมไม่ใช้ structure-from-motion
--------------------------------
คลิปนี้สร้างด้วย AI ฉากหลัง (ซุ้มโค้ง เสา คบไฟ) *ไม่คงที่* ระหว่างเฟรม
เสาหายไปเฉย ๆ ก็มี SfM ที่พึ่งจุดนิ่งในฉากจึงพังตั้งแต่ต้น เราเลยใช้ข้อสมมติ
ที่แข็งแรงกว่าสำหรับคลิปแบบนี้แทน คือ "กล้องวนรอบแกนตั้งแกนเดียว"
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# โครงสร้างข้อมูล
# ---------------------------------------------------------------------------


@dataclass
class ClipFrame:
    """เฟรมเดียวจากคลิป พร้อม mask ที่แยกพื้นหลังแล้ว"""

    index: int                 # เลขเฟรมในคลิปต้นฉบับ
    rgb_path: Path
    mask_path: Path
    _rgb: np.ndarray | None = field(default=None, repr=False)
    _mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            img = cv2.imread(str(self.rgb_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"อ่านเฟรมไม่ได้: {self.rgb_path}")
            self._rgb = img[:, :, ::-1].copy()
        return self._rgb

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            m = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(f"อ่าน mask ไม่ได้: {self.mask_path}")
            self._mask = m > 127
        return self._mask

    def unload(self) -> None:
        """คืนหน่วยความจำ — เฟรมทั้งคลิปใหญ่เกินกว่าจะถือไว้พร้อมกันหมด"""
        self._rgb = None
        self._mask = None

    @property
    def shape(self) -> tuple[int, int]:
        m = self.mask
        return m.shape[0], m.shape[1]


@dataclass
class Camera:
    """พารามิเตอร์กล้องของเฟรมหนึ่ง (ดูสมการการฉายที่หัวไฟล์)"""

    index: int
    theta: float      # เรเดียน 0 = มุมหน้า
    scale: float      # พิกเซลต่อ 1 หน่วยโลก (ตัวละครสูง 1.0)
    cx: float         # คอลัมน์ของแกนหมุนบนภาพ
    cy: float         # แถวของระดับพื้นบนภาพ
    full_body: bool = False   # เฟรมนี้เห็นตัวครบทั้งตัวหรือไม่
    # ตัวคูณแก้เฉพาะเฟรม จากขั้น refine — ไม่ใช่เรขาคณิตของกล้องจริง แต่เป็น
    # การชดเชย "ตัวละครเปลี่ยนสัดส่วนระหว่างคลิป" ซึ่งเกิดกับคลิปที่ AI สร้าง
    # (ตัวกว้างขึ้น 12% ระหว่างต้นคลิปกับท้ายคลิปในคลิปตัวอย่าง)
    ku: float = 1.0   # ตัวคูณมาตราส่วนแนวนอน
    kv: float = 1.0   # ตัวคูณมาตราส่วนแนวตั้ง
    # --- เลนส์ -------------------------------------------------------------
    # กล้องในคลิปนี้อยู่ใกล้ตัวละครมาก (ห่างราวหนึ่งถึงสองเท่าของความสูงตัว)
    # ภาพฉายขนานจึงใช้ไม่ได้: ด้านที่หันเข้ากล้องถูกขยายกว่าด้านไกลหลายสิบ
    # เปอร์เซ็นต์ ถ้าไม่คิดส่วนนี้ เงาจากคนละมุมจะไม่มีทางตรงกัน แล้วการเฉือน
    # จะกินหัวกับแขนขาหายไปหมด เหลือแต่แกนกลางลำตัว
    q: float = 0.0    # ส่วนกลับของระยะกล้อง (1/D) 0 = ภาพฉายขนาน
    h: float = 0.5    # ความสูงของแนวระดับสายตาเทียบความสูงตัวละคร

    @property
    def su(self) -> float:
        return self.scale * self.ku

    @property
    def sv(self) -> float:
        return self.scale * self.kv

    def project(self, pts: np.ndarray) -> np.ndarray:
        """ฉายจุดโลก (N, 3) -> พิกเซล (N, 2) เรียงเป็น (u, v)

        แบบ perspective รอบแกนตั้ง: จุดที่อยู่ใกล้กล้อง (r มาก) ถูกขยายด้วย
        1/(1 - r*q)  ที่ q = 0 สมการนี้ลดรูปกลับไปเป็นภาพฉายขนานพอดี
        """
        c, s = np.cos(self.theta), np.sin(self.theta)
        a = pts[:, 0] * c - pts[:, 2] * s       # ระยะด้านข้างจากแกนหมุน
        r = pts[:, 0] * s + pts[:, 2] * c       # ความลึกเข้าหากล้อง
        g = 1.0 / np.maximum(1.0 - r * self.q, 0.2)
        u = self.su * a * g + self.cx
        v = self.cy - self.sv * (self.h + (pts[:, 1] - self.h) * g)
        return np.stack([u, v], axis=1)

    def with_lens(self, focal: float, height: float) -> "Camera":
        """สำเนาที่ใส่ค่าเลนส์: ระยะกล้อง = focal / มาตราส่วนของเฟรมนี้"""
        return Camera(self.index, self.theta, self.scale, self.cx, self.cy,
                      self.full_body, self.ku, self.kv,
                      q=0.0 if focal <= 0 else self.scale / focal, h=height)

    @property
    def direction(self) -> np.ndarray:
        """ทิศจากตัวละครไปหากล้อง (หน่วยเวกเตอร์)"""
        return np.array([np.sin(self.theta), 0.0, np.cos(self.theta)])


# ---------------------------------------------------------------------------
# ดึงเฟรม + แยกพื้นหลัง
# ---------------------------------------------------------------------------


def extract_frames(video: Path, cache: Path, step: int = 1,
                   jpeg_quality: int = 95) -> list[int]:
    """ดึงเฟรมจากคลิปลงแคช คืนรายการเลขเฟรมที่มี

    เก็บเป็น JPEG เพราะคลิปต้นทางก็ถูกบีบด้วย h264 มาแล้ว การเก็บต่อเป็น PNG
    ได้แค่ไฟล์ใหญ่ขึ้นสิบเท่าโดยไม่ได้รายละเอียดกลับมา
    """
    cache.mkdir(parents=True, exist_ok=True)
    existing = sorted(int(p.stem[1:]) for p in cache.glob("f*.jpg"))
    if existing:
        log.info("ใช้เฟรมในแคชเดิม %d เฟรม (%s)", len(existing), cache)
        return [i for i in existing if i % step == 0]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"เปิดคลิปไม่ได้: {video}")
    idx, kept = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(cache / f"f{idx:04d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        kept.append(idx)
        idx += 1
    cap.release()
    log.info("ดึงเฟรมจาก %s ได้ %d เฟรม -> %s", video.name, len(kept), cache)
    return [i for i in kept if i % step == 0]


def segment_frames(indices: list[int], cache: Path, model: str) -> list[ClipFrame]:
    """แยกพื้นหลังทุกเฟรมด้วย rembg (แคชผลไว้ เพราะช้าที่สุดในกระบวนการ)"""
    mask_dir = cache / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    todo = [i for i in indices
            if not (mask_dir / f"m{i:04d}.png").exists()]
    if todo:
        try:
            from rembg import new_session, remove
        except ImportError as exc:  # pragma: no cover - ขึ้นกับเครื่อง
            raise RuntimeError(
                "ต้องติดตั้ง rembg ก่อนใช้โหมดคลิป:\n"
                r"    .venv\Scripts\python.exe -m pip install rembg onnxruntime"
            ) from exc
        session = new_session(model)
        log.info("แยกพื้นหลัง %d เฟรมด้วย %s ...", len(todo), model)
        for n, i in enumerate(todo, 1):
            img = cv2.imread(str(cache / f"f{i:04d}.jpg"), cv2.IMREAD_COLOR)
            out = remove(img[:, :, ::-1], session=session)
            alpha = np.asarray(out)[:, :, 3]
            cv2.imwrite(str(mask_dir / f"m{i:04d}.png"),
                        _clean_mask(alpha > 127).astype(np.uint8) * 255)
            if n % 20 == 0 or n == len(todo):
                log.info("  แยกแล้ว %d/%d เฟรม", n, len(todo))

    return [ClipFrame(i, cache / f"f{i:04d}.jpg", mask_dir / f"m{i:04d}.png")
            for i in indices]


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """เก็บเฉพาะก้อนใหญ่สุด + ถมรูเล็ก ๆ

    rembg มักทิ้งเศษจุดตามคบไฟหรือขอบเสา ถ้าปล่อยไว้ voxel ที่ควรถูกเฉือนทิ้ง
    จะรอดมาเป็นก้อนลอยรอบตัวละคร
    """
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == biggest).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


# ---------------------------------------------------------------------------
# วัด silhouette
# ---------------------------------------------------------------------------


@dataclass
class MaskStats:
    index: int
    y0: int
    y1: int
    x0: int
    x1: int
    touch_top: bool
    touch_bottom: bool
    core_cx: float      # กึ่งกลางลำตัวช่วงอก-สะโพก (ใช้เดาแกนหมุน)
    area: int


def measure(frames: list[ClipFrame]) -> list[MaskStats]:
    out = []
    for f in frames:
        m = f.mask
        h, w = m.shape
        ys, xs = np.where(m)
        if ys.size == 0:
            raise ValueError(f"เฟรม {f.index} ไม่มีตัวละครใน mask เลย")
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

        # กึ่งกลาง "ลำตัว" ไม่ใช่กึ่งกลาง silhouette ทั้งหมด
        #
        # หางยื่นออกไปข้างเดียวและแกว่งไปมาตลอดคลิป ถ้าใช้ centroid ทั้งก้อน
        # แกนหมุนที่เดาได้จะถูกหางลากไปมาหลายสิบพิกเซล  แถวช่วงกลางตัว
        # (อก-สะโพก) ไม่มีหางกับเท้าปน และอยู่ใกล้แกนหมุนที่สุด
        band0 = y0 + int(0.28 * (y1 - y0))
        band1 = y0 + int(0.55 * (y1 - y0))
        band = m[band0:band1 + 1]
        bxs = np.where(band.any(axis=0))[0]
        core = 0.5 * (bxs.min() + bxs.max()) if bxs.size else 0.5 * (x0 + x1)

        out.append(MaskStats(f.index, y0, y1, x0, x1,
                             touch_top=y0 <= 1, touch_bottom=y1 >= h - 2,
                             core_cx=float(core), area=int(m.sum())))
        f.unload()
    return out


# ---------------------------------------------------------------------------
# มาตราส่วนกับระดับพื้น: ต่อโซ่ด้วยการจับคู่จุดเด่นระหว่างเฟรมติดกัน
# ---------------------------------------------------------------------------


def _pair_similarity(prev: tuple, cur: tuple) -> tuple[float, float, float] | None:
    """คืน (อัตราขยาย, การเลื่อนแนวตั้ง, การไหลแนวนอนของจุดศูนย์กลาง)

    เอาสองค่าแรกไปต่อโซ่ ส่วนค่าที่สามใช้ได้แค่เครื่องหมาย เพราะการหมุน
    ตัวละครทำให้จุดเด่นบนผิวด้านที่หันเข้ากล้องกวาดไปด้านข้าง การเลื่อน
    แนวนอนที่ fit ได้จึงปนการหมุนมาเต็ม ๆ แล้วสะสมเป็น drift ของแกนหมุน
    ถ้าเอาไปต่อโซ่  ส่วนแนวตั้งกับมาตราส่วนไม่โดนผลนั้น เพราะการหมุนรอบ
    แกนตั้งไม่ย้ายอะไรขึ้นลง
    """
    (kp1, des1), (kp2, des2) = prev, cur
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    bf = cv2.BFMatcher()
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=4000, confidence=0.995)
    if M is None or inliers is None or int(inliers.sum()) < 10:
        return None

    scale = float(np.hypot(M[0, 0], M[0, 1]))
    if not (0.8 < scale < 1.25):
        return None

    keep = inliers.ravel().astype(bool)
    c = src[keep].mean(axis=0)
    moved = M @ np.array([c[0], c[1], 1.0])
    return scale, float(M[1, 2]), float(moved[0] - c[0])


def _features(frame: ClipFrame, sift) -> tuple:
    """จุดเด่นเฉพาะในตัวละคร — ฉากหลังของคลิป AI ไม่นิ่ง ใช้อ้างอิงไม่ได้"""
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    m = cv2.erode(frame.mask.astype(np.uint8), np.ones((9, 9), np.uint8))
    return sift.detectAndCompute(gray, m)


def _chain_scale_and_ground(
    frames: list[ClipFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ต่อโซ่อัตราขยายและระดับพื้นจากเฟรมแรกไปเฟรมสุดท้าย

    คืน (log ของมาตราส่วนเทียบเฟรมแรก, แถวของจุดอ้างอิงแนวตั้ง, การไหลแนวนอน)
    สองตัวแรกเป็นค่าสัมพัทธ์ ต้องเอาไปตรึงกับเฟรมที่เห็นตัวครบอีกที

    ตัวที่สามคือการเลื่อนแนวนอนเฉลี่ยของจุดเด่นต่อเฟรม ซึ่งใช้วัดระยะทางไม่ได้
    (ปนการหมุนมาเต็ม ๆ) แต่ *เครื่องหมาย* ของมันบอกทิศที่กล้องวนได้: ผิวด้าน
    ที่หันเข้ากล้องย่อมกวาดสวนทางกับทิศที่กล้องเคลื่อนไปเสมอ
    """
    sift = cv2.SIFT_create(nfeatures=3000)
    log_s = np.zeros(len(frames))
    ty = np.zeros(len(frames))
    flow = np.zeros(len(frames))

    prev = _features(frames[0], sift)
    frames[0].unload()
    misses = 0
    for i in range(1, len(frames)):
        cur = _features(frames[i], sift)
        frames[i].unload()
        est = _pair_similarity(prev, cur)
        if est is None:
            misses += 1
            log_s[i], ty[i] = log_s[i - 1], ty[i - 1]   # ถือค่าเดิมไว้ก่อน
        else:
            s, dy, dx = est
            # จุดอ้างอิงบนภาพ: y_i = s * y_{i-1} + dy  (ยืดรอบจุดกำเนิดภาพ)
            log_s[i] = log_s[i - 1] + np.log(s)
            ty[i] = s * ty[i - 1] + dy
            flow[i] = dx
        prev = cur
    if misses:
        log.warning("จับคู่จุดเด่นไม่ได้ %d คู่เฟรม — ใช้ค่าของเฟรมก่อนหน้าแทน", misses)
    return log_s, ty, flow


def estimate_scale_ground(
    frames: list[ClipFrame], stats: list[MaskStats],
) -> tuple[np.ndarray, np.ndarray, list[bool], np.ndarray]:
    """ประเมินมาตราส่วนกับแถวระดับพื้นของทุกเฟรม

    โซ่จากการจับคู่จุดเด่นให้ *รูปร่าง* ของเส้นกราฟมาถูก (เฟรมไหนกล้องเข้าใกล้
    เฟรมไหนถอยออก) แต่ค่าที่ได้เป็นค่าสัมพัทธ์ และการคูณต่อกัน 240 ครั้งย่อม
    สะสมความคลาด  ต้องตรึงกับสิ่งที่ *วัดได้จากภาพตรง ๆ*

    ของที่วัดได้มีสองอย่าง และคนละเฟรมก็ให้คนละอย่าง
      * เฟรมที่เห็นเท้า   -> แถวล่างสุดของ mask คือระดับพื้น
      * เฟรมที่เห็นหัว    -> แถวบนสุดคือจุดที่ Y = 1 (ตัวละครสูง 1 หน่วย)
    เฟรมที่เห็นทั้งคู่ให้ทั้งสองอย่าง ในคลิปที่กล้องเข้าใกล้จนตัวล้นเฟรม
    เฟรมส่วนใหญ่ยังให้อย่างใดอย่างหนึ่งเสมอ จึงยังตรึงได้ทั้งคลิป

    ตัวแก้ของโซ่เป็นพหุนามดีกรีต่ำสองเส้น (หนึ่งเส้นสำหรับมาตราส่วนใน log-space
    อีกเส้นสำหรับระดับพื้นในหน่วยโลก) fit พร้อมกันด้วย least squares แบบทน
    outlier เพราะแถวบนสุดของหัวขยับได้จริงตามมุม (ปาก สันหนาม) และ mask
    บางเฟรมก็กินเงาหรือคบไฟติดมาด้วย
    """
    from scipy.optimize import least_squares

    log_s_chain, ty_chain, flow = _chain_scale_and_ground(frames)
    n = len(frames)
    idx = np.arange(n, dtype=float)
    t = idx / max(1.0, n - 1.0)        # ปรับช่วงเป็น 0..1 ให้ fit เสถียร

    feet = np.array([not s.touch_bottom for s in stats])
    head = np.array([not s.touch_top for s in stats])
    y_bottom = np.array([s.y1 for s in stats], float)
    y_top = np.array([s.y0 for s in stats], float)
    if feet.sum() < 4 or head.sum() < 4:
        raise ValueError(
            "คลิปนี้ไม่มีเฟรมที่เห็นหัวหรือเท้าพอจะตรึงมาตราส่วนได้ — "
            "ลองใช้คลิปที่กล้องถอยออกมาบ้าง")

    full = [bool(f and h) for f, h in zip(feet, head)]
    whole = np.asarray(full)
    if whole.sum() < 4:
        raise ValueError("ไม่มีเฟรมไหนเห็นตัวละครครบทั้งตัวเลย — ตรึงมาตราส่วนไม่ได้")

    # --- มาตราส่วน: ตรึงกับเฟรมที่เห็นครบตัวเท่านั้น -------------------------
    # ยอดหัวของเฟรมที่เห็นครึ่งตัวใช้ไม่ได้ เพราะจุดสูงสุดของ silhouette ย้าย
    # ไปมาตามมุม (ปากยื่นหน้า สันหนามยื่นหลัง) ความคลาดระดับ 2-3% ของแต่ละ
    # เฟรมนั้นมากพอจะลากเส้นแก้ทั้งเส้นให้เพี้ยน ถ้าปล่อยให้มันมีเสียงข้างมาก
    h_meas = (y_bottom - y_top + 1)[whole]

    def resid_s(p):
        return (np.log(h_meas) - log_s_chain[whole] - np.polyval(p, t[whole])) * 100

    sol_s = least_squares(resid_s, np.zeros(2), loss="soft_l1", f_scale=1.0)
    scale = np.exp(log_s_chain + np.polyval(sol_s.x, t))

    # --- ระดับพื้น: วัดจากภาพโดยตรงทุกเฟรม ไม่ต้องพึ่งโซ่ ----------------------
    # เมื่อรู้มาตราส่วนแล้ว ระดับพื้นอ่านได้จากขอบ mask ที่ยังไม่หลุดเฟรม
    #   เห็นเท้า -> ระดับพื้น = แถวล่างสุด
    #   เห็นหัว  -> ระดับพื้น = แถวบนสุด + มาตราส่วน (ยอดหัวคือ Y = 1)
    # ในคลิปนี้ช่วงกลางที่กล้องเข้าใกล้จนเท้าหลุดเฟรม เหลือแต่ยอดหัวให้ยึด
    # ซึ่งก็เป็นตัวเลือกที่ถูกต้องอยู่แล้ว เพราะช่วงนั้นเห็นแต่ท่อนบน
    # การวางท่อนบนให้ตรงกันจึงสำคัญกว่าการวางขาที่ไม่มีใครเห็น
    g_obs = np.where(feet, y_bottom, y_top + scale)
    w_obs = np.where(feet, 3.0, 1.0)

    # เกลาตามเวลา: ยอดหัวขยับตามมุมอยู่บ้าง และ mask บางเฟรมกินเงาติดมา
    k = 15
    pad_g = np.pad(g_obs * w_obs, k // 2, mode="edge")
    pad_w = np.pad(w_obs, k // 2, mode="edge")
    ker = np.ones(k)
    ground = (np.convolve(pad_g, ker, "valid") / np.convolve(pad_w, ker, "valid"))

    ok = np.asarray(full)
    if ok.any():
        err_h = np.abs(scale[ok] - (y_bottom[ok] - y_top[ok] + 1)) / scale[ok]
        err_g = np.abs(ground[ok] - y_bottom[ok])
        log.info("ตรึงมาตราส่วน: เห็นเท้า %d เฟรม เห็นหัว %d เฟรม — "
                 "บนเฟรมที่เห็นครบ ความสูงคลาด %.1f%% (สูงสุด %.1f%%), "
                 "ระดับพื้นคลาด %.1f px (สูงสุด %.1f px)",
                 int(feet.sum()), int(head.sum()),
                 100 * err_h.mean(), 100 * err_h.max(),
                 err_g.mean(), err_g.max())
    return scale, ground, full, flow


# ---------------------------------------------------------------------------
# มุมกล้อง
#
# คลิปไม่ได้บอกว่าเฟรมไหนกล้องอยู่มุมไหน เราจึงต้องอ่านจากภาพเอง สามคำถาม
#   1. กล้องวนเร็วแค่ไหน (องศาต่อเฟรม)  -> หาจากคาบ: เฟรมไหนเงาซ้ำกับเฟรมต้น
#   2. วนไปทางไหน (ซ้ายหรือขวา)          -> จากทิศที่จุดเด่นบนผิวกวาดไป
#   3. เฟรมไหนคือมุมหน้าพอดี             -> จากความสมมาตรซ้ายขวา + ทิศที่หันหน้า
# ---------------------------------------------------------------------------


def _norm_silhouette(frame: ClipFrame, st: MaskStats, size: int = 72) -> np.ndarray:
    """ย่อเงาให้สูงเท่ากันและจัดกึ่งกลางที่ลำตัว เพื่อเทียบรูปร่างข้ามเฟรม"""
    m = frame.mask
    h = st.y1 - st.y0 + 1
    k = size / h
    small = cv2.resize(m.astype(np.uint8), None, fx=k, fy=k,
                       interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.uint8)
    cx = int(round(st.core_cx * k))
    cy0 = int(round(st.y0 * k))
    half = size // 2
    sx0, sx1 = max(0, cx - half), min(small.shape[1], cx + half)
    dx0 = half - (cx - sx0)
    rows = small[cy0:cy0 + size]
    out[:rows.shape[0], dx0:dx0 + (sx1 - sx0)] = rows[:, sx0:sx1]
    return out > 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def estimate_period(frames: list[ClipFrame], stats: list[MaskStats],
                    full: list[bool]) -> float:
    """หาว่ากล้องวนครบรอบในกี่เฟรม คืนองศาต่อเฟรม (ยังไม่มีเครื่องหมาย)

    วิธี: เงาของมุมเดิมย่อมเหมือนกันที่สุด เทียบเงาเฟรมอ้างอิง (ช่วงต้นคลิป)
    กับทุกเฟรมในช่วงท้าย เฟรมที่เหมือนที่สุดคือเฟรมที่กล้องวนกลับมาที่เดิม

    ใช้เฉพาะเฟรมที่เห็นตัวครบ เพราะต้องย่อให้สูงเท่ากันก่อนเทียบ
    """
    idx = [i for i, ok in enumerate(full) if ok]
    if len(idx) < 8:
        raise ValueError("เฟรมที่เห็นตัวครบมีน้อยเกินกว่าจะหาคาบการวนได้")

    head = [i for i in idx if i < len(frames) * 0.3]
    tail = [i for i in idx if i > len(frames) * 0.6]
    if not head or not tail:
        raise ValueError("ต้องมีเฟรมที่เห็นตัวครบทั้งช่วงต้นและช่วงท้ายคลิป")

    ref_i = head[len(head) // 2]
    ref = _norm_silhouette(frames[ref_i], stats[ref_i])
    frames[ref_i].unload()

    best, best_j = -1.0, tail[-1]
    for j in tail:
        score = _iou(ref, _norm_silhouette(frames[j], stats[j]))
        frames[j].unload()
        if score > best:
            best, best_j = score, j

    span = stats[best_j].index - stats[ref_i].index
    if span < 30:
        raise ValueError("หาคาบการวนไม่ได้ — เฟรมที่เหมือนกันอยู่ใกล้กันเกินไป")
    log.info("กล้องวนครบรอบใน %d เฟรม (เฟรม %d ซ้ำกับเฟรม %d, IoU %.2f)",
             span, stats[best_j].index, stats[ref_i].index, best)
    return float(span)


def _symmetry(frame: ClipFrame, st: MaskStats) -> float:
    """ความสมมาตรซ้ายขวาของท่อนบน — สูงสุดที่มุมหน้ากับมุมหลัง

    ใช้เฉพาะช่วงหัวถึงสะโพก เพราะหางกับขาหลังทำลายความสมมาตรตลอดเวลา
    """
    m = frame.mask
    h = st.y1 - st.y0 + 1
    band = m[st.y0:st.y0 + max(4, int(0.55 * h))]
    c = int(round(st.core_cx))
    half = min(c, band.shape[1] - c - 1)
    if half < 8:
        return 0.0
    left = band[:, c - half:c]
    right = band[:, c + 1:c + 1 + half][:, ::-1]
    return _iou(left, right)


def estimate_phase(frames: list[ClipFrame], stats: list[MaskStats],
                   period: float, flow: np.ndarray) -> tuple[int, float, int]:
    """หาเฟรมที่เป็นมุมหน้า กับทิศที่กล้องวน

    คืน (เลขเฟรมมุมหน้า, ทิศ +1/-1, ตำแหน่งในลิสต์ของเฟรมนั้น)

    ทิศการวน: ผิวที่หันเข้ากล้องย่อมกวาดสวนทางกับการเคลื่อนของกล้องเสมอ
    (จุดที่ Z > 0 มี du/dtheta = -scale*Z ที่ theta = 0) จุดเด่นจึงไหลไปทางซ้าย
    เมื่อกล้องวนไปทาง +X

    เฟสเริ่มต้น: เฟรมที่เงาสมมาตรซ้ายขวาที่สุดคือมุมหน้าหรือมุมหลัง แยกสองอย่าง
    นี้ด้วยการดูว่าเฟรมที่ห่างออกไปอีก 90 องศา "หันหน้า" ไปทางไหน
    """
    from .preprocess import detect_side_facing

    n = len(frames)
    sym = np.array([_symmetry(frames[i], stats[i]) or 0.0 for i in range(n)])
    for f in frames:
        f.unload()

    # เกลาแล้วหาจุดสูงสุด — ความสมมาตรดิบแกว่งตามคุณภาพ mask
    k = 7
    sym_s = np.convolve(np.pad(sym, k // 2, mode="edge"), np.ones(k) / k, "valid")
    cand = int(np.argmax(sym_s))

    # ระยะเป็น "ตำแหน่งในลิสต์" ไม่ใช่เลขเฟรม เผื่อกรณีที่สุ่มเฟรมมาไม่ครบทุกใบ
    per_pos = period / max(1e-6, (stats[-1].index - stats[0].index)) * (n - 1)
    step = max(2, int(round(per_pos / 4)))

    around = flow[max(1, cand - step):cand + step]
    sign = -1 if (around.size and float(np.mean(around)) > 0) else 1
    log.info("ทิศการวน: จุดเด่นไหลเฉลี่ย %.2f px/เฟรม -> กล้องวนไปทาง %s",
             float(np.mean(around)) if around.size else 0.0,
             "+X" if sign > 0 else "-X")

    probe = cand + step if cand + step < n else cand - step
    away = (1 if probe > cand else -1) * sign * 90.0     # มุมของ probe ถ้า cand = มุมหน้า
    facing, conf = detect_side_facing(
        frames[probe].mask, (stats[probe].y0, stats[probe].y1,
                             stats[probe].x0, stats[probe].x1))
    frames[probe].unload()

    # ที่ theta = +90 กล้องอยู่ +X ด้านหน้าตัวละคร (+Z) จึงตกไปทางซ้ายของภาพ
    expect = -1 if away > 0 else 1
    is_front = (facing == expect)
    if conf < 0.05:
        log.warning("ดูไม่ออกว่ามุมข้างหันหน้าไปทางไหน (ความมั่นใจ %.2f) — "
                    "เดาว่าเฟรม %d คือมุมหน้า", conf, stats[cand].index)
        is_front = True
    log.info("เฟรม %d สมมาตรที่สุด (%.2f) จึงเป็นมุม%s "
             "(เฟรม %d หันหน้าไปทาง%s ความมั่นใจ %.2f)",
             stats[cand].index, sym_s[cand], "หน้า" if is_front else "หลัง",
             stats[probe].index, "ขวา" if facing > 0 else "ซ้าย", conf)

    if is_front:
        return int(stats[cand].index), float(sign), cand
    # เฟรมที่สมมาตรที่สุดเป็นมุมหลัง -> มุมหน้าอยู่ห่างออกไปครึ่งรอบ
    half = int(round(per_pos / 2))
    front_pos = cand + half if cand + half < n else cand - half
    return int(stats[front_pos].index), float(sign), front_pos


# ---------------------------------------------------------------------------


class AngleMap:
    """แผนที่ "เฟรมที่เท่าไร -> กวาดไปกี่องศาแล้ว" ที่บังคับให้เพิ่มขึ้นเสมอ

    กล้องในคลิปที่คนถ่ายหรือ AI สร้างไม่ได้วนด้วยความเร็วคงที่ — คลิปตัวอย่าง
    วนช้ามากในช่วงแรก (1 องศา/เฟรม) แล้วเร่งเป็น 3 องศา/เฟรมช่วงกลาง
    ถ้าใช้ความเร็วคงที่ มุมของเฟรมกลางคลิปจะเพี้ยนไปได้ถึง 40 องศา ซึ่งทำให้
    การเฉือนพังทั้งใบ

    เก็บเป็นจุดควบคุมไม่กี่จุด (ทุก ๆ 90 องศา) แล้วลากเส้นโค้งที่ไม่แกว่ง
    (PCHIP) ระหว่างจุด  จุดควบคุมหาได้จากการลองเฉือนแล้ววัดความสอดคล้องของเงา
    ดู ``clipcarve.fit_angles``
    """

    def __init__(self, knots_i: np.ndarray, period: float):
        from scipy.interpolate import PchipInterpolator

        self.knots_i = np.asarray(knots_i, float)
        self.period = float(period)
        n = len(self.knots_i) - 1
        step = 360.0 / n
        deg = np.arange(n + 1) * step
        # ต่อจุดควบคุมวนรอบไว้หัวท้าย เพื่อให้ความเร็วนอกช่วงหนึ่งรอบต่อเนื่อง
        xi = np.concatenate([self.knots_i[-2:-1] - period, self.knots_i,
                             self.knots_i[1:2] + period])
        yi = np.concatenate([deg[-2:-1] - 360.0, deg, deg[1:2] + 360.0])
        self._f = PchipInterpolator(xi, yi, extrapolate=True)

    def degrees(self, index: float | np.ndarray):
        return self._f(index)

    def radians(self, index: float | np.ndarray):
        return np.radians(self._f(index))

    @staticmethod
    def linear(first: int, period: float, n_knots: int = 5) -> "AngleMap":
        step = period / (n_knots - 1)
        return AngleMap(first + np.arange(n_knots) * step, period)

    def resample(self, n_knots: int) -> "AngleMap":
        """สร้างแผนที่หน้าตาเดิมแต่มีจุดควบคุมถี่ขึ้น (เพื่อดัดละเอียดต่อ)"""
        lo, hi = self.knots_i[0], self.knots_i[-1]
        dense = np.linspace(lo, hi, 2000)
        want = np.linspace(0.0, 360.0, n_knots)
        pts = np.interp(want, self.degrees(dense), dense)
        pts[0], pts[-1] = lo, hi
        return AngleMap(pts, self.period)

    def moved(self, k: int, index: float) -> "AngleMap":
        """สำเนาที่ย้ายจุดควบคุมที่ k ไปอยู่ที่เฟรม index"""
        pts = self.knots_i.copy()
        pts[k] = index
        return AngleMap(pts, self.period)

    def describe(self) -> str:
        rate = np.diff(np.asarray(self.knots_i))
        step = 360.0 / (len(self.knots_i) - 1)
        return ", ".join(f"{step * k:.0f}°@เฟรม {self.knots_i[k]:.0f}"
                         for k in range(len(self.knots_i))) + \
            " (ความเร็ว " + ", ".join(f"{step / r:.2f}" for r in rate) + " องศา/เฟรม)"


def build_cameras(stats: list[MaskStats], scale: np.ndarray, ground: np.ndarray,
                  full: list[bool], angles: AngleMap, sign: float,
                  phase: float = 0.0, cx_smooth: float = 9.0) -> list[Camera]:
    """ประกอบพารามิเตอร์กล้องทั้งหมดเข้าด้วยกัน

    มุมมาจาก AngleMap คูณทิศการวน ส่วน cx มาจากกึ่งกลางลำตัวที่วัดได้
    เกลาตามเวลาเพื่อลดการแกว่งจากรูปร่างที่เปลี่ยนไปตามมุม
    """
    cx = np.array([s.core_cx for s in stats], float)
    if cx_smooth > 0 and len(cx) > 5:
        k = max(3, int(cx_smooth) | 1)
        pad = np.pad(cx, k // 2, mode="edge")
        ker = np.ones(k) / k
        cx = np.convolve(pad, ker, mode="valid")

    idx = np.array([s.index for s in stats], float)
    theta = sign * angles.radians(idx) + phase
    return [Camera(index=int(stats[i].index), theta=float(theta[i]),
                   scale=float(scale[i]), cx=float(cx[i]), cy=float(ground[i]),
                   full_body=bool(full[i]))
            for i in range(len(stats))]


@dataclass
class Solution:
    """ผลการประเมินกล้องทั้งคลิป"""

    cams: list[Camera]
    stats: list[MaskStats]
    angles: AngleMap
    sign: float
    front_index: int
    period: float


def solve(frames: list[ClipFrame], n_knots: int = 5) -> Solution:
    """ประเมินกล้องของทุกเฟรมในคลิป — เรียกอันเดียวจบ

    มุมที่ได้ตรงนี้ยังสมมติว่ากล้องวนด้วยความเร็วคงที่ ซึ่งมักไม่จริง
    เรียก ``clipcarve.fit_angles`` ต่อเพื่อดัดจังหวะการวนให้ตรงกับคลิป
    """
    stats = measure(frames)
    scale, ground, full, flow = estimate_scale_ground(frames, stats)
    period = estimate_period(frames, stats, full)
    front, sign, _ = estimate_phase(frames, stats, period, flow)
    angles = AngleMap.linear(front, period, n_knots)
    cams = build_cameras(stats, scale, ground, full, angles, sign)
    log.info("มุมกล้องตั้งต้น: มุมหน้าอยู่ที่เฟรม %d, วนครบรอบใน %.0f เฟรม "
             "(%.3f องศา/เฟรม)", front, period, 360.0 / period)
    return Solution(cams, stats, angles, float(sign), int(front), float(period))


def pick_cardinal(cams: list[Camera]) -> dict[str, Camera]:
    """เลือกเฟรมที่ใกล้สี่มุมหลักที่สุด — ใช้ทำภาพตรวจงานกับตั้งชื่อ"""
    want = {"front": 0.0, "side_r": 90.0, "back": 180.0, "side": 270.0}
    out: dict[str, Camera] = {}
    for name, deg in want.items():
        target = np.radians(deg)
        best = min(cams, key=lambda c: abs(
            np.angle(np.exp(1j * (c.theta - target)))))
        out[name] = best
    return out


def save_solution(sol: "Solution", path: Path) -> None:
    """เก็บผลการประเมินกล้องไว้ใช้ซ้ำ — ขั้นนี้ช้าที่สุดในกระบวนการ"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "knots": list(map(float, sol.angles.knots_i)),
        "period": sol.period, "sign": sol.sign, "front_index": sol.front_index,
        "cams": [dict(index=c.index, theta=c.theta, scale=c.scale, cx=c.cx,
                      cy=c.cy, full_body=c.full_body, ku=c.ku, kv=c.kv,
                      q=c.q, h=c.h) for c in sol.cams],
    }, indent=1), encoding="utf-8")
    log.info("เก็บผลประเมินกล้องไว้ที่ %s (รันครั้งหน้าใส่ --reuse ได้)", path)


def load_solution(path: Path, stats: list[MaskStats]) -> "Solution | None":
    """อ่านผลที่เก็บไว้ คืน None ถ้าไม่มีหรือไม่ตรงกับเฟรมชุดนี้"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cams = [Camera(**c) for c in data["cams"]]
    except (ValueError, TypeError, KeyError) as exc:
        log.warning("อ่านผลประเมินกล้องเดิมไม่ได้ (%s) — ประเมินใหม่", exc)
        return None
    if [c.index for c in cams] != [s.index for s in stats]:
        log.warning("ผลประเมินกล้องเดิมมาจากเฟรมคนละชุด — ประเมินใหม่")
        return None
    log.info("ใช้ผลประเมินกล้องเดิมจาก %s (%d เฟรม)", path, len(cams))
    return Solution(cams, stats, AngleMap(np.array(data["knots"]), data["period"]),
                    float(data["sign"]), int(data["front_index"]),
                    float(data["period"]))


def save_cameras(cams: list[Camera], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        [dict(index=c.index, theta_deg=float(np.degrees(c.theta)),
              scale=c.scale, cx=c.cx, cy=c.cy, full_body=c.full_body)
         for c in cams], indent=1), encoding="utf-8")
    log.info("เขียนพารามิเตอร์กล้อง %d เฟรม -> %s", len(cams), path)
