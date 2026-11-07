r"""เอาภาพนิ่งความละเอียดสูง (หน้า/หลัง/ซ้าย/ขวา) มาเสริมเข้ากับคลิป

ทำไมต้องมีขั้นนี้
-----------------
คลิปกล้องวนรอบตัวให้มุมเยอะก็จริง แต่กล้องเข้าใกล้มากจนตัวละคร**หลุดขอบเฟรม**
ไปเกือบตลอด — ในคลิปตัวอย่างมีแค่ 51 จาก 240 เฟรมที่เห็นครบทั้งตัว และ 47
เฟรมที่หัวโดนขอบบนตัดทิ้ง เมื่อรวมกับความไม่คงรูปของคลิป (ยอดหัวที่แต่ละเฟรม
บอก กระจายกันถึง 6% ของความสูงตัว) ส่วนที่อยู่ไกลจากจุดยึด — ซึ่งก็คือ
**หัว** — จึงถูกเฉือนเหลือแค่กรวยแหลม

ภาพนิ่งชุดสี่มุมไม่มีปัญหานั้นเลย: เห็นเต็มตัว ความละเอียดสูงกว่า และสัดส่วน
คงที่ เอามาเป็นมุมเพิ่มในขั้นเฉือน voxel จึงได้หัวคืนมา และเอาไปร่วมในขั้น
ฉายสีก็ได้ลายผิวที่คมกว่าเฟรมคลิป (942 เทียบกับ ~520 พิกเซลต่อความสูงตัว)

ไม่เชื่อชื่อไฟล์
----------------
"Left.jpeg" กับ "Right.jpeg" ในชุดตัวอย่างหันหน้าไปทางเดียวกันทั้งคู่ ซึ่ง
แปลว่าอย่างน้อยหนึ่งใบถูกพลิกมาแล้ว — เป็นเรื่องปกติของชุดภาพที่คนจัดเอง
โมดูลนี้จึงไม่อ่านมุมจากชื่อไฟล์ แต่**หาเอง**ว่าภาพแต่ละใบตรงกับเฟรมไหนของ
คลิปมากที่สุด (ลองทั้งภาพตรงและภาพพลิก) แล้วใช้มุมของเฟรมนั้น
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .clipviews import _affine
from .config import IMAGE_EXTS, PipelineConfig
from .turntable import Camera, ClipFrame, MaskStats, _clean_mask

log = logging.getLogger(__name__)

# ขนาด canvas ที่ใช้ตอนไล่หามุม — เล็กพอให้สแกนครบ 240 เฟรมได้เร็ว
MATCH_SIZE = 128
# ขนาด canvas ที่ใช้ตอนหาสเกล/ตำแหน่งให้ละเอียด
PLACE_SIZE = 256


@dataclass
class Still:
    """ภาพนิ่งหนึ่งใบที่หาตำแหน่งในระบบพิกัดของคลิปได้แล้ว"""

    name: str
    frame: ClipFrame
    cam: Camera
    stats: MaskStats
    theta_deg: float
    iou: float
    scale_fit: float


# --------------------------------------------------------------------------
# อ่านภาพ + แยกพื้นหลัง
# --------------------------------------------------------------------------
def find_images(folder: Path, skip: tuple[str, ...] = ("up", "down")
                ) -> list[Path]:
    """เก็บไฟล์ภาพในโฟลเดอร์ ยกเว้นมุมบน/ล่างที่เข้า hull รอบตัวไม่ได้"""
    out = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.stem.lower() in skip:
            log.info("ข้ามภาพ %s (มุมบน/ล่างใช้กับ hull รอบตัวไม่ได้)", p.name)
            continue
        out.append(p)
    return out


def segment(paths: list[Path], cache: Path, model: str
            ) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    """แยกพื้นหลังภาพนิ่ง แล้วแคช mask ไว้ (ขั้นนี้ช้าที่สุด)"""
    mask_dir = cache / "stills"
    mask_dir.mkdir(parents=True, exist_ok=True)

    todo = [p for p in paths if not (mask_dir / f"{p.stem}.png").exists()]
    session = None
    if todo:
        try:
            from rembg import new_session, remove
        except ImportError as exc:  # pragma: no cover - ขึ้นกับเครื่อง
            raise RuntimeError(
                "ต้องติดตั้ง rembg ก่อนใช้ภาพนิ่งเสริม:\n"
                r"    .venv\Scripts\python.exe -m pip install rembg onnxruntime"
            ) from exc
        session = new_session(model)
        log.info("แยกพื้นหลังภาพนิ่ง %d ใบด้วย %s ...", len(todo), model)

    out = []
    for p in paths:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"อ่านภาพไม่ได้: {p}")
        rgb = img[:, :, ::-1].copy()
        mpath = mask_dir / f"{p.stem}.png"
        if not mpath.exists():
            from rembg import remove
            alpha = np.asarray(remove(rgb, session=session))[:, :, 3]
            cv2.imwrite(str(mpath), _clean_mask(alpha > 127).astype(np.uint8) * 255)
        mask = cv2.imdecode(np.fromfile(str(mpath), np.uint8),
                            cv2.IMREAD_GRAYSCALE) > 127
        if not mask.any():
            raise ValueError(f"แยกพื้นหลังแล้วไม่เหลืออะไรเลย: {p.name}")
        out.append((p, rgb, mask))
    return out


# --------------------------------------------------------------------------
# วางภาพนิ่งลง canvas กลาง
# --------------------------------------------------------------------------
def _torso_cx(mask: np.ndarray) -> float:
    """กึ่งกลางแนวนอนของ *ช่วงลำตัว* — ทนต่อหางและแขนที่ยื่นออกข้าง

    ใช้ช่วงอกถึงสะโพก (25%-60% ของความสูง) ซึ่งเป็นส่วนที่อยู่ติดแกนหมุนเสมอ
    ถ้าใช้กึ่งกลาง bbox ตัวละครที่มีหางยาวจะถูกเลื่อนไปคนละทิศในแต่ละมุม
    """
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    lo = y0 + int(0.25 * (y1 - y0))
    hi = y0 + int(0.60 * (y1 - y0))
    band = mask[lo:hi + 1]
    if not band.any():
        return float(xs.mean())
    return float(np.nonzero(band)[1].mean())


def place(mask: np.ndarray, rgb: np.ndarray | None, size: int, margin: float,
          scale_mul: float = 1.0, dx: float = 0.0, dy: float = 0.0
          ) -> tuple[np.ndarray, np.ndarray | None]:
    """วางภาพลง canvas แบบเดียวกับที่ ``clipviews._affine`` วางเฟรมของคลิป

    นิยามของ canvas: ตัวละครสูงเต็มพื้นที่โดยเว้นขอบบนล่างด้านละ ``margin``
    ฝ่าเท้าอยู่ที่ขอบล่างของพื้นที่นั้น แกนหมุนอยู่กึ่งกลางภาพ
    """
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    k = size * (1.0 - 2.0 * margin) / (y1 - y0 + 1) * scale_mul
    ground_row = size * (1.0 - margin)
    cx = _torso_cx(mask)
    M = np.float32([[k, 0, size / 2 - k * cx + dx],
                    [0, k, ground_row - k * (y1 + 1) + dy]])
    interp = cv2.INTER_AREA if k < 1.0 else cv2.INTER_LINEAR
    out_mask = cv2.warpAffine(mask.astype(np.uint8) * 255, M, (size, size),
                              flags=interp) > 127
    out_rgb = None
    if rgb is not None:
        out_rgb = cv2.warpAffine(rgb, M, (size, size), flags=interp)
    return out_mask, out_rgb


def _clip_canvas(frame: ClipFrame, cam: Camera, size: int, margin: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """เงาของเฟรมคลิปบน canvas กลาง พร้อมพื้นที่ที่เฟรมนั้นให้ข้อมูลได้

    ``covered`` สำคัญมาก: เฟรมที่ตัวหลุดขอบไปครึ่งตัวไม่ควรถูกตัดสินว่า
    "เงาไม่เหมือน" เพราะส่วนที่หายไปไม่ใช่ความต่างของรูปทรง แต่เป็นเพราะ
    กล้องถ่ายไม่ติด
    """
    M = _affine(cam, size, margin)
    interp = cv2.INTER_AREA if M[0, 0] < 1.0 else cv2.INTER_LINEAR
    m = cv2.warpAffine(frame.mask.astype(np.uint8) * 255, M, (size, size),
                       flags=interp) > 127
    cov = cv2.warpAffine(np.full(frame.mask.shape, 255, np.uint8), M,
                         (size, size), flags=cv2.INTER_NEAREST) > 127
    frame.unload()
    return m, cov


def _iou_where(a: np.ndarray, b: np.ndarray, where: np.ndarray) -> float:
    a, b = a & where, b & where
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 0.0


def score_angles(masks: list[np.ndarray], frames: list[ClipFrame],
                 cams: list[Camera], margin: float, step: int = 1
                 ) -> tuple[np.ndarray, np.ndarray]:
    """ให้คะแนนภาพนิ่งทุกใบกับเฟรมคลิปทุกเฟรม

    คืน (มุมของเฟรมที่ใช้ได้ (องศา), ตาราง IoU ขนาด ภาพนิ่ง x เฟรม)
    วนเฟรมเป็นวงนอกเพราะการ warp เฟรมคลิปคือส่วนที่แพง — ทำครั้งเดียวแล้ว
    เทียบกับภาพนิ่งทุกใบเลย

    **ไม่ลองพลิกซ้ายขวา** โดยตั้งใจ: การพลิกภาพของวัตถุสามมิติไม่ได้ให้มุมมอง
    อีกมุมของวัตถุเดิม นอกจากวัตถุนั้นจะสมมาตรซ้ายขวาพอดี ตัวละครที่มีหาง
    สะบัดไปข้างหนึ่งจึงกลายเป็นตัวละครคนละตัว — และเพราะเงาด้านข้างของคนกับ
    สัตว์สองขาเกือบสมมาตร ภาพข้างที่พลิกแล้วจะไปทับมุมของข้างตรงข้ามได้ง่าย ๆ
    ด้วย IoU สูงลิ่ว ทั้งที่เป็นข้อมูลผิดด้าน
    """
    small = [place(m, None, MATCH_SIZE, margin)[0] for m in masks]
    idx = list(range(0, len(frames), step))
    table = np.full((len(masks), len(idx)), -1.0)
    angles = np.zeros(len(idx))

    for col, i in enumerate(idx):
        angles[col] = np.degrees(cams[i].theta) % 360.0
        cm, cov = _clip_canvas(frames[i], cams[i], MATCH_SIZE, margin)
        if np.count_nonzero(cov) < 0.25 * cov.size:
            continue
        for row, cand in enumerate(small):
            table[row, col] = _iou_where(cand, cm, cov)
    return angles, table


def _gap(a: float, b: float) -> float:
    """ระยะเชิงมุมที่สั้นที่สุดระหว่างสองมุม (องศา)"""
    return abs(float(np.degrees(np.angle(np.exp(1j * np.radians(a - b))))))


def assign(angles: np.ndarray, table: np.ndarray, min_sep: float = 45.0
           ) -> list[tuple[int, float]]:
    """เลือกมุมให้ภาพนิ่งแต่ละใบ โดยห้ามสองใบไปลงมุมใกล้กัน

    ชุดภาพสี่มุมมาจากการวาดตัวละครจากสี่ทิศที่ต่างกัน ถ้าปล่อยให้แต่ละใบ
    เลือกมุมที่คะแนนดีที่สุดอิสระจากกัน ภาพข้างสองใบจะไปลงมุมเดียวกันทั้งคู่
    (เงาด้านข้างสองข้างคล้ายกันมาก) แล้ว hull จะไม่มีมุมจากอีกข้างเลย

    เรียงจากใบที่มั่นใจที่สุดก่อน แล้วกันเขตรอบมุมที่ถูกจองไปแล้ว
    """
    taken: list[float] = []
    out: list[tuple[int, float]] = [(-1, -1.0)] * len(table)
    order = sorted(range(len(table)), key=lambda r: -table[r].max())
    for row in order:
        best_col, best_score = -1, -1.0
        for col in range(len(angles)):
            if table[row, col] < 0:
                continue
            if any(_gap(angles[col], t) < min_sep for t in taken):
                continue
            if table[row, col] > best_score:
                best_col, best_score = col, float(table[row, col])
        if best_col < 0:
            continue
        taken.append(float(angles[best_col]))
        out[row] = (best_col, best_score)
    return out


def snap_to_lattice(degs: list[float], n: int = 4, tol: float = 20.0
                    ) -> list[float] | None:
    """ดัดมุมทั้งชุดให้ลงล็อกที่ห่างกัน 360/n องศาพอดี

    ชุดภาพ turnaround มาตรฐานวาดห่างกัน 90 องศาเป๊ะ ๆ ส่วนมุมที่วัดได้จากการ
    ทับเงามีความคลาดเคลื่อนระดับหลายองศา (เงาเปลี่ยนช้ามากแถวมุมหน้าตรง)
    ถ้าทั้งชุดเข้าล็อกเดียวกันได้ในระยะ ``tol`` ก็ใช้ล็อกนั้นแทนค่าที่วัดมา
    คืน None ถ้าไม่เข้าล็อก — แปลว่าสมมติฐาน "ห่างกันเท่า ๆ กัน" ใช้ไม่ได้
    """
    if len(degs) != n:
        return None
    pitch = 360.0 / n
    grid = np.arange(0.0, pitch, 0.25)
    best, best_err = None, float("inf")
    for phi in grid:
        slots = [(phi + k * pitch) % 360.0 for k in range(n)]
        used: list[int] = []
        err = 0.0
        for d in degs:
            k = min((j for j in range(n) if j not in used),
                    key=lambda j: _gap(d, slots[j]))
            used.append(k)
            err += _gap(d, slots[k]) ** 2
        if err < best_err:
            best_err, best = err, [slots[j] for j in used]
    if best is None:
        return None
    worst = max(_gap(d, b) for d, b in zip(degs, best))
    if worst > tol:
        log.warning("มุมของภาพนิ่งไม่เข้าล็อก %.0f องศา (คลาดสูงสุด %.1f) — "
                    "ใช้มุมที่วัดได้ตามเดิม", pitch, worst)
        return None
    log.info("ดัดมุมภาพนิ่งเข้าล็อก %.0f องศา (คลาดสูงสุด %.1f องศา)",
             pitch, worst)
    return best


def fit_placement(mask: np.ndarray, frame: ClipFrame, cam: Camera,
                  margin: float) -> tuple[float, float, float, float]:
    """ไล่หาสเกล/ตำแหน่งที่ทำให้เงาภาพนิ่งทับเงาเฟรมคลิปได้ดีที่สุด

    ภาพนิ่งวัดความสูงจาก bbox ของตัวเอง ส่วนคลิปวัดจากผลประเมินกล้อง สองค่านี้
    ไม่มีเหตุผลให้ตรงกันเป๊ะ ๆ ถ้าไม่ดัดก่อน เงาที่ขนาดต่างกันไม่กี่เปอร์เซ็นต์
    ก็เฉือนเนื้อหายได้ทั้งแถบ เพราะ visual hull เอา *ค่าน้อยสุด* ของทุกมุม
    """
    cm, cov = _clip_canvas(frame, cam, PLACE_SIZE, margin)
    best = (1.0, 0.0, 0.0, -1.0)
    for s in np.arange(0.90, 1.1001, 0.01):
        for dx in np.arange(-8, 8.1, 2.0):
            for dy in np.arange(-8, 8.1, 2.0):
                cand, _ = place(mask, None, PLACE_SIZE, margin, s, dx, dy)
                score = _iou_where(cand, cm, cov)
                if score > best[3]:
                    best = (float(s), float(dx), float(dy), score)
    return best


# --------------------------------------------------------------------------
def build(folder: Path, cache: Path, frames: list[ClipFrame],
          cams: list[Camera], cfg: PipelineConfig, size: int = 1024,
          min_iou: float = 0.55) -> list[Still]:
    """เตรียมภาพนิ่งทั้งชุดให้พร้อมใช้เป็น "มุมเพิ่ม" ของขั้นเฉือน voxel"""
    paths = find_images(folder)
    if not paths:
        log.warning("ไม่พบภาพนิ่งใน %s — ข้ามการเสริมภาพนิ่ง", folder)
        return []

    margin = cfg.margin
    out_dir = cache / "stills"
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = segment(paths, cache, cfg.rembg_model)
    angles, table = score_angles([m for _, _, m in loaded], frames, cams,
                                 margin)
    picks = assign(angles, table)

    measured = [float(angles[c]) for c, _ in picks if c >= 0]
    snapped = snap_to_lattice(measured) if len(measured) == len(picks) else None
    for (path, _, _), (col, score) in zip(loaded, picks):
        log.info("  %-10s วัดได้ %6.1f องศา (IoU %.3f)", path.name,
                 angles[col] if col >= 0 else float("nan"), score)

    stills: list[Still] = []
    for n, ((path, rgb, mask), (col, _)) in enumerate(zip(loaded, picks)):
        if col < 0:
            log.warning("หามุมที่ว่างให้ %s ไม่ได้ — ข้าม", path.name)
            continue
        deg = snapped[n] if snapped is not None else float(angles[col])

        # ดัดสเกล/ตำแหน่งโดยเทียบกับเฟรมคลิปที่อยู่ใกล้มุมที่เลือกที่สุด
        j = min(range(len(cams)),
                key=lambda i: _gap(np.degrees(cams[i].theta) % 360.0, deg))
        s, dx, dy, iou2 = fit_placement(mask, frames[j], cams[j], margin)

        if iou2 < min_iou:
            log.warning("ภาพ %s ทับเงาคลิปได้แค่ IoU %.2f ที่ %.1f องศา — "
                        "ข้ามไว้ก่อน ดีกว่าเอาไปเฉือนเนื้อหาย",
                        path.name, iou2, deg)
            continue

        canvas_mask, canvas_rgb = place(mask, rgb, size, margin, s,
                                        dx * size / PLACE_SIZE,
                                        dy * size / PLACE_SIZE)
        stem = f"still_{path.stem.lower()}"
        rgb_path, mask_path = out_dir / f"{stem}.jpg", out_dir / f"{stem}_m.png"
        cv2.imwrite(str(rgb_path), canvas_rgb[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(mask_path), canvas_mask.astype(np.uint8) * 255)

        # กล้องของ canvas กลาง: ภาพฉายขนาน (q=0) ตัวสูงเต็มพื้นที่ที่เว้นขอบแล้ว
        cam = Camera(index=-(n + 1), theta=np.radians(deg),
                     scale=size * (1.0 - 2.0 * margin),
                     cx=size / 2.0, cy=size * (1.0 - margin),
                     full_body=True, q=0.0, h=0.5)
        frame = ClipFrame(cam.index, rgb_path, mask_path)
        ys, xs = np.nonzero(canvas_mask)
        stats = MaskStats(index=cam.index, y0=int(ys.min()), y1=int(ys.max()),
                          x0=int(xs.min()), x1=int(xs.max()),
                          touch_top=False, touch_bottom=False,
                          core_cx=_torso_cx(canvas_mask),
                          area=int(canvas_mask.sum()))
        stills.append(Still(path.stem, frame, cam, stats, deg, iou2, s))
        log.info("ภาพ %-10s -> %6.1f องศา  IoU %.3f  สเกล x%.3f",
                 path.name, deg, iou2, s)

    if stills:
        log.info("เสริมภาพนิ่ง %d ใบเข้ากับคลิป (%.0f พิกเซลต่อความสูงตัว "
                 "เทียบกับคลิป ~%.0f)", len(stills), stills[0].cam.scale,
                 float(np.median([c.scale for c in cams])))
    return stills


def debug_sheet(stills: list[Still], frames: list[ClipFrame],
                cams: list[Camera], cfg: PipelineConfig, path: Path) -> None:
    """ภาพตรวจงาน: เงาภาพนิ่ง (เขียว) ทับเงาคลิปที่มุมเดียวกัน (ชมพู)"""
    if not stills:
        return
    S = PLACE_SIZE
    tiles = []
    for st in stills:
        j = min(range(len(cams)),
                key=lambda i: abs(np.angle(np.exp(
                    1j * (cams[i].theta - st.cam.theta)))))
        cm, cov = _clip_canvas(frames[j], cams[j], S, cfg.margin)
        sm = cv2.resize(cv2.imdecode(np.fromfile(str(st.frame.mask_path),
                                                 np.uint8),
                                     cv2.IMREAD_GRAYSCALE),
                        (S, S), interpolation=cv2.INTER_AREA) > 127
        tile = np.zeros((S, S, 3), np.uint8)
        tile[..., 1] = sm * 200            # ภาพนิ่ง = เขียว
        tile[..., 2] = cm * 200            # คลิป = แดง (ทับกันได้เหลือง)
        tile[~cov] //= 3                   # หรี่ส่วนที่คลิปถ่ายไม่ติด
        cv2.putText(tile, f"{st.name} {st.theta_deg:.0f}d IoU{st.iou:.2f}",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet = np.hstack(tiles)
    ok, buf = cv2.imencode(".png", sheet)
    if ok:
        buf.tofile(str(path))
        log.info("ภาพตรวจการวางภาพนิ่ง -> %s", path)
