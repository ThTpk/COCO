"""ขั้นที่ 1 ของ pipeline: แยกตัวละครออกจากพื้นหลัง แล้ว align ทุกมุมให้อยู่ในระบบเดียวกัน

ขั้นนี้สำคัญที่สุดของทั้ง pipeline เพราะภาพ AI-generated แต่ละมุมมักมีสัดส่วนไม่ตรงกัน
(ส่วนสูงต่างกัน จุดศูนย์กลางเลื่อน ระยะกล้องไม่เท่ากัน) ถ้าไม่แก้ก่อน
visual hull จะได้รูปทรงเบี้ยวโดยที่ขั้นถัดไปไม่มีทางรู้เลยว่าผิดตั้งแต่ตรงนี้

หลังขั้นนี้ ทุกมุมจะอยู่บน canvas จัตุรัสขนาดเท่ากัน โดย
  * ยอดหัวอยู่แถวเดียวกัน และฝ่าเท้าอยู่แถวเดียวกัน (scale ด้วยความสูงของ mask)
  * จุดศูนย์กลางแนวนอนตรงกัน
  * มุม back ถูกพลิกซ้ายขวาแล้ว เพื่อให้แกน X ตรงกับมุม front
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from .config import IMAGE_EXTS, VIEW_ALIASES, VIEWS, PipelineConfig, ViewSpec

log = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]  # (y0, y1, x0, x1) ขอบเขตแบบรวมปลาย


# ---------------------------------------------------------------------------
# โครงสร้างผลลัพธ์
# ---------------------------------------------------------------------------


@dataclass
class PreparedView:
    """มุมหนึ่งที่ segment + normalize เรียบร้อยแล้ว พร้อมส่งเข้าขั้น carve"""

    spec: ViewSpec
    source: Path
    rgb: np.ndarray            # (S, S, 3) uint8  — S = cfg.canvas
    mask: np.ndarray           # (S, S) bool
    scale: float               # ตัวคูณจากภาพต้นฉบับมาเป็น canvas
    src_bbox: BBox             # bbox ของ mask ในภาพต้นฉบับ
    src_size: tuple[int, int]  # (h, w) ของภาพต้นฉบับ
    facing: int = 0            # ใช้กับมุมข้างเท่านั้น: +1 = หันไปทางขวาของภาพ

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def coverage(self) -> float:
        """สัดส่วนพิกเซลที่เป็นตัวละคร — ใช้เทียบความสอดคล้องระหว่างมุม"""
        return float(self.mask.mean())

    @property
    def width_profile(self) -> np.ndarray:
        """ความกว้างของ mask ในแต่ละแถว — เป็นลายนิ้วมือของรูปร่าง"""
        return self.mask.sum(axis=1).astype(np.float64)

    @property
    def src_aspect(self) -> float:
        """อัตราส่วน กว้าง/สูง ของ mask ในภาพต้นฉบับ"""
        y0, y1, x0, x1 = self.src_bbox
        return (x1 - x0 + 1) / max(1, y1 - y0 + 1)


# ---------------------------------------------------------------------------
# ค้นหาไฟล์ภาพของแต่ละมุม
# ---------------------------------------------------------------------------


def discover_views(input_dir: Path) -> dict[str, Path]:
    """จับคู่ไฟล์ในโฟลเดอร์กับชื่อมุม โดยดูจากชื่อไฟล์

    รองรับชื่อแบบ front.png / char_back.jpg / 000_f.webp / side_r.png
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ input: {input_dir}")

    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        raise FileNotFoundError(
            f"ไม่มีไฟล์ภาพใน {input_dir} (รองรับ {', '.join(IMAGE_EXTS)})"
        )

    # เรียง alias จากยาวไปสั้น เพื่อให้ "side_r" ชนะ "side" และ "front" ชนะ "f"
    ordered = sorted(
        ((view, alias) for view, aliases in VIEW_ALIASES.items() for alias in aliases),
        key=lambda pair: -len(pair[1]),
    )

    found: dict[str, Path] = {}
    for path in files:
        stem = path.stem.lower()
        tokens = set(stem.replace("-", "_").replace(" ", "_").split("_"))
        for view, alias in ordered:
            if view in found:
                continue
            # alias สั้น (f, b, s) ต้องตรงทั้ง token เพื่อไม่ให้ไปชนชื่ออื่นมั่ว
            # ส่วน alias ยาวยอมให้เป็นส่วนหนึ่งของชื่อได้ เช่น "frontview.jpeg"
            if alias in tokens or stem == alias or (len(alias) >= 4 and alias in stem):
                found[view] = path
                log.info("มุม %-6s <- %s", view, path.name)
                break

    missing = [v for v, spec in VIEWS.items() if spec.required and v not in found]
    if missing:
        raise ValueError(
            "หาไฟล์ของมุมเหล่านี้ไม่เจอ: " + ", ".join(missing) + "\n"
            "ตั้งชื่อไฟล์ให้มีคำว่า front / back / side เช่น front.png, back.png, side.png\n"
            "ไฟล์ที่เจอในโฟลเดอร์: " + ", ".join(p.name for p in files)
        )
    return found


# ---------------------------------------------------------------------------
# โหลดภาพและทำ mask
# ---------------------------------------------------------------------------


def load_rgba(path: Path) -> np.ndarray:
    """อ่านภาพเป็น RGBA uint8 — ใช้ fromfile เพื่อรองรับ path ภาษาไทยบน Windows"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"อ่านภาพไม่ได้ (ไฟล์เสียหรือฟอร์แมตไม่รองรับ): {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return np.ascontiguousarray(img[:, :, [2, 1, 0, 3]])  # BGRA -> RGBA


def _mask_from_alpha(rgba: np.ndarray, cfg: PipelineConfig) -> np.ndarray | None:
    alpha = rgba[:, :, 3]
    transparent = float((alpha < cfg.mask_threshold).mean())
    # ถือว่า alpha ใช้งานได้ เมื่อมีทั้งส่วนโปร่งและส่วนทึบจริง
    if 0.01 < transparent < 0.99:
        return alpha >= cfg.mask_threshold
    return None


def _mask_from_flat_background(
    rgba: np.ndarray, cfg: PipelineConfig
) -> np.ndarray | None:
    """ประมาณสีพื้นหลังจากขอบภาพ แล้วตัดทุกพิกเซลที่ใกล้สีนั้น"""
    rgb = rgba[:, :, :3].astype(np.int16)
    h, w = rgb.shape[:2]
    band = max(2, min(h, w) // 100)

    border = np.concatenate([
        rgb[:band].reshape(-1, 3),
        rgb[-band:].reshape(-1, 3),
        rgb[:, :band].reshape(-1, 3),
        rgb[:, -band:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)

    # ขอบภาพสีไม่สม่ำเสมอ = พื้นหลังไม่ใช่สีพื้น -> วิธีนี้ใช้ไม่ได้
    spread = float(np.percentile(np.abs(border - bg).max(axis=1), 85))
    if spread > cfg.bg_tolerance * 2:
        log.debug("ขอบภาพสีไม่สม่ำเสมอ (spread=%.1f) ข้ามวิธีพื้นหลังสีพื้น", spread)
        return None

    mask = np.abs(rgb - bg).max(axis=2) > cfg.bg_tolerance
    if not 0.01 < float(mask.mean()) < 0.95:
        return None
    return mask


def _mask_from_grabcut(rgba: np.ndarray, cfg: PipelineConfig) -> np.ndarray | None:
    """แยกพื้นหลังด้วย GrabCut — ใช้ได้กับพื้นหลังที่ไล่เฉดหรือมีลวดลาย

    เหมาะกับภาพตัวละครที่อยู่กลางภาพและสีตัดกับพื้นหลัง ซึ่งเป็นลักษณะของ
    ภาพ character sheet แทบทุกใบ ข้อดีคือไม่ต้องโหลดโมเดลจากเน็ตเลย

    เราใส่ trimap เริ่มต้นให้ GrabCut แทนที่จะให้แค่กรอบสี่เหลี่ยม เพราะ
    การบอกว่า "ขอบภาพคือพื้นหลังแน่ ๆ" และ "แกนกลางคือตัวละครแน่ ๆ"
    ทำให้ลู่เข้าเร็วและแม่นกว่ามาก
    """
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    h, w = rgb.shape[:2]

    # ย่อก่อนเพื่อความเร็ว GrabCut เป็น O(พิกเซล) และช้า
    scale = min(1.0, cfg.grabcut_size / max(h, w))
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb.copy()
    sh, sw = small.shape[:2]

    trimap = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    border = max(2, int(0.03 * min(sh, sw)))
    trimap[:border, :] = trimap[-border:, :] = cv2.GC_BGD
    trimap[:, :border] = trimap[:, -border:] = cv2.GC_BGD
    # กล่องกลางภาพ: น่าจะเป็นตัวละคร
    trimap[int(0.12 * sh):int(0.96 * sh), int(0.30 * sw):int(0.70 * sw)] = cv2.GC_PR_FGD
    # แกนกลางลำตัว: ยืนยันว่าเป็นตัวละครแน่นอน
    #
    # แถบนี้ต้องหยุดเหนือเป้า ห้ามลากลงไปถึงช่วงขาเด็ดขาด เพราะกลางภาพ
    # ใต้เป้าลงไปคือ *ช่องว่างระหว่างขา* ซึ่งเป็นพื้นหลัง การมาร์คตรงนั้นว่า
    # เป็นตัวละครแน่นอน จะบังคับให้ GrabCut กลืนพื้นหลังระหว่างขาเข้ามา
    # แล้วขาสองข้างจะติดกันเป็นกระโปรงตลอดไป แก้ทีหลังไม่ได้เลย
    trimap[int(0.25 * sh):int(0.55 * sh), int(0.45 * sw):int(0.55 * sw)] = cv2.GC_FGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, trimap, None, bgd, fgd,
                    cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        log.debug("GrabCut ล้มเหลว: %s", exc)
        return None

    m = (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)
    if not 0.02 < float(m.mean()) < 0.92:
        log.debug("GrabCut ให้ mask ที่กินพื้นที่ %.1f%% ซึ่งดูไม่สมเหตุผล", 100 * m.mean())
        return None

    return cv2.resize(m.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def _mask_from_rembg(rgba: np.ndarray, cfg: PipelineConfig) -> np.ndarray | None:
    try:
        from rembg import new_session, remove
    except ImportError:
        return None

    cache = getattr(_mask_from_rembg, "_sessions", None)
    if cache is None:
        cache = {}
        _mask_from_rembg._sessions = cache
    session = cache.get(cfg.rembg_model)
    if session is None:
        log.info("โหลดโมเดล rembg '%s' (ครั้งแรกจะดาวน์โหลดสักพัก)", cfg.rembg_model)
        session = new_session(cfg.rembg_model)
        cache[cfg.rembg_model] = session

    out = np.asarray(remove(rgba[:, :, :3], session=session))
    return out[:, :, 3] >= cfg.mask_threshold


def clean_mask(mask: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """ปิดรู ลบจุดรบกวน และเก็บเฉพาะก้อนที่ใหญ่ที่สุด"""
    m = mask.astype(np.uint8)
    k = max(3, (min(mask.shape) // 250) | 1)  # บังคับให้เป็นเลขคี่
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)   # ลบ speckle
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)  # ปิดรูเล็ก
    out = ndimage.binary_fill_holes(m.astype(bool))
    assert out is not None

    if cfg.keep_largest_component:
        labels, n = ndimage.label(out)
        if n > 1:
            sizes = ndimage.sum(out, labels, range(1, n + 1))
            out = labels == int(np.argmax(sizes)) + 1
            log.debug("ทิ้งก้อนแยก %d ก้อน (เหลือก้อนใหญ่สุด %.1f%% ของภาพ)",
                      n - 1, 100 * out.mean())
    return out


def extract_mask(rgba: np.ndarray, cfg: PipelineConfig, view: str) -> np.ndarray:
    """ลองแยกพื้นหลังจากวิธีที่ถูกที่สุดไปแพงที่สุด"""
    for label, fn in (
        ("alpha channel", _mask_from_alpha),
        ("พื้นหลังสีพื้น", _mask_from_flat_background),
        ("GrabCut", _mask_from_grabcut),
        ("rembg", _mask_from_rembg),
    ):
        mask = fn(rgba, cfg)
        if mask is not None:
            log.info("มุม %-6s segment ด้วย: %s", view, label)
            return clean_mask(mask, cfg)

    raise RuntimeError(
        f"มุม {view}: แยกตัวละครจากพื้นหลังไม่ได้ "
        "ภาพไม่มี alpha และพื้นหลังไม่ใช่สีพื้น\n"
        "ทางแก้: ติดตั้ง rembg เพิ่ม\n"
        r"    .venv\Scripts\python.exe -m pip install rembg onnxruntime-gpu" "\n"
        "หรือลบพื้นหลังออกเป็น PNG โปร่งใสก่อนเอาเข้า pipeline"
    )


# ---------------------------------------------------------------------------
# normalize: scale ด้วยความสูง + จัดกลาง
# ---------------------------------------------------------------------------


def mask_bbox(mask: np.ndarray) -> BBox:
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("mask ว่าง — ไม่มีพิกเซลของตัวละครเลย")
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def detect_side_facing(mask: np.ndarray, bbox: BBox) -> tuple[int, float]:
    """เดาว่ามุมข้างหันหน้าไปทางไหน จากปลายเท้าที่ยื่นออกไปด้านหน้า

    ท่า A-pose ยืนตรง ปลายเท้าจะยื่นไปทางที่ตัวละครหันหน้าเสมอ
    ซึ่งเป็นสัญญาณที่เชื่อถือได้มากกว่าใบหน้า (เพราะทรงผมด้านหลังมักหักล้างจมูก)

    คืน (ทิศ, ความมั่นใจ) โดยทิศ +1 = หันไปทางขวาของภาพ
    ความมั่นใจคือระยะเลื่อนของ centroid คิดเป็นสัดส่วนของความกว้างลำตัว
    """
    y0, y1, x0, x1 = bbox
    height = y1 - y0 + 1

    feet = mask[max(y0, y1 - int(0.06 * height)) : y1 + 1]
    torso = mask[y0 + int(0.25 * height) : y0 + int(0.65 * height)]
    if feet.sum() == 0 or torso.sum() == 0:
        return 0, 0.0

    feet_cx = float(np.where(feet)[1].mean())
    torso_cx = float(np.where(torso)[1].mean())
    torso_w = max(1.0, float(torso.sum(axis=1).mean()))

    shift = (feet_cx - torso_cx) / torso_w
    return (1 if shift > 0 else -1), abs(shift)


def normalize_view(
    rgba: np.ndarray,
    mask: np.ndarray,
    spec: ViewSpec,
    cfg: PipelineConfig,
    source: Path,
) -> PreparedView:
    """ย่อ/ขยายให้ความสูงของ mask เท่ากันทุกมุม แล้ววางลง canvas จัตุรัส

    scale คิดจาก *ความสูง* เท่านั้น ความกว้างจึงออกมาตามสัดส่วนจริงของภาพนั้น
    ซึ่งเป็นสิ่งที่ต้องการ: มุมหน้าให้ความกว้างลำตัว มุมข้างให้ความหนาลำตัว
    """
    src_h, src_w = mask.shape
    src_bbox = mask_bbox(mask)
    y0, y1, x0, x1 = src_bbox

    facing, confidence = (0, 0.0)
    if spec.axis_u == "z":  # มุมข้าง
        facing, confidence = detect_side_facing(mask, src_bbox)
        log.info("มุม %-6s เดาว่าหันหน้าไปทาง %s (ความมั่นใจ %.2f)",
                 spec.name, "ขวาของภาพ" if facing > 0 else "ซ้ายของภาพ", confidence)

    S = cfg.canvas
    margin = cfg.resolved_margin_px()
    target_h = S - 2 * margin
    scale = target_h / (y1 - y0 + 1)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    rgb_r = cv2.resize(rgba[:, :, :3], (new_w, new_h), interpolation=interp)
    mask_r = cv2.resize(mask.astype(np.float32), (new_w, new_h),
                        interpolation=cv2.INTER_LINEAR) >= 0.5

    # bbox ใหม่หลัง resize (คำนวณซ้ำ ปลอดภัยกว่าคูณ scale ตรง ๆ)
    ny0, ny1, nx0, nx1 = mask_bbox(mask_r)

    # วางให้ยอดหัวอยู่ที่แถว margin
    dy = margin - ny0

    # จัดกลางแนวนอนด้วย centroid ของช่วงขา ไม่ใช่กึ่งกลาง bbox
    #
    # bbox โดนส่วนที่ยื่นออกมาลากไปง่ายมาก ตัวละครที่มีหาง ปีก ผมยาว หรืออาวุธ
    # จะถูกเลื่อนไปคนละทิศในแต่ละมุม (หางชี้ขวาในมุมหน้า แต่ชี้ซ้ายในมุมข้าง)
    # ทำให้สามมุมไม่ตรงแกนกันแล้ว hull เบี้ยว
    # ช่วงขาเป็นจุดอ้างอิงที่นิ่งที่สุด เพราะอยู่กลางลำตัวเสมอในทุกมุม
    leg_top = int(ny0 + 0.62 * (ny1 - ny0))
    legs = mask_r[leg_top : ny1 + 1]
    if legs.any():
        center_x = float(np.where(legs)[1].mean())
        anchor = "ช่วงขา"
    else:
        center_x = float(np.where(mask_r)[1].mean())
        anchor = "ทั้งตัว"
    dx = int(round(S / 2 - center_x))
    log.debug("มุม %s: จัดกลางด้วย centroid %s ที่ x=%.1f (bbox กลางที่ x=%.1f)",
              spec.name, anchor, center_x, 0.5 * (nx0 + nx1))

    canvas_rgb = np.zeros((S, S, 3), np.uint8)
    canvas_mask = np.zeros((S, S), bool)

    # หา region ที่ซ้อนทับกันระหว่างภาพที่เลื่อนแล้วกับ canvas
    sy0, sy1 = max(0, -dy), min(new_h, S - dy)
    sx0, sx1 = max(0, -dx), min(new_w, S - dx)
    if sy0 >= sy1 or sx0 >= sx1:
        raise ValueError(f"มุม {spec.name}: mask หลุดออกนอก canvas ทั้งหมด")

    canvas_rgb[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = rgb_r[sy0:sy1, sx0:sx1]
    canvas_mask[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = mask_r[sy0:sy1, sx0:sx1]

    clipped = int(mask_r.sum() - canvas_mask.sum())
    if clipped > 0.001 * mask_r.sum():
        log.warning("มุม %s: mask ถูกตัดออก %d พิกเซล — ลองเพิ่ม margin ใน config",
                    spec.name, clipped)

    # พลิกซ้ายขวาให้แกนตรงกับมุม front
    flip = spec.flip_u
    if spec.axis_u == "z" and facing < 0:
        # ตัวละครหันไปทางซ้ายของภาพ แต่ระบบพิกัดคาดว่าหน้าอยู่ทาง +Z (ขวา)
        flip = not flip
        log.info("มุม %s: พลิกซ้ายขวาอัตโนมัติ เพื่อให้หันหน้าไปทาง +Z", spec.name)
    if flip:
        canvas_rgb = canvas_rgb[:, ::-1].copy()
        canvas_mask = canvas_mask[:, ::-1].copy()

    canvas_rgb[~canvas_mask] = 0  # ล้างสีพื้นหลังที่เล็ดลอดมา

    return PreparedView(
        spec=spec,
        source=source,
        rgb=canvas_rgb,
        mask=canvas_mask,
        scale=scale,
        src_bbox=src_bbox,
        src_size=(src_h, src_w),
        facing=facing,
    )


# ---------------------------------------------------------------------------
# ตรวจความสอดคล้องระหว่างมุม
# ---------------------------------------------------------------------------


def check_consistency(views: dict[str, PreparedView]) -> list[str]:
    """เตือนเมื่อภาพแต่ละมุมดูเหมือนไม่ใช่ตัวละครเดียวกัน/สัดส่วนเดียวกัน

    นี่คือปัญหาที่พบบ่อยที่สุดของภาพ AI-generated หลายมุม
    ตรวจเจอตรงนี้ดีกว่าไปงงกับ mesh เบี้ยวตอนท้าย
    """
    issues: list[str] = []
    front, back = views.get("front"), views.get("back")

    if front and back:
        # สัดส่วนของ mask ในภาพต้นฉบับควรใกล้กัน เพราะเป็นตัวละครเดียวกัน
        ra, rb = front.src_aspect, back.src_aspect
        rel = abs(ra - rb) / max(ra, rb)
        if rel > 0.15:
            issues.append(
                f"สัดส่วน กว้าง/สูง ของมุม front ({ra:.3f}) กับ back ({rb:.3f}) "
                f"ต่างกัน {rel:.0%} — ภาพสองมุมนี้น่าจะไม่ใช่สัดส่วนเดียวกัน "
                "mesh ที่ได้จะกว้างหรือแคบผิดไปในบางช่วงความสูง"
            )

        # รูปร่าง silhouette ควรคล้ายกันเมื่อมองจากหน้าและหลัง
        pa, pb = front.width_profile, back.width_profile
        if pa.std() > 1e-6 and pb.std() > 1e-6:
            corr = float(np.corrcoef(pa, pb)[0, 1])
            if corr < 0.90:
                issues.append(
                    f"โปรไฟล์ความกว้างของ front กับ back สัมพันธ์กันแค่ {corr:.2f} "
                    "(ควร > 0.90) — ท่าทางในสองภาพอาจไม่เหมือนกัน "
                    "เช่น กางแขนไม่เท่ากัน หรือขาอยู่ตำแหน่งต่างกัน"
                )

    side = views.get("side")
    if side and side.facing == 0:
        issues.append(
            "เดาทิศการหันหน้าของมุมข้างไม่ได้ (หาปลายเท้าไม่เจอ) — "
            "ถ้า mesh ออกมาหน้าหลังสลับกัน ให้ใช้ flag --flip-side"
        )
    elif side:
        _, conf = detect_side_facing(side.mask, mask_bbox(side.mask))
        if conf < 0.08:
            issues.append(
                f"มุมข้าง: ความมั่นใจเรื่องทิศการหันหน้าต่ำ ({conf:.2f}) "
                "ตรวจภาพ debug/02_aligned.png ว่าหันหน้าไปทางขวาหรือไม่ "
                "ถ้าไม่ ให้ใช้ flag --flip-side"
            )

    for name, v in views.items():
        if v.coverage < 0.02:
            issues.append(f"มุม {name}: ตัวละครกินพื้นที่แค่ {v.coverage:.1%} "
                          "ของภาพ — segment อาจผิด")
    return issues


# ---------------------------------------------------------------------------
# ภาพ debug
# ---------------------------------------------------------------------------


def write_debug(views: dict[str, PreparedView], debug_dir: Path) -> None:
    """เขียนภาพรวมทุกมุมหลัง align พร้อมเส้นอ้างอิง เพื่อตรวจด้วยตา"""
    debug_dir.mkdir(parents=True, exist_ok=True)
    order = [n for n in ("front", "back", "side", "side_r") if n in views]

    tiles = []
    for name in order:
        v = views[name]
        tile = v.rgb.copy()
        edge = cv2.morphologyEx(v.mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8)).astype(bool)
        tile[edge] = (0, 255, 128)

        S = tile.shape[0]
        y0, y1, _, _ = mask_bbox(v.mask)
        for y, color in ((y0, (255, 64, 64)), (y1, (64, 128, 255))):
            cv2.line(tile, (0, y), (S, y), color, 2)
        cv2.line(tile, (S // 2, 0), (S // 2, S), (255, 255, 0), 1)
        cv2.putText(tile, f"{name}  facing={v.facing:+d}  scale={v.scale:.3f}",
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(tile)

    sheet = np.concatenate(tiles, axis=1)
    out = debug_dir / "02_aligned.png"
    cv2.imwrite(str(out), sheet[:, :, ::-1])  # RGB -> BGR
    log.info("เขียนภาพตรวจ align: %s", out)

    for name in order:
        v = views[name]
        cv2.imwrite(str(debug_dir / f"01_mask_{name}.png"),
                    (v.mask * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# entry point ของขั้นนี้
# ---------------------------------------------------------------------------


def prepare_views(
    input_dir: Path,
    cfg: PipelineConfig,
    debug_dir: Path | None = None,
    flip_side: bool = False,
) -> dict[str, PreparedView]:
    """โหลดทุกมุม segment แล้ว align ให้อยู่ในระบบพิกัดเดียวกัน"""
    paths = discover_views(input_dir)
    views: dict[str, PreparedView] = {}

    for name, path in paths.items():
        rgba = load_rgba(path)
        mask = extract_mask(rgba, cfg, name)
        spec = VIEWS[name]
        if flip_side and spec.axis_u == "z":
            spec = ViewSpec(spec.name, spec.axis_u, not spec.flip_u,
                            spec.depth_axis, spec.required)
        views[name] = normalize_view(rgba, mask, spec, cfg, path)

    issues = check_consistency(views)
    for msg in issues:
        log.warning("ความสอดคล้องระหว่างมุม: %s", msg)

    if debug_dir is not None and cfg.debug:
        write_debug(views, debug_dir)

    log.info("เตรียมมุมเสร็จ %d มุม: %s", len(views), ", ".join(views))
    return views
