r"""ซ้อนภาพนิ่งหลาย "ชั้น" ของมุมเดียวกัน แล้วให้แต่ละชั้นทำงานที่ตัวเองถนัด

เส้นทางภาพนิ่งเดิม (``src.cli``) มีภาพมุมละใบ ภาพใบนั้นจึงต้องรับหน้าที่
ทุกอย่างพร้อมกัน — บอกเส้นขอบ บอกสี บอกความขรุขระของผิว ทั้งที่ภาพวาด
ใบเดียวทำสามอย่างนี้ได้ไม่ดีเท่ากันเลย เงาที่วาดไว้สวย ๆ กลายเป็นหลุมตอน
อ่านความลึก พื้นหลังฉากวัดดอกทำให้ขอบเสื้อไม่คม

โมเดลที่ 4 ใช้ชุดภาพ *สี่ชุด* ของมุมเดียวกัน ชุดละหน้าที่

=========== ============================== ==========================================
ชั้น        โฟลเดอร์                        ใช้ทำอะไร
=========== ============================== ==========================================
``orig``    ``input/orig``                 ภาพต้นฉบับ: เส้นขอบ + ลากหาง + ผังฝ่าเท้า
``base``    ``input/base mesh``            เรนเดอร์ดินเหนียว: เส้นขอบที่คมที่สุด และ
                                           *แสงเงาที่เป็นรูปทรงล้วน ๆ* (ไม่มีสีมาปน)
``albedo``  ``input/albedo``               สีแบนไม่มีแสงเงา: สีฐานของวัสดุ
``photo``   ``input/photogrammetry``       เรนเดอร์เสมือนถ่ายจริง: ลายผิวความถี่สูง
=========== ============================== ==========================================

สองเรื่องที่ต้องแก้ก่อนจะใช้ประโยชน์จากการซ้อนชั้นได้

1. **สี่ชุดไม่ได้อยู่ตำแหน่งเดียวกัน** แม้ ``preprocess.normalize_view`` จะย่อ
   ให้ความสูงเท่ากันแล้ว แต่สัดส่วนภายใน (หัวโตแค่ไหน ขาแยกแค่ไหน) ยังต่างกัน
   อยู่ ชั้นที่ไม่ตรงจะลากสีไปแปะผิดที่ ขั้น :func:`register` จึงหา
   (มาตราส่วน, เลื่อน x, เลื่อน y) ที่ทำให้แต่ละชั้นทับชั้นอ้างอิงได้ดีที่สุด

2. **บางชั้นไม่ได้วาดมุมเดียวกันจริง ๆ** ในชุดตัวอย่าง ภาพข้างสองใบของชั้น
   ``photo`` ไม่ใช่มุม 90° — หางสะบัดไปคนละทาง ตัวเอียงไปคนละองศา วัด IoU
   เทียบชั้นอื่นได้แค่ 0.60 ขณะที่มุมหน้า/หลังได้ 0.94  ถ้าเชื่อชื่อไฟล์แล้ว
   ใช้ทั้งชุด ลายผิวจะถูกลากไปแปะเหลื่อมทั้งด้านข้าง :func:`register` จึง
   *ตัดชั้นนั้นทิ้งเฉพาะมุมนั้น* แล้วให้ชั้นสำรองมาแทน (ดู :func:`texture_views`)

ผลลัพธ์ของโมดูลนี้คือ

* ``masks`` — เงา "เสียงข้างมาก" ต่อมุม ใช้เฉือน hull  (:func:`consensus`)
* ชุด :class:`~src.preprocess.PreparedView` ต่อชั้น ที่ทุกชั้นใช้เงาใบเดียวกัน
  ต่างกันแค่สี — ขั้น bake จึงได้ texture ที่ลงทะเบียนตรงกันทุกใบ
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .preprocess import PreparedView, prepare_views

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# นิยามชั้น
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerSpec:
    """ชั้นหนึ่งของชุดภาพ — ชื่อโฟลเดอร์ที่ยอมรับ และหน้าที่ของมัน"""

    name: str
    aliases: tuple[str, ...]
    job: str
    required: bool = True


LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec("orig", ("orig", "original", "art", "concept"),
              "เส้นขอบ + ลากหาง + ผังฝ่าเท้า"),
    LayerSpec("base", ("base mesh", "basemesh", "base_mesh", "clay"),
              "เส้นขอบ + ความลึกจากแสงเงา"),
    LayerSpec("albedo", ("albedo", "basecolor", "base_color", "flat"),
              "สีฐานที่ไม่มีแสงเงาติดมา"),
    LayerSpec("photo", ("photogrammetry", "photogram", "photo", "scan"),
              "ลายผิวความถี่สูง"),
)

# ชั้นที่ถือเป็น "ตัวตั้ง" ของระบบพิกัด — ชั้นอื่นถูกดัดมาหาชั้นนี้
#
# เลือก orig เพราะเป็นภาพที่เส้นทางเดิมใช้มาตลอด ตัวเลขทุกตัวใน README
# (แนวคาง ความกว้างคอ ผังฝ่าเท้า) จึงเทียบกับระบบพิกัดเดียวกันได้ทันที
REFERENCE = "orig"

# ลำดับชั้นสำรองเวลาชั้นหนึ่งใช้มุมนั้นไม่ได้ — ไล่จากซ้ายไปขวา
FALLBACK = ("albedo", "orig", "base", "photo")


@dataclass
class Layer:
    """ชั้นหนึ่งที่โหลด segment และดัดเข้าหาชั้นอ้างอิงเรียบร้อยแล้ว"""

    spec: LayerSpec
    folder: Path
    views: dict[str, PreparedView]
    iou: dict[str, float] = field(default_factory=dict)
    fit: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    def agrees(self, view: str, min_iou: float) -> bool:
        """ชั้นนี้วาดมุมนี้ตรงกับชั้นอื่นพอจะเชื่อได้ไหม"""
        return self.iou.get(view, 0.0) >= min_iou


# ---------------------------------------------------------------------------
# หาโฟลเดอร์ + โหลด
# ---------------------------------------------------------------------------


def find_layers(input_dir: Path) -> dict[str, Path]:
    """จับคู่โฟลเดอร์ย่อยใน ``input/`` กับชื่อชั้น (ไม่สนตัวพิมพ์ใหญ่เล็ก)"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ input: {input_dir}")

    dirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    found: dict[str, Path] = {}
    for spec in LAYER_SPECS:
        for d in dirs:
            key = d.name.lower().replace("-", " ").replace("_", " ")
            if any(a.replace("_", " ") == key or a.replace("_", " ") in key
                   for a in spec.aliases):
                found[spec.name] = d
                break

    missing = [s.name for s in LAYER_SPECS if s.required and s.name not in found]
    if missing:
        raise FileNotFoundError(
            "หาโฟลเดอร์ของชั้นเหล่านี้ไม่เจอใน " + str(input_dir) + ": "
            + ", ".join(missing) + "\n"
            "ชื่อที่ยอมรับ: "
            + "; ".join(f"{s.name} <- {'/'.join(s.aliases)}" for s in LAYER_SPECS)
            + "\nโฟลเดอร์ที่เจอ: " + (", ".join(d.name for d in dirs) or "(ไม่มี)")
        )
    return found


def load(input_dir: Path, cfg: PipelineConfig) -> dict[str, Layer]:
    """โหลดทุกชั้น segment + normalize แยกกันชั้นต่อชั้น

    แยก ``prepare_views`` ต่อชั้นโดยตั้งใจ ไม่ใช่โหลดรวมแล้วค่อยแบ่ง เพราะ
    ขั้น normalize ต้องวัดความสูงกับจุดกึ่งกลางจาก mask *ของชุดนั้น* และขั้น
    เดาทิศหันหน้าของมุมข้างก็ต้องตัดสินจากภาพของชุดนั้นเอง ถ้าเอาค่าของชุด
    หนึ่งไปใช้กับอีกชุด ความคลาดจะกลายเป็นการเลื่อนทั้งชุดที่มองไม่เห็น
    """
    folders = find_layers(input_dir)
    layers: dict[str, Layer] = {}
    for spec in LAYER_SPECS:
        if spec.name not in folders:
            continue
        folder = folders[spec.name]
        log.info("\nชั้น %-7s <- %s/   (%s)", spec.name, folder.name, spec.job)
        views = prepare_views(folder, cfg)
        layers[spec.name] = Layer(spec, folder, views)

    if REFERENCE not in layers:
        raise ValueError(f"ต้องมีชั้น {REFERENCE} เป็นตัวตั้งของระบบพิกัด")
    return layers


# ---------------------------------------------------------------------------
# ดัดแต่ละชั้นเข้าหาชั้นอ้างอิง
# ---------------------------------------------------------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def _warp(img: np.ndarray, scale: float, dx: float, dy: float,
          nearest: bool = False) -> np.ndarray:
    """ย่อ/ขยายรอบจุดกึ่งกลาง canvas แล้วเลื่อน — ขนาดภาพคงเดิม"""
    h, w = img.shape[:2]
    m = np.array([[scale, 0.0, (1 - scale) * w * 0.5 + dx],
                  [0.0, scale, (1 - scale) * h * 0.5 + dy]], np.float32)
    flag = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(img, m, (w, h), flags=flag,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _fit_one(ref: np.ndarray, mov: np.ndarray, scales: np.ndarray,
             limit: int) -> tuple[float, float, float, float]:
    """หา (scale, dx, dy, iou) ที่ทำให้ ``mov`` ทับ ``ref`` ได้ดีที่สุด

    ค้นบน mask ที่ย่อลงก่อน (IoU ของเงาเป็นฟังก์ชันเรียบมาก การค้นหยาบจึง
    ไม่พลาดยอด) แล้วค่อยละเอียดรอบคำตอบหยาบที่ความละเอียดเต็ม
    """
    full = ref.shape[0]
    small = 256
    k = small / full
    rs = cv2.resize(ref.astype(np.uint8), (small, small),
                    interpolation=cv2.INTER_AREA) > 0
    ms = cv2.resize(mov.astype(np.uint8), (small, small),
                    interpolation=cv2.INTER_AREA) > 0

    best = (1.0, 0.0, 0.0, _iou(rs, ms))
    lim_s = max(2, int(limit * k))
    for s in scales:
        warped = _warp(ms.astype(np.uint8), s, 0, 0, nearest=True) > 0
        for dy in range(-lim_s, lim_s + 1, 2):
            for dx in range(-lim_s, lim_s + 1, 2):
                shifted = np.roll(warped, (dy, dx), axis=(0, 1))
                v = _iou(rs, shifted)
                if v > best[3]:
                    best = (s, dx / k, dy / k, v)

    # รอบละเอียด: ที่ความละเอียดเต็ม รอบคำตอบหยาบ +-3 px และ +-1 ขั้นมาตราส่วน
    s0, dx0, dy0, _ = best
    step = float(scales[1] - scales[0]) if len(scales) > 1 else 0.0
    fine = (1.0, 0.0, 0.0, 0.0)
    for s in (s0 - step * 0.5, s0, s0 + step * 0.5):
        warped = _warp(mov.astype(np.uint8), s, 0, 0, nearest=True) > 0
        for dy in range(int(dy0) - 4, int(dy0) + 5, 2):
            for dx in range(int(dx0) - 4, int(dx0) + 5, 2):
                v = _iou(ref, np.roll(warped, (dy, dx), axis=(0, 1)))
                if v > fine[3]:
                    fine = (s, float(dx), float(dy), v)
    return fine


def register(layers: dict[str, Layer], cfg: PipelineConfig,
             max_scale: float = 0.08, ref_name: str = REFERENCE) -> list[str]:
    """ดัดทุกชั้นเข้าหาชั้นอ้างอิง แล้วบันทึก IoU ที่ทำได้ของแต่ละมุม

    แก้ *ในที่* — หลังเรียกแล้ว ``layer.views[v].rgb`` กับ ``.mask`` อยู่ใน
    ระบบพิกัดเดียวกับชั้นอ้างอิงทุกใบ
    """
    ref = layers[ref_name]
    limit = max(6, int(0.05 * cfg.canvas))
    scales = np.arange(1.0 - max_scale, 1.0 + max_scale + 1e-9, 0.02)
    notes: list[str] = []

    for v, rv in ref.views.items():
        ref.iou[v] = 1.0
        ref.fit[v] = (1.0, 0.0, 0.0)

    for name, layer in layers.items():
        if name == ref_name:
            continue
        for v, view in layer.views.items():
            if v not in ref.views:
                continue
            base = _iou(ref.views[v].mask, view.mask)
            s, dx, dy, iou = _fit_one(ref.views[v].mask, view.mask, scales, limit)
            if iou > base:
                view.mask = _warp(view.mask.astype(np.uint8), s, dx, dy,
                                  nearest=True) > 0
                view.rgb = _warp(view.rgb, s, dx, dy)
                layer.fit[v] = (s, dx, dy)
            else:
                iou = base
                layer.fit[v] = (1.0, 0.0, 0.0)
            layer.iou[v] = iou
            log.info("ดัดชั้น %-7s มุม %-6s: scale %.3f เลื่อน (%+.0f, %+.0f) "
                     "-> IoU %.3f (เดิม %.3f)", name, v, s, dx, dy, iou, base)

    return notes


# ---------------------------------------------------------------------------
# เงาเสียงข้างมาก
# ---------------------------------------------------------------------------


def consensus(layers: dict[str, Layer], min_iou: float = 0.72,
              agree: float = 0.5) -> tuple[dict[str, np.ndarray], list[str]]:
    """รวมเงาของทุกชั้นที่เชื่อได้ ให้เหลือเงาเดียวต่อมุม

    ทำไมต้องโหวต ไม่ใช่เลือกชั้นที่ดีที่สุดชั้นเดียว: ขั้นแยกพื้นหลังพลาด
    คนละที่กันในแต่ละชั้น (ชั้นดินเหนียวกินเงาบนพื้นเข้ามา ชั้นภาพถ่ายทิ้ง
    ปลายหางที่กลืนกับพื้นสตูดิโอ) ความพลาดพวกนี้ไม่ซ้ำที่กัน เสียงข้างมากจึง
    ลบมันทิ้งได้ ขณะที่การเลือกชั้นเดียวต้องยอมรับความพลาดของชั้นนั้นทั้งหมด

    และทำไมต้อง *เสียงข้างมาก* ไม่ใช่ union หรือ intersection: visual hull
    เอา intersection ข้าม *มุม* อยู่แล้ว ถ้าภายในมุมเดียวกันยังเอา
    intersection อีก ความพลาดของทุกชั้นจะบวกกันหมด (เงาเล็กลงเรื่อย ๆ)
    ส่วน union ก็กลับกัน — เงาพองตามความพลาดของชั้นที่แย่ที่สุด
    """
    masks: dict[str, np.ndarray] = {}
    notes: list[str] = []

    for v in layers[REFERENCE].views:
        voters = [l for l in layers.values()
                  if v in l.views and l.agrees(v, min_iou)]
        dropped = [l.name for l in layers.values()
                   if v in l.views and not l.agrees(v, min_iou)]
        if dropped:
            why = ", ".join(f"{n} (IoU {layers[n].iou.get(v, 0):.2f})"
                            for n in dropped)
            notes.append(f"มุม {v}: ไม่ใช้ชั้น {why} เฉือนรูปทรง "
                         f"— ชั้นนั้นไม่ได้วาดมุมเดียวกัน")
            log.warning("มุม %-6s ตัดชั้น %s ออกจากการโหวต", v, why)

        votes = np.zeros(layers[REFERENCE].views[v].mask.shape, np.int16)
        for l in voters:
            votes += l.views[v].mask.astype(np.int16)
        need = max(1, int(np.ceil(len(voters) * agree)))
        masks[v] = votes >= need
        log.info("มุม %-6s เงาเสียงข้างมาก: %d ชั้นโหวต ต้องได้ %d เสียง "
                 "-> %d พิกเซล", v, len(voters), need,
                 int(np.count_nonzero(masks[v])))
    return masks, notes


# ---------------------------------------------------------------------------
# ประกอบเป็นชุด PreparedView ให้ขั้นถัดไป
# ---------------------------------------------------------------------------


def _clone(view: PreparedView, mask: np.ndarray,
           rgb: np.ndarray, source: Path) -> PreparedView:
    out = copy.copy(view)
    out.mask = mask
    out.rgb = rgb
    out.source = source
    return out


def carving_views(layers: dict[str, Layer],
                  masks: dict[str, np.ndarray]) -> dict[str, PreparedView]:
    """ชุดมุมสำหรับเฉือน hull — เงาเสียงข้างมาก + ภาพต้นฉบับไว้ลากหาง

    ``rgb`` ตรงนี้ไปโผล่ที่ ``tailtrace`` อย่างเดียว (ขั้นเฉือนใช้แต่ mask)
    จึงใช้ของชั้น ``orig`` ซึ่งเป็นชุดที่ค่าตั้งต้นของการลากหางถูกจูนมากับมัน
    """
    ref = layers[REFERENCE]
    return {v: _clone(ref.views[v], masks[v], ref.views[v].rgb,
                      ref.views[v].source)
            for v in ref.views}


def texture_views(layers: dict[str, Layer], name: str,
                  masks: dict[str, np.ndarray], min_iou: float = 0.72,
                  ) -> tuple[dict[str, PreparedView], list[str]]:
    """ชุดมุมสำหรับ bake ของชั้นหนึ่ง — สีของชั้นนั้น บนเงาเสียงข้างมาก

    ใช้ *เงาเดียวกัน* ทุกชั้นโดยตั้งใจ: ถ้าแต่ละชั้นใช้เงาของตัวเอง texel
    ริมขอบจะถูกเลือกมุมคนละมุมในแต่ละแผนที่ แล้วสีฐานกับ normal จะเหลื่อมกัน
    เป็นเส้นขอบคู่รอบตัว

    มุมที่ชั้นนี้ไม่ตรงกับชั้นอื่น (เช่นภาพข้างของชั้นภาพถ่าย ซึ่งเป็นคนละ
    มุมกันจริง ๆ) ถูกแทนด้วยชั้นสำรองที่ตรง — ดีกว่าลากลายไปแปะเหลื่อม
    """
    layer = layers[name]
    ref = layers[REFERENCE]
    out: dict[str, PreparedView] = {}
    notes: list[str] = []

    for v in ref.views:
        src = layer if (v in layer.views and layer.agrees(v, min_iou)) else None
        if src is None:
            for alt in FALLBACK:
                if alt == name or alt not in layers:
                    continue
                if v in layers[alt].views and layers[alt].agrees(v, min_iou):
                    src = layers[alt]
                    break
        if src is None:
            src = ref
        if src is not layer:
            notes.append(f"texture ชั้น {name} มุม {v}: ยืมสีจากชั้น {src.name}")
            log.warning("ชั้น %-7s มุม %-6s ใช้ไม่ได้ (IoU %.2f) -> ยืมชั้น %s",
                        name, v, layer.iou.get(v, 0.0), src.name)
        out[v] = _clone(ref.views[v], masks[v], src.views[v].rgb,
                        src.views[v].source)
    return out, notes


# ---------------------------------------------------------------------------
# ภาพตรวจงาน
# ---------------------------------------------------------------------------


def debug_sheet(layers: dict[str, Layer], masks: dict[str, np.ndarray],
                path: Path, cell: int = 300) -> Path:
    """แผ่นตรวจ: แถวละชั้น ช่องละมุม — สีของชั้นนั้น ทับด้วยขอบเงาที่โหวตได้

    เขียว = ขอบของเงาเสียงข้างมาก, แดง = ขอบของชั้นนั้นเอง ตรงไหนที่สองเส้น
    แยกกันคือตรงที่ชั้นนั้นไม่เห็นด้วยกับชุด
    """
    views = list(layers[REFERENCE].views)
    rows, cols = len(layers), len(views)
    lab = 22
    sheet = np.full((rows * (cell + lab), cols * cell, 3), 22, np.uint8)

    for r, (name, layer) in enumerate(layers.items()):
        for c, v in enumerate(views):
            if v not in layer.views:
                continue
            rgb = layer.views[v].rgb.copy()
            rgb[~layer.views[v].mask] //= 4
            edge_self = cv2.Canny(layer.views[v].mask.astype(np.uint8) * 255,
                                  50, 150) > 0
            edge_cons = cv2.Canny(masks[v].astype(np.uint8) * 255, 50, 150) > 0
            rgb[edge_self] = (255, 60, 60)
            rgb[edge_cons] = (60, 255, 60)
            small = cv2.resize(rgb, (cell, cell), interpolation=cv2.INTER_AREA)
            oy, ox = r * (cell + lab) + lab, c * cell
            sheet[oy:oy + cell, ox:ox + cell] = small
            cv2.putText(sheet, f"{name} / {v}  IoU {layer.iou.get(v, 0):.2f}",
                        (ox + 6, oy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (225, 225, 225), 1, cv2.LINE_AA)

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", sheet[:, :, ::-1])
    if not ok:
        raise RuntimeError("เข้ารหัสแผ่นตรวจไม่สำเร็จ")
    buf.tofile(str(path))
    log.info("แผ่นตรวจชั้นภาพ -> %s", path)
    return path
