r"""แยกวัสดุจากลายบน texture แล้วใส่ลายผิวกับความมันของวัสดุนั้น ๆ

ปัญหาที่แก้
-----------
คลิปต้นทางกว้าง 1280 พิกเซล ตัวละครสูงราว 660 พิกเซล = 1 พิกเซลต่อราว 3 มม.
เกล็ดกว้าง 10-20 มม. จึงยังพอเห็นเป็นลาย แต่ *ร่องระหว่างเกล็ด* ลายทอผ้ายีนส์
(ราว 1 มม.) และเสี้ยนหนังกระเป๋า เล็กกว่าพิกเซลเดียว — ไม่ได้อยู่ในคลิปตั้งแต่
ต้น ไม่ว่าจะเฉือน voxel ละเอียดแค่ไหนหรือฉายสีดีแค่ไหนก็ดึงออกมาไม่ได้

สิ่งที่ *อ่านได้* จากภาพคือ "ตรงนี้เป็นวัสดุอะไร" — สีกับความมันบอกได้ว่าเป็น
เกล็ด ผ้า หนัง หรือโลหะ เมื่อรู้แล้วก็สังเคราะห์ลายละเอียดของวัสดุนั้นใส่กลับ
เข้าไปได้ ซึ่งเป็นวิธีที่ artist เกมทำกันมาตลอด: ภาพถ่ายให้ *สี* ส่วนลายผิว
ระดับมิลลิเมตรมาจากไลบรารีวัสดุ ไม่ได้มาจากภาพถ่าย

ได้อะไรกลับมาสองอย่าง
  * normal map ละเอียดของวัสดุ (เกล็ดหกเหลี่ยม ลายทอผ้า เสี้ยนหนัง)
    ผสมเข้ากับ normal map ที่ดึงมาจากลายจริงในภาพ
  * roughness map — หัวเข็มขัดมัน หนังด้าน ผ้าด้านสนิท เกล็ดกึ่งมัน
    ซึ่งมักเป็นตัวที่ทำให้ "ดูเป็นวัสดุ" มากกว่าความขรุขระเสียอีก

ข้อจำกัด: ลายที่ได้เป็นลาย *สังเคราะห์* ไม่ใช่เกล็ดจริงของตัวละครตัวนี้
ตำแหน่งเกล็ดแต่ละเกล็ดจึงไม่ตรงกับในภาพ และลายไม่ได้ไหลตามรูปร่างของแขนขา
เพราะวางตามกริดของ UV atlas ไม่ได้วางตาม flow map ที่ artist จะวาดให้
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Material:
    """นิยามวัสดุหนึ่งชนิด — ลายผิว ความมัน และวิธีจำแนกจากสี"""

    name: str
    pattern: str        # scales | weave | grain | flat
    size: float         # ขนาดลายเป็นจำนวน texel
    height: float       # ความลึกของลาย (เทียบกันเอง)
    roughness: float    # 0 = เงาเหมือนกระจก, 1 = ด้านสนิท


# ค่าเริ่มต้นที่ออกแบบมาสำหรับตัวละครคนจริง/สัตว์ในชุดผ้า ซึ่งครอบคลุม
# ตัวละครเกมส่วนใหญ่ — ปรับได้ด้วยการส่ง list ของตัวเองเข้า classify()
DEFAULT_MATERIALS: tuple[Material, ...] = (
    # ขนาดเป็นจำนวน texel บน atlas 2048 ของตัวละครสูงราวสองเมตร
    # = ราว 1 มม. ต่อ texel เกล็ด 20 texel จึงกว้างราว 2 ซม. ตามของจริง
    Material("เกล็ด/ผิว", "scales", 20.0, 1.00, 0.55),
    Material("ผ้า",      "weave",   5.5, 0.30, 0.88),
    Material("หนัง",     "grain",   3.5, 0.55, 0.70),
    Material("โลหะ",     "flat",    1.0, 0.00, 0.35),
)
SKIN, CLOTH, LEATHER, METAL = range(4)


# ---------------------------------------------------------------------------
# แยกวัสดุจากสี
# ---------------------------------------------------------------------------


def flatten_shading(texture: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """ลบแสงเงาของทั้งก้อนออก เหลือแต่ "สีของวัสดุ"

    จำเป็นก่อนแยกวัสดุ เพราะ texture ที่ bake มาจากคลิปมีแสงเงาติดมาด้วย
    แขนข้างที่อยู่ในเงากับแขนข้างที่โดนคบไฟมีค่าความสว่างต่างกันเท่ากับ
    ความต่างระหว่างหนังกับผ้าเลย ถ้าไม่ลบออกก่อน k-means จะไปจัดกลุ่มตาม
    "สว่าง/มืด" แทนที่จะจัดตามวัสดุ แล้วแขนข้างที่อยู่ในเงาจะกลายเป็นหนัง

    หารด้วยตัวเองที่เบลอแรง ๆ — แสงเงาเปลี่ยนช้าตามรูปร่าง ส่วนสีของวัสดุ
    เปลี่ยนเร็วตรงรอยต่อเสื้อกับแขน การหารจึงคงสีวัสดุไว้แต่ลบเงาทิ้ง
    """
    lab = cv2.cvtColor(texture, cv2.COLOR_RGB2LAB).astype(np.float32)
    lum = lab[:, :, 0]
    base = cv2.GaussianBlur(lum, (0, 0), 48.0)
    target = float(lum[filled].mean()) if filled.any() else 128.0
    # ปรับเฉพาะความสว่าง ไม่แตะสี — ถ้าหารทั้งสามช่อง สีส้มของเกล็ดจะถูกหาร
    # ด้วยพื้นหลังสีส้มของตัวเองจนกลายเป็นสีเทา แล้วแยกจากผ้าไม่ออกอีกเลย
    lab[:, :, 0] = np.clip(lum / np.maximum(base, 4.0) * target, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def classify(texture: np.ndarray, filled: np.ndarray, clusters: int = 7,
             smooth: int = 9) -> np.ndarray:
    """จัดกลุ่มสีบน texture แล้วตัดสินว่าแต่ละกลุ่มเป็นวัสดุอะไร

    ใช้ k-means ใน Lab ก่อน แล้วค่อยตัดสินทีละกลุ่มจากเฉดสี ไม่ตัดสินทีละ
    พิกเซล เพราะพิกเซลเดียวมีเสียงรบกวนจาก JPEG กับเงาเยอะเกินกว่าจะเชื่อได้
    ส่วนค่าเฉลี่ยของกลุ่มที่มีเป็นแสนพิกเซลนิ่งพอจะแยกผ้ากับหนังออกจากกัน

    กฎการตัดสินตั้งใจให้กว้าง ๆ ไม่ผูกกับตัวละครตัวใดตัวหนึ่ง
      * อิ่มตัวต่ำ สว่างสูง         -> โลหะ (เงินกับเหล็กไม่มีสีของตัวเอง)
      * เฉดอุ่น (แดง-ส้ม-เหลือง)    -> ผิว/เกล็ด
      * เฉดอุ่นแต่เข้ม              -> หนัง
      * ที่เหลือ                    -> ผ้า
    """
    h, w = filled.shape
    flat = flatten_shading(texture, filled)
    px = flat[filled].astype(np.float32)
    if len(px) < clusters * 4:
        return np.full((h, w), CLOTH, np.uint8)

    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8),
                       cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 1.0)
    _, idx, _ = cv2.kmeans(lab, clusters, None, crit, 3,
                           cv2.KMEANS_PP_CENTERS)
    idx = idx.ravel()

    kinds = np.empty(clusters, np.uint8)
    for k in range(clusters):
        sel = idx == k
        if not sel.any():
            kinds[k] = CLOTH
            continue
        mean = px[sel].mean(axis=0)
        hsv = cv2.cvtColor(mean.reshape(1, 1, 3).astype(np.uint8),
                           cv2.COLOR_RGB2HSV)[0, 0]
        hue, sat, val = float(hsv[0]) * 2.0, float(hsv[1]) / 255, float(hsv[2]) / 255
        warm = hue < 55 or hue > 330

        # ลำดับการถามสำคัญ: เนื้อหนังของสัตว์เลื้อยคลานกับหนังฟอกอยู่ในเฉดสี
        # เดียวกันเกือบหมด แยกกันได้ด้วยความอิ่มตัวเท่านั้น (ผิวสด ๆ อิ่มกว่า)
        # จึงต้องถามผิวก่อนหนัง ไม่งั้นแขนที่อยู่ในเงาจะกลายเป็นกระเป๋าหนัง
        if sat < 0.15 and val > 0.55:
            kinds[k] = METAL
        elif warm and sat > 0.38:
            kinds[k] = SKIN
        elif warm and val < 0.45 and sat > 0.20:
            kinds[k] = LEATHER
        else:
            kinds[k] = CLOTH
        log.info("  กลุ่มสี %d: RGB(%3.0f,%3.0f,%3.0f) hue %3.0f sat %.2f val %.2f "
                 "-> %s (%.1f%% ของผิว)", k, *mean, hue, sat, val,
                 DEFAULT_MATERIALS[kinds[k]].name, 100 * sel.mean())

    labels = np.full((h, w), CLOTH, np.uint8)
    labels[filled] = kinds[idx]
    if smooth > 1:
        # เสียงรบกวนทีละพิกเซลไม่ใช่การเปลี่ยนวัสดุ — เอาเสียงข้างมากของย่าน
        labels = cv2.medianBlur(labels, smooth | 1)
    return labels


# ---------------------------------------------------------------------------
# ลายผิวของแต่ละวัสดุ
# ---------------------------------------------------------------------------


def _scales(shape: tuple[int, int], size: float, seed: int = 7) -> np.ndarray:
    """เกล็ดเรียงแบบหกเหลี่ยม — โดมนูนคั่นด้วยร่อง

    วางเป็นแถวสลับเหลื่อมครึ่งเซลล์ ซึ่งคือการเรียงแบบหกเหลี่ยมที่แน่นที่สุด
    และเป็นแบบที่เกล็ดสัตว์เลื้อยคลานเรียงตัวจริง ๆ
    """
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    row_h = size * 0.866
    v = ys / row_h
    row = np.floor(v)
    off = np.where(row.astype(np.int32) % 2 == 0, 0.0, 0.5)
    u = xs / size - off
    col = np.floor(u)

    du = (u - col - 0.5) * 1.15          # บีบแนวนอนเล็กน้อย เกล็ดจะได้ไม่กลมเป๊ะ
    dv = v - row - 0.5
    d = np.sqrt(du * du + dv * dv) / 0.52

    height = np.clip(1.0 - d * d, 0.0, 1.0) ** 0.6      # โดม
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.82, 1.0, size=(int(h / row_h) + 2, int(w / size) + 2))
    height *= jitter[row.astype(np.int32) % jitter.shape[0],
                     (col.astype(np.int32) + 1) % jitter.shape[1]]
    return height - height.mean()


def _weave(shape: tuple[int, int], size: float) -> np.ndarray:
    """ลายทอผ้า — เส้นยืนกับเส้นพุ่งสลับขึ้นลง บวกแนวทแยงแบบผ้าทวิล"""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    warp = np.sin(2 * np.pi * xs / size)
    weft = np.sin(2 * np.pi * ys / size)
    twill = np.sin(2 * np.pi * (xs + ys) / (size * 2.0))
    return 0.6 * np.maximum(warp, weft) + 0.4 * twill


def _grain(shape: tuple[int, int], size: float, seed: int = 11) -> np.ndarray:
    """เสี้ยนหนัง — noise หลายชั้นแล้วตัดความถี่ต่ำทิ้ง"""
    h, w = shape
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    for octave in range(3):
        step = max(2, int(size * (2 ** octave)))
        small = rng.random((h // step + 2, w // step + 2)).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        amp *= 0.55
    return out - cv2.GaussianBlur(out, (0, 0), size * 4)


def detail_height(labels: np.ndarray,
                  materials: tuple[Material, ...] = DEFAULT_MATERIALS,
                  scale: float = 1.0) -> np.ndarray:
    """สนามความสูงของลายวัสดุทั้งภาพ — แต่ละวัสดุใช้ลายของตัวเอง"""
    shape = labels.shape
    out = np.zeros(shape, np.float32)
    for i, mat in enumerate(materials):
        if mat.height <= 0:
            continue
        sel = labels == i
        if not sel.any():
            continue
        size = max(2.0, mat.size * scale)
        if mat.pattern == "scales":
            pat = _scales(shape, size)
        elif mat.pattern == "weave":
            pat = _weave(shape, size)
        elif mat.pattern == "grain":
            pat = _grain(shape, size)
        else:
            continue
        rms = float(np.sqrt((pat ** 2).mean())) or 1.0
        out[sel] = (pat[sel] / rms) * mat.height
        log.info("  ลาย %-8s %-7s ขนาด %.0f texel ครอบ %.1f%% ของผิว",
                 mat.name, mat.pattern, size, 100 * sel.mean())

    # เกลารอยต่อระหว่างวัสดุ ไม่งั้นขอบระหว่างผ้ากับเกล็ดจะเป็นสันคม
    return cv2.GaussianBlur(out, (0, 0), 0.8)


def roughness_map(labels: np.ndarray, detail: np.ndarray,
                  materials: tuple[Material, ...] = DEFAULT_MATERIALS,
                  ) -> np.ndarray:
    """ภาพ metallicRoughness ตามสเปก glTF — G = roughness, B = metallic

    ไม่ตั้ง metallic ให้ตรงไหนเลยแม้แต่โลหะ เพราะผิวโลหะใน PBR สะท้อนภาพ
    สิ่งแวดล้อมเป็นหลัก ฉากที่ไม่มี environment map จึงเรนเดอร์ออกมาเกือบดำ
    ใช้ roughness ต่ำแทน ได้ไฮไลต์วาวแบบโลหะขัดโดยไม่เสี่ยงกลายเป็นก้อนดำ
    """
    rough = np.empty(labels.shape, np.float32)
    for i, mat in enumerate(materials):
        rough[labels == i] = mat.roughness
    # ร่องของลายหยาบกว่ายอด — ฝุ่นกับคราบลงไปจับในร่องเสมอ
    rough = np.clip(rough - 0.08 * np.clip(detail, -2, 2), 0.05, 1.0)
    rough = cv2.GaussianBlur(rough, (0, 0), 1.0)

    # ครึ่งความละเอียดก็พอ ความมันของวัสดุเปลี่ยนช้ากว่าลายมาก และภาพนี้
    # ไปเพิ่มขนาดไฟล์ GLB โดยไม่ได้อะไรกลับมาถ้าเก็บเต็มความละเอียด
    half = cv2.resize(rough, (labels.shape[1] // 2, labels.shape[0] // 2),
                      interpolation=cv2.INTER_AREA)
    img = np.zeros(half.shape + (3,), np.uint8)
    img[:, :, 1] = np.clip(half * 255, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# รวม normal สองชั้น
# ---------------------------------------------------------------------------


def combine_normals(base: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """ซ้อน normal map สองใบเข้าด้วยกันแบบบวกความชัน

    บวกกันตรง ๆ ในค่าสีไม่ถูก เพราะ normal เป็นเวกเตอร์หน่วย การบวกแล้ว
    normalize ทำให้ลายที่ชันกว่ากลบอีกใบหมด  วิธีที่ถูกคือแปลงกลับเป็น
    *ความชัน* (x/z, y/z) ซึ่งบวกกันได้ตามหลักแคลคูลัส แล้วค่อยแปลงกลับ
    """
    a = (base.astype(np.float32) - 127.5) / 127.5
    b = (detail.astype(np.float32) - 127.5) / 127.5
    az = np.maximum(a[:, :, 2], 1e-3)
    bz = np.maximum(b[:, :, 2], 1e-3)
    gx = a[:, :, 0] / az + b[:, :, 0] / bz
    gy = a[:, :, 1] / az + b[:, :, 1] / bz
    vec = np.dstack([gx, gy, np.ones_like(gx)])
    vec /= np.linalg.norm(vec, axis=2, keepdims=True)
    return np.clip(vec * 127.5 + 127.5, 0, 255).astype(np.uint8)


def debug_sheet(texture: np.ndarray, labels: np.ndarray, detail: np.ndarray,
                path) -> None:
    """ภาพตรวจงาน: สี / วัสดุที่แยกได้ / ลายที่สังเคราะห์ขึ้น"""
    from pathlib import Path

    colors = np.array([[255, 170, 90], [90, 130, 255],
                       [150, 95, 60], [220, 220, 230]], np.uint8)
    seg = colors[np.clip(labels, 0, 3)]
    det = cv2.cvtColor(
        np.clip(detail * 60 + 128, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    k = 520
    tiles = [cv2.resize(t, (k, k), interpolation=cv2.INTER_AREA)
             for t in (texture, seg, det)]
    for img, lab in zip(tiles, ("texture", "material", "detail")):
        cv2.putText(img, lab, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
    sheet = np.hstack(tiles)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", sheet[:, :, ::-1])
    if ok:
        buf.tofile(str(path))
        log.info("เขียนภาพตรวจวัสดุ -> %s", path)
