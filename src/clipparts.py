r"""แบ่งตัวละครเป็นส่วน ๆ ตามโครงกระดูก เพื่อให้แต่ละส่วนถูกปั้นด้วยกฎของตัวเอง

ทำไมต้องแบ่ง
------------
กฎชุดเดียวใช้ทั้งตัวไม่ได้ เพราะแต่ละส่วนมีขนาดรายละเอียดต่างกันเป็นสิบเท่า

  * **นิ้วมือ** กว้างราว 2 ซม. ร่องระหว่างนิ้วกว้าง 5 มม.
  * **ลำตัว** กว้าง 50 ซม. รายละเอียดที่สำคัญคือรอยยับเสื้อระดับ 2-3 ซม.

การเกลาผิวด้วย sigma เดียวกันทั้งตัวจึงเป็นการเลือกว่าจะเสียอะไร: ค่าที่พอดี
กับลำตัวกลบนิ้วหายทั้งมือ ส่วนค่าที่พอดีกับนิ้วทิ้งเสียงรบกวนไว้เต็มลำตัว
ที่หนักกว่านั้นคือขั้น opening ที่ลบแผ่นบาง ๆ บนพื้น — มันลบทุกอย่างที่บาง
กว่าสี่ voxel ซึ่งนิ้วมือที่ความละเอียด 320 หนาแค่สามเท่านั้นเอง นิ้วจึงถูก
ลบทิ้งด้วยมือเราเองตั้งแต่ก่อนจะได้เห็น

โมดูลนี้ตอบคำถามเดียว: "จุดนี้อยู่ส่วนไหนของตัว" แล้วปล่อยให้ขั้น carve กับ
ขั้นดันผิวไปเลือกกฎของตัวเอง
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartRule:
    """กฎการปั้นของส่วนหนึ่ง ระยะเป็นมิลลิเมตรบนตัวละครขนาดจริง"""

    name: str
    radius_mm: float      # รัศมีของเขตนี้รอบกระดูก
    protect: bool         # กันจากการ opening/เกลาสนามตอน carve
    edge_mm: float        # ขนาดสามเหลี่ยมที่ต้องการตอนซอย
    depth_mm: float       # ความลึกที่ยอมให้ดันผิวตามความเข้มของสี


# ยิ่งส่วนเล็ก ยิ่งต้องการสามเหลี่ยมถี่และยอมให้ร่องลึกเป็นสัดส่วนที่มากกว่า
DEFAULT_RULES: tuple[PartRule, ...] = (
    PartRule("นิ้ว/มือ",  110.0, True,   2.5, 3.5),
    PartRule("นิ้ว/เท้า", 120.0, True,   3.0, 3.0),
    PartRule("ข้อมือ",     80.0, True,   4.0, 2.5),
    PartRule("คอ",        110.0, True,   5.0, 3.0),
    PartRule("หัว",       170.0, True,   4.0, 2.5),
    PartRule("แขน",       130.0, False,  7.0, 2.0),
    PartRule("ขา",        160.0, False,  9.0, 1.5),
    PartRule("ลำตัว",     300.0, False, 12.0, 1.5),
)
HAND, FOOT, WRIST, NECK, HEAD, ARM, LEG, TORSO = range(8)

# ชื่อข้อต่อแบบ Mixamo เรียกตามกระดูกที่อยู่ *เหนือ* มันขึ้นไป: `LeftHand` คือ
# ข้อมือ ไม่ใช่ฝ่ามือ และกระดูกที่ลากจาก `LeftHand` ไป `LeftForeArm` คือปลายแขน
# ไม่ใช่มือ  ถ้าจับส่วนจากชื่อกระดูกตรง ๆ จะเลื่อนไปหนึ่งข้อทั้งแขนทั้งขา
#
# มือกับเท้าจึงต้องเป็น *ทรงกลมรอบข้อต่อ* ไม่ใช่กระดูก เพราะไม่มีข้อต่อปลายมือ
# ให้ลากไปหา
_JOINT_PART: tuple[tuple[str, int], ...] = (
    ("Hand", HAND), ("ToeBase", FOOT), ("Foot", FOOT),
    ("HeadTop_End", HEAD), ("Head", HEAD), ("Neck", NECK),
)
_BONE_PART: tuple[tuple[str, int], ...] = (
    ("Hand", WRIST),          # ข้อมือ -> ข้อศอก = ปลายแขน
    ("ToeBase", FOOT), ("Foot", LEG), ("Neck", NECK), ("Head", HEAD),
    ("ForeArm", ARM), ("Arm", ARM), ("Shoulder", ARM),
    ("Leg", LEG), ("Spine", TORSO),
)


def _bones(joints: dict[str, list[float]], parents: dict[str, str | None],
           ) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """คืน [(ส่วน, ปลาย A, ปลาย B)] — ถ้า A เท่ากับ B คือทรงกลมรอบข้อต่อ"""
    out = []
    for name, pos in joints.items():
        part = next((p for key, p in _JOINT_PART if key in name), None)
        if part is not None:
            a = np.asarray(pos, float)
            out.append((part, a, a))

    for name, parent in parents.items():
        if parent is None or name not in joints or parent not in joints:
            continue
        part = next((p for key, p in _BONE_PART if key in name), TORSO)
        out.append((part, np.asarray(joints[name], float),
                    np.asarray(joints[parent], float)))
    return out


def _dist_to_segment(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab) or 1e-9
    t = np.clip((pts - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def label(points: np.ndarray, joints: dict, parents: dict, height_m: float,
          unit_m: float, rules: tuple[PartRule, ...] = DEFAULT_RULES,
          ) -> np.ndarray:
    """บอกว่าแต่ละจุดอยู่ส่วนไหน คืน array ของ index ใน ``rules``

    ``points`` อยู่ในหน่วยเดียวกับ ``joints`` และ ``unit_m`` บอกว่าหนึ่งหน่วย
    นั้นเท่ากับกี่เมตร ใช้แปลงรัศมีเป็นมิลลิเมตรให้ตรงกับตัวละครขนาดจริง

    จุดหนึ่งอาจอยู่ในเขตของหลายส่วน (ข้อมืออยู่ในเขตแขนด้วย) ตัดสินด้วยลำดับ
    ใน ``rules`` ซึ่งเรียงจากส่วนเล็กไปใหญ่ — ส่วนที่เล็กกว่าชนะเสมอ เพราะมัน
    คือส่วนที่ต้องการกฎพิเศษ
    """
    out = np.full(len(points), TORSO, np.uint8)
    bones = _bones(joints, parents)
    if not bones:
        return out

    for part in range(len(rules) - 1, -1, -1):
        radius = rules[part].radius_mm / 1000.0 / unit_m
        hit = np.zeros(len(points), bool)
        for bone_part, a, b in bones:
            if bone_part != part:
                continue
            hit |= _dist_to_segment(points, a, b) <= radius
        out[hit] = part

    counts = np.bincount(out, minlength=len(rules))
    log.info("แบ่งส่วนตามโครงกระดูก: %s",
             ", ".join(f"{rules[i].name} {100 * c / len(points):.0f}%"
                       for i, c in enumerate(counts) if c))
    return out


def per_point(points: np.ndarray, labels: np.ndarray, field: str,
              rules: tuple[PartRule, ...] = DEFAULT_RULES) -> np.ndarray:
    """ดึงค่าของกฎ (edge_mm / depth_mm) ออกมาเป็นค่าต่อจุด"""
    table = np.array([getattr(r, field) for r in rules], np.float64)
    return table[np.clip(labels, 0, len(rules) - 1)]


def voxel_protect(shape: tuple[int, int, int], joints: dict, parents: dict,
                  unit_m: float, rules: tuple[PartRule, ...] = DEFAULT_RULES,
                  ) -> np.ndarray:
    """เขตที่ต้องกันจากการ opening/เกลาสนาม บนกริด voxel (iy, ix, iz)

    คิดทีละกระดูกในกล่องของตัวเองเท่านั้น ไม่ไล่ทั้งกริด — กริด 320^3 มี
    voxel สามสิบล้านตัว การวัดระยะถึงกระดูกยี่สิบชิ้นทุกตัวเปลืองเปล่า
    ในเมื่อเขตที่สนใจกินพื้นที่ไม่ถึงหนึ่งเปอร์เซ็นต์
    """
    ny, nx, nz = shape
    out = np.zeros(shape, bool)
    for part, a, b in _bones(joints, parents):
        if not rules[part].protect:
            continue
        r = rules[part].radius_mm / 1000.0 / unit_m
        # ข้อต่ออยู่ในพิกัด (ix, iy, iz) ส่วนกริดเรียง (iy, ix, iz)
        lo = np.minimum(a, b) - r
        hi = np.maximum(a, b) + r
        y0, y1 = max(0, int(lo[1])), min(ny, int(hi[1]) + 2)
        x0, x1 = max(0, int(lo[0])), min(nx, int(hi[0]) + 2)
        z0, z1 = max(0, int(lo[2])), min(nz, int(hi[2]) + 2)
        if y0 >= y1 or x0 >= x1 or z0 >= z1:
            continue
        gy, gx, gz = np.mgrid[y0:y1, x0:x1, z0:z1]
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(float)
        near = _dist_to_segment(pts, a, b).reshape(gy.shape) <= r
        out[y0:y1, x0:x1, z0:z1] |= near
    log.info("เขตที่กันไว้ไม่ให้ถูกกัด/เกลา: %.2f%% ของกริด",
             100 * float(out.mean()))
    return out


def protected_mask(points: np.ndarray, labels: np.ndarray,
                   rules: tuple[PartRule, ...] = DEFAULT_RULES) -> np.ndarray:
    """จุดไหนอยู่ในส่วนที่ต้องกันจากการ opening/เกลาสนาม"""
    table = np.array([r.protect for r in rules], bool)
    return table[np.clip(labels, 0, len(rules) - 1)]
