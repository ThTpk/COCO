"""คลี่ผิว 3 มิติออกเป็นแผ่นแบน (UV) แล้ว bake สีลงเป็นภาพ texture

ทำไมต้องมีขั้นนี้: การเก็บสีไว้ที่ vertex ให้ความละเอียดเท่ากับความถี่ของ vertex
mesh 20,000 จุดบนตัวสูง 2 เมตร = จุดสีหนึ่งจุดต่อผิวราว 1 เซนติเมตร
ลายเกล็ดหรือรอยตะเข็บที่ละเอียดกว่านั้นหายไปตั้งแต่ตอนคำนวณ ทั้งที่ภาพต้นฉบับมีอยู่

texture 2048x2048 ให้จุดสี 4.2 ล้านจุด = ละเอียดกว่าราว 14 เท่าในแต่ละแกน
โดยที่จำนวน polygon เท่าเดิม ต้นทุนตอน render แทบไม่เปลี่ยน
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import PipelineConfig
from .meshing import MeshData
from .preprocess import PreparedView
from .texture import project_colors
from .visual_hull import HullGrid

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# คลี่ผิว
# ---------------------------------------------------------------------------


def unwrap(mesh: MeshData, cfg: PipelineConfig) -> MeshData:
    """คลี่ผิวด้วย xatlas แล้วคืน mesh ที่มีพิกัด UV

    xatlas จะ *ผ่า* ผิวตามแนวที่คลี่แล้วยืดน้อยที่สุด แล้วจัดชิ้นที่คลี่ได้
    (island) ลงในจัตุรัสเดียว vertex ที่อยู่บนรอยผ่าถูกแยกร่างเป็นหลายจุด
    เพราะจุดเดียวกันบนตัวละครต้องมีพิกัด UV คนละที่บนภาพ จำนวน vertex
    จึงเพิ่มขึ้นหลังคลี่ ซึ่งเป็นเรื่องปกติ ไม่ใช่ความผิดพลาด
    """
    try:
        import xatlas
    except ImportError as exc:
        raise RuntimeError(
            "ต้องติดตั้ง xatlas ก่อนใช้โหมด UV:\n"
            r"    .venv\Scripts\python.exe -m pip install xatlas"
        ) from exc

    verts = np.ascontiguousarray(mesh.verts, np.float32)
    faces = np.ascontiguousarray(mesh.faces, np.uint32)

    log.info("กำลังคลี่ผิว %d tris ด้วย xatlas ...", len(faces))
    vmapping, indices, uvs = xatlas.parametrize(verts, faces)

    out = MeshData(
        verts=mesh.verts[vmapping],
        faces=np.asarray(indices, np.int32),
        normals=None if mesh.normals is None else mesh.normals[vmapping],
        colors=None if mesh.colors is None else mesh.colors[vmapping],
        uv=np.asarray(uvs, np.float32),
        history=mesh.history,
    )
    log.info("คลี่ผิวเสร็จ: vertex %d -> %d (จุดบนรอยผ่าถูกแยกร่าง)",
             mesh.n_verts, out.n_verts)
    out.log_stage("uv unwrap")
    return out


# ---------------------------------------------------------------------------
# แปลงสามเหลี่ยมเป็น texel
# ---------------------------------------------------------------------------


def _rasterize(
    mesh: MeshData, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """หาว่า texel ไหนตกอยู่ในสามเหลี่ยมไหน แล้วคิดตำแหน่ง/normal ของ texel นั้น

    ไล่ทีละสามเหลี่ยม คิดกรอบสี่เหลี่ยมที่ครอบมันในพิกัด texel แล้วทดสอบ
    ด้วยพิกัด barycentric ซึ่งให้ทั้งคำตอบว่า "อยู่ในสามเหลี่ยมไหม" และ
    น้ำหนักสำหรับ interpolate ตำแหน่ง 3 มิติกับ normal ในการคำนวณเดียว

    คืน (ตำแหน่ง, normal, mask ของ texel ที่มีเนื้อ) โดย flatten เป็น size*size
    """
    uv = mesh.uv
    if uv is None:
        raise ValueError("ยังไม่ได้คลี่ผิว — เรียก unwrap() ก่อน")
    if mesh.normals is None:
        raise ValueError("ต้องมี vertex normal ก่อน bake")

    pos = np.zeros((size * size, 3), np.float32)
    nrm = np.zeros((size * size, 3), np.float32)
    filled = np.zeros(size * size, bool)

    # UV ของ glTF: (0,0) อยู่มุมซ้ายบนของภาพ v เพิ่มลงล่าง ตรงกับแถวของ numpy
    px = np.clip(uv, 0.0, 1.0) * (size - 1)
    tri_px = px[mesh.faces]                  # (M, 3, 2)
    tri_pos = mesh.verts[mesh.faces]         # (M, 3, 3)
    tri_nrm = mesh.normals[mesh.faces]

    covered = 0
    for i in range(len(mesh.faces)):
        p = tri_px[i]
        x0, y0 = np.floor(p.min(axis=0)).astype(np.int32)
        x1, y1 = np.ceil(p.max(axis=0)).astype(np.int32)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)

        ax, ay = p[0]
        bx, by = p[1]
        cx, cy = p[2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue

        w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / den
        w1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / den
        w2 = 1.0 - w0 - w1

        # เผื่อขอบครึ่ง texel กันสามเหลี่ยมเล็ก ๆ หลุดไปไม่โดน texel ไหนเลย
        eps = -0.002
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue

        b = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)
        ids = ((gy[inside].astype(np.int32)) * size
               + gx[inside].astype(np.int32))

        pos[ids] = b @ tri_pos[i]
        nrm[ids] = b @ tri_nrm[i]
        filled[ids] = True
        covered += len(ids)

    log.info("rasterize: texel ที่มีเนื้อ %d จาก %d (%.1f%% ของภาพ)",
             int(filled.sum()), size * size, 100 * filled.mean())
    return pos, nrm, filled


# ---------------------------------------------------------------------------
# ขยายสีออกนอกขอบ island
# ---------------------------------------------------------------------------


def pad_islands(img: np.ndarray, mask: np.ndarray, iters: int) -> np.ndarray:
    """ขยายสีออกนอกขอบของแต่ละ island

    จำเป็นเพราะตอน render GPU กรองสีแบบ bilinear และใช้ mipmap ซึ่งทั้งสองอย่าง
    ดูดสีจาก texel ข้างเคียง ถ้านอกขอบ island เป็นสีดำ จะเห็นเป็นเส้นดำบาง ๆ
    ตามรอยผ่าทั่วตัวละคร ยิ่งมองไกลยิ่งชัดเพราะ mipmap เฉลี่ยกว้างขึ้น
    """
    out = img.astype(np.float32).copy()
    m = mask.copy()
    k = np.ones((3, 3), np.uint8)

    for _ in range(max(0, iters)):
        grown = cv2.dilate(m.astype(np.uint8), k).astype(bool)
        new = grown & ~m
        if not new.any():
            break
        # ค่าเฉลี่ยของเพื่อนบ้านที่มีสีแล้ว
        acc = cv2.blur(out * m[:, :, None], (3, 3))
        cnt = cv2.blur(m.astype(np.float32), (3, 3))
        out[new] = acc[new] / np.maximum(cnt[new, None], 1e-6)
        m = grown

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def bake(
    mesh: MeshData,
    grid: HullGrid,
    views: dict[str, PreparedView],
    cfg: PipelineConfig,
) -> np.ndarray:
    """ฉายสีจากภาพต้นฉบับลงบน texture ตาม UV ที่คลี่ไว้ คืนภาพ RGB uint8"""
    size = int(cfg.uv_texture_size)
    pos, nrm, filled = _rasterize(mesh, size)

    if not filled.any():
        raise RuntimeError("ไม่มี texel ไหนตกในสามเหลี่ยมเลย — UV อาจเสีย")

    idx = np.flatnonzero(filled)
    log.info("กำลังฉายสีลง %d texel ...", len(idx))
    colors, blind = project_colors(pos[idx], nrm[idx], grid, views, cfg,
                                   return_blind=True)

    img = np.zeros((size * size, 3), np.uint8)
    img[idx] = colors
    img = img.reshape(size, size, 3)

    # --- เติมรูด้วยสีข้างเคียง ไม่ใช่สีเฉลี่ยทั้งตัว ------------------------
    #
    # ซอกที่ไม่มีมุมไหนเห็น (หว่างขากับหาง, ใต้รักแร้) ถ้าเติมด้วยสีเฉลี่ย
    # ของตัวละครจะได้หย่อมสีน้ำตาลกลาง ๆ แปะอยู่กลางขากางเกงสีฟ้า
    # เติมจากเพื่อนบ้านบน atlas ให้สีกลืนไปกับผิวรอบ ๆ แทน
    if blind.any():
        import cv2
        hole = np.zeros(size * size, np.uint8)
        hole[idx[blind]] = 255
        hole = hole.reshape(size, size)
        img = cv2.inpaint(img, hole, 4, cv2.INPAINT_TELEA)
        log.info("เติมรู texture ด้วยสีข้างเคียง %d texel (%.1f%% ของที่มีเนื้อ)",
                 int(blind.sum()), 100 * blind.mean())

    img = pad_islands(img, filled.reshape(size, size), cfg.uv_padding)
    log.info("bake texture %dx%d เสร็จ (ขยายขอบ %d พิกเซล)",
             size, size, cfg.uv_padding)
    return img
