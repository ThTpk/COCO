"""ขั้นที่ 4: ฉายสีจากภาพทุกมุมกลับลงบน mesh (projective texturing)

วิธีคิด: vertex แต่ละจุดมี normal บอกว่ามันหันไปทางไหน มุมกล้องที่หันตรงเข้าหา
vertex นั้นที่สุดคือมุมที่ควรเชื่อถือสีมากที่สุด เราจึงถ่วงน้ำหนักด้วย
`max(0, dot(normal, ทิศกล้อง)) ** texture_power` แล้วเฉลี่ยสีจากทุกมุม

ผลที่ได้เป็น vertex color ซึ่งพอสำหรับดู preview และส่งต่อให้ Blender ไป bake
เป็น texture map จริงในขั้น rig
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import PipelineConfig
from .meshing import MeshData
from .preprocess import PreparedView
from .visual_hull import HullGrid

log = logging.getLogger(__name__)

# ทิศจากตัวละครไปหากล้องของแต่ละมุม ในพิกัด (x, y, z)
VIEW_DIRS: dict[str, tuple[float, float, float]] = {
    "front":  (0.0, 0.0, 1.0),    # กล้องอยู่ +Z
    "back":   (0.0, 0.0, -1.0),   # กล้องอยู่ -Z
    "side":   (-1.0, 0.0, 0.0),   # กล้องอยู่ -X (ด้านขวาของตัวละคร)
    "side_r": (1.0, 0.0, 0.0),    # กล้องอยู่ +X
}


def _edge_extend(rgb: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """ยืดสีจากในตัวละครออกไปนอกขอบ mask

    vertex ที่อยู่ตรงขอบ silhouette มักถูกแมปไปตกนอก mask ไปเล็กน้อยเพราะ
    marching cubes วาง isosurface ไว้กลางระหว่าง voxel ทึบกับว่าง
    ถ้าไม่ยืดสีออกไปก่อน vertex ขอบทั้งแถบจะดูดสีดำของพื้นหลังมาใส่
    """
    if radius <= 0:
        return rgb
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    grown = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    band = grown & ~mask
    if not band.any():
        return rgb
    # inpaint แค่แถบขอบ ถูกกว่าการ inpaint พื้นหลังทั้งภาพมาก
    out = cv2.inpaint(rgb, band.astype(np.uint8), radius, cv2.INPAINT_TELEA)
    return out


def _bilinear(img: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """สุ่มสีแบบ bilinear ที่พิกัดทศนิยม คืน (N, C) float32"""
    h, w = img.shape[:2]
    r = np.clip(rows, 0, h - 1.001)
    c = np.clip(cols, 0, w - 1.001)

    r0, c0 = np.floor(r).astype(np.int32), np.floor(c).astype(np.int32)
    r1, c1 = r0 + 1, c0 + 1
    fr, fc = (r - r0)[:, None], (c - c0)[:, None]

    a = img[r0, c0].astype(np.float32)
    b = img[r0, c1].astype(np.float32)
    d = img[r1, c0].astype(np.float32)
    e = img[r1, c1].astype(np.float32)

    top = a * (1 - fc) + b * fc
    bot = d * (1 - fc) + e * fc
    return top * (1 - fr) + bot * fr


def _sample_mask(mask: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """ค่า mask แบบ bilinear (0..1) ใช้เป็นน้ำหนักความน่าเชื่อถือ"""
    m = _bilinear(mask.astype(np.float32)[:, :, None], rows, cols)
    return m[:, 0]


def project_colors(
    points: np.ndarray,
    normals: np.ndarray,
    grid: HullGrid,
    views: dict[str, PreparedView],
    cfg: PipelineConfig,
) -> np.ndarray:
    """ฉายสีจากทุกมุมลงบนจุดใด ๆ ในพิกัด voxel คืนสี (N, 3) uint8

    ใช้ร่วมกันระหว่างการลงสีทีละ vertex กับการ bake ลง texture ทีละ texel
    ต่างกันแค่ว่าจุดที่ส่งเข้ามาคืออะไร ตรรกะการเลือกมุมและถ่วงน้ำหนักเหมือนกัน
    """
    v = np.asarray(points, np.float64)
    n = np.asarray(normals, np.float64)
    n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)

    ix, iy, iz = v[:, 0], v[:, 1], v[:, 2]
    rows, cols_x, cols_z = grid.voxel_to_canvas(iy, ix, iz)

    total_w = np.zeros(len(v), np.float32)
    total_c = np.zeros((len(v), 3), np.float32)

    for name, view in views.items():
        if name not in VIEW_DIRS:
            log.warning("ไม่รู้ทิศกล้องของมุม %s — ข้ามในขั้น texture", name)
            continue

        d = np.asarray(VIEW_DIRS[name], np.float32)
        facing = n @ d                               # -1..1
        w = np.clip(facing, 0.0, None) ** cfg.texture_power

        cols = cols_x if view.spec.axis_u == "x" else cols_z
        rgb = _edge_extend(view.rgb, view.mask, cfg.texture_edge_extend)

        # vertex ที่แมปไปตกนอก silhouette ของมุมนี้ ไม่น่าเชื่อถือ
        w = w * _sample_mask(view.mask, rows, cols)

        total_c += _bilinear(rgb, rows, cols) * w[:, None]
        total_w += w
        log.debug("มุม %-6s ให้สีกับ %d/%d vertex", name, int((w > 1e-4).sum()), len(v))

    blind = total_w <= 1e-6
    if blind.any():
        # vertex ที่ไม่มีมุมไหนเห็น (ซอกลึก) — เติมด้วยสีเฉลี่ยของตัวละคร
        fallback = np.zeros(3, np.float32)
        weight = 0.0
        for view in views.values():
            if view.mask.any():
                fallback += view.rgb[view.mask].mean(axis=0)
                weight += 1
        fallback = fallback / max(weight, 1.0)
        total_c[blind] = fallback
        total_w[blind] = 1.0
        log.info("เติมสีสำรองให้ %d จุด (%.1f%%) ที่ไม่มีมุมไหนมองเห็น",
                 int(blind.sum()), 100 * blind.mean())

    return np.clip(total_c / total_w[:, None], 0, 255).astype(np.uint8)


def project_vertex_colors(
    mesh: MeshData,
    grid: HullGrid,
    views: dict[str, PreparedView],
    cfg: PipelineConfig,
) -> MeshData:
    """คิดสีของทุก vertex จากภาพทุกมุม แล้วใส่กลับเข้า mesh"""
    if mesh.normals is None:
        raise ValueError("ต้องคิด vertex normal ก่อนทำ texture (เรียก meshing.orient)")

    mesh.colors = project_colors(mesh.verts, mesh.normals, grid, views, cfg)
    mesh.log_stage("vertex colors")
    return mesh
