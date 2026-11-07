r"""ฉายสีจากเฟรมของคลิปลงบนผิวโมเดล (projective texturing แบบหลายสิบมุม)

ต่างจาก ``src/texture.py`` ตรงที่ตัวนั้นรู้จักกล้องแค่สี่ตัวที่ตั้งฉากกัน
(หน้า/หลัง/ข้างซ้าย/ข้างขวา) ส่วนคลิปให้กล้องมาทุก ๆ ไม่กี่องศารอบตัว

การมีมุมถี่ช่วยสองเรื่อง
  * ไม่มีรอยต่อแข็ง ๆ ระหว่างมุม — จุดหนึ่งบนผิวได้สีจากหลายมุมที่เห็นมันชัด
    พอ ๆ กัน แล้วเกลี่ยกันตามว่าหันเข้าหากล้องตัวไหนมากกว่า
  * ซอกที่สี่มุมหลักมองไม่เห็น (ใต้คาง ด้านในต้นขา) มักมีมุมเฉียงสักมุมเห็น

สิ่งที่ยังไม่ทำ: ตรวจการบังกันเอง (occlusion) แบบจริงจังด้วย depth buffer
ใช้การถ่วงน้ำหนักด้วย normal แทน ซึ่งพอสำหรับตัวละครทรงนูนอย่างนี้ และ
ตัดเฉพาะกรณีชัด ๆ ด้วยการเช็กว่าจุดนั้นตกใน silhouette ของมุมนั้นหรือไม่
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import PipelineConfig
from .meshing import MeshData
from .turntable import Camera, ClipFrame
from .visual_hull import HullGrid

log = logging.getLogger(__name__)


def _bilinear(img: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    r = np.clip(rows, 0, h - 1.001)
    c = np.clip(cols, 0, w - 1.001)
    r0, c0 = np.floor(r).astype(np.int32), np.floor(c).astype(np.int32)
    fr, fc = (r - r0)[:, None], (c - c0)[:, None]
    a = img[r0, c0].astype(np.float32)
    b = img[r0, c0 + 1].astype(np.float32)
    d = img[r0 + 1, c0].astype(np.float32)
    e = img[r0 + 1, c0 + 1].astype(np.float32)
    return (a * (1 - fc) + b * fc) * (1 - fr) + (d * (1 - fc) + e * fc) * fr


def _edge_extend(rgb: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """ยืดสีออกนอกขอบ silhouette กัน vertex ริมขอบดูดสีพื้นหลังมาใส่"""
    if radius <= 0:
        return rgb
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    band = cv2.dilate(mask.astype(np.uint8), k).astype(bool) & ~mask
    if not band.any():
        return rgb
    return cv2.inpaint(rgb, band.astype(np.uint8), radius, cv2.INPAINT_TELEA)


def project_colors(points_world: np.ndarray, normals_world: np.ndarray,
                   frames: list[ClipFrame], cams: list[Camera],
                   cfg: PipelineConfig) -> np.ndarray:
    """คิดสีของจุดใด ๆ บนผิว จากทุกเฟรมที่ส่งเข้ามา คืน (N, 3) uint8"""
    n = normals_world / np.clip(
        np.linalg.norm(normals_world, axis=1, keepdims=True), 1e-9, None)

    total_c = np.zeros((len(points_world), 3), np.float32)
    total_w = np.zeros(len(points_world), np.float32)

    for cam, frame in zip(cams, frames):
        facing = n @ cam.direction
        w = np.clip(facing, 0.0, None).astype(np.float32) ** cfg.texture_power
        if not np.any(w > 1e-4):
            frame.unload()
            continue

        uv = cam.project(points_world)
        mask = frame.mask
        h, wd = mask.shape
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < wd - 1)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1))
        w = w * inside
        if not np.any(w > 1e-4):
            frame.unload()
            continue

        rgb = _edge_extend(frame.rgb, mask, cfg.texture_edge_extend)
        # จุดที่ตกนอกเงาของมุมนี้ = มุมนี้มองไม่เห็นจริง ๆ (หรือมุมเพี้ยน)
        w = w * _bilinear(mask.astype(np.float32)[:, :, None],
                          uv[:, 1], uv[:, 0])[:, 0]
        total_c += _bilinear(rgb, uv[:, 1], uv[:, 0]) * w[:, None]
        total_w += w
        frame.unload()

    blind = total_w <= 1e-6
    if blind.any():
        fill = total_c[~blind].sum(axis=0) / max(total_w[~blind].sum(), 1e-6)
        total_c[blind] = fill
        total_w[blind] = 1.0
        log.info("เติมสีเฉลี่ยให้ %d จุด (%.1f%%) ที่ไม่มีมุมไหนมองเห็น",
                 int(blind.sum()), 100 * blind.mean())
    return np.clip(total_c / total_w[:, None], 0, 255).astype(np.uint8)


def _voxel_to_world(points_vox: np.ndarray, grid: HullGrid,
                    axes: tuple) -> np.ndarray:
    """พิกัด voxel (ix, iy, iz) -> พิกัดโลกที่กล้องของคลิปใช้"""
    ys, xs, zs = axes
    out = np.empty_like(points_vox, dtype=np.float64)
    out[:, 0] = np.interp(points_vox[:, 0], np.arange(len(xs)), xs)
    out[:, 1] = np.interp(points_vox[:, 1], np.arange(len(ys)), ys)
    out[:, 2] = np.interp(points_vox[:, 2], np.arange(len(zs)), zs)
    return out


def bake(mesh: MeshData, grid: HullGrid, axes: tuple, frames: list[ClipFrame],
         cams: list[Camera], cfg: PipelineConfig) -> np.ndarray:
    """คลี่ผิวแล้ว bake สีจากคลิปลง texture — คืนภาพ RGB uint8"""
    from .uvmap import _rasterize, pad_islands

    size = int(cfg.uv_texture_size)
    pos, nrm, filled = _rasterize(mesh, size)
    if not filled.any():
        raise RuntimeError("ไม่มี texel ไหนตกในสามเหลี่ยมเลย — UV อาจเสีย")

    idx = np.flatnonzero(filled)
    log.info("กำลังฉายสีจาก %d มุมลง %d texel ...", len(cams), len(idx))
    world = _voxel_to_world(pos[idx], grid, axes)
    # normal อยู่ในพิกัด voxel ซึ่งเป็นลูกบาศก์จริง ทิศจึงตรงกับพิกัดโลกอยู่แล้ว
    colors = project_colors(world, nrm[idx].astype(np.float64), frames, cams, cfg)

    img = np.zeros((size * size, 3), np.uint8)
    img[idx] = colors
    img = pad_islands(img.reshape(size, size, 3),
                      filled.reshape(size, size), cfg.uv_padding)
    log.info("bake texture %dx%d เสร็จ", size, size)
    return img


def vertex_colors(mesh: MeshData, grid: HullGrid, axes: tuple,
                  frames: list[ClipFrame], cams: list[Camera],
                  cfg: PipelineConfig) -> MeshData:
    """ลงสีทีละ vertex (ใช้ตอนไม่ต้องการ texture map)"""
    if mesh.normals is None:
        raise ValueError("ต้องคิด vertex normal ก่อนลงสี")
    world = _voxel_to_world(mesh.verts, grid, axes)
    mesh.colors = project_colors(world, mesh.normals, frames, cams, cfg)
    mesh.log_stage("vertex colors จากคลิป")
    return mesh
