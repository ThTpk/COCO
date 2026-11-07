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


def _depth_of(cam: Camera, pts: np.ndarray) -> np.ndarray:
    """ความลึกเข้าหากล้อง (มากกว่า = ใกล้กล้องกว่า)"""
    return pts[:, 0] * np.sin(cam.theta) + pts[:, 2] * np.cos(cam.theta)


def _visible(cam: Camera, pts: np.ndarray, uv: np.ndarray, shape: tuple,
             step: int = 3, down: int = 2, bias: float = 0.03) -> np.ndarray:
    """จุดไหนที่กล้องตัวนี้เห็นจริง ๆ ไม่ได้ถูกส่วนอื่นของตัวละครบัง

    ทำไมต้องมี: การถ่วงน้ำหนักด้วย normal อย่างเดียวบอกได้แค่ว่าผิวหันไปทางไหน
    ไม่ได้บอกว่ามีอะไรมาขวางหรือเปล่า  ผิวลำตัวที่หันเข้ากล้องแต่มีแขนบังอยู่
    จะดูดสีแขนมาแปะ กลายเป็นเงาแขนลอยอยู่บนอก และหางที่พาดหน้าขาก็ทิ้งรอย
    แบบเดียวกันไว้บนต้นขา

    วิธี: ยิงจุดบนผิวทั้งหมดลงเป็น depth buffer หยาบ ๆ แล้วถามว่าจุดนี้อยู่
    ใกล้กล้องพอ ๆ กับสิ่งที่ใกล้ที่สุดในพิกเซลนั้นไหม — ไม่ต้อง rasterize
    สามเหลี่ยมจริง เพราะ texel บนผิวหนาแน่นกว่าพิกเซลของ buffer อยู่แล้ว
    """
    from scipy import ndimage

    h, w = shape
    hh, ww = h // down + 1, w // down + 1
    depth = _depth_of(cam, pts)

    ui = np.rint(uv[::step, 0] / down).astype(np.int32)
    vi = np.rint(uv[::step, 1] / down).astype(np.int32)
    ok = (ui >= 0) & (ui < ww) & (vi >= 0) & (vi < hh)

    buf = np.full(hh * ww, -1e9, np.float32)
    np.maximum.at(buf, vi[ok] * ww + ui[ok], depth[::step][ok].astype(np.float32))
    buf = buf.reshape(hh, ww)
    # ปิดรูเล็ก ๆ ระหว่างจุด ไม่งั้นพิกเซลที่ไม่มีใครตกจะกลายเป็น "ว่าง"
    buf = ndimage.maximum_filter(buf, size=3)

    qi = np.clip(np.rint(uv[:, 0] / down).astype(np.int32), 0, ww - 1)
    qj = np.clip(np.rint(uv[:, 1] / down).astype(np.int32), 0, hh - 1)
    return depth >= buf[qj, qi] - bias


def _exposure(frames: list[ClipFrame], cams: list[Camera]) -> list[np.ndarray]:
    """ตัวคูณปรับสีของแต่ละเฟรม ให้ทุกเฟรมสว่าง/อมสีเท่า ๆ กัน

    คบไฟกับหมอกในฉากทำให้ตัวละครในแต่ละเฟรมสว่างไม่เท่ากัน เมื่อ texel
    คนละหย่อมเลือกเฟรมคนละใบ ความต่างนั้นจะโผล่เป็นรอยปะบนผิว
    """
    meds = []
    for frame in frames:
        m = frame.mask
        px = frame.rgb[m]
        meds.append(np.median(px, axis=0) if len(px) else np.array([128.] * 3))
        frame.unload()
    target = np.median(np.stack(meds), axis=0)
    return [np.clip(target / np.maximum(md, 1.0), 0.75, 1.35).astype(np.float32)
            for md in meds]


def project_colors(points_world: np.ndarray, normals_world: np.ndarray,
                   frames: list[ClipFrame], cams: list[Camera],
                   cfg: PipelineConfig, blend: float = 6.0,
                   occlusion: bool = True) -> np.ndarray:
    """คิดสีของจุดใด ๆ บนผิว จากทุกเฟรมที่ส่งเข้ามา คืน (N, 3) uint8

    *ไม่* เฉลี่ยทุกมุมเข้าด้วยกัน — เก็บแค่สองมุมที่เห็นจุดนั้นดีที่สุด

    เหตุผล: ตัวละครในคลิปที่ AI สร้างไม่ได้คงรูประหว่างเฟรม จุดเดียวกันบนผิว
    จึงไปตกคนละที่บนลายของแต่ละเฟรม การเฉลี่ยสี่สิบเฟรมจึงเป็นการเกลี่ยลาย
    เกล็ด โซ่ เข็มขัด ให้หายไปหมด เหลือแต่ก้อนสี  มุมเดียวที่คมอยู่แล้ว
    ให้ลายที่คมกว่าการเฉลี่ยเสมอ

    ที่ยังเก็บมุดที่สองไว้ด้วย เพราะถ้าตัดเหลือมุมเดียวเด็ดขาด ตรงที่สองมุม
    ดีพอ ๆ กันจะเกิดรอยต่อคมบนผิว  ``blend`` คุมว่าจะผสมมุมที่สองเข้ามาเร็ว
    แค่ไหน (ยิ่งมากยิ่งผสมน้อย ยิ่งคม)

    คะแนนของแต่ละมุมคิดจากสามอย่างที่เป็นอิสระกัน
      * ผิวหันเข้าหากล้องแค่ไหน (normal · ทิศกล้อง)
      * เฟรมนั้นเห็นตัวละครใหญ่แค่ไหน — เฟรมระยะใกล้ให้ลายละเอียดกว่า
      * จุดนั้นอยู่ลึกเข้าไปในเงาแค่ไหน — ริมขอบ silhouette เป็นที่ที่
        geometry กับ mask เหลื่อมกันเสมอ สีตรงนั้นจึงเชื่อไม่ได้
    """
    n = normals_world / np.clip(
        np.linalg.norm(normals_world, axis=1, keepdims=True), 1e-9, None)
    N = len(points_world)
    gains = _exposure(frames, cams)
    scale_ref = max(c.scale for c in cams)

    best_q = np.zeros(N, np.float32)
    best_c = np.zeros((N, 3), np.float32)
    second_q = np.zeros(N, np.float32)
    second_c = np.zeros((N, 3), np.float32)
    hidden = 0

    for cam, frame, gain in zip(cams, frames, gains):
        facing = (n @ cam.direction).astype(np.float32)
        q = np.clip(facing, 0.0, None) ** cfg.texture_power
        if not np.any(q > 1e-4):
            frame.unload()
            continue

        uv = cam.project(points_world)
        mask = frame.mask
        h, wd = mask.shape
        q *= ((uv[:, 0] >= 0) & (uv[:, 0] < wd - 1)
              & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1))
        if not np.any(q > 1e-4):
            frame.unload()
            continue

        # ลึกเข้าไปในเงาแค่ไหน (หน่วยโลก) — ไล่จาก 0 ที่ขอบ ถึง 1 ที่ลึกพอ
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        edge = _bilinear(dist[:, :, None], uv[:, 1], uv[:, 0])[:, 0] / cam.scale
        q *= np.clip(edge / 0.015, 0.0, 1.0)
        q *= cam.scale / scale_ref

        if occlusion and np.any(q > 1e-4):
            vis = _visible(cam, points_world, uv, mask.shape)
            hidden += int(((q > 1e-4) & ~vis).sum())
            q *= vis

        rgb = _edge_extend(frame.rgb, mask, cfg.texture_edge_extend)
        c = _bilinear(rgb, uv[:, 1], uv[:, 0]) * gain

        better = q > best_q
        mid = (~better) & (q > second_q)
        second_c = np.where(better[:, None], best_c,
                            np.where(mid[:, None], c, second_c))
        second_q = np.where(better, best_q, np.where(mid, q, second_q))
        best_c = np.where(better[:, None], c, best_c)
        best_q = np.where(better, q, best_q)
        frame.unload()

    # ผสมมุมที่สองเข้ามาเฉพาะตอนที่มันดีพอ ๆ กับมุมแรก
    w2 = (second_q / np.maximum(best_q, 1e-9)) ** blend
    out = (best_c + w2[:, None] * second_c) / (1.0 + w2)[:, None]

    blind = best_q <= 1e-9
    if blind.any():
        out[blind] = out[~blind].mean(axis=0) if (~blind).any() else 128
        log.info("เติมสีเฉลี่ยให้ %d จุด (%.1f%%) ที่ไม่มีมุมไหนมองเห็น",
                 int(blind.sum()), 100 * blind.mean())
    if hidden:
        log.info("ตัดการมองของมุมที่ถูกบังไป %.1f ล้านครั้ง (แขน/หางบังลำตัว)",
                 hidden / 1e6)
    return np.clip(out, 0, 255).astype(np.uint8)


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
         cams: list[Camera], cfg: PipelineConfig, blend: float = 6.0,
         occlusion: bool = True) -> np.ndarray:
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
    colors = project_colors(world, nrm[idx].astype(np.float64), frames, cams,
                            cfg, blend=blend, occlusion=occlusion)

    img = np.zeros((size * size, 3), np.uint8)
    img[idx] = colors
    img = pad_islands(img.reshape(size, size, 3),
                      filled.reshape(size, size), cfg.uv_padding)
    log.info("bake texture %dx%d เสร็จ", size, size)
    return img


def vertex_colors(mesh: MeshData, grid: HullGrid, axes: tuple,
                  frames: list[ClipFrame], cams: list[Camera],
                  cfg: PipelineConfig, blend: float = 6.0,
                  occlusion: bool = True) -> MeshData:
    """ลงสีทีละ vertex (ใช้ตอนไม่ต้องการ texture map)"""
    if mesh.normals is None:
        raise ValueError("ต้องคิด vertex normal ก่อนลงสี")
    world = _voxel_to_world(mesh.verts, grid, axes)
    mesh.colors = project_colors(world, mesh.normals, frames, cams, cfg,
                                 blend=blend, occlusion=occlusion)
    mesh.log_stage("vertex colors จากคลิป")
    return mesh
