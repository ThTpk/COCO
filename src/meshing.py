"""ขั้นที่ 3: แปลงกริด occupancy เป็น mesh ที่สะอาดและเบาพอใช้ในเกม

ลำดับงาน
    marching cubes -> เก็บก้อนใหญ่สุด -> Taubin smoothing -> decimate -> คิด normal

หมายเหตุเรื่องระบบพิกัด: ภายในโมดูลนี้ vertex เก็บอยู่ใน *พิกัด voxel index*
เรียงเป็น (ix, iy, iz) เพราะขั้น texture ต้องแมปกลับไปหาพิกเซลบน canvas
การแปลงเป็นหน่วยเมตรจะทำตอนท้ายสุดด้วย `to_world()`
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from skimage import measure

from .config import PipelineConfig
from .visual_hull import HullGrid

log = logging.getLogger(__name__)


@dataclass
class MeshData:
    """mesh ในพิกัด voxel index เรียงเป็น (ix, iy, iz)"""

    verts: np.ndarray                      # (N, 3) float64
    faces: np.ndarray                      # (M, 3) int32
    colors: np.ndarray | None = None       # (N, 3) uint8
    normals: np.ndarray | None = None      # (N, 3) float64
    uv: np.ndarray | None = None           # (N, 2) float32 พิกัดบน texture 0..1
    history: list[str] = field(default_factory=list)

    @property
    def n_verts(self) -> int:
        return len(self.verts)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def log_stage(self, label: str) -> None:
        self.history.append(f"{label}: {self.n_verts} verts / {self.n_faces} tris")
        log.info("%-22s %7d verts  %7d tris", label, self.n_verts, self.n_faces)


# ---------------------------------------------------------------------------
# แปลงไปกลับระหว่าง open3d กับ numpy
# ---------------------------------------------------------------------------


def _to_o3d(mesh: MeshData) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.verts, np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32)),
    )
    if mesh.colors is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.colors, np.float64) / 255.0
        )
    return m


def _from_o3d(m: o3d.geometry.TriangleMesh, history: list[str]) -> MeshData:
    colors = None
    if m.has_vertex_colors():
        colors = np.clip(np.asarray(m.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
    normals = np.asarray(m.vertex_normals) if m.has_vertex_normals() else None
    return MeshData(
        verts=np.asarray(m.vertices),
        faces=np.asarray(m.triangles).astype(np.int32),
        colors=colors,
        normals=normals,
        history=history,
    )


# ---------------------------------------------------------------------------
# marching cubes
# ---------------------------------------------------------------------------


def to_mesh(grid: HullGrid, cfg: PipelineConfig) -> MeshData:
    """ดึง isosurface ออกจากกริด occupancy

    กริดเรียงเป็น (iy, ix, iz) ตามที่ขั้น carve สร้างไว้
    marching_cubes จึงคืน vertex เรียงแบบเดียวกัน ต้องสลับคอลัมน์เป็น (ix, iy, iz)
    """
    volume = grid.occupancy
    if not (volume.min() < cfg.iso_level < volume.max()):
        raise RuntimeError(
            f"ค่า iso_level={cfg.iso_level} อยู่นอกช่วงของกริด "
            f"({volume.min():.3f}..{volume.max():.3f}) — hull ว่างหรือทึบทั้งก้อน"
        )

    verts_yxz, faces, normals_yxz, _ = measure.marching_cubes(
        volume, level=cfg.iso_level, allow_degenerate=False
    )

    # (iy, ix, iz) -> (ix, iy, iz)
    verts = verts_yxz[:, [1, 0, 2]]
    normals = normals_yxz[:, [1, 0, 2]]

    mesh = MeshData(verts=verts, faces=faces.astype(np.int32), normals=normals)
    mesh.log_stage("marching cubes")
    return mesh


# ---------------------------------------------------------------------------
# ทำความสะอาดและลดรูป
# ---------------------------------------------------------------------------


def clean(mesh: MeshData, cfg: PipelineConfig) -> MeshData:
    """ลบ vertex/face ซ้ำและเสีย แล้วเก็บเฉพาะเปลือกที่ใหญ่ที่สุด"""
    m = _to_o3d(mesh)
    m.remove_duplicated_vertices()
    m.remove_duplicated_triangles()
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    # ไม่เรียก remove_non_manifold_edges() ตรงนี้ — ผลจาก marching cubes เป็น
    # manifold อยู่แล้ว การเรียกมันมีแต่จะลบ triangle ดี ๆ ทิ้งจนเกิดรูโหว่

    if cfg.keep_largest_component:
        labels, counts, _ = m.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        if len(counts) > 1:
            biggest = int(np.argmax(counts))
            m.remove_triangles_by_mask(labels != biggest)
            m.remove_unreferenced_vertices()
            log.info("ทิ้งเปลือกแยก %d ชิ้น (เก็บชิ้นที่มี %d tris)",
                     len(counts) - 1, counts[biggest])

    out = _from_o3d(m, mesh.history)
    out.log_stage("clean")
    return out


def smooth(mesh: MeshData, cfg: PipelineConfig) -> MeshData:
    """Taubin smoothing — รักษาปริมาตรดีกว่า Laplacian ที่ทำให้ตัวละครผอมลงเรื่อย ๆ"""
    if cfg.taubin_iters <= 0:
        return mesh
    m = _to_o3d(mesh)
    m = m.filter_smooth_taubin(number_of_iterations=cfg.taubin_iters)
    out = _from_o3d(m, mesh.history)
    out.log_stage(f"taubin smooth x{cfg.taubin_iters}")
    return out


def decimate(mesh: MeshData, cfg: PipelineConfig) -> MeshData:
    """ลดจำนวน triangle ลงมาให้อยู่ในงบที่ตั้งไว้"""
    if mesh.n_faces <= cfg.target_tris:
        log.info("ข้าม decimate — %d tris อยู่ในงบ %d แล้ว",
                 mesh.n_faces, cfg.target_tris)
        return mesh
    m = _to_o3d(mesh)
    m = m.simplify_quadric_decimation(target_number_of_triangles=cfg.target_tris)
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    out = _from_o3d(m, mesh.history)
    out.log_stage(f"decimate -> {cfg.target_tris}")
    return out


def orient(mesh: MeshData) -> MeshData:
    """จัด winding ให้ normal ชี้ออกนอก แล้วคิด vertex normal ใหม่

    ถ้า winding กลับด้าน ปริมาตรที่คำนวณด้วย divergence theorem จะออกมาติดลบ
    """
    m = _to_o3d(mesh)
    m.compute_triangle_normals()

    v = np.asarray(m.vertices)
    f = np.asarray(m.triangles)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    signed_volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    if signed_volume < 0:
        m.triangles = o3d.utility.Vector3iVector(f[:, [0, 2, 1]])
        log.info("winding กลับด้าน (ปริมาตร %.1f) — สลับลำดับ vertex ให้ normal ชี้ออก",
                 signed_volume)

    m.compute_vertex_normals()
    out = _from_o3d(m, mesh.history)
    out.log_stage("orient + normals")
    return out


def build(grid: HullGrid, cfg: PipelineConfig) -> MeshData:
    """รวมทุกขั้นของการทำ mesh"""
    mesh = to_mesh(grid, cfg)
    mesh = clean(mesh, cfg)
    mesh = smooth(mesh, cfg)
    mesh = decimate(mesh, cfg)
    mesh = orient(mesh)
    return mesh


# ---------------------------------------------------------------------------
# แปลงไปหน่วยโลก
# ---------------------------------------------------------------------------


def to_world(mesh: MeshData, grid: HullGrid, height_m: float) -> np.ndarray:
    """แปลง vertex จากพิกัด voxel เป็นเมตร

    จัดให้ Y=0 อยู่ที่ฝ่าเท้า และ X=Z=0 อยู่กลางลำตัว
    ซึ่งเป็น origin ที่ engine และ three.js คาดหวังสำหรับตัวละคร
    """
    v = mesh.verts.astype(np.float64).copy()   # (ix, iy, iz)

    y_span = v[:, 1].max() - v[:, 1].min()
    if y_span <= 0:
        raise ValueError("mesh แบนในแกน Y — ไม่สามารถ scale ตามความสูงได้")
    scale = height_m / y_span

    v *= scale
    v[:, 0] -= 0.5 * (v[:, 0].min() + v[:, 0].max())   # จัดกลางแกน X
    v[:, 2] -= 0.5 * (v[:, 2].min() + v[:, 2].max())   # จัดกลางแกน Z
    v[:, 1] -= v[:, 1].min()                            # ฝ่าเท้าแตะ Y=0

    log.info("ขนาดจริง: กว้าง %.3f m  สูง %.3f m  หนา %.3f m",
             np.ptp(v[:, 0]), np.ptp(v[:, 1]), np.ptp(v[:, 2]))
    return v
