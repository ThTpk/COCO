"""โหลดโมเดลฐาน (base mesh) แล้วจัดให้อยู่ในระบบพิกัดเดียวกับ pipeline

ทำไมต้องมีโมเดลฐาน
------------------
visual hull สร้างรูปทรงจากศูนย์ มันจึงได้แต่ "ก้อนที่เงาถูก" โดยไม่มีความรู้
ว่าร่างกายคนประกอบด้วยอะไร ผลคือจุดที่เงาบอกไม่ได้จะพังเสมอ ไม่ว่าจะเพิ่ม
ความละเอียดแค่ไหน: นิ้วมือ (มุมหน้าเห็นร่องแค่ตอนกางนิ้ว) นิ้วเท้า (อยู่บน
ระนาบที่ไม่มีมุมไหนมองเห็น) และคอ (ถูกบ่าบังทั้งสองมุม)

โมเดลฐานพลิกโจทย์: แทนที่จะ *สร้าง* รูปทรง เรา *ดัด* รูปทรงที่ถูกต้องอยู่แล้ว
ให้ไปตรงกับเงา นิ้วมือห้านิ้วกับคอมีมาตั้งแต่ต้น ไม่ต้องเดาจากเงาอีก และ
topology ที่ได้เป็น quad เรียงสวยพร้อม rig แทนที่จะเป็นผล marching cubes

โมดูลนี้รับผิดชอบแค่ขั้นแรก: อ่านไฟล์เข้ามาแล้วหมุน/ย่อ/ขยับให้อยู่ในระบบ
พิกัดเดียวกับที่ทั้ง pipeline ใช้ (+X ขวา, +Y ขึ้น, +Z ทิศที่หันหน้าไป)
ส่วนการดัดอยู่ใน `basefit.py`
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BLENDER_EXE, PipelineConfig

log = logging.getLogger(__name__)

#: นามสกุลที่ trimesh อ่านได้ตรง ๆ
NATIVE_EXTS = (".glb", ".gltf", ".obj", ".ply", ".stl", ".dae")
#: นามสกุลที่ต้องให้ Blender แปลงให้ก่อน
BLENDER_EXTS = (".fbx", ".blend")
BASE_EXTS = NATIVE_EXTS + BLENDER_EXTS

#: คำในชื่อไฟล์ที่บอกว่าเป็นโมเดลฐาน
BASE_HINTS = ("base", "basemesh", "body", "man", "male", "human", "muscular",
              "anatomy", "mannequin", "ฐาน", "หุ่น")


@dataclass
class BaseMesh:
    """โมเดลฐานที่จัดพิกัดแล้ว"""

    verts: np.ndarray          # (V, 3) float64 — เมตร, +Y ขึ้น, +Z หันหน้า
    faces: np.ndarray          # (F, 3) int32
    source: Path
    transform: np.ndarray      # (4, 4) ที่ใช้แปลงจากไฟล์ต้นฉบับมาเป็นพิกัดนี้
    notes: list[str]

    @property
    def height(self) -> float:
        return float(self.verts[:, 1].max() - self.verts[:, 1].min())

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.verts.min(0), self.verts.max(0)])


# ---------------------------------------------------------------------------
# หาไฟล์
# ---------------------------------------------------------------------------


def find(input_dir: Path) -> Path | None:
    """หาไฟล์โมเดลฐานในโฟลเดอร์ input

    เลือกไฟล์ 3D ที่ใหญ่ที่สุด เพราะโมเดลฐานที่มี topology ละเอียดพอจะดัดได้
    ย่อมใหญ่กว่าไฟล์ 3D อื่นที่อาจหลงเข้ามา และให้คะแนนพิเศษกับชื่อที่บอกว่า
    เป็นโมเดลฐาน เพื่อให้เลือกถูกเมื่อมีหลายไฟล์
    """
    if not input_dir.is_dir():
        return None
    cands = [p for p in sorted(input_dir.iterdir())
             if p.is_file() and p.suffix.lower() in BASE_EXTS]
    if not cands:
        return None

    def score(p: Path) -> tuple[int, int]:
        stem = p.stem.lower()
        hinted = any(h in stem for h in BASE_HINTS)
        return (1 if hinted else 0, p.stat().st_size)

    best = max(cands, key=score)
    if len(cands) > 1:
        log.info("เจอไฟล์ 3D %d ไฟล์ เลือก %s (ใหญ่สุด/ชื่อตรง)",
                 len(cands), best.name)
    return best


# ---------------------------------------------------------------------------
# อ่านไฟล์
# ---------------------------------------------------------------------------


def _fbx_to_glb(path: Path, blender_exe: str, timeout: int = 300) -> Path:
    """ให้ Blender แปลง FBX/blend เป็น GLB เพราะ trimesh อ่านสองอย่างนี้ไม่ได้"""
    exe = Path(blender_exe)
    if not exe.exists():
        raise FileNotFoundError(
            f"ไฟล์ {path.suffix} ต้องให้ Blender แปลงก่อน แต่ไม่พบ Blender ที่ {exe}\n"
            "ตั้ง BLENDER_EXE ให้ถูก หรือ export โมเดลฐานเป็น .glb/.obj มาแทน"
        )
    out = Path(tempfile.gettempdir()) / f"basemesh_{path.stem}.glb"
    script = (
        "import bpy,sys\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        f"p=r'{path}'\n"
        + ("bpy.ops.wm.open_mainfile(filepath=p)\n" if path.suffix.lower() == ".blend"
           else "bpy.ops.import_scene.fbx(filepath=p)\n")
        + f"bpy.ops.export_scene.gltf(filepath=r'{out}', export_format='GLB')\n"
        "print('[base] converted')\n"
    )
    sp = Path(tempfile.gettempdir()) / "basemesh_convert.py"
    sp.write_text(script, encoding="utf-8")
    log.info("แปลง %s เป็น GLB ด้วย Blender ...", path.suffix)
    proc = subprocess.run([str(exe), "--background", "--factory-startup",
                           "--python", str(sp)],
                          capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")
    if "[base] converted" not in (proc.stdout or "") or not out.exists():
        tail = "\n".join((proc.stdout or "").splitlines()[-20:])
        raise RuntimeError(f"Blender แปลง {path.name} ไม่สำเร็จ\n{tail}")
    return out


def _load_raw(path: Path, blender_exe: str):
    import trimesh

    src = path
    if path.suffix.lower() in BLENDER_EXTS:
        src = _fbx_to_glb(path, blender_exe)

    scene = trimesh.load(str(src), process=False, force="scene")
    parts = [g for g in scene.geometry.values()
             if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not parts:
        raise ValueError(f"ไม่เจอ mesh ในไฟล์ {path.name}")

    # ต้องใช้ transform ของแต่ละ node ด้วย ไม่งั้นชิ้นส่วนที่ถูกจัดวางไว้
    # (แขน/มือที่ parent อยู่กับ node อื่น) จะมากองซ้อนกันที่จุดกำเนิด
    mesh = trimesh.util.concatenate(scene.dump(concatenate=False))
    log.info("โหลดโมเดลฐาน %s: %d ชิ้น รวม %d verts / %d tris",
             path.name, len(parts), len(mesh.vertices), len(mesh.faces))

    # เก็บเฉพาะเปลือกที่ใหญ่ที่สุด
    #
    # โมเดลฐานมักมี "ลูกตา" เป็นเปลือกปิดแยกลอยอยู่ข้างในหัว ซึ่งทำให้ mesh
    # ไม่ manifold แล้ว bone heat weighting ของ Blender *ล้มแบบเงียบ ๆ*:
    # operator รายงานสำเร็จ สร้าง vertex group ครบทุกกระดูก แต่ไม่ใส่น้ำหนัก
    # ให้ vertex สักจุดเดียว (วัดได้ 0/21692) แล้วการ pose ก็ไม่ขยับอะไรเลย
    # โดยไม่มี error ให้เห็น
    try:
        comp = mesh.split(only_watertight=False)
    except Exception as exc:                                 # noqa: BLE001
        log.warning("แยกชิ้นส่วนของโมเดลฐานไม่ได้ (%s) — ใช้ทั้งก้อน", exc)
        comp = []
    if len(comp) > 1:
        body = max(comp, key=lambda m: len(m.faces))
        log.info("ทิ้งชิ้นส่วนแยก %d ชิ้น (ลูกตา/ของประกอบ) เหลือเปลือกหลัก "
                 "%d verts / %d tris", len(comp) - 1, len(body.vertices),
                 len(body.faces))
        mesh = body
    return mesh


# ---------------------------------------------------------------------------
# จัดแกนและขนาด
# ---------------------------------------------------------------------------


def _symmetry_score(x: np.ndarray, bins: int = 64) -> float:
    """คะแนนความสมมาตรของการกระจายตัวตามแกนหนึ่ง (1 = สมมาตรเป๊ะ)"""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return 0.0
    h, _ = np.histogram(x, bins=bins, range=(lo, hi))
    h = h.astype(np.float64)
    if h.sum() <= 0:
        return 0.0
    h /= h.sum()
    r = h[::-1]
    denom = float(np.linalg.norm(h) * np.linalg.norm(r))
    return float(h @ r / denom) if denom > 0 else 0.0


def _guess_axes(v: np.ndarray) -> tuple[int, int, int]:
    """เดาว่าแกนไหนคือ ซ้าย-ขวา / ขึ้น / หน้า-หลัง คืน (right, up, front)

    เคยใช้ "แกนที่ยาวที่สุดคือแกนขึ้น" ซึ่งพังกับโมเดลฐานท่า T-pose: ช่วงแขน
    ของคนยาวพอ ๆ กับความสูง (วัดจากไฟล์จริงได้ 19.94 กับ 17.71) แกนแขนจึง
    ชนะไปแบบเฉียดฉิว แล้วโมเดลถูกวางนอนตะแคง

    สัญญาณที่เชื่อถือได้กว่ามากคือ *ความสมมาตร*: ร่างกายคนเป็นภาพกระจกรอบ
    ระนาบกลางตัว การกระจายตัวของ vertex ตามแกนซ้าย-ขวาจึงสมมาตรเกือบเป๊ะ
    ส่วนแกนขึ้น (หัวกับเท้าคนละแบบ) และแกนหน้า-หลัง (อกกับหลังคนละแบบ)
    ไม่สมมาตร เมื่อรู้แกนซ้าย-ขวาแล้ว ที่เหลือแยกด้วยความยาว: คนสูงกว่าหนา
    """
    sym = [_symmetry_score(v[:, a]) for a in range(3)]
    right = int(np.argmax(sym))

    rest = [a for a in range(3) if a != right]
    ext = v.max(0) - v.min(0)
    up, front = (rest[0], rest[1]) if ext[rest[0]] >= ext[rest[1]] \
        else (rest[1], rest[0])
    log.debug("คะแนนความสมมาตรต่อแกน XYZ: %s", np.round(sym, 3).tolist())
    return right, up, front


def _guess_front_sign(v: np.ndarray, up: int, front: int) -> int:
    """ทิศของแกนหน้า: +1 ถ้าตัวละครหันหน้าไปทางค่าบวกของแกนนั้น

    ใช้ปลายเท้า: ฝ่าเท้ายื่นไปข้างหน้ามากกว่าส้นเสมอ เทียบกับแกนกลางลำตัว
    เป็นสัญญาณเดียวกับที่ `preprocess.detect_side_facing` ใช้กับภาพ 2D และ
    เชื่อถือได้กว่าการดูใบหน้า เพราะทรงผม/หมวกด้านหลังมักหักล้างจมูก
    """
    lo, hi = float(v[:, up].min()), float(v[:, up].max())
    h = hi - lo
    feet = v[v[:, up] < lo + 0.06 * h]
    torso = v[(v[:, up] > lo + 0.45 * h) & (v[:, up] < lo + 0.75 * h)]
    if len(feet) < 10 or len(torso) < 10:
        return 1
    shift = float(feet[:, front].mean()) - float(torso[:, front].mean())
    return 1 if shift >= 0 else -1


def normalize(mesh, path: Path, cfg: PipelineConfig,
              up_axis: int | None = None,
              front: tuple[int, int] | None = None) -> BaseMesh:
    """หมุน/ย่อ/ขยับโมเดลฐานให้เข้าระบบพิกัดของ pipeline

    ปลายทาง: +X ขวาของตัวละคร, +Y ขึ้น, +Z ทิศที่หันหน้าไป,
    ฝ่าเท้าอยู่ที่ y = 0, แกนกลางลำตัวอยู่ที่ x = 0, สูงเท่า cfg.character_height_m
    """
    import trimesh

    v = np.asarray(mesh.vertices, np.float64)
    notes: list[str] = []

    right, up_g, front_g = _guess_axes(v)
    up = up_g if up_axis is None else up_axis
    if front is not None:
        f_axis, f_sign = front
    else:
        f_axis = front_g if up_axis is None else \
            [a for a in range(3) if a not in (up, right)][0]
        f_sign = _guess_front_sign(v, up, f_axis)
    right = [a for a in range(3) if a not in (up, f_axis)][0]

    log.info("โมเดลฐาน: แกนขวา = %s, แกนขึ้น = %s, แกนหน้า = %s%s",
             "XYZ"[right], "XYZ"[up], "+-"[f_sign < 0], "XYZ"[f_axis])

    # สร้างเมทริกซ์สลับแกน: คอลัมน์ใหม่ = (right, up, front)
    basis = np.zeros((3, 3))
    basis[0, right] = 1.0
    basis[1, up] = 1.0
    basis[2, f_axis] = f_sign
    # รักษา handedness ไว้ ไม่งั้นโมเดลจะกลับด้านซ้าย-ขวา
    if np.linalg.det(basis) < 0:
        basis[0] *= -1.0
        notes.append("สลับแกนซ้าย-ขวาเพื่อรักษา handedness")

    w = v @ basis.T

    h = float(w[:, 1].max() - w[:, 1].min())
    if h <= 1e-9:
        raise ValueError(f"โมเดลฐาน {path.name} แบนราบ (สูง {h}) — ไฟล์น่าจะเสีย")
    scale = cfg.character_height_m / h
    w *= scale

    # ฝ่าเท้าแตะพื้น และแกนกลางลำตัวอยู่ที่ x = 0
    #
    # ใช้ "ช่วงลำตัว" หาแกนกลาง ไม่ใช่กึ่งกลาง bbox ด้วยเหตุผลเดียวกับขั้น
    # preprocess: แขนที่กางไม่เท่ากันหรืออาวุธที่ถือข้างเดียวจะดึง bbox ไปคนละที่
    lo_y = float(w[:, 1].min())
    band = w[(w[:, 1] > lo_y + 0.45 * cfg.character_height_m)
             & (w[:, 1] < lo_y + 0.75 * cfg.character_height_m)]
    cx = float(np.median(band[:, 0])) if len(band) else float(np.median(w[:, 0]))
    cz = float(np.median(band[:, 2])) if len(band) else float(np.median(w[:, 2]))
    w -= np.array([cx, lo_y, cz])

    xform = np.eye(4)
    xform[:3, :3] = basis * scale
    xform[:3, 3] = -np.array([cx, lo_y, cz])

    faces = np.asarray(mesh.faces, np.int32)
    log.info("โมเดลฐานหลังจัดพิกัด: สูง %.2f m, กว้าง %.2f m, หนา %.2f m",
             w[:, 1].max() - w[:, 1].min(),
             w[:, 0].max() - w[:, 0].min(),
             w[:, 2].max() - w[:, 2].min())
    return BaseMesh(w, faces, path, xform, notes)


def load(path: Path, cfg: PipelineConfig,
         blender_exe: str = BLENDER_EXE, **kw) -> BaseMesh:
    """อ่านไฟล์โมเดลฐานแล้วจัดพิกัดให้พร้อมใช้"""
    return normalize(_load_raw(path, blender_exe), path, cfg, **kw)


# ---------------------------------------------------------------------------
# แปลงเป็น voxel เพื่อใช้ตัวประเมินข้อต่อตัวเดียวกับ hull
# ---------------------------------------------------------------------------


def solidify(verts: np.ndarray, faces: np.ndarray,
             shape: tuple[int, int, int], origin: np.ndarray,
             pitch: float) -> np.ndarray:
    """แปลง mesh เป็นกริดตัน (ny, nx, nz) ด้วยการนับจุดตัดตามแนวตั้ง

    ทำไมเขียนเอง ไม่ใช้ `trimesh.voxelized().fill()`: ตัวนั้นต้องการ mesh ที่
    ปิดสนิท (watertight) ซึ่งโมเดลฐานที่ export มาจากโปรแกรมปั้นแทบไม่เคยเป็น
    (ตา ปาก ช่องใน มักเปิดอยู่) และ fill ของมันต้องใช้ rtree ที่ไม่ได้ติดตั้ง

    วิธีนี้ยิงรังสีขึ้นตามแกน Y ทีละคอลัมน์ นับว่าตัดผิวกี่ครั้ง แล้วเติมเนื้อ
    ระหว่างคู่ที่ 1-2, 3-4, ... ซึ่งทนต่อรูเล็ก ๆ ได้ดีกว่า เพราะรูทำให้จำนวน
    จุดตัดของ *คอลัมน์นั้นคอลัมน์เดียว* เป็นเลขคี่ ไม่ลามไปทั้งก้อน
    """
    ny, nx, nz = shape
    tri = verts[faces]                                   # (F, 3, 3)

    # พิกัดคอลัมน์เป็นหน่วย voxel
    gx = (tri[:, :, 0] - origin[0]) / pitch
    gz = (tri[:, :, 2] - origin[2]) / pitch
    gy = (tri[:, :, 1] - origin[1]) / pitch

    hits: list[list[list[float]]] = [[[] for _ in range(nz)] for _ in range(nx)]

    x_lo = np.floor(gx.min(axis=1)).astype(np.int64)
    x_hi = np.ceil(gx.max(axis=1)).astype(np.int64)
    z_lo = np.floor(gz.min(axis=1)).astype(np.int64)
    z_hi = np.ceil(gz.max(axis=1)).astype(np.int64)
    keep = (x_hi >= 0) & (x_lo < nx) & (z_hi >= 0) & (z_lo < nz)

    for f in np.flatnonzero(keep):
        ax, az = gx[f, 0], gz[f, 0]
        bx, bz = gx[f, 1], gz[f, 1]
        ccx, ccz = gx[f, 2], gz[f, 2]
        den = (bz - ccz) * (ax - ccx) + (ccx - bx) * (az - ccz)
        if abs(den) < 1e-12:
            continue
        xs = np.arange(max(x_lo[f], 0), min(x_hi[f] + 1, nx)) + 0.5
        zs = np.arange(max(z_lo[f], 0), min(z_hi[f] + 1, nz)) + 0.5
        if xs.size == 0 or zs.size == 0:
            continue
        px, pz = np.meshgrid(xs, zs, indexing="ij")
        w0 = ((bz - ccz) * (px - ccx) + (ccx - bx) * (pz - ccz)) / den
        w1 = ((ccz - az) * (px - ccx) + (ax - ccx) * (pz - ccz)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        y = w0 * gy[f, 0] + w1 * gy[f, 1] + w2 * gy[f, 2]
        ix0 = max(x_lo[f], 0)
        iz0 = max(z_lo[f], 0)
        for a, b in zip(*np.nonzero(inside)):
            hits[ix0 + int(a)][iz0 + int(b)].append(float(y[a, b]))

    occ = np.zeros((ny, nx, nz), bool)
    odd = 0
    for ix in range(nx):
        col = hits[ix]
        for iz in range(nz):
            ys = col[iz]
            if len(ys) < 2:
                continue
            ys.sort()
            if len(ys) % 2:
                odd += 1
                ys = ys[:-1]
            for a, b in zip(ys[0::2], ys[1::2]):
                y0 = max(int(np.ceil(a - 0.5)), 0)
                y1 = min(int(np.floor(b - 0.5)), ny - 1)
                if y1 >= y0:
                    occ[y0:y1 + 1, ix, iz] = True
    if odd:
        log.debug("คอลัมน์ที่จุดตัดเป็นเลขคี่ %d คอลัมน์ (mesh มีรู) — ตัดจุดสุดท้ายทิ้ง",
                  odd)
    log.info("แปลงโมเดลฐานเป็น voxel %d x %d x %d — ทึบ %.2f%%",
             ny, nx, nz, 100 * occ.mean())
    return occ
