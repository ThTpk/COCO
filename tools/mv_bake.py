"""อบสีจากภาพถ่ายจริงลงบน mesh ที่ Hunyuan3D ปั้นมา — รันใน Blender เท่านั้น

    blender --background --factory-startup --python tools/mv_bake.py -- \
        --mesh output/croc_mv_shape.glb --views output/mv_views \
        --out output/croc_mv_textured.glb --tex output/croc_mv_albedo.png

mesh ที่ออกจาก Hunyuan3D มีแต่รูปทรง ไม่มี UV ไม่มีสี สคริปต์นี้

  1. นำเข้า .glb เชื่อมจุดซ้ำ แล้วกาง UV ด้วย Smart UV Project ของ Blender
  2. หามุมกล้องของภาพแต่ละใบเอง โดยลองหมุน mesh ทีละ 5 องศาแล้ววัด IoU
     ระหว่างเงาของ mesh กับ alpha ของภาพ — ไม่ต้องเชื่อชื่อไฟล์ว่า "left"
     แปลว่ากล้องอยู่ทางไหน และจับได้ด้วยถ้า mesh ออกมากลับซ้าย-ขวา
  3. ฉายภาพทุกใบกลับลงบน mesh เขียนลง texture atlas ใบเดียว โดยให้น้ำหนัก
     ตามว่าผิวหันเข้าหากล้องแค่ไหน และข้ามพิกเซลที่ถูกส่วนอื่นบังด้วย z-buffer
     แล้วผสมสองความถี่ (multi-band): สีพื้นจากค่าเฉลี่ยของทุกมุมซึ่งไม่มี
     รอยต่อ ส่วนรายละเอียดจากมุมที่หันตรงที่สุดมุมเดียวซึ่งคมที่สุด
  4. ประกอบวัสดุแล้ว export กลับเป็น .glb

ทำไมไม่ใช้ paint model ของ Hunyuan: มันวาดสีขึ้นใหม่จากภาพเดียว ส่วนงานนี้มี
ภาพจริงรอบตัวอยู่แล้วสิบใบ ฉายของจริงลงไปตรง ๆ ได้ผิวที่ตรงกับต้นฉบับกว่า
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np

UP = np.array([0.0, 0.0, 1.0])  # Blender เป็น Z-up (importer แปลงจาก Y-up ของ glTF ให้แล้ว)

# ลำดับที่กล้องวนรอบตัว ห่างกันมุมละเท่า ๆ กัน ใช้ล็อกระยะห่างตอนหามุมจากเงา
RING = ("front", "front-right", "right", "back-right",
        "back", "back-left", "left", "front-left")


def say(msg: str) -> None:
    print(f"[bake] {msg}", flush=True)


# --------------------------------------------------------------------------- scene


def load_mesh(path: Path, weld: bool = True):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"ไม่มี mesh ใน {path}")

    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # เชื่อมจุดซ้ำเฉพาะตอนจะกาง UV ใหม่ — ถ้าไฟล์มี UV มาแล้ว จุดที่ดูเหมือนซ้ำ
    # คือจุดที่ถูกผ่าตามรอยต่อ island การเชื่อมกลับทำให้ UV ของรอยต่อพัง
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if weld:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    for poly in obj.data.polygons:
        poly.use_smooth = True

    say(f"mesh {len(obj.data.vertices)} จุด {len(obj.data.polygons)} หน้า")
    return obj


def unwrap(obj, margin: float, keep: bool) -> None:
    """กาง UV ใหม่ เว้นแต่สั่ง keep แล้วไฟล์มี UV ติดมาอยู่แล้ว

    ถ้า mesh ถูกกางมาดีอยู่แล้ว การกางซ้ำมีแต่เสีย — Smart UV Project ตัดเป็น
    island เล็ก ๆ เต็มไปหมด ใช้เนื้อที่ atlas ได้น้อยกว่าของเดิมหลายเท่า
    """
    if keep and obj.data.uv_layers.active is not None:
        say("ใช้ UV ที่ติดมากับไฟล์")
        return
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=margin)
    bpy.ops.object.mode_set(mode="OBJECT")
    say("กาง UV ใหม่แล้ว")


def orient(obj, mirror: bool, yaw: float) -> None:
    """พลิก/หมุน object จริง แล้ว apply ลงใน mesh data

    การพลิกแกนเดียวทำให้ลำดับจุดของหน้ากลับด้าน นอร์มอลเลยชี้เข้าในทั้งตัว
    จึงต้องสั่งคำนวณนอร์มอลใหม่หลังพลิกทุกครั้ง
    """
    if not mirror and abs(yaw) < 1e-9:
        return
    bpy.context.view_layer.objects.active = obj
    if mirror:
        obj.scale.x = -1.0
    obj.rotation_euler = (0.0, 0.0, yaw)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    if mirror:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()


def mesh_arrays(obj):
    me = obj.data
    me.calc_loop_triangles()

    nv = len(me.vertices)
    verts = np.empty(nv * 3, dtype=np.float64)
    me.vertices.foreach_get("co", verts)
    verts = verts.reshape(nv, 3)

    normals = np.empty(nv * 3, dtype=np.float64)
    me.vertices.foreach_get("normal", normals)
    normals = normals.reshape(nv, 3)

    nt = len(me.loop_triangles)
    tri_v = np.empty(nt * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tri_v)
    tri_v = tri_v.reshape(nt, 3)

    tri_l = np.empty(nt * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("loops", tri_l)
    tri_l = tri_l.reshape(nt, 3)

    nl = len(me.loops)
    uvs = np.empty(nl * 2, dtype=np.float64)
    me.uv_layers.active.data.foreach_get("uv", uvs)
    uvs = uvs.reshape(nl, 2)

    return verts, normals, tri_v, uvs[tri_l]


# --------------------------------------------------------------------------- images


def read_png(path: Path) -> np.ndarray:
    """คืน array (h, w, 4) ค่า 0..1 ตามที่เก็บในไฟล์ (ยังเป็น sRGB ไม่ใช่ linear)"""
    img = bpy.data.images.load(str(path))
    img.colorspace_settings.name = "Non-Color"  # กัน Blender แปลงค่าให้ตอนอ่าน
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    # Blender ไล่แถวจากล่างขึ้นบน ส่วนเราคิดพิกัดภาพจากบนลงล่าง
    return buf.reshape(h, w, 4)[::-1]


def write_png(path: Path, rgba: np.ndarray) -> None:
    h, w = rgba.shape[:2]
    img = bpy.data.images.new("croc_mv_albedo", width=w, height=h, alpha=True)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(rgba[::-1].astype(np.float32).ravel())
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(alpha > 0.5)
    return xs.min(), xs.max() + 1, ys.min(), ys.max() + 1


# --------------------------------------------------------------------------- กล้อง


def basis(az: float) -> tuple[np.ndarray, np.ndarray]:
    """คืน (ทิศจากจุดกำเนิดไปหากล้อง, ทิศขวาของจอ) ที่มุม az เรเดียน"""
    toward = np.array([math.sin(az), -math.cos(az), 0.0])
    right = np.array([math.cos(az), math.sin(az), 0.0])
    return toward, right


def fit_photo(pts_u: np.ndarray, pts_v: np.ndarray, alpha: np.ndarray,
              grid: int = 192, iters: int = 6):
    """จับ mesh ให้ทาบภาพด้วยพื้นที่เงากับจุดศูนย์ถ่วง ไม่ใช่กรอบสี่เหลี่ยม

    กรอบสี่เหลี่ยมขึ้นกับจุดที่ยื่นสุดเพียงไม่กี่จุด — ปลายเล็บหรือหนามหางที่
    mesh ปั้นยาวเกินภาพนิดเดียวก็ดันทั้งกรอบ ทำให้ทั้งตัวเลื่อน สีจากแขนเลย
    ไปตกบนเสื้อ พื้นที่กับจุดศูนย์ถ่วงเฉลี่ยจากทั้งเงา จึงไม่สะเทือนตามจุดเดี่ยว

    ไล่ปรับสเกลกับตำแหน่งสลับกันไปมา เพราะขยับอย่างหนึ่งแล้วอีกอย่างเพี้ยนตาม
    """
    h, w = alpha.shape
    sy, sx = h / grid, w / grid
    rows = np.minimum((np.arange(grid) * sy).astype(np.int32), h - 1)
    cols = np.minimum((np.arange(grid) * sx).astype(np.int32), w - 1)
    ref = alpha[np.ix_(rows, cols)] > 0.5
    ry, rx = np.nonzero(ref)
    ref_area, ref_cx, ref_cy = ref.sum(), rx.mean(), ry.mean()

    x0, x1, y0, y1 = alpha_bbox(alpha)
    umin, umax = pts_u.min(), pts_u.max()
    vmin, vmax = pts_v.min(), pts_v.max()
    s = 0.5 * ((x1 - x0) / max(umax - umin, 1e-9) + (y1 - y0) / max(vmax - vmin, 1e-9))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    cu, cv = (umin + umax) / 2.0, (vmin + vmax) / 2.0

    for _ in range(iters):
        gx = np.clip(((cx + (pts_u - cu) * s) / sx).astype(np.int32), 0, grid - 1)
        gy = np.clip(((cy - (pts_v - cv) * s) / sy).astype(np.int32), 0, grid - 1)
        m = np.zeros((grid, grid), bool)
        m[gy, gx] = True
        my, mx = np.nonzero(m)
        if len(my) == 0:
            break
        cx += (ref_cx - mx.mean()) * sx
        cy += (ref_cy - my.mean()) * sy
        s *= math.sqrt(ref_area / max(m.sum(), 1))

    def to_px(u, v):
        return cx + (u - cu) * s, cy - (v - cv) * s

    return to_px


def fit_screen(pts_u: np.ndarray, pts_v: np.ndarray, box, shape):
    """คืนฟังก์ชันแปลงพิกัดจอ -> พิกเซล โดยจับกรอบ mesh ให้ทับกรอบ alpha ของภาพ"""
    x0, x1, y0, y1 = box
    umin, umax = pts_u.min(), pts_u.max()
    vmin, vmax = pts_v.min(), pts_v.max()
    # สเกลเดียวทั้งสองแกน เฉลี่ยจากสองแกนเพื่อไม่ให้ภาพถูกยืด
    s = 0.5 * ((x1 - x0) / max(umax - umin, 1e-9) + (y1 - y0) / max(vmax - vmin, 1e-9))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cu, cv = (umin + umax) / 2, (vmin + vmax) / 2

    def to_px(u, v):
        return cx + (u - cu) * s, cy - (v - cv) * s  # แกน y ของภาพชี้ลง

    return to_px


GRID = 128  # ความละเอียดของเงาที่ใช้ตอนเทียบมุม ไม่ต้องมาก แค่พอบอกทรงได้


def grow(m: np.ndarray, rounds: int) -> np.ndarray:
    out = m.copy()
    for _ in range(rounds):
        src = out.copy()
        out[1:] |= src[:-1]
        out[:-1] |= src[1:]
        out[:, 1:] |= src[:, :-1]
        out[:, :-1] |= src[:, 1:]
    return out


def mesh_shadow(points: np.ndarray, az: float) -> np.ndarray:
    """เงาของ mesh ที่มุม az ปรับให้เต็มกรอบ เพื่อเทียบกับภาพที่ครอปมาแล้ว"""
    _, right = basis(az)
    u = points @ right
    v = points @ UP

    def cell(a):
        lo, hi = a.min(), a.max()
        return ((a - lo) / max(hi - lo, 1e-9) * (GRID - 1)).astype(np.int32)

    m = np.zeros((GRID, GRID), bool)
    m[GRID - 1 - cell(v), cell(u)] = True
    # mesh เป็นจุด ๆ ต้องพองก่อนจึงจะเทียบกับเงาตันได้
    return grow(m, 2)


def photo_shadow(alpha: np.ndarray) -> np.ndarray:
    x0, x1, y0, y1 = alpha_bbox(alpha)
    crop = alpha[y0:y1, x0:x1] > 0.5
    ys = np.linspace(0, crop.shape[0] - 1, GRID).astype(np.int32)
    xs = np.linspace(0, crop.shape[1] - 1, GRID).astype(np.int32)
    return crop[np.ix_(ys, xs)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1)


def fit_ring(verts: np.ndarray, shadows: dict[str, np.ndarray], order: list[str]):
    """หาว่ากล้องวนรอบตัวไปทางไหน และ mesh หันหน้าเบนไปกี่องศา

    ภาพชุดนี้เป็น turntable แปดมุมห่างกัน 45 องศาตามลำดับชื่อไฟล์ จึงเหลือ
    พารามิเตอร์แค่สามตัว — วนตามเข็มหรือทวนเข็ม, เบนไปกี่องศา, และ mesh
    กลับซ้าย-ขวาไหม การล็อกระยะห่างไว้ทำให้ทนต่อเงาที่คล้ายกันของมุมข้างเคียง
    ซึ่งถ้าปล่อยให้แต่ละภาพหามุมอิสระจะเดาสลับกันมั่ว
    """
    step = 2 * math.pi / len(order)
    nominal = {name: i * step for i, name in enumerate(order)}

    best = None
    for mirror in (False, True):
        pts = verts * np.array([-1.0, 1.0, 1.0]) if mirror else verts
        # เงาของ mesh ไม่ขึ้นกับภาพ คิดครั้งเดียวต่อมุมแล้วใช้ซ้ำทุก candidate
        cache = {a: mesh_shadow(pts, math.radians(a)) for a in range(0, 360, 5)}
        for spin in (1, -1):
            for yaw in range(0, 360, 5):
                total = 0.0
                for name in order:
                    a = round(math.degrees(spin * nominal[name]) + yaw) % 360
                    total += iou(cache[a - a % 5], shadows[name])
                if best is None or total > best[0]:
                    best = (total, mirror, spin, math.radians(yaw))

    total, mirror, spin, yaw = best
    az = {name: (spin * nominal[name] + yaw) % (2 * math.pi) for name in order}
    return mirror, az, total / len(order)


# --------------------------------------------------------------------------- raster


def raster_depth(verts, tris, az, size, to_px):
    """z-buffer แบบ orthographic ของมุม az ใช้ตอบว่าพิกเซลนั้นเห็นผิวที่ลึกเท่าไร"""
    toward, right = basis(az)
    u = verts @ right
    v = verts @ UP
    d = verts @ toward  # ยิ่งมากยิ่งใกล้กล้อง
    px, py = to_px(u, v)

    buf = np.full((size, size), -1e9, dtype=np.float64)
    ax, ay = px[tris[:, 0]], py[tris[:, 0]]
    bx, by = px[tris[:, 1]], py[tris[:, 1]]
    cx, cy = px[tris[:, 2]], py[tris[:, 2]]
    da, db, dc = d[tris[:, 0]], d[tris[:, 1]], d[tris[:, 2]]

    lo_x = np.floor(np.minimum(np.minimum(ax, bx), cx)).astype(np.int32)
    hi_x = np.ceil(np.maximum(np.maximum(ax, bx), cx)).astype(np.int32)
    lo_y = np.floor(np.minimum(np.minimum(ay, by), cy)).astype(np.int32)
    hi_y = np.ceil(np.maximum(np.maximum(ay, by), cy)).astype(np.int32)
    np.clip(lo_x, 0, size - 1, out=lo_x)
    np.clip(hi_x, 0, size, out=hi_x)
    np.clip(lo_y, 0, size - 1, out=lo_y)
    np.clip(hi_y, 0, size, out=hi_y)

    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)

    for i in range(len(tris)):
        if abs(area[i]) < 1e-12 or hi_x[i] <= lo_x[i] or hi_y[i] <= lo_y[i]:
            continue
        xs = np.arange(lo_x[i], hi_x[i]) + 0.5
        ys = np.arange(lo_y[i], hi_y[i]) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        w0 = ((bx[i] - ax[i]) * (gy - ay[i]) - (gx - ax[i]) * (by[i] - ay[i])) / area[i]
        w1 = ((gx - ax[i]) * (cy[i] - ay[i]) - (cx[i] - ax[i]) * (gy - ay[i])) / area[i]
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        if not inside.any():
            continue
        z = da[i] + w1 * (db[i] - da[i]) + w0 * (dc[i] - da[i])
        sub = buf[lo_y[i]:hi_y[i], lo_x[i]:hi_x[i]]
        np.maximum(sub, np.where(inside, z, -1e9), out=sub)
    return buf


def raster_uv(tri_uv, tris, verts, normals, size):
    """แปลงแผ่น UV เป็นแผนที่ตำแหน่ง/นอร์มอลต่อเท็กเซล"""
    pos = np.zeros((size, size, 3))
    nrm = np.zeros((size, size, 3))
    hit = np.zeros((size, size), bool)

    px = tri_uv[:, :, 0] * (size - 1)
    py = (1.0 - tri_uv[:, :, 1]) * (size - 1)
    ax, bx, cx = px[:, 0], px[:, 1], px[:, 2]
    ay, by, cy = py[:, 0], py[:, 1], py[:, 2]

    lo_x = np.clip(np.floor(np.minimum(np.minimum(ax, bx), cx)).astype(np.int32) - 1, 0, size - 1)
    hi_x = np.clip(np.ceil(np.maximum(np.maximum(ax, bx), cx)).astype(np.int32) + 1, 0, size)
    lo_y = np.clip(np.floor(np.minimum(np.minimum(ay, by), cy)).astype(np.int32) - 1, 0, size - 1)
    hi_y = np.clip(np.ceil(np.maximum(np.maximum(ay, by), cy)).astype(np.int32) + 1, 0, size)
    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)

    pa, pb, pc = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    na, nb, nc = normals[tris[:, 0]], normals[tris[:, 1]], normals[tris[:, 2]]

    for i in range(len(tris)):
        if abs(area[i]) < 1e-12 or hi_x[i] <= lo_x[i] or hi_y[i] <= lo_y[i]:
            continue
        xs = np.arange(lo_x[i], hi_x[i]) + 0.5
        ys = np.arange(lo_y[i], hi_y[i]) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        w0 = ((bx[i] - ax[i]) * (gy - ay[i]) - (gx - ax[i]) * (by[i] - ay[i])) / area[i]
        w1 = ((gx - ax[i]) * (cy[i] - ay[i]) - (cx[i] - ax[i]) * (gy - ay[i])) / area[i]
        # เผื่อขอบออกไปเล็กน้อย เพื่อให้สีล้นข้ามรอยต่อ island กันเส้นขาว
        inside = (w0 >= -0.04) & (w1 >= -0.04) & (w0 + w1 <= 1.04)
        if not inside.any():
            continue
        w0c = np.clip(w0, 0, 1)
        w1c = np.clip(w1, 0, 1)
        w2c = np.clip(1 - w0c - w1c, 0, 1)
        sl = (slice(lo_y[i], hi_y[i]), slice(lo_x[i], hi_x[i]))
        take = inside & ~hit[sl]
        if not take.any():
            continue
        p = (w2c[..., None] * pa[i] + w1c[..., None] * pb[i] + w0c[..., None] * pc[i])
        n = (w2c[..., None] * na[i] + w1c[..., None] * nb[i] + w0c[..., None] * nc[i])
        pos[sl][take] = p[take]
        nrm[sl][take] = n[take]
        hit[sl][take] = True
    return pos, nrm, hit


def sample(img: np.ndarray, px: np.ndarray, py: np.ndarray):
    """สุ่มสีแบบ bilinear คืน (rgb, alpha)"""
    h, w = img.shape[:2]
    x = np.clip(px - 0.5, 0, w - 1.001)
    y = np.clip(py - 0.5, 0, h - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    p = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return p[..., :3], p[..., 3]


def _box1d(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """เฉลี่ยแบบหน้าต่างเลื่อน กว้าง 2r+1 ด้วย cumsum — O(n) ไม่ขึ้นกับรัศมี"""
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    ap = np.pad(a, pad, mode="edge")
    zero = np.zeros_like(np.take(ap, [0], axis=axis))
    c = np.concatenate([zero, np.cumsum(ap, axis=axis)], axis=axis)
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def blur_masked(x: np.ndarray, mask: np.ndarray, radius: int,
                passes: int = 3) -> np.ndarray:
    """เบลอโดยนับเฉพาะเท็กเซลที่มีสีจริง

    เท็กเซลว่างในผืน UV ถ้าปล่อยให้ร่วมเฉลี่ยจะลากค่าลงเป็นศูนย์ ขอบของทุก
    เกาะ UV จึงคล้ำ วิธีมาตรฐานคือเบลอ "ค่าคูณ mask" กับ "mask" แยกกัน
    แล้วค่อยหารกันตอนท้าย ได้ค่าเฉลี่ยที่ถ่วงด้วยความมีอยู่จริง
    ทำ box blur ซ้ำหลายรอบแทน Gaussian เพราะเร็วกว่ามากที่รัศมีกว้าง ๆ
    """
    m = mask.astype(np.float32)
    num = (x.astype(np.float32) * m[..., None])
    den = m
    for _ in range(passes):
        for axis in (0, 1):
            num = _box1d(num, radius, axis)
            den = _box1d(den, radius, axis)
    return num / np.maximum(den, 1e-6)[..., None]


def dilate(color: np.ndarray, valid: np.ndarray, rounds: int) -> np.ndarray:
    """ลามสีออกไปกินช่องว่าง เพื่อไม่ให้เห็นเส้นขาวตามรอยต่อ UV ตอน mipmap"""
    out = color.copy()
    ok = valid.copy()
    for _ in range(rounds):
        acc = np.zeros_like(out)
        cnt = np.zeros(ok.shape, np.float64)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s = np.roll(np.roll(out, dy, 0), dx, 1)
            m = np.roll(np.roll(ok, dy, 0), dx, 1)
            acc += s * m[..., None]
            cnt += m
        grow = (~ok) & (cnt > 0)
        out[grow] = acc[grow] / cnt[grow][..., None]
        ok |= grow
    return out


# --------------------------------------------------------------------------- งานหลัก


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--views", type=Path, required=True, help="โฟลเดอร์ภาพ cutout RGBA")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tex", type=Path, required=True)
    ap.add_argument("--size", type=int, default=2048, help="ขนาด texture atlas")
    ap.add_argument("--depth-size", type=int, default=1024)
    ap.add_argument("--margin", type=float, default=0.0012)
    ap.add_argument("--power", type=float, default=3.0, help="ยิ่งสูง ยิ่งเชื่อเฉพาะมุมที่หันตรง")
    ap.add_argument("--edge-guard", type=float, default=0.012,
                    help="หดเงาของภาพเข้ามากี่ส่วนของความสูงภาพ ก่อนรับสี")
    ap.add_argument("--lowband", type=int, default=48,
                    help="รัศมีเบลอของแถบความถี่ต่ำ 0 = ปิด multi-band")
    ap.add_argument("--sharpness", type=float, default=8.0,
                    help="ยิ่งสูง ความถี่สูงยิ่งมาจากมุมที่ดีที่สุดมุมเดียว "
                         "(คมขึ้นแต่เสี่ยงรอยต่อ) 1 = เท่าน้ำหนักปกติ")
    ap.add_argument("--keep-uv", action="store_true",
                    help="ใช้ UV เดิมของไฟล์ แทนที่จะกางใหม่")
    args = ap.parse_args(argv)

    obj = load_mesh(args.mesh, weld=not args.keep_uv)
    unwrap(obj, args.margin, args.keep_uv)
    verts, normals, tris, tri_uv = mesh_arrays(obj)

    images, order = {}, []
    for view in RING:
        hits = list(args.views.glob(f"*_{view}.png"))
        if not hits:
            raise SystemExit(f"ไม่พบภาพมุม {view} ใน {args.views}")
        name = hits[0].stem
        images[name] = read_png(hits[0])
        order.append(name)

    shadows = {n: photo_shadow(img[..., 3]) for n, img in images.items()}
    best_flip, best_az, score = fit_ring(verts, shadows, order)
    say(f"เทียบเงาได้ IoU เฉลี่ย {score:.2f}")

    # หมุนให้ภาพ front อยู่ที่ 0 องศา — ขั้นต่อไป (tools/autorig.py) ต้องการให้
    # ตัวละครหันหน้าไป +Z ของ glTF ซึ่งตรงกับ -Y ของ Blender พอดี
    front_key = next((k for k in best_az if k.endswith("front")), None)
    if front_key is None:
        raise SystemExit("ต้องมีภาพชื่อ ...front.png ไว้อ้างทิศหน้า")
    az0 = best_az[front_key]

    # ดัดตัว object จริงใน Blender ไม่ใช่แค่ array ที่ถืออยู่ ไฟล์ที่ export ออกไป
    # จึงหันหน้าถูกทางด้วย
    orient(obj, mirror=best_flip, yaw=-az0)
    verts, normals, tris, tri_uv = mesh_arrays(obj)
    best_az = {k: v - az0 for k, v in best_az.items()}
    if best_flip:
        say("mesh ออกมากลับซ้าย-ขวา พลิกกลับให้แล้ว")
    say(f"หมุนตัวละคร {math.degrees(-az0):.0f} องศา ให้หันหน้าตรงมุมกล้อง front")
    for name in sorted(best_az):
        say(f"  {name}: {math.degrees(best_az[name]) % 360:.0f} องศา")

    size = args.size
    pos, nrm, hit = raster_uv(tri_uv, tris, verts, normals, size)
    say(f"แผ่น UV ใช้จริง {100 * hit.mean():.1f}% ของพื้นที่")

    nrm_len = np.linalg.norm(nrm, axis=2, keepdims=True)
    nrm = nrm / np.maximum(nrm_len, 1e-9)
    span = float(np.linalg.norm(verts.max(0) - verts.min(0)))

    acc = np.zeros((size, size, 3))
    wsum = np.zeros((size, size))
    # ความถี่สูงมาจากมุมที่หันตรงที่สุด เพราะการเฉลี่ยภาพที่ทาบกันไม่สนิท
    # คือการทำให้เบลอ แต่ "เลือกมุมเดียวที่ชนะ" ก็แรงเกินไป: ตรงที่สองมุมคะแนน
    # ก้ำกึ่งกัน ผู้ชนะจะสลับไปมาเป็นหย่อม ๆ ได้รอยต่อหยักและดูดพิกเซลที่ฉาย
    # ผิด (ผิวแขนไปตกบนเสื้อ) เข้ามาเต็ม ๆ แทนที่จะถูกเฉลี่ยจนจาง
    #
    # เลยใช้ค่าเฉลี่ยที่ถ่วงด้วย w^sharpness แทน ยกกำลังสูง ๆ ทำให้มุมที่ดีที่สุด
    # กินน้ำหนักเกือบทั้งหมด (ได้ความคมเท่าผู้ชนะ) แต่ตรงที่คะแนนก้ำกึ่ง
    # มันยังเกลี่ยให้ จึงไม่มีรอยต่อหยัก — อัตราส่วนนี้ไม่ขึ้นกับสเกลของ w
    # จึงไม่ต้องนอร์มอลไลซ์ก่อน
    acc_hi = np.zeros((size, size, 3))
    wsum_hi = np.zeros((size, size))
    flat_pos = pos[hit]
    flat_nrm = nrm[hit]

    # ให้คะแนนความ "ลึกเข้าไปในเงา" ของทุกพิกเซล ไว้ถ่วงน้ำหนักตอนสุ่มสี
    # ตัดขาดที่ริมขอบทำให้ผิวที่เห็นได้จากมุมเฉียงเท่านั้นกลายเป็นรูดำ
    # ไล่น้ำหนักแทน ริมขอบจึงยังมีสีให้ใช้ แค่แพ้มุมที่เห็นชัดกว่า
    inside = {}
    for name, img in images.items():
        pad = max(1, int(round(args.edge_guard * img.shape[0])))
        depth_in = np.zeros(img.shape[:2])
        shrunk = img[..., 3] > 0.5
        for _ in range(pad):
            shrunk = ~grow(~shrunk, 1)
            depth_in += shrunk
        inside[name] = depth_in / pad
        say(f"  ไล่น้ำหนักขอบ {name} ลึก {pad} พิกเซล")

    for name, img in images.items():
        az = best_az[name]
        toward, right = basis(az)
        mesh_u, mesh_v = verts @ right, verts @ UP
        to_px = fit_photo(mesh_u, mesh_v, img[..., 3])
        # z-buffer ใช้กรอบของตัวเอง ไม่ต้องตรงกับภาพ ขอแค่ครอบ mesh ให้พอดี
        to_depth_px = fit_screen(mesh_u, mesh_v,
                                 (0, args.depth_size, 0, args.depth_size), None)
        depth = raster_depth(verts, tris, az, args.depth_size, to_depth_px)

        # ใช้ค่าสัมบูรณ์ เพราะ mesh จาก marching cubes มีหน้าที่นอร์มอลกลับด้าน
        # ปนอยู่เป็นหย่อม ถ้าตัดด้วยเครื่องหมาย หย่อมพวกนั้นจะไม่ได้สีจากมุมไหนเลย
        # กลายเป็นรูดำ ส่วนเรื่องถูกบังหรือไม่ ปล่อยให้ z-buffer ข้างล่างตัดสิน
        facing = np.abs(flat_nrm @ toward)
        keep = facing > 0.05
        px, py = to_px(flat_pos @ right, flat_pos @ UP)
        dpx, dpy = to_depth_px(flat_pos @ right, flat_pos @ UP)

        ix = np.clip(dpx.astype(np.int32), 0, args.depth_size - 1)
        iy = np.clip(dpy.astype(np.int32), 0, args.depth_size - 1)
        own = flat_pos @ toward
        # ถูกบัง = มีผิวอื่นอยู่หน้ากว่าเราเกินค่าเผื่อ
        visible = own > depth[iy, ix] - 0.012 * span

        rgb, a = sample(img, px, py)
        # กันขอบ: mesh กับภาพไม่ได้ทาบกันเป๊ะ ริมเงาจึงเป็นที่ที่พลาดบ่อยที่สุด
        # — แขนของภาพไปตกบนลำตัวของ mesh ได้ง่าย ๆ หดเงาเข้ามาก่อนแล้วรับเฉพาะ
        # ส่วนในจึงปลอดภัยกว่า ยอมให้ริมขอบว่างแล้วไปเติมด้วยการลามสีทีหลัง
        gy = np.clip(py.astype(np.int32), 0, img.shape[0] - 1)
        gx = np.clip(px.astype(np.int32), 0, img.shape[1] - 1)
        edge = inside[name][gy, gx]
        w = np.where(keep & visible & (a > 0.98),
                     np.clip(facing, 0, 1) ** args.power * edge, 0.0)

        lin = srgb_to_linear(rgb)
        acc[hit] += lin * w[..., None]
        wsum[hit] += w
        w_hi = w ** args.sharpness
        acc_hi[hit] += lin * w_hi[..., None]
        wsum_hi[hit] += w_hi
        say(f"  ฉาย {name} ครอบคลุม {100 * (w > 0).mean():.0f}% ของเท็กเซล")

    covered = wsum > 1e-6
    mean_c = np.zeros((size, size, 3))
    mean_c[covered] = acc[covered] / wsum[covered][..., None]
    say(f"เท็กเซลที่ได้สี {100 * covered.sum() / max(hit.sum(), 1):.1f}% ของที่ใช้จริง")

    if args.lowband > 0:
        # --- multi-band blending ---------------------------------------
        # ค่าเฉลี่ยถ่วงน้ำหนักให้สีพื้นที่ไม่มีรอยต่อ แต่เบลอ เพราะภาพแต่ละใบ
        # ทาบกันไม่สนิทระดับพิกเซล ส่วนมุมที่ชนะมุมเดียวให้รายละเอียดคมแต่มี
        # รอยต่อ เพราะ exposure กับเงาของแต่ละใบไม่เท่ากัน
        #
        # เอาข้อดีทั้งคู่โดยแยกความถี่: ยืนพื้นด้วยภาพที่คม แล้วบวกด้วย
        # "ส่วนต่างระหว่างค่าเฉลี่ยกับภาพคม" ที่เบลอจนเหลือแต่ความถี่ต่ำ
        #
        #     ผล = คม + blur(เฉลี่ย - คม)
        #
        # ส่วนต่างนั้นคือค่าชดเชย exposure/เงา ซึ่งเปลี่ยนช้า ๆ ทั่วผิว
        # เบลอแล้วจึงไม่กินรายละเอียด และเพราะมันเป็นค่าที่เล็กและเรียบอยู่แล้ว
        # การที่รัศมีเบลอล้นข้ามเกาะ UV จึงส่งผลเป็นอันดับสอง ไม่ใช่สีเพี้ยน
        # แบบที่จะเกิดถ้าเบลอตัวสีตรง ๆ
        sharp_img = np.zeros((size, size, 3))
        hot = wsum_hi > 1e-12
        sharp_img[hot] = acc_hi[hot] / wsum_hi[hot][..., None]
        # เขียนอีกแบบที่เท่ากันทางพีชคณิต: คม + blur(เฉลี่ย - คม)
        #                                 = blur(เฉลี่ย) + [คม - blur(คม)]
        # แบบหลังแยก "ความถี่สูง" ออกมาเป็นก้อนเดียวให้จับต้องได้ จึงคุมมันได้
        low = blur_masked(mean_c, covered, args.lowband)
        detail = sharp_img - blur_masked(sharp_img, covered, args.lowband)
        color = np.where(covered[..., None], low + detail, 0.0)
        color = np.clip(color, 0.0, None)
        say(f"multi-band: สีพื้นจากค่าเฉลี่ย รายละเอียดถ่วงด้วย "
            f"w^{args.sharpness:g} (รัศมีเบลอ {args.lowband} พิกเซล)")
    else:
        color = mean_c
        say("multi-band: ปิดอยู่ ใช้ค่าเฉลี่ยถ่วงน้ำหนักล้วน")

    color = dilate(color, covered, rounds=48)
    rgba = np.concatenate([linear_to_srgb(color), np.ones((size, size, 1))], axis=2)
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    write_png(args.tex, rgba)
    say(f"เขียน {args.tex}")

    mat = bpy.data.materials.new("croc_mv")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.62
    bsdf.inputs["Metallic"].default_value = 0.0
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(args.tex))
    tex.image.colorspace_settings.name = "sRGB"
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(args.out), export_format="GLB",
                              use_selection=False, export_yup=True)
    say(f"เขียน {args.out}")


if __name__ == "__main__":
    main()
