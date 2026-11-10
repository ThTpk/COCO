"""เกลารอยต่อระหว่างหัวรายละเอียดสูงกับตัว แล้วบันทึกเป็น mesh ชิ้นเดียว

    blender -b --python blender/merge_head.py -- [ตัวเลือก]
หรือรันใน Python Console ของ Blender:
    p=r"<path>\\blender\\merge_head.py"; exec(compile(open(p,encoding="utf-8").read(),p,"exec"),{"__file__":p,"__name__":"__main__"})

ไฟล์เข้า output/croc_fixA_newhead.glb ได้มาจากการ boolean union ตัว (ซ่อมรูให้ปิดแล้ว
ตัดหัวเดิมออกที่ระดับคอเสื้อ) กับหัวใหม่ จึงเป็นผิวปิดชิ้นเดียวอยู่แล้ว แต่ตรงรอยต่อ
ยังเป็นขั้นบันได สคริปต์นี้

1. ตรวจว่า mesh ปิดสนิทไหม ถ้ายังไม่ปิด (เช่นไฟล์ที่แค่วางสองชิ้นซ้อนกัน) จะ
   อุดรู -> Voxel Remesh หลอมเป็นชิ้นเดียว -> Shrinkwrap ดูดกลับไปหาผิวเดิม
   นอกแถบรอยต่อเพื่อคืนรายละเอียดเล็ก ๆ ที่ voxel ทำมน
2. Smooth เฉพาะแถบรอยต่อ น้ำหนักจางออกจากแนวตัดแบบ smoothstep
3. Decimate ถ้าจำนวน face เกินเป้า
4. Export .glb และบันทึก .blend ไว้แก้ต่อ

แกน Z ของ Blender ตรงกับแกน Y ของไฟล์ glTF แนวตัดปริยาย 0.57

ตัวเลือก
  --in PATH       ไฟล์ต้นทาง (ปริยาย output/croc_fixA_newhead.glb)
  --out PATH      ไฟล์ปลายทาง (ปริยาย output/croc_fixA_newhead_merged.glb)
  --seam F        ระดับแนวตัดแกน Z (ปริยาย 0.57)
  --seams S ...   หลายแนวตัด AXIS:VALUE[:ZMIN:ZMAX] หรือ AXIS:VALUE:GAXIS:MIN:MAX
                  เช่น  Z:0.57   X:-0.445:0.22:0.62   Z:-0.70:X:-0.3:0.3
                  (แกน Blender: X = X ของ glTF, Z = Y ของ glTF)
  --band F        ครึ่งความกว้างแถบที่จะเกลา (ปริยาย 0.05)
  --smooth N      จำนวนรอบ smooth (ปริยาย 60)
  --strength F    ความแรงต่อรอบ (factor) ของ Smooth modifier (ปริยาย 0.8)
  --remesh MODE   auto | always | never (ปริยาย auto = เฉพาะเมื่อ mesh ยังไม่ปิด)
  --voxel F       ขนาด voxel ตอน remesh (ปริยาย 0.0025)
  --faces N       face เป้าหมายหลัง decimate, 0 = ไม่ลด (ปริยาย 400000)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import bmesh
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def args() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=os.path.join(ROOT, "output", "croc_fixA_newhead.glb"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "croc_fixA_newhead_merged.glb"))
    ap.add_argument("--seam", type=float, default=0.57,
                    help="แนวตัดแกน Z เดียว (ใช้เมื่อไม่ได้ระบุ --seams)")
    ap.add_argument("--seams", nargs="+", default=None,
                    help="หลายแนวตัด รูปแบบ AXIS:VALUE[:ZMIN:ZMAX] เช่น X:-0.44:0.22:0.62")
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--smooth", type=int, default=60)
    ap.add_argument("--strength", type=float, default=0.8, help="factor ของ Smooth modifier")
    ap.add_argument("--remesh", choices=("auto", "always", "never"), default="auto")
    ap.add_argument("--voxel", type=float, default=0.0025)
    ap.add_argument("--faces", type=int, default=400000)
    return ap.parse_args(raw)


def log(msg: str, t0: float) -> None:
    print(f"[merge_head {time.time() - t0:6.1f}s] {msg}", flush=True)


def clear_scene() -> None:
    """โหมด headless รีเซ็ตทั้งไฟล์ได้ แต่ถ้ารันจาก Console/Text Editor การรีเซ็ต
    จะลบหน้าต่างที่สคริปต์กำลังรันอยู่ จึงแค่ลบวัตถุและ mesh เดิมทิ้ง"""
    if bpy.app.background:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        return
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me)


def import_mesh(path: str) -> bpy.types.Object:
    """นำเข้า glb รวมทุก mesh เป็นวัตถุเดียว และ apply transform ลงเนื้อ mesh"""
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"ไม่พบ mesh ใน {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in list(bpy.context.scene.objects):
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    return obj


def apply_modifier(obj: bpy.types.Object, mod: bpy.types.Modifier) -> None:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def count_open_edges(obj: bpy.types.Object) -> int:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return n


def parse_seams(a) -> list[tuple[int, float, int, float, float]]:
    """คืนรายการ (แกน, ค่า, แกนจำกัดช่วง, min, max) — แกน 0/1/2 = X/Y/Z ของ Blender

    รูปแบบ  AXIS:VALUE                      แนวตัดเต็มระนาบ
            AXIS:VALUE:MIN:MAX              จำกัดช่วงตามแกน Z
            AXIS:VALUE:GAXIS:MIN:MAX        จำกัดช่วงตามแกน GAXIS
    """
    if not a.seams:
        return [(2, a.seam, 2, -1e9, 1e9)]
    out = []
    for spec in a.seams:
        parts = spec.split(":")
        axis = "XYZ".index(parts[0].upper())
        val = float(parts[1])
        if len(parts) == 2:
            g, lo, hi = 2, -1e9, 1e9
        elif len(parts) == 4:
            g, lo, hi = 2, float(parts[2]), float(parts[3])
        elif len(parts) == 5:
            g, lo, hi = "XYZ".index(parts[2].upper()), float(parts[3]), float(parts[4])
        else:
            raise SystemExit(f"รูปแบบ --seams ไม่ถูกต้อง: {spec}")
        out.append((axis, val, g, lo, hi))
    return out


def smoothstep_falloff(d: float) -> float:
    """1 ที่ d=0 จางเป็น 0 ที่ d>=1"""
    return 0.0 if d >= 1.0 else 1.0 - (3 * d * d - 2 * d * d * d)


def seam_weights(obj: bpy.types.Object, seams, band: float, name: str,
                 invert: bool = False) -> bpy.types.VertexGroup:
    """น้ำหนัก 1 ที่แนวตัด จางเป็น 0 ที่ระยะ band; แนวตัดที่จำกัดช่วง Z จะจางออก
    นอกช่วงนั้นด้วย (กันไม่ให้ไปเกลาส่วนอื่นที่บังเอิญอยู่ระนาบเดียวกัน เช่นปลายหาง)
    invert=True กลับด้าน ใช้คุม shrinkwrap"""
    vg = obj.vertex_groups.new(name=name)
    for v in obj.data.vertices:
        w = 0.0
        for axis, val, g, lo, hi in seams:
            d = abs(v.co[axis] - val) / band
            q = v.co[g]
            dq = 0.0 if lo <= q <= hi else min(abs(q - lo), abs(q - hi)) / band
            w = max(w, smoothstep_falloff(d) * smoothstep_falloff(dq))
        if invert:
            w = 1.0 - w
        if w > 0.0:
            vg.add([v.index], w, "REPLACE")
    return vg


def remesh_and_restore(obj, src, a, t0) -> None:
    """สำหรับ mesh ที่ยังไม่ปิด: อุดรู -> voxel remesh -> shrinkwrap คืนรายละเอียด"""
    before = count_open_edges(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.fill_holes(sides=0)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    after = count_open_edges(obj)
    log(f"อุดรู: ขอบเปิด {before} -> {after}", t0)
    if after:
        log("คำเตือน: ยังมีขอบเปิด voxel remesh อาจทิ้งบางส่วนของ mesh — "
            "ควรรวมชิ้นด้วย boolean ให้ปิดก่อน", t0)

    m = obj.modifiers.new("remesh", "REMESH")
    m.mode = "VOXEL"
    m.voxel_size = a.voxel
    m.adaptivity = 0.0
    apply_modifier(obj, m)
    log(f"voxel remesh {a.voxel}: {len(obj.data.polygons):,} faces", t0)

    keep = seam_weights(obj, parse_seams(a), a.band * 1.5, "detail", invert=True)
    m = obj.modifiers.new("shrink", "SHRINKWRAP")
    m.target = src
    m.wrap_method = "NEAREST_SURFACEPOINT"
    m.vertex_group = keep.name
    apply_modifier(obj, m)
    log("shrinkwrap คืนรายละเอียดนอกแถบรอยต่อแล้ว", t0)


def main() -> int:
    a = args()
    t0 = time.time()
    clear_scene()

    src = import_mesh(a.src)
    src.name = "croc_source"
    log(f"นำเข้า {os.path.basename(a.src)}: {len(src.data.polygons):,} faces, "
        f"ขอบเปิด {count_open_edges(src)}", t0)

    # ทำงานบนสำเนา ต้นฉบับเก็บไว้เทียบ/เป็นเป้า shrinkwrap
    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    bpy.ops.object.duplicate()
    obj = bpy.context.view_layer.objects.active
    obj.name = "croc_merged"
    src.hide_set(True)
    src.hide_render = True

    need = a.remesh == "always" or (a.remesh == "auto" and count_open_edges(obj) > 0)
    if need:
        remesh_and_restore(obj, src, a, t0)
    else:
        log("mesh ปิดสนิทแล้ว ข้าม remesh เพื่อเก็บรายละเอียดเดิมไว้", t0)

    band = seam_weights(obj, parse_seams(a), a.band, "seam")
    # Smooth แบบดั้งเดิม (เฉลี่ยเพื่อนบ้านน้ำหนักเท่ากัน) ลบขั้นบันไดได้แรงกว่า Laplacian
    # ที่ถ่วงตามความยาวขอบ — ฝั่งหัวมี mesh ถี่ ฝั่งตัวห่าง Laplacian จึงขยับฝั่งหัวน้อยมาก
    m = obj.modifiers.new("seam_smooth", "SMOOTH")
    m.factor = a.strength
    m.iterations = a.smooth
    m.vertex_group = band.name
    apply_modifier(obj, m)
    log(f"smooth แถบรอยต่อ {parse_seams(a)} ±{a.band} {a.smooth} รอบ แรง {a.strength}", t0)

    n = len(obj.data.polygons)
    if a.faces and n > a.faces:
        m = obj.modifiers.new("decimate", "DECIMATE")
        m.decimate_type = "COLLAPSE"
        m.ratio = a.faces / n
        m.use_collapse_triangulate = True
        apply_modifier(obj, m)
        log(f"decimate {n:,} -> {len(obj.data.polygons):,} faces", t0)

    log(f"ผลลัพธ์: {len(obj.data.polygons):,} faces, ขอบเปิด {count_open_edges(obj)}", t0)
    bpy.ops.object.shade_smooth()

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=True,
                              export_apply=True)
    blend = os.path.splitext(a.out)[0] + ".blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    log(f"บันทึก {a.out}", t0)
    log(f"บันทึก {blend}", t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
