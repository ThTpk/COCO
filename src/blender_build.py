r"""สคริปต์ที่รันใน interpreter ของ Blender — ขั้น "ปั้น" ของโมเดลที่ 4

    blender --background --factory-startup --python blender_build.py -- \
        --stage cage --mesh output\character4_hull.glb --out output\character4_cage.glb

    blender --background --factory-startup --python blender_build.py -- \
        --stage finish --mesh output\character4_cage_uv.glb \
        --height output\character4_height.png --color output\character4_texture.png \
        --normal output\character4_normal.png --rough output\character4_rough.png \
        --out output\character4.glb --blend output\character4.blend

สองขั้นนี้คือส่วนที่ Blender ทำแทนโค้ด numpy ของโมเดลก่อนหน้า

``cage``
    ``marching cubes`` ให้สามเหลี่ยมที่ขนาดไม่เท่ากันเลย ตรงที่ผิวชันจะมี
    สามเหลี่ยมยาวเรียวพาดอยู่ ซึ่งเป็นโทโพโลยีที่แย่ที่สุดสำหรับการดันผิว —
    ดันเท่ากันทุกจุดแล้วสามเหลี่ยมเรียวจะฉีกก่อนเพื่อน ขั้นนี้ remesh ใหม่
    ด้วย voxel ของ Blender ให้ขอบยาวเท่ากันทั้งตัว แล้ว shrinkwrap กลับไป
    เกาะผิวเดิมเพื่อไม่ให้เส้นขอบที่เฉือนมาอย่างยากลำบากหายไปกับ remesh

``finish``
    ซอยผิวแล้วดันตามแผนที่ความสูงที่ bake มาจากชั้นดินเหนียว จากนั้นต่อวัสดุ
    PBR แล้ว export

**กับดักแกน V**: ``src/export.py`` พลิกภาพแนวตั้งตอนฝังลง GLB ตามแบบแผน
ของ glTF (นับ V จากขอบบน) ส่วนไฟล์ PNG บนดิสก์ยังเป็นด้านตั้งเดิมของ atlas
สคริปต์นี้จึง **พลิกพิกเซลของภาพตอนโหลด** (``load_map``) แทนการแทรก Mapping
node — เพราะ Mapping node จะกลายเป็น ``KHR_texture_transform`` ตอน export
ซึ่งผู้อ่านปลายทางไม่จำเป็นต้องรองรับ ส่วนการพลิกพิกเซลได้ภาพที่ตรงกับที่
``export.py`` ฝังไว้เป๊ะ ๆ แล้วตัว export ของ Blender ก็จัดการแกน UV ต่อเอง
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def say(msg: str) -> None:
    """พิมพ์ด้วย prefix ที่ฝั่ง host กรองเก็บไว้แสดง"""
    print(f"[build] {msg}", flush=True)


# --------------------------------------------------------------------------
# พื้นฐาน
# --------------------------------------------------------------------------
def wipe_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path: Path) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [o for o in bpy.data.objects
              if o not in before and o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"ไม่พบ mesh ใน {path}")

    # รวมทุกชิ้นเป็นก้อนเดียว ขั้น remesh กับ displace ทำงานทีละ object
    obj = meshes[0]
    if len(meshes) > 1:
        ctx = bpy.context
        for o in bpy.data.objects:
            o.select_set(o in meshes)
        ctx.view_layer.objects.active = obj
        bpy.ops.object.join()

    # ตัว import glTF ห่อ mesh ไว้ใต้ node ที่หมุนแกน Y-up เป็น Z-up และอาจมี
    # scale ติดมาด้วย ถ้าไม่ยุบ transform เข้าไปในพิกัดจุดก่อน ค่าที่คิดเป็น
    # "หน่วยเมตร" ทุกตัวหลังจากนี้ (ขนาด voxel ของ remesh, ระยะดันผิว) จะคิด
    # ในพิกัด local ที่ยังไม่ได้คูณ scale แล้วผิดไปทั้งใบโดยไม่มีใครรู้
    activate(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    say(f"อ่าน {path.name}: {len(obj.data.vertices):,} จุด / "
        f"{len(obj.data.polygons):,} หน้า")
    return obj


def activate(obj: bpy.types.Object) -> None:
    for o in bpy.data.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_modifier(obj: bpy.types.Object, name: str) -> None:
    activate(obj)
    bpy.ops.object.modifier_apply(modifier=name)


def tri_count(obj: bpy.types.Object) -> int:
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)


def size_of(obj: bpy.types.Object) -> float:
    """ความสูงของกล่องครอบ (หน่วยเดียวกับฉาก = เมตร)"""
    return max(obj.dimensions)


# --------------------------------------------------------------------------
# ขั้น cage: remesh ให้ขอบยาวเท่ากันทั้งตัว
# --------------------------------------------------------------------------
def stage_cage(args: argparse.Namespace) -> None:
    obj = import_glb(Path(args.mesh))
    height = size_of(obj)
    say(f"ความสูงกล่องครอบ {height:.3f} m")

    # ตัวต้นฉบับไว้เป็นเป้าของ shrinkwrap — ต้องเก็บก่อน remesh จะทับของเดิม
    ref = obj.copy()
    ref.data = obj.data.copy()
    ref.name = "hull_ref"
    bpy.context.collection.objects.link(ref)
    ref.hide_set(True)

    voxel = height / max(32, args.voxel_res)
    obj.data.remesh_voxel_size = voxel
    obj.data.remesh_voxel_adaptivity = 0.0
    activate(obj)
    bpy.ops.object.voxel_remesh()
    say(f"voxel remesh ที่ {voxel * 1000:.1f} mm -> {tri_count(obj):,} หน้า")

    if args.shrink:
        # remesh เดินผิวใหม่ทั้งใบ ผิวที่ได้จึงเลื่อนออกจากผิวเดิมราวครึ่ง
        # voxel ทุกจุด — ซึ่งแปลว่าเส้นขอบที่เฉือนมาจากเงาจะอ้วนขึ้นเท่านั้น
        # ทุกมุม  shrinkwrap ดึงกลับไปเกาะผิวเดิมโดยไม่เสียโทโพโลยีที่เพิ่ง
        # จัดใหม่
        m = obj.modifiers.new("snap", "SHRINKWRAP")
        m.target = ref
        m.wrap_method = "NEAREST_SURFACEPOINT"
        m.offset = 0.0
        apply_modifier(obj, m.name)
        say("shrinkwrap กลับไปเกาะผิว hull เดิม")

    if args.smooth > 0:
        m = obj.modifiers.new("relax", "SMOOTH")
        m.factor = 0.5
        m.iterations = args.smooth
        apply_modifier(obj, m.name)
        say(f"เกลาผิว {args.smooth} รอบ")

    if args.target_tris > 0 and tri_count(obj) > args.target_tris:
        m = obj.modifiers.new("dec", "DECIMATE")
        m.decimate_type = "COLLAPSE"
        m.ratio = args.target_tris / tri_count(obj)
        apply_modifier(obj, m.name)
        say(f"ลดหน้าเหลือ {tri_count(obj):,}")

    bpy.data.objects.remove(ref, do_unlink=True)
    export_glb(Path(args.out))


# --------------------------------------------------------------------------
# ขั้น finish: ซอย + ดันผิว + ต่อวัสดุ
# --------------------------------------------------------------------------
def load_map(path: Path, non_color: bool) -> bpy.types.Image:
    """โหลดแผนที่จากดิสก์ แล้วพลิกแนวตั้งให้ตรงกับที่ GLB ฝังไว้

    ดูเหตุผลเต็ม ๆ ที่หัวไฟล์ — สรุปคือ atlas บนดิสก์เป็นด้านตั้งตรงข้ามกับ
    ที่ glTF ใช้ ถ้าไม่พลิก แต่ละชิ้นส่วนจะไปหยิบสีของชิ้นอื่น
    """
    img = bpy.data.images.load(str(path), check_existing=False)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    w, h = img.size
    buf = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(buf)
    buf = np.ascontiguousarray(buf.reshape(h, w, 4)[::-1]).ravel()
    img.pixels.foreach_set(buf)
    img.pack()
    say(f"แผนที่ {path.name}: {w}x{h} (พลิกแกน V แล้ว)")
    return img


def build_material(name: str, color: Path | None, normal: Path | None,
                   rough: Path | None, strength: float,
                   rough_floor: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")

    def tex(path: Path, y: int, non_color: bool):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = load_map(path, non_color)
        node.location = (-800, y)
        return node

    if color is not None:
        nt.links.new(tex(color, 300, False).outputs["Color"],
                     bsdf.inputs["Base Color"])
    if rough is not None:
        n = tex(rough, 0, True)
        # แผนที่ roughness เก็บค่าไว้ที่ช่องเขียวตามสเปก glTF
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-520, 0)
        nt.links.new(n.outputs["Color"], sep.inputs["Color"])
        rmap = nt.nodes.new("ShaderNodeMapRange")
        rmap.location = (-330, 0)
        rmap.inputs["To Min"].default_value = rough_floor
        rmap.inputs["To Max"].default_value = 1.0
        nt.links.new(sep.outputs["Green"], rmap.inputs["Value"])
        nt.links.new(rmap.outputs["Result"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.55
    if normal is not None:
        n = tex(normal, -300, True)
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.location = (-520, -300)
        nmap.inputs["Strength"].default_value = strength
        nt.links.new(n.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    for key, val in (("Metallic", 0.0), ("IOR", 1.45)):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = val
    return mat


def stage_finish(args: argparse.Namespace) -> None:
    obj = import_glb(Path(args.mesh))
    if not obj.data.uv_layers:
        raise SystemExit("mesh ไม่มี UV — ขั้นนี้ต้องใช้ mesh ที่คลี่ผิวแล้ว")
    uv_name = obj.data.uv_layers[0].name
    height_m = size_of(obj)

    if args.subdiv > 0:
        # SIMPLE ไม่ใช่ CATMULL_CLARK: ที่นี่ต้องการ *จุดเพิ่ม* ไว้รองรับการดัน
        # ผิว ไม่ใช่ผิวที่เรียบขึ้น  Catmull-Clark จะหดเส้นขอบเข้าหาด้านใน
        # ทุกรอบ แล้วนิ้วมือกับปลายหางที่อุตส่าห์เฉือนมาจะเรียวหายไปเอง
        m = obj.modifiers.new("subdiv", "SUBSURF")
        m.subdivision_type = "SIMPLE"
        m.levels = m.render_levels = args.subdiv
        apply_modifier(obj, m.name)
        say(f"ซอยผิว {args.subdiv} ชั้น -> {tri_count(obj):,} หน้า")

    if args.height:
        tex = bpy.data.textures.new("depth", type="IMAGE")
        tex.image = load_map(Path(args.height), non_color=True)
        tex.extension = "EXTEND"
        m = obj.modifiers.new("depth", "DISPLACE")
        m.texture = tex
        m.texture_coords = "UV"
        m.uv_layer = uv_name
        m.direction = "NORMAL"
        m.mid_level = 0.5
        # แผนที่เก็บ -1..1 ไว้ในช่วง 0..1 (0.5 = ผิวเดิม) ระยะจริงจึงเป็น
        # สองเท่าของครึ่งช่วง
        m.strength = 2.0 * args.amp * height_m
        apply_modifier(obj, m.name)
        say(f"ดันผิวตามแผนที่ความลึก +-{args.amp * height_m * 1000:.1f} mm")

    if args.polish > 0:
        # ดันผิวทีละจุดตามแผนที่ย่อมทิ้งเสียงรบกวนระดับ texel ไว้ เกลาเบา ๆ
        # หนึ่งรอบลบมันโดยที่ลายที่กว้างกว่าสามสี่ texel ยังอยู่ครบ
        m = obj.modifiers.new("polish", "SMOOTH")
        m.factor = 0.35
        m.iterations = args.polish
        apply_modifier(obj, m.name)
        say(f"เกลาหลังดันผิว {args.polish} รอบ")

    if args.target_tris > 0 and tri_count(obj) > args.target_tris:
        m = obj.modifiers.new("dec", "DECIMATE")
        m.decimate_type = "COLLAPSE"
        m.use_collapse_triangulate = True
        m.ratio = args.target_tris / tri_count(obj)
        apply_modifier(obj, m.name)
        say(f"ลดหน้าเหลือ {tri_count(obj):,}")

    activate(obj)
    bpy.ops.object.shade_smooth()
    obj.data.materials.clear()
    obj.data.materials.append(build_material(
        "character4",
        Path(args.color) if args.color else None,
        Path(args.normal) if args.normal else None,
        Path(args.rough) if args.rough else None,
        args.normal_strength, args.rough_floor))

    export_glb(Path(args.out))
    if args.blend:
        path = Path(args.blend)
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        say(f"เขียนไฟล์ฉาก -> {path}")


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def export_glb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = dict(filepath=str(path), export_format="GLB")
    # ชื่อพารามิเตอร์ของตัว export glTF เปลี่ยนไปมาระหว่าง Blender แต่ละรุ่น
    # จึงไล่จากครบที่สุดไปน้อยที่สุด
    for extra in ({"export_animations": False, "export_apply": False,
                   "export_yup": True}, {"export_animations": False}, {}):
        try:
            bpy.ops.export_scene.gltf(**base, **extra)
            say(f"export -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
            return
        except TypeError as exc:
            say(f"export ด้วย {sorted(extra)} ไม่ได้: {exc}")
    raise SystemExit("export GLB ไม่สำเร็จ")


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="blender_build")
    p.add_argument("--stage", choices=["cage", "finish"], required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    # cage
    p.add_argument("--voxel-res", type=int, default=240,
                   help="จำนวน voxel ตามความสูงตัวละคร ตอน remesh")
    p.add_argument("--shrink", type=int, default=1)
    p.add_argument("--smooth", type=int, default=4)
    # finish
    p.add_argument("--subdiv", type=int, default=2)
    p.add_argument("--height", default=None)
    p.add_argument("--amp", type=float, default=0.004,
                   help="ความลึกของการดันผิว เป็นสัดส่วนของความสูงตัวละคร")
    p.add_argument("--polish", type=int, default=1)
    p.add_argument("--color", default=None)
    p.add_argument("--normal", default=None)
    p.add_argument("--rough", default=None)
    p.add_argument("--normal-strength", type=float, default=1.0)
    p.add_argument("--rough-floor", type=float, default=0.18)
    p.add_argument("--blend", default=None)
    # ทั้งสองขั้น
    p.add_argument("--target-tris", type=int, default=0)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    wipe_scene()
    if args.stage == "cage":
        stage_cage(args)
    else:
        stage_finish(args)
    say("เสร็จสมบูรณ์")


if __name__ == "__main__":
    main()
