"""ลดขนาดไฟล์ .glb ให้หน้าเว็บโหลดไหว — รันใน Blender เท่านั้น

    blender --background --factory-startup --python tools/glb_slim.py -- \
        --src output/croc_photo_rigged.glb --dst output/croc_photo_small.glb \
        --draco --jpeg 90

ไฟล์ที่ออกจาก pipeline นี้หนักอยู่สองที่คนละเรื่อง จึงมีสองคันโยก

  1. เนื้อ mesh — ตำแหน่ง/นอร์มอล/UV/weight ต่อจุดยอด บวก index
     ลดได้สองทาง: Draco บีบโดยไม่ทิ้งหน้าไหนเลย หรือ decimate ซึ่งทิ้งหน้าจริง
  2. texture — PNG 2048 ไม่บีบอะไรเลย ภาพถ่ายแบบนี้ JPEG เล็กกว่าหลายเท่า
     โดยตาแทบไม่เห็นต่าง (ไม่มี alpha ให้ต้องรักษา)

Draco ต้องโหลดตัวถอดรหัสจาก CDN ตอนเปิดหน้าเว็บ ถ้าไม่อยากพึ่ง ให้ใช้ --faces
แทน แล้วยอมเสียรายละเอียดบางส่วนไป
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import bpy


def say(msg: str) -> None:
    print(f"[slim] {msg}", flush=True)


def load(path: Path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"ไม่มี mesh ใน {path}")
    return meshes


def decimate(obj, target: int) -> None:
    faces = len(obj.data.polygons)
    if target <= 0 or target >= faces:
        return
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("slim", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.use_collapse_triangulate = True
    mod.ratio = target / faces
    bpy.ops.object.modifier_apply(modifier=mod.name)
    say(f"ลดหน้า {faces} -> {len(obj.data.polygons)}")


def trace_image(socket, depth: int = 6):
    """เดินย้อนจากขา input ไปหา Image Texture ที่ป้อนมัน"""
    if depth <= 0 or not socket.is_linked:
        return None
    node = socket.links[0].from_node
    if node.type == "TEX_IMAGE":
        return node.image
    for inp in node.inputs:
        found = trace_image(inp, depth - 1)
        if found is not None:
            return found
    return None


def color_images() -> set[str]:
    """ชื่อภาพที่ทำหน้าที่เป็น "สี" ไม่ใช่ข้อมูลเชิงตัวเลข

    normal map กับ roughness map เก็บตัวเลขไว้ในช่องสี ไม่ใช่ภาพที่ตาคนดู
    JPEG ซึ่งทิ้งรายละเอียดตามที่ตาคนไม่ค่อยเห็น จะทำให้ผิวเป็นคลื่นหรือด่าง
    เลยแตะเฉพาะ base color กับ emission
    """
    names = set()
    for mat in bpy.data.materials:
        if not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            for slot in ("Base Color", "Emission Color", "Emission"):
                inp = node.inputs.get(slot)
                if inp is None:
                    continue
                img = trace_image(inp)
                if img is not None:
                    names.add(img.name)
    return names


def shrink_images(quality: int, map_quality: int, max_size: int, map_size: int) -> None:
    colors = color_images()
    for img in bpy.data.images:
        if img.size[0] == 0:
            continue
        w, h = img.size
        # ไฟล์ที่บีบ mesh มาแล้วเหลือน้ำหนักอยู่ที่ normal map เป็นหลัก และ
        # Blender ตั้งคุณภาพ JPEG ได้ค่าเดียวทั้งไฟล์ตอน export จะลดเฉพาะ
        # แผนที่ให้ต่ำกว่าสีจึงทำไม่ได้ เหลือทางเดียวคือย่อขนาดมันลง
        limit = map_size if (map_size and img.name not in colors) else max_size
        if limit and max(w, h) > limit:
            img.scale(min(w, limit), min(h, limit))
            say(f"ย่อ {img.name} {w}x{h} -> {img.size[0]}x{img.size[1]}")
        if img.name in colors:
            if quality:
                recode_jpeg(img, quality)
        elif map_quality:
            # normal map เป็นตัวที่หนักที่สุดในไฟล์พวกนี้ และมักถูกเซฟมาแบบ
            # เกือบไม่บีบ ลดคุณภาพลงพอประมาณจึงได้เนื้อที่คืนเยอะ แต่ลดแรง
            # ไม่ได้ เพราะ artifact ของ JPEG บนแผนที่ตัวเลขจะโผล่เป็นคลื่นบนผิว
            recode_jpeg(img, map_quality)
        else:
            say(f"{img.name} ไม่ใช่ภาพสี คงรูปแบบเดิม ({img.file_format})")


def recode_jpeg(img, quality: int) -> None:
    """บีบภาพเป็น JPEG จริง ๆ แล้วฝังกลับเข้าไฟล์

    ตั้งแค่ img.file_format ไม่พอ — ตัว export โหมด AUTO ตัดสินจากข้อมูลที่
    ฝังมากับภาพ ไม่ใช่ค่าที่เราเพิ่งตั้ง ภาพที่เข้ามาเป็น PNG จึงออกไปเป็น PNG
    เหมือนเดิม ต้องเขียนเป็น JPEG ลงดิสก์แล้วโหลดกลับมา ให้ข้อมูลที่ฝังอยู่
    เป็น JPEG ตั้งแต่ก่อนถึงขั้น export
    """
    before = len(img.packed_file.data) if img.packed_file else 0
    bpy.context.scene.render.image_settings.file_format = "JPEG"
    bpy.context.scene.render.image_settings.quality = quality

    tmp = Path(tempfile.mkdtemp()) / f"{img.name}.jpg"
    img.file_format = "JPEG"
    img.filepath_raw = str(tmp)
    img.save(quality=quality)

    img.source = "FILE"
    img.filepath = str(tmp)
    img.reload()
    img.pack()
    after = len(img.packed_file.data) if img.packed_file else 0
    say(f"{img.name} -> JPEG q{quality} ({before / 1e6:.2f} -> {after / 1e6:.2f} MB)")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--faces", type=int, default=0, help="จำนวนหน้าเป้าหมาย 0 = ไม่ลด")
    ap.add_argument("--draco", action="store_true")
    ap.add_argument("--draco-level", type=int, default=6)
    ap.add_argument("--jpeg", type=int, default=0, help="คุณภาพ JPEG 0 = คง PNG ไว้")
    ap.add_argument("--map-jpeg", type=int, default=0,
                    help="คุณภาพ JPEG ของ normal/roughness map 0 = ไม่แตะ")
    ap.add_argument("--tex-size", type=int, default=0, help="ย่อ texture ลงเหลือกี่พิกเซล")
    ap.add_argument("--map-size", type=int, default=0,
                    help="ย่อเฉพาะ normal/roughness map ลงเหลือกี่พิกเซล")
    args = ap.parse_args(argv)

    meshes = load(args.src)
    before = args.src.stat().st_size
    for obj in meshes:
        say(f"{obj.name}: {len(obj.data.vertices)} จุด {len(obj.data.polygons)} หน้า")
        decimate(obj, args.faces)

    if args.jpeg or args.map_jpeg or args.tex_size or args.map_size:
        shrink_images(args.jpeg, args.map_jpeg, args.tex_size, args.map_size)

    kw = {}
    if args.draco:
        kw.update(export_draco_mesh_compression_enable=True,
                  export_draco_mesh_compression_level=args.draco_level)
    if args.jpeg or args.map_jpeg:
        kw.update(export_image_format="AUTO",
                  export_jpeg_quality=max(args.jpeg, args.map_jpeg))

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(args.dst), export_format="GLB",
                              use_selection=False, export_yup=True,
                              export_skins=True, export_animations=True,
                              export_animation_mode="ACTIONS", **kw)
    after = args.dst.stat().st_size
    say(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB ({100 * after / before:.0f}%)")


if __name__ == "__main__":
    main()
