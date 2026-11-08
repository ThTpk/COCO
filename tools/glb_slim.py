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


def shrink_images(quality: int, max_size: int) -> None:
    for img in bpy.data.images:
        if img.source != "FILE" and not img.packed_file and img.size[0] == 0:
            continue
        if img.size[0] == 0:
            continue
        w, h = img.size
        if max_size and max(w, h) > max_size:
            img.scale(min(w, max_size), min(h, max_size))
            say(f"ย่อ {img.name} {w}x{h} -> {img.size[0]}x{img.size[1]}")
        # ตั้ง file_format ไว้ล่วงหน้า เพราะตัว export โหมด AUTO ดูค่านี้ว่าจะ
        # เขียนออกเป็นอะไร ส่วน alpha ไม่ต้องห่วง — albedo ชุดนี้ทึบทั้งใบ
        img.file_format = "JPEG"


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--faces", type=int, default=0, help="จำนวนหน้าเป้าหมาย 0 = ไม่ลด")
    ap.add_argument("--draco", action="store_true")
    ap.add_argument("--draco-level", type=int, default=6)
    ap.add_argument("--jpeg", type=int, default=0, help="คุณภาพ JPEG 0 = คง PNG ไว้")
    ap.add_argument("--tex-size", type=int, default=0, help="ย่อ texture ลงเหลือกี่พิกเซล")
    args = ap.parse_args(argv)

    meshes = load(args.src)
    before = args.src.stat().st_size
    for obj in meshes:
        say(f"{obj.name}: {len(obj.data.vertices)} จุด {len(obj.data.polygons)} หน้า")
        decimate(obj, args.faces)

    if args.jpeg or args.tex_size:
        shrink_images(args.jpeg or 90, args.tex_size)

    kw = {}
    if args.draco:
        kw.update(export_draco_mesh_compression_enable=True,
                  export_draco_mesh_compression_level=args.draco_level)
    if args.jpeg:
        kw.update(export_image_format="JPEG", export_jpeg_quality=args.jpeg)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(args.dst), export_format="GLB",
                              use_selection=False, export_yup=True,
                              export_skins=True, export_animations=True,
                              export_animation_mode="ACTIONS", **kw)
    after = args.dst.stat().st_size
    say(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB ({100 * after / before:.0f}%)")


if __name__ == "__main__":
    main()
