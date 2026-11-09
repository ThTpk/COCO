"""เปลี่ยน texture สีของ .glb ที่ rig แล้ว โดยไม่แตะ mesh หรือคลิป — รันใน Blender

    blender --background --factory-startup --python tools/retexture.py -- \
        --src output/croc_photo_rigged.glb --albedo output/croc_photo_albedo_mb.png \
        --dst output/croc_photo_rigged.glb --draco --jpeg 90

อบสีใหม่ด้วย tools/mv_bake.py แล้วได้ atlas ใบใหม่ที่ดีกว่าเดิม แต่ไฟล์ที่หน้าเว็บ
ใช้จริงผ่านขั้นใส่กระดูกมาแล้ว จะย้อนไปอบใหม่ทั้งสายต้องรัน ComfyUI ซ้ำ
ซึ่งไม่มีในเครื่องนี้ ทางลัดที่ปลอดภัยคือสลับเฉพาะภาพ เพราะ atlas ใบใหม่อบด้วย
--keep-uv จึงวางอยู่บนผัง UV เดียวกับของเดิมเป๊ะ ทุกเท็กเซลยังตรงที่เดิม
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blender_rig  # noqa: E402
import glb_slim     # noqa: E402


def say(msg: str) -> None:
    print(f"[retex] {msg}", flush=True)


def swap(albedo: Path) -> int:
    """ชี้ทุกโหนด Image Texture ที่ป้อน Base Color ไปที่ภาพใบใหม่"""
    img = bpy.data.images.load(str(albedo))
    img.colorspace_settings.name = "sRGB"
    img.pack()
    n = 0
    for mat in bpy.data.materials:
        if not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            inp = node.inputs.get("Base Color")
            if inp is None or not inp.is_linked:
                continue
            src = inp.links[0].from_node
            if src.type == "TEX_IMAGE":
                say(f"{mat.name}: {getattr(src.image, 'name', None)} -> {albedo.name}")
                src.image = img
                n += 1
    return n


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--albedo", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--draco", action="store_true")
    ap.add_argument("--jpeg", type=int, default=0)
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.src))
    before = args.src.stat().st_size

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    acts = sorted(a.name for a in bpy.data.actions)
    say(f"เข้ามา: mesh {len(meshes)} armature {len(arms)} clip {acts}")

    if swap(args.albedo) == 0:
        raise SystemExit("ไม่เจอ Image Texture ที่ป้อน Base Color เลย")

    if args.jpeg:
        glb_slim.shrink_images(args.jpeg, 0, 0, 0)

    blender_rig.export_glb(args.dst, args.draco)
    after = args.dst.stat().st_size
    say(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
