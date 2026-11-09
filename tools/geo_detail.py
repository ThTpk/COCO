"""สร้างลายผิวจากรูปทรงของ mesh แทนที่จะลอกมาจากภาพถ่าย — รันใน Blender เท่านั้น

    blender --background --factory-startup --python tools/geo_detail.py -- \
        --mesh output/croc_photo_textured_v1.glb \
        --albedo output/croc_photo_albedo_mb.png \
        --out output/croc_photo_albedo_geo.png --ao

ต่อยอดจาก multi-band ใน tools/mv_bake.py ตรงที่ยังแยกสองความถี่เหมือนเดิม แต่
เปลี่ยน "แหล่ง" ของความถี่สูง

  ความถี่ต่ำ (สีพื้น)   : จาก albedo ที่อบจากภาพถ่าย — ภาพถ่ายเท่านั้นที่รู้ว่า
                          เสื้อสีน้ำเงิน ยีนส์สีคราม หนังสีส้ม
  ความถี่สูง (ลายผิว)   : จากตัว mesh เอง ไม่ใช่จากภาพ

ทำไมถึงดีกว่า: ลายที่คำนวณจาก mesh ทาบกับผิว "ตรงเป๊ะโดยนิยาม" เพราะมันคือผิว
อันเดียวกัน จึงไม่มีวันเบลอจากการทาบพลาด ไม่มีรอยต่อระหว่างมุมกล้อง และไม่มี
เงาของวันที่ถ่ายติดมาด้วย ซึ่งคือสามปัญหาที่เหลืออยู่ของการฉายภาพ

สัญญาณที่ใช้สองตัว อ่านจาก Cycles

  pointiness  ความโค้งของผิว ร่องเป็นลบ สันเป็นบวก → ไล่เส้นเกล็ด รอยพับผ้า
              ขอบเข็มขัด ข้อโซ่ ออกมาได้ตรงตำแหน่งที่มันนูนจริง
  AO          ความอับแสง ซอกลึกยิ่งเข้ม → ให้มิติกับร่องที่ pointiness บอกแค่ว่าโค้ง

คูณเข้ากับสีพื้นแทนที่จะบวก เพราะการบวกทีละช่องสีจะดึง hue เพี้ยน (เคยเจอมาแล้ว
ตอนลองตัดยอดความถี่สูงใน mv_bake) ส่วนการคูณเป็นการปรับความสว่างล้วน สีไม่เปลี่ยน
และเงาบนยีนส์เข้มก็จะอ่อนลงตามสัดส่วนของมันเอง

ค่าปริยายคือ --photo-detail 1.0 คือ "เสริมนูน" ไม่ใช่ "แทนที่ลาย" บทเรียนจากตอน
ลองตั้ง 0: รูปทรงรู้แค่ว่าอะไรนูนอะไรเว้า มันไม่รู้เรื่องสี พอตัดลายจากภาพทิ้ง
สร้อยคอหายไปทั้งเส้น เพราะสร้อยเป็นของที่ "สว่างกว่าเสื้อ" ไม่ใช่ของที่ "นูนกว่าเสื้อ"
ในระดับที่ mesh ความละเอียดนี้จับได้ โลโก้บนเสื้อกับรอยเปื้อนก็หายไปด้วยเหตุผลเดียวกัน

ข้อจำกัด: ขอบสามเหลี่ยมยาวราว 6.8 mm เทียบตัวจริง mesh จึงรู้จักรอยพับผ้า เข็มขัด
สันเกล็ดใหญ่ และขอบกระเป๋าเสื้อ แต่ไม่รู้จักรูขุมขนหรือเกล็ดเม็ดเล็ก ลายละเอียด
กว่านั้นยังต้องพึ่งภาพถ่ายอยู่
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mv_bake  # noqa: E402


def say(msg: str) -> None:
    print(f"[geo] {msg}", flush=True)


def bake_signal(obj, size: int, kind: str, samples: int) -> np.ndarray:
    """เบคสัญญาณเชิงเรขาคณิตลงผัง UV แล้วคืนเป็น array (size, size)"""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.render.bake.margin = 16
    scene.render.bake.use_clear = True

    img = bpy.data.images.new(f"bake_{kind}", size, size, float_buffer=True)
    mat = bpy.data.materials.new(f"bake_{kind}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    if kind == "curv":
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        emit = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(geo.outputs["Pointiness"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        btype = "EMIT"
    else:
        diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
        nt.links.new(diff.outputs["BSDF"], out.inputs["Surface"])
        btype = "AO"

    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    nt.nodes.active = node

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    say(f"เบค {kind} ({btype}) ที่ {size}px samples={samples} …")
    bpy.ops.object.bake(type=btype)

    buf = np.empty(size * size * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    # Blender ไล่แถวจากล่างขึ้นบน ให้ตรงกับที่ mv_bake เขียนผัง UV ไว้
    return buf.reshape(size, size, 4)[::-1, :, 0].astype(np.float64)


def spread(x: np.ndarray, mask: np.ndarray, lo: float = 2.0,
           hi: float = 98.0) -> np.ndarray:
    """ยืดค่าให้เต็มช่วง 0..1 โดยตัดหางสองข้างทิ้ง กัน outlier ลากสเกล"""
    v = x[mask]
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    if b - a < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--albedo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=48,
                    help="รัศมีที่ใช้แยกความถี่ ควรเท่ากับ --lowband ของ mv_bake")
    ap.add_argument("--fine", type=int, default=3,
                    help="รัศมีเบลอตัวเล็ก กิน noise ของ mesh ทิ้ง 0 = ไม่กรอง")
    ap.add_argument("--strength", type=float, default=1.2,
                    help="ความแรงของลายจากรูปทรง")
    ap.add_argument("--photo-detail", type=float, default=1.0,
                    help="เก็บความถี่สูงจากภาพถ่ายไว้เท่าไร 0 = ใช้รูปทรงล้วน")
    ap.add_argument("--ao", action="store_true", help="ผสม ambient occlusion ด้วย")
    ap.add_argument("--ao-mix", type=float, default=0.5)
    ap.add_argument("--ao-samples", type=int, default=24)
    ap.add_argument("--maps", type=Path, default=None,
                    help="เขียนแผนที่เชิงเรขาคณิตออกมาดูด้วย")
    args = ap.parse_args(argv)

    obj = mv_bake.load_mesh(args.mesh, weld=False)
    mv_bake.unwrap(obj, 0.0, True)
    verts, normals, tris, tri_uv = mv_bake.mesh_arrays(obj)
    _pos, _nrm, hit = mv_bake.raster_uv(tri_uv, tris, verts, normals, args.size)
    say(f"ผัง UV ใช้จริง {100 * hit.mean():.1f}% ของพื้นที่")

    curv = spread(bake_signal(obj, args.size, "curv", 1), hit)
    geo = curv
    if args.ao:
        ao = spread(bake_signal(obj, args.size, "ao", args.ao_samples), hit)
        geo = (1.0 - args.ao_mix) * curv + args.ao_mix * ao
        say(f"ผสม curvature กับ AO ที่ {args.ao_mix}")

    # แถบกลาง ไม่ใช่ความถี่สูงล้วน — pointiness ไวต่อ noise ของ mesh มาก
    # จุดกระเด็นทีละเท็กเซลจาก mesh ที่สร้างจากภาพถ่ายจะโผล่เป็นเม็ดเกลือพริกไทย
    # ถ้าปล่อยไว้ ตัดหัวตัดหางด้วยการลบเบลอสองรัศมี (difference of blurs)
    # รัศมีเล็กกินเม็ด noise ทิ้ง รัศมีใหญ่กินสีพื้นทิ้ง เหลือแต่ช่วงที่เป็น
    # สันเกล็ดกับรอยพับจริง ซึ่งกว้างสิบกว่าเท็กเซล
    fine = (mv_bake.blur_masked(geo[..., None], hit, args.fine)[..., 0]
            if args.fine > 0 else geo)
    geo_hp = fine - mv_bake.blur_masked(geo[..., None], hit, args.radius)[..., 0]
    say(f"ลายจากรูปทรง: แกว่ง {geo_hp[hit].std():.4f} "
        f"ช่วง {geo_hp[hit].min():+.3f}..{geo_hp[hit].max():+.3f}")

    alb = mv_bake.srgb_to_linear(mv_bake.read_png(args.albedo)[..., :3])
    low = mv_bake.blur_masked(alb, hit, args.radius)
    det_photo = alb - low

    # คูณ ไม่ใช่บวก — ปรับความสว่างอย่างเดียว hue จึงไม่ขยับ
    mod = np.clip(1.0 + args.strength * geo_hp, 0.15, 3.0)[..., None]
    color = low * mod + args.photo_detail * det_photo
    color = np.clip(color, 0.0, None)

    rgba = np.concatenate([mv_bake.linear_to_srgb(color),
                           np.ones((args.size, args.size, 1))], axis=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mv_bake.write_png(args.out, rgba)
    say(f"เขียน {args.out}")

    if args.maps:
        g = np.repeat(geo[..., None], 3, axis=2)
        mv_bake.write_png(args.maps, np.concatenate(
            [g, np.ones((args.size, args.size, 1))], axis=2))
        say(f"เขียนแผนที่เรขาคณิต {args.maps}")


if __name__ == "__main__":
    main()
