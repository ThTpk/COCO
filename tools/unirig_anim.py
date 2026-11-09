"""ใส่คลิปอนิเมชันให้ .glb ที่มีโครงกระดูกติดมาแล้ว — รันใน Blender เท่านั้น

    blender --background --factory-startup --python tools/unirig_anim.py -- \
        --src output/croc_photo_unirig_textured.glb \
        --dst output/croc_photo_unirig_rigged.glb \
        --merge 1e-6 --smooth --draco --jpeg 90

ต่างจาก tools/autorig.py ตรงที่ไม่ได้สร้างกระดูกเอง ไฟล์ที่เข้ามามีกระดูกจาก
UniRig อยู่แล้ว ขาดแค่คลิป งานที่นี่จึงเหลือแค่ "บอกให้ถูกว่ากระดูกไหนคืออะไร"

ปัญหาเดียวที่ต้องแก้คือชื่อ UniRig ตั้งชื่อกระดูกว่า bone_0..bone_39 ตามลำดับ
ที่มันไล่โครงออกมา ไม่ได้บอกว่าอันไหนคือต้นขาอันไหนคือข้อศอก ส่วนชุดคลิปใน
autorig.clips() เขียนด้วยชื่อแบบ Mixamo (LeftUpLeg, RightForeArm, ...) ไฟล์นี้
จึงเป็นตารางแปลชื่อระหว่างสองฝั่ง อ่านจากตำแหน่งกระดูกในไฟล์จริง

ระบบแกนตรงกับที่ autorig สมมติไว้พอดี ไม่ต้องกลับเครื่องหมายใด ๆ: ใน Blender
ตัวละครหันหน้าไป -Y, +Z ขึ้น, +X คือด้านซ้ายของตัวละคร ตรวจจากไฟล์ได้สามทาง
คือปลายเท้าอยู่หน้าข้อเท้า (y -0.22 -> -0.46), โคนหางอยู่หลังสะโพก (y +0.14)
และขาซ้ายอยู่ฝั่ง +X
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

import autorig          # noqa: E402  ชุดคลิปกลาง ใช้ร่วมกับโมเดลที่ rig เอง
import blender_rig      # noqa: E402  ตัวอบ keyframe + export
import glb_slim         # noqa: E402  weld / smooth / บีบ texture

# ตารางแปลชื่อ อ่านจากตำแหน่งหัวกระดูกใน croc_photo_unirig_textured.glb
# (พิกัดในวงเล็บคือพิกัดแบบ Blender: +X ซ้ายของตัวละคร, -Y หน้า, +Z ขึ้น)
NAME_MAP: dict[str, str] = {
    # แกนกลาง: bone_0 อยู่ระดับสะโพก (z -0.07) ไล่ขึ้นไปจบที่หัว (z +0.69)
    "Hips": "bone_0",
    "Spine": "bone_1",
    "Spine1": "bone_2",
    "Spine2": "bone_3",          # ระดับอก เป็นที่ที่แขนสองข้างแตกออกไป
    "Neck": "bone_4",
    "Head": "bone_5",
    # แขนซ้าย: ไหล่ -> ต้นแขน -> ปลายแขน -> มือ (มือมีสามนิ้วแตกต่อไป)
    "LeftShoulder": "bone_6",
    "LeftArm": "bone_7",
    "LeftForeArm": "bone_8",
    "LeftHand": "bone_9",
    "RightShoulder": "bone_19",
    "RightArm": "bone_20",
    "RightForeArm": "bone_21",
    "RightHand": "bone_22",
    # ขาซ้าย: สะโพก (z -0.13) -> เข่า (-0.50) -> ข้อเท้า (-0.82) -> ปลายเท้า
    "LeftUpLeg": "bone_32",
    "LeftLeg": "bone_33",
    "LeftFoot": "bone_34",
    "LeftToeBase": "bone_35",
    "RightUpLeg": "bone_36",
    "RightLeg": "bone_37",
    "RightFoot": "bone_38",
    "RightToeBase": "bone_39",
}

# หางของ UniRig นับจาก 0 ส่วน autorig.tail_wave ออกชื่อ Tail1..TailN
N_TAIL = 6
NAME_MAP.update({f"Tail{i + 1}": f"tail_{i}" for i in range(N_TAIL)})


# กระดูกขาของ UniRig วางยาวกว่าและชี้คนละมุมกับโครงที่ autorig สร้างเอง มุม
# เท่ากันจึงกวาดปลายเท้าได้ไกลกว่า วัดระยะที่ปลายเท้ากวาดจริงจากไฟล์: สเกล 1.0
# ได้ก้าวยาว 56% ของความสูงตัว ขณะที่โมเดลที่ 6 ซึ่งเป็น mesh ก้อนเดียวกันก้าว
# 36% ขายาวเกินจนดูเป็นการพุ่งมากกว่าเดิน ค่า 0.55 ดึงกลับมาอยู่ที่ 39% คือเท่า
# กับโมเดลที่ 6 หน้าเว็บจะได้เทียบกันได้ว่าสองตัวต่างกันที่วิธีใส่กระดูก
# ไม่ใช่ต่างกันที่จังหวะเดิน
# แตะเฉพาะต้นขา เพราะต้นขาเป็นตัวกำหนดว่าก้าวไกลแค่ไหน ส่วนเข่ากำหนดว่า
# ยกเท้าสูงแค่ไหน ถ้าหารลงทั้งขาเท้าจะไถไปกับพื้นยิ่งกว่าเดิม
SWING_BONES = {"LeftUpLeg", "RightUpLeg"}


def say(msg: str) -> None:
    print(f"[anim] {msg}", flush=True)


def armature() -> object:
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise SystemExit(f"ต้องมี armature ตัวเดียวในไฟล์ แต่เจอ {len(arms)}")
    return arms[0]


def translate(payload: dict, bones: set[str], leg_scale: float) -> dict:
    """เปลี่ยนชื่อกระดูกในชุดคลิปจากแบบ Mixamo เป็นชื่อจริงในไฟล์

    channel ที่หาชื่อปลายทางไม่เจอจะถูกตัดทิ้งตรงนี้ แทนที่จะปล่อยให้ไปเงียบ
    อยู่ใน apply_clips เพราะอยากให้เห็นตั้งแต่ตอนรันว่าแปลไม่ครบตรงไหน
    """
    dropped: set[str] = set()
    for clip in payload.get("clips", []):
        kept = []
        for ch in clip.get("channels", []):
            name = NAME_MAP.get(ch["bone"])
            if name is None or name not in bones:
                dropped.add(ch["bone"])
                continue
            out = {**ch, "bone": name}
            if leg_scale != 1.0 and ch["bone"] in SWING_BONES:
                out["amp"] = ch.get("amp", 0.0) * leg_scale
                out["offset"] = ch.get("offset", 0.0) * leg_scale
            kept.append(out)
        clip["channels"] = kept
        say(f"clip {clip['name']}: {len(kept)} channel")
    if dropped:
        say(f"ข้ามกระดูกที่แปลไม่ได้: {sorted(dropped)}")
    return payload


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--merge", type=float, default=0.0)
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--draco", action="store_true")
    ap.add_argument("--jpeg", type=int, default=0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--leg-scale", type=float, default=0.55,
                    help="ตัวคูณช่วงการแกว่งต้นขา ให้ก้าวเท่าโมเดลที่ 6")
    args = ap.parse_args(argv)

    meshes = glb_slim.load(args.src)
    before = args.src.stat().st_size
    for obj in meshes:
        glb_slim.weld(obj, args.merge)
        if args.smooth:
            glb_slim.smooth(obj)

    arm = armature()
    bones = {b.name for b in arm.data.bones}
    say(f"armature {arm.name}: {len(bones)} กระดูก")

    payload = translate(autorig.clips(N_TAIL, fps=args.fps), bones, args.leg_scale)
    say(f"ตัวคูณช่วงต้นขา {args.leg_scale}")
    made = blender_rig.apply_clips(arm, payload)
    say(f"สร้างอนิเมชัน {len(made)} clip: {', '.join(made)}")

    if args.jpeg:
        glb_slim.shrink_images(args.jpeg, 0, 0, 0)

    blender_rig.export_glb(args.dst, args.draco)
    after = args.dst.stat().st_size
    say(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB ({100 * after / before:.0f}%)")


if __name__ == "__main__":
    main()
