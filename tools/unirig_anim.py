"""ใส่คลิปอนิเมชันให้ .glb ที่มีโครงกระดูกติดมาแล้ว — รันใน Blender เท่านั้น

    blender --background --factory-startup --python tools/unirig_anim.py -- \
        --src output/croc_photo_unirig_textured.glb \
        --dst output/croc_photo_unirig_rigged.glb \
        --merge 1e-6 --smooth --draco --jpeg 90

ต่างจาก tools/autorig.py ตรงที่ไม่ได้สร้างกระดูกเอง ไฟล์ที่เข้ามามีกระดูกจาก
UniRig อยู่แล้ว ขาดแค่คลิป งานที่นี่จึงเหลือแค่ "บอกให้ถูกว่ากระดูกไหนคืออะไร"

ปัญหาเดียวที่ต้องแก้คือชื่อ UniRig ตั้งชื่อกระดูกว่า bone_0..bone_N ตามลำดับ
ที่มันไล่โครงออกมา ไม่ได้บอกว่าอันไหนคือต้นขาอันไหนคือข้อศอก ส่วนชุดคลิปใน
autorig.clips() เขียนด้วยชื่อแบบ Mixamo (LeftUpLeg, RightForeArm, ...) ไฟล์นี้
จึงเป็นตารางแปลชื่อระหว่างสองฝั่ง สร้างขึ้นสด ๆ จากรูปทรงของโครงในไฟล์

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

# ลำดับเลขของ UniRig ไม่คงที่ข้ามการรัน รอบแรกได้ 47 ข้อต่อ มือข้างละสามนิ้ว
# ขาซ้ายเริ่มที่ bone_32 รอบที่สองบน mesh ที่ปรับสัดส่วนแล้วได้ 35 ข้อต่อ
# ขาซ้ายเลื่อนมาอยู่ที่ bone_20 ตารางที่แข็งค่าไว้จึงใช้ได้รอบเดียวแล้วพัง
# เงียบ ๆ (คลิปยังสร้างได้ แต่ไปหมุนกระดูกผิดตัว) ที่นี่เลยไล่โครงเอาเองทุกครั้ง
def build_map(arm) -> tuple[dict[str, str], int]:
    """อ่านโครงกระดูกจากรูปทรงของมันเอง แล้วคืนตารางแปลชื่อแบบ Mixamo

    อาศัยสามอย่างที่จริงไม่ว่า UniRig จะนับเลขอย่างไร: สะโพกคือรากที่มีลูก,
    อกคือกระดูกบนแกนกลางที่แตกออกสามทาง (คอกับแขนสองข้าง) และ +X คือข้างซ้าย
    ของตัวละคร ส่วนหางดูจากชื่อ เพราะ UniRig ตั้งชื่อ tail_* ไว้ให้แล้ว

    ถ้าโครงไม่เข้ารูปที่คาด จะหยุดพร้อมบอกว่าสะดุดตรงไหน ดีกว่าปล่อยให้แปลผิด
    แล้วได้คลิปที่ขยับกระดูกคนละอันกับที่ตั้งใจ
    """
    P = {b.name: arm.matrix_world @ b.head_local for b in arm.data.bones}
    kids = {b.name: [c.name for c in b.children] for b in arm.data.bones}
    body = lambda n: not n.startswith("tail_")

    tails = sorted((n for n in P if n.startswith("tail_")),
                   key=lambda n: int(n.rsplit("_", 1)[1]))

    roots = [b.name for b in arm.data.bones if b.parent is None and kids[b.name]]
    if len(roots) != 1:
        raise SystemExit(f"ต้องมีรากที่มีลูกอันเดียว แต่เจอ {roots}")
    hips = roots[0]

    def chain(start: str, n: int) -> list[str]:
        """ไล่ลงไปตามกิ่ง เลือกลูกที่อยู่ไกลที่สุดเมื่อมีให้เลือกหลายตัว"""
        out = [start]
        while len(out) < n:
            ks = [k for k in kids[out[-1]] if body(k)]
            if not ks:
                raise SystemExit(f"กิ่งจาก {start} สั้นกว่าที่ต้องการ ({out})")
            out.append(max(ks, key=lambda k: (P[k] - P[out[-1]]).length))
        return out

    up = [k for k in kids[hips] if body(k) and P[k].z > P[hips].z]
    if len(up) != 1:
        raise SystemExit(f"ต้องมีแกนกลางเส้นเดียวขึ้นจากสะโพก แต่เจอ {up}")
    spine, cur = [hips], up[0]
    while True:
        spine.append(cur)
        ks = [k for k in kids[cur] if body(k)]
        if len(ks) >= 3:          # ถึงอกแล้ว คอกับแขนสองข้างแตกจากที่นี่
            break
        if len(ks) != 1:
            raise SystemExit(f"แกนกลางขาดตอนที่ {cur} (ลูก {ks})")
        cur = ks[0]
    if len(spine) != 4:
        raise SystemExit(f"คาดว่าแกนกลางถึงอกมีสี่ข้อ แต่ได้ {spine}")

    branches = [k for k in kids[spine[-1]] if body(k)]
    neck = min(branches, key=lambda k: abs(P[k].x))     # คออยู่กลางลำตัว
    arms = [k for k in branches if k != neck]
    if len(arms) != 2:
        raise SystemExit(f"ที่อกต้องแตกเป็นคอกับแขนสองข้าง แต่เจอ {branches}")
    legs = [k for k in kids[hips] if body(k) and P[k].z < P[hips].z]
    if len(legs) != 2:
        raise SystemExit(f"ต้องมีขาสองข้างใต้สะโพก แต่เจอ {legs}")

    sides = lambda ns: (max(ns, key=lambda k: P[k].x), min(ns, key=lambda k: P[k].x))
    m = dict(zip(("Hips", "Spine", "Spine1", "Spine2"), spine))
    m["Neck"], m["Head"] = chain(neck, 2)
    for pre, start in zip(("Left", "Right"), sides(arms)):
        m.update(zip((f"{pre}Shoulder", f"{pre}Arm", f"{pre}ForeArm", f"{pre}Hand"),
                     chain(start, 4)))
    for pre, start in zip(("Left", "Right"), sides(legs)):
        m.update(zip((f"{pre}UpLeg", f"{pre}Leg", f"{pre}Foot", f"{pre}ToeBase"),
                     chain(start, 4)))
    # หางของ UniRig นับจาก 0 ส่วน autorig.tail_wave ออกชื่อ Tail1..TailN
    m.update({f"Tail{i + 1}": n for i, n in enumerate(tails)})

    # กันกรณีที่เงียบที่สุด: ซ้ายขวาสลับกันแล้วเดินถอยหลังโดยไม่มีใครสังเกต
    if not (P[m["LeftUpLeg"]].x > 0 > P[m["RightUpLeg"]].x):
        raise SystemExit("ขาซ้าย/ขวาสลับข้าง — +X ควรเป็นข้างซ้ายของตัวละคร")
    if P[m["LeftToeBase"]].y > P[m["LeftFoot"]].y:
        raise SystemExit("ปลายเท้าควรอยู่หน้าข้อเท้า (-Y คือทิศที่ตัวละครหัน)")
    return m, len(tails)


# กระดูกขาของ UniRig วางยาวกว่าและชี้คนละมุมกับโครงที่ autorig สร้างเอง มุม
# เท่ากันจึงกวาดปลายเท้าได้ไกลกว่า วัดระยะที่ปลายเท้ากวาดจริงจากไฟล์: สเกล 1.0
# ได้ก้าวยาว 56% ของความสูงตัว ขณะที่โมเดลที่ 6 ซึ่งเป็น mesh ก้อนเดียวกันก้าว
# 36% ขายาวเกินจนดูเป็นการพุ่งมากกว่าเดิน ค่า 0.55 ดึงกลับมาอยู่ที่ 39% คือเท่า
# กับโมเดลที่ 6 หน้าเว็บจะได้เทียบกันได้ว่าสองตัวต่างกันที่วิธีใส่กระดูก
# ไม่ใช่ต่างกันที่จังหวะเดิน
# แตะเฉพาะต้นขา เพราะต้นขาเป็นตัวกำหนดว่าก้าวไกลแค่ไหน ส่วนเข่ากำหนดว่า
# ยกเท้าสูงแค่ไหน ถ้าหารลงทั้งขาเท้าจะไถไปกับพื้นยิ่งกว่าเดิม
SWING_BONES = {"LeftUpLeg", "RightUpLeg"}

# เข่าเป็นตัวกำหนดว่ายกเท้าพ้นพื้นแค่ไหน แยกคันโยกไว้ต่างหากจากต้นขาเพราะ
# ทิศที่ UniRig วางเข่าไว้ทำให้มุมงอเท่ากันยกเท้าได้น้อยกว่าโครงของ autorig
KNEE_BONES = {"LeftLeg", "RightLeg"}

# ข้อเท้าเป็นตัวที่ทำให้ "ก้าว" ดูออก วัดเทียบกับโมเดลที่ 6: ที่นั่นปลายเท้า
# ยกขึ้น 0.298 ขณะที่ข้อเท้ายกแค่ 0.057 คือเท้าตะแคงขึ้นลงรับพื้น ส่วนของ
# UniRig ปลายเท้ายก 0.122 น้อยกว่าข้อเท้าเสียอีก เท้าจึงแข็งเป็นแผ่นเดียว
# ลากไปกับพื้น ตัวคูณนี้เปิดมุมข้อเท้าให้เท้าพลิกได้เหมือนกัน ค่า 4.0 ทำให้
# ปลายเท้ายก 0.297 เท่ากับ 0.298 ของโมเดลที่ 6 โดยความยาวก้าวแทบไม่ขยับ
# (0.652 -> 0.668) เพราะข้อเท้าหมุนอยู่กับที่ ไม่ได้พาขาไปข้างหน้า
FOOT_BONES = {"LeftFoot", "RightFoot"}


def say(msg: str) -> None:
    print(f"[anim] {msg}", flush=True)


def armature() -> object:
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise SystemExit(f"ต้องมี armature ตัวเดียวในไฟล์ แต่เจอ {len(arms)}")
    return arms[0]


def translate(payload: dict, name_map: dict[str, str], bones: set[str],
              leg_scale: float, knee_scale: float,
              foot_scale: float) -> dict:
    """เปลี่ยนชื่อกระดูกในชุดคลิปจากแบบ Mixamo เป็นชื่อจริงในไฟล์

    channel ที่หาชื่อปลายทางไม่เจอจะถูกตัดทิ้งตรงนี้ แทนที่จะปล่อยให้ไปเงียบ
    อยู่ใน apply_clips เพราะอยากให้เห็นตั้งแต่ตอนรันว่าแปลไม่ครบตรงไหน
    """
    dropped: set[str] = set()
    for clip in payload.get("clips", []):
        kept = []
        for ch in clip.get("channels", []):
            name = name_map.get(ch["bone"])
            if name is None or name not in bones:
                dropped.add(ch["bone"])
                continue
            out = {**ch, "bone": name}
            k = (leg_scale if ch["bone"] in SWING_BONES else
                 knee_scale if ch["bone"] in KNEE_BONES else
                 foot_scale if ch["bone"] in FOOT_BONES else 1.0)
            if k != 1.0:
                out["amp"] = ch.get("amp", 0.0) * k
                out["offset"] = ch.get("offset", 0.0) * k
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
    # normal map เก็บตัวเลขไว้ในช่องสี ไม่ใช่ภาพให้ตาคนดู กดคุณภาพแรงเท่าสีไม่ได้
    # เพราะ artifact จะโผล่เป็นคลื่นบนผิว จึงแยกคันโยกไว้คนละตัวกับ --jpeg
    ap.add_argument("--map-jpeg", type=int, default=0,
                    help="คุณภาพ JPEG ของ normal/roughness map 0 = คง PNG ไว้")
    ap.add_argument("--map-size", type=int, default=0,
                    help="ย่อ normal/roughness map ลงเหลือกี่พิกเซล 0 = ไม่ย่อ")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--leg-scale", type=float, default=0.55,
                    help="ตัวคูณช่วงการแกว่งต้นขา ให้ก้าวเท่าโมเดลที่ 6")
    ap.add_argument("--knee-scale", type=float, default=1.0,
                    help="ตัวคูณมุมงอเข่า")
    ap.add_argument("--foot-scale", type=float, default=4.0,
                    help="ตัวคูณมุมข้อเท้า ให้ปลายเท้ายกสูงเท่าโมเดลที่ 6")
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

    name_map, n_tail = build_map(arm)
    say(f"ไล่โครงได้ {len(name_map)} ข้อต่อ หาง {n_tail} ปล้อง: "
        + ", ".join(f"{k}={v}" for k, v in name_map.items()))

    payload = translate(autorig.clips(n_tail, fps=args.fps), name_map, bones,
                        args.leg_scale, args.knee_scale, args.foot_scale)
    say(f"ตัวคูณต้นขา {args.leg_scale} เข่า {args.knee_scale} ข้อเท้า {args.foot_scale}")
    made = blender_rig.apply_clips(arm, payload)
    say(f"สร้างอนิเมชัน {len(made)} clip: {', '.join(made)}")

    if args.jpeg or args.map_jpeg or args.map_size:
        glb_slim.shrink_images(args.jpeg, args.map_jpeg, 0, args.map_size)

    blender_rig.export_glb(args.dst, args.draco)
    after = args.dst.stat().st_size
    say(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB ({100 * after / before:.0f}%)")


if __name__ == "__main__":
    main()
