"""ใส่ชุดคลิปท่าทางให้จระเข้ที่ rig ด้วย tools/rig_croc.py แล้ว — รันใน Blender เท่านั้น

    blender --background --factory-startup --python tools/croc_anim.py -- \
        --src output/rig/croc_rigged_apose_v1.blend --dst output/croc_kmitl_rigged.glb \
        --draco --jpeg 90 --tex-size 2048 --map-size 1024

ได้ห้าคลิป: Idle (ยืน), Walk (เดิน), Run (วิ่ง), Jump (กระโดด), TailSwing (แกว่งหาง) ส่วนปุ่ม
"ท่าพัก" ในหน้าเว็บไม่ใช่คลิป มันคือการหยุดเล่นแล้วปล่อยให้ตัวกลับไปที่ rest pose

ต่างจาก tools/unirig_anim.py ตรงที่ไม่ต้องเดาว่ากระดูกไหนคืออะไร โครงชุดนี้มาจาก
output/rig/croc_bones.json ซึ่งวางตำแหน่งไว้เองทีละข้อและตั้งชื่อแบบ Blender/Rigify
(hips, chest, upper_arm.L, thigh.R, tail.01..tail.08) คลิปจึงเขียนอ้างชื่อตรง ๆ ได้
และขยับของที่โครงของ UniRig ไม่มีมาแต่ต้นได้ด้วย — ขากรรไกรและปลายเท้า

ไฟล์ต้นทางที่ควรใช้คือรุ่นที่ผ่าน tools/apose_croc.py มาแล้ว ซึ่งย้าย rest pose จาก
ท่า T เป็นท่า A แขนที่กางออกระดับไหล่ทำให้ท่าเดินดูเหมือนคนเดินสะพายของ ไม่ใช่คนเดิน

ระบบแกนเดียวกับ tools/autorig.py: ใน Blender ตัวละครหันหน้าไป -Y, +Z ขึ้น และ +X คือ
ข้างซ้ายของตัวละคร ตรวจจากไฟล์กระดูกได้สามทาง คือปลายเท้าอยู่หน้าข้อเท้า, โคนหางอยู่
หลังสะโพก และกระดูกที่ลงท้าย .L อยู่ฝั่ง +X ทั้งหมด

ทุก channel คือคลื่นไซน์หนึ่งลูกบนกระดูกหนึ่งชิ้น (blender_rig._wave เป็นคนคิดค่า)
ท่าที่ไม่ใช่คลื่นปิดรอบ เช่น กระโดด ประกอบจาก halfsin สองลูกที่คนละครึ่งของลูป:
ครึ่งแรกย่อ ครึ่งหลังลอย
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blender_rig      # noqa: E402  ตัวอบ keyframe + export
import glb_slim         # noqa: E402  ย่อ/บีบ texture

N_TAIL = 8


def say(msg: str) -> None:
    print(f"[anim] {msg}", flush=True)


def ch(bone, axis="X", amp=0.0, phase=0.0, offset=0.0, harmonic=1.0,
       mode="sin", kind="rot") -> dict:
    return {"bone": bone, "axis": axis, "amp": amp, "phase": phase,
            "offset": offset, "harmonic": harmonic, "mode": mode, "kind": kind}


def tail_wave(axis="Z", first=4.0, last=16.0, step=0.09, phase0=0.0,
              harmonic=1.0, n=N_TAIL) -> list[dict]:
    """คลื่นวิ่งไล่จากโคนหางไปปลายหาง

    หัวใจคือ step: แต่ละปล้องเริ่มขยับช้ากว่าปล้องก่อนหน้าเล็กน้อย หางจึงสะบัด
    เป็นคลื่น ไม่ใช่แกว่งทั้งท่อนพร้อมกันเหมือนไม้แข็ง
    """
    return [ch(f"tail.{i + 1:02d}", axis,
               amp=first + (last - first) * i / max(n - 1, 1),
               phase=phase0 + step * i, harmonic=harmonic)
            for i in range(n)]


# ---------------------------------------------------------------------------
# ชุดท่า — ชื่อ clip คือข้อความบนปุ่มในหน้าเว็บ เพราะหน้าเว็บอ่านชื่อจากไฟล์จริง
# ---------------------------------------------------------------------------

def clips(fps: int = 24, leg: float = 1.0) -> dict:
    # ยืนเฉย ๆ: หายใจที่ลำตัว หัวเอียงช้ากว่าครึ่งหนึ่ง หางแกว่งเบา ๆ
    idle = [
        ch("spine", "X", amp=0.8), ch("chest", "X", amp=1.1, phase=0.06),
        ch("neck", "X", amp=-0.9, phase=0.14),
        ch("head", "Y", amp=1.3, harmonic=0.5, phase=0.25),
        ch("jaw", "X", amp=0.7, offset=0.7, harmonic=0.5, phase=0.1),
        ch("upper_arm.L", "Y", amp=1.6, phase=0.30),
        ch("upper_arm.R", "Y", amp=-1.6, phase=0.30),
        ch("forearm.L", "Y", amp=1.0, phase=0.36),
        ch("forearm.R", "Y", amp=-1.0, phase=0.36),
    ] + tail_wave(first=2.0, last=9.0)

    # เดิน: ขาสองข้างต่างเฟสกันครึ่งลูป แขนแกว่งสวนกับขาข้างเดียวกัน
    walk = [
        ch("thigh.R", "X", amp=-22 * leg, phase=0.00),
        ch("thigh.L", "X", amp=-22 * leg, phase=0.50),
        # เข่างอได้ทางเดียว จึงใช้ halfsin ไม่ใช่ไซน์เต็มลูก
        ch("shin.R", "X", amp=34 * leg, phase=0.55, mode="halfsin"),
        ch("shin.L", "X", amp=34 * leg, phase=0.05, mode="halfsin"),
        ch("foot.R", "X", amp=-12, phase=0.30),
        ch("foot.L", "X", amp=-12, phase=0.80),
        # ปลายเท้าพลิกขึ้นตอนถีบพื้น ไม่งั้นเท้าเป็นแผ่นแข็งลากไปกับพื้น
        ch("toe.R", "X", amp=-14, phase=0.38, mode="halfsin"),
        ch("toe.L", "X", amp=-14, phase=0.88, mode="halfsin"),
        ch("upper_arm.R", "X", amp=15, phase=0.00),
        ch("upper_arm.L", "X", amp=15, phase=0.50),
        ch("forearm.R", "X", amp=9, offset=7, phase=0.10),
        ch("forearm.L", "X", amp=9, offset=7, phase=0.60),
        ch("spine", "Z", amp=3.0), ch("chest", "Z", amp=2.5, phase=0.05),
        ch("hips", "Z", amp=-4.0),
        # ย่อ-ยืดสองครั้งต่อหนึ่งรอบ ตามจังหวะก้าว
        ch("hips", "Z", amp=0.030, harmonic=2, phase=0.25, kind="loc"),
    ] + tail_wave(first=3.0, last=14.0, step=0.10)

    # กระโดด: ครึ่งแรกย่อ ครึ่งหลังลอย — halfsin สองลูกคนละครึ่งลูป
    crouch, fly = dict(mode="halfsin", phase=0.00), dict(mode="halfsin", phase=0.50)
    # กรอบกล้องของหน้าเว็บคำนวณจากกรอบนอกของเมชที่ท่า rest และ field-of-view ถูก
    # ตรึงไว้ที่ 30 องศา เหลือที่ว่างเหนือหัวน้อยมาก ตัวที่ลอยขึ้นเกินราว 5% ของ
    # ความสูงตัวเองหัวจะหลุดเฟรมโดยกล้องไม่ตามขึ้นไป ท่านี้จึงเล่าด้วยการย่อลึก
    # แล้วเก็บขาสูงตอนลอย ไม่ใช่ด้วยการดันตัวขึ้น — สิ่งที่บอกว่าลอยคือเท้าพ้นพื้น
    # กับเงาที่หดลง ไม่ใช่ระยะที่หัวขยับขึ้น
    jump = [
        ch("hips", "Z", amp=-0.20, kind="loc", **crouch),
        ch("hips", "Z", amp=0.08, kind="loc", **fly),
        ch("thigh.L", "X", amp=-46 * leg, **crouch), ch("thigh.R", "X", amp=-46 * leg, **crouch),
        ch("shin.L", "X", amp=72 * leg, **crouch), ch("shin.R", "X", amp=72 * leg, **crouch),
        ch("foot.L", "X", amp=-26, **crouch), ch("foot.R", "X", amp=-26, **crouch),
        # ลอยแล้วเก็บขาขึ้นให้เข่าสูง ปลายเท้าชี้ลง เท้าจึงพ้นพื้นชัดทั้งที่สะโพก
        # แทบไม่ได้ขยับขึ้น
        ch("thigh.L", "X", amp=-44 * leg, **fly), ch("thigh.R", "X", amp=-44 * leg, **fly),
        ch("shin.L", "X", amp=74 * leg, **fly), ch("shin.R", "X", amp=74 * leg, **fly),
        ch("foot.L", "X", amp=26, **fly), ch("foot.R", "X", amp=26, **fly),
        ch("toe.L", "X", amp=18, **fly), ch("toe.R", "X", amp=18, **fly),
        # แขนเหวี่ยงลงหลังตอนย่อ แล้วกางขึ้นตอนลอย — ท่า A เริ่มที่แขนลง 45 องศา
        # อยู่แล้ว มุมกางจึงต้องมากกว่าที่ใช้กับท่า T อยู่ราวเท่านั้นจึงได้แขนเหนือไหล่
        ch("upper_arm.L", "X", amp=26, **crouch), ch("upper_arm.R", "X", amp=26, **crouch),
        # กางแขนได้ถึงแค่ระดับไหล่ด้วยเหตุผลเดียวกับความสูงที่ลอย คือมือที่ชูเหนือ
        # หัวจะพ้นกรอบนอกของท่า rest ซึ่งเป็นกรอบที่กล้องของหน้าเว็บใช้วางระยะ
        ch("upper_arm.L", "Y", amp=-55, **fly), ch("upper_arm.R", "Y", amp=55, **fly),
        ch("forearm.L", "Y", amp=-18, **fly), ch("forearm.R", "Y", amp=18, **fly),
        ch("chest", "X", amp=10, **crouch), ch("chest", "X", amp=-6, **fly),
        ch("neck", "X", amp=-14, **fly),
        ch("jaw", "X", amp=12, **fly),
        # หางยกขึ้นถ่วงตอนลอย ปล้องปลายยกมากกว่าปล้องโคน (หมุน +X = ยกขึ้น)
    ] + [ch(f"tail.{i + 1:02d}", "X", amp=5.0 + 2.5 * i, phase=0.50 + 0.03 * i,
            mode="halfsin") for i in range(N_TAIL)]

    # สะบัดหาง: คลื่นลูกเดียวกับ idle แต่แรงกว่าสามเท่า สะโพกขยับสวนไว้กันตัวลอย
    swing = tail_wave(first=9.0, last=30.0, step=0.12) + [
        ch("hips", "Z", amp=2.2, phase=0.5), ch("chest", "Z", amp=-2.0, phase=0.5),
        ch("head", "Z", amp=3.0, phase=0.55),
    ]

    # วิ่ง: รอบเร็วกว่าเดินเกือบเท่าตัว ก้าวยาวขึ้น เข่าพับลึกตอนเหวี่ยงขากลับ
    # ลำตัวโน้มไปหน้าค้างไว้ (offset) หัวเงยชดเชยให้สายตายังมองตรง แขนงอศอกค้างแล้ว
    # เหวี่ยงแรงสวนขา ตัวเด้งขึ้นลงสองครั้งต่อรอบ และหางสะบัดถ่วงสวนสะโพก
    run = [
        ch("thigh.R", "X", amp=-40 * leg, offset=-6 * leg, phase=0.00),
        ch("thigh.L", "X", amp=-40 * leg, offset=-6 * leg, phase=0.50),
        ch("shin.R", "X", amp=78 * leg, phase=0.58, mode="halfsin"),
        ch("shin.L", "X", amp=78 * leg, phase=0.08, mode="halfsin"),
        ch("foot.R", "X", amp=-20, phase=0.30),
        ch("foot.L", "X", amp=-20, phase=0.80),
        ch("toe.R", "X", amp=-22, phase=0.40, mode="halfsin"),
        ch("toe.L", "X", amp=-22, phase=0.90, mode="halfsin"),
        ch("upper_arm.R", "X", amp=34, phase=0.00),
        ch("upper_arm.L", "X", amp=34, phase=0.50),
        ch("forearm.R", "X", amp=14, offset=-55, phase=0.08),
        ch("forearm.L", "X", amp=14, offset=-55, phase=0.58),
        ch("spine", "X", amp=1.5, offset=7, harmonic=2),
        ch("chest", "X", amp=1.5, offset=6, harmonic=2, phase=0.05),
        ch("neck", "X", amp=-1.5, offset=-9, harmonic=2, phase=0.10),
        ch("spine", "Z", amp=5.0), ch("chest", "Z", amp=4.0, phase=0.05),
        ch("hips", "Z", amp=-7.0),
        ch("jaw", "X", amp=3.0, offset=5.0, harmonic=2, phase=0.2),
        ch("hips", "Z", amp=0.055, harmonic=2, phase=0.25, kind="loc"),
    ] + tail_wave(first=5.0, last=22.0, step=0.11, phase0=0.5)

    out = [
        {"name": "Idle", "frames": 64, "channels": idle},
        {"name": "Walk", "frames": 32, "channels": walk},
        {"name": "Run", "frames": 18, "channels": run},
        {"name": "Jump", "frames": 48, "channels": jump},
        {"name": "TailSwing", "frames": 40, "channels": swing},
    ]
    return {"fps": fps, "clips": out}


# ---------------------------------------------------------------------------


def open_scene(src: Path):
    """เปิดไฟล์ต้นทาง แล้วเหลือไว้แค่เมชกับโครงกระดูก

    .blend ที่ rig_croc.py เซฟไว้มีกล้องกับไฟสามดวงที่ใช้เรนเดอร์ภาพตรวจงานติดมา
    ด้วย ตัว export ของ glTF หยิบกล้องไปเป็น node ในไฟล์ด้วยถ้าไม่เอาออกก่อน
    """
    if src.suffix.lower() == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(src))
    else:
        glb_slim.load(src)

    for o in list(bpy.context.scene.objects):
        if o.type not in ("MESH", "ARMATURE"):
            bpy.data.objects.remove(o, do_unlink=True)

    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise SystemExit(f"ต้องมี armature อันเดียว แต่เจอ {[a.name for a in arms]}")
    arm = arms[0]

    # ท่าทดลองจาก rig_croc.py ไม่ใช่คลิปที่อยากให้ติดไปหน้าเว็บ ถ้าปล่อยไว้มันจะ
    # โผล่เป็นปุ่มอีกปุ่มหนึ่ง เพราะหน้าเว็บสร้างปุ่มจากทุก clip ที่อยู่ในไฟล์
    if arm.animation_data:
        arm.animation_data_clear()
    for act in list(bpy.data.actions):
        bpy.data.actions.remove(act)

    for pb in arm.pose.bones:          # ล้างท่าค้างให้กลับไปที่ rest pose
        pb.matrix_basis.identity()
    return arm


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--leg-scale", type=float, default=1.0,
                    help="ตัวคูณช่วงการแกว่งขาทั้งชุด")
    ap.add_argument("--draco", action="store_true")
    ap.add_argument("--jpeg", type=int, default=0, help="คุณภาพ JPEG ของภาพสี 0 = คง PNG")
    # normal/ORM เก็บตัวเลขไว้ในช่องสี ไม่ใช่ภาพให้ตาคนดู กดคุณภาพแรงเท่าสีไม่ได้
    # เพราะตัวเขียน JPEG ลดความละเอียดช่องสี ซึ่งเป็นช่องที่ normal map เก็บแกน X,Y
    ap.add_argument("--map-jpeg", type=int, default=0,
                    help="คุณภาพ JPEG ของ normal/ORM 0 = คง PNG ไว้")
    ap.add_argument("--tex-size", type=int, default=0, help="ย่อภาพสีเหลือกี่พิกเซล")
    ap.add_argument("--map-size", type=int, default=0, help="ย่อ normal/ORM เหลือกี่พิกเซล")
    args = ap.parse_args(argv)

    before = args.src.stat().st_size
    arm = open_scene(args.src)
    bones = {b.name for b in arm.data.bones}
    say(f"armature {arm.name}: {len(bones)} กระดูก")

    payload = clips(fps=args.fps, leg=args.leg_scale)
    made = blender_rig.apply_clips(arm, payload)
    say(f"สร้างอนิเมชัน {len(made)} clip: {', '.join(made)}")

    if args.jpeg or args.map_jpeg or args.tex_size or args.map_size:
        glb_slim.shrink_images(args.jpeg, args.map_jpeg, args.tex_size, args.map_size)

    blender_rig.export_glb(args.dst, args.draco)
    after = args.dst.stat().st_size
    say(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB ({100 * after / before:.0f}%)")


if __name__ == "__main__":
    main()
