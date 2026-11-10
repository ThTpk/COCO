"""เปลี่ยนท่าปกติ (rest pose) ของโมเดลที่ใส่โครงแล้ว จากท่า T เป็นท่า A

    blender -b --factory-startup --python tools/apose_croc.py -- <rigged.blend> <out_prefix> \
        [--arm 62] [--palm 90]

1. เปิด .blend จาก rig_croc.py ล้าง action แล้วหมุนต้นแขนลงให้ทำมุม --arm องศาใต้แนวนอน
   (ปลายแขนเหยียดตามแนวต้นแขน) แล้วบิดแขนรอบแกนของตัวเอง --palm องศาเพื่อคว่ำมือลง
   ท่า T ที่ปั้นมาวางมือแบบหันฝ่ามือไปข้างหน้า (นิ้วเรียงบน-ล่าง นิ้วโป้งอยู่บน)
   การบิดรอบแกนแขนจึงเป็นการคว่ำมือจริง ๆ ไม่ใช่การหมุนทั้งแขนไปด้านหลัง
   แบ่งมุมบิดเป็นต้นแขน 40% ปลายแขน 60% เพื่อไม่ให้ผิวบิดกองอยู่ที่ข้อเดียว
2. Apply ท่านี้ลงเมช (คัดลอก Armature modifier ไว้ก่อน) แล้วตั้งเป็น rest pose ของ armature
3. สร้าง action "test_pose" ใหม่ (เฟรม 1 = ท่า A, เฟรม 20 = ท่าทดลองแบบเดิม)
4. เรนเดอร์หน้า/ข้าง ทั้งท่า A และท่าทดลอง แล้วบันทึก <out_prefix>.blend และ .glb
"""
import math
import os
import sys

import bpy
from mathutils import Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
arm_deg = float(argv[argv.index("--arm") + 1]) if "--arm" in argv else 62.0
palm_deg = float(argv[argv.index("--palm") + 1]) if "--palm" in argv else 90.0

bpy.ops.wm.open_mainfile(filepath=src)
scn = bpy.context.scene
arm = bpy.data.objects["croc_rig"]
me = bpy.data.objects["croc"]
if arm.animation_data and arm.animation_data.action:
    old = arm.animation_data.action
    arm.animation_data.action = None
    bpy.data.actions.remove(old)
scn.frame_set(1)

bpy.context.view_layer.objects.active = arm
for o in scn.objects:
    o.select_set(o == arm)
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = Quaternion()
    pb.location = (0, 0, 0)
    pb.scale = (1, 1, 1)


def rot(name, axis, deg):
    pb = arm.pose.bones[name]
    loc_axis = pb.bone.matrix_local.to_3x3().inverted() @ Vector(axis)
    pb.rotation_quaternion = Quaternion(loc_axis.normalized(), math.radians(deg)) @ pb.rotation_quaternion


def twist(name, deg):
    """บิดรอบแกนของกระดูกเอง (แกน Y ท้องถิ่นคือหัวไปปลาย) — ใช้คว่ำ/หงายมือ

    คูณต่อท้ายไม่ใช่ต่อหน้า เพราะอยากให้บิดรอบแกนแขน *หลัง* หมุนลงแล้ว ซึ่งก็คือ
    บิดรอบแกนของกระดูกเองก่อนแล้วค่อยหมุนลง ผลเท่ากันเพราะแกนบิดถูกพาไปด้วย
    """
    pb = arm.pose.bones[name]
    pb.rotation_quaternion = pb.rotation_quaternion @ Quaternion((0, 1, 0), math.radians(deg))


def slope(name):  # มุมของกระดูกเทียบแนวนอน (องศา, ลบ = ชี้ลง) ในระนาบหน้า X-Z
    b = arm.data.bones[name]
    d = b.tail_local - b.head_local
    return math.degrees(math.atan2(d.z, abs(d.x)))


for s, sg in (("L", 1), ("R", -1)):
    up = slope(f"upper_arm.{s}")
    fo = slope(f"forearm.{s}")
    turn = arm_deg + up                      # หมุนลงเพิ่มเท่าไรจึงได้ -arm_deg
    rot(f"upper_arm.{s}", (0, 1, 0), sg * turn)
    rot(f"forearm.{s}", (0, 1, 0), sg * max(0.0, (fo - up)))  # ปลายแขนเหยียดตามต้นแขน
    # แกน Y ท้องถิ่นของกระดูกชี้ออกจากตัวทั้งสองข้าง (ซ้าย +x, ขวา -x) มุมบิดที่ทำให้
    # ฝ่ามือคว่ำลงเหมือนกันทั้งคู่จึงต้องกลับเครื่องหมายตามข้าง เหมือนมุมที่หมุนลง
    twist(f"upper_arm.{s}", sg * palm_deg * 0.4)
    twist(f"forearm.{s}", sg * palm_deg * 0.6)
    print(f"[apose] {s}: upper {up:.1f} -> {-arm_deg:.1f} (หมุน {turn:.1f}), "
          f"forearm {fo:.1f}, บิดคว่ำมือ {palm_deg:.0f}")
bpy.context.view_layer.update()

# ---- apply ท่าลงเมช แล้วตั้งเป็น rest pose
bpy.ops.object.mode_set(mode="OBJECT")
mod = next(m for m in me.modifiers if m.type == "ARMATURE")
bpy.context.view_layer.objects.active = me
for o in scn.objects:
    o.select_set(o == me)
bpy.ops.object.modifier_copy(modifier=mod.name)
bpy.ops.object.modifier_apply(modifier=mod.name)
keep = next(m for m in me.modifiers if m.type == "ARMATURE")
keep.name = "Armature"
keep.object = arm
bpy.context.view_layer.objects.active = arm
for o in scn.objects:
    o.select_set(o == arm)
bpy.ops.object.mode_set(mode="POSE")
bpy.ops.pose.armature_apply(selected=False)
print("[apose] rest pose = A-pose")

# ---- action ทดลองใหม่ (ค่าต้นแขนลดลงตามมุมที่หมุนไปแล้ว)
arm.animation_data_create()
arm.animation_data.action = bpy.data.actions.new("test_pose")


def key_all(frame):
    for pb in arm.pose.bones:
        pb.keyframe_insert("rotation_quaternion", frame=frame)


key_all(1)
rot("upper_arm.L", (0, 1, 0), 20); rot("upper_arm.R", (0, 1, 0), -20)
rot("forearm.L", (0, 0, 1), -40); rot("forearm.R", (0, 0, 1), 40)
rot("head", (0, 0, 1), 25); rot("jaw", (1, 0, 0), 15)
rot("chest", (0, 0, 1), -10)
rot("thigh.L", (1, 0, 0), -35); rot("shin.L", (1, 0, 0), 50)
for i in range(1, 9):
    rot(f"tail.{i:02d}", (0, 0, 1), -8)
    rot(f"tail.{i:02d}", (1, 0, 0), 4)
for s in ("L", "R"):
    for k in (1, 2, 3):
        rot(f"finger{k}.01.{s}", (0, 1 if s == "L" else -1, 0), 35)
        rot(f"finger{k}.02.{s}", (0, 1 if s == "L" else -1, 0), 35)
key_all(20)
bpy.ops.object.mode_set(mode="OBJECT")

# ---- เรนเดอร์ (ฉาก กล้อง แสง จากไฟล์เดิม ถ้ามี ไม่งั้นสร้างใหม่)
cam = scn.camera
center = Vector((0, 0, 0))


def light(name, energy, loc, size_):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.size = energy, size_
    ob = bpy.data.objects.new(name, data)
    scn.collection.objects.link(ob)
    ob.location = Vector(loc)
    ob.rotation_euler = (center - ob.location).to_track_quat("-Z", "Y").to_euler()


arm.hide_render = True
for frame, tag in ((1, "rest"), (20, "pose")):
    scn.frame_set(frame)
    for name, az in (("front", 0), ("left", 90)):
        for o in [o for o in scn.objects if o.type == "LIGHT"]:
            bpy.data.objects.remove(o, do_unlink=True)
        a = math.radians(az)
        toward = Vector((math.sin(a), -math.cos(a), 0))
        side = Vector((math.cos(a), math.sin(a), 0))
        light("key", 300, toward * 4 + side * 2 + Vector((0, 0, 3)), 4)
        light("fill", 120, toward * 4 - side * 3 + Vector((0, 0, 0.5)), 5)
        light("rim", 150, -toward * 4 + Vector((0, 0, 3)), 3)
        cam.location = center + toward * 10
        cam.rotation_euler = (-toward).to_track_quat("-Z", "Y").to_euler()
        # ต้องเป็น path เต็ม: Blender อ่าน path สัมพัทธ์ของ render เทียบกับไฟล์ .blend
        # ที่เปิดอยู่ ไม่ใช่เทียบกับที่ที่สั่งรัน ภาพจึงไปโผล่ผิดที่แบบเงียบ ๆ
        scn.render.filepath = os.path.abspath(f"{out}_{tag}_{name}.png")
        bpy.ops.render.render(write_still=True)
        print("[render]", tag, name)

scn.frame_set(1)
arm.hide_render = False
bpy.ops.wm.save_as_mainfile(filepath=out + ".blend")
for o in scn.objects:
    o.select_set(o in (me, arm))
bpy.ops.export_scene.gltf(filepath=out + ".glb", use_selection=True, export_skins=True,
                          export_animations=True)
print("[apose] done", out)
