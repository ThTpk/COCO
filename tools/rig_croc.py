"""ใส่โครงกระดูกให้โมเดลจระเข้ แล้วทดลองจัดท่า

    blender -b --factory-startup --python tools/rig_croc.py -- <model.glb> <bones.json> <out_prefix> [--faces 100000]

1. นำเข้า .glb รวมจุดซ้ำ (UV เก็บต่อมุมหน้า ไม่เสีย) แล้วลดจำนวนหน้าด้วย Decimate
2. สร้าง armature จาก bones.json (พิกัดแกน glTF: +y ขึ้น, +z หน้า) แปลงเป็นแกน Blender
3. ผูกเมชด้วย Automatic Weights (heat) จุดที่ไม่ได้น้ำหนักจะผูกกับกระดูกที่ใกล้ที่สุด
   ขากรรไกรล่างกำหนดน้ำหนักเองจากแนวปาก
4. ทำ action "test_pose" (เฟรม 1 = ท่า T, เฟรม 20 = ท่าทดลอง) เรนเดอร์ภาพหน้า/ข้าง ทั้งสองท่า
ได้ <out_prefix>.blend, <out_prefix>.glb (มี skin + action), <out_prefix>_weights.txt
และ <out_prefix>_{rest,pose}_{front,left}.png
"""
import json
import math
import sys

import bmesh
import bpy
from mathutils import Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, bones_json, out = argv[0], argv[1], argv[2]
target_faces = int(argv[argv.index("--faces") + 1]) if "--faces" in argv else 100000


def g2b(p):  # glTF -> Blender
    return Vector((p[0], -p[2], p[1]))


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
me = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
me.name = "croc"
bpy.context.view_layer.objects.active = me
for o in bpy.context.scene.objects:
    o.select_set(o == me)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# ---- 1. รวมจุดซ้ำตามรอยตัด UV แล้วลดหน้า
bm = bmesh.new()
bm.from_mesh(me.data)
n0 = len(bm.faces)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
bm.to_mesh(me.data)
bm.free()
dec = me.modifiers.new("decimate", "DECIMATE")
dec.ratio = min(1.0, target_faces / max(1, len(me.data.polygons)))
bpy.ops.object.modifier_apply(modifier=dec.name)
for p in me.data.polygons:
    p.use_smooth = True
print(f"[rig] faces {n0:,} -> {len(me.data.polygons):,}")

# ---- 2. armature
B = json.load(open(bones_json, encoding="utf-8"))
arm_data = bpy.data.armatures.new("croc_rig")
arm = bpy.data.objects.new("croc_rig", arm_data)
bpy.context.scene.collection.objects.link(arm)
arm.show_in_front = True
bpy.context.view_layer.objects.active = arm
for o in bpy.context.scene.objects:
    o.select_set(o == arm)
bpy.ops.object.mode_set(mode="EDIT")
eb = {}
for b in B:
    e = arm_data.edit_bones.new(b["name"])
    e.head, e.tail = g2b(b["head"]), g2b(b["tail"])
    d = (e.tail - e.head).normalized()
    # แกน Z ของกระดูก: กระดูกแนวนอนชี้ขึ้น, กระดูกแนวตั้งชี้ไปหน้า (-Y)
    up = Vector((0, 0, 1)) if abs(d.z) < 0.7 else Vector((0, -1, 0))
    e.align_roll(up)
    e.use_deform = b.get("deform", True)
    eb[b["name"]] = e
for b in B:
    if b["parent"]:
        e, p = eb[b["name"]], eb[b["parent"]]
        e.parent = p
        e.use_connect = (e.head - p.tail).length < 1e-4
bpy.ops.object.mode_set(mode="OBJECT")

# ---- 3. ผูกเมช
for o in bpy.context.scene.objects:
    o.select_set(o in (me, arm))
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_AUTO")

deform = [b["name"] for b in B if b.get("deform", True)]
vg = {n: me.vertex_groups.get(n) or me.vertex_groups.new(name=n) for n in deform}
segs = {n: (arm_data.bones[n].head_local.copy(), arm_data.bones[n].tail_local.copy()) for n in deform}


def seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
    return (p - (a + ab * t)).length


verts = me.data.vertices
unweighted = [v.index for v in verts if not any(g.weight > 1e-4 for g in v.groups)]
for i in unweighted:
    p = verts[i].co
    best = min(deform, key=lambda n: seg_dist(p, *segs[n]))
    vg[best].add([i], 1.0, "REPLACE")

# ขากรรไกรล่าง: น้ำหนักไล่ระดับรอบแนวปาก (glTF y ~0.66) ไม่ตัดแข็ง 0/1 กันหน้าแตกเป็นเหลี่ยม
# w = (ต่ำกว่าแนวปาก) x (อยู่ช่วงหน้าหัว ไม่ใช่คอ) x (อยู่ในความกว้างหัว)
def ramp(v, a, b):
    return max(0.0, min(1.0, (v - a) / (b - a)))


jaw_ids = []
gi = {g.index: g.name for g in me.vertex_groups}
for v in verts:
    gx, gy, gz = v.co.x, v.co.z, -v.co.y
    if gy < 0.55 or gz < 0.10:
        continue
    w = ramp(0.675 - gy, 0.0, 0.03) * ramp(gz, 0.13, 0.21) * ramp(0.19 - abs(gx + 0.02), 0.0, 0.04)
    old = {gi[g.group]: g.weight for g in v.groups if gi[g.group] in vg}
    if w <= 0.0:
        if old.get("jaw", 0) > 0:          # ส่วนบนของหัว: เอา jaw ออก ให้ head แทน
            vg["jaw"].remove([v.index])
            vg["head"].add([v.index], old.get("head", 0) + old["jaw"], "REPLACE")
        continue
    jaw_ids.append(v.index)
    rest = {n: wt for n, wt in old.items() if n != "jaw"}
    tot = sum(rest.values()) or 1.0
    for n, wt in rest.items():
        vg[n].add([v.index], (1 - w) * wt / tot, "REPLACE")
    if not rest:
        vg["head"].add([v.index], 1 - w, "REPLACE")
    vg["jaw"].add([v.index], w, "REPLACE")

counts = {n: 0 for n in deform}
for v in verts:
    for g in v.groups:
        name = me.vertex_groups[g.group].name
        if g.weight > 0.5 and name in counts:
            counts[name] += 1
with open(out + "_weights.txt", "w", encoding="utf-8") as f:
    f.write(f"faces {len(me.data.polygons)}  verts {len(verts)}  heat-unweighted {len(unweighted)}"
            f"  jaw {len(jaw_ids)}\n")
    for n in deform:
        f.write(f"{n:16s} {counts[n]}\n")
print(f"[rig] unweighted after heat: {len(unweighted)}  jaw verts: {len(jaw_ids)}")

# ---- 4. ท่าทดลอง
bpy.context.view_layer.objects.active = arm
for o in bpy.context.scene.objects:
    o.select_set(o == arm)
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"


def rot(name, axis, deg):
    pb = arm.pose.bones[name]
    loc_axis = pb.bone.matrix_local.to_3x3().inverted() @ Vector(axis)
    pb.rotation_quaternion = Quaternion(loc_axis.normalized(), math.radians(deg)) @ pb.rotation_quaternion


def key_all(frame):
    for pb in arm.pose.bones:
        pb.keyframe_insert("rotation_quaternion", frame=frame)


arm.animation_data_create()
arm.animation_data.action = bpy.data.actions.new("test_pose")
key_all(1)
rot("upper_arm.L", (0, 1, 0), 55); rot("upper_arm.R", (0, 1, 0), -55)
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
scn = bpy.context.scene
scn.frame_start, scn.frame_end = 1, 20

# ---- เรนเดอร์
for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
    try:
        scn.render.engine = eng
        break
    except TypeError:
        continue
scn.render.resolution_x = scn.render.resolution_y = 900
scn.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("studio")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.92, 0.92, 0.94, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.45
scn.world = world
arm.hide_render = True
center = Vector((0, 0, 0))
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 2.1
cam = bpy.data.objects.new("cam", cam_data)
scn.collection.objects.link(cam)
scn.camera = cam


def light(name, energy, loc, size_):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.size = energy, size_
    ob = bpy.data.objects.new(name, data)
    scn.collection.objects.link(ob)
    ob.location = Vector(loc)
    ob.rotation_euler = (center - ob.location).to_track_quat("-Z", "Y").to_euler()


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
        scn.render.filepath = f"{out}_{tag}_{name}.png"
        bpy.ops.render.render(write_still=True)
        print("[render]", tag, name)

scn.frame_set(1)
arm.hide_render = False
bpy.ops.wm.save_as_mainfile(filepath=out + ".blend")
for o in bpy.context.scene.objects:
    o.select_set(o in (me, arm))
bpy.ops.export_scene.gltf(filepath=out + ".glb", use_selection=True, export_skins=True,
                          export_animations=True)
print("[rig] done", out)
