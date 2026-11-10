"""เรนเดอร์โมเดล .glb (พร้อมวัสดุ PBR) 4 มุม front/left/back/right ด้วยแสงสตูดิโอ

    blender -b --factory-startup --python tools/render_views.py -- <model.glb> <out_prefix> [--size 1024]

กล้อง orthographic ทิศเดียวกับภาพต้นฉบับ (หน้าตัวละครคือ +Z ของ glTF = -Y ของ Blender)
แสง: key ด้านหน้าเฉียงบน + fill อ่อน + rim ด้านหลัง และ world สีขาวนวลคล้ายฉากถ่ายภาพ
ได้ไฟล์ <out_prefix>_<view>.png
"""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
size = int(argv[argv.index("--size") + 1]) if "--size" in argv else 1024

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
mins = Vector((min(min((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
maxs = Vector((max(max((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
center = (mins + maxs) / 2
extent = max(maxs - mins)

scn = bpy.context.scene
for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
    try:
        scn.render.engine = eng
        break
    except TypeError:
        continue
scn.render.resolution_x = scn.render.resolution_y = size
scn.render.film_transparent = False
scn.view_settings.view_transform = "Standard"

world = bpy.data.worlds.new("studio")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.92, 0.92, 0.94, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.45
scn.world = world


def light(name, kind, energy, loc, size_=3.0):
    data = bpy.data.lights.new(name, kind)
    data.energy = energy
    if kind == "AREA":
        data.size = size_
    ob = bpy.data.objects.new(name, data)
    scn.collection.objects.link(ob)
    ob.location = center + Vector(loc)
    d = center - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return ob


cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = extent * 1.08
cam = bpy.data.objects.new("cam", cam_data)
scn.collection.objects.link(cam)
scn.camera = cam

views = {"front": 0, "left": 90, "back": 180, "right": 270}
for name, az in views.items():
    for o in [o for o in scn.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(o, do_unlink=True)
    a = math.radians(az)
    toward = Vector((math.sin(a), -math.cos(a), 0))
    side = Vector((math.cos(a), math.sin(a), 0))
    light("key", "AREA", 300, (toward * 4 + side * 2 + Vector((0, 0, 3)))[:], 4)
    light("fill", "AREA", 120, (toward * 4 - side * 3 + Vector((0, 0, 0.5)))[:], 5)
    light("rim", "AREA", 150, (-toward * 4 + Vector((0, 0, 3)))[:], 3)
    cam.location = center + toward * 10
    cam.rotation_euler = (-toward).to_track_quat("-Z", "Y").to_euler()
    scn.render.filepath = f"{out}_{name}.png"
    bpy.ops.render.render(write_still=True)
    print("[render]", name)
