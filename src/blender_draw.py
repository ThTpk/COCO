r"""สคริปต์ที่รันใน interpreter ของ Blender — อย่าเรียกจาก venv ปกติ

    blender --background --factory-startup --python blender_draw.py -- \
        --mesh output\croc.glb --texture output\croc_texture.png \
        --normal output\croc_normal.png --rough output\croc_rough.png \
        --out output\croc_blender --turns 16

หน้าที่: จัดฉากให้โมเดลที่แกะมาจากคลิป แล้วเรนเดอร์เป็นภาพหลายมุม
  * ต่อวัสดุใหม่จากไฟล์แผนที่ความละเอียดเต็ม (ใน GLB เป็น JPEG ที่บีบแล้ว)
  * ไฟสามดวงแบบสตูดิโอ + พื้นรับเงา
  * กล้องเล็งกึ่งกลางตัว หมุนรอบเป็นรอบเดียวครบ
  * เก็บไฟล์ .blend ไว้เปิดแก้ต่อได้

เรื่องที่ต้องระวังข้ามเวอร์ชัน: ชื่อ engine (`BLENDER_EEVEE` -> `BLENDER_EEVEE_NEXT`)
กับคุณสมบัติของ EEVEE เปลี่ยนไปมาระหว่าง Blender 4.x กับ 5.x สคริปต์จึงถาม
ตัว API ก่อนทุกครั้งแทนที่จะเดาชื่อเอง
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def say(msg: str) -> None:
    """พิมพ์ด้วย prefix ที่ฝั่ง host กรองเก็บไว้แสดง"""
    print(f"[draw] {msg}", flush=True)


# --------------------------------------------------------------------------
# เตรียมฉาก
# --------------------------------------------------------------------------
def wipe_scene() -> None:
    """ล้างทุกอย่างที่ factory startup แถมมา (ลูกบาศก์ กล้อง ไฟ)"""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        raise ValueError(f"ยังอ่านสกุลนี้ไม่ได้: {path.suffix}")
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"ไม่พบ mesh ใน {path}")
    tris = sum(len(o.data.loop_triangles) if o.data.loop_triangles
               else len(o.data.polygons) for o in meshes)
    say(f"อ่าน {path.name}: {len(meshes)} ชิ้น, "
        f"{sum(len(o.data.vertices) for o in meshes):,} จุด, ~{tris:,} หน้า")
    return meshes


def world_bounds(objs: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for o in objs:
        for corner in o.bound_box:
            p = o.matrix_world @ Vector(corner)
            for k in range(3):
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])
    return lo, hi


def drop_to_floor(objs: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    """เลื่อนให้ฝ่าเท้าแตะ z=0 และแกนตัวอยู่ที่จุดกำเนิด"""
    lo, hi = world_bounds(objs)
    shift = Vector((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z))
    for o in objs:
        if o.parent is None:
            o.location += shift
    bpy.context.view_layer.update()
    return world_bounds(objs)


# --------------------------------------------------------------------------
# วัสดุ
# --------------------------------------------------------------------------
def load_image(path: Path, non_color: bool = False) -> bpy.types.Image:
    img = bpy.data.images.load(str(path), check_existing=True)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return img


def build_material(name: str, color: Path | None, normal: Path | None,
                   rough: Path | None, normal_strength: float,
                   rough_floor: float, flip_v: bool = True) -> bpy.types.Material:
    """วัสดุ Principled ที่เสียบแผนที่จากไฟล์ตรง ๆ

    ``flip_v`` กลับแกน V ให้ก่อนสุ่มภาพ เพราะ ``src/export.py`` พลิกภาพแนวตั้ง
    (``texture[::-1]``) ตอนฝังลง GLB ตามแบบแผนของ glTF ที่นับ V จากขอบบน
    ส่วนไฟล์ PNG ที่เขียนคู่กันไว้บนดิสก์ยังเป็นด้านตั้งเดิมของ atlas อยู่
    ถ้าไม่พลิกกลับ แต่ละชิ้นส่วนจะไปหยิบสีของชิ้นอื่น กลายเป็นผิวลายพร้อย
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")

    uv = nt.nodes.new("ShaderNodeTexCoord")
    uv.location = (-1400, 0)
    source = uv.outputs["UV"]

    if flip_v:
        flip = nt.nodes.new("ShaderNodeMapping")
        flip.location = (-1180, 0)
        flip.inputs["Location"].default_value = (0.0, 1.0, 0.0)
        flip.inputs["Scale"].default_value = (1.0, -1.0, 1.0)
        nt.links.new(source, flip.inputs["Vector"])
        source = flip.outputs["Vector"]
        say("UV: พลิกแกน V ให้ตรงกับ atlas ที่เขียนลงดิสก์")

    def tex(path: Path, y: int, non_color: bool):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = load_image(path, non_color)
        node.location = (-820, y)
        nt.links.new(source, node.inputs["Vector"])
        return node

    if color is not None:
        n = tex(color, 300, False)
        nt.links.new(n.outputs["Color"], bsdf.inputs["Base Color"])
        say(f"วัสดุ: สีผิว <- {color.name}")

    if rough is not None:
        n = tex(rough, 0, True)
        # บีบช่วงความมันไม่ให้ดำสนิทจนเงาสะท้อนแข็งเกินจริง
        rmap = nt.nodes.new("ShaderNodeMapRange")
        rmap.location = (-520, 0)
        rmap.inputs["From Min"].default_value = 0.0
        rmap.inputs["From Max"].default_value = 1.0
        rmap.inputs["To Min"].default_value = rough_floor
        rmap.inputs["To Max"].default_value = 1.0
        nt.links.new(n.outputs["Color"], rmap.inputs["Value"])
        nt.links.new(rmap.outputs["Result"], bsdf.inputs["Roughness"])
        say(f"วัสดุ: ความมัน <- {rough.name} (พื้น {rough_floor:.2f})")
    else:
        bsdf.inputs["Roughness"].default_value = 0.55

    if normal is not None:
        n = tex(normal, -300, True)
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.location = (-520, -300)
        nmap.inputs["Strength"].default_value = normal_strength
        nt.links.new(n.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        say(f"วัสดุ: ลายนูน <- {normal.name} (แรง {normal_strength:.2f})")

    for key, val in (("Metallic", 0.0), ("IOR", 1.45)):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = val
    return mat


def clay_material(name: str) -> bpy.types.Material:
    """วัสดุเทาล้วน ไว้ดูรูปทรงโดยไม่ให้ลายสีมาหลอกตา"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.62, 0.60, 0.58, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    return mat


def assign(objs: list[bpy.types.Object], mat: bpy.types.Material) -> None:
    for o in objs:
        o.data.materials.clear()
        o.data.materials.append(mat)


# --------------------------------------------------------------------------
# ไฟ กล้อง พื้น
# --------------------------------------------------------------------------
def add_area(name: str, loc: Vector, aim: Vector, size: float,
             power: float, color=(1.0, 1.0, 1.0)) -> bpy.types.Object:
    data = bpy.data.lights.new(name, type="AREA")
    data.shape = "DISK"
    data.size = size
    data.energy = power
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = loc
    direction = (aim - loc).normalized()
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return obj


def build_lights(center: Vector, height: float) -> None:
    """คีย์หน้าซ้าย ฟิลล์หน้าขวา ริมหลัง — ชุดสตูดิโอมาตรฐาน"""
    r = height * 1.9
    key_p = height * height * 95
    add_area("Key", center + Vector((-r * 0.75, -r * 0.70, height * 1.15)),
             center, height * 0.95, key_p, (1.0, 0.97, 0.92))
    add_area("Fill", center + Vector((r * 0.95, -r * 0.55, height * 0.55)),
             center, height * 1.3, key_p * 0.32, (0.90, 0.94, 1.0))
    add_area("Rim", center + Vector((r * 0.25, r * 0.95, height * 1.45)),
             center, height * 0.7, key_p * 0.80, (0.95, 0.96, 1.0))
    say("ไฟ: key / fill / rim 3 ดวง")


def build_world(strength: float, top=(0.42, 0.47, 0.56),
                bottom=(0.10, 0.10, 0.12)) -> None:
    """ท้องฟ้าไล่เฉดอ่อน ๆ ทำหน้าที่แทน ambient"""
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*bottom, 1.0)
    ramp.color_ramp.elements[1].color = (*top, 1.0)
    grad = nt.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    nt.links.new(geo.outputs["Incoming"], grad.inputs["Vector"])
    nt.links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def build_floor(center: Vector, height: float, shadow_only: bool) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=height * 14,
                                     location=(center.x, center.y, 0.0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    if shadow_only and hasattr(floor, "is_shadow_catcher"):
        floor.is_shadow_catcher = True
        say("พื้น: shadow catcher (เห็นแต่เงา)")
    else:
        mat = bpy.data.materials.new("Floor")
        mat.use_nodes = True
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = (0.14, 0.145, 0.16, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.72
        floor.data.materials.append(mat)
    return floor


def build_camera(center: Vector, size: Vector, lens: float,
                 margin: float) -> tuple[bpy.types.Object, bpy.types.Object]:
    """กล้องแขวนกับ empty ที่กึ่งกลางตัว — หมุน empty = หมุนกล้องรอบตัว"""
    pivot = bpy.data.objects.new("Pivot", None)
    bpy.context.collection.objects.link(pivot)
    pivot.location = center

    data = bpy.data.cameras.new("Camera")
    data.lens = lens
    cam = bpy.data.objects.new("Camera", data)
    bpy.context.collection.objects.link(cam)
    cam.parent = pivot

    # ระยะที่ทำให้ตัวพอดีเฟรม: ด้านที่คับที่สุดต้องอยู่ในมุมรับภาพ
    scene = bpy.context.scene
    sensor = data.sensor_width
    fov_v = 2 * math.atan((sensor * scene.render.resolution_y
                           / scene.render.resolution_x) / (2 * lens))
    fov_h = 2 * math.atan(sensor / (2 * lens))
    half_v = size.z / 2 * margin
    half_h = max(size.x, size.y) / 2 * margin
    dist = max(half_v / math.tan(fov_v / 2), half_h / math.tan(fov_h / 2))

    tilt = math.radians(6.0)
    cam.location = (0.0, -dist * math.cos(tilt), dist * math.sin(tilt))
    cam.rotation_euler = (math.pi / 2 - tilt, 0.0, 0.0)
    scene.camera = cam
    say(f"กล้อง: {lens:.0f} mm ห่าง {dist:.2f} m เล็งที่ z={center.z:.2f}")
    return cam, pivot


# --------------------------------------------------------------------------
# การเรนเดอร์
# --------------------------------------------------------------------------
# enum หลายตัวของ Blender (engine, compute_device_type, denoiser, look)
# สร้างรายการแบบ dynamic จากฝั่ง C ซึ่งในโหมด --background คืนลิสต์ "ว่าง"
# ออกมา ทั้งที่ค่าที่ต้องการใช้ได้จริง เช็คด้วย `in enum_items` จึงไม่ได้ผล —
# ต้องลอง "เขียนค่าลงไป" แล้วดักข้อผิดพลาดแทน
def try_set(owner, attr: str, values: list[str]) -> str | None:
    """ไล่เขียนค่าทีละตัวจนกว่าจะมีตัวที่ Blender ยอมรับ คืนตัวที่ติด"""
    for value in values:
        try:
            setattr(owner, attr, value)
        except (TypeError, ValueError):
            continue
        return value
    return None


def enable_gpu(prefer: str = "auto") -> str:
    """เปิดการ์ดจอให้ Cycles ใช้ คืนชื่อ backend ที่ได้จริง ("CPU" ถ้าไม่ได้)

    ลำดับที่ลอง: OptiX (เร็วสุดบนการ์ด NVIDIA รุ่นใหม่ เพราะใช้ RT core ตรง ๆ)
    -> CUDA -> HIP/oneAPI/Metal สำหรับเครื่องที่ไม่ใช่ NVIDIA
    """
    if "cycles" not in bpy.context.preferences.addons:
        try:
            bpy.ops.preferences.addon_enable(module="cycles")
        except RuntimeError as exc:
            say(f"เปิด add-on cycles ไม่ได้ ({exc}) — ใช้ CPU")
            return "CPU"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    order = (["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"] if prefer == "auto"
             else [prefer.upper()])

    for backend in order:
        if try_set(prefs, "compute_device_type", [backend]) is None:
            continue
        prefs.get_devices()
        cards = [d for d in prefs.devices if d.type == backend]
        if not cards:
            continue
        for d in prefs.devices:
            # เปิดเฉพาะการ์ด ไม่เปิด CPU ร่วม — CPU ที่ช้ากว่ามากจะกลายเป็น
            # ตัวถ่วงเพราะ Cycles รอ tile ของ device ที่ช้าที่สุดอยู่ดี
            d.use = d.type == backend
        say(f"GPU: {backend} — " + ", ".join(d.name for d in cards))
        return backend

    say("ไม่พบการ์ดที่ Cycles ใช้ได้ — เรนเดอร์ด้วย CPU")
    return "CPU"


def engine_id(prefer: str) -> str:
    """ตั้ง render engine ตามที่ขอ คืนชื่อที่ติดจริง

    ``--factory-startup`` ไม่โหลด add-on และ Cycles ก็เป็น add-on จึงต้อง
    เปิดเองก่อน ไม่งั้นจะตกไปใช้ EEVEE เงียบ ๆ ทั้งที่สั่ง Cycles ไว้
    """
    if prefer == "cycles" and "cycles" not in bpy.context.preferences.addons:
        try:
            bpy.ops.preferences.addon_enable(module="cycles")
            say("เปิด add-on cycles (factory-startup ไม่ได้โหลดมาให้)")
        except RuntimeError as exc:
            say(f"เปิด add-on cycles ไม่ได้: {exc}")

    want = (["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"]
            if prefer == "cycles"
            else ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"])
    got = try_set(bpy.context.scene.render, "engine", want)
    if got is None:
        got = bpy.context.scene.render.engine
        say(f"ตั้ง engine ไม่สำเร็จ — ใช้ {got} ตามเดิม")
    elif got != want[0]:
        say(f"ไม่พบ engine {prefer} — ใช้ {got} แทน")
    return got


def setup_render(prefer: str, width: int, height: int, samples: int,
                 transparent: bool, device: str) -> str:
    scene = bpy.context.scene
    r = scene.render
    eng = engine_id(prefer)
    r.engine = eng
    r.resolution_x, r.resolution_y = width, height
    r.resolution_percentage = 100
    r.film_transparent = transparent
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA" if transparent else "RGB"
    r.image_settings.compression = 20

    if eng == "CYCLES":
        cy = scene.cycles
        cy.samples = samples
        cy.use_denoising = True
        cy.max_bounces = 8
        backend = "CPU" if device == "cpu" else enable_gpu(
            "auto" if device in ("auto", "gpu") else device)
        cy.device = "CPU" if backend == "CPU" else "GPU"
        if backend == "OPTIX":
            # OptiX denoiser ทำงานบนการ์ดใบเดียวกัน ไม่ต้องโยนภาพกลับมา CPU
            try_set(cy, "denoiser", ["OPTIX", "OPENIMAGEDENOISE"])
        say(f"Cycles: device {cy.device} ({backend}), denoiser "
            f"{getattr(cy, 'denoiser', '-')}")
    else:
        ee = scene.eevee
        for attr in ("taa_render_samples", "taa_samples"):
            if hasattr(ee, attr):
                setattr(ee, attr, samples)
        for attr in ("use_gtao", "use_raytracing", "use_shadows",
                     "use_soft_shadows"):
            if hasattr(ee, attr):
                setattr(ee, attr, True)
        if hasattr(ee, "shadow_ray_count"):
            ee.shadow_ray_count = 2

    # AgX เก็บไฮไลต์ได้ดีแต่ลดความอิ่มสี ซึ่งกินสีส้มของหนังจระเข้ไปเยอะ
    # look "Punchy" ดึงกลับมาให้ใกล้สีในคลิปเดิม
    vs = scene.view_settings
    try_set(vs, "view_transform", ["AgX", "Filmic", "Standard"])
    try_set(vs, "look", [f"{vs.view_transform} - Punchy", "Punchy",
                         f"{vs.view_transform} - Medium Contrast", "None"])
    say(f"engine {eng} @ {width}x{height}, {samples} samples, "
        f"view transform {vs.view_transform}")
    return eng


def render_turn(pivot: bpy.types.Object, out_dir: Path, turns: int,
                tag: str) -> list[Path]:
    scene = bpy.context.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for k in range(turns):
        deg = 360.0 * k / turns
        pivot.rotation_euler = (0.0, 0.0, math.radians(deg))
        bpy.context.view_layer.update()
        path = out_dir / f"{tag}_{k:02d}_{int(round(deg)):03d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        written.append(path)
        say(f"เรนเดอร์ {deg:5.1f} องศา -> {path.name}")
    pivot.rotation_euler = (0.0, 0.0, 0.0)
    return written


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="blender_draw")
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--texture", type=Path)
    p.add_argument("--normal", type=Path)
    p.add_argument("--rough", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--blend", type=Path)
    p.add_argument("--turns", type=int, default=16)
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=1400)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--engine", choices=["eevee", "cycles"], default="cycles")
    p.add_argument("--device", choices=["auto", "gpu", "cpu", "optix", "cuda"],
                   default="auto",
                   help="อุปกรณ์ที่ให้ Cycles ใช้ (auto = การ์ดจอถ้ามี)")
    p.add_argument("--lens", type=float, default=85.0)
    p.add_argument("--margin", type=float, default=1.18)
    p.add_argument("--normal-strength", type=float, default=1.0)
    p.add_argument("--rough-floor", type=float, default=0.22)
    p.add_argument("--world-strength", type=float, default=0.45)
    p.add_argument("--transparent", action="store_true")
    p.add_argument("--clay", action="store_true",
                   help="เรนเดอร์ชุดเทาล้วนเพิ่มอีกหนึ่งชุด")
    p.add_argument("--no-flip-v", action="store_true",
                   help="ไม่ต้องพลิกแกน V (ใส่เมื่อแผนที่ถูกพลิกมาให้แล้ว)")
    p.add_argument("--keep-material", action="store_true",
                   help="ใช้วัสดุที่ติดมากับไฟล์ ไม่ต้องต่อใหม่จากแผนที่ "
                        "(ไว้เทียบว่าแผนที่ความละเอียดเต็มตรงกับ UV จริงไหม)")
    return p.parse_args(argv)


def main() -> None:
    a = parse_args()
    wipe_scene()

    meshes = import_mesh(a.mesh)
    lo, hi = drop_to_floor(meshes)
    size = hi - lo
    center = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2))
    say(f"ขนาดตัว: กว้าง {size.x:.2f} ลึก {size.y:.2f} สูง {size.z:.2f} m")

    def exists(p: Path | None) -> Path | None:
        if p is None:
            return None
        if not p.exists():
            say(f"ไม่พบไฟล์แผนที่ {p} — ข้าม")
            return None
        return p

    if a.keep_material:
        skin = meshes[0].data.materials[0] if meshes[0].data.materials else None
        say(f"ใช้วัสดุที่ติดมากับไฟล์: {skin.name if skin else '(ไม่มี)'}")
    else:
        skin = build_material("CrocSkin", exists(a.texture), exists(a.normal),
                              exists(a.rough), a.normal_strength,
                              a.rough_floor, flip_v=not a.no_flip_v)
        assign(meshes, skin)

    # ตั้งค่าเรนเดอร์ก่อนวางกล้องเสมอ — ระยะกล้องคำนวณจากอัตราส่วนภาพ
    # ถ้าสลับลำดับจะไปคิดจากค่า default 1920x1080 แล้วถอยกล้องไกลเกินจริง
    eng = setup_render(a.engine, a.width, a.height, a.samples, a.transparent,
                       a.device)

    build_world(a.world_strength)
    build_lights(center, size.z)
    floor = build_floor(center, size.z,
                        shadow_only=a.transparent and eng == "CYCLES")
    if a.transparent and eng != "CYCLES":
        floor.hide_render = True  # EEVEE ไม่มี shadow catcher จริง
    cam, pivot = build_camera(center, size, a.lens, a.margin)

    shots = render_turn(pivot, a.out, a.turns, "turn")

    if a.clay:
        keep = [list(o.data.materials) for o in meshes]
        assign(meshes, clay_material("Clay"))
        shots += render_turn(pivot, a.out, max(4, a.turns // 2), "clay")
        for o, mats in zip(meshes, keep):   # คืนวัสดุเดิมก่อนเก็บไฟล์ .blend
            o.data.materials.clear()
            for m in mats:
                o.data.materials.append(m)

    blend = a.blend or (a.out / f"{a.mesh.stem}.blend")
    blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    say(f"เก็บไฟล์ฉาก -> {blend}")
    say(f"ได้ภาพ {len(shots)} ใบ ที่ {a.out}")
    say("เสร็จสมบูรณ์")


if __name__ == "__main__":
    main()
