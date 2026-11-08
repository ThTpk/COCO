"""Scene plumbing: clearing, lighting, cameras, rendering."""

from __future__ import annotations

import math
import os

import bpy
from mathutils import Vector

from .proportions import HEIGHT


def clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.objects,
                  bpy.data.cameras, bpy.data.lights, bpy.data.images):
        for item in list(block):
            block.remove(item)


def pick_engine() -> str:
    """EEVEE under whichever id this Blender build uses, else Cycles."""
    available = {
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    }
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if name in available:
            return name
    return "CYCLES"


def setup_world(strength: float = 0.35) -> None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.055, 0.058, 0.065, 1.0)
    bg.inputs["Strength"].default_value = strength


def _lamp(name, kind, location, energy, size=1.0, rotation=(0, 0, 0)):
    data = bpy.data.lights.new(name, kind)
    data.energy = energy
    if kind == "AREA":
        data.size = size
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    obj.rotation_euler = rotation
    bpy.context.scene.collection.objects.link(obj)
    return obj


def setup_lights() -> None:
    """Three-point rig scaled to the character, aimed at chest height."""
    h = HEIGHT
    # Watts, at roughly 4 m: the first pass used stage-sized numbers and
    # clipped every surface to white, which hid the shading entirely.
    key = _lamp("Key", "AREA", (-2.4, -3.2, h * 1.15), 420, size=3.0)
    fill = _lamp("Fill", "AREA", (3.4, -2.2, h * 0.75), 150, size=4.0)
    rim = _lamp("Rim", "AREA", (1.0, 3.6, h * 1.3), 320, size=2.5)
    for lamp, target_z in ((key, h * 0.60), (fill, h * 0.50), (rim, h * 0.70)):
        _aim(lamp, Vector((0.0, 0.0, target_z)))


def _aim(obj, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(name: str = "Cam") -> bpy.types.Object:
    data = bpy.data.cameras.new(name)
    data.type = "ORTHO"
    data.ortho_scale = HEIGHT * 1.22
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj
    return obj


def place_camera(cam: bpy.types.Object, angle_deg: float, *,
                 distance: float = 8.0, height: float | None = None) -> None:
    """Orbit the camera around the character at a given yaw.

    0 deg looks at the front (from -Y), 90 deg at the character's right.
    """
    z = HEIGHT * 0.52 if height is None else height
    a = math.radians(angle_deg)
    cam.location = Vector((math.sin(a) * distance, -math.cos(a) * distance, z))
    _aim(cam, Vector((0.0, 0.0, z)))


def setup_render(width: int = 900, height: int = 1200, samples: int = 32) -> None:
    scene = bpy.context.scene
    scene.render.engine = pick_engine()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    available = {
        v.identifier for v in
        bpy.types.ColorManagedViewSettings.bl_rna.properties["view_transform"].enum_items
    }
    for transform in ("AgX", "Filmic", "Standard"):
        if transform in available:
            scene.view_settings.view_transform = transform
            break
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    else:
        eevee = scene.eevee
        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = max(samples, 32)
        for flag in ("use_gtao", "use_raytracing", "use_shadows"):
            if hasattr(eevee, flag):
                setattr(eevee, flag, True)


def render_to(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.context.scene.render.filepath = path
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


def add_ground(size: float = 12.0) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("Ground")
    s = size * 0.5
    mesh.from_pydata(
        [(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0)], [], [(0, 1, 2, 3)]
    )
    mesh.update()
    obj = bpy.data.objects.new("Ground", mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.055, 0.055, 0.058, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    obj.data.materials.append(mat)
    return obj
