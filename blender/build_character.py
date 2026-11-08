"""Build the crocodilian character and write .blend, .glb and a turnaround.

    blender -b --python blender/build_character.py -- [--no-render] [--no-clothes]

Every part is generated from the proportion table in croclib/proportions.py,
which was measured off the reference sheets in input/albedo.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bpy  # noqa: E402

from croclib import body, extremities, head, materials, meshutil as mu, scene  # noqa: E402

try:
    from croclib import clothing
except ImportError:  # built up in stages; the body stands on its own
    clothing = None

OUT = os.path.join(ROOT, "output")


def argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def make(name: str, geom, mat, *, mirror: bool = False, subsurf: int = 0):
    verts, faces = geom
    if not verts:
        return None
    obj = mu.new_object(name, verts, faces)
    mu.recalc_normals(obj)
    if mirror:
        mu.mirror_x(obj)
    if subsurf:
        mu.add_subsurf(obj, subsurf)
    if mirror or subsurf:
        mu.apply_modifiers(obj)
    mu.set_material(obj, mat)
    return obj


def build_skin(mats) -> list[bpy.types.Object]:
    hand_flesh, hand_claws = extremities.hand()
    foot_flesh, foot_claws = extremities.foot()
    eyeballs, eye_hoods = head.eyes()

    skin = mats["skin"]
    hide = mats["hide"]

    parts = [
        make("Torso", body.torso(), skin, subsurf=2),
        make("Neck", body.neck(), skin, subsurf=2),
        make("Arm", body.arm(), skin, mirror=True, subsurf=2),
        make("Leg", body.leg(), skin, mirror=True, subsurf=2),
        make("Hand", hand_flesh, skin, mirror=True, subsurf=2),
        make("Foot", foot_flesh, skin, mirror=True, subsurf=2),
        make("Cranium", head.cranium(), skin, subsurf=2),
        make("Jaw", head.jaw(), skin, subsurf=2),
        make("EyeHood", eye_hoods, skin, subsurf=2),
        make("Tail", body.tail(), hide, subsurf=2),
        make("TailScutes", body.tail_scutes(), hide, subsurf=1),
        make("ArmScutes", body.forearm_spikes(), hide, mirror=True, subsurf=1),
        make("Crest", head.crest(), hide, subsurf=1),
        make("Nostrils", head.nostrils(), skin, subsurf=1),
    ]
    return [p for p in parts if p]


def build_hard(mats) -> list[bpy.types.Object]:
    _, hand_claws = extremities.hand()
    _, foot_claws = extremities.foot()
    eyeballs, _ = head.eyes()
    return [p for p in (
        make("HandClaws", hand_claws, mats["claw"], mirror=True, subsurf=1),
        make("FootClaws", foot_claws, mats["claw"], mirror=True, subsurf=1),
        make("Teeth", head.teeth(), mats["bone"], subsurf=1),
        make("Eyes", eyeballs, mats["eye"], subsurf=1),
    ) if p]


def main() -> None:
    args = argv()
    want_render = "--no-render" not in args
    want_clothes = "--no-clothes" not in args and clothing is not None

    scene.clear()
    scene.setup_world()
    scene.setup_lights()
    scene.add_ground()
    mats = materials.build()

    objects: list[bpy.types.Object] = []
    objects += build_skin(mats)
    objects += build_hard(mats)
    if want_clothes:
        objects += clothing.build(mats, make)

    for obj in objects:
        mu.shade_smooth(obj)

    character = mu.join(objects, "Croc")
    character.rotation_mode = "XYZ"

    os.makedirs(OUT, exist_ok=True)
    tris = sum(len(p.vertices) - 2 for p in character.data.polygons)
    print(f"[build] {len(character.data.vertices)} verts, ~{tris} tris")

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "croc.blend"))

    bpy.ops.object.select_all(action="DESELECT")
    character.select_set(True)
    bpy.context.view_layer.objects.active = character
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(OUT, "croc.glb"),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )
    print("[build] wrote", os.path.join(OUT, "croc.glb"))

    if want_render:
        scene.setup_render()
        cam = scene.setup_camera()
        for label, angle in (("front", 0), ("right", 90), ("back", 180), ("left", 270)):
            scene.place_camera(cam, angle)
            scene.render_to(os.path.join(OUT, "turnaround", f"{label}.png"))
            print("[render]", label)


if __name__ == "__main__":
    main()
