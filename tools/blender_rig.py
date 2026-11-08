"""สคริปต์ที่รันใน interpreter ของ Blender — อย่าเรียกจาก venv ปกติ

    blender --background --factory-startup --python blender_rig.py -- \
        --mesh character.glb --joints character_joints.json --out rigged.glb

หน้าที่: อ่าน mesh + ตำแหน่งข้อต่อ สร้าง armature ตามลำดับชั้นแบบ Mixamo
แล้ว bind เข้ากับ mesh ด้วย automatic weights ก่อน export เป็น GLB

เรื่องระบบพิกัดที่ต้องระวัง: glTF เป็น Y-up แต่ Blender เป็น Z-up
ตัว importer ของ Blender แปลง mesh ให้อัตโนมัติ แต่ตำแหน่งข้อต่อใน JSON
ของเรายังเป็น Y-up อยู่ จึงต้องแปลงเองด้วยสูตร (x, y, z) -> (x, -z, y)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# กระดูกที่ควรต่อเป็นแกนหลักของพ่อแต่ละตัว (ตัวอื่นจะแตกเป็นกิ่ง)
PRIMARY_CHILD = {
    "Hips": "Spine",
    "Spine": "Spine1",
    "Spine1": "Spine2",
    "Spine2": "Neck",
    "Neck": "Head",
    "Head": "HeadTop_End",
    "LeftShoulder": "LeftArm",
    "LeftArm": "LeftForeArm",
    "LeftForeArm": "LeftHand",
    "RightShoulder": "RightArm",
    "RightArm": "RightForeArm",
    "RightForeArm": "RightHand",
    "LeftUpLeg": "LeftLeg",
    "LeftLeg": "LeftFoot",
    "LeftFoot": "LeftToeBase",
    "RightUpLeg": "RightLeg",
    "RightLeg": "RightFoot",
    "RightFoot": "RightToeBase",
}

MIN_BONE_LEN = 1e-3


def say(msg: str) -> None:
    """พิมพ์ด้วย prefix ที่ฝั่ง host กรองเก็บไว้แสดง"""
    print(f"[rig] {msg}", flush=True)


def parse_args() -> dict[str, str]:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("ต้องส่งอาร์กิวเมนต์หลัง -- ")
    rest = argv[argv.index("--") + 1:]
    out: dict[str, str] = {}
    for i in range(0, len(rest) - 1, 2):
        out[rest[i].lstrip("-")] = rest[i + 1]
    for key in ("mesh", "joints", "out"):
        if key not in out:
            raise SystemExit(f"ขาดอาร์กิวเมนต์ --{key}")
    return out


def gltf_to_blender(p) -> Vector:
    """(x, y, z) แบบ Y-up ของ glTF -> Z-up ของ Blender"""
    x, y, z = p
    return Vector((x, -z, y))


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path: Path):
    """นำเข้า GLB แล้วคืน object ที่เป็น mesh"""
    for op in ("import_scene.gltf", "wm.gltf_import"):
        mod, name = op.split(".")
        fn = getattr(getattr(bpy.ops, mod), name, None)
        if fn is None:
            continue
        try:
            fn(filepath=str(path))
            break
        except Exception as exc:  # noqa: BLE001
            say(f"ใช้ bpy.ops.{op} ไม่ได้: {exc}")
    else:
        raise SystemExit("หา operator สำหรับ import glTF ไม่เจอใน Blender รุ่นนี้")

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"ไม่พบ mesh ใน {path}")
    obj = max(meshes, key=lambda o: len(o.data.vertices))
    say(f"นำเข้า mesh '{obj.name}' {len(obj.data.vertices)} verts "
        f"{len(obj.data.polygons)} faces")
    return obj


def clean_mesh(obj) -> None:
    """เชื่อมจุดยอดซ้ำและลบเศษลอย ก่อนส่งให้ bone heat คิด weight

    ตัวคลี่ UV ผ่าจุดยอดตามตะเข็บไว้ ไฟล์ที่ได้จึงมีจุดซ้อนกันเป็นหมื่นจุด
    ขอบที่ตะเข็บเลยกลายเป็นขอบเปิดทั้งเส้น ทำให้ mesh ไม่ manifold และ
    bone heat weighting ล้มเงียบ ๆ (คืน weight ศูนย์ทุกจุด) แล้วตกไปใช้
    envelope ซึ่งให้ขาแทบไม่ขยับตามกระดูก

    เชื่อมจุดได้โดยไม่เสีย UV เพราะ Blender เก็บ UV ไว้ที่มุมของหน้า
    ไม่ใช่ที่ตัวจุดยอด ตะเข็บจึงยังอยู่ครบหลังเชื่อม
    """
    bpy.context.view_layer.objects.active = obj
    before = len(obj.data.vertices)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.normals_make_consistent(inside=False)

    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.object.mode_set(mode="OBJECT")
    bad = sum(1 for e in obj.data.edges if e.select)

    # normal ที่อบมากับไฟล์คือค่าเฉลี่ยรายจุดอยู่แล้ว พอเชื่อมจุดเสร็จ
    # การ shade smooth ให้ผลเดียวกัน และไม่ต้องลุ้นว่า custom normal
    # รอดการเชื่อมมาครบหรือไม่
    try:
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    except Exception:  # noqa: BLE001
        pass
    bpy.ops.object.shade_smooth()

    say(f"ทำความสะอาด mesh: {before} -> {len(obj.data.vertices)} จุดยอด, "
        f"ขอบที่ยัง non-manifold {bad}")


def build_armature(joints: dict[str, list[float]], parents: dict[str, str | None]):
    """สร้าง armature จากตำแหน่งข้อต่อ"""
    arm_data = bpy.data.armatures.new("Armature")
    arm_obj = bpy.data.objects.new("Armature", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    pos = {name: gltf_to_blender(p) for name, p in joints.items()}

    # หาลูกของแต่ละกระดูก เพื่อใช้กำหนดปลายกระดูก
    children: dict[str, list[str]] = {}
    for child, parent in parents.items():
        if parent:
            children.setdefault(parent, []).append(child)

    edit_bones = {}
    for name in parents:
        if name not in pos:
            say(f"ข้ามกระดูก {name} เพราะไม่มีตำแหน่งข้อต่อ")
            continue
        eb = arm_data.edit_bones.new(name)
        eb.head = pos[name]

        kids = children.get(name, [])
        primary = PRIMARY_CHILD.get(name)
        if primary and primary in pos:
            eb.tail = pos[primary]
        elif kids:
            eb.tail = sum((pos[k] for k in kids if k in pos), Vector()) / len(kids)
        else:
            # กระดูกปลายทาง: ยืดต่อไปตามทิศทางจากพ่อ
            parent = parents[name]
            direction = (pos[name] - pos[parent]) if parent in pos else Vector((0, 0, 1))
            if direction.length < MIN_BONE_LEN:
                direction = Vector((0, 0, 1))
            eb.tail = pos[name] + direction.normalized() * max(
                direction.length * 0.5, MIN_BONE_LEN * 10
            )

        if (eb.tail - eb.head).length < MIN_BONE_LEN:
            eb.tail = eb.head + Vector((0, 0, MIN_BONE_LEN * 10))
        edit_bones[name] = eb

    # ผูกพ่อลูก แล้วเชื่อมต่อเมื่อหัวลูกอยู่ที่ปลายพ่อพอดี
    for name, eb in edit_bones.items():
        parent = parents.get(name)
        if parent and parent in edit_bones:
            eb.parent = edit_bones[parent]
            eb.use_connect = (eb.head - edit_bones[parent].tail).length < MIN_BONE_LEN

    bpy.ops.armature.select_all(action="SELECT")
    try:
        bpy.ops.armature.calculate_roll(type="GLOBAL_POS_Z")
    except Exception as exc:  # noqa: BLE001
        say(f"คำนวณ roll ไม่สำเร็จ (ไม่กระทบ skinning): {exc}")

    bpy.ops.object.mode_set(mode="OBJECT")
    say(f"สร้าง armature {len(edit_bones)} กระดูก")
    return arm_obj


def bind(mesh_obj, arm_obj) -> str:
    """ผูก mesh เข้ากับ armature พร้อมคิด skinning weight"""
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    for mode in ("ARMATURE_AUTO", "ARMATURE_ENVELOPE"):
        try:
            bpy.ops.object.parent_set(type=mode)
        except RuntimeError as exc:
            say(f"วิธี {mode} ไม่สำเร็จ: {exc}")
        else:
            # ต้องนับ weight จริง ไม่ใช่เชื่อว่า operator ไม่ error
            #
            # bone heat weighting ล้มแบบเงียบ ๆ ได้: มันรายงานความล้มเหลวเป็น
            # report ไม่ใช่ exception แล้วทิ้ง vertex group เปล่า ๆ ไว้ให้ครบทุก
            # กระดูก  ตัว export glTF เจอ weight ศูนย์ทั้งก้อนก็ข้าม skin ไปเลย
            # ได้ไฟล์ที่ดูเหมือนมีกระดูกแต่ตัวละครไม่ขยับตาม และไม่มีใครแจ้งอะไร
            weighted = sum(1 for v in mesh_obj.data.vertices if v.groups)
            if weighted:
                say(f"bind สำเร็จด้วยวิธี {mode} "
                    f"({weighted}/{len(mesh_obj.data.vertices)} vertex มี weight)")
                return mode
            say(f"วิธี {mode} ไม่ให้ weight สักจุด — ถอดแล้วลองวิธีถัดไป")
            bpy.ops.object.select_all(action="DESELECT")
            mesh_obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_obj
            bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
            mesh_obj.vertex_groups.clear()
            for mod in list(mesh_obj.modifiers):
                mesh_obj.modifiers.remove(mod)
            bpy.ops.object.select_all(action="DESELECT")
            mesh_obj.select_set(True)
            arm_obj.select_set(True)
            bpy.context.view_layer.objects.active = arm_obj

        if mode == "ARMATURE_AUTO":
            say("bone heat weighting มักล้มเมื่อ mesh ไม่ manifold "
                "หรือมีชิ้นส่วนลอย — กำลังลองวิธี envelope แทน")
    raise SystemExit("ผูก mesh เข้ากับ armature ไม่สำเร็จทั้งสองวิธี")


def _wave(ch: dict, t: float) -> float:
    """ค่าของ channel หนึ่งที่ตำแหน่ง t (0..1) ในลูป"""
    import math

    mode = ch.get("mode", "sin")
    if mode == "const":
        return ch.get("offset", 0.0)
    x = 2.0 * math.pi * (ch.get("harmonic", 1.0) * t + ch.get("phase", 0.0))
    s = math.sin(x)
    if mode == "halfsin":
        s = max(0.0, s)
    return ch.get("offset", 0.0) + ch.get("amp", 0.0) * s


def _world_rot_to_basis(pb, rot_3x3):
    """แปลงการหมุนที่คิดในพิกัด armature ให้เป็น rotation ของกระดูกนั้น

    ท่าทางเขียนง่ายที่สุดเมื่อคิดเป็นแกนโลก ("ยกต้นขาไปข้างหน้า" = หมุนรอบ X)
    แต่ Blender เก็บท่าของกระดูกเป็นค่าสัมพัทธ์ในระบบแกนของกระดูกเอง ซึ่งเอียง
    ไปตามทิศที่กระดูกชี้ สูตร M⁻¹ R M คือการย้ายฐานของการหมุนระหว่างสองระบบนั้น
    """
    from mathutils import Matrix  # noqa: F401  (ใช้ชนิดจากผู้เรียก)

    M = pb.bone.matrix_local.to_3x3()
    return (M.inverted() @ rot_3x3 @ M).to_quaternion()


def apply_clips(arm_obj, payload: dict) -> list[str]:
    """สร้าง action หนึ่งตัวต่อหนึ่ง clip แล้วเก็บเข้า NLA เพื่อให้ export ครบ"""
    import math

    from mathutils import Matrix, Vector

    fps = int(payload.get("fps", 24))
    scene = bpy.context.scene
    scene.render.fps = fps

    pose = arm_obj.pose
    for pb in pose.bones:
        pb.rotation_mode = "QUATERNION"

    arm_obj.animation_data_create()
    made: list[str] = []

    for clip in payload.get("clips", []):
        name = clip["name"]
        frames = max(2, int(clip.get("frames", 32)))
        channels = clip.get("channels", [])

        used = sorted({c["bone"] for c in channels if c["bone"] in pose.bones})
        missing = sorted({c["bone"] for c in channels} - set(used))
        if missing:
            say(f"clip {name}: ข้ามกระดูกที่ไม่มีในโครง {missing}")
        if not used:
            say(f"clip {name}: ไม่มีกระดูกให้ขยับเลย ข้าม clip นี้")
            continue

        action = bpy.data.actions.new(name)
        action.use_fake_user = True
        arm_obj.animation_data.action = action
        _ensure_action_slot(arm_obj, action)

        for f in range(frames + 1):
            # เฟรมสุดท้ายซ้ำกับเฟรมแรกพอดี ลูปจึงต่อเนื่องไม่มีสะดุด
            t = (f % frames) / frames

            rot: dict[str, object] = {}
            loc: dict[str, object] = {}
            for ch in channels:
                bone = ch["bone"]
                if bone not in pose.bones:
                    continue
                val = _wave(ch, t)
                if ch.get("kind", "rot") == "loc":
                    v = loc.setdefault(bone, Vector((0.0, 0.0, 0.0)))
                    v["XYZ".index(ch.get("axis", "X"))] += val
                else:
                    R = Matrix.Rotation(math.radians(val), 3, ch.get("axis", "X"))
                    rot[bone] = R @ rot[bone] if bone in rot else R

            for bone in used:
                pb = pose.bones[bone]
                pb.rotation_quaternion = (
                    _world_rot_to_basis(pb, rot[bone]) if bone in rot
                    else (1.0, 0.0, 0.0, 0.0)
                )
                if bone in loc:
                    pb.location = pb.bone.matrix_local.to_3x3().inverted() @ loc[bone]
                else:
                    pb.location = (0.0, 0.0, 0.0)

                pb.keyframe_insert("rotation_quaternion", frame=f + 1)
                pb.keyframe_insert("location", frame=f + 1)

        for fc in _action_fcurves(action):
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"

        # เก็บเข้า NLA เพื่อให้ตัว export เห็นทุก clip ไม่ใช่แค่ action ที่ active
        track = arm_obj.animation_data.nla_tracks.new()
        track.name = name
        track.strips.new(name, 1, action)
        arm_obj.animation_data.action = None

        made.append(name)
        say(f"clip {name}: {frames} เฟรม @ {fps} fps, {len(used)} กระดูก")

    scene.frame_start = 1
    scene.frame_end = max((int(c.get("frames", 32)) + 1
                           for c in payload.get("clips", [])), default=1)
    return made


def _action_fcurves(action) -> list:
    """คืน fcurve ทั้งหมดของ action โดยรองรับทั้ง Blender เก่าและใหม่

    Blender 4.4 เปลี่ยน action เป็นระบบ layer/strip/slot ทำให้ `action.fcurves`
    ที่เคยมีหายไป fcurve ย้ายไปอยู่ใน channelbag ของแต่ละ slot แทน
    """
    fcs = getattr(action, "fcurves", None)
    if fcs is not None:
        return list(fcs)

    out: list = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                out.extend(bag.fcurves)
    return out


def _ensure_action_slot(obj, action) -> None:
    """Blender 4.4+ ใช้ระบบ slot ใน action ต้องผูก slot ให้ด้วยจึงจะ keyframe ได้"""
    ad = obj.animation_data
    if not hasattr(ad, "action_slot"):
        return                      # Blender รุ่นเก่า ไม่มีระบบ slot
    try:
        if getattr(ad, "action_slot", None) is None:
            slots = getattr(action, "slots", None)
            slot = slots[0] if slots else action.slots.new(id_type="OBJECT",
                                                           name=obj.name)
            ad.action_slot = slot
    except Exception as exc:  # noqa: BLE001
        say(f"ผูก action slot ไม่สำเร็จ (อาจไม่กระทบ): {exc}")


def export_glb(path: Path, draco: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = dict(filepath=str(path), export_format="GLB")

    # ไล่จากพารามิเตอร์ครบที่สุดไปน้อยที่สุด เพราะชื่อพารามิเตอร์ของตัว export
    # glTF เปลี่ยนไปมาระหว่าง Blender แต่ละรุ่น
    # export_animation_mode="NLA_TRACKS" คือตัวที่ทำให้ได้ *ทุก* clip ไม่ใช่แค่
    # action ที่ active อยู่ เพราะเราเก็บทุก clip ไว้เป็น NLA track
    # Draco บีบ geometry ได้ราว 4-8 เท่า แต่ไฟล์ที่ได้ต้องมี decoder ฝั่งผู้อ่าน
    # three.js ต้องตั้ง DRACOLoader เพิ่ม (หน้าเว็บในโปรเจกต์นี้ตั้งไว้ให้แล้ว)
    dr = {
        "export_draco_mesh_compression_enable": True,
        "export_draco_mesh_compression_level": 6,
        "export_draco_position_quantization": 14,
        "export_draco_normal_quantization": 10,
        "export_draco_texcoord_quantization": 12,
    } if draco else {}

    attempts = [
        {"export_skins": True, "export_apply": False,
         "export_animations": True, "export_animation_mode": "NLA_TRACKS",
         "export_frame_range": False, "export_current_frame": False, **dr},
        {"export_skins": True, "export_animations": True,
         "export_animation_mode": "ACTIONS"},
        {"export_skins": True, "export_animations": True},
        {},
    ]
    for extra in attempts:
        try:
            bpy.ops.export_scene.gltf(**base, **extra)
            say(f"export -> {path}  (พารามิเตอร์: {sorted(extra) or 'ค่าเริ่มต้น'})")
            return
        except TypeError as exc:
            say(f"export ด้วย {sorted(extra)} ไม่ได้: {exc}")
    raise SystemExit("export GLB ไม่สำเร็จ")


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args["joints"]).read_text(encoding="utf-8"))
    joints = payload["joints"]
    parents = payload["parents"]

    reset_scene()
    mesh_obj = import_glb(Path(args["mesh"]))
    clean_mesh(mesh_obj)
    arm_obj = build_armature(joints, parents)
    bind(mesh_obj, arm_obj)

    anim_path = args.get("anim")
    if anim_path:
        clips = json.loads(Path(anim_path).read_text(encoding="utf-8"))
        made = apply_clips(arm_obj, clips)
        say(f"สร้างอนิเมชัน {len(made)} clip: {', '.join(made)}")

    # ตรวจก่อน export ว่า mesh ผูกกับ armature จริง ไม่ใช่แค่ operator ไม่ error
    #
    # เคยเจอกรณีที่ parent_set คืนค่าสำเร็จ แต่ไฟล์ที่ได้ไม่มี skin เลย ซึ่งเงียบ
    # สนิทจนกว่าจะเปิดดูแล้วพบว่าตัวละครไม่ขยับตามกระดูก
    weighted = sum(1 for v in mesh_obj.data.vertices if v.groups)
    deform = sum(1 for b in arm_obj.data.bones if b.use_deform)
    say(f"ตรวจ weight: vertex ที่มี weight {weighted}/{len(mesh_obj.data.vertices)}, "
        f"กระดูกที่ deform ได้ {deform}/{len(arm_obj.data.bones)}")
    mods = [m.type for m in mesh_obj.modifiers]
    say(f"ก่อน export: parent={getattr(mesh_obj.parent, 'name', None)} "
        f"modifiers={mods} vertex_groups={len(mesh_obj.vertex_groups)}")
    if "ARMATURE" not in mods:
        say("mesh ยังไม่มี armature modifier — ใส่ให้เองแล้วผูก parent")
        mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = arm_obj
        mesh_obj.parent = arm_obj

    export_glb(Path(args["out"]), args.get("draco") == "1")
    say("เสร็จสมบูรณ์")


if __name__ == "__main__":
    main()
