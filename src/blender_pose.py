r"""หมุนโมเดลฐานจากท่าของมันเองไปเป็นท่าของตัวละคร (รันใน Blender เท่านั้น)

ทำไมต้องผ่าน Blender
--------------------
`basefit.tps_warp` ดัดสัดส่วนได้ดี แต่ *หมุนท่อนกระดูกไม่ได้* เพราะสนามการ
เปลี่ยนรูปที่เรียบทำให้จุดสองฝั่งของท่อนวิ่งสวนทางกันไม่ได้ โมเดลฐานที่เป็น
T-pose จึงดัดไปเป็น A-pose ไม่ได้ (วัดได้ว่าต่างกัน 63 องศาที่ปลายแขน)

การหมุนต้องใช้ armature จริง และหัวใจของมันคือ *skinning weight* ที่ดี
Blender มี bone heat weighting ให้อยู่แล้ว ซึ่งดีกว่าน้ำหนักตามระยะทางที่
เขียนเองมาก โดยเฉพาะตรงรักแร้กับขาหนีบที่กระดูกสองท่อนอยู่ใกล้กัน

ขั้นตอน
-------
1. นำเข้าโมเดลฐานที่ normalize แล้ว (ฝั่ง Python ทำมาให้ เป็นเมตร +Y ขึ้น)
2. สร้าง armature ที่ข้อต่อ *ของโมเดลฐานเอง* แล้ว bind ด้วย automatic weights
3. หมุนทีละกระดูกจากโคนไปปลาย ให้ทิศของมันตรงกับทิศของกระดูกเดียวกัน
   ในโครงของตัวละคร — หมุนอย่างเดียว ไม่ยืด ความยาวที่ต่างกันปล่อยให้
   ขั้น TPS ฝั่ง Python เก็บทีหลัง
4. apply modifier แล้ว export เป็น mesh นิ่ง ๆ ที่อยู่ในท่าใหม่

    blender --background --factory-startup --python src/blender_pose.py -- \
        --mesh base.glb --src src_joints.json --dst dst_joints.json --out posed.glb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Matrix, Vector  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_rig as rig  # noqa: E402


def say(msg: str) -> None:
    print(f"[pose] {msg}", flush=True)


def blender_to_gltf(v) -> list[float]:
    """Z-up ของ Blender -> (x, y, z) แบบ Y-up ของ glTF (ผกผันของ gltf_to_blender)"""
    return [float(v.x), float(v.z), float(-v.y)]


def parse_args() -> dict[str, str]:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("ต้องส่งอาร์กิวเมนต์หลัง -- ")
    rest = argv[argv.index("--") + 1:]
    out = {rest[i].lstrip("-"): rest[i + 1] for i in range(0, len(rest) - 1, 2)}
    for key in ("mesh", "src", "dst", "out"):
        if key not in out:
            raise SystemExit(f"ขาดอาร์กิวเมนต์ --{key}")
    return out


def _bone_dir(name: str, joints: dict[str, Vector],
              children: dict[str, list[str]]) -> Vector | None:
    """ทิศที่กระดูกชี้ไป — ต้องใช้กฎเดียวกับตอนสร้าง armature เป๊ะ ๆ

    ไม่งั้นทิศ "ปัจจุบัน" กับทิศ "เป้าหมาย" จะวัดจากคนละนิยาม แล้วมุมที่
    คำนวณได้จะผิดไปทั้งตัว
    """
    primary = rig.PRIMARY_CHILD.get(name)
    if primary and primary in joints:
        tip = joints[primary]
    else:
        kids = [k for k in children.get(name, []) if k in joints]
        if not kids:
            return None
        tip = sum((joints[k] for k in kids), Vector()) / len(kids)
    d = tip - joints[name]
    return d.normalized() if d.length > 1e-6 else None


def pose_to(arm_obj, src: dict, dst: dict, parents: dict) -> int:
    """หมุนกระดูกให้ทิศตรงกับโครงของตัวละคร คืนจำนวนกระดูกที่หมุนจริง"""
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")

    src_v = {k: rig.gltf_to_blender(v) for k, v in src.items()}
    dst_v = {k: rig.gltf_to_blender(v) for k, v in dst.items()}
    children: dict[str, list[str]] = {}
    for child, parent in parents.items():
        if parent:
            children.setdefault(parent, []).append(child)

    # เรียงจากโคนไปปลาย: ต้องหมุนพ่อก่อน ไม่งั้นลูกจะคำนวณจากท่าที่ยังไม่อัปเดต
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        parent = parents.get(name)
        if parent and parent in parents:
            visit(parent)
        seen.add(name)
        order.append(name)

    for name in parents:
        visit(name)

    turned = 0
    scaled = 0
    worst = 0.0
    for name in order:
        pb = arm_obj.pose.bones.get(name)
        if pb is None or name not in dst_v:
            continue
        want = _bone_dir(name, dst_v, children)
        if want is None:
            continue

        cur = (pb.tail - pb.head)
        if cur.length < 1e-6:
            continue
        cur = cur.normalized()
        ang = cur.angle(want, 0.0)
        if ang < 1e-4:
            continue

        # หมุนรอบแกนที่สั้นที่สุด แล้วคูณทับท่าปัจจุบันในพิกัด armature
        rot = cur.rotation_difference(want)
        m = pb.matrix.copy()
        head = m.translation.copy()
        pb.matrix = (Matrix.Translation(head)
                     @ (rot @ m.to_quaternion()).to_matrix().to_4x4())
        # ต้องอัปเดตทันที ไม่งั้นกระดูกลูกจะอ่าน matrix ของพ่อที่ยังไม่ขยับ
        bpy.context.view_layer.update()

        # ยืด/หดกระดูกให้ยาวเท่าของตัวละครด้วย
        #
        # ถ้าหมุนอย่างเดียว ความยาวยังเป็นของโมเดลฐาน ข้อต่อปลายสายจึงไป
        # ไม่ถึงเป้า แล้วความคลาดสะสมลงไปทั้งโซ่ (ข้อศอกเลยเป้า ข้อมือยิ่ง
        # เลยไปอีก) ขั้น TPS ต้องมาดึงระยะไกล ๆ ซึ่งทำให้รูปทรงยืดเพี้ยน
        #
        # ใช้สเกลเท่ากันทุกแกน ไม่ใช่เฉพาะแกนยาว เพื่อไม่ให้หน้าตัดถูกบีบ
        # จนเนื้อบิด (ความหนาที่ถูกต้องให้ขั้น fit_volume จัดการทีหลัง)
        want_len = (dst_v[rig.PRIMARY_CHILD.get(name, name)] - dst_v[name]).length             if rig.PRIMARY_CHILD.get(name) in dst_v else None
        if want_len and want_len > 1e-6:
            have = (pb.tail - pb.head).length
            if have > 1e-6:
                f = max(0.2, min(5.0, want_len / have))
                pb.scale = pb.scale * f
                bpy.context.view_layer.update()
                scaled += 1

        got = (pb.tail - pb.head)
        err = got.normalized().angle(want, 0.0) * 57.2958 if got.length > 1e-9 else -1
        if err > 1.0:
            say(f"  !! {name}: ตั้งทิศไม่ติด เหลือคลาด {err:.0f} องศา")
        turned += 1
        worst = max(worst, ang)

    bpy.ops.object.mode_set(mode="OBJECT")
    say(f"หมุนกระดูก {turned} ท่อน (มากสุด {worst * 57.2958:.0f} องศา) "
        f"ยืด/หด {scaled} ท่อน")
    return turned


def _weighted_count(mesh_obj) -> int:
    return sum(1 for v in mesh_obj.data.vertices if v.groups)


def bind_checked(mesh_obj, arm_obj) -> str:
    """bind แล้ว *ตรวจว่าได้น้ำหนักจริง* ไม่ใช่เชื่อว่า operator สำเร็จ

    bone heat weighting ล้มแบบเงียบได้: operator คืนค่าสำเร็จ สร้าง vertex
    group ครบทุกกระดูก แต่ไม่ใส่น้ำหนักให้ vertex สักจุด (เจอจริงกับโมเดล
    ที่ไม่ manifold — ได้ 0/21692) แล้วการ pose ก็ไม่ขยับอะไรโดยไม่มี error
    ต้องวัดผลลัพธ์เอง แล้วถอยไปวิธีถัดไปถ้าไม่ได้ของ
    """
    for mode in ("ARMATURE_AUTO", "ARMATURE_ENVELOPE"):
        bpy.ops.object.select_all(action="DESELECT")
        mesh_obj.select_set(True)
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
        try:
            bpy.ops.object.parent_set(type=mode)
        except RuntimeError as exc:
            say(f"วิธี {mode} ไม่สำเร็จ: {exc}")
            continue
        n = _weighted_count(mesh_obj)
        if n > 0.5 * len(mesh_obj.data.vertices):
            say(f"bind ด้วย {mode}: {n}/{len(mesh_obj.data.vertices)} vertex มีน้ำหนัก")
            return mode
        say(f"วิธี {mode} ได้น้ำหนักแค่ {n}/{len(mesh_obj.data.vertices)} — ลองวิธีถัดไป")

    weight_by_distance(mesh_obj, arm_obj)
    return "DISTANCE"


def weight_by_distance(mesh_obj, arm_obj) -> None:
    """ใส่น้ำหนักเองจากระยะถึงแกนกระดูก — ใช้เมื่อวิธีของ Blender ไม่ได้ผล

    คุณภาพสู้ heat diffusion ไม่ได้ (ตรงรักแร้กับขาหนีบจะแข็งกว่า) แต่ *ได้
    ของเสมอ* ซึ่งสำคัญกว่าสำหรับงานปรับสัดส่วนที่มุมการหมุนไม่มาก

    ใช้น้ำหนักแบบต่อเนื่อง: หยิบกระดูกใกล้สุด k+1 ท่อนแล้วลบน้ำหนักของท่อน
    ที่ k+1 ออก น้ำหนักของท่อนที่กำลังจะหลุดอันดับจึงไต่ลงถึง 0 ก่อนหลุด
    ไม่งั้นผิวจะฉีกตรงที่ชุดกระดูกเปลี่ยน
    """
    import mathutils  # type: ignore

    deform = [b for b in arm_obj.data.bones if b.use_deform]
    segs = [(b.name, b.head_local.copy(), b.tail_local.copy()) for b in deform]
    if not segs:
        raise SystemExit("armature ไม่มีกระดูกที่ใช้ deform")

    for vg in list(mesh_obj.vertex_groups):
        mesh_obj.vertex_groups.remove(vg)
    groups = {name: mesh_obj.vertex_groups.new(name=name) for name, _, _ in segs}

    k = min(4, len(segs) - 1) if len(segs) > 1 else 1
    mw = mesh_obj.matrix_world
    inv = arm_obj.matrix_world.inverted()

    for v in mesh_obj.data.vertices:
        p = inv @ (mw @ v.co)
        d = []
        for name, a, b in segs:
            ab = b - a
            t = 0.0 if ab.length_squared < 1e-12 else \
                max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
            d.append((max((p - (a + t * ab)).length, 1e-4), name))
        d.sort()
        near = d[: k + 1]
        raw = [(1.0 / dist ** 3, name) for dist, name in near]
        cut = min(w for w, _ in raw)
        trimmed = [(max(w - cut, 0.0), name) for w, name in raw]
        total = sum(w for w, _ in trimmed)
        if total <= 1e-12:
            trimmed = [(1.0, near[0][1])]
            total = 1.0
        for w, name in trimmed:
            if w > 0:
                groups[name].add([v.index], w / total, "REPLACE")

    # ผูก parent แบบธรรมดาแล้วใส่ modifier เอง (parent_set จะไปลบน้ำหนักทิ้ง)
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
    if not any(m.type == "ARMATURE" for m in mesh_obj.modifiers):
        mod = mesh_obj.modifiers.new("Armature", "ARMATURE")
        mod.object = arm_obj
    say(f"ใส่น้ำหนักเองจากระยะ: {_weighted_count(mesh_obj)}/"
        f"{len(mesh_obj.data.vertices)} vertex")


def _report(mesh_obj, label: str) -> None:
    """ขนาดกรอบของ mesh *ตามที่ประเมินจริง* (รวมผลของ modifier)

    ต้องอ่านผ่าน depsgraph ไม่ใช่ `mesh_obj.data` ตรง ๆ เพราะข้อมูลดิบยังเป็น
    ท่า rest อยู่เสมอ การดูแต่ข้อมูลดิบทำให้เข้าใจผิดว่าการ pose ไม่ทำงาน
    """
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh_obj.evaluated_get(dg)
    me = ev.to_mesh()
    if not me.vertices:
        say(f"{label}: ไม่มี vertex")
        return
    co = [mesh_obj.matrix_world @ v.co for v in me.vertices]
    lo = Vector((min(c.x for c in co), min(c.y for c in co), min(c.z for c in co)))
    hi = Vector((max(c.x for c in co), max(c.y for c in co), max(c.z for c in co)))
    say(f"{label}: {len(me.vertices)} verts  ขนาด "
        f"({hi.x - lo.x:.2f}, {hi.y - lo.y:.2f}, {hi.z - lo.z:.2f})")
    ev.to_mesh_clear()


def posed_joints(arm_obj, parents: dict) -> dict:
    """ตำแหน่งข้อต่อ *หลังหมุนแล้ว* อ่านจาก armature ตรง ๆ

    สำคัญกว่าที่คิด: เดิมฝั่ง Python เอา mesh ที่ posed ไป voxelize แล้วเรียก
    ตัวประเมินข้อต่อใหม่ ซึ่งให้คำตอบ *ไม่ตรง* กับที่ armature วางไว้จริง
    (วัดได้ว่าเหลือต่างจากเป้า 60 องศาทั้งที่หมุนไปครบแล้ว) ความคลาดนั้นเป็น
    ของการ *วัด* ไม่ใช่ของการหมุน แล้วขั้น TPS ถัดไปก็ไปดัดแก้สิ่งที่ไม่ได้ผิด

    armature รู้คำตอบอยู่แล้ว จึงอ่านจากมันโดยตรง ไม่ต้องเดาซ้ำ
    """
    out = {}
    for name in parents:
        pb = arm_obj.pose.bones.get(name)
        if pb is None:
            continue
        out[name] = blender_to_gltf(arm_obj.matrix_world @ pb.head)
    # ปลายสาย: ใช้ปลายกระดูกเป็นตำแหน่งของข้อต่อปลาย
    kids = {p for p in parents.values() if p}
    for name in parents:
        if name in kids:
            continue
        pb = arm_obj.pose.bones.get(name)
        if pb is not None:
            out[name] = blender_to_gltf(arm_obj.matrix_world @ pb.head)
    return out


def apply_pose(mesh_obj, arm_obj) -> None:
    """อบท่าใหม่ลงใน vertex แล้วทิ้ง armature

    ต้องการ mesh นิ่ง ๆ ที่อยู่ในท่าใหม่ ไม่ใช่ mesh + armature เพราะขั้นถัดไป
    (TPS ฝั่ง Python) ทำงานกับตำแหน่ง vertex ตรง ๆ และ pipeline จะ rig ใหม่
    ด้วยโครงของตัวละครเองอยู่แล้วตอนท้าย
    """
    bpy.context.view_layer.objects.active = mesh_obj
    weighted = sum(1 for v in mesh_obj.data.vertices if v.groups)
    say(f"vertex groups บน mesh: {len(mesh_obj.vertex_groups)} กลุ่ม, "
        f"vertex ที่มีน้ำหนักจริง {weighted}/{len(mesh_obj.data.vertices)}, "
        f"modifiers: {[m.type for m in mesh_obj.modifiers]}")
    mods = [m for m in mesh_obj.modifiers if m.type == "ARMATURE"]
    if not mods:
        raise SystemExit("ไม่เจอ armature modifier บน mesh — bind ไม่สำเร็จ")
    for m in mods:
        bpy.ops.object.modifier_apply(modifier=m.name)

    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    bpy.ops.object.delete()
    say("อบท่าลง mesh และลบ armature แล้ว")


def main() -> None:
    args = parse_args()
    rig.reset_scene()

    mesh_obj = rig.import_glb(Path(args["mesh"]))

    src = json.loads(Path(args["src"]).read_text(encoding="utf-8"))
    dst = json.loads(Path(args["dst"]).read_text(encoding="utf-8"))
    parents = src["parents"]

    _report(mesh_obj, "หลังนำเข้า")
    arm_obj = rig.build_armature(src["joints"], parents)
    bind_checked(mesh_obj, arm_obj)
    _report(mesh_obj, "หลัง bind")
    pose_to(arm_obj, src["joints"], dst["joints"], parents)
    _report(mesh_obj, "หลังหมุนท่า")

    out_joints = posed_joints(arm_obj, parents)
    jpath = args.get("outjoints")
    if jpath:
        Path(jpath).write_text(json.dumps({"joints": out_joints},
                                          ensure_ascii=False), encoding="utf-8")
        say(f"เขียนข้อต่อหลังหมุน {len(out_joints)} จุด")

    apply_pose(mesh_obj, arm_obj)
    _report(mesh_obj, "หลัง apply")

    rig.export_glb(Path(args["out"]), False)
    say("เสร็จสมบูรณ์")


if __name__ == "__main__":
    main()
