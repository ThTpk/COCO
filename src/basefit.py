"""ดัดโมเดลฐานให้เข้ากับสัดส่วนของตัวละคร

แนวคิด
------
`visual_hull` สร้างรูปทรงจากศูนย์ มันจึงได้แต่ "ก้อนที่เงาถูก" และพังเสมอตรง
จุดที่เงาบอกไม่ได้ — นิ้วมือ นิ้วเท้า คอ ไม่ว่าจะเพิ่มความละเอียดแค่ไหน

โมดูลนี้พลิกโจทย์: เอาโมเดลที่ *ถูกต้องอยู่แล้ว* มาดัดให้ตรงกับเงา แทนที่จะ
สร้างจากเงา สิ่งที่ hull เดาไม่ได้จึงไม่ต้องเดา เพราะมันมีมาในโมเดลฐานแล้ว

ขั้นตอน
-------
1. หาข้อต่อของโมเดลฐานด้วย *ตัวประเมินตัวเดียวกับที่ใช้กับ hull*
   (voxelize โมเดลฐานแล้วเรียก `landmarks.estimate`) — ได้ข้อต่อชื่อเดียวกัน
   ทั้งสองฝั่งโดยอัตโนมัติ ไม่ต้องทำ correspondence เอง ซึ่งเป็นส่วนที่ยาก
   ที่สุดของงาน fit โมเดล
2. จับคู่กระดูกแล้วหา transform (หมุน + ย่อขยาย + เลื่อน) ของแต่ละท่อน
3. ดัด vertex ด้วย linear blend skinning โดยถ่วงน้ำหนักตามระยะถึงกระดูก

ขั้นที่ยังไม่ทำที่นี่: ดัดละเอียดให้ชนเงาทั้ง 4 มุม (อยู่ใน `refine_to_hull`)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .landmarks import BONE_PARENTS

log = logging.getLogger(__name__)

#: กระดูกที่ใช้ดัด — ข้ามหางเพราะโมเดลฐานคนไม่มี และข้ามปลายกระดูก
FIT_BONES = (
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"),
    ("Spine2", "Neck"), ("Neck", "Head"), ("Head", "HeadTop_End"),
    ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase"),
)


def base_joints(base, cfg: PipelineConfig) -> tuple[dict[str, np.ndarray], list[str]]:
    """หาข้อต่อของโมเดลฐาน คืนตำแหน่งเป็น *เมตร* ในระบบพิกัดเดียวกับ base.verts

    ใช้ตัวประเมินตัวเดียวกับที่ใช้กับ hull จึงได้ชื่อข้อต่อชุดเดียวกันเป๊ะ
    ซึ่งทำให้การจับคู่กระดูกระหว่างโมเดลฐานกับตัวละครเป็นเรื่องอัตโนมัติ
    """
    from . import basemesh, landmarks, visual_hull

    pad = 3
    pitch = cfg.character_height_m / cfg.voxel_res
    lo = base.verts.min(axis=0) - pad * pitch
    hi = base.verts.max(axis=0) + pad * pitch
    ext = hi - lo
    shape = (max(int(np.ceil(ext[1] / pitch)), 4),
             max(int(np.ceil(ext[0] / pitch)), 4),
             max(int(np.ceil(ext[2] / pitch)), 4))

    occ = basemesh.solidify(base.verts, base.faces, shape, lo, pitch)
    if not occ.any():
        raise RuntimeError(
            "แปลงโมเดลฐานเป็น voxel แล้วไม่เหลือเนื้อเลย — ไฟล์อาจเป็นผิวเปิด "
            "ทั้งก้อน หรือหน่วยผิดจนเล็กกว่า voxel เดียว"
        )

    grid = visual_hull.HullGrid(occ.astype(np.float32), 1.0, (0.0, 0.0, 0.0),
                                cfg.canvas)
    skel = landmarks.estimate(grid, cfg)

    # voxel (ix, iy, iz) -> เมตร
    joints = {name: np.asarray(p, float)[[0, 1, 2]] * pitch + lo
              for name, p in skel.joints.items()}
    log.info("ข้อต่อของโมเดลฐาน %d จุด", len(joints))
    return joints, list(skel.notes)


# ---------------------------------------------------------------------------
# การดัด
# ---------------------------------------------------------------------------


def tps_warp(
    verts: np.ndarray, src: np.ndarray, dst: np.ndarray, smooth: float = 1e-3
) -> np.ndarray:
    """ดัดจุดทั้งก้อนด้วย thin-plate spline ที่ลากจุดควบคุม src ไปที่ dst

    ทำไมไม่ใช้ linear blend skinning อย่างที่ rig ทั่วไปทำ: LBS ออกแบบมาเพื่อ
    *ขยับท่าทาง* โดยที่สัดส่วนคงเดิม ส่วนงานนี้คือตรงกันข้าม — ท่าทางเหมือนกัน
    อยู่แล้ว (ยืนตรงทั้งคู่) แต่สัดส่วนต่างกัน

    ผลที่ได้ต่างกันมาก: LBS ให้ vertex แต่ละจุดเลือกกระดูกไม่กี่ท่อน จุดที่
    เลือกคนละชุดจึงกระโดดไปคนละทิศ ผิวฉีกเป็นรอยแตก และอกแยกเป็นสองก้อน
    ตรงรอยต่อระหว่างอิทธิพลของไหล่ซ้ายกับขวา (ทดสอบแล้วเห็นชัด)

    TPS ให้สนามการเปลี่ยนรูป *ผืนเดียวที่เรียบทั้งก้อน* ซึ่งผ่านจุดควบคุม
    ทุกจุดพอดี และเป็นสนามที่ "โค้งน้อยที่สุด" ในบรรดาสนามที่ทำได้ จึงไม่มี
    ทางฉีกหรือแยกก้อนโดยนิยาม

    `smooth` > 0 ยอมให้คลาดจากจุดควบคุมได้เล็กน้อยแลกกับความเรียบ กันกรณี
    ที่ข้อต่อสองจุดอยู่ใกล้กันมากจนระบบสมการเกือบเอกฐาน
    """
    n = len(src)
    if n < 4:
        raise ValueError("ต้องมีจุดควบคุมอย่างน้อย 4 จุดจึงจะทำ TPS ได้")

    # phi(r) = r คือ kernel ของ TPS ในสามมิติ
    d = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)
    K = d + smooth * np.eye(n) * float(d.max() or 1.0)
    P = np.hstack([np.ones((n, 1)), src])

    A = np.zeros((n + 4, n + 4))
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T
    rhs = np.zeros((n + 4, 3))
    rhs[:n] = dst

    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    W, Aff = sol[:n], sol[n:]

    out = np.empty_like(verts)
    chunk = 20000                      # กัน matrix (V, n) ใหญ่เกินหน่วยความจำ
    for i in range(0, len(verts), chunk):
        v = verts[i : i + chunk]
        U = np.linalg.norm(v[:, None, :] - src[None, :, :], axis=2)
        out[i : i + chunk] = U @ W + np.hstack([np.ones((len(v), 1)), v]) @ Aff
    return out


def world_to_voxel(
    verts_world: np.ndarray, ref_vox: np.ndarray, ref_world: np.ndarray
) -> np.ndarray:
    """แปลงจุดจากหน่วยเมตรกลับเป็นพิกัด voxel ของกริดเดิม

    อนุมาน transform จาก mesh อ้างอิงที่มีทั้งสองพิกัด (เหมือนที่
    `Skeleton.to_world` ทำ แต่ย้อนทาง) เพื่อให้ขั้น bake texture ซึ่งทำงาน
    ในพิกัด voxel ใช้ mesh ที่ดัดแล้วได้โดยไม่ต้องแก้สูตรที่ไหนอีก
    """
    out = np.empty_like(verts_world)
    for ax in range(3):
        lo_v, hi_v = ref_vox[:, ax].min(), ref_vox[:, ax].max()
        lo_w, hi_w = ref_world[:, ax].min(), ref_world[:, ax].max()
        span = hi_w - lo_w
        scale = (hi_v - lo_v) / span if span > 1e-9 else 1.0
        out[:, ax] = (verts_world[:, ax] - lo_w) * scale + lo_v
    return out


def _pose_gap(src: dict[str, np.ndarray], dst: dict[str, np.ndarray],
              bones: list[tuple[str, str]]) -> tuple[float, str]:
    """มุมที่ต่างกันมากที่สุดระหว่างโครงสองชุด คืน (องศา, ชื่อกระดูก)"""
    worst, worst_bone = 0.0, ""
    for pa, pb in bones:
        u, v = src[pb] - src[pa], dst[pb] - dst[pa]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-6 or nv < 1e-6:
            continue
        ang = float(np.degrees(np.arccos(np.clip((u @ v) / (nu * nv), -1.0, 1.0))))
        if ang > worst:
            worst, worst_bone = ang, f"{pa}->{pb}"
    return worst, worst_bone


def fit_skeleton(
    base, target_joints: dict[str, list[float]], cfg: PipelineConfig,
    parents: dict[str, str | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """ดัดโมเดลฐานให้เข้ากับตัวละคร คืน (vertex เมตร, faces, บันทึก)

    ต้องคืน faces ด้วย ไม่ใช่แค่ vertex: ขั้นหมุนท่าส่งผ่าน Blender ซึ่ง
    export/import ใหม่แล้ว topology เปลี่ยนไป (vertex ถูกแตกแล้วรวมกลับ
    21692 -> 18852) ถ้าผู้เรียกใช้ faces ชุดเดิม index จะเกินช่วงแล้ว
    open3d ที่รับไปต่อจะ *segfault ทั้ง process* โดยไม่มีข้อความบอก

    สองจังหวะ: *หมุน* ท่าด้วย armature ก่อน แล้วค่อย *ดัด* สัดส่วนด้วย TPS
    แยกกันเพราะเครื่องมือคนละตัวเก่งคนละอย่าง — armature หมุนท่อนกระดูกได้
    แต่เปลี่ยนรูปร่างกล้ามเนื้อไม่ได้ ส่วน TPS ตรงกันข้าม
    """
    src, notes = base_joints(base, cfg)
    dst = {k: np.asarray(v, float) for k, v in target_joints.items()}

    def common(s: dict) -> list[tuple[str, str]]:
        return [(a, b) for a, b in FIT_BONES
                if a in s and b in s and a in dst and b in dst]

    # --- จังหวะที่ 1: หมุนท่าถ้าต่างกันมากพอ -------------------------------
    gap, gap_bone = _pose_gap(src, dst, common(src))
    if cfg.base_pose and gap > float(cfg.pose_min_angle):
        log.info("ท่าทางต่างกัน %.0f องศาที่ %s — หมุนด้วย armature ก่อน",
                 gap, gap_bone)
        from .landmarks import BONE_PARENTS
        try:
            base, posed_j = pose_with_blender(
                base, src, target_joints, dict(parents or BONE_PARENTS), cfg)
            if posed_j:
                # ใช้ข้อต่อที่ armature รายงานมา แทนการ voxelize แล้วเดาใหม่
                src = posed_j
            else:
                src, extra = base_joints(base, cfg)
                notes += extra
            gap, gap_bone = _pose_gap(src, dst, common(src))
            log.info("หลังหมุนท่า เหลือต่างกัน %.0f องศาที่ %s", gap, gap_bone)
        except (RuntimeError, FileNotFoundError, OSError) as exc:
            notes.append(f"หมุนท่าด้วย armature ไม่สำเร็จ: {exc}")
            log.warning("%s", notes[-1])

    # --- จังหวะที่ 2: ดัดสัดส่วนด้วย TPS ------------------------------------
    bones = common(src)
    if len(bones) < 6:
        raise RuntimeError(
            f"จับคู่กระดูกได้แค่ {len(bones)} ท่อน ซึ่งน้อยเกินกว่าจะดัดได้\n"
            "โมเดลฐานน่าจะไม่ใช่ท่ายืนตรง หรือ voxel ของมันแยกแขนขาไม่ออก"
        )

    # จุดควบคุม = ข้อต่อที่มีทั้งสองฝั่ง บวกจุดกลางกระดูกเพื่อคุมรูปร่าง
    # ของท่อนยาว ๆ (ต้นแขน ต้นขา) ไม่ให้โป่งหรือคอดตรงกลาง
    names = [n for n in src if n in dst]
    ctrl_s = [src[n] for n in names]
    ctrl_d = [dst[n] for n in names]
    for pa, pb in bones:
        ctrl_s.append(0.5 * (src[pa] + src[pb]))
        ctrl_d.append(0.5 * (dst[pa] + dst[pb]))

    if gap > float(cfg.fit_max_angle):
        notes.append(
            f"ท่าทางของโมเดลฐานกับตัวละครยังต่างกัน {gap:.0f} องศา "
            f"ที่กระดูก {gap_bone} — เกินกว่าที่ TPS จะดัดให้ได้ "
            "รูปทรงช่วงนั้นจะยืดเพี้ยน"
        )
        log.warning("%s", notes[-1])

    verts = base.verts
    out = tps_warp(verts, np.array(ctrl_s), np.array(ctrl_d),
                   smooth=float(cfg.fit_smooth))

    shift = np.linalg.norm(out - verts, axis=1)
    log.info("ดัดโมเดลฐานด้วย %d ท่อนกระดูก — ขยับเฉลี่ย %.1f cm สูงสุด %.1f cm",
             len(bones), 100 * shift.mean(), 100 * shift.max())
    return out, np.asarray(base.faces, np.int32), notes


# ---------------------------------------------------------------------------
# หมุนท่าด้วย armature (ผ่าน Blender) ก่อนค่อยดัดสัดส่วน
# ---------------------------------------------------------------------------


def pose_with_blender(
    base, src_joints: dict[str, np.ndarray], dst_joints: dict[str, list[float]],
    parents: dict[str, str | None], cfg: PipelineConfig,
    blender_exe: str | None = None, timeout: int = 900,
):
    """ให้ Blender หมุนโมเดลฐานไปเป็นท่าของตัวละคร คืน BaseMesh ที่ posed แล้ว

    ส่ง mesh ที่ normalize แล้วไปให้ Blender ไม่ใช่ไฟล์ต้นฉบับ เพื่อให้สอง
    ฝั่งเห็นเรขาคณิตกับระบบพิกัดชุดเดียวกันเป๊ะ ไม่ต้องไปเดาแกนซ้ำอีกรอบ
    """
    import json
    import subprocess
    import tempfile

    import trimesh

    from .basemesh import BaseMesh
    from .config import BLENDER_EXE

    exe = Path(blender_exe or BLENDER_EXE)
    if not exe.exists():
        raise FileNotFoundError(
            f"ขั้นหมุนท่าต้องใช้ Blender แต่ไม่พบที่ {exe}\n"
            "ตั้ง BLENDER_EXE ให้ถูก หรือใช้ --no-base-pose เพื่อข้ามขั้นนี้"
        )

    tmp = Path(tempfile.mkdtemp(prefix="basepose_"))
    mesh_path = tmp / "base.glb"
    trimesh.Trimesh(base.verts, base.faces, process=False).export(str(mesh_path))

    common = {k: v for k, v in parents.items() if k in src_joints}
    (tmp / "src.json").write_text(json.dumps({
        "joints": {k: np.asarray(v, float).tolist() for k, v in src_joints.items()},
        "parents": common,
    }, ensure_ascii=False), encoding="utf-8")
    (tmp / "dst.json").write_text(json.dumps({
        "joints": {k: list(map(float, v)) for k, v in dst_joints.items()},
    }, ensure_ascii=False), encoding="utf-8")

    out_path = tmp / "posed.glb"
    jnt_path = tmp / "posed_joints.json"
    script = Path(__file__).resolve().parent / "blender_pose.py"
    log.info("เรียก Blender หมุนท่าโมเดลฐาน ...")
    proc = subprocess.run(
        [str(exe), "--background", "--factory-startup", "--python", str(script),
         "--", "--mesh", str(mesh_path), "--src", str(tmp / "src.json"),
         "--dst", str(tmp / "dst.json"), "--out", str(out_path),
         "--outjoints", str(jnt_path)],
        capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace")

    for line in (proc.stdout or "").splitlines():
        if line.startswith(("[pose]", "[rig]")):
            log.info("  %s", line.split("]", 1)[1].strip())

    # Blender คืน exit 0 ได้แม้สคริปต์ traceback กลางคัน จึงต้องให้สคริปต์
    # ประกาศความสำเร็จเอง ไม่งั้นจะได้ไฟล์เก่าค้างมาแล้วนึกว่าสำเร็จ
    if "[pose] เสร็จสมบูรณ์" not in (proc.stdout or "") or not out_path.exists():
        tail = "\n".join((proc.stdout or "").splitlines()[-25:])
        err = "\n".join((proc.stderr or "").splitlines()[-10:])
        raise RuntimeError(f"Blender หมุนท่าไม่สำเร็จ\n--- stdout ---\n{tail}\n"
                           f"--- stderr ---\n{err}")

    scene = trimesh.load(str(out_path), process=False, force="scene")
    posed = trimesh.util.concatenate(scene.dump(concatenate=False))
    # Blender เขียน GLB แบบแยก vertex ต่อหน้า (flat shading) จำนวนจุดเลยบวม
    # ขึ้น 5 เท่า รวมกลับให้เหลือ topology เดิมก่อนส่งต่อ
    before = len(posed.vertices)
    posed.merge_vertices()

    # เก็บกวาดหน้าที่เสื่อมสภาพก่อนส่งต่อ
    #
    # การรวม vertex ทำให้หน้าบาง ๆ ยุบจนมี index ซ้ำ กลายเป็นหน้าพื้นที่ศูนย์
    # ซึ่ง open3d ไปคิด normal ต่อแล้ว *segfault ทั้ง process* (เจอจริง
    # exit 139) ไม่ใช่ขึ้น exception ให้จับ จึงต้องกันตั้งแต่ต้นทาง
    bad = len(posed.faces)
    posed.update_faces(posed.nondegenerate_faces())
    posed.remove_unreferenced_vertices()
    bad -= len(posed.faces)

    v = np.asarray(posed.vertices, np.float64)
    if before != len(v) or bad:
        log.info("รวม vertex ที่ exporter แตกไว้: %d -> %d (ทิ้งหน้าเสื่อมสภาพ %d)",
                 before, len(v), bad)

    # glTF ที่ Blender เขียนออกมาเป็น Y-up เหมือนเดิม จึงไม่ต้องหมุนแกนซ้ำ
    # ข้อต่อหลังหมุน อ่านจาก armature ตรง ๆ ไม่ต้องประเมินจาก mesh ใหม่
    joints_after: dict[str, np.ndarray] = {}
    if jnt_path.exists():
        raw = json.loads(jnt_path.read_text(encoding="utf-8")).get("joints", {})
        joints_after = {k: np.asarray(p, float) for k, p in raw.items()}

    log.info("ได้โมเดลฐานท่าใหม่: %d verts — สูง %.2f m กว้าง %.2f m",
             len(v), np.ptp(v[:, 1]), np.ptp(v[:, 0]))
    return (BaseMesh(v, np.asarray(posed.faces, np.int32), base.source,
                     base.transform, list(base.notes)), joints_after)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """normal ต่อ vertex แบบถ่วงน้ำหนักด้วยพื้นที่หน้า

    เขียนเองแทนการเรียก `meshing.orient` เพราะเส้นทางโมเดลฐานไม่ต้องการสิ่งที่
    ฟังก์ชันนั้นทำจริง ๆ (จัด winding จาก hull) และ open3d ข้างในมัน
    *segfault ทั้ง process* (exit 139) กับ mesh ที่มาจากโมเดลฐาน ซึ่งจับไม่ได้
    ด้วย try/except — ล้มทั้ง pipeline โดยไม่มีข้อความบอกสาเหตุ

    winding ของโมเดลฐานถูกต้องมาตั้งแต่ไฟล์ต้นทางอยู่แล้ว จึงไม่ต้องจัดใหม่
    """
    n = np.zeros_like(verts)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # ยาว = 2*พื้นที่
    for i in range(3):
        np.add.at(n, faces[:, i], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


# ---------------------------------------------------------------------------
# อัดปริมาตรให้ชนเงา
# ---------------------------------------------------------------------------

#: กระดูกที่ห้ามแตะ เพราะ hull ตรงนั้นหยาบกว่าโมเดลฐานเสมอ (มือ/เท้าเป็นนวม)
KEEP_BONES = ("Hand", "Foot", "Toe")
#: หัวจะถูกเว้นด้วยหรือไม่ ขึ้นกับ cfg.shape_head
HEAD_BONES = ("Head", "HeadTop")



def _smooth_on_mesh(vals: np.ndarray, faces: np.ndarray, mask: np.ndarray,
                    iters: int) -> np.ndarray:
    """เกลี่ยค่าต่อ vertex ไปตามเส้นเชื่อมของ mesh

    จำเป็นมาก เพราะตัวคูณที่คำนวณต่อ vertex อิสระกันจะกระโดด: จุดที่อยู่ติดกัน
    อาจเจอขอบเงาคนละที่หรือเลือกกระดูกคนละท่อน แล้วถูกดันไปคนละระยะ ผิวจึง
    ระเบิดเป็นเสี่ยง ๆ (เห็นชัดมากตอนทดสอบครั้งแรก)

    `mask` คือ vertex ที่มีค่าเชื่อถือได้ จุดที่ไม่มีข้อมูลจะถูกเติมด้วย
    ค่าเฉลี่ยของเพื่อนบ้านไปเรื่อย ๆ
    """
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    out = vals.copy()
    out[~mask] = np.nan

    for _ in range(max(int(iters), 0)):
        acc = np.zeros(len(vals))
        cnt = np.zeros(len(vals))
        v = np.nan_to_num(out, nan=0.0)
        has = ~np.isnan(out)
        np.add.at(acc, e[:, 0], v[e[:, 1]] * has[e[:, 1]])
        np.add.at(cnt, e[:, 0], has[e[:, 1]].astype(float))
        avg = np.divide(acc, cnt, out=np.full(len(vals), np.nan), where=cnt > 0)
        # จุดที่มีค่าอยู่แล้ว ผสมกับเพื่อนบ้านครึ่งหนึ่ง จุดที่ว่างรับค่าเพื่อนมาเลย
        out = np.where(np.isnan(out), avg,
                       np.where(np.isnan(avg), out, 0.5 * out + 0.5 * avg))
    return np.nan_to_num(out, nan=1.0)


def fit_volume(
    verts_vox: np.ndarray, faces: np.ndarray, occ: np.ndarray,
    joints_vox: dict[str, np.ndarray], cfg: PipelineConfig,
) -> np.ndarray:
    """ขยาย/ย่อหน้าตัดของแต่ละท่อนให้หนาเท่าที่เงาบอก

    ทำไมต้องมีขั้นนี้: TPS ลากเฉพาะ *ข้อต่อ* ไปตรงกัน มันจึงได้ความยาวแขนขา
    ถูกแต่ความหนาเป็นของโมเดลฐานล้วน ๆ ตัวละครที่ล่ำกว่าคนทั่วไปสองเท่าเลย
    ออกมาผอมเก้งก้าง (วัดได้ว่าหนา 0.37 m ขณะที่เงาบอก 0.68 m)

    วิธี: สำหรับแต่ละ vertex หาแกนกระดูกที่ใกล้ที่สุด แล้ววัดว่าจาก *แกนนั้น*
    ออกไปตามทิศของ vertex เงาทึบถึงระยะเท่าไร จากนั้นย้าย vertex ไปที่ระยะนั้น
    ตามสัดส่วน รูปหน้าตัดเดิม (ร่องกล้ามเนื้อ) จึงถูกเก็บไว้ทั้งหมด
    เปลี่ยนแค่ขนาด — ต่างจากการ shrink-wrap ที่ดูดผิวไปแปะ hull จนรายละเอียด
    ของโมเดลฐานหายหมด

    มือ เท้า และหัวไม่ถูกย่อ เพราะ hull ตรงนั้นเป็นนวมกับก้อน การดูดเข้าไป
    หาเงาจะทำลายนิ้วที่เป็นเหตุผลทั้งหมดของการใช้โมเดลฐาน
    """
    ny, nx, nz = occ.shape
    solid = occ > 0.5

    # หัวถูกเว้นจากการขยายตามรัศมีเสมอ — มันมีวิธีของตัวเองใน `fit_head`
    # เพราะปากยื่นไปทางเดียว ซึ่งการขยายตามรัศมีทำไม่ได้
    skip = KEEP_BONES + HEAD_BONES
    names = [n for n in joints_vox if BONE_PARENTS.get(n) in joints_vox]
    segs, keep = [], []
    for n in names:
        p = BONE_PARENTS[n]
        segs.append((joints_vox[p][[1, 0, 2]], joints_vox[n][[1, 0, 2]]))
        keep.append(any(k in n or k in p for k in skip))
    if not segs:
        return verts_vox
    keep = np.array(keep)

    # จุดบนแกนกระดูกที่ใกล้ vertex ที่สุด (ทำงานในลำดับ (iy, ix, iz))
    P = verts_vox[:, [1, 0, 2]].astype(np.float64)
    best_d = np.full(len(P), np.inf)
    foot = np.zeros_like(P)
    bone_of = np.zeros(len(P), np.int32)
    for i, (a, b) in enumerate(segs):
        ab = b - a
        den = float(ab @ ab) or 1.0
        t = np.clip(((P - a) @ ab) / den, 0.0, 1.0)[:, None]
        q = a + t * ab
        d = np.linalg.norm(P - q, axis=1)
        hit = d < best_d
        best_d[hit], foot[hit], bone_of[hit] = d[hit], q[hit], i

    w = P - foot
    r_cur = np.linalg.norm(w, axis=1)
    ok = r_cur > 1e-6
    u = np.zeros_like(w)
    u[ok] = w[ok] / r_cur[ok, None]

    # เดินออกจากแกนตามทิศของ vertex หาว่าเนื้อทึบถึงระยะไหน
    steps = np.linspace(0.0, float(cfg.volume_reach), 56)[None, :]
    rr = r_cur[:, None] * steps
    pts = foot[:, None, :] + u[:, None, :] * rr[:, :, None]
    idx = np.rint(pts).astype(np.int64)
    np.clip(idx[:, :, 0], 0, ny - 1, out=idx[:, :, 0])
    np.clip(idx[:, :, 1], 0, nx - 1, out=idx[:, :, 1])
    np.clip(idx[:, :, 2], 0, nz - 1, out=idx[:, :, 2])
    inside = solid[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]

    # ขอบของ *ก้อนแรก* ที่ติดกับแกน ไม่ใช่จุดทึบสุดท้ายที่เจอ
    #
    # ถ้าเอาจุดทึบสุดท้าย รังสีจะข้ามช่องว่างไปโดนอวัยวะอื่นที่อยู่ไกลออกไป
    # แล้วลากเนื้อตรงนั้นตามไปด้วย เคสจริง: ก้นถูกดันถอยหลัง 30 ซม. ทะลุเข้า
    # ไปในปริมาตร *หาง* ของ hull จนเห็นเป็นกล่องลอยอยู่หลังสะโพก
    # ขอบของก้อนแรกคือผิวจริงของอวัยวะที่ vertex นั้นสังกัดอยู่
    n_steps = inside.shape[1]
    first_gap = np.where(inside.all(axis=1), n_steps,
                         np.argmin(inside, axis=1))
    last = np.clip(first_gap - 1, 0, n_steps - 1)
    found = inside[:, 0] & (first_gap > 0)
    r_hull = np.where(found, rr[np.arange(len(P)), last], r_cur)

    # เชื่อค่าเฉพาะ vertex ที่อยู่ห่างแกนพอสมควร
    #
    # จุดที่เกือบแตะแกนกระดูกมี r_cur เล็กมาก อัตราส่วน r_hull/r_cur จึงพุ่ง
    # ไปหลายสิบเท่าจากความคลาดเคลื่อนไม่กี่ voxel — ใช้ไม่ได้เลย
    min_r = float(cfg.volume_min_radius)
    good = ok & found & ~keep[bone_of] & (r_cur > min_r)

    raw = np.ones(len(P))
    raw[good] = np.clip(r_hull[good] / r_cur[good],
                        float(cfg.volume_min), float(cfg.volume_max))

    scale = _smooth_on_mesh(raw, faces, good, int(cfg.volume_smooth))
    scale = np.clip(scale, float(cfg.volume_min), float(cfg.volume_max))

    # เกลี่ย *เวกเตอร์การขยับ* ไม่ใช่แค่ตัวคูณ
    #
    # ตัวคูณเรียบอย่างเดียวไม่พอ เพราะ vertex ที่อยู่ติดกันแต่ถูกจับให้อยู่
    # คนละกระดูก จะมีแกนอ้างอิง (foot) และทิศ (u) คนละชุด ต่อให้คูณเท่ากัน
    # ก็ถูกดันไปคนละทาง ผิวเลยแตกเป็นแผ่น ๆ ตรงรอยต่อระหว่างท่อน
    # การเกลี่ยที่ปลายทาง (เวกเตอร์การขยับ) แก้ได้ทุกกรณีพร้อมกัน
    move = ok & ~keep[bone_of]
    disp = (foot + u * (r_cur * scale)[:, None]) - P
    disp[~move] = 0.0
    for ax in range(3):
        disp[:, ax] = _smooth_on_mesh(disp[:, ax], faces,
                                      np.ones(len(P), bool),
                                      int(cfg.volume_smooth))
    out = verts_vox.copy()
    out += disp[:, [1, 0, 2]]
    log.info("อัดปริมาตรให้ชนเงา: ขยับ %d/%d vertex "
             "(วัดได้จริง %d จุด, ตัวคูณเฉลี่ย %.2f)",
             int(move.sum()), len(P), int(good.sum()),
             float(scale[move].mean()) if move.any() else 1.0)
    return out


# ---------------------------------------------------------------------------
# ต่ออวัยวะที่โมเดลฐานไม่มี
# ---------------------------------------------------------------------------


def build_tail_mesh(
    path: np.ndarray, radius: np.ndarray, segments: int = 14,
    thickness: float = 1.0, anchor: np.ndarray | None = None,
    blend: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """ปั้นหางเป็นท่อ mesh จากแกนกลางที่ลากมาจากภาพด้านหลัง

    โมเดลฐานเป็นคน จึงไม่มีหางมาให้ ต่างจากเส้นทาง visual hull ที่หางโผล่มา
    เองเพราะมันปั้นทุกอย่างจาก occupancy ก้อนเดียวกัน

    ปั้นเป็นท่อตรง ๆ แทนที่จะไปตัดเอาจาก mesh ของ hull เพราะได้ topology ที่
    สะอาด (วงแหวนเรียงตามแกน) ซึ่ง rig แล้วหางงอได้สวยกว่าผล marching cubes
    มาก และคุมจำนวนหน้าได้

    ใช้ parallel transport หมุนกรอบอ้างอิงตามแกน ไม่ใช่ตรึงแกนอ้างอิงไว้
    ไม่งั้นท่อจะบิดตัวเองตรงช่วงที่หางหักเลี้ยว
    """
    if path is None or len(path) < 3:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)

    P = np.asarray(path, np.float64).copy()
    R = np.asarray(radius, np.float64) * float(thickness)

    # ดึงโคนหางเข้าไปฝังในตัว
    #
    # ขั้นปั้นหางใน hull ดันหางออกไปด้านหลังตาม `tail_z_bias` เพื่อไม่ให้มัน
    # จมอยู่ในลำตัว ซึ่งถูกตอนที่ทุกอย่างเป็น occupancy ก้อนเดียวกันแล้วเบลอ
    # รวมกัน แต่พอหางเป็น mesh แยกชิ้น โคนที่ลอยอยู่หลังตัว 30 voxel จะโผล่
    # เป็นก้อนลอยให้เห็นชัด ต้องลากกลับเข้าหาสะโพกให้ฝังเข้าไปในเนื้อ
    if anchor is not None and len(P) > 4:
        k = max(2, int(blend * len(P)))
        w = np.linspace(1.0, 0.0, k)[:, None] ** 1.5
        P[:k] = P[:k] * (1.0 - w) + np.asarray(anchor, float)[None, :] * w

    # ทิศของแกนที่แต่ละจุด
    t = np.gradient(P, axis=0)
    ln = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.maximum(ln, 1e-9)

    # กรอบอ้างอิงเริ่มต้น: แกนที่ตั้งฉากกับ t[0] มากที่สุด
    seed = np.array([0.0, 0.0, 1.0])
    if abs(seed @ t[0]) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    n = seed - (seed @ t[0]) * t[0]
    n /= np.linalg.norm(n)

    normals = [n]
    for i in range(1, len(P)):
        v = np.cross(t[i - 1], t[i])
        s = np.linalg.norm(v)
        if s < 1e-9:
            normals.append(normals[-1])
            continue
        v /= s
        c = float(np.clip(t[i - 1] @ t[i], -1.0, 1.0))
        ang = np.arccos(c)
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)
        m = rot @ normals[-1]
        m -= (m @ t[i]) * t[i]
        normals.append(m / max(np.linalg.norm(m), 1e-9))
    N = np.array(normals)
    B = np.cross(t, N)

    ang = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    ring = (np.cos(ang)[None, :, None] * N[:, None, :]
            + np.sin(ang)[None, :, None] * B[:, None, :])
    verts = (P[:, None, :] + R[:, None, None] * ring).reshape(-1, 3)

    faces = []
    for i in range(len(P) - 1):
        a, b = i * segments, (i + 1) * segments
        for j in range(segments):
            k = (j + 1) % segments
            faces.append([a + j, b + j, b + k])
            faces.append([a + j, b + k, a + k])

    # ปิดเฉพาะปลายหาง ไม่ปิดโคน
    #
    # โคนหางกว้าง 50 voxel ถ้าปิดเป็นจานแบนแล้วมันโผล่พ้นผิวตัวออกมา จะเห็น
    # เป็นแผ่นสี่เหลี่ยมลอยอยู่หลังสะโพกชัดมาก (เจอจริง) ปล่อยเปิดไว้ดีกว่า
    # เพราะโคนถูกลากเข้าไปฝังในเนื้อสะโพกแล้ว ไม่มีใครมองเห็นรูนั้น
    tip_c = len(verts)
    verts = np.vstack([verts, P[-1][None, :]])
    last = (len(P) - 1) * segments
    for j in range(segments):
        k = (j + 1) % segments
        faces.append([tip_c, last + j, last + k])

    return verts, np.asarray(faces, np.int32)


def attach(parts: list[tuple[np.ndarray, np.ndarray]]
           ) -> tuple[np.ndarray, np.ndarray]:
    """รวมหลายชิ้นเป็น mesh เดียว โดยเลื่อน index ของแต่ละชิ้นให้ถูก"""
    vs, fs, off = [], [], 0
    for v, f in parts:
        if len(v) == 0:
            continue
        vs.append(v)
        fs.append(np.asarray(f, np.int32) + off)
        off += len(v)
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    return np.vstack(vs), np.vstack(fs)


def _ray_surface(
    solid: np.ndarray, origin: np.ndarray, dirs: np.ndarray,
    reach: float, steps: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """ยิงรังสีจาก `origin` ไปตาม `dirs` หาผิวของก้อนแรกที่ติดกับจุดเริ่ม

    คืน (ระยะถึงผิว, เจอหรือไม่) — หยุดที่ช่องว่างแรก ไม่ใช่จุดทึบสุดท้าย
    ไม่งั้นรังสีจะข้ามช่องว่างไปโดนอวัยวะอื่นที่อยู่ไกลออกไป
    """
    ny, nx, nz = solid.shape
    t = np.linspace(0.0, float(reach), steps)[None, :]
    pts = origin[None, None, :] + dirs[:, None, :] * t[:, :, None]
    idx = np.rint(pts).astype(np.int64)
    np.clip(idx[:, :, 0], 0, ny - 1, out=idx[:, :, 0])
    np.clip(idx[:, :, 1], 0, nx - 1, out=idx[:, :, 1])
    np.clip(idx[:, :, 2], 0, nz - 1, out=idx[:, :, 2])
    inside = solid[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]

    n = inside.shape[1]
    first_gap = np.where(inside.all(axis=1), n, np.argmin(inside, axis=1))
    last = np.clip(first_gap - 1, 0, n - 1)
    ok = inside[:, 0] & (first_gap > 1)
    return t[0][last], ok


def fit_head(
    verts_vox: np.ndarray, occ: np.ndarray,
    joints_vox: dict[str, np.ndarray], cfg: PipelineConfig,
) -> np.ndarray:
    """ดัดเฉพาะส่วนหัวของโมเดลฐานให้ไปตรงกับหัวใน hull ด้วย TPS

    ทำไมต้องแยกวิธีให้หัวโดยเฉพาะ: `fit_volume` ขยายเนื้อออกจากแกนกระดูก
    ซึ่งเก่งกับสิ่งที่หนาขึ้น *ทั้งรอบ* (แขน ขา ลำตัว) แต่ปากสัตว์ยื่นไป
    *ทางเดียว* ไกลกว่ารัศมีกะโหลกสามเท่า พอต้องเกลี่ยเวกเตอร์การขยับเพื่อ
    กันผิวแตก ท้ายทอยที่ไม่ต้องขยับก็ถูกเฉลี่ยเข้ามาจนตัวคูณที่หน้าเหลือ
    ราว 1.2 — ปากเลยไม่เคยโผล่

    TPS ไม่มีปัญหานั้น เพราะมันเรียบ *โดยนิยาม* ไม่ต้องเกลี่ยทับทีหลัง
    จึงเก็บการยื่นเฉพาะจุดไว้ได้ครบ

    จุดควบคุมมาจากการยิงรังสีจากกลางหัวออกไปทุกทิศ แล้วดูว่าเงาทึบถึงไหน
    บวกกับจุดตรึงที่คอ (ต้นทาง = ปลายทาง) เพื่อให้การดัดค่อย ๆ จางหายไป
    ไม่ไปลากลำตัวตามมาด้วย
    """
    if not cfg.shape_head:
        return verts_vox
    need = ("Head", "Neck")
    if any(k not in joints_vox for k in need):
        log.warning("ไม่มีข้อต่อ %s — ข้ามการดัดหัว", need)
        return verts_vox

    solid = occ > 0.5
    head = np.asarray(joints_vox["Head"], float)
    neck = np.asarray(joints_vox["Neck"], float)
    # ทำงานในลำดับ (iy, ix, iz) ให้ตรงกับ occupancy
    H = head[[1, 0, 2]]
    Nk = neck[[1, 0, 2]]
    P = verts_vox[:, [1, 0, 2]].astype(np.float64)

    head_h = max(float(P[:, 0].max() - Nk[0]), 1.0)
    sel = P[:, 0] > Nk[0] - cfg.head_blend * head_h
    if sel.sum() < 40:
        log.warning("หา vertex ของหัวไม่พอ (%d) — ข้ามการดัดหัว", int(sel.sum()))
        return verts_vox

    # จุดควบคุม: สุ่ม vertex บนหัวมาชุดหนึ่ง แล้วยิงรังสีจากกลางหัวผ่านมัน
    #
    # ต้องเว้นระยะจากคอให้ชัด ไม่งั้นจุดที่ขยับ 30 voxel จะไปอยู่ติดกับจุด
    # ตรึงที่ห้ามขยับ TPS ต้องสนองข้อบังคับที่ขัดกันในระยะไม่กี่ voxel แล้ว
    # มันแกว่งจนผิวหัวแตกเป็นเสี่ยง ๆ (เจอจริง: หัวกลายเป็นเศษแหลม ๆ)
    idx = np.flatnonzero(sel & (P[:, 0] > Nk[0] + cfg.head_gap * head_h))
    if idx.size > cfg.head_samples:
        idx = idx[np.linspace(0, idx.size - 1, cfg.head_samples).astype(int)]
    v = P[idx] - H
    r = np.linalg.norm(v, axis=1)
    keep = r > 1e-6
    idx, v, r = idx[keep], v[keep], r[keep]
    dirs = v / r[:, None]

    reach = float(cfg.head_reach) * float(np.percentile(r, 90))
    dist, ok = _ray_surface(solid, H, dirs, reach)
    if ok.sum() < 20:
        log.warning("ยิงรังสีหาผิวหัวใน hull ไม่เจอ (%d จุด) — ข้ามการดัดหัว",
                    int(ok.sum()))
        return verts_vox

    # กรองระยะด้วย median ระหว่างทิศที่อยู่ใกล้กัน
    #
    # หัวใน hull มีหนามสันหลังเป็นเงี่ยงแหลม ๆ อยู่บนกะโหลก รังสีไม่กี่เส้น
    # ที่บังเอิญยิงโดนเงี่ยงจะได้ระยะไกลกว่าเพื่อนบ้านสองเท่า แล้วลากผิวหัว
    # ตามไปเป็นหนามแหลมยาว (เจอจริง: หัวงอกเงี่ยงยาวออกมาหลายอัน)
    #
    # median ตัดค่าโดดทิ้งได้โดยไม่กลืนรูปทรงจริง ซึ่งต่างจากค่าเฉลี่ย —
    # และปากที่ยื่นออกมากินหลายสิบทิศติดกัน จึงรอดจากการกรองนี้
    if ok.sum() >= 12:
        D = dirs[ok]
        cosang = D @ D.T
        k = min(int(cfg.head_median_k), len(D))
        nb = np.argpartition(-cosang, k - 1, axis=1)[:, :k]
        dist = dist.copy()
        dist[ok] = np.median(dist[ok][nb], axis=1)

    src = P[idx][ok]
    dst = H[None, :] + dirs[ok] * dist[ok][:, None]
    # กันไม่ให้จุดไหนกระโดดไกลเกินเหตุ (เงาขาดหรือรังสีลอดออกข้าง)
    move = np.linalg.norm(dst - src, axis=1)
    sane = move < cfg.head_max_move * float(np.percentile(r, 90))
    src, dst = src[sane], dst[sane]

    # จุดตรึง: *วงแหวนรอบคอ* เท่านั้น ไม่ใช่จุดกระจายทั่วตัว
    #
    # เคยหว่านจุดตรึงไปทั่วลำตัว ซึ่งทำให้ TPS ต้องสนองข้อบังคับที่ขัดกัน
    # ในระยะใกล้ ๆ (หัวขยายสองเท่า แต่จุดข้าง ๆ ห้ามขยับ) แล้วมันแกว่งเกิน
    # ไปไกลมาก — vertex บางจุดกระเด็นไป 521 voxel ตัวโมเดลกว้างกลายเป็น 1.8 m
    band = np.abs(P[:, 0] - Nk[0]) < 0.12 * head_h
    ring = np.flatnonzero(band)
    if ring.size:
        pick = ring[np.linspace(0, ring.size - 1,
                                min(cfg.head_anchors, ring.size)).astype(int)]
        src = np.vstack([src, P[pick]])
        dst = np.vstack([dst, P[pick]])

    if len(src) < 12:
        log.warning("จุดควบคุมหัวเหลือแค่ %d — ข้ามการดัดหัว", len(src))
        return verts_vox

    warped = tps_warp(P[sel], src, dst, smooth=float(cfg.head_smooth))
    disp = warped - P[sel]

    # ไล่ระดับให้การดัดจางหายที่คอ แล้วตรึงเพดานการขยับ
    #
    # สองอย่างนี้ทำให้ขั้นนี้ *ไม่มีทาง* ไปแตะลำตัว ไม่ว่า TPS จะแกว่งแค่ไหน
    # ซึ่งสำคัญกว่าความเป๊ะ เพราะความเสียหายจากการแกว่งคือทั้งตัวพัง
    t = np.clip((P[sel][:, 0] - Nk[0]) / max(0.45 * head_h, 1.0), 0.0, 1.0)
    ramp = (t * t * (3.0 - 2.0 * t))[:, None]
    disp *= ramp

    # เพดานกันพุ่ง ตั้งหลวม ๆ ไว้เป็นตาข่ายรองสุดท้ายเท่านั้น
    #
    # ถ้าตั้งแน่นจนมันทำงานบ่อย vertex ที่ชนเพดานกับเพื่อนบ้านที่ยังไม่ชน
    # จะถูกตัดคนละระดับ แล้วผิวจะหักเป็นมุมตรงรอยนั้น
    cap = float(cfg.head_max_move) * float(np.percentile(r, 90))
    mag = np.linalg.norm(disp, axis=1, keepdims=True)
    n_cap = int((mag[:, 0] > cap).sum())
    if n_cap:
        disp *= np.minimum(1.0, cap / np.maximum(mag, 1e-9))
        log.info("  ตัดการขยับที่เกินเพดาน %d vertex", n_cap)

    moved = P[sel] + disp
    out = verts_vox.copy()
    out[sel] = moved[:, [1, 0, 2]]
    shift = np.linalg.norm(disp, axis=1)
    log.info("ดัดหัวด้วย TPS: %d จุดควบคุม / %d vertex — ขยับเฉลี่ย %.1f "
             "สูงสุด %.1f voxel", len(src), int(sel.sum()),
             float(shift.mean()), float(shift.max()))
    return out
