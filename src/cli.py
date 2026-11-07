r"""จุดเข้าใช้งานของ pipeline 2D -> 3D

    .venv\Scripts\python.exe -m src.cli
    .venv\Scripts\python.exe -m src.cli --voxel-res 320 --height 2.1 --debug
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
from pathlib import Path

from . import export as export_mod
from . import animation, landmarks, meshing, texture, uvmap, visual_hull
from .config import DEBUG_DIR, INPUT_DIR, OUTPUT_DIR, PipelineConfig
from .preprocess import prepare_views

log = logging.getLogger("2Dto3D")


class _Fmt(logging.Formatter):
    COLORS = {"WARNING": "\033[33m", "ERROR": "\033[31m", "INFO": "", "DEBUG": "\033[90m"}

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = "\033[0m" if color else ""
        return f"{color}{record.getMessage()}{reset}"


def setup_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_Fmt())
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        handlers=[handler], force=True)
    # เงียบ log ภายในของไลบรารีอื่น
    for noisy in ("numba", "PIL", "matplotlib", "onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def banner(step: int, total: int, title: str) -> None:
    log.info("\n\033[1m[%d/%d] %s\033[0m", step, total, title)


def build_parser() -> argparse.ArgumentParser:
    d = PipelineConfig()
    p = argparse.ArgumentParser(
        prog="2Dto3D",
        description="สร้างโมเดล 3D ที่พร้อม rig จากภาพ 2D หลายมุม (หน้า/หลัง/ข้าง)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=INPUT_DIR, help="โฟลเดอร์ภาพต้นทาง")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR, help="โฟลเดอร์ผลลัพธ์")
    p.add_argument("--name", default="character", help="ชื่อไฟล์ผลลัพธ์")

    g = p.add_argument_group("รูปทรง")
    g.add_argument("--voxel-res", type=int, default=d.voxel_res,
                   help="จำนวน voxel ตามความสูง ยิ่งมากยิ่งละเอียด")
    g.add_argument("--carve-mode", choices=["ellipse", "hull"], default=d.carve_mode,
                   help="ellipse = หน้าตัดวงรี (เหมือนร่างกายจริง), hull = ตัดกันตรง ๆ (หน้าตัดเป็นกล่อง)")
    g.add_argument("--cross-section-power", type=float, default=d.cross_section_power,
                   help="2 = วงรี, สูงขึ้น = เข้าใกล้สี่เหลี่ยมมุมมน")
    g.add_argument("--max-aspect", type=float, default=d.max_aspect,
                   help="เพดาน ลึก/กว้าง ของ *หัว* 0 = ไม่จำกัด "
                        "(เปิดเมื่อปากยาวไปลากความลึกของบ่าจนเกิดปีกหมวกรอบคอ)")
    g.add_argument("--body-aspect", type=float, default=d.body_aspect,
                   help="เพดาน ลึก/กว้าง ของลำตัวกับขา ตัดหางที่เชื่อมติดมาในมุมข้าง")
    g.add_argument("--no-neck", action="store_true",
                   help="ไม่ต้องเซาะซอกคอ (หัวจะกลืนเป็นทรงกรวยต่อกับไหล่)")
    g.add_argument("--neck-width", type=float, default=d.neck_width,
                   help="ความกว้างลำคอ เป็นกี่เท่าของครึ่งความกว้างหัว")
    g.add_argument("--neck-shoulder", type=float, default=d.neck_shoulder_min,
                   help="สันบ่าใต้คางบางแค่ไหน (1 = ไม่เซาะ, ต่ำ = คอเด่นขึ้น)")
    g.add_argument("--crest-width", type=float, default=d.crest_width,
                   help="ความกว้างสันหนามเทียบกับความกว้างหัว 0 = ไม่คืนหนาม")
    g.add_argument("--spike-open", type=int, default=d.spike_open,
                   help="ลบส่วนยื่นที่บางกว่ากี่ voxel (หนาม เขา ครีบ) 0 = เก็บไว้ทั้งหมด")
    g.add_argument("--tail-bones", type=int, default=d.tail_bones,
                   help="จำนวนกระดูกในโซ่หาง 0 = ไม่ใส่กระดูกหาง")
    g.add_argument("--no-tail", action="store_true",
                   help="ไม่เก็บเนื้อหาง (ทิ้งส่วนที่ยื่นด้านหลังลำตัวไปเลย)")
    g.add_argument("--depth-median", type=int, default=d.depth_median,
                   help="median filter บนโปรไฟล์ความลึก ตัดหนามสันหลังที่ทำให้เกิดสันวงแหวน")
    g.add_argument("--silhouette-smooth", type=float, default=d.silhouette_smooth,
                   help="เกลาขอบ silhouette ตามแนวตั้ง ลดลอนถี่บนผิว")
    g.add_argument("--profile-smooth", type=float, default=d.profile_smooth,
                   help="เกลาโปรไฟล์ความลึก/ความกว้างตามแนวตั้ง ลดผิวขั้นบันได")
    g.add_argument("--keep-side-extras", action="store_true",
                   help="ไม่ตัดหาง/เป้/ปีก ออกจาก mask ด้านข้าง")
    g.add_argument("--hull-blur", type=float, default=d.hull_blur,
                   help="เบลอ occupancy ก่อน marching cubes เพื่อลบเหลี่ยม")
    g.add_argument("--smooth", type=int, default=d.taubin_iters,
                   help="จำนวนรอบ Taubin smoothing")
    g.add_argument("--target-tris", type=int, default=d.target_tris,
                   help="งบ triangle ของ mesh สุดท้าย")
    g.add_argument("--height", type=float, default=d.character_height_m,
                   help="ความสูงจริงของตัวละคร (เมตร)")

    g = p.add_argument_group("มือ / เท้า")
    g.add_argument("--base-mesh", action="store_true",
                   help="ทดลองใช้โมเดลฐานใน input/ แทนรูปทรงจาก hull "
                        "(ยังพัฒนาไม่เสร็จ ผลปัจจุบันแย่กว่า hull)")
    g.add_argument("--no-base-pose", action="store_true",
                   help="ไม่ต้องหมุนท่าโมเดลฐานด้วย armature ก่อนดัด")
    g.add_argument("--no-fit-volume", action="store_true",
                   help="ไม่ต้องอัดปริมาตรโมเดลฐานให้ชนเงา")
    g.add_argument("--volume-max", type=float, default=d.volume_max,
                   help="เพดานการขยายหน้าตัดของโมเดลฐาน")
    g.add_argument("--no-detail-hands", action="store_true",
                   help="ไม่ต้องกันเขตมือออกจากกฎเกลาผิวของลำตัว (มือจะเป็นนวม)")
    g.add_argument("--no-detail-feet", action="store_true",
                   help="ไม่ต้องกันเขตเท้าออกจากกฎเกลาผิวของลำตัว")
    g.add_argument("--no-sole-plan", action="store_true",
                   help="ไม่ใช้ภาพมุมล่าง (down/sole/bottom) เป็นผังฝ่าเท้า")
    g.add_argument("--no-top-plan", action="store_true",
                   help="ไม่ใช้ภาพมุมบน (top/up) บอกความลึกของหาง")
    g.add_argument("--detail-blur", type=float, default=d.detail_blur,
                   help="เบลอเขตมือ/เท้าด้วย sigma เท่านี้ (ลำตัวใช้ --hull-blur)")
    g.add_argument("--palm-round", type=float, default=d.palm_round,
                   help="ความหนาฝ่ามือเป็นกี่เท่าของความกว้างมือ")
    g.add_argument("--hand-length-frac", type=float, default=d.hand_length_frac,
                   help="ความยาวมือคิดเป็นกี่ส่วนของความยาวแขนที่แยกจากลำตัว")

    g = p.add_argument_group("ภาพ")
    g.add_argument("--canvas", type=int, default=d.canvas, help="ความละเอียด canvas ที่ align")
    g.add_argument("--texture-mode", choices=["uv", "vertex"], default=d.texture_mode,
                   help="uv = bake ลง texture (คมกว่ามาก), vertex = เก็บสีที่ vertex")
    g.add_argument("--no-draco", action="store_true",
                   help="ไม่บีบ geometry ด้วย Draco (ไฟล์ใหญ่ขึ้นแต่เปิดได้ทุกที่)")
    g.add_argument("--texture-quality", type=int, default=d.texture_quality,
                   help="คุณภาพ JPEG ของ texture ใน GLB (0 = เก็บเป็น PNG)")
    g.add_argument("--texture-glb-size", type=int, default=d.texture_glb_size,
                   help="ย่อ texture ใน GLB เหลือด้านละเท่านี้ (0 = ไม่ย่อ)")
    g.add_argument("--texture-size", type=int, default=d.uv_texture_size,
                   help="ขนาด texture ด้านละกี่พิกเซล")
    g.add_argument("--rembg-model", default=d.rembg_model,
                   help="โมเดลแยกพื้นหลัง เช่น isnet-general-use, u2net, u2net_human_seg")
    g.add_argument("--flip-side", action="store_true",
                   help="พลิกมุมข้างซ้ายขวา ใช้เมื่อตรวจทิศหันหน้าผิด")

    g = p.add_argument_group("ผลลัพธ์")
    g.add_argument("--no-anim", action="store_true",
                   help="rig อย่างเดียว ไม่ต้องใส่อนิเมชัน")
    g.add_argument("--fps", type=int, default=d.fps, help="เฟรมต่อวินาทีของอนิเมชัน")
    g.add_argument("--rig", choices=["none", "blender"], default=d.rig,
                   help="none = export mesh เปล่าไป rig ที่ Mixamo, blender = rig อัตโนมัติ")
    g.add_argument("--formats", default="glb",
                   help="ฟอร์แมตที่ต้องการ คั่นด้วยจุลภาค เช่น glb,obj,ply")
    g.add_argument("--no-debug", action="store_true", help="ไม่เขียนภาพ debug")
    g.add_argument("-v", "--verbose", action="store_true", help="log ละเอียด")
    return p


def config_from_args(a: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        canvas=a.canvas,
        rembg_model=a.rembg_model,
        texture_mode=a.texture_mode,
        uv_texture_size=a.texture_size,
        texture_quality=a.texture_quality,
        draco=not a.no_draco,
        texture_glb_size=a.texture_glb_size,
        voxel_res=a.voxel_res,
        hull_blur=a.hull_blur,
        carve_mode=a.carve_mode,
        cross_section_power=a.cross_section_power,
        side_main_component_only=not a.keep_side_extras,
        spike_open=a.spike_open,
        max_aspect=a.max_aspect,
        body_aspect=a.body_aspect,
        neck_carve=not a.no_neck,
        neck_width=a.neck_width,
        neck_shoulder_min=a.neck_shoulder,
        crest_width=a.crest_width,
        silhouette_smooth=a.silhouette_smooth,
        depth_median=a.depth_median,
        keep_tail=not a.no_tail,
        detail_hands=not a.no_detail_hands,
        detail_feet=not a.no_detail_feet,
        use_sole_plan=not a.no_sole_plan,
        use_base_mesh=a.base_mesh,
        base_pose=not a.no_base_pose,
        fit_volume=not a.no_fit_volume,
        volume_max=a.volume_max,
        detail_blur=a.detail_blur,
        palm_round=a.palm_round,
        hand_length_frac=a.hand_length_frac,
        tail_bones=a.tail_bones,
        profile_smooth=a.profile_smooth,
        taubin_iters=a.smooth,
        target_tris=a.target_tris,
        character_height_m=a.height,
        rig=a.rig,
        animate=not a.no_anim,
        fps=a.fps,
        export_formats=tuple(f.strip() for f in a.formats.split(",") if f.strip()),
        debug=not a.no_debug,
    )


def run(a: argparse.Namespace) -> int:
    cfg = config_from_args(a)
    out_dir: Path = a.output
    debug_dir = out_dir / "debug"
    t_start = time.perf_counter()
    warnings: list[str] = []

    total = 6 if cfg.rig == "none" else 7

    banner(1, total, "เตรียมภาพ: แยกพื้นหลัง + align ทุกมุม")
    views = prepare_views(a.input, cfg, debug_dir, flip_side=a.flip_side)

    banner(2, total, "Visual hull: ตัด voxel ด้วย silhouette")
    # ภาพมุมล่าง (ถ้ามี) ไม่เข้า visual hull เพราะเป็นภาพ perspective
    # ใช้เป็นผังฝ่าเท้าอย่างเดียว ดูเหตุผลใน soleplan.py
    from . import soleplan, topplan
    sole_plans = soleplan.load(a.input, cfg)
    # ภาพมุมบนให้ระนาบ X-Z ซึ่งเป็นที่เดียวที่บอก *ความลึก* ของหางได้
    top = None if a.no_top_plan else topplan.load(a.input, cfg)
    grid = visual_hull.carve(views, cfg, sole_plans, top)
    for note in visual_hull.carve_report(grid, views):
        log.warning("คุณภาพ hull: %s", note)
        warnings.append(note)

    banner(3, total, "สร้าง mesh: marching cubes -> smooth -> decimate")
    mesh = meshing.build(grid, cfg)

    # --- โมเดลฐาน: ดัดโมเดลที่ถูกต้องอยู่แล้วให้เข้าสัดส่วนตัวละคร -----------
    #
    # ต้องทำตรงนี้ *ก่อน* ขั้น texture เพราะขั้นนั้นต้องฉายสีลงบน mesh ตัวจริง
    # ที่จะถูก export ไม่ใช่ตัวที่ถูกทิ้ง และต้องทำ *หลัง* meshing เพราะต้องใช้
    # mesh จาก hull มาอนุมาน transform ระหว่างพิกัด voxel กับเมตร
    base_notes: list[str] = []
    if cfg.use_base_mesh:
        from . import basefit, basemesh
        base_path = basemesh.find(a.input)
        if base_path is not None:
            skel_pre = landmarks.estimate(grid, cfg)
            ref_world = meshing.to_world(mesh, grid, cfg.character_height_m)
            joints_m = skel_pre.to_world(mesh.verts, ref_world)

            base = basemesh.load(base_path, cfg)
            fitted, fit_faces, base_notes = basefit.fit_skeleton(
                base, joints_m, cfg)
            verts_vox = basefit.world_to_voxel(fitted, mesh.verts, ref_world)
            if cfg.fit_volume:
                # ข้อต่อของตัวละครในพิกัด voxel — ใช้ตัวเดียวกับที่ประเมินมา
                jv = {k: np.asarray(v, float) for k, v in skel_pre.joints.items()}
                verts_vox = basefit.fit_volume(verts_vox, fit_faces,
                                               grid.occupancy, jv, cfg)
            # หัวมีวิธีของตัวเอง: TPS ลากผิวหน้าไปหาผิวหัวใน hull โดยตรง
            # การขยายตามรัศมีทำปากที่ยื่นไปทางเดียวไม่ได้
            verts_vox = basefit.fit_head(verts_vox, grid.occupancy, jv, cfg)
            # ต่อหางเข้าไป: โมเดลฐานเป็นคน จึงไม่มีหางมาให้ ต่างจากเส้นทาง
            # hull ที่หางโผล่มาเองเพราะปั้นทุกอย่างจาก occupancy ก้อนเดียวกัน
            if cfg.attach_tail and grid.tail_path is not None:
                hips = skel_pre.joints.get("Hips")
                tv, tf = basefit.build_tail_mesh(
                    grid.tail_path, grid.tail_radius,
                    segments=cfg.tail_segments, thickness=cfg.tail_thickness,
                    anchor=None if hips is None else np.asarray(hips, float),
                    blend=cfg.tail_root_blend)
                if len(tv):
                    verts_vox, fit_faces = basefit.attach(
                        [(verts_vox, fit_faces), (tv, tf)])
                    log.info("ต่อหางเข้ากับโมเดลฐาน: +%d verts / +%d tris",
                             len(tv), len(tf))
            mesh = meshing.MeshData(verts=verts_vox, faces=fit_faces)
            # ต้องคิด normal ใหม่: ขั้น texture กับ export ต้องใช้ และ normal
            # เดิมของโมเดลฐานใช้ไม่ได้แล้วหลังโดนดัด
            #
            # คิดเองด้วย numpy ไม่เรียก meshing.orient เพราะ open3d ข้างในนั้น
            # segfault ทั้ง process กับ mesh ที่มาจากโมเดลฐาน (จับไม่ได้ด้วย
            # try/except) และ winding ของโมเดลฐานถูกต้องมาแต่ต้นอยู่แล้ว
            mesh.normals = basefit.vertex_normals(mesh.verts, mesh.faces)
            mesh.log_stage(f"ดัดโมเดลฐาน {base_path.name}")
            log.info("ใช้โมเดลฐานแทนผล hull: %d verts / %d tris",
                     mesh.n_verts, mesh.n_faces)
            warnings.extend(base_notes)
        else:
            log.info("ไม่เจอโมเดลฐานใน %s — ใช้รูปทรงจาก visual hull ตามเดิม",
                     a.input)

    banner(4, total, "ฉายสีจากภาพลงบน mesh")
    tex_img = None
    if cfg.texture_mode == "uv":
        mesh = uvmap.unwrap(mesh, cfg)
        tex_img = uvmap.bake(mesh, grid, views, cfg)
        tex_path = out_dir / f"{a.name}_texture.png"
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite(str(tex_path), tex_img[:, :, ::-1])
        log.info("เขียน texture -> %s (%.2f MB)",
                 tex_path, tex_path.stat().st_size / 1e6)
    else:
        mesh = texture.project_vertex_colors(mesh, grid, views, cfg)

    banner(5, total, "ประเมินตำแหน่งข้อต่อ")
    skel = landmarks.estimate(grid, cfg)
    warnings.extend(skel.notes)

    verts_vox = mesh.verts.copy()
    verts_world = meshing.to_world(mesh, grid, cfg.character_height_m)

    if cfg.debug:
        landmarks.draw_debug(grid, skel, debug_dir / "03_skeleton.png")
    joints_path = landmarks.write_json(skel, verts_vox, verts_world,
                                       out_dir / f"{a.name}_joints.json")

    banner(6, total, "เขียนไฟล์ผลลัพธ์")
    written = export_mod.export(verts_world, mesh, out_dir, a.name,
                                cfg.export_formats, texture=tex_img)
    if cfg.texture_quality > 0:
        for p in written.values():
            if p.suffix.lower() == ".glb":
                export_mod.recompress_textures(p, cfg.texture_quality,
                                               cfg.texture_glb_size)

    export_mod.write_summary(
        out_dir / f"{a.name}_summary.json", mesh, verts_world,
        extra={
            "views": {n: str(v.source) for n, v in views.items()},
            "voxel_grid": list(grid.shape),
            "warnings": warnings,
            "joints": str(joints_path),
        },
    )

    if cfg.rig == "blender":
        banner(7, total, "สร้างโครงกระดูก อนิเมชัน และ export GLB")

        anim_path = None
        if cfg.animate:
            n_tail = sum(1 for n in skel.joints
                         if n.startswith("Tail") and n != "TailEnd")
            clips = animation.default_clips(n_tail)
            anim_path = out_dir / f"{a.name}_clips.json"
            anim_path.write_text(
                json.dumps(animation.to_payload(clips, cfg.fps),
                           indent=2, ensure_ascii=False),
                encoding="utf-8")
            log.info("เขียนนิยามอนิเมชัน %d clip (%s) -> %s",
                     len(clips), ", ".join(c.name for c in clips), anim_path)

        from .rigging import rig_with_blender
        rigged = rig_with_blender(written["glb"], joints_path,
                                  out_dir / f"{a.name}_rigged.glb", anim_path,
                                  draco=cfg.draco)
        if cfg.texture_quality > 0:
            export_mod.recompress_textures(rigged, cfg.texture_quality,
                                           cfg.texture_glb_size)
        log.info("ได้ไฟล์ที่ rig แล้ว: %s (%.2f MB)",
                 rigged, rigged.stat().st_size / 1e6)

    elapsed = time.perf_counter() - t_start
    log.info("\n\033[1mเสร็จใน %.1f วินาที\033[0m", elapsed)
    log.info("ผลลัพธ์อยู่ที่ %s", out_dir)

    if warnings:
        log.warning("\n\033[1mมีข้อควรตรวจ %d ข้อ:\033[0m", len(warnings))
        for w in warnings:
            log.warning("  - %s", w)
    return 0


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    setup_logging(a.verbose)
    try:
        return run(a)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("\n\033[31mหยุดทำงาน: %s\033[0m", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
