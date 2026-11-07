r"""สร้างโมเดล 3D จาก *คลิป* กล้องวนรอบตัวละคร

    .venv\Scripts\python.exe -m src.clipcli --video input\clip.mp4
    .venv\Scripts\python.exe -m src.clipcli --views 24 --rig blender

ต่างจาก ``src.cli`` อย่างไร
--------------------------
``src.cli`` รับภาพนิ่ง 4 มุมที่วาดมาให้ตั้งฉากกันพอดี ส่วนตัวนี้รับคลิปที่
กล้องวนรอบตัว แล้วต้องหาเองว่าเฟรมไหนกล้องอยู่มุมไหน ห่างแค่ไหน สูงแค่ไหน
ก่อนจะเอาเงาจากทุกมุมมาเฉือน voxel

สองโหมด
  hull  (ค่าเริ่มต้น) เฉือนจากหลายสิบมุมพร้อมกันด้วยสนามระยะ — หน้าตัดเป็น
        รูปจริงของตัวละคร ไม่ใช่วงรีที่เดาเอาจากสองแกนแบบเส้นทางภาพนิ่ง
  views ประกอบภาพสี่มุมหลักจากคลิปแล้วส่งเข้า ``src.cli`` ตามเดิม —
        ได้ของครบทุกอย่างที่เส้นทางภาพนิ่งมี (เซาะคอ ลากหาง ผังฝ่าเท้า)
        แต่เสียข้อมูลมุมเฉียงไปทั้งหมด
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from . import (clipcarve, clipdepth, clipmaterial, clipparts, cliptexture,
               clipviews, meshing, turntable, uvmap)
from .cli import _Fmt, banner, setup_logging
from .config import INPUT_DIR, OUTPUT_DIR, PipelineConfig

log = logging.getLogger("2Dto3D.clip")


def build_parser() -> argparse.ArgumentParser:
    d = PipelineConfig()
    p = argparse.ArgumentParser(
        prog="2Dto3D-clip",
        description="สร้างโมเดล 3D จากคลิปกล้องวนรอบตัวละคร",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", type=Path, default=INPUT_DIR / "clip.mp4",
                   help="ไฟล์คลิปต้นทาง")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR, help="โฟลเดอร์ผลลัพธ์")
    p.add_argument("--name", default="character2", help="ชื่อไฟล์ผลลัพธ์")
    p.add_argument("--cache", type=Path, default=None,
                   help="โฟลเดอร์เก็บเฟรมกับ mask (ค่าเริ่มต้น: output/clip_cache)")
    p.add_argument("--mode", choices=["hull", "views"], default="hull",
                   help="hull = เฉือนจากทุกมุมพร้อมกัน, views = ประกอบภาพ 4 มุมแล้วส่งเข้า src.cli")

    g = p.add_argument_group("การประเมินกล้อง")
    g.add_argument("--solve-step", type=int, default=1,
                   help="ใช้ทุกกี่เฟรมในขั้นประเมินกล้อง (1 = ทุกเฟรม แม่นที่สุด)")
    g.add_argument("--reuse", action="store_true",
                   help="ใช้ผลประเมินกล้องที่เก็บไว้ในแคช ถ้ามี (ข้ามขั้นที่ช้าที่สุด)")
    g.add_argument("--no-fit-angles", action="store_true",
                   help="ไม่ต้องดัดจังหวะการวน (เชื่อว่ากล้องวนด้วยความเร็วคงที่)")
    g.add_argument("--fit-lens", action="store_true",
                   help="ลองหาระยะกล้อง/ความสูงสายตาที่ทำให้เงาสอดคล้องที่สุด")
    g.add_argument("--refine", type=int, default=2,
                   help="จำนวนรอบที่ดัดมาตราส่วน/ตำแหน่งของแต่ละเฟรมให้เข้ากับ hull")
    g.add_argument("--rembg-model", default=d.rembg_model, help="โมเดลแยกพื้นหลัง")

    g = p.add_argument_group("ภาพนิ่งเสริม")
    g.add_argument("--stills", type=Path, default=None, metavar="DIR",
                   help="โฟลเดอร์ภาพนิ่งสี่มุม (เช่น input) เอามาเป็นมุมเพิ่ม "
                        "ตอนเฉือน voxel และตอนฉายสี — มุมอ่านจากภาพเอง "
                        "ไม่ได้อ่านจากชื่อไฟล์")
    g.add_argument("--stills-min-iou", type=float, default=0.55,
                   help="ภาพนิ่งต้องทับเงาคลิปได้อย่างน้อยเท่านี้จึงจะเอามาใช้")
    g.add_argument("--stills-only-above", type=float, default=0.85,
                   metavar="FRAC",
                   help="เหนือความสูงนี้ (สัดส่วนของตัว) ให้ภาพนิ่งเป็นคนตัดสิน"
                        "รูปทรงแทนเฟรมคลิป — ช่วงหัวที่คลิปมักถ่ายไม่ติด "
                        "(1.0 = ไม่ใช้กลไกนี้)")

    g = p.add_argument_group("รูปทรง (โหมด hull)")
    g.add_argument("--views", type=int, default=24,
                   help="จำนวนมุมที่เอามาเฉือน กระจายเท่า ๆ กันรอบตัว")
    g.add_argument("--drop", type=int, default=2,
                   help="ยอมให้กี่มุมค้านได้ ก่อนจะเชื่อว่าตรงนั้นไม่มีเนื้อ")
    g.add_argument("--voxel-res", type=int, default=288,
                   help="จำนวน voxel ตามความสูง")
    g.add_argument("--field-smooth", type=float, default=2.0,
                   help="เกลาสนามระยะกี่ voxel ก่อนดึงผิว")
    g.add_argument("--tol-views", type=int, default=8,
                   help="ต้องมีกี่มุมเห็น ถึงจะยอมให้ค้านได้เพิ่มอีกหนึ่งมุม")
    g.add_argument("--open", type=int, default=2, dest="open_iters",
                   help="ลบแผ่นบางกว่ากี่ voxel (เงาบนพื้น/ขอบเฟรม)")
    g.add_argument("--height", type=float, default=d.character_height_m,
                   help="ความสูงจริงของตัวละคร (เมตร)")
    g.add_argument("--smooth", type=int, default=d.taubin_iters,
                   help="จำนวนรอบ Taubin smoothing")
    g.add_argument("--target-tris", type=int, default=d.target_tris,
                   help="งบ triangle ของ mesh สุดท้าย")

    g = p.add_argument_group("ผลลัพธ์")
    g.add_argument("--texture-size", type=int, default=d.uv_texture_size)
    g.add_argument("--texture-glb-size", type=int, default=d.texture_glb_size,
                   help="ย่อ texture ที่ฝังใน GLB ให้ไม่เกินกี่พิกเซล "
                        "(0 = ไม่ย่อ) ไฟล์ PNG บนดิสก์ยังเป็นความละเอียดเต็ม "
                        "— แผนที่ 4096 สามใบทำให้ GLB โตถึง 15 MB ซึ่งหนัก "
                        "เกินไปสำหรับหน้าเว็บ")
    g.add_argument("--texture-mode", choices=["uv", "vertex"], default=d.texture_mode)
    g.add_argument("--texture-views", type=int, default=48,
                   help="จำนวนเฟรมที่เอามาฉายสี (มากกว่าตอนเฉือนได้ ไม่เสียรูปทรง)")
    g.add_argument("--texture-blend", type=float, default=6.0,
                   help="ยิ่งมากยิ่งคม: 0 = เฉลี่ยสองมุมที่ดีที่สุดเท่า ๆ กัน, "
                        "สูง = ใช้มุมที่ดีที่สุดมุมเดียวเกือบตลอด")
    g.add_argument("--normal-strength", type=float, default=1.0,
                   help="ความแรงของ normal map ที่สร้างจากลายบน texture "
                        "(0 = ไม่ต้องสร้าง) สูงเกินผิวจะดูเป็นรอยบุบ")
    g.add_argument("--no-parts", action="store_true",
                   help="ไม่ต้องแยกกฎตามส่วนของตัว (นิ้ว ข้อมือ คอ) ใช้กฎเดียวทั้งตัว")
    g.add_argument("--displace", type=float, default=1.0,
                   help="ตัวคูณความลึกตอนดันผิวตามความเข้มของสี (0 = ไม่ดัน)")
    g.add_argument("--displace-tris", type=int, default=260_000,
                   help="งบสามเหลี่ยมหลังซอยเฉพาะส่วนที่มีรายละเอียด")
    g.add_argument("--no-materials", action="store_true",
                   help="ไม่ต้องแยกวัสดุแล้วใส่ลายเกล็ด/ผ้า/หนัง กับความมันของแต่ละวัสดุ")
    g.add_argument("--detail-scale", type=float, default=1.0,
                   help="ตัวคูณขนาดลายวัสดุ (เล็กลง = เกล็ดถี่ขึ้น)")
    g.add_argument("--detail-strength", type=float, default=0.35,
                   help="ความลึกของลายวัสดุที่สังเคราะห์ขึ้น")
    g.add_argument("--no-occlusion", action="store_true",
                   help="ไม่ต้องตรวจว่าจุดบนผิวถูกแขน/หางบังในมุมนั้นหรือไม่")
    g.add_argument("--rig", choices=["none", "blender"], default="none")
    g.add_argument("--no-anim", action="store_true", help="rig อย่างเดียว ไม่ใส่อนิเมชัน")
    g.add_argument("--formats", default="glb")
    g.add_argument("--no-draco", action="store_true")
    g.add_argument("--no-debug", action="store_true")
    g.add_argument("-v", "--verbose", action="store_true")
    return p


def _config(a: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        rembg_model=a.rembg_model,
        texture_mode=a.texture_mode,
        uv_texture_size=a.texture_size,
        texture_glb_size=a.texture_glb_size,
        draco=not a.no_draco,
        voxel_res=a.voxel_res,
        taubin_iters=a.smooth,
        target_tris=a.target_tris,
        character_height_m=a.height,
        rig=a.rig,
        animate=not a.no_anim,
        export_formats=tuple(f.strip() for f in a.formats.split(",") if f.strip()),
        debug=not a.no_debug,
    )


def _even_angles(cams, n: int) -> list[int]:
    """เลือกเฟรมที่มุมกระจายเท่า ๆ กันรอบตัว คืน index ในลิสต์"""
    picked: list[int] = []
    for want in np.arange(n) * (2 * np.pi / n):
        j = min(range(len(cams)),
                key=lambda i: abs(np.angle(np.exp(1j * (cams[i].theta - want)))))
        if j not in picked:
            picked.append(j)
    return sorted(picked)


def solve_clip(a: argparse.Namespace, cfg: PipelineConfig, cache: Path):
    """ดึงเฟรม แยกพื้นหลัง แล้วประเมินกล้องของทุกเฟรม"""
    banner(1, 6, "อ่านคลิป: ดึงเฟรม + แยกพื้นหลัง")
    indices = turntable.extract_frames(a.video, cache, step=a.solve_step)
    frames = turntable.segment_frames(indices, cache, cfg.rembg_model)

    banner(2, 6, "ประเมินกล้อง: มาตราส่วน ระดับพื้น และมุมของทุกเฟรม")
    saved = cache / "solution.json"
    if a.reuse:
        sol = turntable.load_solution(saved, turntable.measure(frames))
        if sol is not None:
            return frames, sol

    sol = turntable.solve(frames)
    if not a.no_fit_angles:
        sol = clipcarve.fit_angles(frames, sol)
    if a.fit_lens:
        sol.cams, _, _ = clipcarve.fit_lens(frames, sol.cams, sol.stats)
    turntable.save_solution(sol, saved)
    return frames, sol


def run_hull(a, cfg, frames, sol, out_dir: Path, cache: Path) -> int:
    pick = _even_angles(sol.cams, a.views)
    F = [frames[i] for i in pick]
    C = [sol.cams[i] for i in pick]
    S = [sol.stats[i] for i in pick]

    # ภาพนิ่งเข้ามาก่อนขั้น refine เพื่อให้ถูกดัดมาตราส่วนไปพร้อมกับเฟรมคลิป
    # โดยมีเฟรมคลิปจำนวนมากกว่าเป็นตัวตรึงค่ากลางไว้
    stills = []
    if a.stills is not None:
        banner(2, 6, "เสริมภาพนิ่งสี่มุมเข้ากับชุดมุมของคลิป")
        from . import clipstills
        stills = clipstills.build(a.stills, cache, frames, sol.cams, cfg,
                                  min_iou=a.stills_min_iou)
        if stills:
            if cfg.debug:
                clipstills.debug_sheet(stills, frames, sol.cams, cfg,
                                       out_dir / "debug" / "07_stills.png")
            F += [s.frame for s in stills]
            C += [s.cam for s in stills]
            S += [s.stats for s in stills]

    log.info("ใช้ %d มุมในการเฉือน: %s", len(C),
             ", ".join(f"{np.degrees(c.theta) % 360:.0f}°" for c in C[:12])
             + (" ..." if len(C) > 12 else ""))

    banner(3, 6, "Visual hull: เฉือน voxel ด้วยเงาจากทุกมุม")
    if a.refine > 0:
        bounds = clipcarve.bounds_from(C, S)
        ys, xs, zs, _ = clipcarve._axes(bounds, 128)
        for i in range(a.refine):
            field = clipcarve.sdf_field(F, C, ys, xs, zs, drop=a.drop)
            C = clipcarve.refine(F, C, field > 0, ys, xs, zs)[0]
            bounds = clipcarve.bounds_from(C, S)
            ys, xs, zs, _ = clipcarve._axes(bounds, 128)
        if stills:
            # ขั้น refine คืนกล้องชุดใหม่ ต้องอัปเดตฝั่งภาพนิ่งด้วย ไม่งั้น
            # ขั้นฉายสีจะใช้มาตราส่วนเก่าแล้วลายผิวเลื่อนไปจากรูปทรง
            for st, cam in zip(stills, C[len(pick):]):
                st.cam = cam

    # เหนือแนวคอขึ้นไป เฟรมคลิปมักถ่ายไม่ติดหรือมาตราส่วนเพี้ยน ให้ภาพนิ่ง
    # ที่เห็นหัวเต็ม ๆ เป็นคนตัดสินรูปทรงแทน (ดู clipcarve.sdf_field)
    trust = solo_above = None
    if stills and a.stills_only_above < 1.0:
        trust = [False] * len(pick) + [True] * len(stills)
        solo_above = a.stills_only_above

    grid, field, axes = clipcarve.carve_sdf(
        F, C, S, cfg, a.voxel_res, drop=a.drop, smooth=a.field_smooth,
        open_iters=a.open_iters, tol_views=a.tol_views,
        trust=trust, solo_above=solo_above)

    # ---- แยกกฎตามส่วนของตัว ---------------------------------------------
    # ขั้นเก็บกวาดที่จำเป็นกับลำตัว (ลบแผ่นบาง เกลาสนาม) เป็นตัวทำลายมือ
    # กับเท้า จึงอ่านโครงกระดูกจากกริดหยาบก่อน แล้วประกอบกริดใหม่โดยยกเว้น
    # เขตเล็ก ๆ เหล่านั้น — ใช้สนามระยะเดิม ไม่ต้องเฉือนซ้ำ
    from . import landmarks

    parts = None
    if not a.no_parts:
        rough = clipcarve._to_hull_grid(
            (field > 0).astype(np.float32),
            clipcarve.bounds_from(C, S), axes[0][1] - axes[0][0], cfg)
        try:
            skel0 = landmarks.estimate(rough, cfg)
        except ValueError as exc:
            log.warning("อ่านโครงกระดูกจากกริดหยาบไม่ได้ (%s) — ข้ามการแยกส่วน", exc)
        else:
            unit_m = (axes[0][1] - axes[0][0]) * cfg.character_height_m
            protect = clipparts.voxel_protect(
                field.shape, skel0.joints, skel0.parents, unit_m)
            grid = clipcarve.assemble(
                field, clipcarve.bounds_from(C, S), axes[0][1] - axes[0][0],
                cfg, smooth=a.field_smooth, open_iters=a.open_iters,
                protect=protect)
            parts = (skel0, unit_m)

    banner(4, 6, "สร้าง mesh: marching cubes -> smooth -> decimate")
    mesh = meshing.build(grid, cfg)

    banner(5, 6, "ฉายสีจากเฟรมของคลิปลงบนผิว")
    tex_pick = _even_angles(sol.cams, max(a.texture_views, a.views))
    TF = [frames[i] for i in tex_pick]
    TC = [sol.cams[i] for i in tex_pick]
    if stills:
        # ภาพนิ่งคมกว่าเฟรมคลิปเกือบสองเท่า ใส่เข้าไปในชุดที่ให้เลือกตอนฉายสี
        # ด้วย — ขั้น bake เลือกมุมที่ผิวหันเข้าหามากที่สุดอยู่แล้ว
        TF += [s.frame for s in stills]
        TC += [s.cam for s in stills]
        log.info("ฉายสีจาก %d เฟรมคลิป + ภาพนิ่ง %d ใบ", len(tex_pick),
                 len(stills))
    tex_img = nrm_img = rough_img = None
    if cfg.texture_mode == "uv":
        mesh = uvmap.unwrap(mesh, cfg)
        baked = cliptexture.bake(
            mesh, grid, axes, TF, TC, cfg, blend=a.texture_blend,
            occlusion=not a.no_occlusion, normal_strength=a.normal_strength)
        tex_img, nrm_img = baked.color, baked.normal

        if a.displace > 0:
            banner(5, 6, "ดันผิวตามความเข้มของสี ทีละส่วนของตัว")
            height = cliptexture.shading_height(baked.color, baked.filled)
            step = axes[0][1] - axes[0][0]
            unit_m = parts[1] if parts else step * cfg.character_height_m
            if parts:
                labels = clipparts.label(mesh.verts, parts[0].joints,
                                         parts[0].parents,
                                         cfg.character_height_m, unit_m)
                amp = clipparts.per_point(mesh.verts, labels, "depth_mm")
                edge = clipparts.per_point(mesh.verts, labels, "edge_mm")
            else:
                amp = np.full(len(mesh.verts), 2.0)
                edge = np.full(len(mesh.verts), 8.0)
            to_vox = 1.0 / (1000.0 * unit_m)
            mesh = clipdepth.displace(
                mesh, height, amp * a.displace * to_vox, edge * to_vox,
                budget=a.displace_tris)

        if not a.no_materials:
            banner(5, 6, "แยกวัสดุจากลาย แล้วใส่ลายผิวของวัสดุนั้น")
            labels = clipmaterial.classify(baked.color, baked.filled)
            detail = clipmaterial.detail_height(labels, scale=a.detail_scale)
            rough_img = clipmaterial.roughness_map(labels, detail)
            dn = cliptexture.normal_from_height(detail * a.detail_strength)
            nrm_img = dn if nrm_img is None else                 clipmaterial.combine_normals(nrm_img, dn)
            if cfg.debug:
                clipmaterial.debug_sheet(
                    baked.color, labels, detail,
                    out_dir / "debug" / "06_materials.png")

        import cv2
        for img, suffix in ((tex_img, "texture"), (nrm_img, "normal"),
                            (rough_img, "rough")):
            if img is None:
                continue
            path = out_dir / f"{a.name}_{suffix}.png"
            ok, buf = cv2.imencode(".png", img[:, :, ::-1])
            if ok:
                buf.tofile(str(path))
                log.info("เขียน %-7s -> %s (%.2f MB)", suffix, path,
                         path.stat().st_size / 1e6)
    else:
        mesh = cliptexture.vertex_colors(mesh, grid, axes, TF, TC, cfg,
                                         blend=a.texture_blend,
                                         occlusion=not a.no_occlusion)

    banner(6, 6, "ประเมินข้อต่อ แล้วเขียนไฟล์ผลลัพธ์")
    from . import export as export_mod
    from . import landmarks

    warnings: list[str] = []
    skel = landmarks.estimate(grid, cfg)
    warnings.extend(skel.notes)
    verts_world = meshing.to_world(mesh, grid, cfg.character_height_m)
    if cfg.debug:
        landmarks.draw_debug(grid, skel, out_dir / "debug" / "03_skeleton.png")
    joints_path = landmarks.write_json(skel, mesh.verts, verts_world,
                                       out_dir / f"{a.name}_joints.json")

    written = export_mod.export(verts_world, mesh, out_dir, a.name,
                                cfg.export_formats, texture=tex_img,
                                normal=nrm_img, rough=rough_img)
    if cfg.texture_quality > 0:
        for p in written.values():
            if p.suffix.lower() == ".glb":
                export_mod.recompress_textures(p, cfg.texture_quality,
                                               cfg.texture_glb_size)
    export_mod.write_summary(
        out_dir / f"{a.name}_summary.json", mesh, verts_world,
        extra={
            "source": str(a.video),
            "mode": "hull",
            "views_used": [int(sol.cams[i].index) for i in pick],
            "view_angles": [round(float(np.degrees(sol.cams[i].theta) % 360), 1)
                            for i in pick],
            "voxel_grid": list(grid.shape),
            "warnings": warnings,
            "joints": str(joints_path),
        })

    if cfg.rig == "blender":
        _rig(a, cfg, skel, written["glb"], joints_path, out_dir)
    for w in warnings:
        log.warning("ข้อควรตรวจ: %s", w)
    return 0


def _rig(a, cfg, skel, glb_path: Path, joints_path: Path, out_dir: Path) -> None:
    from . import animation, export as export_mod
    from .rigging import rig_with_blender

    anim_path = None
    if cfg.animate:
        n_tail = sum(1 for n in skel.joints
                     if n.startswith("Tail") and n != "TailEnd")
        clips = animation.default_clips(n_tail)
        anim_path = out_dir / f"{a.name}_clips.json"
        anim_path.write_text(json.dumps(animation.to_payload(clips, cfg.fps),
                                        indent=2, ensure_ascii=False),
                             encoding="utf-8")
    rigged = rig_with_blender(glb_path, joints_path,
                              out_dir / f"{a.name}_rigged.glb", anim_path,
                              draco=cfg.draco)
    if cfg.texture_quality > 0:
        export_mod.recompress_textures(rigged, cfg.texture_quality,
                                       cfg.texture_glb_size)
    log.info("ได้ไฟล์ที่ rig แล้ว: %s (%.2f MB)", rigged,
             rigged.stat().st_size / 1e6)


def run_views(a, cfg, frames, sol, out_dir: Path) -> int:
    """ประกอบภาพสี่มุมจากคลิป แล้วส่งต่อให้ pipeline ของภาพนิ่ง"""
    banner(3, 4, "ประกอบภาพสี่มุมหลักจากคลิป")
    views = clipviews.build_views(frames, sol.cams, sol.stats, cfg)
    view_dir = out_dir / "clip_views"
    clipviews.write_views(views, view_dir)
    if cfg.debug:
        clipviews.debug_sheet(views, out_dir / "debug" / "05_clip_views.png")

    banner(4, 4, "ส่งต่อให้ pipeline ภาพนิ่ง (src.cli)")
    from . import cli as still_cli

    argv = ["--input", str(view_dir), "--output", str(out_dir),
            "--name", a.name, "--height", str(a.height),
            "--rig", a.rig, "--formats", a.formats]
    if a.no_anim:
        argv.append("--no-anim")
    if a.no_draco:
        argv.append("--no-draco")
    if a.no_debug:
        argv.append("--no-debug")
    return still_cli.run(still_cli.build_parser().parse_args(argv))


def run(a: argparse.Namespace) -> int:
    cfg = _config(a)
    out_dir: Path = a.output
    out_dir.mkdir(parents=True, exist_ok=True)
    # แคชอยู่ที่เดิมเสมอ ไม่ผูกกับ --output เพราะการแยกพื้นหลังทั้งคลิป
    # ใช้เวลาหลายนาที จะได้ใช้ซ้ำได้แม้เปลี่ยนโฟลเดอร์ผลลัพธ์
    cache = a.cache or (OUTPUT_DIR / "clip_cache")
    t0 = time.perf_counter()

    frames, sol = solve_clip(a, cfg, cache)
    turntable.save_cameras(sol.cams, cache / "cameras.json")

    rc = (run_hull(a, cfg, frames, sol, out_dir, cache) if a.mode == "hull"
          else run_views(a, cfg, frames, sol, out_dir))

    log.info("\n\033[1mเสร็จใน %.0f วินาที\033[0m — ผลลัพธ์อยู่ที่ %s",
             time.perf_counter() - t0, out_dir)
    return rc


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
