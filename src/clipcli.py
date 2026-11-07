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

from . import clipcarve, cliptexture, clipviews, meshing, turntable, uvmap
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
    g.add_argument("--texture-mode", choices=["uv", "vertex"], default=d.texture_mode)
    g.add_argument("--texture-views", type=int, default=48,
                   help="จำนวนเฟรมที่เอามาฉายสี (มากกว่าตอนเฉือนได้ ไม่เสียรูปทรง)")
    g.add_argument("--texture-blend", type=float, default=6.0,
                   help="ยิ่งมากยิ่งคม: 0 = เฉลี่ยสองมุมที่ดีที่สุดเท่า ๆ กัน, "
                        "สูง = ใช้มุมที่ดีที่สุดมุมเดียวเกือบตลอด")
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


def run_hull(a, cfg, frames, sol, out_dir: Path) -> int:
    pick = _even_angles(sol.cams, a.views)
    F = [frames[i] for i in pick]
    C = [sol.cams[i] for i in pick]
    S = [sol.stats[i] for i in pick]
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

    grid, field, axes = clipcarve.carve_sdf(
        F, C, S, cfg, a.voxel_res, drop=a.drop, smooth=a.field_smooth,
        open_iters=a.open_iters, tol_views=a.tol_views)

    banner(4, 6, "สร้าง mesh: marching cubes -> smooth -> decimate")
    mesh = meshing.build(grid, cfg)

    banner(5, 6, "ฉายสีจากเฟรมของคลิปลงบนผิว")
    tex_pick = _even_angles(sol.cams, max(a.texture_views, a.views))
    TF = [frames[i] for i in tex_pick]
    TC = [sol.cams[i] for i in tex_pick]
    tex_img = None
    if cfg.texture_mode == "uv":
        mesh = uvmap.unwrap(mesh, cfg)
        tex_img = cliptexture.bake(mesh, grid, axes, TF, TC, cfg,
                                   blend=a.texture_blend,
                                   occlusion=not a.no_occlusion)
        import cv2
        tex_path = out_dir / f"{a.name}_texture.png"
        ok, buf = cv2.imencode(".png", tex_img[:, :, ::-1])
        if ok:
            buf.tofile(str(tex_path))
            log.info("เขียน texture -> %s (%.2f MB)", tex_path,
                     tex_path.stat().st_size / 1e6)
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
                                cfg.export_formats, texture=tex_img)
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

    rc = (run_hull(a, cfg, frames, sol, out_dir) if a.mode == "hull"
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
