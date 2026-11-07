r"""โมเดลที่ 4: ซ้อนชุดภาพสี่ชั้น แล้วให้ Blender เป็นคนปั้น

    .venv\Scripts\python.exe -m src.stackcli --name character4 --height 2.1

สิ่งที่ต่างจากสามโมเดลก่อนหน้า

* **ภาพหนึ่งใบไม่ต้องทำทุกอย่างอีกแล้ว** — ``src/stackviews.py`` รับชุดภาพ
  สี่ชั้นของมุมเดียวกัน (ต้นฉบับ / เรนเดอร์ดินเหนียว / สีแบน / ภาพเสมือนถ่าย)
  ดัดให้ลงทะเบียนตรงกัน แล้วให้แต่ละชั้นออกเสียงเฉพาะเรื่องที่ตัวเองเชื่อถือได้
* **ขั้นปั้นย้ายไปอยู่ใน Blender** — ``src/blender_build.py`` remesh ด้วย voxel
  ของ Blender, shrinkwrap กลับไปเกาะเส้นขอบเดิม, ซอยผิว แล้วดันผิวด้วย
  Displace modifier ตามแผนที่ความลึกที่ bake มาจากชั้นดินเหนียว

ลำดับงาน

====== ================================================= ======================
ขั้น   ทำอะไร                                             ใช้ชั้นไหน
====== ================================================= ======================
1      segment + ดัดทุกชั้นให้ลงทะเบียนตรงกัน               ทุกชั้น
2      เฉือน visual hull ด้วยเงา "เสียงข้างมาก"             ทุกชั้นที่ผ่านเกณฑ์
3      Blender: voxel remesh + shrinkwrap + เกลา            —
4      คลี่ UV แล้ว bake สีของทุกชั้นลง atlas ใบเดียวกัน     ทุกชั้น
5      แปลงแสงเงาดินเหนียวเป็นความลึก + ลายผิวเป็น normal    base, photo, albedo
6      Blender: ซอยผิว + Displace + ต่อวัสดุ + export        —
7      ประเมินข้อต่อ แล้ว rig                                —
====== ================================================= ======================
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import animation, basefit, clipmaterial, cliptexture, landmarks
from . import export as export_mod
from . import meshing, soleplan, stackviews, texture, uvmap, visual_hull
from .config import BLENDER_EXE, INPUT_DIR, OUTPUT_DIR, PipelineConfig

log = logging.getLogger("2Dto3D.stack")


def banner(step: int, total: int, title: str) -> None:
    log.info("\n\033[1m[%d/%d] %s\033[0m", step, total, title)


# ---------------------------------------------------------------------------
# เรียก Blender
# ---------------------------------------------------------------------------


def run_blender(stage: str, args: list[str], expect: Path, exe: str,
                timeout: int = 2400) -> Path:
    """รัน ``src/blender_build.py`` หนึ่งขั้น แล้วตรวจว่าได้ไฟล์จริง

    Blender คืน exit code 0 ได้แม้สคริปต์จะ traceback กลางคัน จึงต้องให้
    สคริปต์ประกาศความสำเร็จของตัวเอง ไม่งั้นจะได้ไฟล์เก่าจากการรันครั้งก่อน
    มาแทนแล้วนึกว่าสำเร็จ (บทเรียนเดียวกับขั้น rig)
    """
    blender = Path(exe)
    if not blender.exists():
        raise FileNotFoundError(
            f"ไม่พบ Blender ที่ {blender}\n"
            "ตั้ง environment variable BLENDER_EXE ให้ชี้ไปที่ blender.exe")

    script = Path(__file__).resolve().parent / "blender_build.py"
    expect.parent.mkdir(parents=True, exist_ok=True)
    stamp = expect.stat().st_mtime if expect.exists() else 0.0

    cmd = [str(blender), "--background", "--factory-startup",
           "--python", str(script), "--", "--stage", stage] + args
    log.info("เรียก Blender (ขั้น %s)", stage)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace")

    out = proc.stdout or ""
    for line in out.splitlines():
        if line.startswith("[build]"):
            log.info("  %s", line[7:].strip())

    done = "[build] เสร็จสมบูรณ์" in out
    fresh = expect.exists() and expect.stat().st_mtime > stamp
    if proc.returncode != 0 or not done or not fresh:
        why = ("สคริปต์หยุดกลางคัน (ไม่มีสัญญาณจบงาน)" if proc.returncode == 0
               else f"exit {proc.returncode}")
        if done and not fresh:
            why = f"สคริปต์บอกว่าจบแล้วแต่ไม่ได้เขียน {expect.name} ใหม่"
        tail = "\n".join(out.splitlines()[-30:])
        err = "\n".join((proc.stderr or "").splitlines()[-15:])
        raise RuntimeError(f"Blender ขั้น {stage} ไม่สำเร็จ: {why}\n"
                           f"--- stdout ---\n{tail}\n--- stderr ---\n{err}")
    log.info("Blender ขั้น %s ใช้เวลา %.1f วินาที", stage, time.time() - t0)
    return expect


def read_glb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """อ่าน mesh กลับจาก GLB ที่ Blender เขียน คืน (verts_world, faces)"""
    import trimesh

    scene = trimesh.load(str(path), process=False)
    if isinstance(scene, trimesh.Scene):
        # ต้องใช้ dump() ไม่ใช่ scene.geometry โดยตรง — Blender เขียน mesh ไว้
        # ใต้ node ที่มี transform ของตัวเอง ถ้าหยิบ geometry ตรง ๆ จะได้พิกัด
        # ก่อนคูณ transform แล้วตัวละครจะไปโผล่ผิดที่/ผิดมาตราส่วนแบบเงียบ ๆ
        mesh = (scene.to_geometry() if hasattr(scene, "to_geometry")
                else scene.dump(concatenate=True))
    else:
        mesh = scene
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise RuntimeError(f"ไม่พบ mesh ใน {path}")
    return (np.asarray(mesh.vertices, np.float64),
            np.asarray(mesh.faces, np.int32))


# ---------------------------------------------------------------------------
# bake ทุกชั้นลง atlas ใบเดียวกัน
# ---------------------------------------------------------------------------


def bake_layers(mesh: meshing.MeshData, grid: visual_hull.HullGrid,
                sets: dict[str, dict], cfg: PipelineConfig,
                ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """ฉายสีของทุกชั้นลง atlas ใบเดียวกัน โดย rasterize แค่ครั้งเดียว

    rasterize คือขั้นที่แพงที่สุดของการ bake (ไล่สามเหลี่ยมทุกใบลงตาราง texel)
    และผลของมันไม่ขึ้นกับว่าจะเอาสีของชั้นไหนมาฉาย — ทุกชั้นใช้ mesh, UV และ
    เงาชุดเดียวกันหมด ต่างแค่ภาพ RGB  ทำครั้งเดียวแล้วใช้ซ้ำจึงเร็วขึ้นเท่ากับ
    จำนวนชั้น และสำคัญกว่านั้นคือ **ทุกแผนที่ลงทะเบียนตรงกันแบบรับประกัน**:
    texel เดียวกันบนทุกใบคือจุดเดียวกันบนตัวละคร
    """
    size = int(cfg.uv_texture_size)
    pos, nrm, filled = uvmap._rasterize(mesh, size)
    if not filled.any():
        raise RuntimeError("ไม่มี texel ไหนตกในสามเหลี่ยมเลย — UV อาจเสีย")
    idx = np.flatnonzero(filled)
    mask2d = filled.reshape(size, size)
    log.info("rasterize ครั้งเดียว: %d texel อยู่บนผิว (%.1f%% ของ atlas)",
             len(idx), 100 * len(idx) / (size * size))

    out: dict[str, np.ndarray] = {}
    for name, views in sets.items():
        colors = texture.project_colors(pos[idx], nrm[idx], grid, views, cfg)
        img = np.zeros((size * size, 3), np.uint8)
        img[idx] = colors
        out[name] = uvmap.pad_islands(img.reshape(size, size, 3), mask2d,
                                      cfg.uv_padding)
        log.info("  bake ชั้น %-7s -> atlas %dx%d", name, size, size)
    return out, mask2d


# ---------------------------------------------------------------------------
# ลบเส้น wireframe ออกจากภาพเรนเดอร์ดินเหนียว
# ---------------------------------------------------------------------------


def strip_wireframe(img: np.ndarray, width: int) -> np.ndarray:
    """ลบเส้นตารางของ topology ที่ติดมากับภาพเรนเดอร์ดินเหนียว

    ชั้น ``base mesh`` เรนเดอร์มาพร้อม wireframe ของ quad ทุกเส้น ซึ่งเป็น
    *ข้อมูลของโมเดลต้นทาง* ไม่ใช่ความขรุขระของผิว ถ้าปล่อยไว้ ขั้นอ่าน
    ความลึกจากแสงเงาจะอ่านเส้นพวกนั้นเป็นร่อง แล้ว Displace modifier จะดัน
    ผิวจริงให้กลายเป็นผ้านวมตารางทั้งตัว (เห็นชัดมากในภาพเรนเดอร์ clay)

    เส้นเหล่านี้มีลักษณะเฉพาะที่ใช้ลบได้โดยไม่แตะรูปทรงจริง: **มืดกว่ารอบข้าง
    และบางกว่าทุกอย่างที่เป็นของจริงบนผิว** morphological closing (ขยายแล้ว
    หด) ลบโครงสร้าง *มืด* ที่แคบกว่า kernel ออกตามนิยาม ส่วนสันเกล็ดกับรอยยับ
    ผ้าซึ่งกว้างกว่านั้นยังอยู่ครบ

    ข้อแลกเปลี่ยนที่ต้องรู้: ร่องจริงที่แคบพอ ๆ กับเส้น wireframe (ร่องระหว่าง
    นิ้ว ตะเข็บเสื้อ) ก็ตื้นขึ้นด้วย — ที่ atlas 2048 ของตัวสูงสองเมตร หนึ่ง
    texel ราว 1.5 มม. kernel 5 texel จึงลบได้เฉพาะร่องที่แคบกว่า 7 มม.
    """
    k = max(3, int(width) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    log.info("ลบเส้น wireframe ด้วย closing %dx%d texel "
             "(ความสว่างเฉลี่ยขึ้น %.1f ระดับ)", k, k,
             float(out.mean()) - float(img.mean()))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    d = PipelineConfig()
    p = argparse.ArgumentParser(
        prog="2Dto3D-stack",
        description="โมเดลที่ 4: ซ้อนชุดภาพสี่ชั้น แล้วปั้นด้วย Blender",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", type=Path, default=INPUT_DIR)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    p.add_argument("--name", default="character4")
    p.add_argument("--height", type=float, default=2.1,
                   help="ความสูงจริงของตัวละคร (เมตร)")

    g = p.add_argument_group("ชั้นภาพ")
    g.add_argument("--min-iou", type=float, default=0.72,
                   help="ชั้นที่ทับชั้นอ้างอิงได้ต่ำกว่านี้ ถือว่าไม่ใช่มุมเดียวกัน")
    g.add_argument("--agree", type=float, default=0.5,
                   help="ต้องได้เสียงกี่ส่วนของชั้นที่โหวต จึงนับเป็นเนื้อ")
    g.add_argument("--max-scale", type=float, default=0.08,
                   help="ยอมให้ดัดมาตราส่วนของชั้นได้ไม่เกินนี้")

    g = p.add_argument_group("รูปทรง")
    g.add_argument("--voxel-res", type=int, default=d.voxel_res)
    g.add_argument("--remesh-res", type=int, default=240,
                   help="ความละเอียด voxel remesh ของ Blender (ต่อความสูงตัว)")
    g.add_argument("--cage-tris", type=int, default=140_000,
                   help="จำนวนสามเหลี่ยมของ cage ที่ส่งไปคลี่ UV")
    g.add_argument("--final-tris", type=int, default=400_000,
                   help="จำนวนสามเหลี่ยมของโมเดลที่ export")
    g.add_argument("--subdiv", type=int, default=2,
                   help="ซอยผิวกี่ชั้นก่อนดันผิว (0 = ไม่ซอย)")
    g.add_argument("--amp", type=float, default=0.004,
                   help="ความลึกของการดันผิว เป็นสัดส่วนของความสูงตัวละคร")
    g.add_argument("--form-sigma", type=float, default=16.0,
                   help="ขอบบนของย่านความถี่ที่อ่านเป็นรูปทรงจากแสงเงาดินเหนียว")
    g.add_argument("--form-blur", type=float, default=2.5,
                   help="ขอบล่างของย่านนั้น (เบลอทิ้งก่อน เพื่อตัด noise ระดับ texel)")
    g.add_argument("--wire-close", type=int, default=5,
                   help="ขนาด kernel (texel) ที่ใช้ลบเส้น wireframe ของชั้นดินเหนียว")
    g.add_argument("--no-displace", action="store_true",
                   help="ข้ามการดันผิว (ได้ cage เรียบ ๆ)")

    g = p.add_argument_group("ผิว")
    g.add_argument("--texture-size", type=int, default=d.uv_texture_size)
    g.add_argument("--detail-strength", type=float, default=1.1,
                   help="ความแรงของ normal ที่มาจากลายผิวภาพเสมือนถ่าย")
    g.add_argument("--pattern-scale", type=float, default=1.0,
                   help="ตัวคูณขนาดลายเกล็ด/ผ้า/หนังที่สังเคราะห์ขึ้น")
    g.add_argument("--no-materials", action="store_true")

    g = p.add_argument_group("ปลายทาง")
    g.add_argument("--rig", choices=["blender", "none"], default="blender")
    g.add_argument("--no-animate", action="store_true")
    g.add_argument("--blender", default=BLENDER_EXE)
    g.add_argument("--keep-steps", action="store_true",
                   help="เก็บไฟล์ระหว่างทาง (hull / cage) ไว้ตรวจ")
    g.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(a: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        voxel_res=a.voxel_res,
        character_height_m=a.height,
        uv_texture_size=a.texture_size,
        texture_mode="uv",
        rig=a.rig,
        animate=not a.no_animate,
    )


def run(a: argparse.Namespace) -> int:
    cfg = config_from_args(a)
    out_dir: Path = a.output
    debug_dir = out_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    warnings: list[str] = []
    total = 7 if cfg.rig == "none" else 8

    # --- 1. ชั้นภาพ ---------------------------------------------------------
    banner(1, total, "อ่านชุดภาพสี่ชั้น แล้วดัดให้ลงทะเบียนตรงกัน")
    layers = stackviews.load(a.input, cfg)
    warnings += stackviews.register(layers, cfg, max_scale=a.max_scale)
    masks, notes = stackviews.consensus(layers, min_iou=a.min_iou,
                                        agree=a.agree)
    warnings += notes
    if cfg.debug:
        stackviews.debug_sheet(layers, masks, debug_dir / "08_layers.png")

    # --- 2. visual hull ------------------------------------------------------
    banner(2, total, "เฉือน visual hull ด้วยเงาเสียงข้างมาก")
    carve_views = stackviews.carving_views(layers, masks)
    sole = soleplan.load(layers[stackviews.REFERENCE].folder, cfg)
    grid = visual_hull.carve(carve_views, cfg, sole)
    for note in visual_hull.carve_report(grid, carve_views):
        log.warning("คุณภาพ hull: %s", note)
        warnings.append(note)

    hull = meshing.build(grid, cfg)
    hull_world = meshing.to_world(hull, grid, cfg.character_height_m)
    hull_glb = out_dir / f"{a.name}_hull.glb"
    export_mod.export(hull_world, hull, out_dir, f"{a.name}_hull", ("glb",))

    # --- 3. Blender: remesh --------------------------------------------------
    banner(3, total, "Blender: voxel remesh + shrinkwrap + เกลาผิว")
    cage_glb = out_dir / f"{a.name}_cage.glb"
    run_blender("cage", [
        "--mesh", str(hull_glb), "--out", str(cage_glb),
        "--voxel-res", str(a.remesh_res),
        "--target-tris", str(a.cage_tris),
    ], cage_glb, a.blender)

    cage_world, cage_faces = read_glb(cage_glb)
    cage_vox = basefit.world_to_voxel(cage_world, hull.verts, hull_world)
    mesh = meshing.MeshData(verts=cage_vox, faces=cage_faces)
    mesh.normals = basefit.vertex_normals(mesh.verts, mesh.faces)
    mesh.log_stage("Blender voxel remesh")
    log.info("cage จาก Blender: %d verts / %d tris", mesh.n_verts, mesh.n_faces)

    # --- 4. คลี่ UV + bake ทุกชั้น -------------------------------------------
    banner(4, total, "คลี่ผิว แล้ว bake สีของทุกชั้นลง atlas ใบเดียวกัน")
    mesh = uvmap.unwrap(mesh, cfg)
    sets = {}
    for name in layers:
        views, tnotes = stackviews.texture_views(layers, name, masks,
                                                 min_iou=a.min_iou)
        warnings += tnotes
        sets[name] = views
    baked, filled = bake_layers(mesh, grid, sets, cfg)

    # --- 5. แผนที่ผิว --------------------------------------------------------
    banner(5, total, "อ่านความลึกจากแสงเงาดินเหนียว + ลายผิวจากภาพเสมือนถ่าย")
    color = baked["albedo"]

    # ความลึก: ชั้นดินเหนียวไม่มีสีของวัสดุเลย ความสว่างทุกจุดจึงเป็นเรื่อง
    # ของรูปทรงล้วน ๆ ซึ่งเป็นสมมติฐานที่ shading_height ต้องการพอดี
    # (กับภาพสีต้องเปิด chroma_guard ไว้กันขอบเสื้อกลายเป็นร่อง — ที่นี่ไม่ต้อง)
    clay = strip_wireframe(baked["base"], a.wire_close)

    # สองย่านความถี่ ไม่ใช่ย่านเดียว: `shading_height` เป็น high-pass (ลบ
    # ตัวเองที่เบลอด้วย sigma ออก) ซึ่งแปลว่า sigma *ยิ่งมากยิ่งเก็บของหยาบ
    # ไว้น้อยลงไม่ได้* — มันเก็บทุกอย่างที่ละเอียดกว่า sigma รวมทั้ง noise
    # ระดับ texel ด้วย  การเบลอก่อนแล้วค่อย high-pass จึงได้ band-pass จริง ๆ
    # (ย่าน form_blur..form_sigma) ซึ่งเป็นย่านของกล้ามเนื้อกับรอยยับผ้า
    form = cliptexture.shading_height(
        cv2.GaussianBlur(clay, (0, 0), a.form_blur), filled,
        sigma=a.form_sigma, chroma_guard=False)
    fine = cliptexture.shading_height(clay, filled, sigma=3.0,
                                      chroma_guard=False)
    height = np.clip(0.80 * form + 0.45 * fine, -1.0, 1.0)

    # ลายผิว: ภาพเสมือนถ่ายเก็บเกล็ด รอยด้าย หัวเข็มขัดไว้ครบ แต่ *สี* ของมัน
    # มีแสงสตูดิโอติดมาด้วย จึงใช้เฉพาะความถี่สูงมาเป็น normal ไม่เอาไปเป็นสีฐาน
    detail = cliptexture.shading_height(baked["photo"], filled, sigma=3.0)
    normal = cliptexture.normal_from_height(detail * a.detail_strength)

    rough = None
    if not a.no_materials:
        # แยกวัสดุจาก *ชั้นสีแบน* ไม่ใช่ภาพที่มีแสงเงา — k-means ใน Lab จัด
        # กลุ่มตามความสว่างได้ง่ายกว่าตามเฉดสี ภาพที่มีเงาจึงทำให้แขนข้างที่
        # อยู่ในเงากลายเป็นวัสดุคนละชนิดกับแขนอีกข้าง (โมเดลที่ 3 ต้องมีขั้น
        # `flatten_shading` มาแก้ทีหลัง ส่วนที่นี่ต้นทางไม่มีเงามาแต่แรก)
        labels = clipmaterial.classify(color, filled)
        tiles = clipmaterial.detail_height(labels, scale=a.pattern_scale)
        rough = clipmaterial.roughness_map(labels, tiles)
        normal = clipmaterial.combine_normals(
            normal, cliptexture.normal_from_height(tiles))
        if cfg.debug:
            clipmaterial.debug_sheet(color, labels, tiles,
                                     debug_dir / "09_materials.png")

    height_png = out_dir / f"{a.name}_height.png"
    maps = {"texture": color, "normal": normal, "rough": rough,
            "clay": baked["base"], "detail": baked["photo"]}
    for suffix, img in maps.items():
        if img is None:
            continue
        path = out_dir / f"{a.name}_{suffix}.png"
        ok, buf = cv2.imencode(".png", img[:, :, ::-1])
        if ok:
            buf.tofile(str(path))
            log.info("เขียน %-7s -> %s (%.2f MB)", suffix, path,
                     path.stat().st_size / 1e6)
    ok, buf = cv2.imencode(
        ".png", np.clip(height * 127.0 + 128.0, 0, 255).astype(np.uint8))
    if not ok:
        raise RuntimeError("เขียนแผนที่ความลึกไม่สำเร็จ")
    buf.tofile(str(height_png))
    log.info("เขียน height  -> %s (%.2f MB)", height_png,
             height_png.stat().st_size / 1e6)

    # --- 6. Blender: ปั้นจริง ------------------------------------------------
    banner(6, total, "Blender: ซอยผิว + ดันผิวตามความลึก + ต่อวัสดุ")
    cage_uv = out_dir / f"{a.name}_cage_uv.glb"
    mesh_world = meshing.to_world(mesh, grid, cfg.character_height_m)
    export_mod.export(mesh_world, mesh, out_dir, f"{a.name}_cage_uv", ("glb",),
                      texture=color)

    final_glb = out_dir / f"{a.name}.glb"
    args = ["--mesh", str(cage_uv), "--out", str(final_glb),
            "--subdiv", str(a.subdiv),
            "--amp", str(a.amp),
            "--target-tris", str(a.final_tris),
            "--color", str(out_dir / f"{a.name}_texture.png"),
            "--normal", str(out_dir / f"{a.name}_normal.png"),
            "--blend", str(out_dir / f"{a.name}.blend")]
    if not a.no_displace:
        args += ["--height", str(height_png)]
    if rough is not None:
        args += ["--rough", str(out_dir / f"{a.name}_rough.png")]
    run_blender("finish", args, final_glb, a.blender)
    if cfg.texture_quality > 0:
        # Blender ฝังแผนที่เป็น PNG ซึ่งใหญ่กว่าที่จำเป็นหลายเท่าสำหรับ
        # ไฟล์ที่จะเอาไปเปิดบนเว็บ — บีบเป็น JPEG ด้วยเกณฑ์เดียวกับโมเดลอื่น
        export_mod.recompress_textures(final_glb, cfg.texture_quality,
                                       cfg.texture_glb_size)
        log.info("GLB หลังบีบแผนที่: %.2f MB",
                 final_glb.stat().st_size / 1e6)

    # --- 7. ข้อต่อ -----------------------------------------------------------
    banner(7, total, "ประเมินตำแหน่งข้อต่อ")
    skel = landmarks.estimate(grid, cfg)
    warnings.extend(skel.notes)
    if cfg.debug:
        landmarks.draw_debug(grid, skel, debug_dir / "10_skeleton.png")
    joints_path = landmarks.write_json(skel, mesh.verts, mesh_world,
                                       out_dir / f"{a.name}_joints.json")

    export_mod.write_summary(
        out_dir / f"{a.name}_summary.json", mesh, mesh_world,
        extra={
            "layers": {n: str(l.folder) for n, l in layers.items()},
            "layer_iou": {n: l.iou for n, l in layers.items()},
            "voxel_grid": list(grid.shape),
            "final_glb": str(final_glb),
            "blend": str(out_dir / f"{a.name}.blend"),
            "warnings": warnings,
            "joints": str(joints_path),
        })

    # --- 8. rig --------------------------------------------------------------
    if cfg.rig == "blender":
        banner(8, total, "สร้างโครงกระดูก อนิเมชัน และ export GLB")
        anim_path = None
        if cfg.animate:
            n_tail = sum(1 for n in skel.joints
                         if n.startswith("Tail") and n != "TailEnd")
            clips = animation.default_clips(n_tail)
            anim_path = out_dir / f"{a.name}_clips.json"
            anim_path.write_text(
                json.dumps(animation.to_payload(clips, cfg.fps), indent=2,
                           ensure_ascii=False), encoding="utf-8")
            log.info("เขียนนิยามอนิเมชัน %d clip -> %s", len(clips), anim_path)

        from .rigging import rig_with_blender
        rigged = rig_with_blender(final_glb, joints_path,
                                  out_dir / f"{a.name}_rigged.glb", anim_path,
                                  draco=cfg.draco, blender_exe=a.blender)
        if cfg.texture_quality > 0:
            export_mod.recompress_textures(rigged, cfg.texture_quality,
                                           cfg.texture_glb_size)
        log.info("ได้ไฟล์ที่ rig แล้ว: %s (%.2f MB)", rigged,
                 rigged.stat().st_size / 1e6)

    if not a.keep_steps:
        for p in (hull_glb, cage_glb, cage_uv):
            p.unlink(missing_ok=True)

    log.info("\n\033[1mเสร็จใน %.1f วินาที\033[0m",
             time.perf_counter() - t_start)
    log.info("ผลลัพธ์อยู่ที่ %s", out_dir)
    if warnings:
        log.warning("\n\033[1mมีข้อควรตรวจ %d ข้อ:\033[0m", len(warnings))
        for w in warnings:
            log.warning("  - %s", w)
    return 0


def main(argv: list[str] | None = None) -> int:
    from .cli import setup_logging

    a = build_parser().parse_args(argv)
    setup_logging(a.verbose)
    try:
        return run(a)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("\n\033[31m%s\033[0m", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
