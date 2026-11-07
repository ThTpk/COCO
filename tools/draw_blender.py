r"""เอาโมเดลที่แกะจากคลิปไปจัดฉากและเรนเดอร์ใน Blender

    .venv\Scripts\python.exe tools\draw_blender.py --name croc
    .venv\Scripts\python.exe tools\draw_blender.py --name croc --turns 24 --clay
    .venv\Scripts\python.exe tools\draw_blender.py --name croc --device cpu

ไฟล์นี้อยู่ฝั่ง venv ปกติ ทำหน้าที่หาไฟล์ให้ครบ เรียก Blender แบบ headless
แล้วประกอบภาพที่ได้เป็น contact sheet ใบเดียว ส่วนโค้ดที่ใช้ ``bpy`` จริง
อยู่ที่ ``src/blender_draw.py`` เพราะรันได้แต่ใน interpreter ของ Blender

ค่าเริ่มต้นเรนเดอร์ด้วย Cycles บนการ์ดจอ (OptiX ถ้าเป็น NVIDIA) — ใส่
``--device cpu`` ถ้าอยากให้การ์ดว่างไว้ทำอย่างอื่น
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BLENDER_EXE, OUTPUT_DIR  # noqa: E402

log = logging.getLogger("2Dto3D.draw")


def find_inputs(out_dir: Path, name: str, rigged: bool) -> dict[str, Path | None]:
    """หา mesh กับแผนที่ผิวของโมเดลชื่อนี้"""
    mesh = out_dir / (f"{name}_rigged.glb" if rigged else f"{name}.glb")
    if not mesh.exists():
        other = out_dir / (f"{name}.glb" if rigged else f"{name}_rigged.glb")
        if other.exists():
            log.warning("ไม่พบ %s — ใช้ %s แทน", mesh.name, other.name)
            mesh = other
        else:
            raise FileNotFoundError(
                f"ไม่พบโมเดลชื่อ {name} ใน {out_dir}\n"
                f"รัน `python -m src.clipcli --name {name}` ก่อน")
    maps = {}
    for key, suffix in (("texture", "texture"), ("normal", "normal"),
                        ("rough", "rough")):
        p = out_dir / f"{name}_{suffix}.png"
        maps[key] = p if p.exists() else None
        if maps[key] is None:
            log.warning("ไม่พบแผนที่ %s ของ %s — Blender จะข้ามช่องนั้น",
                        suffix, name)
    return {"mesh": mesh, **maps}


def run_blender(args: argparse.Namespace, found: dict[str, Path | None],
                script: Path, out_dir: Path) -> None:
    exe = Path(args.blender)
    if not exe.exists():
        raise FileNotFoundError(
            f"ไม่พบ Blender ที่ {exe}\n"
            "ตั้ง environment variable BLENDER_EXE ให้ชี้ไปที่ blender.exe")

    cmd = [str(exe), "--background", "--factory-startup",
           "--python", str(script), "--",
           "--mesh", str(found["mesh"]),
           "--out", str(out_dir),
           "--turns", str(args.turns),
           "--width", str(args.width),
           "--height", str(args.height),
           "--samples", str(args.samples),
           "--engine", args.engine,
           "--device", args.device,
           "--lens", str(args.lens),
           "--margin", str(args.margin),
           "--normal-strength", str(args.normal_strength)]
    for key in ("texture", "normal", "rough"):
        if found[key] is not None:
            cmd += [f"--{key}", str(found[key])]
    if args.transparent:
        cmd.append("--transparent")
    if args.clay:
        cmd.append("--clay")
    if args.keep_material:
        cmd.append("--keep-material")
    if args.no_flip_v:
        cmd.append("--no-flip-v")

    log.info("เรียก Blender: %s", exe.name)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    done = False
    tail: list[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        del tail[:-30]
        if line.startswith("[draw]"):
            log.info("  %s", line[6:].strip())
            if "เสร็จสมบูรณ์" in line:
                done = True
    proc.wait()

    # Blender คืน exit code 0 ได้แม้สคริปต์จะ traceback กลางคัน จึงต้องให้
    # สคริปต์ประกาศความสำเร็จของตัวเอง ไม่งั้นจะเจอภาพเก่าค้างแล้วนึกว่าสำเร็จ
    if proc.returncode != 0 or not done:
        why = ("สคริปต์หยุดกลางคัน (ไม่มีสัญญาณจบงาน)" if proc.returncode == 0
               else f"exit {proc.returncode}")
        raise RuntimeError("Blender ทำงานไม่สำเร็จ: " + why + "\n--- ท้าย log ---\n"
                           + "\n".join(tail))
    log.info("Blender ใช้เวลา %.1f วินาที", time.time() - t0)


def contact_sheet(shots: list[Path], path: Path, cols: int,
                  cell: int = 300) -> Path:
    """เรียงภาพที่เรนเดอร์ได้ลงแผ่นเดียว ไว้ตรวจทุกมุมพร้อมกัน"""
    rows = (len(shots) + cols - 1) // cols
    lab = 20
    sheet = np.full((rows * (cell + lab), cols * cell, 3), 24, np.uint8)
    for k, shot in enumerate(shots):
        img = cv2.imdecode(np.fromfile(str(shot), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("อ่านภาพไม่ได้: %s", shot)
            continue
        h, w = img.shape[:2]
        s = min(cell / w, cell / h)
        small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        r, c = divmod(k, cols)
        oy = r * (cell + lab) + lab + (cell - sh) // 2
        ox = c * cell + (cell - sw) // 2
        sheet[oy:oy + sh, ox:ox + sw] = small
        cv2.putText(sheet, shot.stem.split("_", 1)[1],
                    (c * cell + 8, r * (cell + lab) + lab - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
                    cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", sheet)
    if not ok:
        raise RuntimeError("เข้ารหัส contact sheet ไม่สำเร็จ")
    buf.tofile(str(path))
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="draw_blender",
        description="จัดฉากและเรนเดอร์โมเดลจากคลิปด้วย Blender",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--name", default="croc", help="ชื่อโมเดลใน output/")
    p.add_argument("--models", type=Path, default=OUTPUT_DIR,
                   help="โฟลเดอร์ที่เก็บ .glb กับแผนที่ผิว")
    p.add_argument("--out", type=Path, default=None,
                   help="โฟลเดอร์ผลลัพธ์ (ค่าเริ่มต้น output/<name>_blender)")
    p.add_argument("--rigged", action="store_true",
                   help="ใช้ไฟล์ที่ rig แล้วแทน mesh เปล่า")
    p.add_argument("--turns", type=int, default=16, help="จำนวนมุมที่เรนเดอร์")
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=1400)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--engine", choices=["cycles", "eevee"], default="cycles")
    p.add_argument("--device", choices=["auto", "gpu", "cpu", "optix", "cuda"],
                   default="auto")
    p.add_argument("--lens", type=float, default=85.0)
    p.add_argument("--margin", type=float, default=1.18)
    p.add_argument("--normal-strength", type=float, default=1.0)
    p.add_argument("--transparent", action="store_true",
                   help="พื้นหลังโปร่งใส (เห็นแต่เงาบนพื้น)")
    p.add_argument("--clay", action="store_true",
                   help="เรนเดอร์ชุดเทาล้วนเพิ่ม ไว้ดูรูปทรงล้วน ๆ")
    p.add_argument("--keep-material", action="store_true",
                   help="ใช้วัสดุที่ติดมาใน .glb ไม่ต้องต่อใหม่จากแผนที่")
    p.add_argument("--no-flip-v", action="store_true",
                   help="ไม่ต้องพลิกแกน V ของแผนที่ (ปกติต้องพลิก)")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--blender", default=BLENDER_EXE)
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = a.out or (OUTPUT_DIR / f"{a.name}_blender")
    found = find_inputs(a.models, a.name, a.rigged)
    log.info("โมเดล: %s (%.2f MB)", found["mesh"],
             found["mesh"].stat().st_size / 1e6)

    script = Path(__file__).resolve().parent.parent / "src" / "blender_draw.py"
    run_blender(a, found, script, out_dir)

    for tag, cols in (("turn", a.cols), ("clay", a.cols)):
        # กัน contact sheet ของรอบก่อนไม่ให้ถูกนับเป็นภาพเรนเดอร์ของรอบนี้
        # (ชื่อ clay_sheet.png เข้าแพตเทิร์น clay_*.png ของตัวเองพอดี)
        shots = [p for p in sorted(out_dir.glob(f"{tag}_*.png"))
                 if not p.stem.endswith("_sheet")]
        if not shots:
            continue
        sheet = contact_sheet(shots, out_dir / f"{tag}_sheet.png", cols)
        log.info("contact sheet (%s) -> %s (%.2f MB)", tag, sheet,
                 sheet.stat().st_size / 1e6)

    log.info("ไฟล์ฉาก Blender: %s", out_dir / f"{found['mesh'].stem}.blend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
