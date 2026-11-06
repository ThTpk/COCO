"""ขั้นที่ 7 (ทางเลือก): เรียก Blender แบบ headless มาสร้างโครงกระดูกและ skinning

แยกเป็นสองไฟล์เพราะโค้ดที่ใช้ `bpy` รันได้เฉพาะใน interpreter ของ Blender เท่านั้น
ไฟล์นี้อยู่ฝั่ง venv ปกติ ทำหน้าที่เรียก process และแปลผลลัพธ์
ส่วน `blender_rig.py` คือสคริปต์ที่ Blender รันจริง
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .config import BLENDER_EXE

log = logging.getLogger(__name__)


def rig_with_blender(
    mesh_glb: Path,
    joints_json: Path,
    out_glb: Path,
    anim_json: Path | None = None,
    draco: bool = False,
    blender_exe: str = BLENDER_EXE,
    timeout: int = 600,
) -> Path:
    """สร้าง armature จากข้อต่อที่ประเมินไว้ แล้ว bind เข้ากับ mesh"""
    exe = Path(blender_exe)
    if not exe.exists():
        raise FileNotFoundError(
            f"ไม่พบ Blender ที่ {exe}\n"
            "ตั้ง environment variable BLENDER_EXE ให้ชี้ไปที่ blender.exe "
            "หรือใช้ --rig none เพื่อข้ามขั้นนี้"
        )

    script = Path(__file__).resolve().parent / "blender_rig.py"
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(exe), "--background", "--factory-startup",
        "--python", str(script), "--",
        "--mesh", str(mesh_glb),
        "--joints", str(joints_json),
        "--out", str(out_glb),
    ]
    if anim_json is not None:
        cmd += ["--anim", str(anim_json)]
    cmd += ["--draco", "1" if draco else "0"]
    log.info("เรียก Blender: %s", exe.name)

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace")

    for line in (proc.stdout or "").splitlines():
        if line.startswith("[rig]"):
            log.info("  %s", line[5:].strip())

    # Blender คืน exit code 0 ได้แม้สคริปต์จะ traceback กลางคัน จึงเชื่อ
    # returncode อย่างเดียวไม่ได้ ต้องให้สคริปต์ประกาศความสำเร็จของตัวเองด้วย
    # ไม่งั้นจะได้ไฟล์เก่าค้างจากการรันครั้งก่อนมาแทน แล้วนึกว่าสำเร็จ
    done = "[rig] เสร็จสมบูรณ์" in (proc.stdout or "")

    if proc.returncode != 0 or not out_glb.exists() or not done:
        tail = "\n".join((proc.stdout or "").splitlines()[-25:])
        err = "\n".join((proc.stderr or "").splitlines()[-15:])
        why = ("สคริปต์หยุดกลางคัน (ไม่มีสัญญาณจบงาน)" if proc.returncode == 0
               else f"exit {proc.returncode}")
        raise RuntimeError(
            f"Blender ทำงานไม่สำเร็จ: {why}\n"
            f"--- stdout ---\n{tail}\n--- stderr ---\n{err}"
        )

    log.info("rig เสร็จ -> %s (%.2f MB)", out_glb, out_glb.stat().st_size / 1e6)
    return out_glb
