r"""ตรวจว่าเครื่องพร้อมรัน pipeline แล้วหรือยัง

    .venv\Scripts\python.exe tools\check_env.py
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))


def check_python() -> None:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}  ({sys.executable})"
    record(OK if (v.major, v.minor) >= (3, 10) else BAD, "Python", detail)
    in_venv = sys.prefix != sys.base_prefix
    record(OK if in_venv else WARN, "venv",
           "กำลังรันใน venv" if in_venv else "รันบน Python global — ควรใช้ .venv ของโปรเจกต์")


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        record(BAD, "torch", "ยังไม่ได้ติดตั้ง — ดู SETUP.md")
        return

    build = torch.__version__
    record(OK, "torch", build)

    if "+cpu" in build:
        record(BAD, "torch CUDA build",
               "เป็น wheel สาย CPU — ต้องลงใหม่จาก index cu128 (ดู SETUP.md)")
        return

    if not torch.cuda.is_available():
        record(BAD, "torch.cuda.is_available()",
               "False — driver หรือ wheel ไม่ตรงกัน")
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    record(OK, "GPU", f"{name}  sm_{cap[0]}{cap[1]}  {total:.1f} GB")
    record(OK, "torch CUDA runtime", str(torch.version.cuda))

    arches = torch.cuda.get_arch_list()
    want = f"sm_{cap[0]}{cap[1]}"
    record(OK if want in arches else WARN, f"wheel รองรับ {want}",
           ", ".join(arches) or "ไม่มีข้อมูล")

    # ทดสอบคำนวณจริงบน GPU — is_available() ผ่านแต่คำนวณพังก็มี
    try:
        a = torch.randn(512, 512, device="cuda")
        err = float((a @ a - (a @ a)).abs().max())
        torch.cuda.synchronize()
        record(OK if err == 0.0 else WARN, "matmul บน GPU", "คำนวณได้จริง")
    except Exception as exc:  # noqa: BLE001
        record(BAD, "matmul บน GPU", f"{type(exc).__name__}: {exc}")


def check_nvcc() -> None:
    exe = shutil.which("nvcc")
    if not exe:
        record(WARN, "nvcc",
               "ไม่พบใน PATH — คอมไพล์ CUDA extension ไม่ได้ (ดู SETUP.md ขั้นที่ 1)")
        return
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             timeout=30).stdout
        ver = next((ln.strip() for ln in out.splitlines() if "release" in ln), out.strip())
        is_12 = ", 12." in ver or "release 12" in ver
        record(OK if is_12 else WARN, "nvcc", ver +
               ("" if is_12 else "  <- ไม่ใช่สาย 12.x, third-party repo อาจคอมไพล์ไม่ผ่าน"))
    except Exception as exc:  # noqa: BLE001
        record(WARN, "nvcc", f"เรียกไม่สำเร็จ: {exc}")


def check_msvc() -> None:
    if shutil.which("cl"):
        record(OK, "cl.exe (MSVC)", "พบใน PATH")
        return
    roots = [
        Path(r"C:\Program Files\Microsoft Visual Studio"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
    ]
    found = [p for r in roots if r.exists() for p in r.glob("*/*/VC/Tools/MSVC/*")]
    if found:
        record(WARN, "cl.exe (MSVC)",
               f"ติดตั้งแล้วที่ {found[0].parent.parent.parent.parent} "
               "แต่ยังไม่อยู่ใน PATH — ใช้ 'x64 Native Tools Command Prompt' เวลาคอมไพล์")
    else:
        record(WARN, "cl.exe (MSVC)",
               "ไม่พบ VS Build Tools — คอมไพล์ CUDA extension ไม่ได้ (ดู SETUP.md ขั้นที่ 1)")


def check_packages() -> None:
    for mod, label in [
        ("numpy", "numpy"), ("scipy", "scipy"), ("cv2", "opencv-python"),
        ("skimage", "scikit-image"), ("open3d", "open3d"),
        ("trimesh", "trimesh"), ("pygltflib", "pygltflib"), ("PIL", "pillow"),
    ]:
        try:
            m = importlib.import_module(mod)
            record(OK, label, getattr(m, "__version__", "?"))
        except ImportError:
            record(BAD, label, "ขาด — pip install -r requirements.txt")
    try:
        importlib.import_module("rembg")
        record(OK, "rembg (optional)", "พร้อมใช้")
    except ImportError:
        record(WARN, "rembg (optional)",
               "ไม่มี — ใช้ได้เฉพาะภาพที่มี alpha หรือพื้นหลังสีพื้น")


def check_blender() -> None:
    from src.config import BLENDER_EXE
    p = Path(BLENDER_EXE)
    if not p.exists():
        record(BAD, "Blender", f"ไม่พบที่ {p} — ตั้ง env var BLENDER_EXE ชี้ไปที่ blender.exe")
        return
    try:
        out = subprocess.run([str(p), "--version"], capture_output=True, text=True,
                             timeout=90).stdout.strip().splitlines()
        record(OK, "Blender", out[0] if out else str(p))
    except Exception as exc:  # noqa: BLE001
        record(WARN, "Blender", f"เรียกไม่สำเร็จ: {exc}")


def check_inputs() -> None:
    from src.config import IMAGE_EXTS, INPUT_DIR
    if not INPUT_DIR.exists():
        record(WARN, "input/", "ยังไม่มีโฟลเดอร์")
        return
    imgs = [p for p in INPUT_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if imgs:
        record(OK, "input/", f"{len(imgs)} ภาพ: " + ", ".join(p.name for p in imgs[:6]))
    else:
        record(WARN, "input/", "ว่าง — วางภาพ front/back/side ลงในโฟลเดอร์นี้")


def main() -> int:
    for fn in (check_python, check_torch, check_nvcc, check_msvc,
               check_packages, check_blender, check_inputs):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            record(BAD, fn.__name__, f"{type(exc).__name__}: {exc}")

    width = max(len(n) for _, n, _ in results) + 2
    print("\n" + "=" * 78)
    print("  ตรวจสภาพแวดล้อม 2Dto3D")
    print("=" * 78)
    for status, name, detail in results:
        print(f" {status} {name:<{width}} {detail}")
    print("=" * 78)

    bad = sum(1 for s, _, _ in results if s == BAD)
    warn = sum(1 for s, _, _ in results if s == WARN)
    print(f"  ผ่าน {len(results) - bad - warn}   เตือน {warn}   ไม่ผ่าน {bad}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
