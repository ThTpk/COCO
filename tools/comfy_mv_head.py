"""สั่ง ComfyUI ปั้น mesh จากภาพสี่มุม ด้วย Hunyuan3D-2mv

ประกอบ workflow แบบ API ขึ้นเองแทนที่จะโหลด .json จาก UI เพราะ workflow ตัวอย่าง
ของ wrapper พ่วง custom node ชุด comfyui-essentials กับโมเดล delight/paint ที่เครื่อง
นี้ไม่มี เส้นทางนี้ใช้เฉพาะโหนดของ wrapper กับโหนดแกนของ ComfyUI จึงรันได้เลย

สีไม่ได้มาจาก ComfyUI — ขั้นนี้เอาแต่รูปทรง แล้วไปอบสีใน Blender ด้วยภาพต้นฉบับ
(tools/mv_texture.py) ซึ่งได้ผิวตรงกับภาพจริงมากกว่าให้ paint model วาดใหม่

เรียก: python tools/comfy_mv.py --out output/croc_mv_shape.glb
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = "http://127.0.0.1:8188"
# Hy3DExportMesh ไม่ได้ลง history ว่าเขียนไฟล์ไหน และ output ไปอยู่คนละที่ขึ้นกับว่า
# เซิร์ฟเวอร์ถูกสั่งเปิดยังไง (Comfy Desktop ชี้ไป ComfyUI-Shared ส่วนการเปิดเอง
# แบบ headless ใช้ของในโฟลเดอร์ติดตั้ง) จึงกวาดดูทุกที่แล้วเอาไฟล์ใหม่สุด
COMFY_ROOT = Path.home() / "AppData/Local/Comfy-Desktop"
COMFY_OUTS = [COMFY_ROOT / "ComfyUI-Shared/output",
              *(COMFY_ROOT / "ComfyUI-Installs").glob("*/ComfyUI/output")]

MODEL = "hunyuan3d-dit-v2-mv_fp16_tf5.safetensors"
VIEWS = ("front", "left", "right")


def build(seed: int, steps: int, cfg: float, octree: int, faces: int, prefix: str) -> dict:
    """คืน workflow รูปแบบ API — dict ของ node_id -> {class_type, inputs}"""
    g: dict[str, dict] = {
        "loader": {
            "class_type": "Hy3DModelLoader",
            "inputs": {"model": MODEL, "attention_mode": "sdpa", "cublas_ops": False},
        },
    }

    for v in VIEWS:
        # LoadImage คืน MASK ที่เป็น "ส่วนโปร่งใส" ต้องกลับด้านก่อนจะเอาไปเป็น alpha
        g[f"load_{v}"] = {
            "class_type": "LoadImage",
            "inputs": {"image": f"croc_head_{v}.png"},
        }
        g[f"inv_{v}"] = {
            "class_type": "InvertMask",
            "inputs": {"mask": [f"load_{v}", 1]},
        }
        g[f"rgba_{v}"] = {
            "class_type": "JoinImageWithAlpha",
            "inputs": {"image": [f"load_{v}", 0], "alpha": [f"inv_{v}", 0]},
        }

    g["gen"] = {
        "class_type": "Hy3DGenerateMeshMultiView",
        "inputs": {
            "pipeline": ["loader", 0],
            "guidance_scale": cfg,
            "steps": steps,
            "seed": seed,
            "scheduler": "FlowMatchEulerDiscreteScheduler",
            **{v: [f"rgba_{v}", 0] for v in VIEWS},
        },
    }
    g["decode"] = {
        "class_type": "Hy3DVAEDecode",
        "inputs": {
            "vae": ["loader", 1],
            "latents": ["gen", 0],
            "box_v": 1.01,
            "octree_resolution": octree,
            "num_chunks": 8000,
            "mc_level": 0,
            "mc_algo": "mc",
            "enable_flash_vdm": True,
            "force_offload": True,
        },
    }
    g["post"] = {
        "class_type": "Hy3DPostprocessMesh",
        "inputs": {
            "trimesh": ["decode", 0],
            "remove_floaters": True,
            "remove_degenerate_faces": True,
            "reduce_faces": True,
            "max_facenum": faces,
            "smooth_normals": False,
        },
    }
    g["save"] = {
        "class_type": "Hy3DExportMesh",
        "inputs": {
            "trimesh": ["post", 0],
            "filename_prefix": prefix,
            "file_format": "glb",
            "save_file": True,
        },
    }
    return g


def upload(img: Path) -> None:
    """ส่งภาพเข้าโฟลเดอร์ input ของ ComfyUI ผ่าน API

    ก๊อปไฟล์ไปวางเองไม่พอ — ComfyUI รุ่นนี้ตรวจชื่อภาพกับทะเบียน asset ที่สแกนไว้
    ตอนบูต ไฟล์ที่โผล่มาทีหลังจะถูกปฏิเสธว่า "Invalid image file"
    """
    crlf = "\r\n"
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(img.name)[0] or "application/octet-stream"
    head = (
        f"--{boundary}{crlf}"
        f'Content-Disposition: form-data; name="image"; filename="{img.name}"{crlf}'
        f"Content-Type: {ctype}{crlf}{crlf}"
    )
    tail = (
        f"{crlf}--{boundary}{crlf}"
        f'Content-Disposition: form-data; name="overwrite"{crlf}{crlf}true{crlf}'
        f"--{boundary}--{crlf}"
    )
    body = head.encode() + img.read_bytes() + tail.encode()
    req = urllib.request.Request(
        SERVER + "/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        SERVER + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(SERVER + path, timeout=60) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "croc_head_shape.glb")
    ap.add_argument("--images", type=Path, default=ROOT / "output" / "head_in",
                    help="โฟลเดอร์ภาพ cutout จาก tools/mv_prep.py --set mv")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=5.5)
    ap.add_argument("--octree", type=int, default=320, help="ยิ่งสูงยิ่งละเอียด แต่กิน VRAM")
    ap.add_argument("--faces", type=int, default=120000)
    args = ap.parse_args()

    try:
        get("/system_stats")
    except (urllib.error.URLError, TimeoutError) as e:
        raise SystemExit(f"ต่อ ComfyUI ที่ {SERVER} ไม่ได้ ({e}) — เปิด Comfy Desktop ก่อน")

    for v in VIEWS:
        src = args.images / f"croc_head_{v}.png"
        if not src.exists():
            raise SystemExit(f"ไม่พบ {src} — รัน tools/mv_prep.py --set mv ก่อน")
        upload(src)
    print(f"อัปโหลดภาพ {len(VIEWS)} ใบแล้ว")

    prefix = "3D/croc_head"
    wf = build(args.seed, args.steps, args.cfg, args.octree, args.faces, prefix)
    client = str(uuid.uuid4())
    pid = post("/prompt", {"prompt": wf, "client_id": client})["prompt_id"]
    print(f"คิวแล้ว prompt_id={pid}")

    t0 = time.time()
    while True:
        time.sleep(5)
        hist = get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                for kind, data in status.get("messages", []):
                    if kind == "execution_error":
                        print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])
                raise SystemExit("workflow พัง — ดูรายละเอียดข้างบน")
            break
        q = get("/queue")
        running = len(q.get("queue_running", []))
        print(f"  ... {time.time() - t0:.0f}s (กำลังรัน {running})", flush=True)

    cands = [p for root in COMFY_OUTS
             for p in (root / Path(prefix).parent).glob(Path(prefix).name + "*.glb")]
    if not cands:
        raise SystemExit("ไม่เจอไฟล์ .glb ใน " + ", ".join(str(r) for r in COMFY_OUTS))
    src = max(cands, key=lambda p: p.stat().st_mtime)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, args.out)
    print(f"เสร็จใน {time.time() - t0:.0f}s -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
