"""แปลงชื่อ tensor ใน checkpoint ของ Hunyuan3D ให้ตรงกับ transformers รุ่นใหม่

Hunyuan3D-2mv ใช้ DINOv2-giant เป็น image encoder ส่วน transformers 5.x เปลี่ยน
ชื่อโมดูลข้างในไปแล้ว ตัว wrapper ประกอบโมเดลจาก transformers ที่ติดตั้งอยู่
แล้วโหลด state dict แบบ strict จึงพังทันทีที่ชื่อไม่ตรง:

    KeyError: 'main_image_encoder.model.encoder.layer.0.attention.q_proj.weight'

ที่เปลี่ยนมีสองกลุ่ม

  attention.attention.{query,key,value}  ->  attention.{q,k,v}_proj
  attention.output.dense                 ->  attention.o_proj
  mlp.weights_in                         ->  mlp.gate_proj + mlp.up_proj
  mlp.weights_out                        ->  mlp.down_proj

กลุ่มหลังไม่ใช่แค่เปลี่ยนชื่อ — SwiGLU เดิมเก็บสองเมทริกซ์ต่อกันเป็นก้อนเดียว
แล้วค่อย chunk ตอนรัน ของใหม่แยกเป็นสอง Linear ตั้งแต่แรก จึงต้องผ่าครึ่ง
(ครึ่งแรกคือ gate เพราะโค้ดเดิมเอา chunk แรกเข้า silu)

เขียนไฟล์ใหม่แบบไหลทีละ tensor ไม่ได้อ่านทั้ง 4.9 GB ขึ้น RAM

    python tools/hy3d_convert_ckpt.py <src.safetensors> <dst.safetensors>
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# ขนาดไบต์ต่อค่า ของ dtype ที่โผล่ใน checkpoint พวกนี้
ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1}

PREFIX = "conditioner.main_image_encoder.model."
RENAME = {
    "attention.attention.query": "attention.q_proj",
    "attention.attention.key": "attention.k_proj",
    "attention.attention.value": "attention.v_proj",
    "attention.output.dense": "attention.o_proj",
    "mlp.weights_out": "mlp.down_proj",
}


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    return head, 8 + n


def plan(head: dict) -> list[tuple[str, list[str], list[int], str, tuple[int, int]]]:
    """คืนรายการงาน (ชื่อเดิม, ชื่อใหม่ที่จะออก, shape, dtype, ช่วงไบต์ต้นทาง)

    ชื่อใหม่เป็น list เพราะ weights_in ตัวเดียวแตกออกเป็นสอง tensor
    """
    jobs = []
    for key, info in head.items():
        if key == "__metadata__":
            continue
        new = [key]
        if key.startswith(PREFIX):
            tail = key[len(PREFIX):]
            for old, repl in RENAME.items():
                if old in tail:
                    new = [PREFIX + tail.replace(old, repl)]
                    break
            else:
                if "mlp.weights_in" in tail:
                    new = [PREFIX + tail.replace("mlp.weights_in", "mlp.gate_proj"),
                           PREFIX + tail.replace("mlp.weights_in", "mlp.up_proj")]
        jobs.append((key, new, info["shape"], info["dtype"], tuple(info["data_offsets"])))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()

    head, data_start = read_header(args.src)
    jobs = plan(head)
    split = sum(1 for _, new, *_ in jobs if len(new) > 1)
    renamed = sum(1 for old, new, *_ in jobs if len(new) == 1 and new[0] != old)
    print(f"{len(jobs)} tensor — เปลี่ยนชื่อ {renamed}, ผ่าครึ่ง {split}")

    # รอบแรก: วางผังว่า tensor ไหนอยู่ช่วงไบต์ไหนของไฟล์ใหม่
    out_head: dict[str, dict] = {}
    cursor = 0
    layout: list[tuple[str, tuple[int, int], int, int]] = []  # ชื่อเดิม, ช่วงต้นทาง, ครึ่งที่, จำนวนครึ่ง
    for key, new, shape, dtype, span in jobs:
        parts = len(new)
        for i, name in enumerate(new):
            out_shape = list(shape)
            if parts > 1:
                out_shape[0] //= parts
            size = int(np.prod(out_shape)) * ITEMSIZE[dtype]
            out_head[name] = {"dtype": dtype, "shape": out_shape,
                              "data_offsets": [cursor, cursor + size]}
            layout.append((key, span, i, parts))
            cursor += size

    blob = json.dumps(out_head, separators=(",", ":")).encode()
    pad = (-(len(blob)) % 8)
    blob += b" " * pad

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.src.open("rb") as fin, args.dst.open("wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for key, span, part, parts in layout:
            lo, hi = span
            length = hi - lo
            chunk = length // parts
            fin.seek(data_start + lo + part * chunk)
            # ผ่าตามแกนแรก ซึ่งเก็บต่อกันอยู่แล้ว จึงตัดเป็นช่วงไบต์ได้ตรง ๆ
            remaining = chunk
            while remaining:
                buf = fin.read(min(remaining, 1 << 22))
                if not buf:
                    raise SystemExit(f"อ่าน {key} ไม่ครบ")
                fout.write(buf)
                remaining -= len(buf)

    print(f"เขียน {args.dst} ({args.dst.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
