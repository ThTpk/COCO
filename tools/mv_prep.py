"""เตรียมภาพ photometry พื้นขาวให้พร้อมเข้า Hunyuan3D multiview

Hunyuan3D-2mv รับภาพ RGBA ที่ลบพื้นหลังแล้ว ภาพชุด input/photometry-whitebg/
เป็น JPEG/WebP พื้นขาวมีเงาตกกระทบ จึงต้อง

1. ลบพื้นหลัง — เลือกพิกเซลที่ "ขาว/เทาซีด" และต่อถึงขอบภาพ เงาบนพื้นจึงหลุดไป
   ด้วยเพราะมันต่อกับขอบ ส่วนฟันขาวกลางปากไม่ต่อกับขอบ จึงรอด
2. ครอปตามตัว แล้วแปะกลางผืนจัตุรัส เว้นขอบเท่ากันทุกมุม — ตัวแบบจึงอยู่
   ตำแหน่งเดียวกันทุกมุมมอง ซึ่งเป็นสิ่งที่ multiview model คาดหวัง
3. ย่อเป็น 518x518 ตามขนาดที่ image encoder ใช้

เรียก: python tools/mv_prep.py [--size 518] [--out <dir>]
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "input" / "photometry-whitebg"

# ชุด "mv" — สี่มุมฉากที่ Hunyuan3D-2mv รับเป็นเงื่อนไข ชื่อคีย์คือชื่อ input ของโหนด
MV_VIEWS = ("front", "left", "right", "back")

# ชุด "texture" — แปดมุมรอบตัวสำหรับฉายสีกลับลงบน mesh ยิ่งมากยิ่งมีมุมให้เลือก
# ไม่เอา up/down เพราะสองใบนั้นเป็นภาพ perspective แรง ฉายแบบ orthographic ไม่ลง
TEX_VIEWS = ("front", "front-right", "right", "back-right",
             "back", "back-left", "left", "front-left")


def background_mask(rgb: np.ndarray) -> np.ndarray:
    """คืน mask ของพื้นหลัง — พิกเซลซีดที่เดินจากขอบภาพมาถึงได้

    เกณฑ์ "ซีด" คือสว่างพอและสีจางพอ (ช่วง max-min ของ R,G,B แคบ) ซึ่งครอบคลุม
    ทั้งพื้นขาวและเงาสีเทา แต่ไม่โดนเสื้อน้ำเงินหรือหนังสีส้ม
    """
    v = rgb.max(axis=2).astype(np.int16)
    chroma = v - rgb.min(axis=2).astype(np.int16)
    pale = (v > 150) & (chroma < 32)

    h, w = pale.shape
    seen = np.zeros_like(pale)
    q: deque[tuple[int, int]] = deque()

    for x in range(w):
        for y in (0, h - 1):
            if pale[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if pale[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))

    # flood fill สี่ทิศ เขียนเป็น loop ตรง ๆ เพราะทำครั้งเดียวกับภาพ 10 ใบ
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and pale[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def cutout(path: Path, size: int, margin: float = 0.06) -> Image.Image:
    im = Image.open(path).convert("RGB")
    rgb = np.asarray(im)
    alpha = (~background_mask(rgb)).astype(np.uint8) * 255

    ys, xs = np.nonzero(alpha)
    if len(ys) == 0:
        raise SystemExit(f"{path.name}: ลบพื้นหลังแล้วไม่เหลืออะไรเลย")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    rgba = np.dstack([rgb, alpha])[y0:y1, x0:x1]
    # ทำให้พิกเซลโปร่งใสเป็นขาวสนิท กันขอบดำตอนย่อภาพ
    cut = Image.fromarray(rgba, "RGBA")

    bw, bh = cut.size
    span = max(bw, bh)
    canvas = int(round(span * (1 + 2 * margin)))
    sheet = Image.new("RGBA", (canvas, canvas), (255, 255, 255, 0))
    sheet.paste(cut, ((canvas - bw) // 2, (canvas - bh) // 2))
    return sheet.resize((size, size), Image.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=("mv", "texture"), default="mv")
    ap.add_argument("--size", type=int, default=0, help="0 = ใช้ค่าปริยายของชุดนั้น")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    views = MV_VIEWS if args.set == "mv" else TEX_VIEWS
    size = args.size or (518 if args.set == "mv" else 1024)

    args.out.mkdir(parents=True, exist_ok=True)
    for view in views:
        src = SRC / f"{view}.webp"
        if not src.exists():
            raise SystemExit(f"ไม่พบ {src}")
        out = args.out / f"croc_mv_{view}.png"
        cutout(src, size).save(out)
        print(f"{src.name} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
