r"""เรนเดอร์ mesh เป็นภาพ PNG หลายมุม เพื่อตรวจรูปทรงด้วยตาโดยไม่ต้องเปิดโปรแกรม 3D

    .venv\Scripts\python.exe tools\render_preview.py output\character.glb -o preview.png
    .venv\Scripts\python.exe tools\render_preview.py output\character.glb --zoom hands

เป็น rasterizer แบบ z-buffer เขียนด้วย numpy ล้วน ไม่ต้องมี GPU หรือ OpenGL
เพราะการตรวจงานขั้นนี้ต้องการแค่ "รูปทรงอ่านออกไหม" ไม่ต้องการความสวย
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    scene = trimesh.load(str(path), process=False)
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    else:
        mesh = scene
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32)


def rasterize(
    verts: np.ndarray,
    faces: np.ndarray,
    size: int,
    yaw: float,
    pitch: float = 0.0,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    return_depth: bool = False,
) -> np.ndarray:
    """ฉายภาพแบบ orthographic แล้วแรเงาด้วยมุมของ normal เทียบแสงจากหน้า-บน"""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    v = verts @ rot_y.T @ rot_x.T

    lo, hi = (v.min(0), v.max(0)) if bounds is None else bounds
    center = 0.5 * (lo + hi)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1])) * 1.06 or 1.0
    scale = size / span

    px = (v[:, 0] - center[0]) * scale + size / 2
    py = size / 2 - (v[:, 1] - center[1]) * scale
    # แกนลึกต้องสเกลเท่ากับแกนภาพ ไม่งั้น normal ที่คำนวณจาก cross product
    # จะโดนหน่วยพิกเซลกลบจนชี้เข้ากล้องหมด แล้วภาพออกมาขาวแบน
    pz = (v[:, 2] - center[2]) * scale

    tri = np.stack([px, py, pz], 1)[faces]  # (F, 3, 3)

    # normal ในพิกัดกล้อง ใช้แรเงาอย่างเดียว
    e1 = tri[:, 1, :3] - tri[:, 0, :3]
    e2 = tri[:, 2, :3] - tri[:, 0, :3]
    n = np.cross(e1, e2)
    n[:, 1] *= -1  # แกน y ของภาพชี้ลง
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-12)

    # หันด้าน normal เข้าหากล้องก่อนคำนวณแสง
    #
    # เคยใช้ abs(n . light) เพื่อกัน winding ที่ไม่สม่ำเสมอ ซึ่งสร้าง *วงเข้ม
    # ปลอม* รอบทุกส่วนโค้ง: ตรงที่ normal ตั้งฉากกับแสง ค่าจะเป็น 0 แล้วสว่าง
    # ขึ้นอีกฝั่ง เกิดเป็นเส้นวงรีสีเข้มคาดกลางหัวที่ไม่มีอยู่จริงในโมเดล
    # (ตรวจแล้วว่า voxel ตรงนั้นโตขึ้นเรียบทุกแถว) ซึ่งพาให้ไล่แก้ผิดจุด
    #
    # เราเห็นเฉพาะผิวที่หันเข้ากล้องอยู่แล้ว การบังคับ normal ให้ชี้เข้ากล้อง
    # จึงถูกต้องเสมอ และไม่ต้องเชื่อ winding ของไฟล์
    n *= np.sign(n[:, 2:3]) + (n[:, 2:3] == 0)
    light = np.array([0.35, 0.55, 0.76])
    light = light / np.linalg.norm(light)
    lam = np.clip(n @ light, 0.0, 1.0)
    shade = 0.16 + 0.84 * lam ** 0.8

    depth = np.full((size, size), -np.inf, np.float64)
    color = np.zeros((size, size), np.float64)

    # เรียงจากไกลไปใกล้แล้ววาดทับ (painter) พร้อมเช็ค z-buffer ต่อพิกเซล
    order = np.argsort(tri[:, :, 2].mean(1))
    for f in order:
        t = tri[f]
        if not np.isfinite(t).all():
            continue
        x0 = max(int(np.floor(t[:, 0].min())), 0)
        x1 = min(int(np.ceil(t[:, 0].max())) + 1, size)
        y0 = max(int(np.floor(t[:, 1].min())), 0)
        y1 = min(int(np.ceil(t[:, 1].max())) + 1, size)
        if x0 >= x1 or y0 >= y1:
            continue

        xs = np.arange(x0, x1) + 0.5
        ys = np.arange(y0, y1) + 0.5
        gx, gy = np.meshgrid(xs, ys)

        ax, ay = t[0, 0], t[0, 1]
        bx, by = t[1, 0], t[1, 1]
        cx, cy_ = t[2, 0], t[2, 1]
        den = (by - cy_) * (ax - cx) + (cx - bx) * (ay - cy_)
        if abs(den) < 1e-12:
            continue
        w0 = ((by - cy_) * (gx - cx) + (cx - bx) * (gy - cy_)) / den
        w1 = ((cy_ - ay) * (gx - cx) + (ax - cx) * (gy - cy_)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z = w0 * t[0, 2] + w1 * t[1, 2] + w2 * t[2, 2]
        sub = depth[y0:y1, x0:x1]
        hit = inside & (z > sub)
        if not hit.any():
            continue
        sub[hit] = z[hit]
        color[y0:y1, x0:x1][hit] = shade[f]

    if return_depth:
        # ไว้ตรวจว่ารอยที่เห็นเป็นรูปทรงจริงหรือเป็นแค่เงา: ความลึกไม่โกหก
        return depth

    img = np.zeros((size, size, 3), np.uint8)
    body = depth > -np.inf
    g = (np.clip(color, 0, 1) * 255).astype(np.uint8)
    img[body] = np.stack([g, g, g], -1)[body]
    img[~body] = (24, 26, 30)
    return img


ZOOMS = {
    "hands": (0.00, 0.62, 0.34, 0.52),   # (x0, x1, y0, y1) เป็นสัดส่วนของกรอบ
    "feet":  (0.10, 0.90, 0.00, 0.22),
    "head":  (0.20, 0.80, 0.80, 1.00),
}


def crop_bounds(verts: np.ndarray, zoom: str) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = verts.min(0), verts.max(0)
    fx0, fx1, fy0, fy1 = ZOOMS[zoom]
    span = hi - lo
    new_lo = lo.copy()
    new_hi = hi.copy()
    new_lo[0], new_hi[0] = lo[0] + fx0 * span[0], lo[0] + fx1 * span[0]
    new_lo[1], new_hi[1] = lo[1] + fy0 * span[1], lo[1] + fy1 * span[1]
    return new_lo, new_hi


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="เรนเดอร์ mesh หลายมุมเป็น PNG")
    p.add_argument("mesh", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("preview.png"))
    p.add_argument("--size", type=int, default=420)
    p.add_argument("--zoom", choices=sorted(ZOOMS), default=None,
                   help="ซูมเฉพาะส่วน (hands / feet / head)")
    p.add_argument("--angles", default="0,45,90,180",
                   help="มุม yaw เป็นองศา คั่นด้วยจุลภาค")
    p.add_argument("--pitch", type=float, default=0.0, help="ก้มกล้องกี่องศา")
    a = p.parse_args(argv)

    import cv2

    verts, faces = load_mesh(a.mesh)
    print(f"{a.mesh.name}: {len(verts)} verts / {len(faces)} tris  "
          f"bounds={np.round(verts.max(0) - verts.min(0), 3)}")

    bounds = crop_bounds(verts, a.zoom) if a.zoom else None
    tiles = []
    for deg in (float(s) for s in a.angles.split(",")):
        img = rasterize(verts, faces, a.size, np.radians(deg),
                        np.radians(a.pitch), bounds)
        cv2.putText(img, f"{deg:g}deg", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 220, 120), 1, cv2.LINE_AA)
        tiles.append(img)

    sheet = np.concatenate(tiles, axis=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), sheet[:, :, ::-1])
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
