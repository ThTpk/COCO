"""เติม NORMAL แบบเกลี่ยเรียบให้ไฟล์ .glb ที่ไม่มี normal มาด้วย

กลไกเนื้อแท้: ตามสเปก glTF ถ้า primitive ไม่มี attribute NORMAL
เบราว์เซอร์ต้อง shade แบบ flat คือใช้ normal ของหน้าสามเหลี่ยมตรง ๆ
ผิวจึงออกมาเป็นเหลี่ยม สคริปต์นี้คำนวณ normal เฉลี่ยรายจุดยอดแล้วเขียน
กลับเข้าไฟล์ ทำให้แสงไล่ข้ามรอยต่อสามเหลี่ยมอย่างต่อเนื่อง

จุดที่ต้องระวังและสคริปต์นี้จัดการให้แล้ว: ตัวคลี่ UV (เช่น xatlas)
จะ "ผ่า" จุดยอดตามตะเข็บ ทำให้จุดเดียวกันในเชิงเรขาคณิตกลายเป็นหลายจุด
ถ้าเฉลี่ยตาม index ตรง ๆ จะเห็นรอยแสงเป็นเส้นตามตะเข็บ เลยต้อง
จับกลุ่มจุดที่พิกัดตรงกันก่อนแล้วค่อยเฉลี่ย

ใช้: python tools/smooth_normals.py เข้า.glb ออก.glb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glbmesh import append_view, read_accessor, read_glb, write_glb  # noqa: E402

# คอนโซลวินโดวส์ไทยเป็น cp874 เขียนลูกศรหรืออักษรนอกตารางไม่ได้ บังคับเป็น utf-8
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def smooth_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """normal เฉลี่ยรายจุดยอด ถ่วงน้ำหนักด้วยพื้นที่สามเหลี่ยม

    ไม่ normalize cross product ก่อนบวก ความยาวของมันคือสองเท่าของพื้นที่
    สามเหลี่ยมอยู่แล้ว สามเหลี่ยมใหญ่จึงมีน้ำหนักมากตามธรรมชาติ
    """
    pos = positions.astype(np.float64, copy=False)

    # จับกลุ่มจุดที่พิกัดตรงกัน เพื่อให้ตะเข็บ UV ไม่กลายเป็นรอยแสง
    _, group = np.unique(pos, axis=0, return_inverse=True)
    group = group.reshape(-1)

    a, b, c = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    face_n = np.cross(b - a, c - a)

    acc = np.zeros((group.max() + 1, 3), dtype=np.float64)
    for col in range(3):
        np.add.at(acc, group[faces[:, col]], face_n)

    length = np.linalg.norm(acc, axis=1, keepdims=True)
    # จุดโดดเดี่ยวหรือจุดที่หน้ารอบตัวหักล้างกันพอดี ปล่อยเป็นเวกเตอร์ขึ้นแทนศูนย์
    degenerate = (length < 1e-20).reshape(-1)
    acc[degenerate] = (0.0, 1.0, 0.0)
    length[degenerate] = 1.0

    return (acc / length)[group].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument(
        "--force",
        action="store_true",
        help="คำนวณใหม่ทับของเดิม แม้ primitive นั้นจะมี NORMAL อยู่แล้ว",
    )
    args = ap.parse_args()

    gltf, blob = read_glb(args.src)
    touched = 0

    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue  # ไม่ใช่สามเหลี่ยม ไม่มี normal ให้เกลี่ย
            if "NORMAL" in prim["attributes"] and not args.force:
                print(f"ข้าม {mesh.get('name', '?')}: มี NORMAL อยู่แล้ว")
                continue
            if "indices" not in prim:
                print(f"ข้าม {mesh.get('name', '?')}: ไม่มี index")
                continue

            positions = read_accessor(gltf, blob, prim["attributes"]["POSITION"])
            faces = read_accessor(gltf, blob, prim["indices"]).reshape(-1, 3)
            normals = smooth_normals(positions, faces.astype(np.int64))

            view = append_view(gltf, blob, normals.tobytes(), target=34962)
            gltf["accessors"].append(
                {
                    "bufferView": view,
                    "componentType": 5126,
                    "type": "VEC3",
                    "count": int(normals.shape[0]),
                    # min/max ไม่บังคับสำหรับ NORMAL แต่ใส่ไว้ช่วยให้ validator เงียบ
                    "min": [float(v) for v in normals.min(axis=0)],
                    "max": [float(v) for v in normals.max(axis=0)],
                }
            )
            prim["attributes"]["NORMAL"] = len(gltf["accessors"]) - 1
            touched += 1
            print(
                f"{mesh.get('name', '?')}: {len(positions):,} จุดยอด "
                f"{len(faces):,} สามเหลี่ยม: เติม NORMAL แล้ว"
            )

    if not touched:
        print("ไม่มี primitive ไหนต้องแก้ ไม่ได้เขียนไฟล์ออก")
        return 1

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    write_glb(args.dst, gltf, blob)
    print(
        f"เขียน {args.dst} แล้ว "
        f"({args.src.stat().st_size / 1048576:.2f} MB -> "
        f"{args.dst.stat().st_size / 1048576:.2f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
