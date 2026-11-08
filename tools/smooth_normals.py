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
import struct
import sys
from pathlib import Path

import numpy as np

# คอนโซลวินโดวส์ไทยเป็น cp874 เขียนลูกศรหรืออักษรนอกตารางไม่ได้ บังคับเป็น utf-8
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

# ชนิดข้อมูลของ accessor ที่สคริปต์นี้ต้องอ่าน
COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path: Path) -> tuple[dict, bytearray]:
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise SystemExit(f"{path} ไม่ใช่ไฟล์ .glb")
    if version != 2:
        raise SystemExit(f"รองรับเฉพาะ glTF 2.0 ไฟล์นี้เป็นเวอร์ชัน {version}")

    gltf: dict | None = None
    blob = bytearray()
    off = 12
    while off < len(data):
        length, kind = struct.unpack_from("<II", data, off)
        body = data[off + 8 : off + 8 + length]
        if kind == JSON_CHUNK:
            gltf = json.loads(body)
        elif kind == BIN_CHUNK:
            blob = bytearray(body)
        off += 8 + length

    if gltf is None:
        raise SystemExit(f"{path} ไม่มี JSON chunk")
    return gltf, blob


def write_glb(path: Path, gltf: dict, blob: bytearray) -> None:
    # glTF บังคับว่าทุก chunk ต้องยาวเป็นพหุคูณของ 4
    # JSON เติมด้วยช่องว่าง ส่วน BIN เติมด้วยศูนย์
    raw_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    raw_json += b" " * (-len(raw_json) % 4)
    raw_bin = bytes(blob)
    raw_bin += b"\x00" * (-len(raw_bin) % 4)

    total = 12 + 8 + len(raw_json) + 8 + len(raw_bin)
    out = bytearray()
    out += struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<II", len(raw_json), JSON_CHUNK) + raw_json
    out += struct.pack("<II", len(raw_bin), BIN_CHUNK) + raw_bin
    path.write_bytes(out)


def read_accessor(gltf: dict, blob: bytearray, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    if "sparse" in acc:
        raise SystemExit("ยังไม่รองรับ accessor แบบ sparse")

    dtype = COMPONENT_DTYPE[acc["componentType"]]
    ncomp = TYPE_COUNT[acc["type"]]
    count = acc["count"]
    view = gltf["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    item = np.dtype(dtype).itemsize * ncomp
    stride = view.get("byteStride") or item

    if stride == item:
        flat = np.frombuffer(blob, dtype=dtype, count=count * ncomp, offset=start)
        # ต้อง copy เพราะ view ที่ค้างอยู่จะล็อก bytearray ไม่ให้ขยายตอนต่อท้าย
        return flat.reshape(count, ncomp).copy()

    # ข้อมูลวางแบบ interleaved ต้องหยิบทีละแถวตามระยะก้าว
    raw = np.frombuffer(blob, dtype=np.uint8, count=count * stride, offset=start)
    rows = raw.reshape(count, stride)[:, :item].copy()
    return rows.view(dtype).reshape(count, ncomp)


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


def append_view(gltf: dict, blob: bytearray, payload: bytes) -> int:
    blob += b"\x00" * (-len(blob) % 4)  # bufferView ต้องเริ่มที่ตำแหน่งหาร 4 ลงตัว
    offset = len(blob)
    blob += payload
    gltf.setdefault("bufferViews", []).append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(payload), "target": 34962}
    )
    gltf["buffers"][0]["byteLength"] = len(blob)
    return len(gltf["bufferViews"]) - 1


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

            view = append_view(gltf, blob, normals.tobytes())
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
