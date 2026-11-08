"""อ่าน/เขียน .glb เท่าที่เครื่องมือในโฟลเดอร์นี้ต้องใช้ ด้วย numpy ล้วน

แยกออกมาเพราะทั้ง smooth_normals และ autorig ต้องแกะ binary chunk เหมือนกัน
ไม่ได้ตั้งใจให้ครอบคลุมสเปก glTF ทั้งหมด รองรับเท่าที่ไฟล์ในโปรเจกต์นี้เป็น
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path: Path) -> tuple[dict, bytearray]:
    data = Path(path).read_bytes()
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
    raw_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    raw_json += b" " * (-len(raw_json) % 4)
    raw_bin = bytes(blob)
    raw_bin += b"\x00" * (-len(raw_bin) % 4)

    total = 12 + 8 + len(raw_json) + 8 + len(raw_bin)
    out = bytearray()
    out += struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<II", len(raw_json), JSON_CHUNK) + raw_json
    out += struct.pack("<II", len(raw_bin), BIN_CHUNK) + raw_bin
    Path(path).write_bytes(out)


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
        # copy เพราะ view ที่ค้างอยู่จะล็อก bytearray ไม่ให้ขยายตอนต่อท้าย
        return flat.reshape(count, ncomp).copy()

    raw = np.frombuffer(blob, dtype=np.uint8, count=count * stride, offset=start)
    rows = raw.reshape(count, stride)[:, :item].copy()
    return rows.view(dtype).reshape(count, ncomp)


def append_view(gltf: dict, blob: bytearray, payload: bytes, target: int | None = None) -> int:
    blob += b"\x00" * (-len(blob) % 4)
    offset = len(blob)
    blob += payload
    view: dict = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
    if target is not None:
        view["target"] = target
    gltf.setdefault("bufferViews", []).append(view)
    gltf["buffers"][0]["byteLength"] = len(blob)
    return len(gltf["bufferViews"]) - 1


def first_triangle_mesh(gltf: dict, blob: bytearray) -> tuple[np.ndarray, np.ndarray]:
    """คืน (vertices Nx3, faces Mx3) ของ primitive สามเหลี่ยมก้อนใหญ่สุด"""
    best = None
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4 or "indices" not in prim:
                continue
            acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            if best is None or acc["count"] > best[0]:
                best = (acc["count"], prim)
    if best is None:
        raise SystemExit("ไม่พบ primitive สามเหลี่ยมในไฟล์นี้")

    prim = best[1]
    verts = read_accessor(gltf, blob, prim["attributes"]["POSITION"]).astype(np.float64)
    faces = read_accessor(gltf, blob, prim["indices"]).reshape(-1, 3).astype(np.int64)
    return verts, faces
