"""ขั้นที่ 6: เขียน mesh ออกเป็นไฟล์ที่ engine และ three.js อ่านได้

GLB เป็นรูปแบบหลัก เพราะเป็นไฟล์เดียวจบ (geometry + สี + ภายหลังคือ skeleton
และ animation) และ `GLTFLoader` ของ three.js โหลดได้ตรง ๆ
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import trimesh

from .meshing import MeshData

log = logging.getLogger(__name__)


def build_trimesh(
    verts_world: np.ndarray,
    mesh: MeshData,
    texture: np.ndarray | None = None,
    normal: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """ประกอบเป็น trimesh พร้อมสี — ใช้ texture ถ้ามี ไม่งั้นใช้ vertex color

    ไม่ใส่ทั้งคู่พร้อมกัน เพราะใน glTF ค่า COLOR_0 จะถูก *คูณ* เข้ากับสีจาก
    texture ผลคือภาพมืดลงเป็นสองเท่าโดยไม่มีใครตั้งใจ
    """
    kwargs: dict = {}

    if texture is not None and mesh.uv is not None:
        from PIL import Image

        # ต้องกำหนด metallic/roughness เอง ค่าเริ่มต้นของ glTF คือ
        # metallicFactor = 1.0 ซึ่งแปลว่า "ผิวโลหะเต็มร้อย" ผิวโลหะสะท้อนภาพ
        # สิ่งแวดล้อมเป็นหลัก ฉากที่ไม่มี environment map จึงเรนเดอร์ออกมา
        # เกือบดำสนิท ทั้งที่ texture ถูกต้องทุกอย่าง
        # trimesh พลิกแกน V ตอนเขียน glTF (v -> 1-v, ส่วน U คงเดิม) เพราะมันเก็บ
        # UV ตามแบบ OpenGL ที่จุดกำเนิดอยู่มุมซ้าย*ล่าง* ส่วน glTF อยู่ซ้าย*บน*
        # เรา bake ภาพไว้ตามแบบ glTF จึงต้องพลิกภาพกลับมาชดเชยให้ตรงกัน
        #
        # ถ้าลืมข้อนี้ อาการที่ได้คือ texture ติดครบแต่แมปผิดที่เป็นหย่อม ๆ
        # (หน้าอกกลายเป็นผ้ายีนส์) ไม่ใช่ภาพกลับหัวทั้งใบอย่างที่คาด เพราะแต่ละ
        # island ไปตกบนบริเวณอื่นของ atlas คนละที่กัน
        extra = {}
        roughness = 0.85
        if normal is not None:
            # normal map ต้องพลิกแกน V ให้ตรงกับ baseColor เพราะใช้ UV ชุดเดียวกัน
            extra["normalTexture"] = Image.fromarray(
                np.ascontiguousarray(normal[::-1]))
            # ผิวที่ด้านสนิทสะท้อนแสงเท่ากันทุกทิศ การบิด normal จึงแทบไม่
            # เปลี่ยนอะไรเลย ต้องเหลือความมันไว้บ้าง ร่องเกล็ดกับรอยยับผ้า
            # ถึงจะมีไฮไลต์วิ่งให้เห็นว่าผิวไม่ได้เรียบ
            roughness = 0.7
        material = trimesh.visual.material.PBRMaterial(
            name="character",
            baseColorTexture=Image.fromarray(
                np.ascontiguousarray(texture[::-1])),
            metallicFactor=0.0,
            roughnessFactor=roughness,
            **extra,
        )
        kwargs["visual"] = trimesh.visual.TextureVisuals(
            uv=np.asarray(mesh.uv, np.float32),
            material=material,
        )
    elif mesh.colors is not None:
        rgba = np.empty((len(mesh.colors), 4), np.uint8)
        rgba[:, :3] = mesh.colors
        rgba[:, 3] = 255
        kwargs["vertex_colors"] = rgba

    tm = trimesh.Trimesh(
        vertices=np.asarray(verts_world, np.float32),
        faces=np.asarray(mesh.faces, np.int32),
        process=False,   # อย่าให้ trimesh ไปรวม vertex เอง จะทำ UV และสีเพี้ยน
        **kwargs,
    )
    tm.fix_normals()
    return tm


def export(
    verts_world: np.ndarray,
    mesh: MeshData,
    out_dir: Path,
    name: str = "character",
    formats: tuple[str, ...] = ("glb",),
    texture: np.ndarray | None = None,
    normal: np.ndarray | None = None,
) -> dict[str, Path]:
    """เขียนไฟล์ตามฟอร์แมตที่ขอ คืน dict ของ {ฟอร์แมต: path}"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tm = build_trimesh(verts_world, mesh, texture, normal)

    # บังคับให้คำนวณ vertex normal ไว้ก่อน แล้วสั่ง exporter ให้เขียนลงไฟล์ด้วย
    #
    # ถ้า glTF ไม่มี attribute NORMAL ตัวโหลด (three.js, Blender, engine ทั่วไป)
    # จะคิด normal ใหม่แบบ flat ต่อสามเหลี่ยม ผิวที่เรียบสนิทจะกลายเป็นเหลี่ยม
    # และเพราะ decimate สร้างสามเหลี่ยมยาวตามแนวนอน (ผิวถูกปั้นทีละแถวความสูง)
    # แต่ละแถบจะอ่านออกมาเป็นสันวงแหวนรอบตัว ทั้งที่เรขาคณิตไม่มีสันอยู่จริง
    _ = tm.vertex_normals

    written: dict[str, Path] = {}
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        if fmt == "glb":
            path.write_bytes(tm.export(file_type="glb", include_normals=True))
        else:
            tm.export(str(path))
        size_mb = path.stat().st_size / 1e6
        log.info("เขียน %-4s -> %s  (%.2f MB)", fmt.upper(), path, size_mb)
        written[fmt] = path
    return written


def recompress_textures(path: Path, quality: int, max_size: int = 0) -> dict:
    """เข้ารหัสภาพใน GLB ใหม่เป็น JPEG แล้วอัดบัฟเฟอร์ใหม่ทั้งก้อน

    texture กิน 70-80% ของขนาดไฟล์ การเก็บเป็น PNG (ไม่สูญเสีย) จึงแพงมาก
    สำหรับภาพถ่าย/ภาพวาดที่มีไล่เฉดเต็มไปหมด JPEG เล็กกว่าหลายเท่าโดยตาแทบ
    แยกไม่ออก และ glTF รองรับ image/jpeg มาแต่ต้น ไม่ต้องใช้ส่วนขยายใด ๆ
    จึงเปิดได้ทุกที่โดยไม่ต้องตั้ง decoder เพิ่ม ต่างจาก KTX2 หรือ WebP

    เปลี่ยนขนาดภาพแล้ว byteLength ของ bufferView เปลี่ยนตาม ทุก offset หลังจาก
    นั้นจึงเลื่อนหมด ต้องอัดบัฟเฟอร์ใหม่ทั้งก้อนแล้วเขียน offset ใหม่ทุกตัว
    """
    import io

    from PIL import Image
    from pygltflib import GLTF2

    before = path.stat().st_size
    g = GLTF2().load(str(path))
    blob = g.binary_blob()

    views = []
    for bv in g.bufferViews:
        off = bv.byteOffset or 0
        views.append(bytearray(blob[off : off + bv.byteLength]))

    # normal map เก็บ *ทิศทาง* ไว้ในค่าสี ไม่ใช่ภาพให้คนดู artifact ของ JPEG
    # จึงโผล่เป็นคลื่นบนผิวตอนโดนแสง — บีบได้ แต่ต้องคุณภาพสูงกว่าภาพสีปกติ
    # (เก็บเป็น PNG ไปเลยก็ได้ แต่ 2048x2048 กิน 6 MB ซึ่งหนักเกินไปสำหรับเว็บ)
    keep = set()
    for m in g.materials or []:
        if m.normalTexture is not None and m.normalTexture.index is not None:
            keep.add(g.textures[m.normalTexture.index].source)

    changed = 0
    for i, im in enumerate(g.images):
        if im.bufferView is None:
            continue
        raw = bytes(views[im.bufferView])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if max_size and max(img.size) > max_size:
            img = img.resize((max_size, max_size), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG",
                 quality=min(97, quality + 12) if i in keep else quality,
                 optimize=True, progressive=True, subsampling=0)
        new = buf.getvalue()
        if len(new) >= len(raw):
            continue                      # บีบแล้วไม่เล็กลง ก็ไม่ต้องเปลี่ยน
        views[im.bufferView] = bytearray(new)
        im.mimeType = "image/jpeg"
        changed += 1

    if not changed:
        return {"before": before, "after": before, "images": 0}

    # อัดใหม่ทั้งก้อน จัดให้ทุก bufferView เริ่มที่ตำแหน่งหาร 4 ลงตัว
    out = bytearray()
    for i, bv in enumerate(g.bufferViews):
        out += b"\x00" * ((-len(out)) % 4)
        bv.byteOffset = len(out)
        bv.byteLength = len(views[i])
        out += views[i]
    out += b"\x00" * ((-len(out)) % 4)

    g.buffers[0].byteLength = len(out)
    g.set_binary_blob(bytes(out))
    g.save(str(path))

    after = path.stat().st_size
    log.info("บีบ texture %d ภาพใน %s: %.2f -> %.2f MB (เหลือ %.0f%%)",
             changed, path.name, before / 1e6, after / 1e6, 100 * after / before)
    return {"before": before, "after": after, "images": changed}


def write_summary(
    path: Path,
    mesh: MeshData,
    verts_world: np.ndarray,
    extra: dict | None = None,
) -> Path:
    """สรุปผลการรันเป็น JSON ไว้ตรวจย้อนหลังและเทียบระหว่างการตั้งค่า"""
    payload = {
        "verts": int(mesh.n_verts),
        "tris": int(mesh.n_faces),
        "has_vertex_colors": mesh.colors is not None,
        "bounds_m": {
            "width_x": float(np.ptp(verts_world[:, 0])),
            "height_y": float(np.ptp(verts_world[:, 1])),
            "depth_z": float(np.ptp(verts_world[:, 2])),
        },
        "stages": mesh.history,
    }
    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("เขียนสรุปผล -> %s", path)
    return path
