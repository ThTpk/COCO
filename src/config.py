"""ค่าตั้งกลางของ pipeline 2D -> 3D"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
DEBUG_DIR = OUTPUT_DIR / "debug"

BLENDER_EXE = os.environ.get(
    "BLENDER_EXE",
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
)

# ---------------------------------------------------------------------------
# นิยามของมุมกล้อง
#
# ใช้ระบบพิกัดแบบ glTF / three.js:  +X = ขวาของตัวละคร, +Y = ขึ้น, +Z = เข้าหาผู้ชม
#
# แต่ละมุมบอกว่า "พิกเซล (u, v) ของภาพ แมปไปแกนโลกไหน"
#   axis_u : แกนโลกที่วิ่งตามความกว้างของภาพ (u เพิ่ม -> ค่าแกนนั้นเพิ่ม)
#   axis_v : แกนโลกที่วิ่งตามความสูงของภาพ (v เพิ่มคือลงล่าง -> ค่าแกนนั้นลด)
#   flip_u : ภาพนี้ต้องพลิกซ้ายขวาก่อนใช้หรือไม่
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewSpec:
    name: str
    axis_u: str          # "x" | "z"
    flip_u: bool         # พลิกแนวนอนก่อนใช้
    depth_axis: str      # แกนที่มุมนี้ "มองไม่เห็น" (ใช้ตอน carve)
    required: bool = True


VIEWS: dict[str, ViewSpec] = {
    # มองจากด้านหน้า: กล้องอยู่ +Z มองไปทาง -Z
    #   u เพิ่ม (ซ้าย->ขวาของภาพ) = +X,  มองไม่เห็นแกน Z
    "front": ViewSpec("front", axis_u="x", flip_u=False, depth_axis="z"),
    # มองจากด้านหลัง: กล้องอยู่ -Z มองไปทาง +Z
    #   u เพิ่มในภาพ = -X ของโลก  ->  ต้องพลิกซ้ายขวาให้ตรงกับ front
    "back": ViewSpec("back", axis_u="x", flip_u=True, depth_axis="z"),
    # มองจากด้านข้าง (ข้างซ้ายของตัวละคร): กล้องอยู่ -X มองไปทาง +X
    #   u เพิ่มในภาพ = +Z,  มองไม่เห็นแกน X
    "side": ViewSpec("side", axis_u="z", flip_u=False, depth_axis="x"),
    # มุมเสริม ถ้ามีภาพข้างอีกฝั่ง
    "side_r": ViewSpec("side_r", axis_u="z", flip_u=True, depth_axis="x", required=False),
}

# ชื่อไฟล์ที่ยอมรับต่อมุม (ไม่สนตัวพิมพ์ใหญ่เล็ก)
VIEW_ALIASES: dict[str, tuple[str, ...]] = {
    "front": ("front", "f", "หน้า", "0", "000"),
    "back":  ("back", "b", "rear", "หลัง", "180"),
    "side":  ("side", "s", "left", "side_l", "ข้าง", "270"),
    "side_r": ("side_r", "right", "90"),
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass
class PipelineConfig:
    # --- preprocess ---
    canvas: int = 1024              # ความสูงของ canvas หลัง normalize (พิกเซล)
    margin: float = 0.04            # เว้นขอบบน/ล่างเป็นสัดส่วนของ canvas
    mask_threshold: int = 127       # ตัด alpha/ความสว่างที่ค่านี้
    bg_tolerance: int = 18          # ความคลาดสีพื้นหลัง (ใช้เมื่อไม่มี alpha)
    # โมเดล rembg: isnet-general-use ใช้ได้กับทุกวัตถุ รวมถึงตัวละครที่ไม่ใช่มนุษย์
    # (u2net_human_seg เทรนมากับคนจริง จะพลาดกับสัตว์ประหลาด/หุ่นยนต์/ตัวการ์ตูน)
    rembg_model: str = "isnet-general-use"
    grabcut_size: int = 768         # ย่อภาพเหลือด้านยาวเท่านี้ก่อนรัน GrabCut
                                    # ต่ำกว่านี้ช่องแคบอย่างหว่างขาจะหายไป
    grabcut_iters: int = 7          # จำนวนรอบของ GrabCut

    # --- visual hull ---
    voxel_res: int = 320            # จำนวน voxel ตามความสูงของตัวละคร
                                    # หัวสูงราว 1/7 ของตัว ที่ 256 เหลือแค่ ~36 voxel
                                    # ซึ่งน้อยเกินกว่าจะเก็บรูปปากกับกะโหลกไว้ได้
    # เบลอ occupancy ก่อน marching cubes
    # นอกจากลบเหลี่ยมแล้ว ยังทำหน้าที่ *หลอมก้อนที่ปั้นแยกกันให้กลืนเป็นเนื้อ
    # เดียวกัน* (ลำตัว + ก้อนปาก + หาง) ซึ่ง union แบบ boolean ทำไม่ได้
    # ต่ำกว่า ~2 จะเห็นรอยต่อเป็นจานแยกชิ้นระหว่างหัวกับไหล่
    hull_blur: float = 2.6
    iso_level: float = 0.5

    # วิธีสร้างหน้าตัดแนวนอน
    #   "hull"    = ตัดกันตรง ๆ ตามนิยาม visual hull -> หน้าตัดเป็นสี่เหลี่ยม
    #   "ellipse" = หน้าตัดเป็นวงรี/superellipse -> เหมือนร่างกายจริงมากกว่ามาก
    # ด้วยภาพแค่ 3 มุม (0/90/180) มีแค่สองแกนที่ให้ข้อมูลจริง หน้าตัดแบบ hull
    # จึงออกมาเป็นกล่องเสมอ "ellipse" แลกความเป๊ะของ silhouette ด้านข้าง
    # กับรูปทรงที่เป็นอินทรีย์กว่า ซึ่งคุ้มกว่ามากสำหรับตัวละคร
    carve_mode: str = "ellipse"
    cross_section_power: float = 2.2   # 2 = วงรี, สูงขึ้น = เข้าใกล้สี่เหลี่ยมมุมมน
    depth_falloff: float = 0.5         # ชิ้นส่วนที่แคบ (แขน) ได้ความลึกตามสัดส่วนยกกำลังนี้
    # เพดานอัตราส่วน ลึก/กว้าง ของหน้าตัดลำตัว
    # กันไม่ให้ส่วนที่ยื่นไปข้างหน้า (ปากสัตว์) ลากความลึกของตัวเองไปครอบ
    # ความกว้างของบ่า จนเกิดแผ่นแบนคาดรอบคอ ตั้ง 0 เพื่อปิดการจำกัด
    max_aspect: float = 1.15
    # แถวที่กว้างกี่เท่าของความกว้างหัว จึงจะถูกจำกัดเต็มที่ (ไล่ระดับระหว่างนั้น)
    aspect_blend: float = 2.0
    # ความกว้างของก้อนปากที่คืนกลับมา เป็นกี่เท่าของความกว้างหัว
    # 0 = ไม่คืน (ตัวละครจะหน้าสั้นกุดแต่คอเนียน)
    snout_width: float = 1.15
    # ความกว้างของสันหนามที่คืนกลับมา เป็นกี่เท่าของความกว้างหัว
    # หนามอยู่บนแนวกลางหลัง จึงต้องแคบกว่าลำตัวมาก 0 = ไม่คืนหนาม
    crest_width: float = 0.45
    crest_min: float = 2.0          # ส่วนยื่นต้องลึกกว่านี้ (voxel) จึงนับเป็นหนาม
    # ใช้เฉพาะก้อนที่ใหญ่ที่สุดของ mask ด้านข้างในแต่ละแถว
    # ตัดส่วนที่ยื่นออกไปอย่างหางหรือเป้สะพายหลังทิ้ง ไม่ให้ไปพองความหนาทั้งตัว
    side_main_component_only: bool = True
    # เกลาโปรไฟล์ความลึก/ความกว้างตามแนวตั้งกี่แถว ก่อนเอาไปปั้นหน้าตัด
    # 0 = ไม่เกลา ซึ่งจะได้ผิวเป็นวงแหวนขั้นบันได
    profile_smooth: float = 3.0
    # ลบส่วนยื่นที่บางกว่านี้ (หน่วย voxel) ออกจาก silhouette ทั้งสองแกน
    # หนามสันหลัง เขา ครีบ ผมแหลม ถ้าไม่ลบจะกลายเป็นสันวงแหวนรอบตัว
    # และทำให้รูปทรงกระโดดเป็นขั้นบันไดตามแนวตั้ง ตั้ง 0 เพื่อเก็บไว้ทั้งหมด
    spike_open: int = 3
    # เกลาขอบ silhouette ตามแนวตั้งกี่ voxel ก่อนปั้นหน้าตัด
    # เส้นขอบที่สั่น 1-2 px ระหว่างแถว จะกลายเป็นลอนถี่พาดทั่วตัวถ้าไม่เกลา
    silhouette_smooth: float = 1.5
    # median filter บนโปรไฟล์ขอบหน้า/ขอบหลัง (หน่วย voxel ตามแนวตั้ง)
    # ตัดหนามสันหลังที่เป็นค่ากระโดดออกไป โดยไม่ทำให้รูปทรงจริงเบลอ
    # ต้องกว้างกว่าระยะห่างระหว่างหนามสองอัน จึงจะตัดได้หมด
    depth_median: int = 13

    # --- หาง / ส่วนยื่นด้านหลัง ---
    # เก็บก้อนที่อยู่ด้านหลังลำตัวใน mask ด้านข้าง กลับมาเป็นปริมาตรของตัวเอง
    # แทนที่จะทิ้งไป ทำให้ได้เนื้อหางที่กระดูกหางยึดเกาะได้จริง
    keep_tail: bool = True
    # --- ลากหางจากภาพด้านหลัง (ภาพเดียวเป็นแหล่งอ้างอิง) ---
    # หางห้อยทับกางเกงอยู่ใน silhouette จึงต้องแยกด้วยสี ไม่ใช่ขอบ
    tail_band_top: float = 0.40     # เริ่มมองหาหางที่กี่ส่วนจากยอดหัวลงมา
    tail_max_jump: float = 0.06     # ศูนย์กลางหางขยับได้กี่ส่วนของ canvas ต่อแถว
    tail_trace_min_score: float = 0.12   # คะแนนขั้นต่ำจึงจะเชื่อว่าเป็นหางจริง
    tail_path_smooth: float = 6.0   # เกลาเส้นทางหางตามแนวตั้ง
    # ดันหางออกไปด้านหลังกี่เท่าของรัศมี 0 = ครึ่งหนึ่งจมอยู่ในตัว, 1 = พ้นตัวพอดี
    tail_z_bias: float = 0.7
    tail_thickness: float = 1.0     # ตัวคูณความหนาหางโดยรวม
    tail_bones: int = 5             # จำนวนกระดูกในโซ่หาง
    # ท่อนขาที่อ้วนเกินกี่เท่าของความกว้างที่ติดตามมา ถือว่ามีหางเชื่อมติดมา
    # แล้วจะถูกตัดเหลือเฉพาะความกว้างของขา ต่ำไปจะกินขา สูงไปจะเหลือหางติด
    leg_clip_factor: float = 1.45

    # --- meshing ---
    taubin_iters: int = 14          # Taubin smoothing รักษาปริมาตรดีกว่า Laplacian
                                    # มากเกินไปจะกินรายละเอียดเล็ก ๆ อย่างปากกับนิ้ว
    target_tris: int = 40_000       # งบ polygon สำหรับเกม / three.js
    keep_largest_component: bool = True

    # --- texture ---
    texture_mode: str = "uv"        # "uv" = bake ลงภาพ, "vertex" = สีที่ vertex
    uv_texture_size: int = 2048
    # ขยายสีออกนอกขอบ island กี่พิกเซล กันเส้นดำตามรอยผ่าตอน GPU กรองสี
    uv_padding: int = 8
    # บีบ texture ใน GLB เป็น JPEG คุณภาพเท่านี้ (0 = เก็บเป็น PNG ไม่บีบ)
    # texture กิน 70-80% ของขนาดไฟล์ ตัวเลขนี้จึงมีผลมากที่สุด
    # บีบ geometry ด้วย Draco — เล็กลงราวครึ่งหนึ่ง
    # ผู้อ่านต้องตั้ง decoder เพิ่ม (Blender, three.js, Babylon, Unity glTFast,
    # Godot รองรับหมด แต่เครื่องมือเก่าบางตัวไม่รองรับ)
    draco: bool = True
    texture_quality: int = 88
    # ย่อ texture ใน GLB ลงเหลือด้านละเท่านี้ (0 = ไม่ย่อ)
    # ไฟล์ PNG ข้าง ๆ ยังเป็นความละเอียดเต็มเสมอ ไว้ retouch
    texture_glb_size: int = 0
    texture_power: float = 3.0      # ยิ่งสูง มุมที่หันตรงยิ่งมีน้ำหนักเหนือมุมเฉียง
    texture_edge_extend: int = 10   # ยืดสีออกนอก mask กี่พิกเซล กัน vertex ขอบดูดสีดำ

    # --- world ---
    character_height_m: float = 1.70   # ความสูงจริงของตัวละคร ใช้ scale mesh เป็นเมตร

    # --- rig ---
    rig: str = "blender"            # "blender" | "none"  ("none" = rig เองใน Mixamo)
    animate: bool = True            # สร้าง clip อนิเมชันใส่ไปในไฟล์ที่ rig แล้ว
    fps: int = 24

    # --- export ---
    export_formats: tuple[str, ...] = ("glb",)
    debug: bool = True

    def resolved_margin_px(self) -> int:
        return int(round(self.canvas * self.margin))
