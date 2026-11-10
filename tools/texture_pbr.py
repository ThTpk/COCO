"""สร้างวัสดุ PBR จาก texture สีที่ฉายมาแล้ว: roughness/metallic แยกตามชนิดวัสดุ + normal map จากลายเกล็ด

    python tools/texture_pbr.py <model.glb> <out_prefix> [--normal-strength 6]

เขียน <out_prefix>_orm.png (R=occlusion 1, G=roughness, B=metallic ตามมาตรฐาน glTF),
<out_prefix>_normal.png (tangent space แกน +Y ขึ้นแบบ glTF) และ <out_prefix>.glb ที่ผูกทั้งสามแผ่น

แยกชนิดวัสดุจากสีของเท็กเซล + ตำแหน่ง 3 มิติ (แกน glTF ของโมเดลชุด croc_full_*):
  กำไล   |x| 0.74-0.83 สีเทาจาง        -> โลหะ  roughness 0.35  metallic 1
  อัญมณี  เขียวเด่น ในแถบกำไล           -> roughness 0.10
  ฟัน/เล็บ สีขาวอมเทา บนหัว มือ เท้า       -> roughness 0.30
  ผิวเกล็ด สีส้ม                         -> roughness 0.55 (หนังสัตว์เลื้อยคลานกึ่งมัน)
  เสื้อ / ยีนส์ สีน้ำเงิน                  -> roughness 0.85 / 0.90
normal map มาจากความสว่างที่ผ่าน high-pass (ร่องเกล็ดมืด = ต่ำ) แรงเต็มบนผิว เบาบนผ้า และกันขอบเกาะ UV
"""
import argparse
import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("glb"); ap.add_argument("out")
ap.add_argument("--normal-strength", type=float, default=6.0)
a = ap.parse_args()

m = trimesh.load(a.glb, force="mesh")
mat = m.visual.material
base = (mat.baseColorTexture if getattr(mat, "baseColorTexture", None) is not None else mat.image).convert("RGB")
tex = np.asarray(base).astype(np.float32)
H, W = tex.shape[:2]
uv, F = m.visual.uv, m.faces
C = m.vertices[F].mean(1)

# ตำแหน่ง 3 มิติของทุกเท็กเซล: วาด id ของหน้าลงผืน UV แล้วขยายขึ้น
R = 2048
fid = Image.new("RGB", (R, R), (0, 0, 0)); d = ImageDraw.Draw(fid)
P = np.c_[uv[:, 0] * R, (1 - uv[:, 1]) * R]
for i in range(len(F)):
    k = i + 1
    d.polygon([tuple(P[j]) for j in F[i]], fill=(k & 255, (k >> 8) & 255, (k >> 16) & 255))
ia = np.asarray(fid).astype(np.int64)
ids = ia[..., 0] + (ia[..., 1] << 8) + (ia[..., 2] << 16) - 1
ids = cv2.resize(ids.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
valid = ids >= 0
pos = C[np.clip(ids, 0, None)]
x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
ax = np.abs(x)

r, g, b = tex[..., 0], tex[..., 1], tex[..., 2]
mx, mn = tex.max(2), tex.min(2)
sat = (mx - mn) / np.maximum(mx, 1)
lum = 0.299 * r + 0.587 * g + 0.114 * b

skin = (r > b + 20)
cloth = (b > r + 5)
shirt = cloth & (y > -0.02)
jeans = cloth & (y <= -0.02)
band = (ax > 0.73) & (ax < 0.84) & (y > 0.2)
gem = (ax > 0.6) & (y > 0.2) & (g > r + 8) & (g > b + 5)
metal = band & ~gem & (sat < 0.35) & (lum > 70)
headr = y > 0.62
extremity = (ax > 0.83) | (y < -0.8) | headr
ivory = extremity & (sat < 0.30) & (lum > 120) & ~metal   # ฟัน เล็บ

rough = np.full((H, W), 0.6, np.float32)
metal_m = np.zeros((H, W), np.float32)
rough[skin] = 0.55
rough[shirt] = 0.85
rough[jeans] = 0.90
rough[ivory] = 0.30
rough[metal] = 0.35; metal_m[metal] = 1.0
rough[gem] = 0.10
# ขอบวัสดุนุ่มลงเล็กน้อย กันรอยหยักตอนเรนเดอร์
rough = cv2.GaussianBlur(rough, (0, 0), 1.0)
metal_m = cv2.GaussianBlur(metal_m, (0, 0), 1.0)
for nm, mk in [("skin", skin), ("shirt", shirt), ("jeans", jeans), ("ivory", ivory), ("metal", metal), ("gem", gem)]:
    print(f"  {nm:6s} {100 * (mk & valid).sum() / valid.sum():5.1f}% ของเท็กเซล")

orm = np.dstack([np.full((H, W), 255, np.uint8), (rough * 255).astype(np.uint8), (metal_m * 255).astype(np.uint8)])
Image.fromarray(orm).save(a.out + "_orm.png")

# --- normal map -------------------------------------------------------------
L = lum / 255.0
hp = L - cv2.GaussianBlur(L, (0, 0), 8)            # ลายเกล็ด/รอยผ้า
hp = np.clip(hp, -0.15, 0.15)
gain = np.where(skin, 1.0, np.where(cloth, 0.35, 0.6)).astype(np.float32)
h = cv2.GaussianBlur(hp * gain, (0, 0), 0.8)
ok = valid.astype(np.uint8)
inner = cv2.erode(ok, np.ones((3, 3), np.uint8)).astype(bool)   # ไม่คิดเกรเดียนต์ข้ามขอบเกาะ UV
gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
gx[~inner] = 0; gy[~inner] = 0
s = a.normal_strength * W / 1024.0
n = np.dstack([-gx * s, gy * s, np.ones_like(gx)])   # +Y = ทิศ V (ขึ้น) ตาม glTF
n /= np.linalg.norm(n, axis=2, keepdims=True)
Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(a.out + "_normal.png")

pbr = trimesh.visual.material.PBRMaterial(
    baseColorTexture=base,
    metallicRoughnessTexture=Image.open(a.out + "_orm.png"),
    normalTexture=Image.open(a.out + "_normal.png"),
    metallicFactor=1.0, roughnessFactor=1.0)
m.visual = trimesh.visual.TextureVisuals(uv=uv, material=pbr)
m.export(a.out + ".glb")
print("เขียน", a.out + ".glb")
