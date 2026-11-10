"""ปรับ texture ตามสไตล์ที่ต้องการ: เสื้อผ้าสะอาด สีสด และผิวเป็นสีส้มตามรหัสที่กำหนด

    python tools/texture_stylize.py <model.glb> <out.png> <out.glb> [--skin "#F16322"] [--cloth-sat 1.35]

1. เสื้อ/ยีนส์: หาจุดที่ไม่ใช่สีผ้า (จุดส้ม/น้ำตาล/ดำ ที่ฉายมาผิด ทำให้ดูขาด) ในบริเวณเสื้อผ้า
   ที่เป็นก้อนเล็ก แล้วเติมด้วยสีผ้าข้างเคียง (เว้นโลโก้หน้าอกและคอเสื้อที่เห็นผิวคอจริง)
   ลบปื้นมืด-สว่างความถี่ต่ำที่ทำให้ผ้าดูเก่าซีด แล้วเพิ่มความอิ่มสีใน CIELAB
2. ผิว: ย้ายค่าเฉลี่ย Lab ของผิวไปที่สีเป้าหมาย คงลายเกล็ด (ส่วนต่างจากค่าเฉลี่ยเฉพาะที่)
   เว้นฟัน เล็บ กำไล อัญมณี
พิกัดที่ใช้แบ่งพื้นที่เป็นแกน glTF ของโมเดลชุด croc_full_*
"""
import argparse
import cv2
import numpy as np
import scipy.sparse as sp
import trimesh
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("glb"); ap.add_argument("out_png"); ap.add_argument("out_glb")
ap.add_argument("--skin", default="#F16322", help="ส้ม KMITL")
ap.add_argument("--skin-high", default="#FFCD39", help="เหลือง KMITL ใช้กับจุดสว่าง")
ap.add_argument("--highlight", type=float, default=0.45, help="ผสมเหลืองที่จุดสว่างมากสุดกี่ส่วน")
ap.add_argument("--cloth-sat", type=float, default=1.5)
ap.add_argument("--cloth-lift", type=float, default=5.0, help="เพิ่มความสว่างผ้า (หน่วย L)")
a = ap.parse_args()


def to_lab(rgb):
    return cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def from_lab(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255, 0, 255)


def norm_blur(img, mask, sigma):
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(img * (m[..., None] if img.ndim == 3 else m), (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    den_ = den[..., None] if img.ndim == 3 else den
    return num / np.maximum(den_, 1e-6), den


m = trimesh.load(a.glb, force="mesh")
mat = m.visual.material
tex = np.asarray((mat.baseColorTexture if getattr(mat, "baseColorTexture", None) is not None
                  else mat.image).convert("RGB")).astype(np.float32)
H, W = tex.shape[:2]
uv, F = m.visual.uv, m.faces
C = m.vertices[F].mean(1)

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
cloth = (b > r + 5)

# ---------------------------------------------------------------- เสื้อผ้า
# ---- แยกผิว/ผ้า ระดับหน้า (face) ด้วยสี + ตำแหน่ง + โหวตเสียงข้างมากของหน้าข้างเคียง
# 1) สีเริ่มต้นของแต่ละหน้า: น้ำเงิน = ผ้า, ส้ม = ผิว, อื่น ๆ = ยังไม่รู้
# 2) ตำแหน่งที่แน่ใจ: แขนพ้นแขนเสื้อ, หัวเหนือปกเสื้อ, เท้า, หางช่วงล่าง = ผิว; ปลายขากางเกง = ผ้า
# 3) โหวต 40 รอบบนกราฟหน้าที่ติดกัน — จุดสีผิดเล็ก ๆ ถูกกลืนหาย ส่วนพื้นที่จริง (คอเสื้อ) อยู่รอด
fc = []
for wts in ((1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)):
    pp = (uv[F] * np.array(wts)[None, :, None]).sum(1)
    fc.append(tex[np.clip(((1 - pp[:, 1]) * H).astype(int), 0, H - 1),
                  np.clip((pp[:, 0] * W).astype(int), 0, W - 1)])
fc = np.mean(fc, 0)
fx, fy, fz = C[:, 0], C[:, 1], C[:, 2]
flab = np.full(len(F), -1)
flab[fc[:, 2] > fc[:, 0] + 5] = 1
flab[fc[:, 0] > fc[:, 2] + 15] = 0
force_skin = (np.abs(fx) > 0.50) | (fy < -0.82) | (fy > 0.74) | ((fz < -0.2) & (fy < -0.3))
force_cloth = (fy > -0.79) & (fy < -0.70) & (fz > -0.2) & (np.abs(fx) < 0.4)
# ขากางเกงทั้งสองข้าง (ต่ำกว่าเอว เหนือชายขา ด้านหน้าของหาง) = ผ้าทั้งหมด
legs_col = ((fx > -0.42) & (fx < -0.10)) | ((fx > 0.06) & (fx < 0.38))
force_cloth |= legs_col & (fy > -0.70) & (fy < -0.06) & (fz > -0.12)
force_skin &= ~force_cloth
flab[force_skin] = 0; flab[force_cloth] = 1
geo = trimesh.Trimesh(m.vertices.copy(), F.copy(), process=False)
geo.merge_vertices(merge_tex=True, merge_norm=True)
adj = geo.face_adjacency
n = len(F)
A = sp.coo_matrix((np.ones(len(adj) * 2), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])),
                  shape=(n, n)).tocsr()
for _ in range(40):
    oh = np.zeros((n, 2)); oh[flab == 0, 0] = 1; oh[flab == 1, 1] = 1
    v = A @ oh + 0.5 * oh
    new_l = np.where(v.sum(1) > 0, v.argmax(1), flab)
    new_l[force_skin] = 0; new_l[force_cloth] = 1
    if (new_l == flab).all():
        break
    flab = new_l
# ---- ขอบขากางเกงที่ข้อเท้า: วัดจากภาพถ่าย หน้า -0.775 / ข้าง -0.80 / ส้นเท้า -0.825
# ผ้า = อยู่ในท่อขา (รัศมี < 0.10 จากแกนขา) และสูงกว่าแนวขอบ; หลังเท้า นิ้ว ส้น = ผิว
for x0, x1 in ((-0.42, -0.10), (0.06, 0.38)):
    col = (fx > x0) & (fx < x1)
    ref = col & (fy > -0.78) & (fy < -0.70) & (fz > -0.15)
    cx, cz = fx[ref].mean(), fz[ref].mean()
    rad = np.hypot(fx - cx, fz - cz)
    th = np.arctan2(fz - cz, fx - cx)          # +z (หน้า) = +pi/2
    hem = -0.80 + 0.025 * np.sin(th)
    band = col & (fy > -0.95) & (fy < -0.70) & (fz > -0.15) & (rad < 0.32)
    flab[band] = np.where((fy[band] > hem[band]) & (rad[band] < 0.10), 1, 0)
# ---- ชายแขนเสื้อ: วัดจากภาพหน้า/หลัง ด้านบนยาวถึง |x|~0.46 ใต้แขน ~0.40 (ตัดเป็นแนวเอียงสะอาด)
armr = (np.abs(fx) > 0.36) & (fy > 0.15) & (fy < 0.75)
hemx = np.clip(0.40 + 0.35 * (fy - 0.33), 0.39, 0.47)
flab[armr] = np.where(np.abs(fx[armr]) < hemx[armr], 1, 0)
# ---- โคนหาง: ส่วนที่อยู่หลังระนาบตัดตอนต่อหาง (z < -0.19) ใต้ชายเสื้อ = ผิวทั้งหมด
flab[(fy > -0.36) & (fy < -0.06) & (fz < -0.19)] = 0
np.save(a.out_png.rsplit(".", 1)[0] + "_facelabel.npy", flab)
tl = np.where(valid, flab[np.clip(ids, 0, None)], -1)
logo = (x > -0.24) & (x < 0.0) & (y > 0.28) & (y < 0.50) & (z > 0.08)
head = y > 0.70
arms = ax > 0.47
feet = y < -0.80
skin_rg = valid & (tl == 0) & ~logo
shirt_rg = valid & (tl == 1) & (y > -0.02)
jeans_rg = valid & (tl == 1) & (y <= -0.02)
print(f"  แยกส่วน: ผิว {skin_rg.mean() / valid.mean():.0%} เสื้อ {shirt_rg.mean() / valid.mean():.0%} "
      f"ยีนส์ {jeans_rg.mean() / valid.mean():.0%} ของเท็กเซล")
out = tex.copy()
for name, rg in (("shirt", shirt_rg), ("jeans", jeans_rg)):
    # ในบริเวณผ้า ทุกเท็กเซลที่ไม่ใช่สีผ้าคือสีที่ฉายมาผิด (ยกเว้นโลโก้)
    bad = rg & ~cloth & ~logo
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & rg & ~logo
    src = rg & cloth & ~bad
    filled = np.zeros_like(bad)
    for sg in (4, 8, 16, 32, 64, 128, 256):
        mean, den = norm_blur(out, src, sg)
        ok = bad & ~filled & (den > 0.05)
        out[ok] = mean[ok]; filled |= ok
    out[bad & ~filled] = out[src].mean(0)
    print(f"  {name}: เติมสีผ้าแทนจุดเสีย {bad.sum():,} พิกเซล")
    lab = to_lab(out)
    clothpx = rg & ~(logo & ~cloth)
    low, _ = norm_blur(lab[..., 0], clothpx, 40)
    target = np.median(lab[..., 0][clothpx])
    lab[..., 0] = np.where(clothpx, lab[..., 0] + (target - low) * 0.75 + a.cloth_lift, lab[..., 0])
    lab[..., 1] = np.where(clothpx, lab[..., 1] * a.cloth_sat, lab[..., 1])
    lab[..., 2] = np.where(clothpx, lab[..., 2] * a.cloth_sat, lab[..., 2])
    out = np.where(clothpx[..., None], from_lab(lab), out)

# ---------------------------------------------------------------- ผิว
r, g, b = tex[..., 0], tex[..., 1], tex[..., 2]
mx, mn = tex.max(2), tex.min(2)
sat = (mx - mn) / np.maximum(mx, 1)
lum = 0.299 * r + 0.587 * g + 0.114 * b
# ---- กำไลข้อมือ: แถบรอบข้อมือ (ช่วง |x| ต่อแขน) ที่สีหม่นแบบโลหะ และอยู่ใกล้แกนข้อมือ (ตัดนิ้วโป้งออก)
fsat = (fc.max(1) - fc.min(1)) / np.maximum(fc.max(1), 1)
fcuff = np.zeros(len(F), bool)
for sg, (c0, c1) in ((-1, (0.755, 0.83)), (1, (0.715, 0.80))):
    band = (sg * fx > c0) & (sg * fx < c1) & (fy > 0.2)
    seed = band & (fsat < 0.45)
    cy, cz = np.median(fy[seed]), np.median(fz[seed])
    rad = np.hypot(fy - cy, fz - cz)
    rlim = np.percentile(rad[seed], 80) + 0.006
    fcuff |= band & (rad < rlim)
    print(f"  กำไล {'ซ้าย' if sg < 0 else 'ขวา'}: {(band & (rad < rlim)).sum():,} หน้า (รัศมี < {rlim:.3f})")
bracelet = valid & fcuff[np.clip(ids, 0, None)]
ivory = (sat < 0.30) & (lum > 120) & (head | arms | feet)
gem = (g > r + 8) & (g > b + 5) & arms
# ผิว = บริเวณผิวตามตำแหน่ง (หัว แขนพ้นแขนเสื้อ เท้า หาง คอเสื้อ) ทั้งหมด ไม่ดูสี
# ยกเว้นกำไล อัญมณี ฟัน/เล็บ — ปื้นน้ำเงิน/ดำที่ฉายมาผิดบนหัวหรือหางจึงกลายเป็นผิวด้วย
skin = skin_rg & ~bracelet & ~ivory & ~gem


def hex_lab(h):
    h = h.lstrip("#")
    return to_lab(np.array([[[int(h[i:i + 2], 16) for i in (0, 2, 4)]]], np.float32))[0, 0]


mid = hex_lab(a.skin)                       # ส้ม KMITL
high = hex_lab(a.skin_high)                 # เหลือง KMITL สำหรับจุดสว่าง
low_c = mid.copy(); low_c[0] = mid[0] - 24  # ส้มเข้มโทนเดียวกันสำหรับร่องเกล็ด
lab = to_lab(out)
local, _ = norm_blur(lab[..., 0], skin, 24)
glob = lab[..., 0][skin].mean()
t = (lab[..., 0] - local) + 0.25 * (local - glob)       # ลายเกล็ด + เงาช่วงกว้างหนึ่งในสี่
t = np.clip(t / 22.0, -1.0, 1.0)[..., None]
ramp = np.where(t < 0, mid + (low_c - mid) * (-t), mid + (high - mid) * (t * a.highlight))
rec = from_lab(ramp)
soft = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 1.5)[..., None]
out = out * (1 - soft) + rec * soft
# ---- กำไล: ปรับเป็นเงินรมดำแบบในภาพถ่าย (ลดความอิ่มสี ยกความสว่าง เพิ่มคอนทราสต์ลาย)
labc = to_lab(out)
Lc = labc[..., 0][bracelet]
labc[..., 0] = np.where(bracelet, np.clip((labc[..., 0] - Lc.mean()) * 1.9 + 60, 12, 92), labc[..., 0])
labc[..., 1] = np.where(bracelet, labc[..., 1] * 0.2 + 1.0, labc[..., 1])
labc[..., 2] = np.where(bracelet, labc[..., 2] * 0.25 + 7.0, labc[..., 2])
out = np.where(bracelet[..., None], from_lab(labc), out)
print(f"  ผิว: โทน KMITL {a.skin} -> {a.skin_high} ({skin.sum():,} พิกเซล)")

Image.fromarray(out.astype(np.uint8)).save(a.out_png)
mat.baseColorTexture = Image.open(a.out_png)
m.export(a.out_glb)
print("เขียน", a.out_glb)
