r"""Visual hull จากคลิปกล้องวนรอบตัว — เฉือน voxel ด้วย silhouette หลายสิบมุม

ต่างจาก ``src/visual_hull.py`` อย่างไร
-------------------------------------
ตัวนั้นรับภาพ 4 มุมที่ตั้งฉากกัน แล้วปั้น *หน้าตัด* ของแต่ละแถวจากความกว้าง
(มุมหน้า) กับความลึก (มุมข้าง) โดยสมมติว่าหน้าตัดเป็นวงรี เพราะสองมุมไม่พอ
จะบอกรูปหน้าตัดได้จริง

ตัวนี้มี silhouette ทุก 3 องศารอบตัว จึงเฉือนตรง ๆ ได้ตามนิยาม visual hull:
voxel ไหนตกนอกเงาของมุมใดมุมหนึ่ง = ไม่มีเนื้อ  หน้าตัดที่ได้จึงเป็นรูปจริง
ของตัวละคร ไม่ใช่วงรีที่เราเดาให้

ปัญหาเฉพาะของคลิปนี้: ตัวล้นเฟรม
---------------------------------
ช่วงกลางคลิปกล้องเข้าใกล้จนขาหลุดขอบล่าง ถ้าเอา mask มาเฉือนตรง ๆ ขาจะถูก
ตัดทิ้งทั้งท่อน เพราะพิกเซลนอกเฟรมไม่ใช่ตัวละคร "ตามที่ mask บอก"

วิธีแก้คือแยก "นอกเงา" ออกจาก "นอกเฟรม" ให้ชัด
  * voxel ฉายตกในเฟรม และตกนอกเงา  -> นับเป็นหลักฐานว่าไม่มีเนื้อ
  * voxel ฉายตกนอกเฟรม             -> ไม่รู้ ไม่นับอะไรเลย
ขาจึงถูกเฉือนด้วยเฉพาะมุมที่เห็นขาจริง ๆ (ช่วงต้นกับท้ายคลิปที่กล้องถอยออก
ราว 100 องศารอบมุมหน้า) ซึ่งพอให้หน้าตัดของขาออกมาใช้ได้
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import PipelineConfig
from .turntable import Camera, ClipFrame, MaskStats
from .visual_hull import HullGrid

log = logging.getLogger(__name__)


@dataclass
class Bounds:
    """กล่องครอบตัวละครในหน่วยโลก (ตัวละครสูง 1.0 ฝ่าเท้าอยู่ที่ y = 0)"""

    radius: float      # รัศมีรอบแกนหมุน
    y_min: float
    y_max: float


def bounds_from(cams: list[Camera], stats: list[MaskStats],
                pad: float = 0.03) -> Bounds:
    """กล่องครอบ จากขอบ mask ของทุกเฟรมแปลงเป็นหน่วยโลก

    รัศมีคิดจากระยะแนวนอนที่ไกลจากแกนหมุนที่สุดเท่าที่เฟรมไหนเคยเห็น — ของที่
    ยื่นออกไปไกลสุดอย่างหางหรือมือจะโผล่ในบางมุมเท่านั้น จึงต้องเอา max ทั้งคลิป
    """
    r, y_lo, y_hi = 0.0, 0.0, 0.0
    for c, s in zip(cams, stats):
        r = max(r, abs(s.x0 - c.cx) / c.su, abs(s.x1 - c.cx) / c.su)
        y_hi = max(y_hi, (c.cy - s.y0) / c.sv)
        if not s.touch_bottom:
            y_lo = min(y_lo, (c.cy - s.y1) / c.sv)
    b = Bounds(radius=r + pad, y_min=y_lo - pad, y_max=y_hi + pad)
    log.info("กล่องครอบ: รัศมี %.3f, y %.3f .. %.3f (หน่วย = ความสูงตัวละคร)",
             b.radius, b.y_min, b.y_max)
    return b


# ---------------------------------------------------------------------------
# กริด
# ---------------------------------------------------------------------------


def _axes(bounds: Bounds, res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """พิกัดโลกที่จุดกึ่งกลางของ voxel แต่ละชั้น พร้อมขนาด voxel"""
    step = (bounds.y_max - bounds.y_min) / res
    ny = res
    nx = nz = int(np.ceil(2 * bounds.radius / step))
    ys = bounds.y_min + (np.arange(ny) + 0.5) * step
    xs = -bounds.radius + (np.arange(nx) + 0.5) * step
    zs = -bounds.radius + (np.arange(nz) + 0.5) * step
    return ys, xs, zs, step


def _accumulate(frames: list[ClipFrame], cams: list[Camera],
                ys: np.ndarray, xs: np.ndarray, zs: np.ndarray,
                ) -> tuple[np.ndarray, np.ndarray]:
    """นับว่าแต่ละ voxel ถูกกี่มุมบอกว่า "เห็นแล้วไม่มีเนื้อ" จากทั้งหมดกี่มุมที่เห็น

    คิดแยกแกนได้ เพราะการฉายของ turntable แยกส่วน: คอลัมน์ u ขึ้นกับ (x, z)
    เท่านั้น ส่วนแถว v ขึ้นกับ y เท่านั้น จึงคิด u เป็นตาราง (nx, nz) หนึ่งครั้ง
    กับ v เป็นเวกเตอร์ (ny,) หนึ่งครั้ง แล้วหยิบค่า mask ด้วย broadcast รวดเดียว
    """
    ny, nx, nz = len(ys), len(xs), len(zs)
    n_seen = np.zeros((ny, nx, nz), np.uint16)
    n_out = np.zeros((ny, nx, nz), np.uint16)

    gx = xs[:, None]
    gz = zs[None, :]
    slab = max(8, 2 ** 22 // max(1, nx * nz))   # แบ่งเป็นแผ่น ๆ กันหน่วยความจำบวม
    for cam, frame in zip(cams, frames):
        m = frame.mask
        h, w = m.shape
        c, s = np.cos(cam.theta), np.sin(cam.theta)
        g = 1.0 / np.maximum(1.0 - (gx * s + gz * c) * cam.q, 0.2)

        u = cam.su * (gx * c - gz * s) * g + cam.cx
        ui = np.rint(u).astype(np.int32)
        ok_u = (ui >= 0) & (ui < w)
        np.clip(ui, 0, w - 1, out=ui)

        for y0 in range(0, ny, slab):
            yy = ys[y0:y0 + slab]
            v = cam.cy - cam.sv * (cam.h + (yy[:, None, None] - cam.h) * g)
            vi = np.rint(v).astype(np.int32)
            ok_v = (vi >= 0) & (vi < h)
            np.clip(vi, 0, h - 1, out=vi)

            inside = m[vi, ui[None, :, :]]
            seen = ok_v & ok_u[None, :, :]
            n_seen[y0:y0 + slab] += seen
            n_out[y0:y0 + slab] += seen & ~inside
        frame.unload()
    return n_seen, n_out


def sdf_field(frames: list[ClipFrame], cams: list[Camera],
              ys: np.ndarray, xs: np.ndarray, zs: np.ndarray,
              drop: int = 1, min_views: int = 3,
              tol_views: int = 8, trust: list[bool] | None = None,
              solo_above: float | None = None) -> np.ndarray:
    """ระยะจากผิว hull ของแต่ละ voxel (บวก = อยู่ใน, หน่วย = ความสูงตัวละคร)

    ต่างจากการนับโหวตแบบ "ใน/นอก" ตรงที่เก็บ *ระยะ* ไว้ด้วย ซึ่งให้สองอย่าง

    1. ผิวเรียบระดับย่อยกว่า voxel — การนับโหวตด้วยมุมสิบกว่ามุมให้ค่าได้แค่
       สิบกว่าระดับ ผิวที่ marching cubes ดึงออกมาจึงเป็นขั้นบันไดตามจำนวนมุม
       สนามระยะเปลี่ยนค่าต่อเนื่อง ผิวจึงเรียบ
    2. ทนมุมที่เพี้ยนได้ — visual hull ตามตำราคือ min ของทุกมุม มุมเดียวที่
       มาตราส่วนผิดจะกัดเนื้อหายทั้งแถบ  เราใช้ค่าที่น้อยเป็นอันดับ (drop+1)
       แทน จึงยอมให้มุมเพี้ยนได้ ``drop`` มุมโดยไม่กระทบรูปทรง

    voxel ที่ฉายออกนอกเฟรมของมุมไหน มุมนั้นไม่ออกเสียงเลย (ไม่ใช่ "อยู่นอก")
    ขาที่หลุดขอบล่างในมุมข้างจึงไม่ถูกมุมข้างเฉือนทิ้ง

    ``trust`` + ``solo_above``: เหนือความสูง ``solo_above`` (หน่วย = ความสูง
    ตัวละคร) ให้เฉพาะมุมที่ ``trust`` เป็นจริงเท่านั้นที่ออกเสียง

    มีไว้เพื่อหัวโดยเฉพาะ ในคลิปที่กล้องเข้าใกล้ ตัวละครหลุดขอบบนของเฟรมบ่อย
    และความคลาดของมาตราส่วนรายเฟรมก็ไปโผล่แรงที่สุดตรงจุดที่ไกลจากพื้นที่สุด
    (วัดจากคลิปตัวอย่าง: ยอดหัวที่แต่ละเฟรมบอก กระจายกัน 6% ของความสูงตัว)
    ผลคือปากจระเข้ที่ยื่นไปข้างหน้าโดนมุมที่เพี้ยนเฉือนกลับเข้ามาจนเหลือกรวย
    ส่วนภาพนิ่งสี่มุมเห็นหัวเต็ม ๆ และสัดส่วนคงที่ จึงให้มันเป็นคนตัดสินแทน
    """
    ny, nx, nz = len(ys), len(xs), len(zs)
    BIG = np.float32(1e3)
    keep = [np.full((ny, nx, nz), BIG, np.float32) for _ in range(drop + 1)]
    seen = np.zeros((ny, nx, nz), np.uint8)

    gx = xs[:, None]
    gz = zs[None, :]
    slab = max(8, 2 ** 22 // max(1, nx * nz))
    if trust is None:
        trust = [True] * len(cams)
    n_mute = sum(1 for t in trust if not t)
    if solo_above is not None and n_mute:
        log.info("เหนือ %.2f ของความสูงตัว ให้เฉพาะ %d มุมที่เชื่อถือได้ออกเสียง "
                 "(ปิดเสียง %d มุม)", solo_above, len(cams) - n_mute, n_mute)

    for cam, frame, ok_high in zip(cams, frames, trust):
        m = frame.mask
        h, w = m.shape
        inside = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        outside = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
        signed = ((inside - outside) / cam.scale).astype(np.float32)

        c, s = np.cos(cam.theta), np.sin(cam.theta)
        g = 1.0 / np.maximum(1.0 - (gx * s + gz * c) * cam.q, 0.2)
        u = cam.su * (gx * c - gz * s) * g + cam.cx
        ui = np.rint(u).astype(np.int32)
        ok_u = (ui >= 0) & (ui < w)
        np.clip(ui, 0, w - 1, out=ui)

        for y0 in range(0, ny, slab):
            yy = ys[y0:y0 + slab]
            v = cam.cy - cam.sv * (cam.h + (yy[:, None, None] - cam.h) * g)
            vi = np.rint(v).astype(np.int32)
            ok = (vi >= 0) & (vi < h) & ok_u[None, :, :]
            if solo_above is not None and not ok_high:
                ok &= (yy[:, None, None] <= solo_above)
            np.clip(vi, 0, h - 1, out=vi)

            d = np.where(ok, signed[vi, ui[None, :, :]], BIG)
            sl = slice(y0, y0 + slab)
            seen[sl] += ok
            # แทรกค่าใหม่ลงในรายการ "ค่าที่น้อยที่สุด drop+1 ตัว"
            for k in range(drop + 1):
                lo = np.minimum(keep[k][sl], d)
                d = np.maximum(keep[k][sl], d)
                keep[k][sl] = lo
        frame.unload()

    # จะยอมให้กี่มุมค้านได้ ต้องขึ้นกับว่า voxel นั้นมีมุมเห็นกี่มุม ไม่ใช่
    # ตัวเลขตายตัว
    #
    # ตรงลำตัวมีครบทุกมุมเห็น การยอมให้ค้านสองสามมุมจึงแค่ชดเชยสัดส่วนตัวที่
    # เพี้ยนไประหว่างคลิป  แต่ใต้เข่าลงไปเหลือมุมเห็นแค่ไม่กี่มุม (มุมอื่น
    # ขาหลุดขอบล่างไปแล้ว) การยอมให้ค้านได้เท่ากันตรงนั้นเท่ากับเชื่อมุมเดียว
    # แล้วปลายนิ้วเท้าจะยืดเป็นลิ่มแบน ๆ ไปข้างหน้า เพราะไม่มีมุมไหนเหลือ
    # มาบอกว่าเท้ายาวแค่ไหน
    field = keep[0]
    for k in range(1, drop + 1):
        enough = seen >= tol_views * k
        if enough.any():
            field = np.where(enough, keep[k], field)

    # มุมเดียวหรือสองมุมยืนยันไม่พอจะบอกว่ามีเนื้อตรงนั้นจริง — ไม่งั้นพื้นที่
    # ที่เกือบทุกมุมมองไม่เห็น (ใต้ฝ่าเท้า นอกขอบล่างของเฟรมระยะใกล้) จะกลาย
    # เป็นแผ่นเนื้อแบน ๆ ปูอยู่กับพื้น
    low = seen < max(min_views, 1)
    if low.any():
        log.info("ตัด %d voxel (%.1f%%) ที่มีมุมเห็นไม่ถึง %d มุม",
                 int(low.sum()), 100 * float(low.mean()), max(min_views, 1))
    return np.where(low, -BIG, field)


def _to_hull_grid(occ: np.ndarray, bounds: Bounds, step: float,
                  cfg: PipelineConfig) -> HullGrid:
    """ห่อ occupancy ให้เป็น HullGrid ที่ขั้นหลัง ๆ ของ pipeline หลักใช้ต่อได้

    ขั้น texture กับ landmarks คิดพิกัดผ่าน ``voxel_to_canvas`` เราจึงต้อง
    นิยาม canvas สมมติขึ้นมาหนึ่งใบ: จัตุรัสด้าน cfg.canvas ที่ตัวละครยืนเต็ม
    ความสูงโดยเว้นขอบตาม cfg.margin — เหมือนที่ ``preprocess.normalize_view``
    ทำกับภาพนิ่งทุกมุม เพื่อให้สองเส้นทางคิดพิกัดเป็นภาษาเดียวกัน
    """
    S = cfg.canvas
    k = S * (1.0 - 2.0 * cfg.margin)        # พิกเซลต่อหนึ่งหน่วยโลก
    ground_row = S * (1.0 - cfg.margin)
    origin = (ground_row - bounds.y_min * k,     # แถวของ iy = 0
              S / 2 - bounds.radius * k,         # คอลัมน์ของ ix = 0
              S / 2 - bounds.radius * k)
    return HullGrid(occ.astype(np.float32), voxel_px=step * k,
                    origin_px=origin, canvas=S)


# ---------------------------------------------------------------------------
# เฉือน
# ---------------------------------------------------------------------------


def vote_field(frames: list[ClipFrame], cams: list[Camera],
               ys: np.ndarray, xs: np.ndarray, zs: np.ndarray,
               min_views: int = 6) -> np.ndarray:
    """สัดส่วนของมุมที่ "เห็นแล้วบอกว่ามีเนื้อ" ของแต่ละ voxel (0..1)

    visual hull ตามตำราคือ "ทุกมุมต้องเห็นว่ามีเนื้อ" ซึ่งใช้กับคลิปนี้ไม่ได้
    เพราะตัวละครในคลิปที่ AI สร้างไม่ได้นิ่งสนิท: หางแกว่ง แขนขยับเข้าออก
    สัดส่วนตัวค่อย ๆ เพี้ยนระหว่างต้นคลิปกับท้ายคลิป (เทียบเฟรมมุมเดียวกัน
    ที่ห่างกันหนึ่งรอบได้ IoU แค่ 0.68)  การบังคับให้ทุกมุมเห็นตรงกันจึงเหลือ
    แต่แกนกลางลำตัว แขนกับหางหายหมด

    เก็บเป็น "คะแนนโหวต" แทน แล้วให้ขั้นถัดไปเลือกเส้นแบ่งเอง = ได้รูปทรง
    ที่มุมส่วนใหญ่เห็นตรงกัน ซึ่งเป็นค่ากลางของตัวละครตลอดคลิป
    """
    n_seen, n_out = _accumulate(frames, cams, ys, xs, zs)
    with np.errstate(invalid="ignore", divide="ignore"):
        vote = np.where(n_seen >= min_views,
                        1.0 - n_out / np.maximum(n_seen, 1), 0.0)
    blind = int((n_seen < min_views).sum())
    if blind:
        log.info("มี %d voxel (%.1f%%) ที่แทบไม่มีมุมไหนเห็น — ถือว่าว่าง",
                 blind, 100 * blind / vote.size)
    return vote.astype(np.float32)


def carve_sdf(frames: list[ClipFrame], cams: list[Camera], stats: list[MaskStats],
              cfg: PipelineConfig, res: int, drop: int = 2, min_views: int = 3,
              smooth: float = 2.0, open_iters: int = 2, tol_views: int = 8,
              bounds: Bounds | None = None, trust: list[bool] | None = None,
              solo_above: float | None = None
              ) -> tuple[HullGrid, np.ndarray, tuple]:
    """เฉือน voxel จากหลายมุมด้วยสนามระยะ แล้วคืนกริดที่พร้อมเข้าขั้น meshing

    ขั้นตอน
      1. สนามระยะจากทุกมุม (``sdf_field``) — ทนมุมเพี้ยนได้ ``drop`` มุม
      2. เกลาสนามเล็กน้อย ลดรอยต่อระหว่างมุมที่เห็นไม่ตรงกัน
      3. เปิด (erode แล้ว dilate) เพื่อลบแผ่นบาง ๆ ที่ไม่ใช่เนื้อตัวละคร
         — เงาบนพื้นกับขอบล่างของเฟรมระยะใกล้ทิ้งแผ่นแบนปูพื้นไว้เสมอ
         ส่วนแขนขาหนาหลายสิบ voxel จึงรอดจากขั้นนี้
      4. เก็บเฉพาะก้อนใหญ่สุด
    """
    from scipy import ndimage

    bounds = bounds or bounds_from(cams, stats)
    ys, xs, zs, step = _axes(bounds, res)
    log.info("กริด %d x %d x %d voxel (%.1f M) จาก %d มุม",
             len(ys), len(xs), len(zs),
             len(ys) * len(xs) * len(zs) / 1e6, len(cams))

    field = sdf_field(frames, cams, ys, xs, zs, drop=drop,
                      min_views=min_views, tol_views=tol_views,
                      trust=trust, solo_above=solo_above)
    grid = assemble(field, bounds, step, cfg, smooth=smooth,
                    open_iters=open_iters)
    return grid, field, (ys, xs, zs)


def assemble(field: np.ndarray, bounds: Bounds, step: float,
             cfg: PipelineConfig, smooth: float = 2.0, open_iters: int = 2,
             protect: np.ndarray | None = None) -> HullGrid:
    """แปลงสนามระยะเป็นกริด occupancy พร้อมเก็บกวาด

    ``protect`` คือเขตที่ห้ามกัดและห้ามเกลา — นิ้วมือที่ความละเอียด 320 หนา
    แค่สาม voxel ส่วน opening ลบทุกอย่างที่บางกว่าสี่ และการเกลาด้วย sigma
    สอง voxel ก็ถมร่องระหว่างนิ้วจนเรียบ สองขั้นนั้นจำเป็นกับลำตัวแต่เป็น
    ตัวทำลายมือกับเท้า จึงต้องแยกกฎตามส่วน ไม่ใช่ตั้งค่าเดียวทั้งตัว
    """
    from scipy import ndimage

    if smooth > 0:
        blurred = ndimage.gaussian_filter(field, smooth)
        field = np.where(protect, field, blurred) if protect is not None             else blurred

    solid = field > 0
    if open_iters > 0:
        core = ndimage.binary_erosion(solid, iterations=open_iters)
        core = _largest_blob(core)
        kept = solid & ndimage.binary_dilation(core, iterations=open_iters + 1)
        if protect is not None:
            # ในเขตที่กันไว้ยังต้องกวาดอยู่ แค่เบามือลง: ไม่กวาดเลยจะเหลือ
            # สะพานหนาหนึ่ง voxel ซึ่งกลายเป็นขอบ non-manifold ตอน marching
            # cubes แล้วทำให้ bone heat weighting ของ Blender แก้ไม่ออก
            # (อาการคือได้ไฟล์ที่ไม่มี skin เลยโดยไม่มี error อะไรแจ้ง)
            soft = ndimage.binary_erosion(solid, iterations=1)
            soft = solid & ndimage.binary_dilation(_largest_blob(soft),
                                                   iterations=2)
            kept = kept | (soft & protect)
        solid = kept
    solid = _largest_blob(solid)

    band = 2.5 * step
    occ = np.clip(0.5 + field / (2.0 * band), 0.0, 1.0).astype(np.float32)
    occ = np.where(ndimage.binary_dilation(solid, iterations=2), occ, 0.0)

    log.info("เฉือนเสร็จ: มีเนื้อ %.2f%% ของกริด (%.2f M voxel)",
             100 * solid.mean(), solid.sum() / 1e6)
    return _to_hull_grid(occ, bounds, step, cfg)


def _largest_blob(solid: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, n = ndimage.label(solid)
    if n <= 1:
        return solid
    sizes = np.bincount(labels.ravel())[1:]
    keep = 1 + int(np.argmax(sizes))
    dropped = sizes.sum() - sizes.max()
    log.info("hull แตกเป็น %d ก้อน — เก็บก้อนใหญ่สุด ทิ้งเศษ %d voxel (%.2f%%)",
             n, int(dropped), 100 * dropped / max(1, sizes.sum()))
    return labels == keep


# ---------------------------------------------------------------------------
# ตรวจความสอดคล้อง: ฉายกลับไปเทียบกับเงาต้นฉบับ
# ---------------------------------------------------------------------------


def project_silhouette(solid: np.ndarray, cam: Camera, ys: np.ndarray,
                       xs: np.ndarray, zs: np.ndarray,
                       shape: tuple[int, int], down: int = 4) -> np.ndarray:
    """ฉายก้อน voxel ลงบนภาพของมุมหนึ่ง คืนภาพขาวดำที่ย่อแล้ว"""
    h, w = shape
    iy, ix, iz = np.nonzero(solid)
    c, s = np.cos(cam.theta), np.sin(cam.theta)
    g = 1.0 / np.maximum(1.0 - (xs[ix] * s + zs[iz] * c) * cam.q, 0.2)
    u = cam.su * (xs[ix] * c - zs[iz] * s) * g + cam.cx
    v = cam.cy - cam.sv * (cam.h + (ys[iy] - cam.h) * g)
    ui = np.rint(u / down).astype(np.int32)
    vi = np.rint(v / down).astype(np.int32)
    hh, ww = h // down, w // down
    ok = (ui >= 0) & (ui < ww) & (vi >= 0) & (vi < hh)
    img = np.zeros((hh, ww), bool)
    img[vi[ok], ui[ok]] = True

    # ภาพที่ได้เป็นจุดศูนย์กลางของ voxel กระจายห่างกันตามขนาด voxel
    # ต้องถมให้เต็มรอยเท้าของ voxel ก่อน ไม่งั้นเงาที่ฉายได้จะพรุนเป็นตะแกรง
    # แล้ว IoU ตกทั้งที่รูปทรงถูกต้อง
    step = float(ys[1] - ys[0]) if len(ys) > 1 else 0.01
    k = max(3, int(np.ceil(cam.sv * step / down)) | 1)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((k, k), np.uint8)).astype(bool)


def consistency(frames: list[ClipFrame], cams: list[Camera],
                stats: list[MaskStats], res: int, vote: float) -> float:
    """คะแนนความสอดคล้องของเงาทั้งคลิปกับ hull ที่เฉือนได้จากเงาเหล่านั้นเอง

    เป็นเกณฑ์กลางสำหรับ "พารามิเตอร์ชุดนี้ดีแค่ไหน": ถ้ามุมหรือระยะผิด เงาที่
    ฉายกลับจาก hull จะผอมกว่า mask จริง IoU จึงตก — ไม่ต้องมีเฉลยจากที่ไหน
    """
    bounds = bounds_from(cams, stats)
    ys, xs, zs, _ = _axes(bounds, res)
    field = vote_field(frames, cams, ys, xs, zs, min_views=4)
    solid = field >= vote
    if solid.sum() < 500:
        return 0.0
    solid = _largest_blob(solid)
    return float(silhouette_scores(solid, frames, cams, ys, xs, zs).mean())


def fit_angles(frames: list[ClipFrame], sol, cfg: PipelineConfig | None = None,
               res: int = 96, vote: float = 0.85, sub: int = 3,
               rounds: int = 2, span: float = 0.22, steps: int = 7):
    """ดัดจังหวะการวนของกล้องให้ตรงกับคลิป

    กล้องในคลิปตัวอย่างวนช้าช่วงต้น (1.0 องศา/เฟรม) แล้วเร่งเป็นเกือบ
    3 องศา/เฟรมช่วงกลาง สมมติว่าความเร็วคงที่จึงทำให้มุมของเฟรมกลางคลิป
    เพี้ยนไปกว่า 40 องศา — มากพอจะทำให้ "มุมข้าง" ที่เลือกได้กลายเป็นมุมเฉียง
    และทำให้ hull ที่เฉือนได้พังทั้งใบ

    วิธีหา: เลื่อนจุดควบคุมทีละจุด (ทุก ๆ 90 องศา) แล้ววัดว่าเงาจากทุกเฟรม
    สอดคล้องกันขึ้นหรือลง — ไม่ต้องรู้เฉลย ไม่ต้องให้คนมาชี้ว่าเฟรมไหนคือมุมข้าง

    คืน Solution ตัวใหม่ที่มุมถูกดัดแล้ว
    """
    from .turntable import build_cameras

    F = frames[::sub]
    S = sol.stats[::sub]
    scale = np.array([c.scale for c in sol.cams])
    cx = np.array([c.cx for c in sol.cams])
    cy = np.array([c.cy for c in sol.cams])
    full = [c.full_body for c in sol.cams]

    def cams_for(angles):
        out = build_cameras(sol.stats, scale, cy, full, angles, sol.sign,
                            cx_smooth=0.0)
        for c, x in zip(out, cx):      # cx ถูกเกลามาแล้วรอบแรก อย่าเกลาซ้ำ
            c.cx = float(x)
        return out

    angles = sol.angles
    best = consistency(F, cams_for(angles)[::sub], S, res, vote)
    log.info("ดัดจังหวะการวน: เริ่มที่ IoU %.3f — %s", best, angles.describe())

    def sweep(am, width_frac, n_rounds):
        nonlocal best
        n = len(am.knots_i)
        for _ in range(n_rounds):
            improved = False
            for k in range(1, n - 1):    # จุดหัวท้ายคือมุมหน้า ตรึงไว้
                lo, hi = am.knots_i[k - 1], am.knots_i[k + 1]
                width = width_frac * (hi - lo)
                for cand in am.knots_i[k] + np.linspace(-width, width, steps):
                    if not (lo + 2 < cand < hi - 2):
                        continue
                    trial = am.moved(k, float(cand))
                    score = consistency(F, cams_for(trial)[::sub], S, res, vote)
                    if score > best + 1e-4:
                        best, am, improved = score, trial, True
            if not improved:
                break
        return am

    angles = sweep(angles, span, rounds)
    # รอบละเอียด: เพิ่มจุดควบคุมเป็นสองเท่าแล้วขยับทีละน้อย เพื่อจับจังหวะ
    # ที่เปลี่ยนเร็วกว่าช่วงละ 90 องศา
    fine = len(angles.knots_i) * 2 - 1
    angles = sweep(angles.resample(fine), span * 0.6, 1)
    log.info("ดัดจังหวะการวนเสร็จ: IoU %.3f — %s", best, angles.describe())

    sol.angles = angles
    sol.cams = cams_for(angles)
    return sol


def fit_lens(frames: list[ClipFrame], cams: list[Camera], stats: list[MaskStats],
             res: int = 96, vote: float = 0.85,
             focals: tuple[float, ...] = (700, 900, 1150, 1500, 2200, 4000, 0.0),
             heights: tuple[float, ...] = (0.35, 0.5, 0.65),
             ) -> tuple[list[Camera], float, float]:
    """หาค่าเลนส์ (ระยะกล้อง + ความสูงสายตา) ที่ทำให้เงาทุกมุมสอดคล้องกันที่สุด

    คลิปไม่ได้บอกสเปกกล้องมา แต่ค่าที่ถูกต้องมีลายเซ็นชัดเจน: ถ้าระยะกล้อง
    ผิด เงาที่ฉายจาก hull จะไม่ทับ mask โดยเฉพาะส่วนที่อยู่ไกลจากกึ่งกลางภาพ
    (หัวกับเท้า) จึงหาได้ด้วยการลองแล้ววัด IoU

    ``focal`` เป็นความยาวโฟกัสหน่วยพิกเซล ระยะกล้องของแต่ละเฟรมคิดจาก
    focal / มาตราส่วนของเฟรมนั้น — เลนส์ตัวเดียวกันทั้งคลิป แต่กล้องเข้าใกล้
    ออกห่างได้ ซึ่งตรงกับที่เห็นในคลิป  0 = ภาพฉายขนาน (กล้องอยู่ไกลอนันต์)
    """
    sub = max(1, len(frames) // 45)
    F = frames[::sub]
    S = stats[::sub]
    best = (-1.0, 0.0, 0.5)
    for focal in focals:
        for h in heights:
            C = [c.with_lens(focal, h) for c in cams[::sub]]
            bounds = bounds_from(C, S)
            ys, xs, zs, _ = _axes(bounds, res)
            field = vote_field(F, C, ys, xs, zs, min_views=4)
            solid = field >= vote
            if solid.sum() < 500:
                continue
            solid = _largest_blob(solid)
            score = float(silhouette_scores(solid, F, C, ys, xs, zs).mean())
            log.info("  ลองเลนส์ focal=%-6s สายตาสูง %.2f -> IoU %.3f",
                     "ขนาน" if focal <= 0 else f"{focal:.0f}", h, score)
            if score > best[0]:
                best = (score, focal, h)
    score, focal, h = best[0], best[1], best[2]
    log.info("เลือกเลนส์: %s, แนวสายตาสูง %.2f ของความสูงตัว (IoU %.3f)",
             "ภาพฉายขนาน" if focal <= 0 else
             f"focal {focal:.0f} px (กล้องห่างราว {focal / np.median([c.scale for c in cams]):.1f} เท่าความสูงตัว)",
             h, score)
    return [c.with_lens(focal, h) for c in cams], float(focal), float(h)


def _best_fit(proj: np.ndarray, mask: np.ndarray, cx: float, cy: float,
              down: int, shifts: range, scales: np.ndarray,
              ) -> tuple[float, float, float, float, float]:
    """ขยับ/ยืดเงาที่ฉายจาก hull ให้ทับ mask ให้มากที่สุด คืน (IoU, dx, dy, ku, kv)

    แทนที่จะฉาย hull ใหม่ทุกครั้งที่ลองค่าใหม่ (แพง) ใช้วิธีแปลง *ภาพเงา* แทน
    ซึ่งให้ผลเหมือนกันเป๊ะ เพราะการเลื่อน cx กับการคูณมาตราส่วนรอบจุด (cx, cy)
    เป็น affine บนระนาบภาพอยู่แล้ว
    """
    m = mask
    pc, pr = cx / down, cy / down
    h, w = proj.shape

    def iou_of(img: np.ndarray) -> float:
        union = np.count_nonzero(img | m)
        return np.count_nonzero(img & m) / union if union else 0.0

    def warped(ku: float, kv: float) -> np.ndarray:
        if ku == 1.0 and kv == 1.0:
            return proj
        M = np.float32([[ku, 0, pc - pc * ku], [0, kv, pr - pr * kv]])
        return cv2.warpAffine(proj.astype(np.uint8), M, (w, h)) > 0

    # ไล่ทีละแกน (coordinate descent) แทนการค้นตารางสี่มิติ: ตารางเต็มต้อง
    # ประเมินหลักพันครั้งต่อเฟรม ซึ่งช้าเกินกว่าจะวนหลายรอบได้
    ku = kv = 1.0
    dx = dy = 0
    best = iou_of(proj)
    for _ in range(2):
        base = warped(ku, kv)
        for ddy in shifts:
            rolled = np.roll(base, ddy, axis=0)
            for ddx in shifts:
                score = iou_of(np.roll(rolled, ddx, axis=1))
                if score > best:
                    best, dx, dy = score, ddx, ddy
        # ลำดับต้องเป็น "ยืดรอบ (cx, cy) ก่อน แล้วค่อยเลื่อน" ให้ตรงกับที่
        # กล้องตัวใหม่จะฉายออกมาจริง ๆ (ดูสมการในคลาส Camera)
        for a in scales:
            for b in scales:
                cand = np.roll(np.roll(warped(float(a), float(b)), dy, axis=0),
                               dx, axis=1)
                score = iou_of(cand)
                if score > best:
                    best, ku, kv = score, float(a), float(b)
    return best, float(dx), float(dy), ku, kv


def refine(frames: list[ClipFrame], cams: list[Camera], solid: np.ndarray,
           ys: np.ndarray, xs: np.ndarray, zs: np.ndarray,
           down: int = 4, max_shift: int = 5, max_scale: float = 0.08,
           ) -> tuple[list[Camera], np.ndarray]:
    """ปรับพารามิเตอร์ของแต่ละเฟรมให้เงาที่ฉายจาก hull ทับ mask ได้ดีที่สุด

    ทำไมต้องมีขั้นนี้: ตัวละครในคลิปที่ AI สร้างไม่ได้มีสัดส่วนคงที่ตลอดคลิป
    เฟรมท้ายคลิปตัวกว้างกว่าเฟรมต้นคลิปราว 12% ซึ่งมากเกินกว่าจะมองข้าม —
    การเฉือนด้วยเงาที่ขนาดไม่ตรงกันกินเนื้อแขนขาหายไปทั้งท่อน

    สิ่งที่แก้ให้คือ "มาตราส่วนและตำแหน่งของแต่ละเฟรม" ไม่ใช่มุมกล้อง เพราะ
    มุมมาจากคาบการวนซึ่งวัดได้แม่นอยู่แล้ว ตรึงค่าเฉลี่ยของตัวคูณไว้ที่ 1
    เพื่อไม่ให้ทั้งระบบค่อย ๆ หดลงทุกรอบที่ refine
    """
    scales = np.round(np.arange(1.0 - max_scale, 1.0 + max_scale + 1e-9,
                                max_scale / 2), 4)
    shifts = range(-max_shift, max_shift + 1)
    out, scores = [], []
    for cam, frame in zip(cams, frames):
        m = frame.mask
        proj = project_silhouette(solid, cam, ys, xs, zs, m.shape, down)
        small = cv2.resize(m.astype(np.uint8),
                           (proj.shape[1], proj.shape[0])) > 0
        score, dx, dy, ku, kv = _best_fit(proj, small, cam.cx, cam.cy,
                                          down, shifts, scales)
        out.append(Camera(cam.index, cam.theta, cam.scale,
                          cam.cx + dx * down, cam.cy + dy * down,
                          cam.full_body, cam.ku * ku, cam.kv * kv,
                          q=cam.q, h=cam.h))
        scores.append(score)
        frame.unload()

    # ตรึงค่ากลาง: ยอมให้แต่ละเฟรม *ต่างจากเพื่อน* ได้ แต่ห้ามทั้งกลุ่มเลื่อน
    ku_mid = float(np.exp(np.mean([np.log(c.ku) for c in out])))
    kv_mid = float(np.exp(np.mean([np.log(c.kv) for c in out])))
    dy_mid = float(np.mean([c.cy - o.cy for c, o in zip(out, cams)]))
    for c in out:
        c.ku /= ku_mid
        c.kv /= kv_mid
        c.cy -= dy_mid
    log.info("refine: IoU เฉลี่ย %.3f (ต่ำสุด %.3f) — "
             "ตัวคูณแนวนอน %.3f..%.3f แนวตั้ง %.3f..%.3f",
             float(np.mean(scores)), float(np.min(scores)),
             min(c.ku for c in out), max(c.ku for c in out),
             min(c.kv for c in out), max(c.kv for c in out))
    return out, np.asarray(scores)


def silhouette_scores(solid: np.ndarray, frames: list[ClipFrame],
                      cams: list[Camera], ys: np.ndarray, xs: np.ndarray,
                      zs: np.ndarray, down: int = 4) -> np.ndarray:
    """IoU ระหว่างเงาที่ฉายจาก hull กับ mask จริง ของทุกเฟรม

    ตัวเลขนี้คือคำตอบของคำถาม "พารามิเตอร์กล้องที่ประเมินมาถูกไหม"
    ถ้ามุมหรือมาตราส่วนเพี้ยน hull จะถูกเฉือนจนผอมกว่าเงาจริง IoU จะตก
    """
    out = np.zeros(len(cams))
    for i, (cam, frame) in enumerate(zip(cams, frames)):
        m = frame.mask
        proj = project_silhouette(solid, cam, ys, xs, zs, m.shape, down)
        small = cv2.resize(m.astype(np.uint8), (proj.shape[1], proj.shape[0]),
                           interpolation=cv2.INTER_AREA) > 0
        # เทียบเฉพาะในกรอบภาพ — ส่วนที่ล้นเฟรมไม่มีใครรู้ว่าควรมีหรือไม่มี
        inter = np.count_nonzero(proj & small)
        union = np.count_nonzero(proj | small)
        out[i] = inter / union if union else 0.0
        frame.unload()
    return out
