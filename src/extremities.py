"""หามือกับเท้าใน silhouette แล้วกันไม่ให้ขั้นตอนเกลาผิวกลืนนิ้วหายไป

ทำไมต้องมีโมดูลนี้แยกออกมา
---------------------------
ทุกขั้นตอนที่ทำให้ "ลำตัวดูดี" ล้วนทำลายมือกับเท้าทั้งหมด:

    spike_open        ลบส่วนยื่นที่บางกว่า 7 voxel  -> นิ้วกับเล็บหายเกลี้ยง
    silhouette_smooth เบลอขอบตามแนวตั้ง             -> ร่องระหว่างนิ้วถูกถม
    depth_falloff     ชิ้นแคบได้ความลึกตามลำตัว     -> นิ้วกลายเป็นครีบแบน
    hull_blur 2.6     เบลอ occupancy ก่อน marching  -> อะไรที่เล็กกว่า ~5
                                                       voxel ละลายหายไป

สี่ข้อนี้ถูกต้องสำหรับลำตัว หัว และหาง ซึ่งเป็นก้อนใหญ่ที่ต้องการผิวเรียบ
แต่ผิดหมดสำหรับมือกับเท้า ซึ่ง *ทั้งชิ้น* มีขนาดเท่ากับรายละเอียดที่ขั้นตอน
พวกนั้นตั้งใจจะลบ ทางแก้ไม่ใช่การปิดขั้นตอนเหล่านั้น (ลำตัวจะกลับไปเป็น
ขั้นบันได) แต่คือ *จำกัดขอบเขต* ของมัน: หาว่ามือกับเท้าอยู่ตรงไหน แล้วใช้
กฎคนละชุดเฉพาะในเขตนั้น

เขตที่หาได้จะถูกใช้ใน 3 ที่
  1. `visual_hull._remove_spikes`  — คืน silhouette ดิบในเขตนี้
  2. `visual_hull._fill_elliptical` — นิ้วได้หน้าตัด *กลม* ไม่ใช่แผ่นแบน
  3. `visual_hull.carve`            — เบลอด้วย sigma ที่เล็กกว่าในเขตนี้
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import PipelineConfig

log = logging.getLogger(__name__)


@dataclass
class Extremities:
    """เขตของมือและเท้าในพิกัด voxel slab (แถว x แกน X)"""

    foot_rows: np.ndarray      # bool[ny]      แถวที่เป็นเท้า
    hand_rows: np.ndarray      # bool[ny]      แถวที่เป็นมือ
    zone: np.ndarray           # bool[ny, nx]  พิกเซลที่อยู่ในมือหรือเท้า
    z_push: np.ndarray         # float[ny]     ดันนิ้วไปข้างหน้ากี่ส่วนของช่วงลึก
    notes: list[str]
    y_ankle: int = -1          # แถวบนสุดของเท้า
    toe_gaps: tuple = ()       # ((x_lo, x_hi, x_ball), ...) ร่องระหว่างนิ้วเท้า
                               # x_ball = แกนกลางของเท้าข้างที่ร่องนั้นอยู่
    foot_spans: tuple = ()     # ((x_lo, x_hi), ...) ช่วงแกน X ของเท้าแต่ละข้าง

    @property
    def any(self) -> bool:
        return bool(self.zone.any())

    def row_band(self, pad: int = 0) -> np.ndarray:
        """แถวทั้งหมดที่อยู่ในเขต เผื่อขอบ `pad` แถว"""
        rows = self.foot_rows | self.hand_rows
        if pad > 0:
            rows = ndimage.binary_dilation(rows, np.ones(2 * pad + 1, bool))
        return rows


def _runs(row: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[brk + 1]))
    ends = np.concatenate((idx[brk], [idx[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_len]


# ---------------------------------------------------------------------------
# เท้า
# ---------------------------------------------------------------------------


def _find_ankle(fb: np.ndarray, cfg: PipelineConfig) -> int:
    """แถวที่เท้าจบและข้อเท้าเริ่ม

    รูปร่างตามแนวตั้งของขาท่อนล่างมีจังหวะชัดเจน: จากพื้นขึ้นมา เท้าผายออก
    จนกว้างสุดราวกลางฝ่าเท้า แล้วแคบลงอย่างรวดเร็วจนถึงข้อเท้า จากนั้น
    *หยุดแคบ* แล้วค่อย ๆ กว้างขึ้นไปเป็นน่อง จุดที่หยุดแคบคือข้อเท้า

    หาแบบนี้แทนการใช้สัดส่วนตายตัว เพราะตัวละครแต่ละตัวเท้าสั้นยาวไม่เท่ากัน
    (ตัวที่เดินด้วยปลายเท้าแบบสัตว์ เท้าจะสูงกว่าคนมาก)
    """
    ny = fb.shape[0]
    width = fb.sum(axis=1).astype(np.float64)
    width = ndimage.uniform_filter1d(width, 3, mode="nearest")

    lo = max(1, int(0.015 * ny))
    hi = max(lo + 4, int(0.18 * ny))
    y_peak = lo + int(np.argmax(width[lo:hi]))

    y = y_peak
    while y + 1 < hi and width[y + 1] < width[y] - 1e-9:
        y += 1
    return int(np.clip(y, int(0.025 * ny), hi))


def find_toe_gaps(
    front: np.ndarray, y_ankle: int, cfg: PipelineConfig
) -> list[tuple[float, float]]:
    """หาร่องระหว่างนิ้วเท้าจาก silhouette ของ *มุมหน้าอย่างเดียว*

    ทำไมมุมหน้าอย่างเดียว ไม่ใช่ fb ที่ตัดกับมุมหลังแล้ว: นิ้วเท้าชี้ไป
    ข้างหน้า มุมหน้าจึงเห็นร่องระหว่างนิ้วเต็ม ๆ ส่วนมุมหลังเห็นแต่ส้นเท้า
    การ AND สองมุมจึงลบร่องทิ้งเกือบหมดก่อนที่เราจะได้เห็นมัน

    นับแบบโหวต: แต่ละคอลัมน์ x ได้คะแนนเมื่อ "อยู่ในช่วงของเท้าแต่ไม่มีเนื้อ"
    ในแถวล่าง ๆ แถวเดียวเชื่อไม่ได้เลยเพราะขอบ mask สั่นและเล็บแต่ละนิ้วสูง
    ไม่เท่ากัน แต่พอรวมหลายแถวแล้วร่องจริงจะโผล่ขึ้นมาชัด
    """
    ny, nx = front.shape
    if y_ankle < 4:
        return []
    y_hi = max(3, int(cfg.toe_split_band * y_ankle))

    # ช่วงของเท้าแต่ละข้าง: ยูเนียนของทุกแถวในแถบนิ้ว
    #
    # ห้ามอ่านจากแถวเดียวที่ข้อเท้า เพราะนิ้วก้อยกับเล็บที่กางออกอยู่ *นอก*
    # ช่วงของข้อเท้าไปหลาย voxel แล้วร่องของนิ้วฝั่งนั้นจะถูกตัดทิ้งหมด
    # (วัดได้ว่าเท้าซ้ายทั้งข้างหลุดการตรวจด้วยเหตุนี้) ยูเนียนของทุกแถว
    # ปิดร่องระหว่างนิ้วไปในตัว จึงได้ขอบเขตเท้าทั้งข้างเป็นก้อนเดียวพอดี
    union = front[1 : y_hi + 1].any(axis=0)
    feet = [r for r in _runs(union, max(2, nx // 110))
            if r[1] - r[0] >= nx // 40]
    if not feet:
        return []

    votes = np.zeros(nx, np.int32)
    rows = 0
    for y in range(1, y_hi + 1):
        rr = _runs(front[y], max(2, nx // 200))
        if len(rr) < 2:
            continue
        rows += 1
        for lo, hi in feet:
            inside = [r for r in rr if r[0] >= lo - 2 and r[1] <= hi + 2]
            if len(inside) < 2:
                continue
            for a, b in zip(inside[:-1], inside[1:]):
                votes[a[1] + 1 : b[0]] += 1

    if rows == 0 or votes.max() == 0:
        return []

    keep = votes >= max(2, int(cfg.toe_gap_votes * rows))
    gaps: list[tuple[float, float, float]] = []
    for lo, hi in _runs(keep, 1):
        # ร่องที่กว้างเกินครึ่งหนึ่งของเท้าไม่ใช่ร่องนิ้ว แต่คือช่องระหว่างเท้าสองข้าง
        width = hi - lo + 1
        host = next((f for f in feet if lo >= f[0] and hi <= f[1]), None)
        if host is not None and width <= 0.45 * (host[1] - host[0]):
            gaps.append((float(lo), float(hi), 0.5 * (host[0] + host[1])))

    log.info("ร่องระหว่างนิ้วเท้า %d ร่อง จาก %d แถวล่างสุด: %s",
             len(gaps), rows, ", ".join(f"{a:.0f}-{b:.0f}" for a, b, _ in gaps))
    return gaps


def carve_toe_grooves(
    occ: np.ndarray, ext: "Extremities", cz: np.ndarray, rz: np.ndarray,
    valid: np.ndarray, cfg: PipelineConfig,
) -> int:
    """ลากร่องระหว่างนิ้วเท้าไปตามความยาวนิ้ว แทนที่จะมีแค่ที่ขอบล่างสุด

    ปัญหา: มุมหน้าเห็นร่องระหว่างนิ้วเฉพาะช่วง 2 ซม. ล่างสุดของเท้าเท่านั้น
    (สูงกว่านั้นนิ้วรวมเป็นหลังเท้าก้อนเดียว ซึ่งถูกต้องตามกายวิภาค)
    visual hull จึงได้เท้าที่เป็นก้อนรองเท้าบูทมีรอยหยักจาง ๆ ที่พื้นรองเท้า

    สิ่งที่เรา *รู้* เพิ่มคือทิศ: นิ้วเท้าทอดไปข้างหน้า (+Z) จากโคนนิ้วถึง
    ปลายเล็บ ร่องที่เห็นที่ขอบล่างจึงต้องลากต่อไปตลอดความยาวนิ้ว ไม่ใช่
    หยุดแค่ขอบ นี่เป็นความรู้เชิงรูปร่าง ไม่ใช่ข้อมูลจากภาพ — แบบเดียวกับ
    ที่โมดูลนี้ใช้กับหาง (ท่อกลม) และปาก (ก้อนเรียว)

    ร่องลึกที่สุดที่ปลายนิ้วด้านล่าง แล้วตื้นหายไปเมื่อเข้าใกล้โคนนิ้ว
    """
    if not ext.toe_gaps or ext.y_ankle < 4:
        return 0

    ny, nx, nz = occ.shape
    y_top = max(2, int(cfg.toe_height_frac * ext.y_ankle))
    zs = np.arange(nz, dtype=np.float32)
    xs = np.arange(nx, dtype=np.float32)
    removed = 0

    for y in range(0, min(y_top, ny)):
        if not valid[y]:
            continue
        z_front = float(cz[y] + rz[y])
        z_ball = float(cz[y] - cfg.toe_ball_frac * rz[y])
        depth = max(z_front - z_ball, 1.0)
        # ร่องเต็มความลึกที่ปลายนิ้ว ตื้นลงจนเป็นศูนย์ที่โคนนิ้ว
        s_z = np.clip((zs - z_ball) / depth, 0.0, 1.0)
        # และตื้นลงเมื่อสูงขึ้นไปทางหลังเท้า
        #
        # ยกกำลังน้อยกว่า 1 เพื่อให้ร่องยัง "เต็ม" อยู่เกือบตลอดความสูงของนิ้ว
        # แล้วค่อยหายไปเร็ว ๆ ตรงหลังเท้า ซึ่งตรงกับรูปจริง: นิ้วแยกกันชัด
        # เกือบถึงโคน ไม่ได้ค่อย ๆ เชื่อมกันตั้งแต่กลางนิ้ว
        s_y = float(np.clip(1.0 - y / y_top, 0.0, 1.0)) ** 0.4
        if s_y <= 0.02:
            continue

        plane = occ[y]
        for lo, hi, x_ball in ext.toe_gaps:
            c_tip = 0.5 * (lo + hi)
            half = max(0.5 * (hi - lo + 1), 1.0)
            # ร่องแผ่ออกเป็นรูปพัดจากโคนนิ้ว ไม่ใช่เส้นขนานกับแกน Z
            #
            # นิ้วเท้ากางออกจากจุดเดียวกันที่ฝ่าเท้า (เห็นชัดมากในภาพมุมบน)
            # ร่องที่ลากขนานกันจะไปเฉือนข้างนิ้วก้อยทิ้งตอนปลาย และเปิดไม่พอ
            # ตรงที่นิ้วแยกกันจริง การให้มันแผ่ออกตามแนวรัศมีจึงตรงกว่ามาก
            c = x_ball + (c_tip - x_ball) * (cfg.toe_fan + (1.0 - cfg.toe_fan) * s_z)
            w = half * s_y * s_z                      # (nz,)
            cut = np.abs(xs[:, None] - c[None, :]) <= w[None, :]
            removed += int((plane & cut).sum())
            plane &= ~cut

    if removed:
        log.info("เซาะร่องระหว่างนิ้วเท้า %d ร่อง ถึงแถว %d (-%d voxel)",
                 len(ext.toe_gaps), y_top, removed)
    return removed


def apply_sole_plan(
    occ: np.ndarray, ext: "Extremities", plans: list, cz: np.ndarray,
    rz: np.ndarray, valid: np.ndarray, cfg: PipelineConfig,
) -> int:
    """บีบหน้าตัดของเท้าให้เป็น *รูปฝ่าเท้าจริง* ที่อ่านจากภาพมุมล่าง

    วิธีคิด: visual hull ให้ *ขอบเขต* ของเท้ามาถูกแล้ว (กว้างเท่าไรจากมุมหน้า
    ยาวเท่าไรจากมุมข้าง) สิ่งที่มันไม่รู้คือเนื้อกระจายตัวยังไงในระนาบ X-Z
    มันจึงเดาเป็นวงรีตัน ผังฝ่าเท้าตอบคำถามนั้นพอดี เราจึงแค่เอามา *ตัด*
    ส่วนที่ไม่ใช่เนื้อเท้าออก (ร่องระหว่างนิ้ว ซอกโคนนิ้ว ส่วนเว้าข้างฝ่าเท้า)
    ไม่ได้เอามาแทนที่ขอบเขต ขนาดจึงยังมาจากภาพมุมมาตรฐานเหมือนเดิม

    นิ้วแยกกันชัดที่พื้น แล้วค่อย ๆ กลืนเป็นหลังเท้าก้อนเดียวเมื่อสูงขึ้น
    ทำด้วยการขยาย (dilate) ผังทีละนิดตามความสูง ซึ่งปิดร่องจากปลายเข้าหาโคน
    ตามลำดับเดียวกับเท้าจริง
    """
    if not plans or not ext.foot_spans or ext.y_ankle < 4:
        return 0

    from . import soleplan

    ny, nx, nz = occ.shape
    y_merge = max(2, int(cfg.sole_merge_frac * ext.y_ankle))
    body_cx = 0.5 * (ext.foot_spans[0][0] + ext.foot_spans[-1][1])
    removed = 0

    for x_lo, x_hi in ext.foot_spans:
        side = -1 if 0.5 * (x_lo + x_hi) < body_cx else 1
        plan = next((p for p in plans if p.side == side), plans[0])

        # ช่วงความลึกของเท้า อ่านจากแถวที่เท้ากว้างที่สุด (กลางฝ่าเท้า)
        rows = [y for y in range(min(y_merge, ny)) if valid[y]]
        if not rows:
            continue
        y_ref = max(rows, key=lambda y: rz[y])
        z_lo = int(np.floor(cz[y_ref] - rz[y_ref]))
        z_hi = int(np.ceil(cz[y_ref] + rz[y_ref]))
        z_lo, z_hi = max(z_lo, 0), min(z_hi, nz - 1)
        w, d = x_hi - x_lo + 1, z_hi - z_lo + 1
        if w < 6 or d < 6:
            continue

        max_r = max(1, int(round(cfg.sole_merge_radius * w)))

        # กรอบปลายทางที่ *ขอบบนของเขต* ซึ่งเป็นแถวที่ silhouette เชื่อถือได้
        #
        # ห้ามใช้ขนาดของแต่ละแถวเป็นเป้า เพราะแถวกลาง ๆ ของเท้าวัดได้เล็ก
        # ผิดปกติ (ขอบล่างของเท้าในภาพเป็นเส้นโค้ง แถวกลางจึงเห็นแค่บางส่วน)
        # ถ้าไล่เข้าหาค่านั้น เท้าจะคอดเป็นเอวตรงกลางแล้วบานออกทั้งบนและล่าง
        top = occ[min(y_merge, ny - 1), x_lo : x_hi + 1]
        tx = np.flatnonzero(top.any(axis=1))
        tz = np.flatnonzero(top.any(axis=0))
        if tx.size and tz.size:
            ax, bx = x_lo + int(tx[0]), x_lo + int(tx[-1])
            az, bz = int(tz[0]), int(tz[-1])
        else:
            ax, bx, az, bz = x_lo, x_hi, z_lo, z_hi

        for y in rows:
            t = min(y / y_merge, 1.0)
            lo_x = int(round(x_lo * (1 - t) + ax * t))
            hi_x = int(round(x_hi * (1 - t) + bx * t))
            lo_z = int(round(z_lo * (1 - t) + az * t))
            hi_z = int(round(z_hi * (1 - t) + bz * t))
            if hi_x - lo_x < 4 or hi_z - lo_z < 4:
                continue

            r = int(round(max_r * t ** cfg.sole_merge_power))
            base = soleplan.resample(plan, hi_x - lo_x + 1, hi_z - lo_z + 1)
            stamp = base
            if r:
                # ใช้ closing ไม่ใช่ dilation
                #
                # dilation ขยาย *ขอบนอก* ของเท้าออกไปด้วย พอขึ้นไปถึงแถวบน ๆ
                # ของเขต รัศมีโตพอที่จะดันรูปเท้าจนเต็มกรอบสี่เหลี่ยม เท้าเลย
                # กลายเป็นแผ่นกระดานแบน ๆ ยื่นออกมาทุกด้าน
                # closing ถมเฉพาะ *ร่องแคบ* (ซอกระหว่างนิ้ว) โดยขอบนอกอยู่ที่เดิม
                # ซึ่งตรงกับสิ่งที่ต้องการพอดี: นิ้วกลืนกันเป็นหลังเท้า แต่เท้า
                # ไม่บวมขึ้น
                # ต้องเติมขอบก่อน: ผังเท้าแตะขอบอาเรย์ทุกด้าน (มันคือ bounding
                # box ของตัวเอง) ถ้าปิดโดยไม่เติมขอบ ขั้น erosion จะกินเนื้อ
                # ริมขอบหายไปเป็นแถบกว้าง r voxel รอบตัว เท้าเลยหดเป็นสี่เหลี่ยม
                st = np.ones((2 * r + 1, 2 * r + 1), bool)
                pad = np.pad(base, r, constant_values=False)
                closed = ndimage.binary_closing(pad, st)
                stamp = closed[r:-r, r:-r]

            # มนมุมกรอบด้วย superellipse
            #
            # ผังที่ผ่าน closing รัศมีใหญ่ ๆ จะเต็มกรอบจนมุมเป็นฉาก เท้าเลย
            # ออกมาเป็นกล่อง เท้าจริงมีสันข้างที่โค้ง superellipse กำลัง 4
            # มนมุมให้โดยยังเก็บนิ้วที่ยื่นไปชนขอบกรอบด้านหน้าไว้ได้
            if cfg.sole_round_power > 0:
                gx = np.abs((np.arange(stamp.shape[0], dtype=np.float32)
                             - 0.5 * (stamp.shape[0] - 1))
                            / max(0.5 * stamp.shape[0], 1.0))
                gz = np.abs((np.arange(stamp.shape[1], dtype=np.float32)
                             - 0.5 * (stamp.shape[1] - 1))
                            / max(0.5 * stamp.shape[1], 1.0))
                q = cfg.sole_round_power
                stamp = stamp & ((gx[:, None] ** q + gz[None, :] ** q) <= 1.0)

            # *แทนที่* หน้าตัดทั้งแถว ไม่ใช่แค่ตัดส่วนเกินออก
            #
            # แถวล่างสุดของเท้าคือฝ่าเท้าที่แปะอยู่กับพื้น ซึ่ง "ควร" เป็น
            # รอยเท้าเต็ม ๆ แต่ silhouette มุมหน้าที่แถวนั้นเห็นแค่ปลายเล็บ
            # (ขอบล่างของเท้าในภาพเป็นเส้นโค้ง ไม่ใช่เส้นตรง) ถ้าเอาผังไป
            # *ตัด* อย่างเดียว ก็ไม่มีอะไรให้ตัด เพราะเนื้อแทบไม่มีอยู่แล้ว
            # ผังฝ่าเท้าคือข้อมูลที่ถูกต้องกว่าสำหรับแถวพวกนี้ จึงให้มันเป็น
            # ตัวกำหนดรอยเท้าไปเลย
            before = int(occ[y, x_lo : x_hi + 1].sum())
            occ[y, x_lo : x_hi + 1] = False
            occ[y, lo_x : hi_x + 1, lo_z : hi_z + 1] = stamp
            removed += abs(int(occ[y, x_lo : x_hi + 1].sum()) - before)

    if removed:
        log.info("แทนหน้าตัดเท้าด้วยผังฝ่าเท้า %d ข้าง ถึงแถว %d (ต่าง %d voxel)",
                 len(ext.foot_spans), y_merge, removed)
    return removed


# ---------------------------------------------------------------------------
# มือ
# ---------------------------------------------------------------------------


def _arm_runs(fb: np.ndarray, min_run: int, y_floor: int) -> tuple[list, list]:
    """ท่อนแขนในแต่ละแถว แยกเป็นฝั่งซ้ายกับฝั่งขวา

    วิธี: วัดช่วงแกน X ของ *ขาทั้งสองข้าง* ที่ระดับกลางต้นขา ซึ่งเป็นแถบที่
    มีแต่ขาแน่ ๆ (เหนือเท้า ใต้สะโพก ไม่มีแขนลงมาถึง) แล้วถือว่าอะไรก็ตาม
    ที่อยู่ *นอก* ช่วงนั้นไปทางซ้าย คือแขนซ้าย นอกไปทางขวาคือแขนขวา

    เคยลองใช้ "ท่อนนอกสุดเมื่อมีตั้งแต่ 3 ท่อน" ซึ่งพังสองแบบ: แถวที่ขาสอง
    ข้างแยกกันจะนับขาเป็นแขน และที่สำคัญกว่าคือตอนที่มือแยกเป็นหลายท่อน
    (นิ้วแยกกัน!) มันคืนแค่ท่อนแรก เขตของมือจึงคลุมแค่นิ้วชุดเดียว แล้ว
    นิ้วที่เหลือหลุดออกนอกเขตไปโดนกฎของลำตัวเล่นงานตามเดิม

    การวัดจากช่วงขาคืนค่าเป็น *ช่วงรวม* ของทุกท่อนฝั่งนั้น จึงคลุมนิ้วครบ
    """
    ny, nx = fb.shape
    left: list[tuple[int, int] | None] = [None] * ny
    right: list[tuple[int, int] | None] = [None] * ny

    # ช่วงของขาที่ระดับกลางต้นขา — เฉลี่ยหลายแถวกันขอบ mask สั่น
    lo_rows, hi_rows = int(0.22 * ny), int(0.34 * ny)
    spans = [(min(r[0] for r in rr), max(r[1] for r in rr))
             for rr in (_runs(fb[y], min_run) for y in range(lo_rows, hi_rows))
             if rr]
    if not spans:
        return left, right
    body_lo = float(np.median([s[0] for s in spans]))
    body_hi = float(np.median([s[1] for s in spans]))
    log.debug("ช่วงขาที่ใช้แยกแขน: x %.0f-%.0f", body_lo, body_hi)

    for y in range(max(0, y_floor + 1), ny):
        rr = _runs(fb[y], min_run)
        if len(rr) < 2:
            continue
        outer_l = [r for r in rr if r[1] < body_lo]
        outer_r = [r for r in rr if r[0] > body_hi]
        if outer_l:
            left[y] = (min(r[0] for r in outer_l), max(r[1] for r in outer_l))
        if outer_r:
            right[y] = (min(r[0] for r in outer_r), max(r[1] for r in outer_r))
    return left, right


def _contiguous_band(arm: list, ny: int, min_rows: int) -> tuple[int, int] | None:
    """ช่วงแถวที่ต่อเนื่องกันและสูงที่สุดของแขนข้างหนึ่ง

    แขนเป็นชิ้นเดียวจากไหล่ถึงปลายนิ้ว ช่วงแถวของมันจึงต่อเนื่อง ส่วนท่อน
    นอกสุดที่โผล่มาเป็นหย่อม ๆ ที่ระดับอื่น (หางที่สะบัดออกข้าง เข็มขัด
    กระเป๋า) ไม่ต่อเนื่องกับไหล่ จึงตัดทิ้งได้ด้วยการเลือกช่วงที่ต่อกับ
    แถวบนสุดเท่านั้น
    """
    rows = [y for y in range(ny) if arm[y] is not None]
    if not rows:
        return None
    y_top = rows[-1]
    y = y_top
    gap = 0
    while y > 0:
        if arm[y - 1] is not None:
            gap = 0
        else:
            gap += 1
            if gap > 3:         # ยอมให้ขาดได้ไม่กี่แถว (ขอบ mask สั่น)
                break
        y -= 1
    y_bot = y + gap
    if y_top - y_bot + 1 < min_rows:
        return None
    return y_bot, y_top


def _hand_span(arm: list, band: tuple[int, int],
               cfg: PipelineConfig) -> tuple[int, int] | None:
    """ช่วงแถวของ *มือ* ในแขนข้างหนึ่ง คืน (แถวปลายนิ้ว, แถวข้อมือ)

    ปลายนิ้วคือแถวล่างสุดที่ยังมีแขนอยู่ ส่วนข้อมือคิดเป็นสัดส่วนของความยาว
    แขนตั้งแต่รักแร้ลงมา

    เคยลองหาข้อมือจาก "จุดที่แขนแคบที่สุด" ตามกายวิภาค แต่ใช้ไม่ได้จริง:
    ตัวละครจำนวนมากใส่สนับข้อมือ ปลอกแขน หรือถุงมือที่หนากว่าข้อมือเอง
    และตัวละครกล้ามโตจะมีปลายแขนกว้างขึ้นเรื่อย ๆ จนไม่มีรอยคอดเลย
    ค่าต่ำสุดที่หาได้จึงเป็นแค่สัญญาณรบกวน และวางข้อมือไว้กลางฝ่ามือ

    เขตที่กว้างเกินไปไม่เสียหาย (แค่ผิวแถวนั้นเกลาน้อยลง) แต่เขตที่แคบ
    เกินไปทำให้ฝ่ามือถูกเบลอกลืนกับนิ้ว สัดส่วนตายตัวจึงปลอดภัยกว่า
    """
    y_tip, y_top = band
    length = y_top - y_tip
    if length < 6:
        return None
    y_wrist = y_tip + max(3, int(cfg.hand_length_frac * length))
    return y_tip, int(min(y_wrist, y_top))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def detect(fb: np.ndarray, cfg: PipelineConfig,
           front: np.ndarray | None = None) -> Extremities:
    """หาเขตมือและเท้าจาก silhouette ด้านหน้า/หลัง (ก่อนถูกลบหนาม)

    `front` คือ silhouette ของมุมหน้ามุมเดียว ใช้เฉพาะตอนหาร่องนิ้วเท้า
    ซึ่งมุมหลังมีแต่จะลบทิ้ง
    """
    ny, nx = fb.shape
    min_run = max(1, nx // 110)
    notes: list[str] = []

    foot_rows = np.zeros(ny, bool)
    hand_rows = np.zeros(ny, bool)
    zone = np.zeros((ny, nx), bool)
    z_push = np.zeros(ny, np.float32)
    toe_gaps: list[tuple[float, float, float]] = []
    foot_spans: tuple = ()

    # --- เท้า ---------------------------------------------------------------
    if cfg.detail_feet:
        y_ankle = _find_ankle(fb, cfg)
        foot_rows[: y_ankle + 1] = True
        zone[: y_ankle + 1] = fb[: y_ankle + 1]
        # นิ้วเท้าชี้ไปข้างหน้า จึงต้องวางไว้ค่อนไปทางขอบหน้าของช่วงลึก
        z_push[: y_ankle + 1] = cfg.toe_z_push
        log.info("เขตเท้า: แถว 0-%d (%.1f%% ของความสูง)",
                 y_ankle, 100 * (y_ankle + 1) / ny)
        # ช่วงแกน X ของเท้าแต่ละข้าง = ยูเนียนของทุกแถวในเขตเท้า
        union = fb[: y_ankle + 1].any(axis=0)
        foot_spans = tuple(r for r in _runs(union, max(2, nx // 110))
                           if r[1] - r[0] >= nx // 40)
        if cfg.toe_grooves:
            toe_gaps = find_toe_gaps(front if front is not None else fb,
                                     y_ankle, cfg)
            if not toe_gaps:
                notes.append(
                    "แยกนิ้วเท้าจากภาพด้านหน้าไม่ได้ — เท้าจะออกมาเป็นก้อนเดียว "
                    "ภาพเสริมที่ช่วยได้มากคือภาพเท้ามองจากด้านบนหรือจากใต้ฝ่าเท้า"
                )
                log.warning(notes[-1])
    else:
        y_ankle = -1

    # --- มือ ----------------------------------------------------------------
    if cfg.detail_hands:
        left, right = _arm_runs(fb, min_run, y_ankle)
        found = 0
        for side, arm in (("ซ้าย", left), ("ขวา", right)):
            band = _contiguous_band(arm, ny, max(8, int(0.08 * ny)))
            if band is None:
                continue
            span = _hand_span(arm, band, cfg)
            if span is None:
                continue
            y_tip, y_wrist = span
            found += 1
            for y in range(y_tip, y_wrist + 1):
                r = arm[y]
                if r is None:
                    continue
                hand_rows[y] = True
                # เผื่อขอบออกข้างละนิด กันนิ้วที่กางออกนอกช่วงที่วัดได้
                a = max(0, r[0] - min_run)
                b = min(nx - 1, r[1] + min_run)
                zone[y, a : b + 1] |= fb[y, a : b + 1]
            log.info("เขตมือฝั่ง%s: แถว %d-%d (ยาว %.1f%% ของความสูง)",
                     side, y_tip, y_wrist, 100 * (y_wrist - y_tip + 1) / ny)
        if found == 0:
            notes.append(
                "หามือในภาพหน้า/หลังไม่เจอ (แขนไม่แยกออกจากลำตัวใน silhouette) — "
                "นิ้วจะถูกเกลาหายเหมือนเดิม ลองใช้ภาพที่กางแขนออกจากตัวมากขึ้น"
            )
            log.warning(notes[-1])

    if cfg.detail_dilate > 0 and zone.any():
        k = int(cfg.detail_dilate)
        zone = ndimage.binary_dilation(zone, np.ones((2 * k + 1, 2 * k + 1), bool))

    return Extremities(foot_rows, hand_rows, zone, z_push, notes,
                       y_ankle=y_ankle, toe_gaps=tuple(toe_gaps),
                       foot_spans=foot_spans)
