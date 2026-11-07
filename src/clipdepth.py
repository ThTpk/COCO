r"""ดันผิวจริงตามความเข้มของสี — shape from shading ที่ลงไปถึงเรขาคณิต

normal map ทำให้ *แสง* ตกกระทบเหมือนผิวขรุขระ แต่ผิวยังเรียบอยู่: เงาไม่ตก
ใส่กันเอง ขอบ silhouette ไม่หยัก และร่องระหว่างนิ้วก็ยังไม่มีอยู่จริง

โมดูลนี้เอาความสูงชุดเดียวกัน (``cliptexture.shading_height``) ไปขยับ vertex
จริง ๆ ตามแนว normal  สิ่งที่ visual hull ทำไม่ได้เลยคือส่วนเว้าที่ไม่เคยโผล่
บนเส้นขอบของมุมไหน — ร่องระหว่างนิ้ว ซอกใต้คาง รอยยับเสื้อ ล้วนเป็นแบบนั้น
แต่มันมืดกว่ารอบข้างเสมอ เพราะแสงรอบตัวเข้าไปไม่ถึง (ambient occlusion)
ความมืดนั้นจึงเป็นข้อมูลความลึกที่เงาไม่มีให้

ปัญหาเดียวคือ mesh จาก marching cubes ที่ decimate เหลือเจ็ดหมื่นสามเหลี่ยม
มีขอบยาวราว 1-2 ซม. ซึ่งหยาบกว่าร่องนิ้วที่กว้าง 5 มม. หลายเท่า จะดัน vertex
ให้เกิดร่องก็ไม่มี vertex ให้ดัน ต้องซอยสามเหลี่ยมก่อน — แต่ซอยทั้งตัวจะได้
สามเหลี่ยมหลายล้าน จึงซอยเฉพาะที่ที่ความสูงมีรายละเอียดจริง ๆ
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .meshing import MeshData

log = logging.getLogger(__name__)


def _sample(field: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """อ่านค่าจากภาพที่พิกัด UV แบบ bilinear (UV ของ glTF: v วิ่งลง)"""
    h, w = field.shape
    x = np.clip(uv[:, 0], 0, 1) * (w - 1)
    y = np.clip(uv[:, 1], 0, 1) * (h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top = field[y0, x0] * (1 - fx) + field[y0, x1] * fx
    bot = field[y1, x0] * (1 - fx) + field[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def subdivide_conforming(verts: np.ndarray, faces: np.ndarray,
                         want: np.ndarray, attrs: dict[str, np.ndarray],
                         ) -> tuple[np.ndarray, np.ndarray, dict]:
    """ซอยเฉพาะสามเหลี่ยมที่เลือก โดยไม่ทิ้งรอยแยกไว้ตามขอบเขต

    ซอยแบบเลือกตรง ๆ ใช้ไม่ได้: สามเหลี่ยมที่ถูกซอยจะมีจุดใหม่กลางขอบ ส่วน
    เพื่อนบ้านที่ไม่ถูกซอยยังเป็นขอบตรงเส้นเดียว จุดใหม่นั้นจึงลอยอยู่กลางขอบ
    ของเพื่อนบ้าน (T-junction) พอดันผิว จุดนั้นเคลื่อนออกจากขอบ เกิดเป็นรอย
    แยกให้เห็นทะลุเข้าไปในตัว — อาการคือเส้นดำ/ขาวบาง ๆ พาดไปทั่วผิว

    วิธีมาตรฐานคือ red-green refinement: สามเหลี่ยมที่เลือก (แดง) แบ่งสี่
    ส่วนเพื่อนบ้านที่โดนแบ่งขอบเดียว (เขียว) แบ่งสองเพื่อปิดรอยต่อ ส่วนที่โดน
    แบ่งตั้งแต่สองขอบขึ้นไปก็ยกระดับเป็นแดงไปเลย วนจนไม่มีใครเปลี่ยนอีก
    """
    nf = len(faces)
    pairs = np.sort(np.stack([faces[:, [0, 1]], faces[:, [1, 2]],
                              faces[:, [2, 0]]], axis=1), axis=2)
    uniq, inv = np.unique(pairs.reshape(-1, 2), axis=0, return_inverse=True)
    inv = inv.reshape(nf, 3)

    red = np.asarray(want, bool).copy()
    for _ in range(8):
        split = np.zeros(len(uniq), bool)
        split[inv[red].ravel()] = True
        promote = (~red) & (split[inv].sum(axis=1) >= 2)
        if not promote.any():
            break
        red |= promote

    split = np.zeros(len(uniq), bool)
    split[inv[red].ravel()] = True
    which = np.flatnonzero(split)
    mid_id = np.full(len(uniq), -1, np.int64)
    mid_id[which] = len(verts) + np.arange(len(which))
    lo, hi = uniq[which, 0], uniq[which, 1]

    verts = np.vstack([verts, 0.5 * (verts[lo] + verts[hi])])
    attrs = {k: np.vstack([v, 0.5 * (v[lo] + v[hi])]) for k, v in attrs.items()}

    m = mid_id[inv]                       # จุดกลางของแต่ละขอบ (-1 = ไม่ถูกแบ่ง)
    n_split = (m >= 0).sum(axis=1)
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    out = [faces[n_split == 0]]

    r = red
    if r.any():
        out.append(np.concatenate([
            np.stack([v0[r], m[r, 0], m[r, 2]], axis=1),
            np.stack([m[r, 0], v1[r], m[r, 1]], axis=1),
            np.stack([m[r, 2], m[r, 1], v2[r]], axis=1),
            np.stack([m[r, 0], m[r, 1], m[r, 2]], axis=1)]))

    green = (~red) & (n_split == 1)
    for k, (a, b, c) in enumerate(((v0, v1, v2), (v1, v2, v0), (v2, v0, v1))):
        g = green & (m[:, k] >= 0)
        if g.any():
            out.append(np.concatenate([
                np.stack([a[g], m[g, k], c[g]], axis=1),
                np.stack([m[g, k], b[g], c[g]], axis=1)]))

    return verts, np.vstack(out).astype(np.int64), attrs


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(verts)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norm, 1e-12, None)


def _detail_per_face(field: np.ndarray, uv: np.ndarray, faces: np.ndarray,
                     ) -> np.ndarray:
    """ความแรงของรายละเอียดบนแต่ละสามเหลี่ยม — ใช้ตัดสินว่าคุ้มที่จะซอยไหม

    วัดจากช่วงความสูงที่สามเหลี่ยมนั้นคร่อมอยู่ (ต่างกันมาก = มีร่องหรือสัน
    พาดผ่าน) ไม่ใช่ค่าเฉลี่ย เพราะผิวที่ลึกเท่ากันทั้งแผ่นไม่ต้องซอย
    """
    h = _sample(field, uv)
    at = h[faces]
    return at.max(axis=1) - at.min(axis=1)


def displace(mesh: MeshData, field: np.ndarray, amplitude, target_edge,
             budget: int = 400_000, detail_floor: float = 0.25,
             smooth: int = 2, blur: float = 2.2) -> MeshData:
    """ซอยสามเหลี่ยมเฉพาะที่ที่มีรายละเอียด แล้วดัน vertex ตามความสูง

    ``amplitude`` และ ``target_edge`` อยู่ในหน่วยเดียวกับ ``mesh.verts``
    (พิกัด voxel) ส่วน ``field`` คือความสูง -1..1 ในพิกัด UV

    ซอยแบบวนรอบ: รอบหนึ่งแบ่งสี่เฉพาะสามเหลี่ยมที่ยังใหญ่เกินและมีรายละเอียด
    พาดผ่าน จนกว่าจะหมดงบหรือเล็กพอแล้ว — ทำให้สามเหลี่ยมไปกองตรงนิ้วมือ
    นิ้วเท้า รอยยับเสื้อ แทนที่จะกระจายเท่ากันทั้งตัวซึ่งเปลืองเปล่า
    """
    if mesh.uv is None:
        raise ValueError("ต้องคลี่ UV ก่อนถึงจะดันผิวตามลายได้")

    # ตัดความถี่ที่สูงเกินกว่าสามเหลี่ยมจะรับไหวทิ้งก่อน
    #
    # ความสูงมาจาก texel ที่ละเอียดราว 1 มม. ส่วนสามเหลี่ยมหลังซอยยังกว้าง
    # 2-3 มม. การดันตามทุกความถี่จึงไม่ได้ร่องนิ้วเพิ่ม แต่ได้ผิวยู่ยี่เต็มตัว
    # จากเสียงรบกวนของ JPEG กับลายเกล็ดที่ติดมาในสี — ส่วนรายละเอียดที่ละเอียด
    # กว่านั้น normal map รับไปแล้ว
    if blur > 0:
        field = cv2.GaussianBlur(field, (0, 0), blur)

    verts = np.asarray(mesh.verts, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    uv = np.asarray(mesh.uv, np.float64)
    start = len(faces)

    # ทั้งสองค่าเป็นรายจุด เพื่อให้แต่ละส่วนของตัวมีกฎของตัวเอง — นิ้วมือ
    # ต้องการสามเหลี่ยมถี่กว่าลำตัวห้าเท่า และยอมให้ร่องลึกกว่าด้วย
    amp = np.broadcast_to(np.asarray(amplitude, np.float64), (len(verts),)).copy()
    edge = np.broadcast_to(np.asarray(target_edge, np.float64), (len(verts),)).copy()

    for _ in range(6):
        tri = verts[faces]
        edges = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(axis=1)
        detail = _detail_per_face(field, uv, faces)
        want = (edges > edge[faces].min(axis=1)) & (detail > detail_floor)

        # เกินงบก็ซอยเท่าที่งบให้ โดยเรียงจากสามเหลี่ยมที่คุ้มที่สุดก่อน
        # (ใหญ่และมีรายละเอียดพาดผ่านมาก) — ไม่ใช่ไม่ซอยอะไรเลย
        room = (budget - len(faces)) // 3
        if not want.any() or room <= 0:
            break
        idx = np.flatnonzero(want)
        if len(idx) > room:
            best = np.argsort(detail[idx] * edges[idx])[::-1][:room]
            want = np.zeros(len(faces), bool)
            want[idx[best]] = True
        verts, faces, attrs = subdivide_conforming(
            verts, faces, want,
            {"uv": uv, "amp": amp[:, None], "edge": edge[:, None]})
        uv, amp, edge = attrs["uv"], attrs["amp"].ravel(), attrs["edge"].ravel()

    normals = _vertex_normals(verts, faces)
    h = _sample(field, uv)

    # จุดที่อยู่บนรอยผ่าของ UV atlas ถูกแยกร่างเป็นหลายจุดที่ *ตำแหน่งเดียวกัน*
    # แต่ชี้ไปคนละที่บน texture ถ้าปล่อยให้แต่ละร่างดันคนละระยะ ผิวจะปริออก
    # เป็นรอยแยกตามรอยผ่าทั่วตัว — เฉลี่ยทั้งความสูงและ normal ให้ทุกร่าง
    # ของจุดเดียวกันเท่ากันก่อน
    group = _weld(verts)
    h = _group_mean(h[:, None], group)[:, 0]
    normals = _group_mean(normals, group)
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)

    verts = verts + normals * (h * amp)[:, None]

    if smooth > 0:
        # เกลาบนโครงที่เชื่อมร่างซ้ำของรอยผ่าเข้าด้วยกันแล้ว ไม่งั้นสองร่างของ
        # จุดเดียวกันจะถูกเกลาคนละทาง แล้วผิวปริออกตามรอยผ่าอีกรอบ
        verts = _taubin(verts, faces, smooth, group=group)

    out = MeshData(verts=verts, faces=faces.astype(np.int32),
                   uv=uv.astype(np.float32), history=list(mesh.history))
    out.normals = _vertex_normals(verts, faces)
    log.info("ดันผิวตามความเข้มของสี: สามเหลี่ยม %d -> %d "
             "(ลึกสุด %.2f-%.2f voxel แล้วแต่ส่วนของตัว)",
             start, len(faces), float(amp.min()), float(amp.max()))
    out.log_stage("displace จากความเข้มของสี")
    return out


def _weld(verts: np.ndarray, digits: int = 5) -> np.ndarray:
    """จัดกลุ่มจุดที่อยู่ตำแหน่งเดียวกัน คืน index ของกลุ่มต่อจุด"""
    key = np.round(verts * (10 ** digits)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv


def _group_mean(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    acc = np.zeros((group.max() + 1, values.shape[1]), np.float64)
    np.add.at(acc, group, values)
    return acc[group] / np.bincount(group, minlength=len(acc))[group, None]


def _taubin(verts: np.ndarray, faces: np.ndarray, rounds: int,
            lam: float = 0.5, mu: float = -0.53,
            group: np.ndarray | None = None) -> np.ndarray:
    """เกลาเบา ๆ แบบรักษาปริมาตร — กันยอดแหลมจาก texel ที่สว่าง/มืดผิดปกติ"""
    if group is None:
        group = np.arange(len(verts))
    n = int(group.max()) + 1
    e = group[np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])]
    e = np.vstack([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=n).astype(np.float64)
    deg[deg == 0] = 1.0

    out = np.zeros((n, 3))
    np.add.at(out, group, verts)
    out /= np.bincount(group, minlength=n)[:, None]
    for i in range(2 * rounds):
        acc = np.zeros_like(out)
        np.add.at(acc, e[:, 0], out[e[:, 1]])
        delta = acc / deg[:, None] - out
        out = out + (lam if i % 2 == 0 else mu) * delta
    return out[group]
