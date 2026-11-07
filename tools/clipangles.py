r"""เก็บภาพตัวละครจากคลิป "หลายมุม" ไว้เป็นชุดอ้างอิง ก่อนเอาไปขึ้นรูปใน 3D

    .venv\Scripts\python.exe tools\clipangles.py --views 36
    .venv\Scripts\python.exe tools\clipangles.py --views 24 --out output\croc_views

ทำอะไร
------
อ่านผลประเมินกล้องที่ ``src.clipcli`` เก็บไว้ในแคช แล้วเลือกเฟรมที่มุมกระจาย
เท่า ๆ กันรอบตัว ตัดพื้นหลังออกตาม mask เขียนเป็น PNG โปร่งใสมุมละไฟล์
(ชื่อไฟล์บอกมุมจริง เช่น ``a090_f0101.png`` = 90° จากเฟรมที่ 101) พร้อม
contact sheet รวมทุกมุมไว้ดูทีเดียว และ ``angles.json`` ที่เก็บพารามิเตอร์
กล้องของมุมที่เลือกไว้ให้ขั้นถัดไปใช้ต่อ

ต้องมีแคชอยู่ก่อน — ถ้ายังไม่เคยรัน ``src.clipcli`` กับคลิปนี้ สคริปต์จะดึง
เฟรมและแยกพื้นหลังเอง (ช้า) แต่จะไม่ประเมินกล้องใหม่ให้ เพราะขั้นนั้นกินเวลา
หลายนาที — ให้รัน ``src.clipcli`` ก่อนหนึ่งรอบ
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import turntable  # noqa: E402
from src.config import INPUT_DIR, OUTPUT_DIR, PipelineConfig  # noqa: E402

log = logging.getLogger("2Dto3D.angles")


def even_angles(cams: list, n: int) -> list[int]:
    """เลือก index ของเฟรมที่มุมกระจายเท่า ๆ กันรอบตัว"""
    picked: list[int] = []
    for want in np.arange(n) * (2 * np.pi / n):
        j = min(range(len(cams)),
                key=lambda i: abs(np.angle(np.exp(1j * (cams[i].theta - want)))))
        if j not in picked:
            picked.append(j)
    return picked


def cutout(frame, pad: int = 12) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """ตัดพื้นหลังออก คืนภาพ RGBA ที่ครอบเฉพาะตัวละคร"""
    mask = frame.mask
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError(f"เฟรม {frame.index} ไม่มีตัวละครใน mask")
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + 1 + pad)

    rgb = frame.rgb[y0:y1, x0:x1]
    a = (mask[y0:y1, x0:x1].astype(np.uint8) * 255)
    # เกลาขอบ alpha ไม่ให้เป็นฟันเลื่อย แล้วกัดเข้าในนิดหนึ่งกันขอบพื้นหลังติดมา
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    rgba = np.dstack([rgb, a])
    return rgba, (x0, y0, x1, y1)


def contact_sheet(tiles: list[np.ndarray], labels: list[str],
                  cols: int, cell: int = 320) -> np.ndarray:
    """เรียงทุกมุมลงแผ่นเดียว พื้นหลังเทาเข้มให้เห็นขอบตัวละครชัด"""
    rows = (len(tiles) + cols - 1) // cols
    pad_label = 22
    sheet = np.full((rows * (cell + pad_label), cols * cell, 3), 38, np.uint8)
    for k, (tile, label) in enumerate(zip(tiles, labels)):
        h, w = tile.shape[:2]
        s = min((cell - 16) / w, (cell - 16) / h)
        small = cv2.resize(tile, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        r, c = divmod(k, cols)
        oy = r * (cell + pad_label) + pad_label + (cell - sh) // 2
        ox = c * cell + (cell - sw) // 2
        alpha = small[:, :, 3:4].astype(np.float32) / 255.0
        patch = sheet[oy:oy + sh, ox:ox + sw]
        sheet[oy:oy + sh, ox:ox + sw] = (
            small[:, :, :3] * alpha + patch * (1 - alpha)).astype(np.uint8)
        cv2.putText(sheet, label, (c * cell + 10,
                                   r * (cell + pad_label) + pad_label - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1,
                    cv2.LINE_AA)
    return sheet


def write_png(path: Path, img: np.ndarray) -> None:
    """เขียน PNG ผ่าน buffer — cv2.imwrite เขียนพาธภาษาไทย/ยูนิโค้ดไม่ได้บนวินโดวส์"""
    bgr = img[:, :, [2, 1, 0, 3]] if img.shape[2] == 4 else img[:, :, ::-1]
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"เข้ารหัส PNG ไม่สำเร็จ: {path}")
    buf.tofile(str(path))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="clipangles",
        description="เก็บภาพตัวละครหลายมุมจากคลิป ไว้เป็นชุดอ้างอิง",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--video", type=Path, default=INPUT_DIR / "clip.mp4")
    p.add_argument("--cache", type=Path, default=OUTPUT_DIR / "clip_cache")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "croc_views")
    p.add_argument("--views", type=int, default=36, help="จำนวนมุมที่เก็บ")
    p.add_argument("--cols", type=int, default=6, help="จำนวนคอลัมน์ของ contact sheet")
    p.add_argument("--cell", type=int, default=320, help="ขนาดช่องใน contact sheet")
    p.add_argument("--rembg-model", default=PipelineConfig().rembg_model)
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    indices = turntable.extract_frames(a.video, a.cache, step=1)
    frames = turntable.segment_frames(indices, a.cache, a.rembg_model)
    stats = turntable.measure(frames)
    sol = turntable.load_solution(a.cache / "solution.json", stats)
    if sol is None:
        log.error("ยังไม่มีผลประเมินกล้องในแคช — รัน `python -m src.clipcli` "
                  "กับคลิปนี้หนึ่งรอบก่อน แล้วค่อยเรียกสคริปต์นี้")
        return 2

    pick = even_angles(sol.cams, a.views)
    a.out.mkdir(parents=True, exist_ok=True)

    tiles: list[np.ndarray] = []
    labels: list[str] = []
    meta: list[dict] = []
    for i in pick:
        cam = sol.cams[i]
        frame = frames[i]
        deg = float(np.degrees(cam.theta) % 360.0)
        rgba, (x0, y0, x1, y1) = cutout(frame)
        name = f"a{deg:05.1f}_f{frame.index:04d}.png"
        write_png(a.out / name, rgba)
        tiles.append(rgba)
        labels.append(f"{deg:5.1f}  f{frame.index}")
        meta.append({
            "file": name,
            "frame": int(frame.index),
            "angle_deg": round(deg, 2),
            "scale_px_per_unit": round(float(cam.scale), 3),
            "axis_px": [round(float(cam.cx), 2), round(float(cam.cy), 2)],
            "eye_height": round(float(cam.h), 4),
            "inv_distance": round(float(cam.q), 5),
            "full_body": bool(cam.full_body),
            "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "px_tall": int(y1 - y0),
        })
        frame.unload()
        log.info("เก็บมุม %5.1f องศา (เฟรม %3d) -> %s  [%dx%d]",
                 deg, frame.index, name, x1 - x0, y1 - y0)

    order = np.argsort([m["angle_deg"] for m in meta])
    sheet = contact_sheet([tiles[k] for k in order], [labels[k] for k in order],
                          a.cols, a.cell)
    sheet_path = a.out / "contact_sheet.png"
    write_png(sheet_path, sheet)

    (a.out / "angles.json").write_text(json.dumps({
        "video": str(a.video),
        "views": len(meta),
        "note": "angle_deg 0 = มุมหน้า, เพิ่มทวนเข็มเมื่อมองจากด้านบน",
        "frames": [meta[k] for k in order],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("")
    log.info("เก็บครบ %d มุม -> %s", len(meta), a.out)
    log.info("contact sheet -> %s (%.2f MB)", sheet_path,
             sheet_path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
