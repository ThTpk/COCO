"""นิยามอนิเมชันแบบข้อมูลล้วน ให้ฝั่ง Blender เอาไปสร้าง action

ทุก channel คือคลื่นไซน์หนึ่งลูกบนกระดูกหนึ่งชิ้น

    มุม(t) = offset + amp * wave(2π * (harmonic * t + phase))

`t` คือตำแหน่งในลูป 0..1 ซึ่งทำให้ทุก clip วนต่อเนื่องกันได้เองโดยอัตโนมัติ
วิธีนี้พอสำหรับท่าเดิน ท่ายืนหายใจ และการสะบัดหาง ซึ่งล้วนเป็นการแกว่งเป็นจังหวะ

ระบบแกนเป็นของ Blender (Z ขึ้น) หลังจาก importer แปลงจาก glTF แล้ว
ตัวละครหันหน้าไปทาง -Y

    หมุนรอบ X = แกว่งหน้า-หลัง (ยกขา ก้าวเดิน)   ค่าบวก = ไปข้างหลัง
    หมุนรอบ Y = เอียงซ้าย-ขวา
    หมุนรอบ Z = บิดลำตัว / สะบัดหางซ้ายขวา
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Channel:
    """คลื่นหนึ่งลูกบนกระดูกหนึ่งชิ้น"""

    bone: str
    axis: str = "X"            # "X" | "Y" | "Z" ในพิกัด armature
    amp: float = 0.0           # องศา (kind="rot") หรือเมตร (kind="loc")
    phase: float = 0.0         # เลื่อนเฟส 0..1 ของหนึ่งลูป
    offset: float = 0.0        # ค่าคงที่ที่บวกเพิ่มตลอดทั้ง clip
    harmonic: float = 1.0      # จำนวนรอบต่อหนึ่งลูป
    mode: str = "sin"          # "sin" | "halfsin" (ครึ่งบวกเท่านั้น) | "const"
    kind: str = "rot"          # "rot" | "loc"


@dataclass
class Clip:
    name: str
    frames: int = 32           # ความยาวลูป (เฟรม)
    channels: list[Channel] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "frames": self.frames,
                "channels": [asdict(c) for c in self.channels]}


# ---------------------------------------------------------------------------
# ตัวช่วยสร้างโซ่หาง
# ---------------------------------------------------------------------------


def tail_wave(
    n_bones: int,
    axis: str = "Z",
    amp_first: float = 4.0,
    amp_last: float = 16.0,
    phase_step: float = 0.09,
    phase0: float = 0.0,
    harmonic: float = 1.0,
) -> list[Channel]:
    """คลื่นวิ่งไล่จากโคนหางไปปลายหาง

    หัวใจคือ `phase_step`: แต่ละข้อเริ่มขยับช้ากว่าข้อก่อนหน้าเล็กน้อย
    หางจึงดูเหมือนสะบัดเป็นคลื่น ไม่ใช่แกว่งทั้งท่อนพร้อมกันเหมือนไม้แข็ง
    และแอมพลิจูดไล่จากน้อยที่โคนไปมากที่ปลาย ตามที่หางจริงเคลื่อนที่
    """
    out: list[Channel] = []
    for i in range(n_bones):
        t = i / max(n_bones - 1, 1)
        out.append(Channel(
            bone=f"Tail{i + 1}",
            axis=axis,
            amp=amp_first + (amp_last - amp_first) * t,
            phase=phase0 + phase_step * i,
            harmonic=harmonic,
        ))
    return out


# ---------------------------------------------------------------------------
# clip มาตรฐาน
# ---------------------------------------------------------------------------


def idle(n_tail: int) -> Clip:
    """ยืนหายใจ หางแกว่งเบา ๆ"""
    ch = [
        Channel("Spine", "X", amp=0.8, harmonic=1),
        Channel("Spine1", "X", amp=1.2, harmonic=1, phase=0.05),
        Channel("Spine2", "X", amp=0.9, harmonic=1, phase=0.10),
        Channel("Neck", "X", amp=-0.8, harmonic=1, phase=0.15),
        Channel("Head", "Y", amp=1.2, harmonic=0.5, phase=0.25),
        Channel("LeftArm", "Y", amp=1.5, harmonic=1, phase=0.3),
        Channel("RightArm", "Y", amp=-1.5, harmonic=1, phase=0.3),
        Channel("Hips", "Z", amp=0.0, kind="loc", harmonic=2),
    ]
    ch += tail_wave(n_tail, amp_first=2.0, amp_last=9.0, harmonic=1.0)
    return Clip("Idle", frames=64, channels=ch)


def walk(n_tail: int) -> Clip:
    """เดินหนึ่งรอบ (ซ้าย-ขวาครบ)

    ขาสองข้างเป็นคลื่นเดียวกันต่างเฟสครึ่งลูป แขนแกว่งสวนทางกับขาข้างเดียวกัน
    เข่าใช้ halfsin เพราะเข่างอได้ทางเดียว
    """
    ch = [
        # ขา: ค่าลบ = เหวี่ยงไปข้างหน้า
        Channel("RightUpLeg", "X", amp=-26, phase=0.00),
        Channel("LeftUpLeg", "X", amp=-26, phase=0.50),
        Channel("RightLeg", "X", amp=40, phase=0.55, mode="halfsin"),
        Channel("LeftLeg", "X", amp=40, phase=0.05, mode="halfsin"),
        Channel("RightFoot", "X", amp=-10, phase=0.30),
        Channel("LeftFoot", "X", amp=-10, phase=0.80),
        # แขนแกว่งสวนกับขาข้างเดียวกัน
        Channel("RightArm", "X", amp=18, phase=0.00),
        Channel("LeftArm", "X", amp=18, phase=0.50),
        Channel("RightForeArm", "X", amp=10, offset=8, phase=0.10),
        Channel("LeftForeArm", "X", amp=10, offset=8, phase=0.60),
        # ลำตัวบิดสวนกับสะโพก
        Channel("Spine1", "Z", amp=4, phase=0.00),
        Channel("Spine2", "Z", amp=3, phase=0.05),
        Channel("Hips", "Z", amp=-5, phase=0.00),
        # ย่อ-ยืดตามจังหวะก้าว สองครั้งต่อหนึ่งรอบ
        Channel("Hips", "Z", amp=0.035, harmonic=2, phase=0.25, kind="loc"),
    ]
    ch += tail_wave(n_tail, amp_first=3.0, amp_last=14.0,
                    phase_step=0.10, harmonic=1.0)
    return Clip("Walk", frames=32, channels=ch)


def tail_swing(n_tail: int) -> Clip:
    """สะบัดหางอย่างเดียว ใช้ตรวจว่ากระดูกหางผูกกับเนื้อถูกหรือไม่"""
    ch = tail_wave(n_tail, amp_first=8.0, amp_last=28.0,
                   phase_step=0.12, harmonic=1.0)
    ch.append(Channel("Hips", "Z", amp=2.0, harmonic=1, phase=0.5))
    return Clip("TailSwing", frames=40, channels=ch)


def default_clips(n_tail: int) -> list[Clip]:
    """ชุด clip ที่สร้างให้อัตโนมัติ — ข้าม clip หางเมื่อตัวละครไม่มีหาง"""
    clips = [idle(n_tail), walk(n_tail)]
    if n_tail > 0:
        clips.append(tail_swing(n_tail))
    return clips


def to_payload(clips: list[Clip], fps: int) -> dict:
    return {"fps": fps, "clips": [c.to_dict() for c in clips]}
