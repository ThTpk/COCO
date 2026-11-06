# การติดตั้ง Toolchain

เครื่องเป้าหมาย: Windows 11, RTX 5070 Laptop (Blackwell, `sm_120`), driver 610.71, Python 3.12.10

## ชุดเวอร์ชันที่เลือกและเหตุผล

| ส่วนประกอบ | เวอร์ชัน | เหตุผล |
|---|---|---|
| CUDA channel | `cu128` | CUDA 12.8 เป็นรุ่นแรกที่รองรับ `sm_120` |
| torch | `2.11.0+cu128` | ตัวใหม่สุดในสาย cu128 |
| torchvision | `0.26.0+cu128` | เวอร์ชันที่คู่กับ torch 2.11 |
| CUDA Toolkit | `12.8` | repo ส่วนใหญ่ยังเขียนบน CUDA 12.x — CUDA 13 ทำ `setup.py` พังบ่อย |
| VS Build Tools | `2022` (17.14) | ต้องใช้ `cl.exe` เวลาคอมไพล์ CUDA extension |

> **สำคัญ:** อย่าลง CUDA Toolkit 13.x แม้ winget จะเสนอเป็น default (13.5)
> `nvdiffrast`, `pytorch3d`, `spconv` ยังมีปัญหากับ CUDA 13 อยู่

---

## ขั้นที่ 1 — VS Build Tools + CUDA Toolkit (ต้องใช้สิทธิ์ Administrator)

เปิด **PowerShell แบบ Run as Administrator** แล้วรันสองคำสั่งนี้

ตัวแรก VS Build Tools พร้อม workload C++ (~5–7 GB ใช้เวลาสักพัก):

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements --accept-package-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

ตัวที่สอง CUDA Toolkit 12.8 — **ต้องระบุ `--version 12.8`** ไม่งั้น winget จะลง 13.5 ให้:

```powershell
winget install --id Nvidia.CUDA --version 12.8 --accept-source-agreements --accept-package-agreements
```

## ขั้นที่ 2 — ตรวจผล

ปิด terminal เดิมแล้วเปิดใหม่ (เพื่อให้ `PATH` อัปเดต) จากนั้น:

```powershell
nvcc --version
```

ควรขึ้น `release 12.8` ถ้า `nvcc` ยังไม่เจอ ให้เพิ่ม path นี้เข้า `PATH`:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
```

## ขั้นที่ 3 — ตรวจทั้งระบบด้วยสคริปต์

```powershell
.venv\Scripts\python.exe tools\check_env.py
```

---

## หมายเหตุเรื่อง venv

โปรเจกต์นี้ใช้ `.venv` แยก **ไม่ไปแตะ Python global ของเครื่อง** เพราะ global มี `torch 2.10.0+cpu`
ติดตั้งอยู่พร้อมกับ cupy / dash / flask / librosa — การสลับ torch ตัวนั้นเป็น CUDA เสี่ยงทำงานอื่นพัง

ลง dependency ที่เหลือ:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Blender

พบ Blender 5.2 ติดตั้งอยู่แล้วที่ `C:\Program Files\Blender Foundation\Blender 5.2`
ขั้น rigging เรียก Blender แบบ headless ไม่ต้องลง `bpy` เป็น pip package
