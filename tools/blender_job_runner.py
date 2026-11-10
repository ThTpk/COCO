"""ตัวรับงานใน Blender ที่เปิดค้างไว้: วางไฟล์ .json ใน output/jobs แล้วมันจะรัน Blender แบบ
background ให้ทีละงาน ใช้แทนการพิมพ์คำสั่งในหน้าต่าง Blender ซ้ำ ๆ

เปิดใช้ (ใน Python Console ของ Blender):
    p=r"<โปรเจกต์>\\tools\\blender_job_runner.py"; exec(open(p,encoding="utf-8").read(),{"__file__":p})

ไฟล์งาน output/jobs/<ชื่อ>.json:
    {"script": "tools/mv_bake.py", "args": ["--mesh", "output/x.glb", ...]}
path ที่ขึ้นต้นด้วย tools/ หรือ output/ หรือ input/ จะถูกแปลงเป็น path เต็มของโปรเจกต์
สถานะ: <ชื่อ>.json -> <ชื่อ>.running -> <ชื่อ>.done (หรือ .failed) และ log อยู่ที่ <ชื่อ>.log
"""
import json
import os
import subprocess

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = os.path.join(ROOT, "output", "jobs")
os.makedirs(JOBS, exist_ok=True)
_state = {"proc": None, "name": None, "log": None}


def _abs(v):
    if isinstance(v, str) and v.split("/")[0] in ("tools", "output", "input"):
        return os.path.join(ROOT, *v.split("/"))
    return v


def _tick():
    st = _state
    if st["proc"] is not None:
        rc = st["proc"].poll()
        if rc is None:
            return 2.0
        st["log"].close()
        base = os.path.join(JOBS, st["name"])
        os.replace(base + ".running", base + (".done" if rc == 0 else ".failed"))
        st["proc"] = None
    pending = sorted(f for f in os.listdir(JOBS) if f.endswith(".json"))
    if not pending:
        return 2.0
    name = pending[0][:-5]
    base = os.path.join(JOBS, name)
    job = json.load(open(base + ".json", encoding="utf-8"))
    os.replace(base + ".json", base + ".running")
    cmd = [bpy.app.binary_path, "-b", "--factory-startup", "--python", _abs(job["script"]), "--"]
    cmd += [_abs(a) for a in job.get("args", [])]
    st["log"] = open(base + ".log", "w", encoding="utf-8")
    st["proc"] = subprocess.Popen(cmd, stdout=st["log"], stderr=subprocess.STDOUT)
    st["name"] = name
    print("[jobs] เริ่ม", name)
    return 2.0


if not bpy.app.timers.is_registered(_tick):
    bpy.app.timers.register(_tick, first_interval=1.0, persistent=True)
print("[jobs] ตัวรับงานทำงานแล้ว เฝ้าที่", JOBS)
