"""ย่อ texture ที่ pack อยู่ใน .glb แล้วเขียนไฟล์ใหม่ โดยไม่แตะ geometry

    python tools/glb_retex.py <src.glb> <dst.glb> <ขนาดภาพ> <คุณภาพ JPEG>
    python tools/glb_retex.py output/web/croc_web_v1.glb output/web/croc_web_v1_2k.glb 2048 88

มีไว้เพราะตัวเลือกย่อภาพของ Blender ตอน export ไม่ได้แตะภาพที่ pack มาในไฟล์
(log ขึ้น "Image not available. Keeping packed image") atlas 4096 PNG จึงติดไป
ทั้งใบ ตัวนี้แกะ bufferView ของทุกภาพออกมาย่อแล้วบีบเป็น JPEG ประกอบ buffer
ใหม่ให้ offset เรียงใหม่ทั้งก้อน ส่วน bufferView ของ geometry ที่ Draco บีบไว้
แล้วผ่านไปทั้งดุ้น ไม่ได้ถอดออกมาเข้ารหัสใหม่
"""
import json,struct,io,sys
from PIL import Image
src,dst,size,q=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
b=open(src,'rb').read(); L=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+L])
bl=struct.unpack('<I',b[20+L:24+L])[0]; BIN=b[28+L:28+L+bl]
views=[BIN[v.get('byteOffset',0):v.get('byteOffset',0)+v['byteLength']] for v in j['bufferViews']]
for im in j['images']:
    i=im['bufferView']; img=Image.open(io.BytesIO(views[i])).convert('RGB')
    if img.size[0]>size: img=img.resize((size,size),Image.LANCZOS)
    o=io.BytesIO(); img.save(o,'JPEG',quality=q,optimize=True,progressive=False); views[i]=o.getvalue(); im['mimeType']='image/jpeg'
out=bytearray()
for v,d in zip(j['bufferViews'],views):
    while len(out)%4: out+=b'\0'
    v['byteOffset']=len(out); v['byteLength']=len(d); out+=d
while len(out)%4: out+=b'\0'
j['buffers'][0]['byteLength']=len(out)
js=json.dumps(j,separators=(',',':')).encode()
while len(js)%4: js+=b' '
total=12+8+len(js)+8+len(out)
with open(dst,'wb') as f:
    f.write(struct.pack('<III',0x46546C67,2,total)); f.write(struct.pack('<II',len(js),0x4E4F534A)); f.write(js)
    f.write(struct.pack('<II',len(out),0x004E4942)); f.write(out)
print(dst,round(total/1e6,2),'MB')
