"""ซ่อมจุดสีผิดบน texture: สีน้ำเงิน/ดำที่ไปตกบนท้ายทอย มือ และเท้า

    python tools/texture_repair.py <model.glb> <out.png> <out.glb>

ระบุพื้นที่ด้วยพิกัด 3 มิติ (แกน glTF) แล้วเติมสีจากผิวส้มข้างเคียงด้วย normalized blur หลายสเกล
กำไลข้อมือ (|x| 0.74-0.83) ถูกกันไว้ไม่ให้กลายเป็นสีผิว ค่าขอบเขตผูกกับโมเดลชุด croc_full_* นี้
"""
import trimesh, numpy as np, cv2, sys
from PIL import Image, ImageDraw
GLB,OUTPNG,OUTGLB=sys.argv[1:4]
m=trimesh.load(GLB,force='mesh'); mat=m.visual.material
tex=np.asarray(mat.baseColorTexture.convert('RGB')).astype(np.float32); H,W=tex.shape[:2]
uv=m.visual.uv; F=m.faces; C=m.vertices[F].mean(1)
R=2048; fid=Image.new('RGB',(R,R),(0,0,0)); d=ImageDraw.Draw(fid); P=np.c_[uv[:,0]*R,(1-uv[:,1])*R]
for i in range(len(F)):
    k=i+1; d.polygon([tuple(P[j]) for j in F[i]],fill=(k&255,(k>>8)&255,(k>>16)&255))
a=np.asarray(fid).astype(np.int64); ids=a[...,0]+(a[...,1]<<8)+(a[...,2]<<16)-1
ids=cv2.resize(ids.astype(np.int32),(W,H),interpolation=cv2.INTER_NEAREST).astype(np.int64)
valid=ids>=0; pos=C[np.clip(ids,0,None)]; x,y,z=pos[...,0],pos[...,1],pos[...,2]
Rr,G,B=tex[...,0],tex[...,1],tex[...,2]
L=0.299*Rr+0.587*G+0.114*B
blue=(B>Rr+5); dark=L<45
skin=(Rr>B+25)&valid
notskin=~(Rr>B+15)
ax=np.abs(x)
head=(y>0.72)
hands=(ax>0.47)
feet=(y<-0.81)&(z>-0.25)
bracelet=(ax>0.74)&(ax<0.83)
region=valid&(head|hands|feet)
bad=region&(((head|feet|(hands&~bracelet))&notskin)|(bracelet&(blue|dark)))
print('bad texels',bad.sum(),'head',(bad&head).sum(),'hands',(bad&hands).sum(),'feet',(bad&feet).sum())
# fill: normalized convolution from skin texels in same region class, multi-scale
src=skin&region&~bad
out=tex.copy()
filled=np.zeros_like(bad)
for sig in [6,12,24,48,96,192,384]:
    k=int(sig*3)|1
    num=cv2.GaussianBlur(tex*src[...,None],(k,k),sig); den=cv2.GaussianBlur(src.astype(np.float32),(k,k),sig)
    ok=bad&~filled&(den>0.05)
    out[ok]=num[ok]/den[ok][...,None]; filled|=ok
rest=bad&~filled
for nm,rg in [('head',head),('hands',hands),('feet',feet)]:
    s_=src&rg
    if s_.any(): out[rest&rg]=tex[s_].mean(0)
filled|=rest
print('filled',filled.sum(),'of',bad.sum(),'fallback',rest.sum())
# keep fine scale texture: add high-frequency luminance from local skin neighbourhood by reusing original luminance variation (scaled)
detail=(L-cv2.GaussianBlur(L,(0,0),3))[...,None]*0.5
out[filled]=np.clip(out[filled]+detail[filled]*(~blue[filled])[...,None],0,255)
# soften edges
m8=filled.astype(np.float32); m8=cv2.GaussianBlur(m8,(0,0),1.5)[...,None]
final=tex*(1-m8)+out*m8
Image.fromarray(final.clip(0,255).astype(np.uint8)).save(OUTPNG)
mat.baseColorTexture=Image.open(OUTPNG); m.export(OUTGLB); print('written')
