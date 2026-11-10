"""เทียบสี texture ของโมเดลกับภาพต้นฉบับ 4 มุม (front/left/back/right)

    python tools/texture_compare.py <model.glb> <out.jpg> [โฟลเดอร์ภาพต้นฉบับ]

เรนเดอร์โมเดลแบบสี texture ล้วน (ไม่มีแสง) ให้ขนาด/ตำแหน่งตรงภาพต้นฉบับ แล้ววัด ΔE (CIELAB)
ต่อพิกเซลและสีเฉลี่ยแยกส่วน ผิว/เสื้อ/ยีนส์ เขียนภาพเทียบ + <out>.txt + <out>.json (ใช้กับ texture_color_match.py)
"""
import trimesh, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import sys
GLB=sys.argv[1]; OUT=sys.argv[2]
import os
SRC=(sys.argv[3] if len(sys.argv)>3 else os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','input','photogrammetry-whitebg-fix'))+os.sep
m=trimesh.load(GLB,force='mesh')
mat=m.visual.material
tex=np.asarray((mat.baseColorTexture if getattr(mat,'baseColorTexture',None) is not None else mat.image).convert('RGB'))
uv=m.visual.uv; H,W=tex.shape[:2]
fuv=uv[m.faces].mean(1); fcol=tex[np.clip(((1-fuv[:,1])*H).astype(int),0,H-1),np.clip((fuv[:,0]*W).astype(int),0,W-1)]
V=m.vertices; F=m.faces; N=m.face_normals
def Ry(a):a=np.radians(a);return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
def mask_of(rgb):
    mx=rgb.max(2).astype(int); ch=mx-rgb.min(2); pale=(mx>150)&(ch<32)
    lab,_=ndimage.label(pale); edge=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])))-{0}
    fg=~np.isin(lab,list(edge)); fg=ndimage.binary_opening(fg,iterations=2)
    l,n=ndimage.label(fg); sz=ndimage.sum(fg,l,range(1,n+1)); return l==(np.argmax(sz)+1)
def lab(rgb):
    c=rgb/255.0; c=np.where(c>0.04045,((c+0.055)/1.055)**2.4,c/12.92)
    M=np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]]); xyz=c@M.T/np.array([0.95047,1,1.08883])
    f=np.where(xyz>0.008856,np.cbrt(xyz),7.787*xyz+16/116)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)
views={'front':0,'left':270,'back':180,'right':90}
SC=0.5
rows=[]; report=[]; STATS={}
for v,yaw in views.items():
    org=Image.open(SRC+v+'.jpeg').convert('RGB'); org=org.resize((int(org.width*SC),int(org.height*SC)),Image.LANCZOS)
    o=np.asarray(org); om=mask_of(o)
    ys=np.where(om.any(1))[0]; top,bot=ys.min(),ys.max()
    P=V@Ry(yaw).T; s=(bot-top)/np.ptp(P[:,1]); 
    vy=lambda y: top+(P[:,1].max()-y)*s
    # horizontal: torso band 38-52%
    r0,r1=int(top+0.38*(bot-top)),int(top+0.52*(bot-top))
    cx_img=np.nonzero(om[r0:r1])[1].mean()
    yb_hi=P[:,1].max()-0.38*np.ptp(P[:,1]); yb_lo=P[:,1].max()-0.52*np.ptp(P[:,1])
    selv=(P[:,1]<yb_hi)&(P[:,1]>yb_lo); cu=P[selv,0].mean()
    xy=np.c_[cx_img+(P[:,0]-cu)*s, top+(P[:,1].max()-P[:,1])*s]
    n=N@Ry(yaw).T; z=P[F].mean(1)[:,2]
    img=Image.new('RGB',org.size,(255,255,255)); mk=Image.new('L',org.size,0)
    d=ImageDraw.Draw(img); dm=ImageDraw.Draw(mk)
    for i in np.argsort(z):
        if n[i,2]>0:
            pts=[tuple(p) for p in xy[F[i]]]; d.polygon(pts,fill=tuple(int(c) for c in fcol[i])); dm.polygon(pts,fill=255)
    r=np.asarray(img); rm=np.asarray(mk)>0
    both=om&rm
    dE=np.linalg.norm(lab(o.astype(float))-lab(r.astype(float)),axis=-1)
    # regions from ORIGINAL colors
    R,G,B=[o[...,i].astype(int) for i in range(3)]
    skin=(R>B+25); cloth=(B>R+10)
    hem=int(top+0.505*(bot-top))  # shirt/jeans split approx
    yy=np.arange(o.shape[0])[:,None]
    regs={'skin/scales':skin,'shirt':cloth&(yy<hem),'jeans':cloth&(yy>=hem)}
    line=[v, f'overlap {100*both.sum()/om.sum():.0f}%', f'mean dE {dE[both].mean():.1f}']
    for k,rg in regs.items():
        mm=both&rg
        if mm.sum()<100: continue
        STATS.setdefault(v,{})[k]=dict(orig=lab(o[mm].astype(float)).mean(0).tolist(),model=lab(r[mm].astype(float)).mean(0).tolist())
        line.append(f'{k}: orig {tuple(o[mm].mean(0).astype(int))} model {tuple(r[mm].mean(0).astype(int))} dE {dE[mm].mean():.1f}')
    report.append(' | '.join(line)); print(report[-1])
    heat=np.zeros(o.shape,np.uint8)+255
    hv=np.clip(dE/30,0,1); heat[both]=(np.stack([255*hv,255*(1-hv)*0.8,40*np.ones_like(hv)],-1)[both]).astype(np.uint8)
    rows.append((org,img,Image.fromarray(heat)))
w,h=rows[0][0].size
cw=int(w*0.55); x0=(w-cw)//2
S=Image.new('RGB',(cw*4,h*3),'white')
for k,(a,b,c) in enumerate(rows):
    for j,im in enumerate((a,b,c)): S.paste(im.crop((x0,0,x0+cw,h)),(k*cw,j*h))
dd=ImageDraw.Draw(S)
for k,v in enumerate(views): dd.text((k*cw+8,8),v,fill='black')
for j,t in enumerate(['original','model (unlit texture)','color diff dE (green=close, red=far)']): dd.text((8,j*h+24),t,fill='red')
S.save(OUT,quality=90)
open(OUT+'.txt','w').write('\n'.join(report))
import json; json.dump(STATS,open(OUT+'.json','w'))
