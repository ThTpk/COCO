"""ปรับสี texture ให้สีเฉลี่ยแต่ละส่วน (ผิว เสื้อ ยีนส์) ตรงกับภาพต้นฉบับแยกตามมุม

    python tools/texture_color_match.py <model.glb> <stats.json จาก texture_compare> <out.png> <out.glb>

ค่าชดเชย Lab ของแต่ละมุมถูกผสมตามทิศของนอร์มอล ผิวด้านข้างจึงได้ค่าของภาพข้าง ด้านหน้าได้ของภาพหน้า
รันซ้ำ (compare -> match) 2 รอบจะลู่เข้าเหลือ ΔE ของสีเฉลี่ยต่ำกว่า 1
"""
import trimesh, numpy as np, json, sys
from PIL import Image, ImageDraw, ImageFilter
GLB=sys.argv[1]; STATS=json.load(open(sys.argv[2])); OUTPNG=sys.argv[3]; OUTGLB=sys.argv[4]
m=trimesh.load(GLB,force='mesh'); mat=m.visual.material
img=(mat.baseColorTexture if getattr(mat,'baseColorTexture',None) is not None else mat.image).convert('RGB')
tex=np.asarray(img).astype(float); H,W=tex.shape[:2]
uv=m.visual.uv; F=m.faces; N=m.face_normals; C=m.vertices[F].mean(1)
# face-id raster in UV space (2048) -> upsample
R=2048
fid=Image.new('RGB',(R,R),(0,0,0)); d=ImageDraw.Draw(fid)
P=np.c_[uv[:,0]*R,(1-uv[:,1])*R]
for i in range(len(F)):
    k=i+1; d.polygon([tuple(P[j]) for j in F[i]],fill=(k&255,(k>>8)&255,(k>>16)&255))
a=np.asarray(fid).astype(np.int64); ids=a[...,0]+(a[...,1]<<8)+(a[...,2]<<16)-1
ids=np.asarray(Image.fromarray(ids.astype(np.int32)).resize((W,H),Image.NEAREST)).astype(np.int64)
print('raster done, covered',(ids>=0).mean())
def lab(rgb):
    c=rgb/255.0; c=np.where(c>0.04045,((c+0.055)/1.055)**2.4,c/12.92)
    M=np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]]); xyz=c@M.T/np.array([0.95047,1,1.08883])
    f=np.where(xyz>0.008856,np.cbrt(xyz),7.787*xyz+16/116)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)
def unlab(L):
    fy=(L[...,0]+16)/116; fx=fy+L[...,1]/500; fz=fy-L[...,2]/200
    f=np.stack([fx,fy,fz],-1); xyz=np.where(f>0.2069,f**3,(f-16/116)/7.787)*np.array([0.95047,1,1.08883])
    Mi=np.linalg.inv(np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]]))
    c=xyz@Mi.T; c=np.where(c>0.0031308,1.055*np.clip(c,0,None)**(1/2.4)-0.055,12.92*c)
    return np.clip(c*255,0,255)
# per-view per-class offsets
dirs={'front':np.array([0,0,1.]),'back':np.array([0,0,-1.]),'left':np.array([1.,0,0]),'right':np.array([-1.,0,0])}
classes=['skin/scales','shirt','jeans']
off={c:{v:np.array(STATS[v][c]['orig'])-np.array(STATS[v][c]['model']) for v in dirs if c in STATS[v]} for c in classes}
for c in classes: print(c,{v:np.round(o,1).tolist() for v,o in off[c].items()})
# face class from texel colour at face centroid
fuv=uv[F].mean(1); fc=tex[np.clip(((1-fuv[:,1])*H).astype(int),0,H-1),np.clip((fuv[:,0]*W).astype(int),0,W-1)]
# per face offset = blend of view offsets by normal
wsum=np.zeros(len(F)); 
Wd={v:np.clip(N@dv,0,None)**2+1e-3 for v,dv in dirs.items()}
tot=sum(Wd.values())
face_off={}
for c in classes:
    o=np.zeros((len(F),3))
    for v in dirs: o+=Wd[v][:,None]*off[c][v][None,:]
    face_off[c]=o/tot[:,None]
# texel class masks
Lt=lab(tex); R_,G_,B_=tex[...,0],tex[...,1],tex[...,2]
skin=(R_>B_+20); cloth=(B_>R_+5)
valid=ids>=0
yid=np.where(valid,C[np.clip(ids,0,None),1],0)
HEM=-0.02
cls={'skin/scales':skin&valid,'shirt':cloth&valid&(yid>HEM),'jeans':cloth&valid&(yid<=HEM)}
out=Lt.copy()
for c,mask in cls.items():
    fo=face_off[c][np.clip(ids,0,None)]
    out[mask]+=fo[mask]
res=unlab(out)
# feather: blend corrected/uncorrected softly at class edges via blurred alpha
alpha=np.zeros((H,W))
for mask in cls.values(): alpha[mask]=1
alpha=np.asarray(Image.fromarray((alpha*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))).astype(float)/255
final=tex*(1-alpha[...,None])+res*alpha[...,None]
Image.fromarray(final.astype(np.uint8)).save(OUTPNG)
mat.baseColorTexture=Image.open(OUTPNG)
m.export(OUTGLB); print('written')
