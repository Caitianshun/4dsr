"""Fixed development frames and original ROI coordinates, with image-level traceability."""
from dv_common import *
from PIL import Image,ImageDraw,ImageFont
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)

def main():
    p=read(OUT/'protocol.json');roi=read(bound(p['roi']))['regions_by_camera_xyxy_exclusive'];manifest=read(bound(p['manifest']));data=bound(p['manifest']).parent
    index=read(OUT/'checkpoint_index.json')['checkpoints'];sets=sorted(set(v['repeat'] for v in index.values()));dest=OUT/'visuals';dest.mkdir(exist_ok=True);artifacts=[]
    def image(label,c,f):
        if label=='HR':path=data/f'hr/{c}/{f:04d}.png'
        elif label=='LR':path=data/f'lr/{c}/{f:04d}.png'
        else:path=OUT/'evaluation'/label/c/'predictions'/c/f'{f:04d}.png'
        im=Image.open(path).convert('RGB')
        if label=='LR':im=im.resize((1344,1008),Image.Resampling.BICUBIC)
        return im,path
    def sheet(name,columns,entries,size):
        # entries list of (camera, frame, ROI name, box); every crop comes from a fixed registry.
        cw,ch=size;canvas=Image.new('RGB',(len(columns)*cw,len(entries)*(ch+46)),(245,245,245));dr=ImageDraw.Draw(canvas);trace=[]
        for y,(c,f,r,box) in enumerate(entries):
            for x,label in enumerate(columns):
                im,path=image(label,c,f)
                if box:im=im.crop(box)
                im=im.resize(size,Image.Resampling.LANCZOS);canvas.paste(im,(x*cw,y*(ch+46)+46));dr.text((x*cw+4,y*(ch+46)+2),label,fill='black',font=FONT);dr.text((x*cw+4,y*(ch+46)+23),f'{c} f{f} {r}',fill='black',font=FONT)
                trace.append(dict(label=label,camera=c,frame=f,roi=r,box=box,image=str(path),sha256=sha(path)))
        path=dest/(name+'.png');canvas.save(path);artifacts.append(dict(path=str(path),sha256=sha(path),inputs=trace))
    for r in sets:
        cols=['HR','U6000',f'r{r}_J_joint',f'r{r}_A_sh',f'r{r}_S_cov']
        for c in ['cam00','cam01']:
            for f in [40,80]:
                sheet(f'r{r}_{c}_full_f{f}',cols,[(c,f,'full',None)],(448,336))
                if c in roi:sheet(f'r{r}_{c}_roi_f{f}',['LR',*cols],[(c,f,k,v) for k,v in roi[c].items()],(168,168))
            for fs in [[38,40,42],[78,80,82]]:
                if c in roi:box=roi[c]['hand_utensil_pan_reference'];name='hand'
                else:box=None;name='full'
                sheet(f'r{r}_{c}_adjacent_{fs[1]}',cols,[(c,f,name,box) for f in fs],(252,252) if box else (336,252))
    write(OUT/'visuals.json',dict(status='exported_not_yet_reviewed',fixed_frames=[40,80],adjacent_frames=[[38,40,42],[78,80,82]],cam01_roi='No registered ROI, use full frame',artifacts=artifacts,script_sha256=sha(__file__)))
if __name__=='__main__':main()
