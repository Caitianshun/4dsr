"""Full-image signed H before ROI extraction; separately gated float HR evaluation."""
import argparse
from common_cov import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--arm',required=True);ap.add_argument('--repeat',type=int,required=True);ap.add_argument('--step',type=int,required=True);ap.add_argument('--hr',action='store_true');a=ap.parse_args();p=load_protocol(a.protocol);a.out.mkdir(parents=True,exist_ok=False);tick=time.time();torch.set_num_threads(4)
    if a.hr:assert read(Path(p['_root'])/'train_decision.json')['status']=='frozen'
    reads,blocked=guard_images(p,privileged=a.hr);_,data,masks=load_data(p,include_dev=a.hr,hr=a.hr);m=fixed.load_baked(a.checkpoint);before=model_hash(m);rows=[];values={};cksha=sha(a.checkpoint)
    if a.hr:
        import lpips
        metric=lpips.LPIPS(net='alex',spatial=False).cuda().eval().requires_grad_(False)
    with torch.no_grad():
        for c,r in data.items():
            pred=render(m,r['camera']);h=detail.highpass(pred,(252,336));areas=regions(p,c,pred,masks);values[c]=metrics_from_render(pred,r,masks.get(c))
            identity=dict(stage='A',repeat=a.repeat,arm=a.arm,step=a.step,camera=c,frame=40,checkpoint_sha256=cksha)
            def record(reg,ref,key,val,pixels):rows.append(dict(**identity,region=reg,reference=ref,metric=key,value=float(val),pixels=int(pixels)))
            record('full','LR','L1',values[c]['LR'],252*336)
            for reg,mask in areas.items():
                pixels=1344*1008 if mask is None else int(mask.sum())
                if 'teacher' in r:
                    for name,error in [('RGB_L1',(pred-r['teacher']).abs()),('H_L1',(h-r['teacher_H']).abs())]:
                        val=float((error if mask is None else error[:,mask]).double().mean());record(reg,'teacher',name,val,pixels);values[c][reg+'_'+name]=val
                if a.hr and (reg=='full' or reg in p['rois'].get(c,{})):
                    arr=evalmod.legacy.image_array(pred);gt=evalmod.legacy.image_array(r['hr'])
                    if reg!='full':
                        x0,y0,x1,y1=[4*v for v in p['rois'][c][reg]];arr=arr[y0:y1,x0:x1].copy();gt=gt[y0:y1,x0:x1].copy()
                    quality=evalmod.legacy.spatial_metrics(arr,gt,np.zeros(gt.shape[:2],bool),metric)['full']
                    for name in ['psnr','ssim','lpips_alex']:record(reg,'HR',name,quality[name],pixels)
            if a.hr and c in ['cam00','cam01','cam02','cam03']:
                from PIL import Image
                Image.fromarray((pred.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(a.out/f'{c}.png')
    assert before==model_hash(m);csvwrite(a.out/'metrics.csv',rows);write(a.out/'complete.json',dict(status='completed',stage='A',arm=a.arm,repeat=a.repeat,step=a.step,checkpoint=str(a.checkpoint),checkpoint_sha256=cksha,values=values,rows=rows,state_unchanged=True,seconds=time.time()-tick,counts=COUNTS,image_reads=reads,blocked=blocked,HR=a.hr,source_sha256=sources()))
if __name__=='__main__':main()
