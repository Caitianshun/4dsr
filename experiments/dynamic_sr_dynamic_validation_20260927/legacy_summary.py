"""Read-only 63 raw renders. Existing quality rows remain the authoritative HR values."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/dynamic_sr_covariance_probe_20260927'))
from common_cov import *
OUT=ROOT/'output/dynamic_sr_dynamic_validation_20260927'

def main():
    torch.set_num_threads(4);started=time.time()
    old=ROOT/'output/dynamic_sr_covariance_probe_20260927';p=load_protocol(old/'protocol.json')
    idx=read(old/'checkpoint_index.json');items={'AB600':idx['parent'],**{x['arm']+'600':x for x in idx['checkpoints'] if x['step']==600}}
    old_files={'AB600':old/'parent_hr/complete.json','C600':old/'r1_C/hr600/complete.json','S600':old/'r1_S/hr600/complete.json'}
    for k,f in list(old_files.items()):
        if not f.exists():
            fs=list((old/('r1_'+k[0])).glob('*/complete.json')) if k!='AB600' else []
            old_files[k]=next(x for x in fs if read(x).get('HR') and read(x).get('step')==600)
    source_rows={k:read(f)['rows'] for k,f in old_files.items()}
    _,data,masks=load_data(p,include_dev=True,hr=True);rows=[];roirows=[]
    for label,entry in items.items():
        assert sha(entry['path'])==entry['sha256'];model=fixed.load_baked(entry['path']);before=model_hash(model)
        with torch.no_grad():
            for c,r in data.items():
                pred=render(model,r['camera']);hp=detail.highpass(pred,(252,336));hy=detail.highpass(r['hr'],(252,336))
                q={x['metric']:x['value'] for x in source_rows[label] if x['camera']==c and x['region']=='full' and x['reference']=='HR'}
                oldh=next((x['value'] for x in source_rows[label] if x['camera']==c and x['region']=='full' and x['reference']=='teacher' and x['metric']=='H_L1'),None)
                vals=metrics_from_render(pred,r);newh=vals.get('H');
                if oldh is not None:assert abs(newh-oldh)<1e-7,(label,c,newh,oldh)
                row=dict(endpoint=label,camera=c,frame=40,checkpoint_sha256=entry['sha256'],PSNR=q['psnr'],SSIM=q['ssim'],LPIPS=q['lpips_alex'],LR_L1=vals['LR'],teacher_H=oldh,H_HR=float((hp-hy).abs().double().mean()),teacher_to_HR_H=float((r['teacher_H']-hy).abs().double().mean()) if 'teacher' in r else None,quality_source=str(old_files[label]),new_columns='H_HR; teacher_to_HR_H; raw-render LR verification')
                rows.append(row)
                for name,mask in regions(p,c,pred,masks).items():
                    if name not in p['rois'].get(c,{}):continue
                    roirows.append(dict(endpoint=label,camera=c,region=name,H_HR=float((hp-hy).abs()[:,mask].double().mean()),teacher_H=float((hp-r['teacher_H']).abs()[:,mask].double().mean()),teacher_to_HR_H=float((r['teacher_H']-hy).abs()[:,mask].double().mean())))
        assert model_hash(model)==before
        del model;torch.cuda.empty_cache()
    fields=['PSNR','SSIM','LPIPS','LR_L1','teacher_H','H_HR','teacher_to_HR_H'];deltas=[]
    for c in data:
        a=next(r for r in rows if r['endpoint']=='S600' and r['camera']==c);b=next(r for r in rows if r['endpoint']=='C600' and r['camera']==c)
        x=dict(camera=c)
        for k in fields:
            x[k+'_S']=a[k];x[k+'_C']=b[k];x[k+'_delta']=None if a[k] is None else a[k]-b[k]
            if k!='PSNR':x[k+'_relative']=None if b[k] is None or abs(b[k])<1e-6 else (a[k]-b[k])/b[k]
        deltas.append(x)
    remaining=[x for x in deltas if x['camera'] not in ['cam00','cam01','cam02','cam03']]
    summary={k:dict(mean=float(np.mean([x[k+'_delta'] for x in remaining])),median=float(np.median([x[k+'_delta'] for x in remaining])),worst_camera=(min if k in ['PSNR','SSIM'] else max)(remaining,key=lambda x:x[k+'_delta'])['camera']) for k in fields}
    summary.update(psnr_loss_over_020=[r['camera'] for r in remaining if r['PSNR_delta']<-.2],lpips_harm_over_005=[r['camera'] for r in remaining if r['LPIPS_relative']>.05])
    csvwrite(OUT/'legacy_per_view.csv',rows);csvwrite(OUT/'legacy_S_minus_C.csv',deltas);csvwrite(OUT/'legacy_roi.csv',roirows)
    write(OUT/'legacy_summary.json',dict(status='completed',rows=rows,deltas=deltas,remaining17=summary,rgb_forwards=COUNTS['rgb_forwards'],parameter_updates=0,seconds=time.time()-started,gpu=torch.cuda.get_device_name(),old_decision_sha256=sha(old/'decision.json'),old_protocol_sha256=sha(old/'protocol.json'),source_sha256=sources(),script_sha256=sha(__file__),note='Old decision remains fail. These are development diagnostics; no new static qualification. H computed on raw floats before any display clamp.'))

if __name__=='__main__':main()
