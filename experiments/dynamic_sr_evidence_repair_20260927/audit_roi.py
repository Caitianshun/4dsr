"""Additional raw fixed-ROI diagnostics; no optimization, all points unchanged."""
import time
from context import *
from renderer import render_model
from fixed_model import load_baked,render_baked
from evaluate_probes import scalar_error
from footprint_reference import audit as audit_footprint

def main():
    out=OUT/'roi_fit_v1';out.mkdir(exist_ok=False);torch.set_num_threads(4);p=paths();m=load_manifest(p['manifest']);root=Path(m['_root']);rois=read(p['roi'])['regions_by_camera_xyxy_exclusive'];obs={(o['camera_id'],o['frame_index']):o for o in m['observations']};teachers={(e['camera'],e['frame']):e for e in read(p['teacher'])['entries']};started=time.time();rows=[]
    reference_dir=OUT/'footprint_reference_v1';reference_dir.mkdir(exist_ok=False);references=[]
    with torch.inference_mode():
        for mode in ['Shared40','Baked40','references']:
            jobs=read(OUT/'probe_evaluation'/mode/'complete.json')['methods']
            for name,item in jobs.items():
                ck=item['checkpoint'];assert sha256(ck)==item['checkpoint_sha256'];baked=mode=='Baked40';model=load_baked(ck) if baked else motion.load_model(ck,m);fn=render_baked if baked else render_model
                for r in item['rows']:
                    cam,frame=r['camera'],r['frame']
                    if cam not in rois:continue
                    o=obs[cam,frame];raw=fn(model,evalmod.render_camera(m,o,0))['render'];lr=image_tensor(root/o['lr_path']);low=downsample(raw,lr.shape[-2:]);h=detail.highpass(raw,lr.shape[-2:]);hr=image_tensor(root/o['hr_path']);hh=detail.highpass(hr,lr.shape[-2:]);entry=teachers.get((cam,frame));target=image_tensor(root/entry['relative_path']) if entry else None;ht=detail.highpass(target,lr.shape[-2:]) if entry else None
                    fp=next(x for x in item['footprints'] if x['camera']==cam and x['frame']==frame)
                    ref=audit_footprint(model,baked,evalmod.render_camera(m,o,0),fp['path'],reference_dir/f'{name}_{cam}_{frame:04d}.npz',rois[cam]);references.append(dict(run_id=name,camera=cam,frame=frame,**ref))
                    for region,(x0,y0,x1,y1) in rois[cam].items():
                        sl=(slice(None),slice(y0,y1),slice(x0,x1));small=(slice(None),slice(y0//4,y1//4),slice(x0//4,x1//4));row={k:r[k] for k in ['run_id','parent_run_id','checkpoint_sha256','renderer_id','metric_version','camera','frame','split','source_path']};row['region']=region;row['lr']=scalar_error(low[small],lr[small]);row['hr_H']=scalar_error(h[sl],hh[sl])
                        if target is not None:row.update(teacher_rgb=scalar_error(raw[sl],target[sl]),teacher_H=scalar_error(h[sl],ht[sl]),teacher_hr_H=scalar_error(ht[sl],hh[sl]))
                        rows.append(row)
                del model;torch.cuda.empty_cache()
    write_json(reference_dir/'complete.json',dict(status='completed',rows=references,parameter_updates=0))
    write_json(out/'complete.json',dict(status='completed',rows=rows,parameter_updates=0,seconds=time.time()-started,gpu=torch.cuda.get_device_name(),source_sha256=sha256(__file__),note='HR-defined frozen ROI used for independent evaluation only; H calculated full-image before cropping. LR ROI coordinates are exactly divided by4, no resampling of predictions.'))
if __name__=='__main__':main()
