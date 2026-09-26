"""Full float endpoint errors, effective state drift, and 2x2 state sensitivity."""
import argparse
import importlib.util
import math
import time
import numpy as np
from PIL import Image
from shared import *
_ev=evaluator();render_camera,legacy=_ev.render_camera,_ev.legacy

def regions(shape,dynamic):
    h,w=shape;yy,xx=np.indices(shape)
    border=(xx<w*.1)|(xx>=w*.9)|(yy<h*.1)|(yy>=h*.9)
    return dict(full=np.ones(shape,bool),border10=border,center80=~border,
        top10=yy<h*.1,bottom10=yy>=h*.9,left10=xx<w*.1,right10=xx>=w*.9,
        dynamic=dynamic,static=~dynamic)

def stat(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(n=0)

def project(xyz,cal):
    points=np.c_[xyz,np.ones(len(xyz))]@np.asarray(cal['w2c'])[:3].T
    q=points@np.asarray(cal['K_lr']).T
    xy=q[:,:2]/np.where(np.abs(q[:,2:])>1e-10,q[:,2:],np.nan)
    return xy,points[:,2]>0

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT/'p0/drift');a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic()
    p=paths();m=load_manifest(p['manifest']);obs={(o['camera_id'],o['frame_index']):o for o in m['observations']}
    models={};states={};results={};maps={};masks={};rows=[];state_rows=[]
    for label in ['U6000','U18000','U40000']:
        model=load_model(p['methods'][label]['checkpoint'],m);model.g._deformation.eval();models[label]=model
        states[label]={}
        for f in [0,40,80,118]:
            states[label][f]=effective_state(model,f/300)
            np.savez_compressed(a.out/f'{label}_state_{f:04d}.npz',**{k:v.cpu().numpy() for k,v in states[label][f].items()},base_count=len(model.g._xyz))
        results[label]={}
        with torch.inference_mode():
            for camera,split in [('cam00','test'),('cam01','dev')]:
                old=read(Path(p['methods'][label]['evaluations'][split])/'metrics.json')
                maskpath=Path(old['evaluation_caches'][camera]['path'])/'dynamic_mask.png'
                assert sha256(maskpath)==old['evaluation_caches'][camera]['dynamic_mask_sha256']
                masks[camera]=regions((1008,1344),np.array(Image.open(maskpath))>0)
                error_dir=a.out/'errors'/label/camera;error_dir.mkdir(parents=True)
                total=np.zeros((1008,1344),np.float64);parity=[]
                for f in range(0,120,2):
                    o=obs[camera,f];cam=render_camera(m,o,f)
                    raw=render_model(model,cam)['render'];pred=legacy.image_array(raw)
                    gtpath=Path(m['_root'])/o['hr_path'];assert sha256(gtpath)==o['hr_sha256']
                    gt=legacy.read_rgb(gtpath);err=((pred.astype(np.float64)-gt)**2).mean(-1)
                    np.save(error_dir/f'{f:04d}.npy',err.astype(np.float32));total+=err
                    region={n:dict(sse_rgb=float(err[mask].sum()*3),mse=float(err[mask].mean()),psnr=legacy.metric_psnr(float(err[mask].mean())),pixels=int(mask.sum())) for n,mask in masks[camera].items()}
                    rows.append(dict(label=label,camera=camera,frame=f,regions=region))
                    prev=next(r for r in old['rows'] if r['frame_index']==f)
                    parity.append(abs(region['full']['psnr']-prev['spatial']['full']['psnr']))
                    if f in [0,38,40,42,78,80,82,118]:legacy.write_rgb(a.out/'predictions'/label/camera/f'{f:04d}.png',pred)
                assert max(parity)<.001,(label,camera,max(parity))
                maps[label,camera]=total/60
                np.save(a.out/f'{label}_{camera}_mean_error.npy',(total/60).astype(np.float32))
                results[label][camera]=dict(existing_float_psnr_maxabs=max(parity),regions={n:dict(sse_rgb=float(total[mask].sum()*3),mse=float(total[mask].mean()/60),psnr_frame_mean=float(np.mean([r['regions'][n]['psnr'] for r in rows if r['label']==label and r['camera']==camera])),fraction_of_total_sse=float(total[mask].sum()/total.sum())) for n,mask in masks[camera].items()})
                print(label,camera,'complete',flush=True)
    deltas=[]
    for camera in ['cam00','cam01']:
        for early,late in [('U6000','U18000'),('U18000','U40000'),('U6000','U40000')]:
            totals={n:dict(net=0.,positive=0.,negative_magnitude=0.) for n in masks[camera]};per_frame=[]
            positive=np.zeros((1008,1344),np.float64);negative=positive.copy()
            for f in range(0,120,2):
                e=np.load(a.out/'errors'/early/camera/f'{f:04d}.npy').astype(np.float64)
                l=np.load(a.out/'errors'/late/camera/f'{f:04d}.npy').astype(np.float64);delta=l-e
                pos=np.maximum(delta,0);neg=np.maximum(-delta,0);positive+=pos;negative+=neg
                fr={}
                for n,mask in masks[camera].items():
                    vals=dict(net=float(delta[mask].sum()*3),positive=float(pos[mask].sum()*3),negative_magnitude=float(neg[mask].sum()*3))
                    for k,v in vals.items():totals[n][k]+=v
                    fr[n]=vals
                per_frame.append(dict(frame=f,regions=fr))
            for n,v in totals.items():
                v['positive_fraction']=v['positive']/totals['full']['positive']
                v['negative_fraction']=v['negative_magnitude']/totals['full']['negative_magnitude']
            np.savez_compressed(a.out/f'{camera}_{early}_to_{late}_delta.npz',signed=(positive-negative)/60,positive=positive/60,negative_magnitude=negative/60)
            deltas.append(dict(camera=camera,early=early,late=late,regions=totals,frames=per_frame))
    # Geometry states refer to exact concatenated render order, never parent_ids as base indices.
    for label in states:
        model=models[label];nb=len(model.g._xyz);aabb=np.stack([v.cpu().numpy() for v in model.g._deformation.deformation_net.get_aabb])
        lo,hi=aabb.min(0),aabb.max(0)
        for f,state in states[label].items():
            cur={k:v.cpu().numpy() for k,v in state.items()};ref={k:v.cpu().numpy() for k,v in states['U6000'][f].items()}
            temporal={k:v.cpu().numpy() for k,v in states[label][0].items()}
            inside=np.all((cur['xyz']>=lo)&(cur['xyz']<=hi),axis=1)
            train_frustum=np.zeros(len(inside),bool);projections={}
            for camera in m['cameras']:
                xy,front=project(cur['xyz'],m['cameras'][camera]);oldxy,oldfront=project(ref['xyz'],m['cameras'][camera])
                valid=front&oldfront&np.isfinite(xy).all(1)&np.isfinite(oldxy).all(1)
                if camera in m['splits']['train']:train_frustum|=front&(xy[:,0]>=0)&(xy[:,0]<336)&(xy[:,1]>=0)&(xy[:,1]<252)
                if camera in ['cam00','cam01','cam02']:projections[camera]=stat(np.linalg.norm(xy[valid]-oldxy[valid],axis=1))
            dist=np.linalg.norm(cur['xyz']-ref['xyz'],axis=1)
            e0=np.linalg.eigvalsh(ref['cov'].astype(np.float64));e1=np.linalg.eigvalsh(cur['cov'].astype(np.float64))
            groups=dict(all=np.ones(len(inside),bool),base=np.arange(len(inside))<nb,child=np.arange(len(inside))>=nb,aabb_inside=inside,aabb_outside=~inside,train_frustum_union=train_frustum,outside_train_frustum_union=~train_frustum)
            state_rows.append(dict(label=label,frame=f,projections_lr_pixels=projections,groups={n:dict(world_displacement=stat(dist[mask]),
                covariance_eigenvalue_absolute_delta=stat(np.abs(e1-e0)[mask].ravel()),alpha_absolute_delta=stat(np.abs(cur['opacity']-ref['opacity'])[mask].ravel()),
                sh_absolute_delta=stat(np.abs(cur['sh']-ref['sh'])[mask].ravel()),within_checkpoint_world_from_frame0=stat(np.linalg.norm(cur['xyz']-temporal['xyz'],axis=1)[mask])) for n,mask in groups.items()}))
    swap=[]
    with torch.inference_mode():
        for camera in ['cam00','cam01','cam02','cam06','cam12','cam18']:
            for f in [0,40,80,118]:
                o=obs[camera,f];cam=render_camera(m,o,f);gt=legacy.read_rgb(Path(m['_root'])/o['hr_path'])
                for geo in ['U6000','U40000']:
                    for color in ['U6000','U40000']:
                        s=states[geo][f];raw=rasterize(models[geo].g,cam,s['xyz'],s['cov'],s['opacity'],states[color][f]['sh'])['render']
                        parity=float((raw-render_model(models[geo],cam)['render']).abs().max()) if geo==color else None
                        if parity is not None:assert parity==0
                        pred=legacy.image_array(raw);mse=float(np.mean((pred.astype(np.float64)-gt)**2))
                        swap.append(dict(camera=camera,frame=f,gamma=geo,color=color,psnr=legacy.metric_psnr(mse),diagonal_maxabs=parity))
                        if camera in ['cam00','cam01','cam02'] and f in [40,80]:legacy.write_rgb(a.out/'swaps'/f'{camera}_{f:04d}_{geo}_{color}.png',pred)
    write_json(a.out/'summary.json',dict(status='completed',endpoint_errors=results,error_rows=rows,deltas=deltas,state_drift=state_rows,state_swaps=swap,
        precision='float render; RGB eval clamp identical to original; no PNG quantization for errors',
        evidence_limit='Swaps are sensitivity diagnostics, not additive causal contributions. Frustum/AABB membership is model projection, not verified real visibility. Alpha is activated Gaussian opacity, not cumulative rendered alpha. Depth not exported.',
        seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),parameter_updates=0))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for row,camera in enumerate(['cam00','cam01']):
        z=np.load(a.out/f'{camera}_U6000_to_U40000_delta.npz')
        for j,k in enumerate(['signed','positive','negative_magnitude']):
            im=axes[row,j].imshow(z[k],cmap='coolwarm' if k=='signed' else 'inferno',vmin=-.003 if k=='signed' else 0,vmax=.003)
            axes[row,j].set_title(camera+' '+k);axes[row,j].axis('off');fig.colorbar(im,ax=axes[row,j],shrink=.55)
    fig.savefig(a.out/'delta_maps.png',dpi=130);plt.close(fig)

if __name__=='__main__':main()
