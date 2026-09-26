"""Join privileged region masks only after legal one-step probes have finished."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def corr(a,b):
    a=a.astype(np.float64).ravel();b=b.astype(np.float64).ravel()
    return None if len(a)<16 or a.var()<1e-8 or b.var()<1e-8 else float(np.corrcoef(a,b)[0,1])


def main():
    p=argparse.ArgumentParser()
    for k in ['probe','attributes','reference-eval','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);j=read(a.reference_eval/'metrics.json')
    masks={}
    for c,info in j['evaluation_caches'].items():
        path=Path(info['path'])/'dynamic_mask.png';assert sha(path)==info['dynamic_mask_sha256'];masks[c]=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)>0
    probe=read(a.probe/'complete.json');rows=[]
    for r in probe['rows']:
        path=a.probe/r['maps'];assert sha(path)==r['maps_sha256'];maps=np.load(path);mask=masks[r['camera']].astype(np.uint8)
        outer=np.zeros(mask.shape,bool);h,w=mask.shape;outer[:h//10]=True;outer[-h//10:]=True;outer[:,:w//10]=True;outer[:,-w//10:]=True
        inside=cv2.erode(mask,np.ones((11,11),np.uint8)).astype(bool);outside=~cv2.dilate(mask,np.ones((11,11),np.uint8)).astype(bool)
        regions=dict(change_boundary=(~inside)&(~outside)&(~outer),dynamic_interior=inside&(~outer),static_interior=outside&(~outer),outer_background=outside&outer)
        for name,region in regions.items():
            rows.append(dict(camera=r['camera'],frame=r['frame'],region=name,pixels=int(region.sum()),response={k:float(v[region].mean()) if region.any() else None for k,v in maps.items()}))
    groups={}
    for group in probe['rows'][0]['gradients']:
        rr=[r['gradients'][group] for r in probe['rows']];valid=[x['cosine'] for x in rr if x['cosine'] is not None]
        groups[group]=dict(median_cosine=float(np.median(valid)) if valid else None,negative_cosine_observations=sum(x<0 for x in valid),valid_observations=len(valid),median_element_conflict_fraction=float(np.median([x['sign_conflict_fraction'] for x in rr if x['sign_conflict_fraction'] is not None])))
    regional={}
    for name in ['change_boundary','dynamic_interior','static_interior','outer_background']:
        rr=[r for r in rows if r['region']==name];regional[name]={arm:float(np.mean([r['response'][arm] for r in rr if r['response'][arm] is not None])) for arm in rr[0]['response']}
    attrs=read(a.attributes/'complete.json');comparisons=[]
    for r in attrs['rows']:
        path=a.attributes/r['path'];assert sha(path)==r['sha256'];d=np.load(path)
        dyn=cv2.resize(masks[r['camera']].astype(np.uint8),(336,252),interpolation=cv2.INTER_NEAREST)>0
        valid=(d['U_alpha']>.9)&(d['U_expected_z']>0)
        for kind in ['correct','spatial_shift48','wrong_time40']:
            def other(key):
                image=d[('O_wrong_time' if kind=='wrong_time40' else 'O')+'_'+key]
                return np.roll(image,48,axis=-1) if kind=='spatial_shift48' else image
            common=valid&(other('alpha')>.9)&(other('expected_z')>0)
            for reg,mask in [('full',common),('dynamic',common&dyn),('static',common&(~dyn))]:
                values={}
                for key in ['alpha','expected_z','variance_z','rgb']:
                    x=d['U_'+key];y=other(key)
                    x=x[:,mask] if key=='rgb' else x[mask];y=y[:,mask] if key=='rgb' else y[mask]
                    values[key]=dict(pearson=corr(x,y),mean_absolute_difference=float(np.abs(x-y).mean()) if len(x) else None)
                comparisons.append(dict(camera=r['camera'],frame=r['frame'],pairing=kind,region=reg,common_coverage=float(mask.mean()),metrics=values))
    aggregate={}
    for kind in ['correct','spatial_shift48','wrong_time40']:
        aggregate[kind]={}
        for reg in ['full','dynamic','static']:
            rr=[r for r in comparisons if r['pairing']==kind and r['region']==reg]
            aggregate[kind][reg]={key:{metric:float(np.mean([r['metrics'][key][metric] for r in rr if r['metrics'][key][metric] is not None])) if any(r['metrics'][key][metric] is not None for r in rr) else None for metric in ['pearson','mean_absolute_difference']} for key in ['alpha','expected_z','variance_z','rgb']}
    (a.out/'complete.json').write_text(json.dumps(dict(status='completed',gradient_groups=groups,probe_regions=regional,probe_region_rows=rows,attribute_comparison=aggregate,attribute_rows=comparisons,attribute_point_summaries=attrs['rows'],source_sha256=sha(__file__),information_boundary='HR temporal masks read here only after legal probe has completed. They are change-region proxies, not semantic boundaries; no outputs flow back to RGB or future depth training.',interpretation='Gradient norms have incompatible units and do not give error percentages. Shared U/O topology establishes index identity, not unique geometry. Wrong-time background is not necessarily a negative; report dynamic separately.'),indent=2))


if __name__=='__main__':main()
