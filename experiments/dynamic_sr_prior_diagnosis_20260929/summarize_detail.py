"""CPU-only paired region, temporal and teacher-fit summary after fixed endpoints."""
from dv_common import *
from summarize import csvwrite
from statistics import mean


def main():
    p=require_run_root(OUT)
    assert read(OUT/'final_integrity.json')['status']=='completed'
    regions=read(OUT/'fixed_regions.json')
    assert regions['status']=='completed'
    inputs=[];rows=[];pairs=[]
    for rep in ['1','2']:
        for arm in ['J1','T']:
            path=bound(p['historical_J1'][rep]['endpoint']) if arm=='J1' else OUT/'evaluation'/f'r{rep}_T/endpoint.json'
            inputs.append(entry(path));q=read(path)
            for scope,values in list(q['cameras'].items())+[('train76',q['train76'])]:
                rows.append(dict(repeat=rep,arm=arm,scope=scope,**values))
    for a in rows:
        if a['arm']!='T':continue
        b=next(b for b in rows if b['arm']=='J1' and b['repeat']==a['repeat'] and b['scope']==a['scope'])
        pairs.append(dict(repeat=a['repeat'],scope=a['scope'],**{'delta_'+k:a[k]-b[k] for k in a if k not in ['repeat','arm','scope'] and k in b}))
    average=[]
    for scope in ['cam00','cam01','train76']:
        selected=[r for r in pairs if r['scope']==scope]
        average.append(dict(scope=scope,**{k:mean(r[k] for r in selected) for k in selected[0] if k.startswith('delta_')}))
    region_average=[]
    for camera,region in sorted({(r['camera'],r['region']) for r in regions['paired_deltas']}):
        selected=[r for r in regions['paired_deltas'] if r['camera']==camera and r['region']==region]
        region_average.append(dict(camera=camera,region=region,paired_observations=len(selected),
            **{k:mean(r[k] for r in selected) for k in selected[0] if k.startswith('delta_')}))
    sizes={}
    for key,path in [('texture_cache',OUT/'legal/texture_maps'),('run1',OUT/'runs/r1_T'),('run2',OUT/'runs/r2_T')]:
        files=[f for f in path.rglob('*') if f.is_file()]
        sizes[key]=dict(bytes=sum(f.stat().st_size for f in files),files=len(files))
    checkpoint_bytes={label:(OUT/'runs'/label/'attempt_01/train/checkpoint_12000.pt').stat().st_size for label in ['r1_T','r2_T']}
    import torch
    def tensor_bytes(value):
        if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
        if isinstance(value,dict):return sum(tensor_bytes(x) for x in value.values())
        if isinstance(value,(tuple,list)):return sum(tensor_bytes(x) for x in value)
        return 0
    inference_payload={}
    for rep in ['1','2']:
        for arm in ['J1','T']:
            path=bound(p['historical_J1'][rep]['checkpoint']) if arm=='J1' else OUT/'runs'/f'r{rep}_T/attempt_01/train/checkpoint_12000.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            # GaussianModel.capture: xyz, deformation state, deformation table,
            # SH, scales, rotations and opacities; exclude densification/Adam.
            payload=checkpoint['model'][1:9]
            inference_payload[f'r{rep}_{arm}']=tensor_bytes(payload)+tensor_bytes(checkpoint['motion_refinement']['children'])
            del checkpoint
    for name,values in [('detail_quality.csv',rows),('detail_paired_deltas.csv',pairs),
                        ('detail_mean_deltas.csv',average),('fixed_region_mean_deltas.csv',region_average)]:
        csvwrite(OUT/name,values)
    result=dict(status='completed',rows=rows,paired_deltas=pairs,mean_deltas=average,
        fixed_region_mean_deltas=region_average,storage=sizes,full_training_checkpoint_bytes=checkpoint_bytes,
        inference_tensor_payload_bytes=inference_payload,
        evidence=inputs+[entry(OUT/'fixed_regions.json')],script=entry(Path(__file__)),
        interpretation={
            'region':'fixed image coordinates, two frames and two paired suffixes; not tracked semantic masks',
            'temporal':'GT-relative flow-warp L1 on existing fixed dynamic masks; lower is better; estimated flow is diagnostic, never training supervision',
            'H_HR':'legacy signed X-Up(D(X)) residual L1 against true HR, with existing D clamp; not a strict linear high-pass or the orthonormal DCT budget',
            'teacher_H':'same signed residual L1 to frozen teacher on train76; not whole-image teacher error; historical J1 does not contain full-RGB teacher fit',
            'training_checkpoint':'contains optimizer/RNG/recovery state, not deployable model size',
            'inference_tensor_payload':'raw tensor bytes for saved Gaussian/deformation/child parameter and buffer state, excludes both Adams, RNG and training accumulators; not a compressed export'})
    write(OUT/'detail_summary.json',result)
    print(json.dumps(dict(mean_deltas=average,region_deltas=region_average),indent=2))


if __name__=='__main__':main()
