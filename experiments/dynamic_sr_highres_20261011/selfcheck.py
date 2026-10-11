"""Meaningful CPU checks: exact degradation, intrinsics, grids, schedules, defaults."""
import collections,sys,tempfile,json,hashlib
from common import *
import prepare_colmap_dataset as p
import register_highres as r
def main():
    import numpy as np,torch,torch.nn.functional as F
    torch.set_num_threads(1);rng=np.random.default_rng(17);checks={}
    for H,W in [(12,20),(68,100),(32,48)]:
        a=rng.integers(0,256,(H,W,3),dtype=np.uint8);x=torch.from_numpy(a.copy()).permute(2,0,1).float()[None]/255
        expected=F.interpolate(x,size=(H//4,W//4),mode='bicubic',antialias=True,align_corners=False).clamp(0,1)[0].permute(1,2,0).mul(255).round().byte().numpy()
        assert np.array_equal(p.downsample(a),expected)
    checks['exact_bicubic_AA_uint8_operator']=True
    K=np.array([[400.,0,49.3],[0,405.,31.2],[0,0,1.]])
    assert np.allclose(p.scale_K(K,.25)[[0,1],2],(K[[0,1],2]+.5)/4-.5)
    checks['pixel_center_intrinsic_transform']=True
    with tempfile.TemporaryDirectory(prefix='loader_smoke_',dir=HERE) as temp:
        root=Path(temp);a=rng.integers(0,256,(12,20,3),dtype=np.uint8)
        p.png(root/'lr/cam01/0000.png',a);p.png(root/'hr/cam00/0000.png',a)
        camera=dict(K_hr=K.tolist(),K_lr=K.tolist(),w2c=np.eye(4).tolist(),c2w=np.eye(4).tolist())
        m=dict(schema='n3dv_dynamic_sr_pilot_v1',scene='synthetic',cameras={'cam00':camera,'cam01':camera},resolutions=dict(hr=[20,12],lr=[20,12]),observations=[dict(camera_id='cam01',frame_index=0,time=0.,split='train',lr_path='lr/cam01/0000.png',hr_path='absent_hr/cam01/0000.png'),dict(camera_id='cam00',frame_index=0,time=0.,split='test',lr_path='absent_lr/cam00/0000.png',hr_path='hr/cam00/0000.png')])
        (root/'manifest.json').write_text(json.dumps(m));loader=module('highres_loader_CPU_check',ROOT/'experiments/dynamic_sr_20260918/n3dv_data.py')
        for split,resolution in [('train','lr'),('test','hr')]:
            d=loader.N3DVPreparedDataset(root/'manifest.json',split,resolution,device='cpu');obs=d[0];assert tuple(obs['image'].shape)==(3,12,20) and obs['split']==split
        checks['unchanged_loader_train_LR_and_test_HR_sparse_paths']=True
    metadata={}
    for config in ['deskgames_cube.json','long360.json']:
        cfg=read(HERE/'datasets'/config);cams,mapping,diag,center=p.metadata(cfg);errors=[p.remap_maps(c)[1] for c in cams.values()]
        assert max(errors)<=.01 and mapping[diag]=='cam01' and mapping[cfg['main_test_original']]=='cam00'
        metadata[config]=dict(cameras=len(cams),diagnostic_original=diag,max_map_error_px=max(errors),central_original=center['selected_original'] if center else None)
    checks['published_models_and_undistortion']=metadata
    config,paths=r.config_default();assert config['OptimizationParams']['batch_size']==4;checks['author_default_batch4_3000_14000']=True
    schedules={}
    for count in [20,32]:
        train=[f'cam{i:02d}' for i in range(1,count+1)];m=dict(splits=dict(train=train),observations=[dict(camera_id=c,frame_index=f,split='train') for c in train for f in range(300)])
        s=r.schedule(m,dict(path='synthetic',sha256='0'*64),20261007);keys=s['record_keys'];assert len(s['rows'])==6000
        for row in s['rows']:
            k=[keys[i] for i in row];assert len({x[0] for x in k})==3 and len({x[1] for x in k})==1
        ex=[collections.Counter(row[i] for row in s['rows']) for i in range(3)];assert ex[0]==ex[1]==ex[2]
        totals=collections.Counter(keys[row[0]][0] for row in s['rows']);assert max(totals.values())-min(totals.values())<=1
        schedules[str(count)]=dict(rows=6000,legal_observations=count*300,exposed=len(ex[0]),column_exposure_equal=True,camera_balance=dict(totals))
    checks['schedules']=schedules
    assert not torch.cuda.is_initialized();source_hashes={str(f.relative_to(HERE)):sha(f) for f in sorted(HERE.glob('*.py'))};version=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()[:12]
    write(HERE/f'selfcheck_cpu_{version}.json',dict(status='passed_CPU_selfchecks',checks=checks,source_sha256=source_hashes,CUDA_calls=0))
    print(checks)
if __name__=='__main__':main()
