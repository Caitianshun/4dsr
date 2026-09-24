"""CPU end-of-batch identity, optimizer-step, time-basis and result completeness audit."""
from pathlib import Path
import json
import torch
from summarize import read,write,sha
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_geometry_residual_20260924'


def main():
    records=[];indices=[]
    for scene in ['cook','discussion']:
        assert read(OUT/f'{scene}_finalization_v1/complete.json')['status']=='completed_unified_summary_views_storage'
        mapping=read(OUT/f'{scene}_methods.json')['methods']
        ref=read(Path(mapping['U']['train_dir'])/'config.json');ref_done=read(Path(mapping['U']['train_dir'])/'complete.json')
        for method in ['G','L']:
            folder=Path(mapping[method]['train_dir']);config=read(folder/'config.json');done=read(folder/'complete.json')
            assert done['status']=='completed' and done['parameter_updates']==6000 and not done['smoke']
            for key in ['initial_identity','manifest_sha256','parent_sha256','selection_sha256','teacher_index_sha256','schedule_sha256','gpu','torch','seed','scheduler_offset']:
                assert config[key]==ref[key],(scene,method,key)
            for key in ['draw_sha256','lr_draw_sha256','sr_frame_sha256']:assert done[key]==ref_done[key],(scene,method,key)
            assert set(config['teacher_cameras']).isdisjoint({'cam00','cam01'})
            sources=[]
            for row in config['sources']:
                path=folder/'sources'/Path(row['snapshot']).name;assert sha(path)==row['sha256'];sources.append(dict(path=str(path),sha256=row['sha256']))
            for step in [1200,6000]:
                cp=folder/f'checkpoint_{step}.pt';ck=torch.load(cp,map_location='cpu',weights_only=False)
                state=ck['geometry_residual'];tensors=state['tensors'];coeff=tensors['coeff']
                expected=44324 if scene=='cook' else 35576
                assert tuple(coeff.shape)==(expected,8,3) and torch.isfinite(coeff).all()
                assert state['kind']==method and state['basis']['frames']==list(range(0,120,2))
                newopt=state['optimizer'];assert len(newopt['state'])==1
                assert float(next(iter(newopt['state'].values()))['step'])==step
                group=next(g for g in ck['model'][12]['param_groups'] if g['name']=='xyz')
                assert newopt['param_groups'][0]['lr']==group['lr']
                for key in ['betas','eps','weight_decay']:assert newopt['param_groups'][0][key]==group[key]
                child_ids=tensors['child_indices'];base_count=len(ck['model'][1])
                assert torch.equal(child_ids,torch.arange(base_count,base_count+expected))
                assert len(child_ids.unique())==expected
                assert read(folder/f'residual_statistics_{step}.json')['children']==expected
                indices.append(dict(scene=scene,method=method,step=step,path=str(cp),sha256=sha(cp),bytes=cp.stat().st_size,
                                    coefficient_parameters=coeff.numel(),coefficient_bytes=coeff.numel()*coeff.element_size(),
                                    optimizer_step=int(step),coefficient_lr=group['lr'],training_gpu=config['gpu'],status='completed'))
                del ck,coeff,state,tensors
            records.append(dict(scene=scene,method=method,initial_state_and_input_identity_matches_U=True,
                all_three_sampling_hashes_match_U=True,source_snapshots=sources,
                observations={k:read(Path(v)/'complete.json')['observations'] for k,v in mapping[method]['evaluations'].items()},
                training_seconds=done['train_s'],peak_allocated_gb=done['peak_gb']))
    write(OUT/'checkpoint_index.json',dict(status='completed_verified_local_checkpoints',checkpoints=indices))
    write(OUT/'protocol_integrity.json',dict(status='passed',runs=records,scope='No quality threshold; exact identity and completeness audit only'))
    print(json.dumps(dict(status='passed',runs=len(records),checkpoints=len(indices))))

if __name__=='__main__':main()
