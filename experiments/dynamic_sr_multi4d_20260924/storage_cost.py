"""Serialize complete inference state separately from resumable training state.

These explicit-schema archives retain all branch weights, deformation networks,
fixed charts and scalar rendering settings. They are not upstream PLY exports or
compressed deployment formats. No Adam, sampling/RNG or controller state.
"""
import argparse
from pathlib import Path
import torch
from summarize import read,write,sha
from full_state import cpu,digest


def tensors(x):
    if isinstance(x,torch.Tensor):yield x
    elif isinstance(x,dict):
        for v in x.values():yield from tensors(v)
    elif isinstance(x,(tuple,list)):
        for v in x:yield from tensors(v)


def archive(state):
    if state.get('schema')=='multi4d_full_state_v1':
        branches=[]
        for b in state['branches']:
            branches.append(dict(parameters={k:v['value'] for k,v in b['parameters'].items()},
                modules={k:v['state'] for k,v in b['modules'].items()},values=b['values'],
                fixed_tensors={k:v for k,v in b['tensors'].items() if k=='_deformation_table'}))
        return dict(schema='multi4d_inference_v1',branches=branches,hidden=state['hidden_config'])
    model=state['model'];assert len(model)==14
    payload=dict(schema='wu_full_system_inference_v1',hidden=state['hidden'],
        base=dict(zip(['active_sh_degree','xyz','deformation','deformation_table','sh_dc','sh_rest','logscale','rotation','opacity'],model[:9])))
    if 'motion_refinement' in state:
        r=state['motion_refinement'];payload['refinement']={k:r[k] for k in ['branch','original_count','children']}
    if 'geometry_residual' in state:
        r=state['geometry_residual'];payload['geometry_residual']={k:v for k,v in r.items() if 'optimizer' not in k and k!='capacity'}
    return payload


def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);rows=[]
    for name,item in read(a.methods)['methods'].items():
        checkpoint=Path(item['checkpoint']);s=torch.load(checkpoint,map_location='cpu',weights_only=False)
        payload=cpu(archive(s));path=a.out/(name+'_inference.pt');torch.save(payload,path)
        restored=torch.load(path,map_location='cpu',weights_only=False);assert digest(payload)==digest(restored)
        rows.append(dict(method=name,training_checkpoint_bytes=checkpoint.stat().st_size,
            inference_archive_bytes=path.stat().st_size,inference_tensor_bytes=sum(t.numel()*t.element_size() for t in tensors(payload)),
            archive=str(path.resolve()),archive_sha256=sha(path),source_checkpoint_sha256=sha(checkpoint),
            note='All branches/networks/fixed charts; uncompressed explicit-schema archive, not a stock-upstream checkpoint or compressed codec'))
    write(a.out/'complete.json',dict(status='verified_inference_state_archives',rows=rows))


if __name__=='__main__':main()
