"""CPU accounting archive: inference tensors only, no Adam/RNG/training gradients.

This is an explicit storage account, not a replacement training-checkpoint loader.
The tested geometry loader continues to consume complete training checkpoints.
"""
import argparse
from pathlib import Path
import json
import torch
from summarize import read, write, sha, load_methods


def clean(value):
    if torch.is_tensor(value): return value.detach().cpu().contiguous().clone()
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return type(value)(clean(v) for v in value)
    return value


def tensor_bytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)): return sum(tensor_bytes(v) for v in value)
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
    for name,item in load_methods(a.methods).items():
        ck=torch.load(item['checkpoint'],map_location='cpu',weights_only=False);m=ck['model']
        payload=dict(schema='explicit_inference_tensor_account_v1',active_sh_degree=m[0],
            xyz=m[1],deformation=m[2],sh_dc=m[4],sh_rest=m[5],logscale=m[6],quaternion=m[7],opacity=m[8],
            hidden=ck['hidden'],branch=ck['motion_refinement']['branch'],children=ck['motion_refinement']['children'])
        if 'geometry_residual' in ck:
            g=ck['geometry_residual'];payload['geometry_residual']={k:v for k,v in g.items() if k!='optimizer'}
        payload=clean(payload);file=a.out/f'{name}_inference_tensors.pt';torch.save(payload,file)
        restored=torch.load(file,map_location='cpu',weights_only=False)
        assert tensor_bytes(restored)==tensor_bytes(payload)
        row=dict(method=name,checkpoint=str(item['checkpoint']),checkpoint_sha256=sha(item['checkpoint']),
            inference_tensor_bytes=tensor_bytes(payload),inference_archive=str(file),inference_archive_bytes=file.stat().st_size,
            inference_archive_sha256=sha(file),interpretation='Base attributes, shared deformation tensors, fixed charts and children, residual coefficients/buffers; no optimizer. Includes required metadata in serialized size. Accounting archive, not evaluated standalone loader.')
        rows.append(row);del ck,m,payload,restored
    write(a.out/'storage.json',dict(status='completed_inference_account',methods=rows))

if __name__=='__main__':main()
