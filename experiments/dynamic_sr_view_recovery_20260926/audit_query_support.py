"""CPU post-audit: canonical HexPlane queries vs deformed-center coverage."""
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/dynamic_sr_view_recovery_20260926'

def main():
    assets=json.loads((OUT/'assets.json').read_text());m=json.loads(Path(assets['manifest']['path']).read_text())
    result={}
    for label,asset in assets['endpoints'].items():
        ck=torch.load(asset['path'],map_location='cpu',weights_only=False);c=ck['motion_refinement']['children']
        base=ck['model'][1];child=c['origin']+(c['chart']@(1.5*c['offset_raw'].tanh()).unsqueeze(-1)).squeeze(-1)
        xyz=torch.cat((base,child)).detach().numpy();bounds=np.asarray(asset['aabb']);lo,hi=bounds.min(0),bounds.max(0)
        inside=np.all((xyz>=lo)&(xyz<=hi),axis=1);nb=len(base);groups=dict(base=np.arange(len(xyz))<nb,child=np.arange(len(xyz))>=nb)
        entry=dict(query_space='Canonical base xyz / bounded child xyz, actual HexPlane inputs; distinct from deformed-center AABB diagnostic',
            aabb=asset['aabb'],canonical={g:dict(points=int(mask.sum()),outside=int((mask&~inside).sum()),outside_fraction=float((~inside[mask]).mean())) for g,mask in groups.items()},frames=[])
        for f in [0,40,80,118]:
            cur=np.load(OUT/f'p0/drift/{label}_state_{f:04d}.npz')['xyz'];old=np.load(OUT/f'p0/drift/U6000_state_{f:04d}.npz')['xyz']
            for camera in ['cam00','cam01','cam02']:
                cal=m['cameras'][camera]
                def proj(x):
                    p=np.c_[x,np.ones(len(x))]@np.asarray(cal['w2c'])[:3].T;q=p@np.asarray(cal['K_lr']).T
                    return q[:,:2]/np.where(np.abs(q[:,2:])>1e-10,q[:,2:],np.nan),p[:,2]>0
                xy,front=proj(cur);ref,rf=proj(old);finite=front&rf&np.isfinite(xy).all(1)&np.isfinite(ref).all(1)
                both=finite&(xy[:,0]>=0)&(xy[:,0]<336)&(xy[:,1]>=0)&(xy[:,1]<252)&(ref[:,0]>=0)&(ref[:,0]<336)&(ref[:,1]>=0)&(ref[:,1]<252)
                delta=np.linalg.norm(xy-ref,axis=1)
                for group,mask in groups.items():
                    def stat(q):return dict(n=len(q),median=float(np.median(q)),p95=float(np.quantile(q,.95)),mean=float(q.mean())) if len(q) else dict(n=0)
                    entry['frames'].append(dict(frame=f,camera=camera,group=group,front_finite=stat(delta[finite&mask]),both_in_frustum=stat(delta[both&mask])))
        result[label]=entry
    (OUT/'p0/query_support.json').write_text(json.dumps(dict(status='completed',endpoints=result,
        warning='Front-finite projections can have huge near-plane outliers; use medians and both-in-frustum diagnostics. Frustum membership does not prove actual light contribution or true visibility.'),indent=2))

if __name__=='__main__':main()
