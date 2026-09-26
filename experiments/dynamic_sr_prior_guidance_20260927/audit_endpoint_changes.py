"""Effective attribute movement of every completed RGB endpoint, on fixed times."""
import argparse
import numpy as np
from shared import *
from gradient_policy import effective_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    methods=read(a.methods)['methods'];protocol=read(OUT/'protocol.json');m=load_manifest(protocol['manifest']['path'])
    baseline={};rows=[]
    with torch.no_grad():
        for name in ['U6000','shared_17550']:
            model=load_model(methods[name]['checkpoint'],m);baseline[name]={f:{k:v.cpu() for k,v in effective_state(model,f/300).items()} for f in [0,40,80,118]};del model
        for name,v in methods.items():
            if name=='U6000':continue
            model=load_model(v['checkpoint'],m)
            for f in [0,40,80,118]:
                state={k:x.cpu() for k,x in effective_state(model,f/300).items()}
                for ref in ['U6000','shared_17550']:
                    s=baseline[ref][f];d=(state['xyz']-s['xyz']).norm(dim=-1);cov=(state['cov']-s['cov']).norm(dim=(-1,-2));op=(state['opacity']-s['opacity']).abs()
                    rows.append(dict(method=name,reference=ref,frame=f,center_median=float(d.median()),center_p95=float(torch.quantile(d,.95)),cov_frobenius_median=float(cov.median()),opacity_abs_median=float(op.median()),sh_abs_mean=float((state['sh']-s['sh']).abs().mean())))
            del model;torch.cuda.empty_cache()
    write_json(a.out,dict(status='completed',rows=rows,source_sha256=sha256(__file__),gpu=torch.cuda.get_device_name(),parameter_updates=0,interpretation='Matched stored point identity, world-scale units not calibrated meters. Final attribute movement includes LR, SR via remaining pathways, and inherited Adam. It is not an isolated measure of SR direct positional gradients or true geometry error.'))


if __name__=='__main__':main()
