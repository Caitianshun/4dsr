"""Call unchanged full_initialization.generate; source pixels remain train LR only."""
import argparse, sys
from pathlib import Path
from common import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--seeds',nargs='+',type=int,default=[20261007]);p.add_argument('--cpu-threads',type=int,choices=[1,2],default=2);a=p.parse_args()
    path=local(a.manifest)
    if not path.is_relative_to(DATA/'prepared') or read(path)['scene'] not in CONDITIONS:raise ValueError('New highres manifest required')
    sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_multiview_footprint_20261007'))
    generator=module('highres_unchanged_initialization',ROOT/'experiments/dynamic_sr_multiview_footprint_20261007/full_initialization.py')
    receipts=[generator.generate(path,s,a.cpu_threads) for s in a.seeds]
    for seed in a.seeds:
        for d in [path.parent/f'initialization_seed_{seed}',*(path.parent/f'initialization_seed_{seed}').rglob('*')]:
            if d.is_dir():d.chmod(0o2775)
    write(OUT/'registry'/'initializations'/f'{read(path)["scene"]}.json',dict(status='completed_LR_only_initialization',receipts=[r['derived_manifest'] for r in receipts],source=entry(__file__),unchanged_generator=entry(generator.__file__),CUDA_calls=0))
if __name__=='__main__':main()
