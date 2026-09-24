"""Cook-only frame-zero LR cloud and frozen native no-replacement schedules."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from multi_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    m=json.loads(a.manifest.read_text());assert m['scene']=='cook_spinach'
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'experiments/dynamic_sr_20260918'))
    from prepare_n3dv import make_initialization
    (a.out/'lr').symlink_to((a.manifest.parent/'lr').resolve(),target_is_directory=True)
    initialization=make_initialization(a.out,m['cameras'],m['splits']['train'],[0])
    records=[r for r in m['observations'] if r['split']=='train'];rng=random.Random(20260924)
    s=dict(seed=20260924,record_keys=[[r['camera_id'],r['frame_index']] for r in records],rule='native uniform pop without replacement, independent frozen RNG')
    for stage,steps in [('coarse',2000),('fine',20000)]:
        bag=list(range(len(records)));rows=[]
        for step in range(steps):
            ids=[]
            for _ in range(2):
                ids.append(bag.pop(rng.randint(0,len(bag)-1)))
                if not bag:bag=list(range(len(records)))
            rows.append(ids)
        s[stage]=rows;s[stage+'_counts']=dict(Counter(str(i) for row in rows for i in row))
    (a.out/'multi_schedule.json').write_text(json.dumps(s,indent=2)+'\n')
    (a.out/'input_identity.json').write_text(json.dumps(dict(manifest_sha256=sha(a.manifest),initialization=initialization,
        triangulation_source_sha256=sha(root/'experiments/dynamic_sr_20260918/prepare_n3dv.py'),schedule_sha256=sha(a.out/'multi_schedule.json')),indent=2)+'\n')


if __name__=='__main__':main()
