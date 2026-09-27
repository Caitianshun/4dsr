"""One-time anchor choice from LR tracks and immutable premeasurement rules."""
import argparse
import random
import shutil
import subprocess
from probe_common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--gpu',required=True);parser.add_argument('--out',type=Path,default=OUT);a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    old=ROOT/'output/dynamic_sr_evidence_repair_20260927';prior=read(old/'protocol.json');m=legacy.load_manifest(prior['manifest']['path']);root=Path(m['_root'])
    trackfile=old/'support_v1/tracks.json';tracks=read(trackfile);assert len({t['track_id'] for t in tracks})==len(tracks)
    anchor='cam02';candidates=[]
    for c in prior['train_cameras']:
        if c==anchor:continue
        common=[t for t in tracks if t['frame']==40 and anchor in t['projections'] and c in t['projections']]
        grids={cam:len({tuple(np.floor(np.array(t['projections'][cam]['observed_xy'])/16).astype(int)) for t in common}) for cam in [anchor,c]}
        candidates.append(dict(camera=c,common_tracks=len(common),a_grids=grids[anchor],b_grids=grids[c],eligible=len(common)>=12 and min(grids.values())>=4,track_ids=[t['track_id'] for t in common]))
    candidates.sort(key=lambda x:(-x['common_tracks'],x['camera']));eligible=[x for x in candidates if x['eligible']];chosen=eligible[0]['camera'] if eligible else None
    masks={};selected=[]
    if chosen:
        selected=[t for t in tracks if t['frame']==40 and anchor in t['projections'] and chosen in t['projections']]
        for c in [anchor,chosen]:
            yy,xx=np.indices((252,336));mask=np.zeros((252,336),bool);clipped=0
            for t in selected:
                xy=np.array(t['projections'][c]['observed_xy']);assert len(t['projections'])>=3 and any(p['role']=='held_out_validation' for p in t['projections'].values())
                mask|=(xx-xy[0])**2+(yy-xy[1])**2<=16;clipped+=int(xy[0]-4<0 or xy[1]-4<0 or xy[0]+4>335 or xy[1]+4>251)
            p=a.out/'masks'/f'{c}.npz';p.parent.mkdir(exist_ok=True);np.savez_compressed(p,M_lr=mask,M_hr=mask.repeat(4,0).repeat(4,1));masks[c]=dict(path=str(p.resolve()),sha256=sha(p),pixels_lr=int(mask.sum()),pixels_hr=int(mask.sum()*16),clipped_disks=clipped)
    files=[];inventory=read(prior['teacher']['path']);teachers={(e['camera'],e['frame']):e for e in inventory['entries']}
    for o in m['observations']:
        if o['split']=='train' and o['frame_index']==40:
            for typ,p,h in [('LR',root/o['lr_path'],o['lr_sha256']),('teacher',root/teachers[o['camera_id'],40]['relative_path'],teachers[o['camera_id'],40]['sha256'])]:
                assert sha(p)==h;files.append(dict(camera=o['camera_id'],frame=40,reference=typ,path=str(p),sha256=h))
    assert len(files)==38;assert sha(prior['start'])==prior['start_sha256']
    rng=random.Random(2026092701);sequence=[]
    while len(sequence)<600:
        part=list(prior['train_cameras']);rng.shuffle(part);sequence.extend(part)
    config=read(old/'fixed_time/Baked40/config.json');rates={g['group']:{k:g[k] for k in ['lr','betas','eps']} for g in config['groups']}
    p=dict(schema=1,registered_unix=time.time(),status='registered' if chosen else 'insufficient_common_support',scene='cook_spinach',frame=40,time=40/300,manifest=prior['manifest'],teacher_index=prior['teacher'],start=prior['start'],start_sha256=prior['start_sha256'],physical_gpu=a.gpu,
        source_commit='b32a7112876ed5e43ca722e319f53dcd880576cf',plan_sha256=sha(a.plan),tracks=dict(path=str(trackfile),sha256=sha(trackfile)),a=anchor,b=chosen,anchor_candidates=candidates,common_track_ids=[t['track_id'] for t in selected],masks=masks,
        mask_definition='Union of radius4 LR disks centered at observed_xy, integer OpenCV pixel centers; nearest4x replication; diagnostic only',train_cameras=prior['train_cameras'],training_files=files,seed=2026092701,lr_sequence=sequence[:600],rates=rates,
        arms={'A_half':[.05,0.],'A_full':[.1,0.],'B_half':[0.,.05],'AB':[.05,.05]},order=['A_half','A_full','B_half','AB'],steps=600,save_steps=[0,300,600],trainable=['sh_dc','sh_rest'],frozen=['xyz','logscale','quaternion','opacity'],epsilons=[1.,.25],groups={'SH':['sh_dc','sh_rest'],'xyz':['xyz'],'shape':['logscale','quaternion'],'opacity':['opacity'],'all':list(rates)},
        thresholds=dict(min_denominator=1e-6,own_H_gain=.05,cross_H=.02,lr_harm=.02,transfer_absolute=1e-7,transfer_relative=1e-4),budget=dict(training=4800,first_round=2400,diagnostic_adam_candidates=12,engineering_updates=8,total_optimizer_steps=4820),operator=detail.OPERATOR,
        information_boundary='No HR or dev image reads until final train-only repeat decision is frozen; historical ROI only diagnostic, never anchor choice or training weighting')
    shutil.copyfile(a.plan,a.out/'adopted_plan.codex.md');write(a.out/'common_tracks.json',selected);write(a.out/'protocol.json',p)
    csvwrite(a.out/'pair_support.csv',[dict(**{k:v for k,v in x.items() if k!='track_ids'},selected=x['camera']==chosen,selection_rule='max unique LR shared tracks; ties camera ascending',a_mask_pixels=masks.get(anchor,{}).get('pixels_lr'),b_mask_pixels=masks.get(x['camera'],{}).get('pixels_lr')) for x in candidates])
    write(a.out/'registration_sources.json',sources());print(json.dumps(dict(a=anchor,b=chosen,selected_tracks=len(selected),masks=masks,status=p['status']),ensure_ascii=False))
if __name__=='__main__':main()
