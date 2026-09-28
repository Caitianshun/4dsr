"""Complete-state, accounting and video-decode verification; no new training/rendering."""
import subprocess
import time
from dv_common import *
from schedule import plain
from evaluation_adapter import checkpoint_family
import torch

def finite(x):
    if torch.is_tensor(x):return not x.is_floating_point() or bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,(tuple,list)):return all(finite(v) for v in x)
    return True

def main():
    p=require_run_root(OUT);ph=sha(OUT/'protocol.json');assert sources()==p['sources'];bound(p['parent'])
    idx=read(OUT/'checkpoint_index.json');checks=[];videos=[];signatures=[]
    for task in p['task_plan']:
        label=task['task_id'];record=idx['checkpoints'][label];seen=set();recovered=[]
        for attempt in record['attempts']:
            directory=local(attempt['path'])/'train'
            if not (directory/'config.json').exists():continue
            cfg=read(directory/'config.json');assert cfg['sources']==p['sources'] and cfg['physical_gpu']==p['training']['physical_gpu']
            reload=read(directory/'reload_audit.json');assert reload['two_adam_exact'] and reload['global_rng_exact'] and reload['prefix_verified']
            if (directory/'complete.json').exists():
                assert read(directory/'complete.json')['source_unchanged'] and read(directory/'image_reads.json')['all_allowed']
            audit=read(directory/'first_update_audit.json');assert audit['policy']['policy']=='joint' and not audit['SR_position_detach']
            signatures.append((audit['policy']['stored_parameters'],audit['policy']['parameter_shapes']))
            for step in [9000,12000]:
                path=directory/f'checkpoint_{step}.pt'
                if not path.exists():continue
                meta=read(path.with_suffix('.json'));assert sha(path)==meta['sha256']
                ck=torch.load(path,map_location='cpu',weights_only=False);m=ck['metadata']
                assert m==meta['metadata'] and m['protocol_sha256']==ph and m['task_id']==label
                assert m['points']==132972 and m['intervention_step']==step
                assert ck['model'][12]['state'] and ck['motion_refinement']['child_optimizer']['state']
                assert set(ck['rng'])=={'torch','cuda','numpy','python'}
                schedule=read(bound(p['schedules'][task['repeat']]));assert plain(ck['samplers'])==schedule['states'][str(step)]
                prior=ck['temporal_prior'];assert prior['cursor']==step and prior['mode']==task['mode'] and prior['index']==p['priors'][task['mode']]
                assert prior['sources']==p['sources'] and checkpoint_family(ck)=='temporal_prior'
                assert finite(ck['model']) and finite(ck['motion_refinement'])
                value=dict(task=label,step=step,**entry(path),bytes=path.stat().st_size,two_adam_states=True,rng_and_sampler_exact=True,prior_identity=True,finite=True)
                checks.append(value);recovered.append(value);seen.add(step);del ck
        assert seen=={9000,12000};record['milestones']=recovered
        receipt=read(OUT/'evaluation'/label/'complete.json')
        assert receipt['checkpoint_sha256']==record['sha256'] and receipt['rgb_forwards']==196 and receipt['protocol_sha256']==ph
        assert sha(OUT/'evaluation'/label/'endpoint.json')==receipt['endpoint_sha256']
        directory=local(receipt['attempt_directory'])
        import imageio_ffmpeg
        for camera in ['cam00','cam01']:
            path=directory/camera/'preview60.mp4'
            run=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-','-progress','pipe:1'],capture_output=True,text=True,check=True)
            counts=[int(l.split('=')[1]) for l in run.stdout.splitlines() if l.startswith('frame=')];assert counts[-1]==60
            videos.append(dict(endpoint=label,camera=camera,**entry(path),decoded_frames=60,human_continuous_viewing=False))
    assert all(x==signatures[0] for x in signatures)
    write(OUT/'checkpoint_index.json',idx)
    b=read(OUT/'cost.json');assert b['effective_updates']==24000 and b['actual_updates']<=30000 and b['retry_updates']==b['actual_updates']-24000
    assert b['effective_training_rgb_forwards']==48000 and b['effective_adam_calls']==48000
    if b['retry_updates']==0:assert b['training_rgb_forwards']==48000 and b['adam_calls']==48000
    priors={mode:read(bound(e)) for mode,e in p['priors'].items()}
    for mode,index in priors.items():
        assert len(index['entries'])==1140 and index['mode']==mode
        for e in index['entries']:bound(e)
    b['prior_preparation']={m:{k:v[k] for k in ['inference_seconds_total','new_calls','new_inference_seconds','source_frame_instances','wall_seconds','disk_write_seconds','peak_gpu_gb','disk_bytes','image_cache_hits','image_cache_misses']} for m,v in priors.items()}
    b['teacher_generators']={m:dict(gpu=v['identity']['gpu'],physical_gpu=v['identity']['physical_gpu'],host=v['identity']['host'],commit=v['identity']['commit'],weights=v['identity']['weights']) for m,v in priors.items()}
    b['reference_evaluation']=dict(rgb_forwards=480,seconds=sum(read(f)['seconds'] for f in (OUT/'references').glob('*direct*.json')),updates=0)
    b['engineering']=dict(formal_updates=0,teacher_smoke_calls=1,teacher_smoke_frame_instances=7,
        note='One local 3090 full-shape Video7 generation; semantic and receipt tests perform no 3D training.')
    write(OUT/'cost.json',b)
    write(OUT/'video_integrity.json',dict(status='all_decoded',videos=videos,decoded_frames=480,human_continuous_viewing=False))
    write(OUT/'final_integrity.json',dict(status='passed',verified_unix=time.time(),protocol_sha256=ph,checkpoints=checks,
        stored_parameters=signatures[0][0],equal_model_storage_and_shapes=True,sources_unchanged=True,all_training_image_reads_whitelisted=True,
        new_evaluation_rgb_forwards=784,reference_rgb_forwards=480,effective_updates=24000,actual_updates=b['actual_updates'],retry_updates=b['retry_updates'],
        scope='CPU hashes, complete checkpoints, finite state, exact schedules and receipts. Video decoding does not imply continuous visual viewing.'))
    print(json.dumps(dict(status='passed',checkpoints=len(checks),video_frames=480)))

if __name__=='__main__':main()
