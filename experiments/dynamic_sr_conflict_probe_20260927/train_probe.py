"""600 fixed-footprint SH-only steps, exactly three RGB forwards per update."""
import argparse
from collections import Counter
from probe_common import *
from parameter_groups import *

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--protocol',type=Path,required=True);pa.add_argument('--out',type=Path,required=True);pa.add_argument('--arm',required=True);pa.add_argument('--repeat-id',type=int,required=True);a=pa.parse_args();p=load_protocol(a.protocol);out=a.out;out.mkdir(parents=True,exist_ok=False)
    started=time.time();torch.set_num_threads(4);reads,blocked=guard_images(p);assert os.environ.get('CUDA_VISIBLE_DEVICES')==p['physical_gpu'];_,data,masks=load_data(p);model=load_initial(p);opt=configure(model,p);geometry=geometry_hash(model);initial_hash=model_hash(model);sh0={n:cpu(getattr(model,n)) for n in p['trainable']};wa,wb=p['arms'][a.arm];identity=sources()
    config=dict(arm=a.arm,repeat_id=a.repeat_id,protocol_sha256=sha(a.protocol),initial_model_hash=initial_hash,geometry_hash=geometry,gpu_uuid=p['physical_gpu'],gpu_name=torch.cuda.get_device_name(),coefficients=[wa,wb],sequence=p['lr_sequence'],source_sha256=identity,optimizer=[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],requires_grad={n:v.requires_grad for n,v in model.named_parameters()},preload_seconds=time.time()-started)
    write(out/'config.json',config);torch.cuda.reset_peak_memory_stats();loop=time.time();train_s=0.;exposure=Counter();forward_start=COUNTS['rgb_forwards']
    def save(step):
        assert_frozen(model,geometry);path=out/f'checkpoint_{step}.pt'
        torch.save(dict(baked=fixed.state(model),optimizer=cpu(opt.state_dict()),metadata=dict(arm=a.arm,repeat_id=a.repeat_id,step=step,frame=40,initial_model_hash=initial_hash,geometry_hash=geometry,protocol_sha256=sha(a.protocol))),path)
        write(path.with_suffix('.json'),dict(path=str(path.resolve()),sha256=sha(path),model_hash=model_hash(model),step=step,geometry_hash=geometry))
    save(0)
    with (out/'training.jsonl').open('w',buffering=1) as log:
        for step,c in enumerate(p['lr_sequence'],1):
            tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            r=render(model,data[c]['camera']);ra=render(model,data[p['a']]['camera']);rb=render(model,data[p['b']]['camera'])
            lr=(downsample(r,(252,336))-data[c]['lr']).abs().mean();ta=(ra-data[p['a']]['teacher']).abs().mean();tb=(rb-data[p['b']]['teacher']).abs().mean();loss=lr+wa*ta+wb*tb
            assert bool(torch.isfinite(loss));loss.backward()
            assert all(v.grad is None for n,v in model.named_parameters() if n in p['frozen'])
            assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for n,v in model.named_parameters() if n in p['trainable'])
            opt.step();torch.cuda.synchronize();train_s+=time.perf_counter()-tick;exposure[c]+=1
            log.write(json.dumps(dict(step=step,camera=c,a=p['a'],b=p['b'],a_coefficient=wa,b_coefficient=wb,lr=float(lr),teacher_a=float(ta),teacher_b=float(tb),loss=float(loss),train_seconds=train_s,loop_wall_seconds=time.time()-loop,forwards=3))+'\n')
            if step in p['save_steps']:save(step)
    assert COUNTS['rgb_forwards']-forward_start==1800;assert min(exposure.values())==31 and max(exposure.values())==32;assert_frozen(model,geometry)
    deltas={n:cpu(getattr(model,n))-sh0[n] for n in p['trainable']};assert any(torch.count_nonzero(v)>0 for v in deltas.values());assert all(sha(f)==h for f,h in identity.items())
    write(out/'complete.json',dict(status='completed',arm=a.arm,repeat_id=a.repeat_id,actual_updates=600,training_forwards=1800,exposure=dict(exposure),geometry_hash=geometry,geometry_unchanged=True,SH_changed=True,SH_update_statistics=named_statistics(deltas,model.base_count),HR_training_reads=0,image_reads=reads,blocked=blocked,source_unchanged=True,preload_seconds=config['preload_seconds'],train_seconds=train_s,loop_wall_seconds=time.time()-loop,total_wall_seconds=time.time()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_uuid=p['physical_gpu'],counts=COUNTS))
if __name__=='__main__':main()
