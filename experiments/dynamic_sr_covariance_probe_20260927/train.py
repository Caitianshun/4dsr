"""Paired fixed-time training with complete checkpoint state and strict freeze contract."""
import argparse
from collections import Counter
from common_cov import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--arm',choices=['C','S'],required=True);ap.add_argument('--repeat',type=int,required=True);a=ap.parse_args();p=load_protocol(a.protocol);a.out.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4)
    assert os.environ['CUDA_VISIBLE_DEVICES']==p['physical_gpu'];reads,blocked=guard_images(p);_,data,masks=load_data(p);m=initial(p);opt=configure(m,p,p['active'][a.arm]);freeze=digest(contract_state(m,a.arm));parent_cov=cpu(covariance(m));source=sources();active_count=sum(v.numel() for v in m.parameters() if v.requires_grad);assert active_count=={'C':6382656,'S':7313460}[a.arm]
    manifest=dict(run_id=f'A_r{a.repeat}_{a.arm}',stage='A',repeat=a.repeat,arm=a.arm,parent=p['parent'],initial_model_hash=model_hash(m),protocol_sha256=sha(a.protocol),source_sha256=source,gpu_uuid=p['physical_gpu'],gpu_name=torch.cuda.get_device_name(),active_names=p['active'][a.arm],active_parameter_ids={n:id(v) for n,v in m.named_parameters() if v.requires_grad},active_scalars=active_count,optimizer_initial=cpu(opt.state_dict()),freeze_hash=freeze,sequence=p['sequences'][str(a.repeat)],start_step=0,end_step=600,started_unix=started)
    write(a.out/'run_manifest.json',manifest);ledger=dict(path=str(a.out/'ledger.json'),kind='formal',started_updates=0,adam_calls=0);write(ledger['path'],ledger);peak=0;elapsed=0.;exposure=Counter()
    def save(step):
        check_contract(m,a.arm,freeze);path=a.out/f'checkpoint_{step}.pt';torch.save(snapshot(m,opt,p,a.arm,a.repeat,step),path);write(path.with_suffix('.json'),dict(path=str(path),sha256=sha(path),model_hash=model_hash(m),arm=a.arm,repeat=a.repeat,step=step,freeze_hash=freeze))
    save(0);torch.cuda.reset_peak_memory_stats()
    with (a.out/'training.jsonl').open('w',buffering=1) as f:
        for step,c in enumerate(p['sequences'][str(a.repeat)],1):
            tick=time.perf_counter();values=update(m,opt,data,c,p,ledger);elapsed+=time.perf_counter()-tick;exposure[c]+=1
            assert all(bool(torch.isfinite(v).all()) for v in m.parameters())
            f.write(json.dumps(dict(step=step,**values,train_seconds=elapsed))+'\n')
            if step in p['save_steps']:save(step)
    check_contract(m,a.arm,freeze);changed=not torch.equal(parent_cov,cpu(covariance(m)));assert changed==(a.arm=='S');assert not blocked and all(sha(k)==v for k,v in source.items());assert COUNTS['rgb_forwards']==1800
    write(a.out/'complete.json',dict(status='completed',stage='A',arm=a.arm,repeat=a.repeat,updates=600,adam_calls=600,training_forwards=1800,exposure=dict(exposure),teacher_exposure={'cam02':600,'cam03':600},freeze_pass=True,covariance_changed=changed,finite_pass=True,source_unchanged=True,train_seconds=elapsed,total_wall_seconds=time.time()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),image_reads=reads,blocked=blocked,gpu_uuid=p['physical_gpu']))
if __name__=='__main__':main()
