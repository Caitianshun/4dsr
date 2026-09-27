"""Four disposable Adam steps; exact freezes and zero-coefficient graph checks."""
import argparse
from probe_common import *
from parameter_groups import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);a=parser.parse_args();p=load_protocol(a.protocol);out=Path(p['_root'])/'engineering';out.mkdir(exist_ok=False)
    torch.set_num_threads(4);reads,blocked=guard_images(p);_,data,masks=load_data(p);rows=[];steps=0;tick=time.time()
    model=load_initial(p);opt=configure(model,p);pred=render(model,data[p['b']]['camera']);zero=0.*(pred-data[p['b']]['teacher']).abs().mean();zero.backward()
    zero_exact=all(v.grad is not None and torch.count_nonzero(v.grad)==0 for n,v in model.named_parameters() if n in p['trainable']);assert zero_exact
    for arm,(wa,wb) in p['arms'].items():
        model=load_initial(p);opt=configure(model,p);before=geometry_hash(model);sh_before={n:cpu(getattr(model,n)) for n in p['trainable']};assert not opt.state
        x=render(model,data[p['lr_sequence'][0]]['camera']);pa=render(model,data[p['a']]['camera']);pb=render(model,data[p['b']]['camera'])
        loss=(downsample(x,(252,336))-data[p['lr_sequence'][0]]['lr']).abs().mean()+wa*(pa-data[p['a']]['teacher']).abs().mean()+wb*(pb-data[p['b']]['teacher']).abs().mean()
        loss.backward();opt.step();steps+=1;write(out/'update_ledger.json',dict(actual_updates=steps,max_updates=8))
        assert_frozen(model,before);delta={n:cpu(getattr(model,n))-sh_before[n] for n in p['trainable']};assert any(torch.count_nonzero(v)>0 for v in delta.values())
        rows.append(dict(arm=arm,loss=float(loss),geometry_unchanged=True,optimizer_names=[g['name'] for g in opt.param_groups],sh_updates=named_statistics(delta,model.base_count)))
    original=load_initial(p);before=model_hash(original);values=observe(original,data,masks);assert before==model_hash(original)
    write(out/'complete.json',dict(status='passed',actual_updates=steps,zero_SR_gradient_exact=bool(zero_exact),rows=rows,readonly_state_unchanged=True,initial_metrics=values,seconds=time.time()-tick,counts=COUNTS,image_reads=reads,HR_reads=0,blocked=blocked,source_sha256=sources()));print('engineering passed,4 disposable steps')
if __name__=='__main__':main()
