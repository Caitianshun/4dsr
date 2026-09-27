"""Eight discarded updates verify new active groups and full save/restore continuity."""
import argparse
from common_cov import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();p=load_protocol(a.protocol);a.out.mkdir(parents=True,exist_ok=False);tick=time.time();torch.set_num_threads(4);reads,blocked=guard_images(p);_,data,_=load_data(p);parent=initial(p);parent_hash=model_hash(parent);rows=[];limits=read(ROOT/'output/dynamic_sr_prior_guidance_20260927/protocol.json')['engineering'];write(a.out/'limits.json',limits);ledger=dict(path=str(a.out/'ledger.json'),kind='engineering',started_updates=0,adam_calls=0);write(ledger['path'],ledger)
    with torch.no_grad():reference={c:render(parent,data[c]['camera']) for c in ['cam02','cam03']}
    for arm in ['C','S']:
        m=initial(p);opt=configure(m,p,p['active'][arm]);assert not opt.state;freeze=digest(contract_state(m,arm));assert model_hash(m)==parent_hash
        with torch.no_grad():parity={c:float((render(m,data[c]['camera'])-reference[c]).abs().max()) for c in reference}
        assert max(parity.values())==0
        cov0=cpu(covariance(m));update(m,opt,data,p['sequences']['1'][0],p,ledger);check_contract(m,arm,freeze)
        changed=not torch.equal(cov0,cpu(covariance(m)));assert changed==(arm=='S')
        shape_grad={n:float(getattr(m,n).grad.norm()) for n in ['logscale','quaternion']} if arm=='S' else {}
        path=a.out/f'{arm}_step1.pt';ck=snapshot(m,opt,p,arm,1,1);torch.save(ck,path)
        # Next-step reference and restored execution consume identical RNG and sampler position.
        update(m,opt,data,p['sequences']['1'][1],p,ledger);expected=fixed.state(m);opt_expected=cpu(opt.state_dict())
        restored,ro,pos=restore(path,p,arm,1);assert pos==1 and digest(fixed.state(restored))==digest(ck['baked']) and digest(cpu(ro.state_dict()))==digest(ck['optimizer']);assert digest(rng_state())==digest(ck['rng'])
        update(restored,ro,data,p['sequences']['1'][pos],p,ledger);check_contract(restored,arm,freeze)
        diffs={n:float((v-expected['values'][n]).abs().max()) for n,v in cpu(fixed.state(restored))['values'].items()}
        # CUDA raster backward uses atomics. Save/restore identities above are exact;
        # trajectory drift is reported rather than mislabelled bitwise determinism.
        with torch.no_grad():
            base_image=render(m,data['cam02']['camera']);resume_image=render(restored,data['cam02']['camera'])
        replay=fixed.Baked(ck['baked']['values'],ck['baked']['base_count'],ck['baked']['degree']).cuda();rp=configure(replay,p,p['active'][arm]);rp.load_state_dict(ck['optimizer']);set_rng(ck['rng']);update(replay,rp,data,p['sequences']['1'][1],p,ledger)
        with torch.no_grad():replay_image=render(replay,data['cam02']['camera'])
        comparisons={}
        for label,other,im in [('resume',restored,resume_image),('repeat',replay,replay_image)]:
            delta=im-base_image
            comparisons[label]=dict(tensor_maxabs=max(float((cpu(v)-expected['values'][n]).abs().max()) for n,v in other.named_parameters()),meanabs=float(delta.abs().mean()),rmse=float(delta.square().mean().sqrt()),render_maxabs=float(delta.abs().max()))
        r,t=comparisons['resume'],comparisons['repeat']
        tests=dict(tensor=r['tensor_maxabs']<=max(limits['tensor_floor'],limits['replay_envelope']*t['tensor_maxabs']),mean=r['meanabs']<=limits['render_meanabs'],rms=r['rmse']<=limits['render_rmse'],rms_envelope=r['rmse']<=max(limits['render_rmse_floor'],limits['replay_envelope']*t['rmse']))
        write(a.out/(arm+'_replay.json'),dict(comparisons=comparisons,tests=tests,limits=limits))
        assert all(tests.values()),tests
        rows.append(dict(arm=arm,zero_RGB_maxabs=parity,empty_Adam=True,initial_parameter_hash=parent_hash,active_scalars=sum(v.numel() for v in restored.parameters() if v.requires_grad),freeze_pass=True,shape_grad_norm=shape_grad,covariance_changed=changed,restore_parameter_Adam_RNG_sampler_exact=True,next_update_maxabs=diffs,replay_comparisons=comparisons,replay_tests=tests,next_Adam_hash_equal=digest(cpu(ro.state_dict()))==digest(opt_expected)))
    assert ledger['started_updates']==8 and not blocked
    write(a.out/'complete.json',dict(status='completed',rows=rows,updates=8,adam_calls=8,training_forwards=24,readonly_forwards=12,seconds=time.time()-tick,image_reads=reads,blocked=blocked,counts=COUNTS,source_sha256=sources()))
if __name__=='__main__':main()
