"""Read-only independent endpoint and identity audit; no renderer or optimizer step."""
from common_cov import *
from decide import pair,hr

def main():
    p=load_protocol(OUT/'protocol.json');d=read(OUT/'decision.json');checks=[];parent=torch.load(p['parent']['path'],map_location='cpu',weights_only=False)['baked'];pv=parent['values']
    for rr in d['training']['pairs']:
        repeat=rr['repeat'];assert pair(OUT,repeat)==rr
        for arm in p['order'][str(repeat)]:
            manifest=read(OUT/f'r{repeat}_{arm}/train/run_manifest.json');allowed=set(p['active'][arm]);assert not manifest['optimizer_initial']['state']
            for step in [0,300,600]:
                path=OUT/f'r{repeat}_{arm}/train/checkpoint_{step}.pt';ck=torch.load(path,map_location='cpu',weights_only=False);v=ck['baked']['values'];assert ck['baked']['base_count']==88648 and ck['baked']['degree']==3 and len(v['xyz'])==132972
                fixed_names=['xyz','opacity']+(['logscale','quaternion'] if arm=='C' else []);assert all(torch.equal(v[n],pv[n]) for n in fixed_names);assert all(torch.isfinite(x).all() for x in v.values());assert set(g['name'] for g in ck['optimizer']['param_groups'])==allowed
                if step==0:assert all(torch.equal(v[n],pv[n]) for n in v) and not ck['optimizer']['state']
                else:assert all(int(x['step'])==step for x in ck['optimizer']['state'].values())
                assert ck['sampler']['position']==step and ck['sampler']['sequence']==p['sequences'][str(repeat)] and set(ck['rng'])=={'python','numpy','torch','cuda'}
                checks.append(dict(path=str(path),sha256=sha(path),arm=arm,step=step,frozen_parameters=fixed_names,finite=True,Adam_step=step,complete_RNG_and_sampler=True))
    assert hr(OUT)==d
    for item in p['training_files']+list(p['masks'].values())+[p['manifest'],p['teacher_index'],p['parent'],p['roi_definition']]:assert sha(item['path'])==item['sha256']
    frozen=d['training']['frozen_unix'];commands=[json.loads(x) for x in (OUT/'commands.jsonl').read_text().splitlines()];assert all(x['returncode']==0 for x in commands)
    for cmd in commands:
        if '--hr' in cmd['command']:
            target=Path(cmd['command'][cmd['command'].index('--out')+1]);r=read(target/'complete.json');assert cmd['completed']-r['seconds']>=frozen
    for f in [OUT/'parent_train/complete.json',OUT/'engineering_v2/complete.json']+list(OUT.glob('r*/train/complete.json'))+list(OUT.glob('r*/eval_*/complete.json')):
        r=read(f);assert not r['blocked'];assert all(x in {item['path'] for item in p['training_files']} for x in r['image_reads'])
    write(OUT/'integrity.json',dict(status='passed',checks=checks,data_hashes=True,ROI_immutable=True,HR_after_train_decision=True,training_image_whitelist=True,decision_exact_recomputation=True,zero_parameters_exact=True,complete_checkpoint_state=True,checked_unix=time.time()))
    print('integrity passed; six checkpoints, input identities, exact decisions and HR access order')
if __name__=='__main__':main()
