"""Event-driven GPU screening DAG; local HR expansion/initialization are prerequisites."""
import argparse,json
from common import *
import make_tasks
def build(condition,seed,z5_receipt,z5_samples):
    all_tasks=make_tasks.build(condition,seed);key=f'{condition}_seed{seed}';teacher=f'output/dynamic_sr_highres_20261011/registry/teachers/{condition}';base=f'output/dynamic_sr_highres_20261011/registry/full_SR_refinement/{condition}/seed_{seed}/Bsync';runs=f'output/dynamic_sr_highres_20261011/runs/{key}'
    gpu_tasks=[dict(t) for t in all_tasks if t['script'].endswith('/entry.py')];selected={t['name'] for t in gpu_tasks}
    result=[]
    def task(name,script,deps,done,argv,device='gpu'):
        return dict(name=f'{key}_{name}',script=script,deps=[f'{key}_{d}' for d in deps],done=done,argv=argv,device=device)
    def prior_fit(arm,checkpoint,dependency):
        out=f'{runs}/prior_fit_{arm}'
        return task(f'prior_fit_{arm}','prior_fit.py',[dependency,'register_sr'],f'{out}/prior_fit_complete.json',['--checkpoint',checkpoint,'--baseline-sr',base,'--z5-samples',str(z5_samples),'--hr-root',f'data/dynamic_sr/highres_20261011/hr_full/{condition}','--out',out,'--cull'])
    args=['--plan',f'{teacher}/plan.json','--z5-receipt',str(z5_receipt),'--z5-samples',str(z5_samples)]
    result.append(task('teacher_cost','teacher_screened.py',[],f'{teacher}/teacher_cost.json',['cost',*args]))
    result.append(task('teacher','teacher_screened.py',['teacher_cost'],f'{teacher}/acceptance_complete.json',['run',*args]))
    manifest=f'data/dynamic_sr/highres_20261011/prepared/{condition}/manifest_train_ready_seed{seed}.json'
    result.append(task('register_sr','register_highres.py',['teacher'],f'{base}/config.json',['sr','--manifest',manifest,'--seed',str(seed),'--teacher-index',f'{teacher}/teacher_index_seed{seed}.json'],'cpu'))
    for t in gpu_tasks:
        t['script']='entry.py';t.pop('owner',None);t['device']='gpu';t['deps']=[d for d in t['deps'] if d in selected or d.endswith('_register_sr')]
        result.append(t)
        if t['name'].endswith('_fine'):
            cp=f'{runs}/lr_parent/checkpoint_fine_14000.pt';ev=f'{runs}/eval_LRparent17k';ex=f'{runs}/extras_LRparent17k'
            result.append(task('eval_LRparent17k','entry.py',['fine'],f'{ev}/summary.json',['eval','--checkpoint',cp,'--cull','--out',ev,'--label',f'{key}_LRparent17k']))
            result.append(task('extras_LRparent17k','entry.py',['eval_LRparent17k','register_sr'],f'{ex}/extras.json',['extras','--checkpoint',cp,'--baseline-sr',base,'--cull','--diagnostics','--out',ex]))
            result.append(prior_fit('LRparent17k',cp,'fine'))
        if '_eval_' in t['name']:
            arm=t['name'].split('_eval_',1)[1];cp=f'{runs}/{arm}/checkpoint_06000.pt' if arm in ['lam0','base','oracleHR'] else f'{runs}/{arm}/checkpoint_fine_14000.pt';ex=f'{runs}/extras_{arm}'
            result.append(task(f'extras_{arm}','entry.py',[f'eval_{arm}','register_sr'],f'{ex}/extras.json',['extras','--checkpoint',cp,'--baseline-sr',base,'--cull','--diagnostics','--out',ex]))
            if arm in ['SRrecon','HRref','base']:result.append(prior_fit(arm,cp,arm))
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=list(CONDITIONS),required=True);p.add_argument('--seed',type=int,default=20261007);p.add_argument('--z5-receipt',required=True);p.add_argument('--z5-samples',required=True);p.add_argument('--out',required=True,type=Path);a=p.parse_args();tasks=build(a.condition,a.seed,a.z5_receipt,a.z5_samples)
    write(local(a.out),dict(schema='highres_screening_event_DAG_v1',condition=a.condition,seed=a.seed,tasks=tasks,external_prerequisites=['prepared/initialized manifest and LR registration','teacher plan; Z5 metrics+sample inventory','separate local-produced HR full storage copied byte-for-byte'],budgets=dict(LRparent17k=17000,lam0_base_oracle=23000,SRrecon_HRref=17000),extras_scope='unchanged modsr defaults: cam00 stride10 (30frames), cam01 phase15 stride30 (10frames), trainLR QC15/150/285; no300frame ROI claim',prior_fit_scope='all train cameras at15/150/285; LRparent17k/SRrecon/HRref/base; same HR+teacher positions; no heldout pixels or parameter updates'))
if __name__=='__main__':main()
