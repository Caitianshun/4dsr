"""Write the one-seed headroom DAG; does not submit any GPU jobs."""
import argparse,json
from common import *
def build(condition,seed=20261007):
    key=f'{condition}_seed{seed}';registry=f'output/dynamic_sr_highres_20261011/registry';runs=f'output/dynamic_sr_highres_20261011/runs/{key}';manifest=f'data/dynamic_sr/highres_20261011/prepared/{condition}/manifest_train_ready_seed{seed}.json';producer=f'data/dynamic_sr/highres_20261011/prepared/{condition}/manifest.json';parent=f'{registry}/full_LR_prefixes/{condition}/seed_{seed}';base=f'{registry}/full_SR_refinement/{condition}/seed_{seed}/Bsync';teacher=f'{registry}/teachers/{condition}';hrroot=f'data/dynamic_sr/highres_20261011/hr_full/{condition}';tasks=[]
    def task(name,script,deps,done,argv):tasks.append(dict(name=f'{key}_{name}',script=f'../dynamic_sr_highres_20261011/{script}',deps=[f'{key}_{d}' for d in deps],done=done,argv=argv,owner='UNALLOCATED'))
    task('register_lr','register_highres.py',[],f'{parent}/config.json',['lr','--manifest',manifest,'--seed',str(seed)])
    task('plan_teacher','teacher_highres.py',[],f'{teacher}/plan.json',['plan','--manifest',producer,'--seeds',str(seed)])
    task('teacher_cost','teacher_highres.py',['plan_teacher'],f'{teacher}/teacher_cost.json',['cost','--plan',f'{teacher}/plan.json'])
    # Full teacher is gated externally on Z5 (three-image inference cost is insufficient).
    task('teacher','teacher_highres.py',['teacher_cost'],f'{teacher}/acceptance_complete.json',['run','--plan',f'{teacher}/plan.json'])
    task('register_sr','register_highres.py',['register_lr','teacher'],f'{base}/config.json',['sr','--manifest',manifest,'--seed',str(seed),'--teacher-index',f'{teacher}/teacher_index_seed{seed}.json'])
    task('expand_hr','expand_hr_highres.py',[],f'{hrroot}/hr_full_storage.json',['--manifest',producer,'--workers','4'])
    coarse=f'{runs}/lr_coarse';fine=f'{runs}/lr_parent'
    task('coarse','entry.py',['register_lr'],f'{coarse}/complete.json',['lr','--parent',parent,'--coarse-only','--arm','author_coarse','--out',coarse])
    task('fine','entry.py',['coarse'],f'{fine}/complete.json',['lr','--parent',parent,'--fine0',f'{coarse}/checkpoint_fine_00000.pt','--arm','gateaff','--prune-gate','none','--affine','--out',fine])
    for arm,lam in [('lam0','0'),('base','0.2'),('oracleHR','0.2')]:
        out=f'{runs}/{arm}';deps=['fine','register_sr']+(['expand_hr'] if arm=='oracleHR' else [])
        argv=['sr','--parent-checkpoint',f'{fine}/checkpoint_fine_14000.pt','--baseline-sr',base,'--lambda-sr',lam,'--dssim','0.2','--prune-gate','none','--affine','--arm',arm,'--out',out]
        if arm=='oracleHR':argv+=['--teacher-source','hr','--hr-root',hrroot]
        task(arm,'entry.py',deps,f'{out}/complete.json',argv)
        evaluation=f'{runs}/eval_{arm}';task(f'eval_{arm}','entry.py',[arm],f'{evaluation}/summary.json',['eval','--checkpoint',f'{out}/checkpoint_06000.pt','--cull','--out',evaluation,'--label',f'{key}_{arm}'])
    for arm,source,deps in [('SRrecon','teacher',['register_lr','register_sr']),('HRref','hr',['register_lr','expand_hr'])]:
        c=f'{runs}/{arm}_coarse';f=f'{runs}/{arm}';extra=['--image-source',source]
        extra+=['--image-baseline-sr',base] if source=='teacher' else ['--hr-root',hrroot]
        task(f'{arm}_coarse','entry.py',deps,f'{c}/complete.json',['lr','--parent',parent,'--coarse-only','--arm',f'{arm}_coarse','--out',c,*extra])
        task(arm,'entry.py',[f'{arm}_coarse'],f'{f}/complete.json',['lr','--parent',parent,'--fine0',f'{c}/checkpoint_fine_00000.pt','--arm',arm,'--prune-gate','none','--affine','--out',f,*extra])
        e=f'{runs}/eval_{arm}';task(f'eval_{arm}','entry.py',[arm],f'{e}/summary.json',['eval','--checkpoint',f'{f}/checkpoint_fine_14000.pt','--cull','--out',e,'--label',f'{key}_{arm}'])
    return tasks
def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=list(CONDITIONS),required=True);p.add_argument('--seed',type=int,choices=SEEDS,default=20261007);p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=local(a.out)
    if not out.is_relative_to(OUT/'tasks'):raise ValueError('Task table must be in new tasks namespace')
    tasks=build(a.condition,a.seed);write(out,tasks);print(json.dumps(dict(status='task_table_written_not_launched',tasks=len(tasks),path=str(out),requires='Z1,Z5 gates, initialized manifest, resources allocated by root, same GPU model paired within condition')))
if __name__=='__main__':main()
