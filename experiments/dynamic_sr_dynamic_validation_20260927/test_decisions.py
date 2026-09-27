"""Small decision fixtures required by the adopted protocol; no GPU updates."""
import copy
from dv_common import *
from decide import decide_set,finalize
p=read(OUT/'protocol.json')['decision']
u=dict(cameras={c:dict(psnr=30.,lpips=.2,dynamic_lpips=.2,temporal=.01) for c in ['cam00','cam01']},train={f'cam{i:02}':dict(psnr=32.,lpips=.1) for i in range(2,21)},train76_lr=.01)
def sample():
    e={n:copy.deepcopy(u) for n in ['U6000','J_joint','A_sh','S_cov']}
    for n in ['A_sh','S_cov']:e[n]['cameras']['cam00']['psnr']+=.2
    return e
checks=[]
def check(name,x):assert x,name;checks.append(name)
e=sample();e['S_cov']['teacher_H']=999;d=decide_set(e,p);check('teacher alone does not fail',d['candidates']['S_cov']['eligible'])
for fraction in [.02,.05]:
    e=sample();e['S_cov']['train76_lr']*=1+fraction;d=decide_set(e,p);check(f'boundary {fraction}',d['candidates']['S_cov']['eligible']);check(f'observation {fraction}',d['candidates']['S_cov']['quality_tradeoff']==(fraction>.02))
e=sample();e['S_cov']['train76_lr']*=1.05001;check('over5 fails',not decide_set(e,p)['candidates']['S_cov']['eligible'])
e=sample();e['S_cov']['cameras']['cam00']['psnr']=30;e['S_cov']['train76_lr']*=.1;check('no main gain fails',not decide_set(e,p)['candidates']['S_cov']['eligible'])
e=sample();e['S_cov']['train']['cam20']['psnr']-=.2001;check('one camera cannot hide',not decide_set(e,p)['candidates']['S_cov']['eligible'])
e=sample();first=decide_set(e,p)
for n in ['A_sh','S_cov']:e[n]['cameras']['cam00'].update(psnr=30.,dynamic_lpips=.19)
second=decide_set(e,p);check('disjoint modes',not finalize(first,second,p)['confirmed'])
e=sample();e['S_cov']['cameras']['cam00']['psnr']=30.;first=decide_set(e,p);e=sample();e['A_sh']['cameras']['cam00']['psnr']=30.;second=decide_set(e,p);check('disjoint candidates',not finalize(first,second,p)['confirmed'])
e=sample();first=decide_set(e,p);check('simpler A chosen',finalize(first,first,p)['selected']=='A_sh')
e=sample();e['J_joint']['train76_lr']=1e-8;check('tiny denominator fails',not decide_set(e,p)['candidates']['S_cov']['eligible'])
write(OUT/'decision_tests.json',dict(status='passed',checks=checks,protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(__file__)))
print('passed',len(checks))
