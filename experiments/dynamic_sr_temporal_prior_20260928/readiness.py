"""Input and implementation integrity checks only; no quality qualification gate."""
from dv_common import *

def issue():
    p=require_run_root(OUT);assert sources()==p['sources']
    for k in ['parent','manifest','teacher','lr_curve','old_schedule']:bound(p[k])
    for e in list(p['schedules'].values())+list(p['priors'].values()):bound(e)
    for e in p['training_files']:bound(e)
    for e in p['inherited_evidence'].values():bound(e)
    tests=read(OUT/'control_tests.json');assert tests['status']=='passed'
    previous=read(bound(p['inherited_evidence']['prior_readiness']))['sources']
    for name,h in previous.items():
        if name in p['sources']:assert p['sources'][name]==h,name
    r=dict(status='ready_for_quality_experiment',protocol_sha256=sha(OUT/'protocol.json'),parent_sha256=p['parent']['sha256'],
        sources=p['sources'],evidence=p['inherited_evidence'],control_tests=entry(OUT/'control_tests.json'),issued_unix=time.time(),
        arithmetic='J_joint LR+regularization backward then .1 target L1 joint backward then both Adams; only frozen target changed')
    path=OUT/'research_readiness.json'
    if path.exists():
        old=read(path);assert old['sources']==r['sources'] and old['protocol_sha256']==r['protocol_sha256']
    else:write(path,r)
    return r

def validate(p,source=None):
    r=read(OUT/'research_readiness.json')
    assert r['status']=='ready_for_quality_experiment' and r['protocol_sha256']==sha(OUT/'protocol.json')
    assert r['parent_sha256']==p['parent']['sha256'] and r['sources']==p['sources']==(sources() if source is None else source)
    bound(r['control_tests'])
    return r

if __name__=='__main__':print(json.dumps(issue()))
