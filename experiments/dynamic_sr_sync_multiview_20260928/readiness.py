"""Exact inherited identities plus CPU semantics; no numerical replay quality gate."""
from dv_common import *


def issue():
    p=require_run_root(OUT);source=sources();assert source==p['sources']
    for e in p['inherited_evidence'].values():bound(e)
    inherited=read(bound(p['inherited_evidence']['prior_readiness']))['sources']
    for path,h in inherited.items():
        if path in source:assert source[path]==h,path
    for key in ['parent','manifest','teacher','lr_curve','old_schedule']:bound(p[key])
    for e in list(p['schedules'].values())+list(p['pair_schedules'].values()):bound(e)
    tests=read(OUT/'control_tests.json');assert tests['status']=='passed'
    r=dict(status='ready_for_quality_experiment',protocol_sha256=sha(OUT/'protocol.json'),
        parent_sha256=p['parent']['sha256'],sources=source,evidence=p['inherited_evidence'],
        numerical_replay='warning',control_tests=entry(OUT/'control_tests.json'),issued_unix=time.time(),
        arithmetic='Original differentiable joint render and backward; sequential .05 SR gradients then both Adams once')
    path=OUT/'research_readiness.json'
    if path.exists():
        old=read(path);assert old['sources']==source and old['protocol_sha256']==r['protocol_sha256']
    else:write(path,r)
    return r


def validate(p,source=None):
    r=read(OUT/'research_readiness.json')
    assert r['status']=='ready_for_quality_experiment'
    assert r['protocol_sha256']==sha(OUT/'protocol.json') and r['parent_sha256']==p['parent']['sha256']
    assert r['sources']==p['sources']==(sources() if source is None else source)
    for e in r['evidence'].values():bound(e)
    bound(r['control_tests'])
    return r


if __name__=='__main__':print(json.dumps(issue()))
