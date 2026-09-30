"""Fail closed on protocol/source/input mismatches, never on interim quality."""
from dv_common import *

def validate(p, source=None):
    assert p['status']=='frozen_after_diagnosis'
    assert p['sources']==(sources() if source is None else source)
    for k in ['parent','manifest','teacher','lr_curve','old_schedule']:
        bound(p[k])
    for e in p['schedules'].values():bound(e)
    tests=read(bound(p['control_tests']))
    assert tests['status']=='passed'
    for t in p['task_plan']:
        assert t['arm'] in ['T','S','TS'] and t['stop']==12000
    if any('T' in t['arm'] for t in p['task_plan']):
        idx=read(bound(p['texture']['maps']))
        assert len(idx['entries'])==1140 and idx['information_boundary']=='legal_train_LR_and_U6000_only'
    return dict(status='ready', protocol_sha256=sha(OUT/'protocol.json'))
