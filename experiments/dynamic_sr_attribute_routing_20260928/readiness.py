"""Reuse exact identity/routing evidence; numerical replay warnings are descriptive."""
from dv_common import *

def check_old_evidence(p):
    evidence = p['inherited_evidence']
    for e in evidence.values(): bound(e)
    audit = read(bound(evidence['routing']))
    assert {r['arm'] for r in audit['rows']} == {'J_joint', 'A_sh', 'S_cov'}
    for r in audit['rows']:
        assert r['zero_rgb_exact'] and r['zero_identity']
        assert r['policy']['shared_head_alias_check']
        if r['arm'] != 'J_joint':
            assert r['isolated_center_check']['maxabs'] == 0
            assert all(x['maxabs'] == 0 for x in r['gradient_checks'])
            assert all(x['allowed'] or x['exact_outside'] for x in r['gradient_checks'])
    for r in p['inherited_reload_audits']: 
        a = read(bound(r)); assert a['global_rng_exact'] and a['two_adam_exact'] and a['prefix_verified']
    assert read(bound(evidence['old_decision']))['formal_updates'] == 0
    assert read(bound(evidence['old_budget']))['engineering_updates'] == 32
    deployed = read(bound(evidence['completed_fixture_config']))['sources']
    source = sources()
    # All shared arithmetic used by the new facade must be the audited implementation.
    for name, h in deployed.items():
        if name in source: assert source[name] == h, name
    return source

def issue():
    p = require_run_root(OUT); source = check_old_evidence(p)
    for key in ['parent', 'manifest', 'teacher', 'lr_curve', 'old_schedule']: bound(p[key])
    for e in p['schedules'].values(): bound(e)
    tests = read(OUT / 'control_tests.json'); assert tests['status'] == 'passed'
    value = dict(status='ready_for_quality_experiment', protocol_sha256=sha(OUT/'protocol.json'),
        parent_sha256=p['parent']['sha256'], sources=source, numerical_replay='warning',
        old_failed_receipts_unchanged=True, evidence=p['inherited_evidence'],
        reload_audits=p['inherited_reload_audits'],
        arithmetic_changes='none: SRAttributeRouter delegates unchanged audited selected/backward_prior functions',
        control_changes=['new run root and readiness checks', 'six unconditional paired tasks',
                         'attempt recovery', 'actual/effective budget ledger', 'three primary quality metrics'],
        control_tests=entry(OUT/'control_tests.json'), issued_unix=time.time())
    path = OUT/'research_readiness.json'
    if path.exists():
        old = read(path)
        assert old['sources'] == source and old['protocol_sha256'] == value['protocol_sha256']
    else: write(path, value)
    return value

def validate(p, source=None):
    r = read(OUT/'research_readiness.json')
    assert r['status'] == 'ready_for_quality_experiment'
    assert r['protocol_sha256'] == sha(OUT/'protocol.json') and r['parent_sha256'] == p['parent']['sha256']
    assert r['sources'] == (sources() if source is None else source)
    assert r['numerical_replay'] == 'warning'
    for e in r['evidence'].values(): bound(e)
    return r

if __name__ == '__main__': print(json.dumps(issue(), ensure_ascii=False))
