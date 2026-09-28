"""One explicit run root, immutable source identities, and actual/effective accounting."""
from pathlib import Path
import hashlib
import json
import sys
import time
import os
import fcntl

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN_ID = HERE.name
ORIGIN = Path('/home/cai_tianshun/Project/4dsr')
OLD = ROOT / 'output/dynamic_sr_sync_multiview_20260928'
OUT = Path(os.environ.get('FOURDSR_RUN_ROOT', str(ROOT / 'output' / RUN_ID)))
if not OUT.is_absolute(): OUT = ROOT / OUT
OUT = OUT.resolve()
assert OUT != OLD.resolve()

def local(path):
    p = Path(path)
    if not p.is_absolute(): return ROOT / p
    try: return ROOT / p.relative_to(ORIGIN)
    except ValueError: return p

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def write(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    t.replace(p)

def entry(p):
    p = Path(p)
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p))

def bound(e):
    p = local(e['path']); assert sha(p) == e['sha256'], p
    return p

def setup():
    import importlib.util
    path=ROOT/'experiments/dynamic_sr_prior_guidance_20260927/shared.py'
    if 'shared' in sys.modules:
        shared=sys.modules['shared'];assert Path(shared.__file__).resolve()==path.resolve()
    else:
        spec=importlib.util.spec_from_file_location('shared',path)
        shared=importlib.util.module_from_spec(spec);sys.modules['shared']=shared;spec.loader.exec_module(shared)
    policy_path=ROOT/'experiments/dynamic_sr_prior_guidance_20260927/gradient_policy.py'
    if 'gradient_policy' not in sys.modules:
        spec=importlib.util.spec_from_file_location('gradient_policy',policy_path)
        policy=importlib.util.module_from_spec(spec);sys.modules['gradient_policy']=policy;spec.loader.exec_module(policy)
    assert Path(sys.modules['gradient_policy'].__file__).resolve()==policy_path.resolve()
    expected={'common':'dynamic_sr_20260918/common.py','n3dv_data':'dynamic_sr_20260918/n3dv_data.py',
        'training_support':'dynamic_sr_detail_supervision_20260924/training_support.py',
        'motion_model':'dynamic_sr_motion_bound_20260923/motion_model.py'}
    for name,rel in expected.items():
        assert Path(sys.modules[name].__file__).resolve()==(ROOT/'experiments'/rel).resolve(),name
    sys.path.insert(0, str(HERE))
    return shared

def sources():
    paths = [HERE/n for n in ['dv_common.py','schedule.py','sr_attribute_router.py','temporal_prior.py','train.py','readiness.py','runtime_identity.py']]
    dependencies = {
        'dynamic_sr_dynamic_validation_20260927': ['routing.py', 'train.py', 'train76.py', 'evaluate_endpoint.py'],
        'dynamic_sr_prior_guidance_20260927': ['gradient_policy.py', 'shared.py'],
        'dynamic_sr_detail_supervision_20260924': ['training_support.py', 'detail_loss.py', 'evaluate.py'],
        'dynamic_sr_motion_bound_20260923': ['motion_model.py'],
        'dynamic_sr_20260918': ['common.py', 'n3dv_data.py', 'run_experiment.py', 'evaluate.py'],
        'dynamic_sr_20260920': ['resume_control.py'],
        'dynamic_sr_controlled_headroom_20260926': ['evaluate.py'],
    }
    for folder, names in dependencies.items():
        paths.extend(ROOT / 'experiments' / folder / n for n in names)
    result = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}
    default_upstream = ROOT/'vendor/4dgs' if (ROOT/'vendor/4dgs').exists() else Path('/home/cai_tianshun/Project/4dgs')
    upstream = Path(os.environ.get('FOURDSR_UPSTREAM', str(default_upstream)))
    for n in ['scene/gaussian_model.py', 'scene/deformation.py', 'scene/hexplane.py', 'gaussian_renderer/__init__.py']:
        result['upstream/' + n] = sha(upstream / n)
    return result

def require_run_root(path):
    assert local(path).resolve() == OUT, (path, OUT)
    p = read(OUT / 'protocol.json')
    assert p['run_id'] == RUN_ID and local(p['run_root']).resolve() == OUT
    return p

def budget_update(fn):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = OUT / 'budget.json'
        d = read(path) if path.exists() else dict(actual_updates=0, completed_update_rounds=0,
            effective_updates=0, adam_calls=0, training_rgb_forwards=0,
            by_attempt={}, committed_steps={}, historical_engineering_updates=32, new_engineering_updates=0)
        fn(d); d['updated_unix'] = time.time(); write(path, d)

def debit(kind, label, updates=0, adam=0, rgb=0, completed=0):
    assert kind == 'formal'
    def update(d):
        cap = read(OUT / 'protocol.json')['budget']['actual_formal_max']
        assert d['actual_updates'] + updates <= cap, 'Actual formal update budget exhausted'
        d['actual_updates'] += updates; d['completed_update_rounds'] += completed
        d['adam_calls'] += adam; d['training_rgb_forwards'] += rgb
        r = d['by_attempt'].setdefault(label, dict(actual_updates=0, completed_update_rounds=0, adam_calls=0, rgb_forwards=0))
        r['actual_updates'] += updates; r['completed_update_rounds'] += completed
        r['adam_calls'] += adam; r['rgb_forwards'] += rgb
    budget_update(update)

def commit_progress(task_id, step):
    def update(d):
        assert 6000 <= step <= 12000
        d['committed_steps'][task_id] = max(step, d['committed_steps'].get(task_id, 6000))
        d['effective_updates'] = sum(s - 6000 for s in d['committed_steps'].values())
        assert d['effective_updates'] <= read(OUT / 'protocol.json')['budget']['effective_formal_max']
    budget_update(update)
