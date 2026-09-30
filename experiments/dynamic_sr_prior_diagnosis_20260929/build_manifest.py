"""Register verified historical assets before diagnosis; never fabricate runs."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/dynamic_sr_prior_diagnosis_20260929'
OLD = ROOT / 'output/dynamic_sr_temporal_prior_20260928'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def verify(e):
    p = Path(e['path'])
    if not p.is_absolute(): p = ROOT / p
    if p.is_absolute() and not p.exists() and '/Project/4dsr/' in str(p):
        p = ROOT / str(p).split('/Project/4dsr/', 1)[1]
    assert p.is_file(), p
    digest = sha(p); assert digest == e['sha256'], p
    return dict(path=str(p.relative_to(ROOT)), sha256=digest, bytes=p.stat().st_size)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = read(OLD / 'protocol.json')
    names = ['protocol.json', 'baseline_registry.json', 'prior_index.json',
             'checkpoint_index.json', 'main_quality.csv', 'quality_gaps.csv', 'metrics_per_frame.csv']
    required = {n: dict(path=str((OLD/n).relative_to(ROOT)), sha256=sha(OLD/n)) for n in names}
    assets = {k: verify(p[k]) for k in ['manifest', 'parent', 'teacher', 'lr_curve', 'old_schedule', 'roi']}
    assets['schedules'] = {k: verify(v) for k,v in p['schedules'].items()}
    baselines = read(OLD / 'baseline_registry.json')
    models = {}
    for label in ['LR-direct-HRrender', 'HR-direct-6k']:
        record = baselines['references'][label]
        models[label] = {**record, 'checkpoint': verify(record['checkpoint'])}
    models['U6000'] = dict(checkpoint=assets['parent'], points=132972, total_updates=14200)
    for rep, item in p['historical_J1'].items():
        models[f'r{rep}_J1'] = dict(checkpoint=verify(item['checkpoint']),
                                  endpoint=verify(item['endpoint']), points=132972, total_updates=20200)
    for label, item in p['historical_multiview'].items():
        ep = read(ROOT/item['endpoint']['path'])
        ck = ROOT/'output/dynamic_sr_sync_multiview_20260928/runs'/label/'attempt_01/train/checkpoint_12000.pt'
        receipt = read(ROOT/item['receipt']['path'])
        assert ck.is_file() and sha(ck) == receipt['checkpoint_sha256']
        models[label] = dict(checkpoint=dict(path=str(ck.relative_to(ROOT)), sha256=sha(ck)),
                             endpoint=verify(item['endpoint']), points=132972, total_updates=20200)
    publish = ROOT.parent/'4dsr-github'
    head = subprocess.check_output(['git','-C',str(publish),'rev-parse','HEAD'], text=True).strip()
    diff = subprocess.check_output(['git','-C',str(publish),'status','--short'], text=True)
    result = dict(status='verified', created_unix=time.time(), source_plan=dict(
        path=str((OUT/'source_plan/dynamic_sr_prior_diagnosis_execution_2026-09-29.codex.md').relative_to(ROOT)),
        sha256=sha(OUT/'source_plan/dynamic_sr_prior_diagnosis_execution_2026-09-29.codex.md')),
        public_head=head, publication_clone_initial_diff=diff, inherited=required,
        assets=assets, models=models, diagnosis_models=['LR-direct-HRrender','HR-direct-6k','U6000','r1_J1','r1_Async2'],
        information_boundary='legal train LR/SwinIR/LR-only priors separated from HR and dev diagnostic assets',
        observation_protocol=dict(dev_cameras=['cam00','cam01'],dev_frames=list(range(0,120,2)),
            train_cameras=p['evaluation']['train_cameras'],train_frames=[0,40,80,118],
            image_prior_observations=92,flow_cameras=['cam02','cam06','cam12','cam18'],
            flow_pairs=[[38,40],[40,42],[78,80],[80,82]],flow_both_directions=True),
        optimization=dict(parent_step=6000,stop=12000,save=[9000,12000],points=132972,
            sr_weight=0.1,lr_offset=7200,repeats=2,maximum_new_effective_updates=36000,
            branch_decision='diagnosis first; T and at most one G/S; no extra capacity arm'),
        primary_metrics=['psnr','ssim','lpips'], development_views=True,
        complete_from_scratch_independent_seeds=False)
    write(OUT/'audit_manifest.json',result)
    shutil.copyfile(OLD/'baseline_registry.json', OUT/'baseline_registry.json')
    print(json.dumps(dict(status='verified',models=list(models),head=head)))

if __name__ == '__main__': main()
