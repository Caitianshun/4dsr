"""Re-render the cam00 test views of the 6 main-table x4 baseline cells whose 300 test PNGs bl4_cleanup deleted on A100 after
scoring (Claude 4dsr-a4, 2026-11-02): Yang / 4C4D on meetroom_discussion, meetroom_vrheadset (strict4d runs) and Multi4D on the
main-table reruns Discussion07_a2, VRHeadset07_a2. Needed for the temporal read-out and the shared qualitative-figure tree.

Same frozen renderers and the same plan wrappers as bl4_eval_strict4d.render_scope / bl4_multi4d_v2.render_scope (imported, not
copied), same final checkpoints (sha256 checked against the cell's cleanup_receipt.json), same environments; only the output
directory is new:  output/dynamic_sr_modules_20261010/qualfig_20261102/rerender/<cell>/render_test
Afterwards every PNG is compared with render_sha256 of the original render_index.json (kept by bl4_cleanup) -> compare.json.
  rerender_bl4.py run CELL [CELL ...]          host .venv python: per cell sets the original env, spawns the method python
  rerender_bl4.py inner CELL                   (internal) the render itself
CELL: yang:meetroom_discussion | yang:meetroom_vrheadset | 4c4d:<same> | multi4d:meetroom_discussion | multi4d:meetroom_vrheadset
"""
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BL4 = HERE.parent / 'bl4_20261024'
OUT = ROOT / 'output/dynamic_sr_modules_20261010/qualfig_20261102/rerender'
SCENES = ('meetroom_discussion', 'meetroom_vrheadset')
LABEL = dict(meetroom_discussion='Discussion07', meetroom_vrheadset='VRHeadset07')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def parse(cell):
    m, s = cell.split(':')
    if m not in ('yang', '4c4d', 'multi4d') or s not in SCENES:
        raise SystemExit(f'unknown cell {cell}')
    return m, s


def run_dir(m, s):
    if m == 'multi4d':
        return ROOT / 'output/dynamic_sr_modules_20261010/bl4_20261024/multi4d/runs' / f'{LABEL[s]}_a2'
    return ROOT / 'output/dynamic_sr_modules_20261010/bl4_20261024/strict4d/runs' / f'{m}_{s}'


def checkpoint(m, s):
    run = run_dir(m, s)
    ck = run / ('train/fine_20000.pt' if m == 'multi4d' else 'model/chkpnt30000.pth')
    receipt = json.loads((run / 'cleanup_receipt.json').read_text())
    want = receipt['final_checkpoint']['sha256']
    if Path(receipt['final_checkpoint']['path']).name != ck.name or sha(ck) != want:
        raise SystemExit(f'{m}:{s} final checkpoint differs from cleanup_receipt.json')
    return ck, want


def inner(cell):
    m, s = parse(cell)
    sys.path.insert(0, str(BL4))
    out = OUT / f'{m}_{s}' / 'render_test'
    if out.exists():
        raise FileExistsError(f'preserve existing re-render {out}')
    ck, _ = checkpoint(m, s)
    if m == 'multi4d':
        V = importlib.import_module('bl4_multi4d_v2')
        path = V.HERE / 'frozen' / V.FROZEN_RENDER[0]
        if not V.sha(path).startswith(V.FROZEN_RENDER[1]):
            raise ValueError('frozen render_wu_multi.py copy changed')
        spec = importlib.util.spec_from_file_location('render_wu_multi_frozen', path)
        R = importlib.util.module_from_spec(spec)
        sys.modules['render_wu_multi_frozen'] = R
        spec.loader.exec_module(R)
        R.ROOT = V.ROOT
        R.OLD = V.ROOT / 'deployment/dynamic_lr_baselines_20261020/methods/wu_multi'
        args = argparse.Namespace(spec=V.spec_path(s), scope='test', out=out, checkpoint=ck, frames=None, execute=True,
                                  plan_only=False, cpu_threads=4)
        orig, hits = R.json.loads, []

        def loads(text, *a, **k):  # identical to bl4_multi4d_v2.render_scope
            v = orig(text, *a, **k)
            if isinstance(v, dict) and 'observations' in v and (v.get('physical_lr_derivation') or {}).get('formation') == 'area4':
                hits.append(v['physical_lr_derivation'])
                v = {k2: x for k2, x in v.items() if k2 != 'physical_lr_derivation'}
            return v
        R.json.loads = loads
        try:
            spec_d, geometry, rows, ckp, expected = R.plan(args)
        finally:
            R.json.loads = orig
        if len(hits) != 1:
            raise RuntimeError('expected exactly one area4 manifest read in plan()')
        expected.update(input_protocol='area4', physical_lr_derivation=hits[0],
                        plan_wrapper=dict(path=str(Path(V.__file__).resolve().relative_to(ROOT)), sha256=V.sha(Path(V.__file__))),
                        rerender=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(Path(__file__))))
        out.mkdir(parents=True)
        R.write(out / 'expected_views.json', expected)
        R.execute(args, spec_d, geometry, rows, ckp, expected)
    else:
        E = importlib.import_module('bl4_eval_strict4d')
        R = E.load_frozen()
        run = run_dir(m, s)
        args = argparse.Namespace(project_root=E.ROOT, method=m, runtime=E.ROOT / E.BASE / 'runtime' / E.RTN[m],
                                  dataset=E.ROOT / E.OUTB / f'inputs_{m}' / s,
                                  manifest=E.ROOT / f'data/dynamic_sr/full_time_20261007/prepared/{s}_area4/manifest_train_ready_seed20261007.json',
                                  checkpoint=ck, checkpoint_sha256=None, config=run / 'native_config.yaml',
                                  scope='test', output=out, plan_only=False, execute=True)
        plan, mm, config = E.make_plan_area4(R, args)
        plan['rerender'] = dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(Path(__file__)))
        out.mkdir(parents=True)
        R.write_json(out / 'expected_views.json', plan, immutable=True)
        R.execute(args, plan, mm, config)


def environment(m):
    env = os.environ.copy()
    if m == 'multi4d':
        sys.path.insert(0, str(BL4))
        V = importlib.import_module('bl4_multi4d_v2')
        return V.environment(), V.PY_MULTI
    env.update(PYTHONNOUSERSITE='1', TORCH_CUDA_ARCH_LIST='8.0', OMP_NUM_THREADS='4', MAX_JOBS='2',
               TORCH_EXTENSIONS_DIR=str(ROOT / f'.cache/strict4d_extensions_20261020/{m}'))   # as bl4_eval_strict4d.main
    env.setdefault('CUDA_HOME', str(ROOT / 'tools/cuda-12.8-minimal'))
    env['PATH'] = env['CUDA_HOME'] + '/bin:' + env.get('PATH', '/usr/bin:/bin')
    env['LD_LIBRARY_PATH'] = ':'.join([str(ROOT / 'tools/runtime-libs'), env['CUDA_HOME'] + '/lib64', env.get('LD_LIBRARY_PATH', '')])
    return env, ROOT / f'envs/dynamic_lr_baselines_20261020/{m}/bin/python'


def compare(m, s):
    orig = json.loads((run_dir(m, s) / 'eval_area4/render_test/render_index.json').read_text())
    new_dir = OUT / f'{m}_{s}' / 'render_test'
    new = json.loads((new_dir / 'render_index.json').read_text())
    a = {r['path']: r['render_sha256'] for r in orig['records'] if r['camera_id'] == 'cam00'}
    b = {r['path']: r['render_sha256'] for r in new['records'] if r['camera_id'] == 'cam00'}
    actual = {p: sha(new_dir / p) for p in b}
    rec = dict(cell=f'{m}:{s}', frames_original=len(a), frames_new=len(b), index_matches_files=all(actual[p] == b[p] for p in b),
               identical_to_original=sum(a.get(p) == b[p] for p in b), same_paths=set(a) == set(b),
               original_index_sha256=sha(run_dir(m, s) / 'eval_area4/render_test/render_index.json'),
               checkpoint_sha256=checkpoint(m, s)[1], time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (OUT / f'{m}_{s}' / 'compare.json').write_text(json.dumps(rec, indent=1) + '\n')
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('run', 'inner'))
    p.add_argument('cells', nargs='+')
    a = p.parse_args()
    if a.mode == 'inner':
        return inner(a.cells[0])
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise SystemExit('the dispatcher must bind CUDA_VISIBLE_DEVICES')
    bad = 0
    for cell in a.cells:
        m, s = parse(cell)
        env, py = environment(m)
        (OUT / f'{m}_{s}').mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        with open(OUT / f'{m}_{s}' / 'render_test.log', 'a') as log:
            r = subprocess.run([str(py), str(Path(__file__)), 'inner', cell], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT)
        rec = dict(cell=cell, rc=r.returncode, seconds=round(time.monotonic() - t0, 1))
        if r.returncode == 0:
            rec.update(compare(m, s))
        print(json.dumps(rec), flush=True)
        bad |= r.returncode != 0
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
