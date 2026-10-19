"""Register immutable stage specs and a bounded one-GPU STG execution plan.

Does not upload files, reserve a GPU or launch a child. The run pipeline requires
the existing remote broker and remote SHA256 verification before execution.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def prepare(mode, destination, output, spec_root, input_verification,
            gpu_uuid='GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b'):
    if destination.exists():
        raise FileExistsError('Preserve registered plans; select a new plan path')
    paths = list(HERE.glob('*.py')) + [
        ROOT/'experiments/dynamic_sr_multi4d_area_20261017/evaluate.py',
        ROOT/'experiments/dynamic_sr_20260918/evaluate.py',
        ROOT/'experiments/dynamic_sr_same_observation_20260930/frequency.py',
        ROOT/'output/dynamic_sr_stg_area_20261018/adapter/adapted_train.py',
        ROOT/'output/dynamic_sr_stg_area_20261018/adapter/helper_model.py',
        ROOT/'output/dynamic_sr_stg_area_20261018/adapter/identity.json',
    ]
    stages = []
    for start in ([0] if mode == 'smoke' else range(0, 300, 50)):
        for method in ('areaL0', 'areaTm', 'areaOracle'):
            stem = f'Cut07_f{start:03d}_{start+49:03d}_{method}'
            spec = spec_root / (stem + '.json')
            stages.append(dict(name=stem, spec=str(spec.relative_to(ROOT)), spec_sha256=sha(spec),
                               segment_start=start, method=method,
                               relative_output=f'{method}/f{start:03d}_{start+49:03d}',
                               iterations=40 if mode == 'smoke' else 25000))
    plan = dict(schema='stg_pipeline_plan_v1', mode=mode, host='oran-A100-Ubuntu',
                gpu_uuid=gpu_uuid, output=output,
                input_verification=input_verification,
                python_paths=['third_party/stg_20261018/pydeps', 'third_party/stg_20261018/runtime',
                              'third_party/stg_20261018/runtime/thirdparty/gaussian_splatting'],
                torch_home='.cache/torch', cuda_home='tools/cuda-12.8-minimal',
                max_seconds=3600 if mode=='smoke' else 129600,
                max_stage_seconds=600 if mode=='smoke' else 7200, minimum_free_bytes=12 << 30,
                stages=stages,
                source_files=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in sorted(set(paths))],
                budget='Cut only; 6 independent 50-frame models per arm, no hyperparameter search; Cook needs registered comparison gate',
                resume='forbidden; native PLY+decoder supports rendering only',
                preparation_only=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(plan=str(destination), sha256=sha(destination), stages=len(stages))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('smoke','formal'), required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--spec-root', type=Path, default=ROOT/'output/dynamic_sr_stg_area_20261018/specs_v1')
    p.add_argument('--input-verification', required=True)
    p.add_argument('--gpu-uuid', default='GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b')
    a = p.parse_args()
    prepare(a.mode, a.plan.resolve(), a.output, a.spec_root.resolve(), a.input_verification, a.gpu_uuid)
