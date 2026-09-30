"""Freeze one from-initialization, same-observation LR/SR training run."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/dynamic_sr_same_observation_20260930'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / 'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    teacher_path = ROOT / 'output/dynamic_sr_detail_supervision_20260924/teacher_inventory_v1/cook_spinach/teacher_index.json'
    teacher = json.loads(teacher_path.read_text())
    assert sha(manifest_path) == teacher['manifest_sha256']
    by_key = {(r['camera'], r['frame']): r for r in teacher['entries']}
    observations = []
    data_root = manifest_path.parent
    for r in manifest['observations']:
        if r['split'] != 'train':
            continue
        t = by_key[r['camera_id'], r['frame_index']]
        assert r['lr_sha256'] == t['lr_sha256']
        item = dict(camera=r['camera_id'], frame=r['frame_index'],
                    lr_path=r['lr_path'], lr_sha256=r['lr_sha256'],
                    sr_path=t['relative_path'], sr_sha256=t['sha256'])
        assert sha(data_root / item['lr_path']) == item['lr_sha256']
        assert sha(data_root / item['sr_path']) == item['sr_sha256']
        observations.append(item)
    assert len(observations) == 1140
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in [
        Path(__file__), Path(__file__).with_name('train.py'),
        ROOT / 'experiments/dynamic_sr_20260918/common.py',
        ROOT / 'experiments/dynamic_sr_20260918/n3dv_data.py']}
    protocol = dict(schema=1, method='P1-from-start', scene=manifest['scene'],
                    manifest_path=manifest_path.relative_to(ROOT).as_posix(),
                    manifest_sha256=sha(manifest_path),
                    initialization=manifest['initialization'],
                    teacher_index_path=teacher_path.relative_to(ROOT).as_posix(),
                    teacher_index_sha256=sha(teacher_path), teacher=teacher['teacher_config'],
                    observations=observations, sources=sources,
                    training=dict(seed=20260918, coarse_steps=1000, fine_steps=19200,
                                  total_updates=20200, max_points=120000, sr_weight=0.1,
                                  physical_gpu='1', checkpoints=[7000, 14200, 20200],
                                  render_calls_per_update=1, backward_calls_per_update=1,
                                  optimizer_calls_per_update=1),
                    loss='mean(abs(D(R[c,t])-LR[c,t])) + 0.1*mean(abs(R[c,t]-SwinIR[c,t])) + original fine-stage regularization',
                    initialization_policy='Original train-LR-only points and new Gaussian/deformation parameters; no trained checkpoint',
                    stages='Both LR and SR active from coarse step1 and every fine step; coarse omits deformation as original backbone',
                    densification='Original warmup schedule: 300<stage_step<min(4000,stage_steps-500); densify every100 while count<120000; soft point limit',
                    comparison='One independent training run; historical two-suffix averages share U6000; matched total updates do not match capacity or computation')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    print(json.dumps(dict(protocol=str(OUT/'protocol.json'), inputs=len(observations),
                          sources=sources), ensure_ascii=False))


if __name__ == '__main__':
    main()
