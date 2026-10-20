"""Freeze the original-LR two-scene comparison and its input identities on CPU."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'output/dynamic_lr_baselines_20261020'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    records = {}
    scenes = []
    for name, label in [('cook_spinach', 'Cook'), ('cut_roasted_beef', 'Cut')]:
        directory = ROOT / 'data/dynamic_sr/full_time_20261007/prepared' / name
        path = directory / 'manifest_train_ready_seed20261007.json'
        manifest = json.loads(path.read_text())
        assert manifest['frame_indices'] == list(range(300))
        assert manifest['splits']['test'] == ['cam00']
        assert manifest['resolutions'] == {'hr': [1344, 1008], 'lr': [336, 252]}
        assert 'physical_lr_derivation' not in manifest
        assert manifest['initialization']['train_only_lr']
        records[str(path.relative_to(ROOT))] = dict(sha256=sha(path), role='metadata', bytes=path.stat().st_size)
        for observation in manifest['observations']:
            if observation['split'] == 'train':
                key, role = 'lr', 'training_LR'
            elif observation['split'] == 'test':
                key, role = 'hr', 'evaluation_only_HR'
            else:
                continue
            asset = directory / observation[key + '_path']
            records[str(asset.relative_to(ROOT))] = dict(sha256=observation[key + '_sha256'], role=role, bytes=asset.stat().st_size)
        init = manifest['initialization']
        for key in ('npz', 'ply'):
            asset = directory / init[key + '_path']
            records[str(asset.relative_to(ROOT))] = dict(sha256=init[key + '_sha256'], role='existing_train_LR_initialization_optional_per_method', bytes=asset.stat().st_size)
        scenes.append(dict(scene=name, label=label, manifest=str(path.relative_to(ROOT)), manifest_sha256=sha(path),
                           train_cameras=manifest['splits']['train'], test_cameras=['cam00'], frames=300,
                           time_convention=manifest['time_convention'], resolutions=manifest['resolutions'],
                           initialization=init, training_LR_count=300 * len(manifest['splits']['train'])))

    def check(item):
        name, entry = item
        actual = sha(ROOT / name)
        if actual != entry['sha256']:
            raise ValueError('Input SHA256 mismatch: ' + name)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, records.items()))
    inventory = dict(schema='dynamic_lr_input_inventory_v1', files=records,
                     file_count=len(records), total_bytes=sum(x['bytes'] for x in records.values()),
                     verified_local=True)
    inventory_path = HERE / 'input_inventory.json'
    inventory_path.write_text(json.dumps(inventory, indent=2) + '\n')
    protocol = dict(schema='dynamic_lr_original_comparison_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                    authorization='2026-10-20 user explicitly requests all six dynamic reconstruction methods on CTS and A100; Cook and Cut only; original LR and no proposed gain modules; supervise through completion.',
                    original_LR=True, use_area4_regenerated_inputs=False, scenes=scenes,
                    forbidden_training_additions=['SR_teacher', 'SR_loss', 'HR_training_pixels', 'test_camera_pixels',
                        'proposed_image_displacement', 'proposed_affine_color', 'proposed_visibility_cull', 'HR_render_then_area_LR_loss'],
                    methods=['Wu4DGS', 'Yang4DGS', '4C4D', 'Multi4D', 'LocalDyGS', 'CEM4DGS'],
                    initialization_policy='Author-specific training-LR initialization; disclose data-interface and initialization adaptations. No HR or held-out pixels.',
                    endpoint_policy='Author schedule; no selection using test metrics. Reuse completed identity-matched clean endpoints instead of duplicate training.',
                    evaluation=dict(view='cam00', resolution=[1344,1008], frames=list(range(300)),
                                    metrics=['PSNR', 'SSIM', 'LPIPS'], same_evaluator=True,
                                    fixed_visual_frames=[0,150,299], include_whole_image_and_predefined_face_ROI=True,
                                    cross_hardware_training_disclosed=True),
                    inventory=dict(path=str(inventory_path.relative_to(ROOT)), sha256=sha(inventory_path)),
                    status='inputs_verified_locally_methods_preparing')
    (HERE / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(status=protocol['status'], scenes=[x['scene'] for x in scenes],
                         files=len(records), total_bytes=inventory['total_bytes'])))


if __name__ == '__main__':
    main()
