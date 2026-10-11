"""Training-view SR image versus reconstruction diagnostics, without updates.

CPU preflight validates existing metadata and selected image hashes before any
runtime/model import. --check-only prints that receipt without creating --out.
Formal execution reuses cullretrain's parameter-only loader, renderer, culler
and frozen legacy PSNR/SSIM/AlexNet LPIPS helpers. This is training-view fitting
evidence, separate from heldout novel-view evaluation and from optimization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, OUT, CONDITIONS, entry, local, mkdir, read, sha, write

FRAMES = (15, 150, 285)
FROZEN = dict(
    generator_sha256='d7be23d15473082c57e1e4091bfea9eaafad0de97ef7379db39308ae14c6cbcc',
    network_sha256='1d650d2c1c4519d95db771863691c6b239a0822222f287f6a8342a9e0eccbc6e',
    checkpoint_sha256='4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215',
    scale=4, window=8, tile=0, overlap=32, precision='float32',
    padding='official_mirror_concat_next_window')


def require(value, message):
    if not value:
        raise ValueError(message)


def child(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute(), 'Absolute image child path refused')
    target = (Path(root) / relative).resolve()
    require(target.is_relative_to(Path(root).resolve()), 'Image child escapes source root')
    return target


def bound(ref):
    path = local(ref['path'])
    require(sha(path) == ref['sha256'], f'Bound source changed: {path}')
    return path


def png_size(path):
    with Path(path).open('rb') as f:
        header = f.read(24)
    require(header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR', 'PNG image required')
    return list(struct.unpack('>II', header[16:24]))


def keys(rows, camera='camera', frame='frame'):
    values = [(r[camera], int(r[frame])) for r in rows]
    require(len(values) == len(set(values)), 'Duplicate camera/frame population')
    return set(values)


def validate_population(manifest, samples, dataset_config, teacher_index):
    """Pure metadata checks; usable in CPU negative interface acceptance."""
    train = manifest['splits']['train']
    require(train and 'cam00' not in train and len(train) == len(set(train)), 'Invalid training camera boundary')
    require(set(train) == set(manifest['cameras']) - {'cam00'}, 'Prepared training cameras omitted')
    require(manifest['splits']['test'] == ['cam00'] and not manifest['splits'].get('dev'), 'Unexpected heldout split')
    require(manifest['scene'] in CONDITIONS, 'Fresh highres condition required')
    width, height = CONDITIONS[manifest['scene']]
    require(manifest['schema'] == 'n3dv_dynamic_sr_pilot_v1' and manifest.get('full_time_decode_verified') is True and manifest['frame_indices'] == list(range(300)), 'Prepared full-time manifest required')
    require(manifest['resolutions'] == dict(hr=[width,height],lr=[width//4,height//4]), 'Prepared condition grid differs')
    require(samples.get('test_views_used') is False and samples.get('teacher_cache') is False, 'Z5 diagnostic role differs')
    require(dataset_config['id'] == samples['dataset'], 'Z5 dataset identity differs')
    require(len(train) == {'DeskGames':20,'Long360':32}[dataset_config['dataset']], 'Published training-camera count differs')
    require(dataset_config['rungs'].get(samples['rung']) == manifest['scene'], 'Z5 condition/rung differs')
    require(samples['start'] == manifest['original_start'], 'Z5 selected window differs')
    require(dataset_config['source'] == manifest['source'], 'Z5 source provenance differs')
    require(teacher_index['status'] == 'completed_teacher_inventory', 'Full accepted teacher inventory required')
    require(teacher_index['scene'] == manifest['scene'], 'Teacher scene differs')
    require(teacher_index['train_cameras'] == train and teacher_index['frame_indices'] == list(range(300)), 'Teacher train/time boundary differs')
    expected_full = {(c, f) for c in train for f in range(300)}
    require(keys(teacher_index['entries']) == expected_full, 'Incomplete or heldout teacher population')
    observations = [r for r in manifest['observations'] if r['split'] == 'train']
    require(keys(observations, 'camera_id', 'frame_index') == expected_full, 'Prepared train LR population differs')
    require(keys(samples['entries']) == {(c, f) for c in train for f in FRAMES}, 'Z5 must contain every train camera at15/150/285 only')
    for row in samples['entries']:
        require(row['source_frame'] == manifest['original_start'] + int(row['frame']), 'Z5 source frame differs')
        require(row['original_camera'] == manifest['cameras'][row['camera']]['original_number'], 'Z5 original camera mapping differs')
        other = row['frame'] + 30 if row['frame'] + 30 < 300 else row['frame'] - 30
        require(row['companion_frame'] == other, 'Z5 companion time differs')
    return train


def preflight(a):
    checkpoint, baseline, samples_path, hr_root = map(local, (a.checkpoint, a.baseline_sr, a.z5_samples, a.hr_root))
    config_path = baseline / 'config.json'
    config = read(config_path)
    manifest_path = bound(config['data']['manifest'])
    manifest = read(manifest_path)
    samples = read(samples_path)
    dataset_path = bound(samples['dataset_config'])
    dataset = read(dataset_path)
    index_path = bound(config['dependencies']['teacher'])
    index = read(index_path)
    train = validate_population(manifest, samples, dataset, index)
    require(index['manifest_sha256'] == sha(manifest_path), 'Teacher seed manifest differs')
    require(index['seed_consumer'] == config['seed'] and config['scene'] == manifest['scene'], 'Teacher seed consumer differs')
    require(index['producer_manifest'] == manifest['original_full_manifest'], 'Teacher producer manifest differs')
    producer = bound(index['producer_manifest'])
    acceptance_path = index_path.parent / 'acceptance_complete.json'
    acceptance = read(acceptance_path)
    require(acceptance.get('status') == 'completed_full_teacher_CPU_acceptance', 'Full teacher acceptance not complete')
    require(entry(index_path) in acceptance['indexes'] and acceptance['teacher_images'] == len(index['entries']), 'Teacher acceptance/index binding differs')
    require(acceptance['producer_manifest'] == index['producer_manifest'], 'Teacher acceptance producer differs')
    cfg_path = bound(index['producer_config'])
    teacher_config = read(cfg_path)
    require(teacher_config == index['teacher_config'] and sha(cfg_path) == index['teacher_config_sha256'], 'Teacher cache configuration differs')
    require(all(teacher_config.get(k) == v for k, v in FROZEN.items()), 'Frozen teacher recipe differs')
    require(teacher_config['manifest_sha256'] == sha(producer), 'Teacher cache producer differs')
    generation = read(bound(index['producer_summary']))
    require(generation.get('selection_complete') is True and generation.get('selected') == len(index['entries']), 'Teacher generation population incomplete')
    for teacher in index['entries']:
        target = child(manifest_path.parent,teacher['relative_path'])
        require(target.is_file() and target.with_suffix('.json').is_file(), 'Accepted full teacher cache is now incomplete')
    storage_path = hr_root / 'hr_full_storage.json'
    storage = read(storage_path)
    require(storage['manifest'] in (entry(manifest_path), manifest['original_full_manifest']), 'HR storage belongs to another manifest/window')
    bound(storage['manifest'])
    require(storage.get('LR_roundtrip_all_frames_SHA_equal') is True, 'HR full LR roundtrip not verified')
    require(set(storage['cameras']) == set(train), 'HR full training camera inventory differs')
    for c in train:
        require(storage['cameras'][c]['frames'] == 300 and set(storage['cameras'][c]['hr_sha256']) == {str(f) for f in range(300)}, 'HR full frame inventory incomplete')
    observations = {(r['camera_id'], int(r['frame_index'])): r for r in manifest['observations'] if r['split'] == 'train'}
    teachers = {(r['camera'], int(r['frame'])): r for r in index['entries']}
    selected = []
    for sample in sorted(samples['entries'], key=lambda r: (r['camera'], r['frame'])):
        key = (sample['camera'], int(sample['frame']))
        obs, teacher = observations[key], teachers[key]
        require(obs['time'] == key[1] / 300 and obs['original_frame_index'] == sample['source_frame'], 'Prepared observation time differs')
        for name, relative in sample['paths'].items():
            require(sha(child(samples_path.parent, relative)) == sample['sha256'][name], 'Z5 image changed')
        lr = child(manifest_path.parent, obs['lr_path'])
        sr = child(manifest_path.parent, teacher['relative_path'])
        hr = child(hr_root, obs['hr_path'])
        lr_sha, sr_sha, hr_sha = sha(lr), sha(sr), sha(hr)
        require(lr_sha == obs['lr_sha256'] == sample['sha256']['lr'] == teacher['lr_sha256'], 'Prepared LR / Z5 / teacher input SHA differs')
        require(sr_sha == teacher['sha256'] and sha(sr.with_suffix('.json')) == teacher['receipt_sha256'], 'Accepted teacher image/receipt changed')
        sr_record = read(sr.with_suffix('.json'))
        require(sr_record['input_sha256'] == lr_sha and sr_record['output_sha256'] == sr_sha, 'Teacher receipt source differs')
        require(hr_sha == storage['cameras'][key[0]]['hr_sha256'][str(key[1])] == sample['sha256']['hr'], 'HRfull / Z5 HR SHA differs')
        require(png_size(lr) == manifest['resolutions']['lr'] and png_size(sr) == png_size(hr) == manifest['resolutions']['hr'], 'Selected image dimensions differ')
        selected.append(dict(camera=key[0], frame=key[1], observation=obs,
                             sources=dict(LR=entry(lr), teacher=entry(sr), teacher_receipt=entry(sr.with_suffix('.json')), HR=entry(hr),
                                          Z5_HR=entry(child(samples_path.parent, sample['paths']['hr'])), Z5_LR=entry(child(samples_path.parent, sample['paths']['lr'])))))
    training_path = checkpoint.parent / 'run.json'
    training = read(training_path)
    require(training['scene'] == manifest['scene'] and training['seed'] == config['seed'], 'Checkpoint training scene/seed differs')
    parent_config = local(training['parent'])/'config.json' if training['kind'] == 'lr' else local(training['baseline_sr'])/'config.json'
    require(read(parent_config)['data']['manifest'] == entry(manifest_path), 'Checkpoint training manifest differs')
    checkpoint_ref = entry(checkpoint)
    require(read(checkpoint.with_suffix('.json'))['sha256'] == checkpoint_ref['sha256'], 'Checkpoint sidecar SHA differs')
    completed = read(checkpoint.parent/'complete.json')
    require(completed.get('status') == 'completed' and completed.get('checkpoint') == checkpoint_ref, 'Completed checkpoint endpoint differs')
    mod_path = checkpoint.parent/'modsr.json'
    require(not mod_path.exists() or not read(mod_path).get('time_full_range', False), 'Unsupported altered time-coordinate mapping')
    refs = dict(source=entry(__file__),checkpoint=checkpoint_ref,manifest=entry(manifest_path),baseline_sr=entry(config_path),
                teacher_index=entry(index_path),teacher_acceptance=entry(acceptance_path),hr_storage=entry(storage_path),z5_samples=entry(samples_path),
                training_run=entry(training_path),dataset_config=entry(dataset_path),producer_config=entry(cfg_path))
    receipt = dict(status='passed_CPU_prior_fit_preflight', scene=manifest['scene'],seed=config['seed'],train_cameras=train,
                   observations=len(selected),sampled_frames=list(FRAMES),population_keys=[[s['camera'],s['frame']] for s in selected],
                   training=False,CUDA_imports=0,checkpoint_loaded=False,optimizer_calls=0,parameter_updates=0,heldout_pixels_used=False,**refs)
    return dict(receipt=receipt, selected=selected, manifest=manifest, manifest_path=manifest_path, config=config, checkpoint=checkpoint)


class EvaluationAffine:
    """CameraAffine.apply exactly, from saved tensors; never constructs Adam."""
    def __init__(self, state, cameras, torch_module, device='cuda'):
        require(state['names'] == list(cameras), 'Affine camera list differs')
        self.torch = torch_module
        self.index = {c:i for i,c in enumerate(cameras)}
        self.delta = state['delta'].detach().to(device=device,dtype=torch_module.float32).clone().requires_grad_(False)
        require(tuple(self.delta.shape) == (len(cameras),3,4) and bool(torch_module.isfinite(self.delta).all()), 'Invalid saved affine tensor')
        self.eye = torch_module.eye(3,device=device,dtype=torch_module.float32)

    def apply(self, image, camera):
        d = self.delta[self.index[camera]] - self.delta.mean(0)
        return self.torch.einsum('ij,jhw->ihw',self.eye+d[:,:3],image) + d[:,3,None,None]


def run(a, checked):
    # All provenance/role checks above precede importing torch or a CUDA extension.
    import numpy as np
    import torch
    import torch.nn.functional as F
    import lpips
    sys.path.insert(0, str(ROOT/'experiments/dynamic_sr_cull_retrain_20261009'))
    import cullretrain as cr
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = local(a.out)
    require(out.is_relative_to(OUT) and not out.exists(), 'Fresh highres output directory required')
    mkdir(out)
    write(out/'preflight.json', checked['receipt'])
    started = time.monotonic()
    checkpoint = checked['checkpoint']
    payload_cpu = torch.load(checkpoint,map_location='cpu',weights_only=False)
    plan = payload_cpu['plan']
    require(plan['data']['manifest'] == checked['receipt']['manifest'], 'Embedded checkpoint manifest differs')
    del payload_cpu
    g, payload = cr.load_for_render(checkpoint, plan)
    affine_state = payload.get('affine')
    del payload
    _, _, pipe, _ = cr.configuration(plan)
    _, manifest = cr.plan_scene(plan)
    affine = None
    if affine_state is not None:
        affine = EvaluationAffine(affine_state,manifest['splits']['train'],torch)
    record = dict(checked['receipt'],schema='highres_train_prior_fit_v1',status='running_train_view_diagnostic',
                  posthoc_cull=bool(a.cull),host=socket.gethostname(),gpu=torch.cuda.get_device_name(),
                  runtime=cr.rt.check_runtime(),affine_corrected_reported=affine is not None,
                  optimizer_initializations=0,points_loaded=int(g.get_xyz.shape[0]))
    if a.cull:
        record['cull'] = cr.cull(g,cr.Culler(manifest),None,'posthoc',optimizer_backed=False)
    record['points_evaluated'] = int(g.get_xyz.shape[0])
    legacy, _, _, _ = cr.ev.metric_helpers()
    metric = lpips.LPIPS(net='alex',version='0.1',spatial=False,pnet_rand=False,pretrained=True,verbose=False).cuda().eval().requires_grad_(False)
    background = torch.ones(3,device='cuda') if cr.white_background(plan) else torch.zeros(3,device='cuda')
    write(out/'run.json',record)
    rows, per_image = [], []
    def quality(pred,target):
        empty = np.zeros(target.shape[:2],bool)
        q = legacy.spatial_metrics(pred,target,empty,metric,'cuda')['full']
        return dict(psnr=q['psnr'],ssim=q['ssim'],lpips=q['lpips_alex'],mse=q['mse'])
    with torch.no_grad():
        for uid, item in enumerate(checked['selected']):
            obs = item['observation']
            cam = cr.native.hr_camera(cr.ev.camera_item(manifest,obs,torch),uid,manifest)
            state = cr.native.native_effective_state(g,float(obs['time']),torch)
            raw = cr.native.native_rgb(cam,g,state,pipe,background)['render'].detach()
            lr, teacher, hr = [legacy.read_rgb(local(item['sources'][key]['path'])) for key in ('LR','teacher','HR')]
            pred = legacy.image_array(raw)
            require(pred.shape == teacher.shape == hr.shape, 'Rendered/teacher/HR image grids differ')
            x = torch.from_numpy(np.ascontiguousarray(lr.transpose(2,0,1)))[None]
            bicubic = F.interpolate(x,size=hr.shape[:2],mode='bicubic',align_corners=False).clamp(0,1)[0].permute(1,2,0).numpy()
            scores = dict(teacher_vs_HR=quality(teacher,hr),bicubic_vs_HR=quality(bicubic,hr),
                          render_raw_vs_HR=quality(pred,hr),render_raw_vs_teacher=quality(pred,teacher))
            hashes = dict(render_raw_float32_sha256=hashlib.sha256(np.ascontiguousarray(pred,dtype=np.float32).tobytes()).hexdigest())
            previews = {}
            preview_images = dict(teacher=teacher,render_raw=pred,HR=hr,bicubic=bicubic)
            if affine is not None:
                corrected = legacy.image_array(affine.apply(raw,item['camera']).clamp(0,1))
                scores.update(render_corrected_vs_HR=quality(corrected,hr),render_corrected_vs_teacher=quality(corrected,teacher))
                hashes['render_corrected_float32_sha256'] = hashlib.sha256(np.ascontiguousarray(corrected,dtype=np.float32).tobytes()).hexdigest()
                preview_images['render_corrected'] = corrected
            if item['camera'] == 'cam01':
                from PIL import Image
                mkdir(out/'previews')
                for name,pixels in preview_images.items():
                    path = out/'previews'/f"cam01_{item['frame']:04d}_{name}.png"
                    with path.open('xb') as f:
                        Image.fromarray((np.clip(pixels,0,1)*255+.5).astype(np.uint8)).save(f,format='PNG')
                    previews[name] = entry(path)
            row = dict(scope='train_prior_fit_diagnostic',camera=item['camera'],frame=item['frame'])
            for name, values in scores.items():
                row.update({name+'_'+k:v for k,v in values.items()})
            rows.append(row)
            image_record = dict(camera=item['camera'],frame=item['frame'],source_frame=obs['original_frame_index'],time=obs['time'],
                                source=entry(__file__),sources=item['sources'],rendered_pixel_hashes=hashes,metrics=scores,
                                previews=previews,
                                scope='training-view diagnostic only; no optimizer or parameter updates')
            path = write(out/'observations'/f"{item['camera']}_{item['frame']:04d}.json",image_record)
            per_image.append(entry(path))
            print(json.dumps(dict(stage='prior_fit',camera=item['camera'],frame=item['frame'])),flush=True)
    names = sorted(set().union(*(set(r) for r in rows)))
    with (out/'metrics_per_observation.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)
    aggregate = {k:float(np.mean([r[k] for r in rows])) for k in names if k not in ('scope','camera','frame')}
    summary = dict(record,status='completed_train_view_prior_fit',seconds=time.monotonic()-started,
                   aggregate_equal_observation_mean=aggregate,
                   metric_protocol='RGB[0,1]; full-frame PSNR; frozen11x11Gaussian SSIM; AlexNet LPIPSv0.1; no shave or uint8 requantization of renders',
                   metric_source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in (Path(cr.__file__),Path(cr.rt.__file__),Path(cr.ev.__file__),cr.ev.LEGACY)},
                   limitations=['Training-view fitting is separate from heldout novel-view quality.',
                     'Direct teacher images, HRref and SRrecon scores are empirical references, never strict bounds.',
                     'Agreement with a teacher does not prove that its details are correct.',
                     'Raw render and training-camera affine-corrected render are reported separately; correction uses learned parameters without refitting.',
                     'HR is opened only for diagnostic metrics and never passed to an optimizer.'])
    metrics = write(out/'prior_fit.json',summary)
    observation_index = write(out/'observation_index.json',dict(population_keys=record['population_keys'],observations=per_image))
    complete = dict(checked['receipt'],schema='highres_train_prior_fit_complete_v1',status='completed_train_view_prior_fit',
                    posthoc_cull=bool(a.cull),metrics=entry(metrics),csv=entry(out/'metrics_per_observation.csv'),
                    observation_index=entry(observation_index),seconds=summary['seconds'])
    write(out/'prior_fit_complete.json',complete)
    return complete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','baseline-sr','z5-samples','hr-root','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cull',action='store_true')
    p.add_argument('--check-only',action='store_true',help='CPU provenance checks only; no runtime import, checkpoint load or output creation')
    a = p.parse_args()
    require(local(a.out).is_relative_to(OUT) and not local(a.out).exists(), 'Fresh highres output directory required')
    checked = preflight(a)
    if a.check_only:
        print(json.dumps(checked['receipt'],ensure_ascii=False,indent=2));return
    try:
        print(json.dumps(run(a,checked),ensure_ascii=False))
    except BaseException:
        if local(a.out).exists():
            write(local(a.out)/f'failure_{time.time_ns()}.json',dict(status='failed_prior_fit_preserved',source=entry(__file__),error=traceback.format_exc(),training=False))
        raise


if __name__ == '__main__':
    main()
