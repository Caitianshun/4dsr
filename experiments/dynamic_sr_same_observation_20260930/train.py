"""Single HR render and combined LR/SR backward from the first coarse update."""
import argparse
from collections import OrderedDict, Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_20260918'))
import numpy as np
import torch
from common import (UPSTREAM, downsample, image_tensor, named_parameters, new_model,
                    render_image, resized_camera, save_checkpoint, seed_all, sha256, write_json)
from n3dv_data import N3DVPreparedDataset, load_initial_points, load_manifest, observation_to_4dgs_camera


def train(protocol_path, out):
    p = json.loads(protocol_path.read_text())
    tr = p['training']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == tr['physical_gpu']
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'state.json', dict(status='validating_inputs', pid=os.getpid()))
    manifest_path = ROOT / p['manifest_path']
    assert sha256(manifest_path) == p['manifest_sha256']
    m = load_manifest(manifest_path)
    data_root = Path(m['_root'])
    for path, identity in p['sources'].items():
        assert sha256(ROOT / path) == identity, path
    initial_path = data_root / p['initialization']['npz_path']
    assert sha256(initial_path) == p['initialization']['npz_sha256']
    allowed = {str(initial_path.resolve()): p['initialization']['npz_sha256']}
    for r in p['observations']:
        assert r['camera'] in m['splits']['train'] and r['camera'] not in ['cam00', 'cam01']
        for kind in ['lr', 'sr']:
            path = data_root / r[kind+'_path']
            assert sha256(path) == r[kind+'_sha256'], path
            allowed[str(path.resolve())] = r[kind+'_sha256']
    reads = Counter()
    def audit_open(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.npz', '.npy']:
            name = str(path.resolve())
            assert name in allowed, ('Unregistered training input', name)
            reads[name] += 1
    sys.addaudithook(audit_open)
    torch.set_num_threads(4)
    seed_all(tr['seed'])
    data = N3DVPreparedDataset(m, 'train', 'lr', cache=True)
    records = [data[i] for i in range(len(data))]
    assert [[r['camera_id'], r['frame_index']] for r in records] == [[r['camera'],r['frame']] for r in p['observations']]
    w, hgt = m['resolutions']['hr']
    cameras = [resized_camera(observation_to_4dgs_camera(r, i), hgt, w) for i,r in enumerate(records)]
    initial = load_initial_points(m)
    centers = np.stack([np.asarray(m['cameras'][c]['c2w'])[:3, 3] for c in m['splits']['train']])
    g, hidden, optim, extent = new_model(initial['points'], initial['colors'], centers)
    assert len(g._xyz) == p['initialization']['point_count']
    source_dir = out / 'sources'
    for rel in p['sources']:
        dest = source_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    upstream_sources = {}
    for rel in ['arguments/__init__.py', 'gaussian_renderer/__init__.py', 'scene/gaussian_model.py', 'scene/deformation.py', 'scene/hexplane.py']:
        path = UPSTREAM / rel
        if path.exists():
            upstream_sources[rel] = sha256(path)
            dest = source_dir / 'upstream' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    config = dict(**p, parent_checkpoint=None,
                  parent_checkpoint_loaded=False, initialization_points=len(g._xyz),
                  training_host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  gpu_visible=os.environ['CUDA_VISIBLE_DEVICES'], torch=str(torch.__version__),
                  cuda=torch.version.cuda, python=sys.version, upstream=str(UPSTREAM),
                  upstream_sources=upstream_sources, hidden=vars(hidden), optim=vars(optim),
                  teacher_inputs=[dict(camera=r['camera'],frame=r['frame'],relative_path=r['sr_path'],sha256=r['sr_sha256']) for r in p['observations']])
    write_json(out / 'config.json', config)
    write_json(out / 'input_validation.json', dict(status='passed', files=len(allowed),
                  sha256=allowed, no_HR_or_heldout_input=True))
    # Bounded GPU cache, matching the recent prior branches' 64-image capacity.
    teachers = OrderedDict()
    def teacher(index):
        if index not in teachers:
            teachers[index] = image_tensor(data_root / p['observations'][index]['sr_path'])
        teachers.move_to_end(index)
        while len(teachers) > 64:
            teachers.popitem(last=False)
        return teachers[index]
    draw = random.Random(tr['seed'])
    draw_hash = hashlib.sha256()
    started = time.monotonic()
    total_step = 0
    torch.cuda.reset_peak_memory_stats()
    write_json(out / 'state.json', dict(status='training', pid=os.getpid(), total_updates=tr['total_updates']))
    with (out / 'training.jsonl').open('w', buffering=1) as log:
        for stage, steps in [('coarse', tr['coarse_steps']), ('fine', tr['fine_steps'])]:
            g.training_setup(optim)
            for step in range(1, steps + 1):
                total_step += 1
                g.update_learning_rate(step)
                if step % 1000 == 0:
                    g.oneupSHdegree()
                idx = draw.randrange(len(records))
                draw_hash.update(f'{idx}\n'.encode())
                target_lr = records[idx]['image'].cuda()
                target_sr = teacher(idx)
                g.optimizer.zero_grad(set_to_none=True)
                package = render_image(g, cameras[idx], stage)
                prediction = package['render']
                loss_lr = (downsample(prediction, target_lr.shape[-2:]) - target_lr).abs().mean()
                loss_sr = (prediction - target_sr).abs().mean()
                reg = (g.compute_regulation(hidden.time_smoothness_weight, hidden.l1_time_planes,
                                             hidden.plane_tv_weight) if stage == 'fine' else prediction.new_zeros(()))
                loss = loss_lr + tr['sr_weight'] * loss_sr + reg
                assert bool(torch.isfinite(loss)), (stage, step)
                if step == 1:
                    # Output-gradient audit does not traverse/update Gaussian parameters.
                    gl = torch.autograd.grad(loss_lr, prediction, retain_graph=True)[0]
                    gs = torch.autograd.grad(loss_sr, prediction, retain_graph=True)[0]
                    gt = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
                    maximum = float((gt - gl - tr['sr_weight'] * gs).abs().max())
                    assert maximum < 1e-7 and float(gl.norm()) > 0 and float(gs.norm()) > 0
                    write_json(out / f'first_update_{stage}.json', dict(stage=stage, total_step=total_step,
                        camera=records[idx]['camera_id'], frame=records[idx]['frame_index'],
                        same_render_tensor=True, same_LR_and_SR_observation=True, LR_weight=1.0,
                        SR_weight=tr['sr_weight'], LR_output_grad_norm=float(gl.norm()),
                        SR_output_grad_norm=float(gs.norm()), combined_output_gradient_error=maximum,
                        render_calls=1, parameter_backward_calls=1, optimizer_calls=1,
                        parent_checkpoint_loaded=False, SR_active_from_first_update=True))
                    del gl, gs, gt
                loss.backward()
                with torch.no_grad():
                    if 300 < step < min(4000, steps - 500):
                        visible = package['visibility_filter']
                        g.max_radii2D[visible] = torch.maximum(g.max_radii2D[visible], package['radii'][visible])
                        g.add_densification_stats(package['viewspace_points'].grad, visible)
                        if step % 100 == 0 and len(g._xyz) < tr['max_points']:
                            g.densify(.0002, .005, extent, None, 5, 5, str(out), step, stage)
                    g.optimizer.step()
                if step == 1 or total_step % 100 == 0 or total_step in tr['checkpoints']:
                    row = dict(stage=stage, stage_step=step, total_step=total_step,
                               lr_l1=float(loss_lr), sr_l1=float(loss_sr), sr_weighted_l1=float(tr['sr_weight']*loss_sr),
                               regularization=float(reg), total_loss=float(loss), points=len(g._xyz),
                               camera=records[idx]['camera_id'],frame=records[idx]['frame_index'],
                               elapsed_s=time.monotonic()-started, peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                               render_calls=total_step, parameter_backward_calls=total_step,
                               optimizer_calls=total_step, draw_sha256=draw_hash.hexdigest())
                    log.write(json.dumps(row)+'\n')
                    print(json.dumps(row), flush=True)
                if total_step in tr['checkpoints']:
                    metadata = dict(scene=m['scene'], method=p['method'], stage=stage, step=step,
                                    total_updates=total_step, coarse_updates=min(total_step,tr['coarse_steps']),
                                    fine_updates=max(0,total_step-tr['coarse_steps']),
                                    manifest_sha=p['manifest_sha256'], manifest=manifest_path.as_posix(),
                                    seed=tr['seed'], extent=extent, points=len(g._xyz),
                                    observation='same_observation_LR_SR_from_start', parent_checkpoint=None,
                                    privileged_train_hr=False, protocol_sha256=sha256(protocol_path),
                                    draw_sha256=draw_hash.hexdigest(), elapsed_s=time.monotonic()-started)
                    save_checkpoint(out / f'checkpoint_{total_step}.pt',g,hidden,optim,metadata)
                    write_json(out / f'checkpoint_{total_step}.json', dict(metadata=metadata,
                        sha256=sha256(out/f'checkpoint_{total_step}.pt')))
    assert total_step == tr['total_updates']
    assert all(sha256(ROOT/rel) == value for rel,value in p['sources'].items())
    assert all(bool(torch.isfinite(v).all()) for v in named_parameters(g).values())
    write_json(out / 'image_reads.json', dict(actual_reads=dict(reads), all_allowed=all(x in allowed for x in reads),
                                            no_HR_or_heldout_input=True))
    # Train-only immediate exit validation using the final model and legal LR/SR.
    torch.cuda.synchronize()
    train_seconds = time.monotonic()-started
    with torch.no_grad():
        image = render_image(g,cameras[0])['render']
        assert bool(torch.isfinite(image).all())
        low = downsample(image, records[0]['image'].shape[-2:])
        check = dict(camera=records[0]['camera_id'],frame=records[0]['frame_index'],
                     lr_l1=float((low-records[0]['image'].cuda()).abs().mean()),
                     sr_l1=float((image-teacher(0)).abs().mean()),parameter_updates=0,rgb_forwards=1)
    write_json(out / 'exit_validation.json', check)
    result = dict(status='completed_training', updates=total_step, rgb_forwards=total_step,
                  backward_calls=total_step, optimizer_calls=total_step, coarse_updates=tr['coarse_steps'],
                  fine_updates=tr['fine_steps'], points=len(g._xyz), train_seconds=train_seconds,
                  peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                  final_checkpoint=f'checkpoint_{total_step}.pt',final_sha256=sha256(out/f'checkpoint_{total_step}.pt'),
                  immediate_exit_validation=check, parent_checkpoint_loaded=False,
                  combined_supervision_from_first_step=True, training_host=socket.gethostname(),
                  gpu=torch.cuda.get_device_name(), physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    write_json(out / 'complete.json', result)
    write_json(out / 'state.json', result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        train(args.protocol, args.out)
    except BaseException:
        args.out.mkdir(parents=True,exist_ok=True)
        write_json(args.out / 'failed.json', dict(status='failed', traceback=traceback.format_exc()))
        raise
