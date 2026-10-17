"""Paired-form SR stage for physically sampled LR (Claude, 4dsr-7c, 2026-10-17). Copy of
experiments/dynamic_sr_compress_20261013/pairedsr.py (mode paired only; pcgrad / rangenull / --lr-dir removed) with three
options; cullretrain.py is imported, never modified.

  physpaired.py --lr-operator {bicaa,area} --teacher-band {all,high,zero} [--teacher-cut C] -- sr <cullretrain sr args>

Every step renders the same three same-time training views once (one shared deformation); every render gets its own LR
term and its own teacher-band term (all three views carry LR evidence, unlike the split Bsync recipe where only view a does):
    L = (1/3) sum_v l_LR(v) + (2 lambda_sr / 3) sum_v l_T(v) + R
    l_LR(v) = (1-d) mean|D(R_v) - Y_v| + d (1 - SSIM(D(R_v), Y_v))          D = area (4x4 box) or the registered bicubic-AA
    l_T(v)  = all : (1-d) mean|R_v - T_v| + d (1 - SSIM(R_v, T_v))            (pairedsr.py P-Y)
              high: mean|HP_C(R_v - T_v)|       teacher only above C cycles / HR px (default 0.25, first null of the 4x4 box)
              zero: mean|HP_C(R_v)|             same band and weight, target 0: a null-space smoothness prior, no teacher
R_v is the render after the camera's colour affine; d = --dssim; HP_C = radial FFT high-pass per channel.
Schedule, teachers, regulariser, colour affine, topology (srfix --mode fix) and optimizer order are cullretrain.run_sr's.
Writes <out>/physpaired.json with first-step checks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CULL = HERE.parent / 'dynamic_sr_cull_retrain_20261009'


def block_project(gs, gl):
    """PCGrad on 4x4 HR blocks: remove from gs its component along gl wherever the two conflict."""
    c, h, w = gs.shape
    if h % 4 or w % 4:
        raise ValueError(f'HR size {h}x{w} is not a multiple of 4')
    bs = gs.reshape(c, h // 4, 4, w // 4, 4).permute(1, 3, 0, 2, 4).reshape(h // 4, w // 4, -1)
    bl = gl.reshape(c, h // 4, 4, w // 4, 4).permute(1, 3, 0, 2, 4).reshape(h // 4, w // 4, -1)
    dot = (bs * bl).sum(-1, keepdim=True)
    norm = (bl * bl).sum(-1, keepdim=True)
    conflict = dot < 0
    bp = bs - (dot.clamp(max=0) / (norm + 1e-20)) * bl
    out = bp.reshape(h // 4, w // 4, c, 4, 4).permute(2, 0, 3, 1, 4).reshape(c, h, w)
    after = (bp * bl).sum(-1)
    return out, dict(conflict_share=float(conflict.float().mean()), min_dot_after=float(after.min()),
                     removed_energy_share=float(((bs - bp) ** 2).sum() / ((bs ** 2).sum() + 1e-30)))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog='physpaired.py')
    p.add_argument('--lr-operator', choices=('bicaa', 'area'), required=True)
    p.add_argument('--teacher-band', choices=('all', 'high', 'zero'), required=True)
    p.add_argument('--teacher-cut', type=float, default=0.25)
    p.add_argument('rest', nargs=argparse.REMAINDER)
    a0 = p.parse_args(argv)
    a0.mode, a0.lr_dir = 'paired', None
    rest = a0.rest[1:] if a0.rest[:1] == ['--'] else a0.rest
    if rest[:1] != ['sr']:
        raise SystemExit('expected: physpaired.py --lr-operator O --teacher-band B -- sr <cullretrain sr arguments>')
    for path in (CULL, HERE.parent / 'dynamic_sr_modules_20261010'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import cullretrain as cr
    import rt
    import srfix
    if srfix.cr is not cr:
        raise SystemExit('srfix imported a different cullretrain module')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    losses, prefix, native = cr.losses, cr.prefix, cr.native
    sys.path.insert(0, str(HERE))
    import physsr
    if a0.lr_operator == 'area':
        losses.degradation = physsr.area_degradation
    masks = {}

    def highpass(x):
        h_, w_ = x.shape[-2:]
        key = (h_, w_)
        if key not in masks:
            fy = torch.fft.fftfreq(h_, device=x.device)[:, None]
            fx = torch.fft.rfftfreq(w_, device=x.device)[None, :]
            masks[key] = (torch.sqrt(fy ** 2 + fx ** 2) >= a0.teacher_cut).float()
        return torch.fft.irfft2(torch.fft.rfft2(x) * masks[key], s=(h_, w_))

    counts = dict(images=0)
    receipt = None

    checks = {}
    stats = dict(steps=0, conflict_share=0.0, removed_energy_share=0.0)

    def up(x, size):
        return F.interpolate(x[None], size=tuple(size), mode='bicubic', align_corners=False)[0]

    def run_sr(a):  # cullretrain.run_sr with the loss block replaced; everything else line for line
        from scene.gaussian_model import GaussianModel
        from utils.loss_utils import ssim
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'complete.json').exists():
            raise FileExistsError('Completed run preserved: ' + str(out))
        base = cr.read(Path(a.baseline_sr) / 'config.json')
        definition = base['definition']
        if definition != dict(schedule='rows', lr_weights=[1.0, 0.0, 0.0], sr_weights=[0.0, 0.05, 0.05], X=False, E=False):
            raise ValueError('Only the registered Bsync definition is supported')
        parent_ckpt = Path(a.parent_checkpoint)
        payload = torch.load(parent_ckpt, map_location='cuda', weights_only=False)
        plan = payload['plan']
        if plan['configuration'] != base['configuration']:
            raise ValueError('Parent configuration differs from the baseline SR configuration')
        h, opt, pipe, sh = cr.configuration(plan)
        manifest_path, manifest = cr.plan_scene(plan)
        if cr.rel(manifest_path) != base['data']['manifest']['path']:
            raise ValueError('Manifest differs from the baseline SR plan')
        dataset = cr.N3DVPreparedDataset(manifest, 'train', 'lr', device='cpu', cache=False)
        extent = plan['data']['cameras_extent']
        torch.set_num_threads(a.cpu_threads)
        started = time.time()
        lam_total = 2 * a.lambda_sr
        record = dict(kind='sr', arm=a.arm, args=vars(a), parent_checkpoint=cr.entry(parent_ckpt), baseline_sr=cr.rel(a.baseline_sr),
                      scene=plan['scene'], seed=plan['seed'], lambda_SR=a.lambda_sr, dssim=a.dssim, host=cr.socket.gethostname(),
                      gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(),
                      sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(cr.__file__), Path(rt.__file__))}, started_unix=started,
                      paired=dict(mode='paired', lr_operator=a0.lr_operator, teacher_band=a0.teacher_band, teacher_cut=a0.teacher_cut,
                                  lambda_total=lam_total, lr_weight_per_view=1 / 3, sr_weight_per_view=lam_total / 3))
        g = GaussianModel(sh, h)
        g._deformation = g._deformation.cuda()
        g.restore(payload['model'], opt)
        g._deformation_accum = payload['deformation_accum']
        prefix.restore_rng(payload['rng'], torch, np)
        affine = None
        if a.affine:
            affine = cr.CameraAffine(manifest['splits']['train'], lr_init=a.affine_sr_lr, lr_final=a.affine_sr_lr, max_steps=cr.SR_STEPS)
            if payload.get('affine') is not None:
                affine.load_state_dict(payload['affine'])
                for group in affine.optimizer.param_groups:
                    group['lr'] = a.affine_sr_lr
                record['affine_inherited'] = True
        del payload
        g.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name in ('max_radii2D', 'xyz_gradient_accum', 'denom'):
                getattr(g, name).zero_()
        table = cr.read(cr.ROOT / base['dependencies']['schedule']['path'])
        rows = table[definition['schedule']]
        record_keys = [tuple(k) for k in table['record_keys']]
        teacher_index = cr.read(cr.ROOT / base['dependencies']['teacher']['path'])
        teacher_rows = {(e['camera'], int(e['frame'])): e for e in teacher_index['entries']}
        mroot = Path(manifest['_root'])
        teacher_paths = {key: native.safe_child(mroot, e['relative_path']) for key, e in teacher_rows.items()}
        stop = a.stop or cr.SR_STEPS

        def fetch(i):
            row = rows[i]
            items = [dataset[index] for index in row]
            keys = [record_keys[index] for index in row]
            teachers = [cr.read_teacher(teacher_paths[keys[j]]) for j in (0, 1, 2)]  # paired: all three views
            return row, items, keys, teachers

        prefetch = cr.Prefetch(fetch, list(range(stop)))
        culler = cr.Culler(manifest) if a.cull else None
        record.update(culler=culler.describe() if culler else None, prune_gate=a.prune_gate, affine=bool(a.affine),
                      cull_interval=a.cull_interval, schedule=cr.entry(cr.ROOT / base['dependencies']['schedule']['path']),
                      teacher_index=cr.entry(cr.ROOT / base['dependencies']['teacher']['path']))
        rt.write_json(out / 'run.json', record)
        background = torch.ones(3, dtype=torch.float32, device='cuda') if cr.white_background(plan) else torch.zeros(3, dtype=torch.float32, device='cuda')
        counts_topo = dict(densify=0, prune=0, pruned_points=0, grow=0, opacity_reset=0)
        cull_rows = []
        log = (out / 'trajectory.jsonl').open('a', buffering=1)
        torch.cuda.reset_peak_memory_stats()
        d = a.dssim
        try:
            if culler is not None:
                cull_rows.append(dict(cr.cull(g, culler, log, 'sr_start'), cursor=0))
            for cursor in range(1, stop + 1):
                tick = time.time()
                g.update_learning_rate(cr.LR_OFFSET + cursor)
                if (cr.LR_OFFSET + cursor) % 1000 == 0:
                    g.oneupSHdegree()
                g.optimizer.zero_grad(set_to_none=True)
                row, items, keys, teacher_cpu = prefetch.get(cursor - 1)
                if len({k[0] for k in keys}) != 3 or len({int(k[1]) for k in keys}) != 1:
                    raise ValueError('Registered same-time batch differs')
                cameras = [native.hr_camera(obs, i, manifest) for i, obs in zip(row, items)]
                lrs = [obs['image'].cuda() for obs in items]
                teachers = [t.cuda() for t in teacher_cpu]
                state = native.native_effective_state(g, cameras[0].time, torch)
                packages = [native.native_rgb(c, g, state, pipe, background) for c in cameras]
                rgbs = [q['render'] for q in packages]
                if affine is not None:
                    rgbs = [affine.apply(rgb, obs['camera_id']) for rgb, obs in zip(rgbs, items)]
                # ---- loss block (the only change against cullretrain.run_sr) ----
                lr_l1, lr_terms, sr_l1, sr_terms = [], [], [], []
                for rgb, lr, t in zip(rgbs, lrs, teachers):
                    down = losses.degradation(rgb, lr.shape[-2:])
                    l1 = (down - lr).abs().mean()
                    lr_l1.append(l1)
                    lr_terms.append((1 - d) * l1 + d * (1 - ssim(down[None], lr[None])) if d else l1)
                    sr_l1.append((rgb - t).abs().mean().detach())
                    if a0.teacher_band == 'all':
                        s1 = (rgb - t).abs().mean()
                        sr_terms.append((1 - d) * s1 + d * (1 - ssim(rgb[None], t[None])) if d else s1)
                    elif a0.teacher_band == 'high':
                        sr_terms.append(highpass(rgb - t).abs().mean())
                    else:
                        sr_terms.append(highpass(rgb).abs().mean())
                lr_term = sum(lr_terms) / 3
                sr_term = lam_total * sum(sr_terms) / 3
                regularizer = g.compute_regulation(h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight) if h.time_smoothness_weight != 0 else rgbs[0].sum() * 0.
                loss = lr_term + sr_term + regularizer
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite SR objective')
                step_info = {}
                loss.backward()
                if cursor == 1:
                    recomputed = float(sum(float(x) for x in lr_terms) / 3 + lam_total * sum(float(x) for x in sr_terms) / 3 + float(regularizer))
                    checks.update(first_loss=float(loss.detach()), recomputed=recomputed, lr_terms=[float(x) for x in lr_terms],
                                  sr_terms=[float(x) for x in sr_terms], lambda_total=lam_total,
                                  viewspace_grads_present=[q['viewspace_points'].grad is not None for q in packages])
                stats['steps'] += 1
                # ---- end of loss block ----
                statistics = prefix.combine_batch_statistics(packages, torch)
                before = int(g.get_xyz.shape[0])
                with torch.no_grad():
                    cr.topology(g, opt, 'fine', cursor, extent, out, statistics, a.prune_gate, counts_topo)
                    g.optimizer.step()
                    g.optimizer.zero_grad(set_to_none=True)
                if affine is not None:
                    affine.step(cursor)
                if culler is not None and cursor % a.cull_interval == 0:
                    cull_rows.append(dict(cr.cull(g, culler, log, f'sr_{cursor}'), cursor=cursor))
                log.write(json.dumps(dict(cursor=cursor, loss=float(loss.detach()), LR_L1=[float(v.detach()) for v in lr_l1],
                                          SR_L1=[float(v) for v in sr_l1], LR_term=float(lr_term.detach()),
                                          SR_term=float(sr_term.detach()), regularizer=float(regularizer.detach()),
                                          points_before=before, points_after=int(g.get_xyz.shape[0]), keys=[list(k) for k in keys],
                                          seconds=time.time() - tick, **step_info)) + '\n')
                del items, cameras, lrs, teachers, packages, rgbs, state, loss
            prefetch.close()
            if culler is not None and stop % a.cull_interval != 0:
                cull_rows.append(dict(cr.cull(g, culler, log, f'sr_end_{stop}'), cursor=stop))
            path = cr.save(out / f'checkpoint_{stop:05d}.pt', cr.SCHEMA_SR, g, plan,
                           dict(affine=affine.state_dict() if affine else None, metadata=dict(cursor=stop, arm=a.arm)))
            torch.cuda.synchronize()
            result = dict(status='completed', cursor=stop, seconds=time.time() - started, counts=counts_topo, culls=cull_rows,
                          points=int(g.get_xyz.shape[0]), peak_GB=torch.cuda.max_memory_allocated() / 1e9,
                          affine=affine.summary() if affine else None, checkpoint=cr.entry(path))
            rt.write_json(out / 'complete.json', dict(record, **result))
            return result
        except BaseException:
            prefetch.close()
            rt.write_json(out / f'failure_{time.time_ns()}.json', dict(record, error=traceback.format_exc()))
            raise

    out = Path(rest[rest.index('--out') + 1])
    out = out if out.is_absolute() else cr.ROOT / out
    if (out / 'complete.json').exists():
        raise FileExistsError('Completed run preserved: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    cr.run_sr = run_sr
    sys.argv = ['srfix.py', '--mode', 'fix', 'cullretrain', *rest]
    srfix.main()
    n = max(stats['steps'], 1)
    summary = dict(kind='physpaired_sr', lr_operator=a0.lr_operator, teacher_band=a0.teacher_band, teacher_cut=a0.teacher_cut,
                   degradation=f'{losses.degradation.__module__}.{losses.degradation.__name__}',
                   source={'physpaired.py': rt.sha(Path(__file__)), 'physsr.py': rt.sha(Path(physsr.__file__))},
                   entry='srfix --mode fix cullretrain (run_sr replaced)', argv=rest, checks=checks, steps=stats['steps'])
    rt.write_json(out / 'physpaired.json', summary)
    print(json.dumps(dict(physpaired=[a0.lr_operator, a0.teacher_band], checks=checks), default=str)[:2000], flush=True)


if __name__ == '__main__':
    main()
