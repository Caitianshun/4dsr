"""Bounded continuation experiment; child exit triggers evaluation immediately.

The manager blocks in waitpid (subprocess.run) and Future.result, with no status
polling. Local numerical verification and reporting do not require an LLM call.
Run with system Python so an optional external event listener can use pidfds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
NEW = Path(__file__).resolve().parent
OLD = ROOT / 'experiments/dynamic_sr_20260918'
ORIGINAL = ROOT / 'output/dynamic_sr_20260919'
OUT = ROOT / 'output/dynamic_sr_20260920/continuation'
PY = '/home/cai_tianshun/Project/4dgs/.venv/bin/python'
CASES = ('joint', 'appearance_only')
SCENES = ('cook_spinach', 'meetroom_discussion')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def checked_run(command, log, env):
    # No timeout or repeated status reads: return itself is the completion event.
    started = time.monotonic()
    with Path(log).open('x') as handle:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Child exit {result.returncode}: {log}')
    return {'seconds': time.monotonic() - started, 'finished': now(),
            'log': str(log), 'command': list(map(str, command))}


def verify_run(scene, mode):
    name = scene + '_' + mode
    run = OUT / name
    config, done = read(run / 'config.json'), read(run / 'complete.json')
    ev = read(OUT / (name + '_evaluation/metrics.json'))
    sampling = read(OUT / (name + '_sampling/sampling.json'))
    cksha = sha(run / 'checkpoint_final.pt')
    assert ev['checkpoint_sha256'] == sampling['checkpoint_sha256'] == cksha
    assert config['manifest_sha256'] == ev['manifest_sha256'] == sampling['manifest_sha256']
    assert ev['manifest_sha256'] == sha(ev['manifest'])
    assert done['intervention_step'] == 18000
    assert done['additional_steps'] == 12000 and done['full_endpoint'] is True
    if mode == 'appearance_only':
        assert done['frozen_verification']['passed'] is True
    assert len(ev['rows']) == 60 and sum('temporal' in x for x in ev['rows']) == 59
    assert ev['gaussian_count'] == done['points'] == config['initial_points']
    assert len(read(run / 'fit_18000.json')['rows']) == 16
    assert sum(read(run / 'exposure.json').values()) == 12000
    assert read(run / 'full_exposure.json') == read(ORIGINAL / (scene + '_sr_w01/exposure.json'))
    for region in ('full', 'dynamic', 'static'):
        for metric in ('psnr', 'ssim', 'lpips_alex' if region == 'full' else 'lpips_alex_spatial_mask'):
            values = [r['spatial'][region][metric] for r in ev['rows']]
            assert all(math.isfinite(v) for v in values)
            assert abs(sum(values) / len(values) - ev['aggregate'][region][metric + '_mean']) < 1e-7
    return {'status': 'passed', 'finished': now(), 'checkpoint_sha256': cksha,
            'manifest_sha256': config['manifest_sha256'], 'points': done['points'],
            'frame_count': 60, 'temporal_pairs': 59, 'train_diagnostics': 16}


def scene_worker(scene, gpu, dependency_pid=None):
    state = OUT / (scene + '_state.json')
    s = {'status': 'running', 'pid': os.getpid(), 'scene': scene, 'gpu': gpu,
         'started': now(), 'completed': [], 'current': None,
         'completion_mechanism': 'blocking subprocess return, not polling'}
    write(state, s)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
    source = ORIGINAL / (scene + '_sr_w01')
    manifest = read(source / 'config.json')['manifest']
    try:
        if dependency_pid is not None:
            s.update(status='waiting_gpu', dependency_pid=dependency_pid)
            write(state, s)
            try:
                fd = os.pidfd_open(dependency_pid)
            except ProcessLookupError:
                fd = None
            if fd is not None:
                try:
                    select.select([fd], [], [])  # GPU predecessor exit event; no timeout/polling.
                finally:
                    os.close(fd)
            dependency = read(ROOT / 'output/dynamic_sr_20260920/output_swap/complete.json')
            assert dependency['status'] == 'complete', 'Output swap dependency did not finish successfully'
            s.update(status='running', gpu_acquired=now())
            write(state, s)
        for mode in CASES:
            name = scene + '_' + mode
            run = OUT / name
            stages = [
                ('training', [PY, str(NEW / 'resume_control.py'), '--source-run', str(source),
                              '--mode', mode, '--out', str(run), '--steps', '12000']),
                ('evaluation', [PY, str(OLD / 'evaluate.py'), '--manifest', manifest,
                                '--checkpoint', str(run / 'checkpoint_final.pt'),
                                '--out', str(OUT / (name + '_evaluation')), '--no-video']),
                ('sampling', [PY, str(OLD / 'evaluate.py'), '--manifest', manifest,
                              '--checkpoint', str(run / 'checkpoint_final.pt'),
                              '--out', str(OUT / (name + '_sampling')), '--sampling-only', '--no-video'])]
            for stage, command in stages:
                s.update(current={'mode': mode, 'stage': stage, 'started': now()})
                write(state, s)
                event = checked_run(command, OUT / (name + '_' + stage + '.log'), env)
                s['completed'].append({'mode': mode, 'stage': stage, **event})
                write(state, s)
            verified = verify_run(scene, mode)
            write(OUT / (name + '_verification.json'), verified)
        s.update(status='complete', current=None, finished=now())
        write(state, s)
        return s
    except BaseException as e:
        s.update(status='failed', error=repr(e), finished=now())
        write(state, s)
        raise


def region_values(ev, region):
    x = ev['aggregate'][region]
    perceptual = 'lpips_alex_mean' if region == 'full' else 'lpips_alex_spatial_mask_mean'
    return dict(psnr=x['psnr_mean'], ssim=x['ssim_mean'], lpips=x[perceptual], mse=x['mse_mean'])


def summarize():
    result = {'created': now(), 'scope': 'development-only matched continuation; no checkpoint selection',
              'scenes': {}, 'information_boundary': 'training LR and frozen SR priors only; HR is evaluation-only'}
    lines = ['# 固定空间支撑续训：自动数值汇总', '',
             '固定追加 12k，比较原 SR 0.1 的 6k→18k 阶段。appearance_only 冻结位置、形状、透明度和共享形变特征，继续更新 SH 外观；不是仅纹理更新。', '',
             '这些 cam00 已用于研究开发，不是新的独立测试。区域 LPIPS 为原全图空间 LPIPS 图的固定掩码均值。', '']
    for scene in SCENES:
        src = ORIGINAL / (scene + '_sr_w01')
        old = read(ORIGINAL / (scene + '_sr_w01_evaluation/metrics.json'))
        cases = {}
        lines += [f'## {scene}', '', '| 分支 / 区域 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |', '| --- | ---: | ---: | ---: |']
        for mode in CASES:
            name = scene + '_' + mode
            run = OUT / name
            ev = read(OUT / (name + '_evaluation/metrics.json'))
            fit, done = read(run / 'fit_18000.json'), read(run / 'complete.json')
            cases[mode] = {'regions': {r: region_values(ev, r) for r in ('full', 'dynamic', 'static')},
                           'fit': fit['aggregate'], 'train_lr_psnr': fit['lr_psnr'],
                           'temporal': ev['temporal_aggregate'], 'complete': done,
                           'metrics': str(OUT / (name + '_evaluation/metrics.json'))}
            for region, values in cases[mode]['regions'].items():
                lines.append(f"| {mode} / {region} | {values['psnr']:.5f} | {values['ssim']:.6f} | {values['lpips']:.6f} |")
        deltas = {r: {k: cases['appearance_only']['regions'][r][k] - cases['joint']['regions'][r][k]
                      for k in ('psnr', 'ssim', 'lpips', 'mse')} for r in ('full', 'dynamic', 'static')}
        parity = {r: {k: cases['joint']['regions'][r][k] - region_values(old, r)[k]
                      for k in ('psnr', 'ssim', 'lpips', 'mse')} for r in ('full', 'dynamic', 'static')}
        # Numerical screening only, not a statistical equivalence test.
        parity_ok = all(abs(parity[r]['psnr']) <= .05 and abs(parity[r]['ssim']) <= .001
                        and abs(parity[r]['lpips']) <= .002 for r in parity)
        result['scenes'][scene] = {'cases': cases, 'appearance_minus_joint': deltas,
                                  'joint_minus_historical': parity,
                                  'historical_agreement_screen': parity_ok,
                                  'initial_train_fit': read(src / 'fit_6000.json')['aggregate']}
        lines += ['', f"恢复轨迹数值筛查：{'通过' if parity_ok else '需排查'}（全图及各 ROI 的旧终点差异：PSNR≤0.05dB、SSIM≤0.001、LPIPS≤0.002；这不是统计等价性检验）。", '',
                  '| appearance_only − joint | ΔPSNR | ΔSSIM | ΔLPIPS |', '| --- | ---: | ---: | ---: |']
        for r, d in deltas.items():
            lines.append(f"| {r} | {d['psnr']:+.5f} | {d['ssim']:+.6f} | {d['lpips']:+.6f} |")
        lines += ['', '| 训练图拟合 | 对 HR PSNR | 对 HR SSIM | 对 HR LPIPS | 对 SR 先验 PSNR |', '| --- | ---: | ---: | ---: | ---: |']
        for label, fit in [('原 6k', read(src / 'fit_6000.json')['aggregate'])] + [(m, cases[m]['fit']) for m in CASES]:
            h = fit['render_hr']
            lines.append(f"| {label} | {h['psnr']:.5f} | {h['ssim']:.6f} | {h['lpips_alex']:.6f} | {fit['render_prior']['psnr']:.5f} |")
        lines += ['']
    lines += ['## 解释边界', '',
              '这份程序汇总只报告已完成的数值与完整性；研究判断应联合输出交换、固定位置图像、训练拟合和时序误差。固定支撑若有效，可能体现正则化或参数更新限制，不单独证明真实几何更准确。若恢复筛查失败，优先排查实现，暂不解释机制。', '',
              '所有分支、成本、冻结验收和原始指标路径保存在同目录 summary.json；训练退出后直接触发评测，没有训练状态轮询。']
    write(OUT / 'summary.json', result)
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    return result


def notify(status):
    executable = shutil.which('notify-send')
    if not executable:
        return {'status': 'unavailable'}
    title = '4DSR 续训与验证已完成' if status == 'complete' else '4DSR 续训检查发现异常'
    body = '本地评测、核验和汇总已自动生成，请查看 continuation/summary.md。' if status == 'complete' else '失败状态和日志已保存，未自动重试。'
    try:
        r = subprocess.run([executable, '--app-name=4DSR', title, body], capture_output=True, text=True, timeout=10)
        return {'status': 'sent' if r.returncode == 0 else 'failed', 'returncode': r.returncode, 'stderr': r.stderr[-500:]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'status': 'failed', 'error': repr(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--gpu1-dependency-pid', type=int,
                        help='Wait on the output-swap process exit before using GPU1')
    args = parser.parse_args()
    if args.summarize_only:
        summarize()
        return
    OUT.mkdir(parents=True, exist_ok=False)
    state = OUT / 'state.json'
    s = {'status': 'running', 'pid': os.getpid(), 'started': now(), 'completed_scenes': [],
         'scenes': dict(zip(SCENES, [0, 1])), 'script_sha256': sha(__file__),
         'protocol_sha256': sha(ROOT / 'docs/dynamic_sr_next_validation_protocol_2026-09-20.md'),
         'completion_mechanism': 'subprocess return + Future completion; no polling'}
    write(state, s)
    errors = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = {executor.submit(scene_worker, scene, gpu,
                                 args.gpu1_dependency_pid if gpu == 1 else None): scene
                 for scene, gpu in s['scenes'].items()}
        for future in as_completed(tasks):
            scene = tasks[future]
            try:
                future.result()
                s['completed_scenes'].append(scene)
            except Exception as e:
                errors[scene] = repr(e)
            write(state, s)
    try:
        if errors:
            raise RuntimeError(str(errors))
        summarize()
        s.update(status='complete', finished=now())
    except Exception as e:
        s.update(status='failed', finished=now(), error=repr(e))
    write(state, s)
    s['desktop_notification'] = notify(s['status'])
    write(state, s)
    if s['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
