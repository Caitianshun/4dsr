"""Call deployed author SR4D training unchanged, with only generic data adapter.

Original AST, losses, average pooling, save/densification/Adam order, and final
iteration without Adam are retained. No full optimizer-resume claim is made.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import time
import traceback

import numpy as np
import torch

from sr4d_common import (ManifestData, GenericScene, camera_geometry_test,
                         configure_upstream, sha256, write_json)


def module_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hashes(root):
    files = ['train.py', 'train_2.py', 'arguments/__init__.py', 'gaussian_renderer/__init__.py',
             'scene/gaussian_model.py', 'scene/deform_model.py', 'utils/time_utils.py',
             'utils/tasa_utils.py', 'utils/loss_utils.py', 'utils/general_utils.py']
    out = {name: sha256(root / name) for name in files}
    out['adapter/train_sr4d.py'] = sha256(__file__)
    out['adapter/sr4d_common.py'] = sha256(Path(__file__).with_name('sr4d_common.py'))
    return out


def import_coarse(source, target, iteration, manifest_hash, sources):
    source = Path(source).resolve()
    config = json.loads((source / 'config.json').read_text())
    if config['manifest_sha256'] != manifest_hash:
        raise ValueError('Cannot reuse a coarse model trained on a different manifest')
    if config['source_hashes'] != sources:
        raise ValueError('Cannot reuse a coarse model with different training source')
    receipt = json.loads((source / 'coarse_complete.json').read_text())
    if not receipt['complete'] or receipt['iterations'] != iteration:
        raise ValueError('Require the declared completed coarse endpoint')
    for folder in ('point_cloud', 'deform_lr'):
        src = source / folder / f'iteration_{iteration}'
        dst = target / folder / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    write_json(target / 'coarse_import.json', {'source': str(source), 'iteration': iteration,
               'receipt_sha256': sha256(source / 'coarse_complete.json'),
               'copied_files': {str(p.relative_to(target)): sha256(p)
                                for folder in ('point_cloud', 'deform_lr')
                                for p in (target / folder).rglob('*') if p.is_file()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--scene', help='Optional expected manifest scene, checked if supplied')
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['coarse', 'fine', 'both'], default='both')
    parser.add_argument('--coarse-checkpoint', help='Completed generic coarse RUN DIRECTORY for a fine-only run')
    parser.add_argument('--coarse-steps', type=int, default=20000)
    parser.add_argument('--fine-steps', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--edgemap-loss', type=float, default=.01)
    parser.add_argument('--tex-rate', type=float, default=.001)
    parser.add_argument('--opacity-reg', type=float, default=.1)
    parser.add_argument('--save-iterations', nargs='*', type=int, default=[6000, 12000, 18000, 20000, 30000, 40000])
    parser.add_argument('--smoke', action='store_true', help='Label engineering run; budget must be set explicitly')
    parser.add_argument('--inspect-only', action='store_true', help='CPU-only frozen data/geometry checks; no upstream import')
    args = parser.parse_args()
    if min(args.coarse_steps, args.fine_steps) < 1:
        parser.error('Stage iteration counts must be positive')
    if args.stage == 'fine' and not args.coarse_checkpoint:
        parser.error('Fine-only requires --coarse-checkpoint RUN_DIRECTORY')
    if args.coarse_checkpoint and args.stage != 'fine':
        parser.error('--coarse-checkpoint is only valid with --stage fine')
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory; original training has no exact mid-stage resume')
    out.mkdir(parents=True)
    started = time.monotonic()
    try:
        data = ManifestData(args.manifest)
        if args.scene and args.scene != data.scene:
            raise ValueError('Unexpected manifest scene')
        evidence = data.verify_training_inputs()
        evidence['projection_max_error_pixels_cpu'] = camera_geometry_test(data)
        write_json(out / 'input_validation.json', evidence)
        if args.inspect_only:
            write_json(out / 'status.json', {'state': 'inspection_complete', 'gpu_used': False, **evidence})
            print(json.dumps(evidence), flush=True)
            return
        root = configure_upstream(args.upstream)
        from arguments import ModelParams, PipelineParams, OptimizationParams
        options = argparse.ArgumentParser()
        mp, pp, op = ModelParams(options), PipelineParams(options), OptimizationParams(options)
        values = options.parse_args([])
        dataset, pipe, opt = mp.extract(values), pp.extract(values), op.extract(values)
        dataset.source_path, dataset.model_path = str(data.root), str(out)
        dataset.load2gpu_on_the_fly = True
        dataset.white_background = dataset.is_blender = dataset.is_6dof = False
        dataset.eval = True
        opt.iterations = opt.iterations_lr = args.coarse_steps
        opt.iterations_hr = args.fine_steps
        opt.edgemap_loss, opt.tex_rate, opt.opacity_reg = args.edgemap_loss, args.tex_rate, args.opacity_reg
        hashes = source_hashes(root)
        config = {'arguments': vars(args), 'scene': data.scene, 'manifest_sha256': evidence['manifest_sha256'],
                  'source_hashes': hashes, 'options': vars(opt), 'initialization_sha256': evidence['initialization_sha256'],
                  'input_degradation': data.document['degradation'], 'train_cameras': data.document['splits']['train'],
                  'training_observations': len(data.train_rows), 'frames': data.document['frame_indices'],
                  'hr_wh': data.hr_wh, 'lr_wh': data.lr_wh, 'time_convention': data.document['time_convention'],
                  'method': 'SR4D deployed author training, generic multiview data interface',
                  'internal_degradation': 'author fixed avg_pool2d x4, RGB and deployed audited texture path',
                  'ast': 'unchanged author 1 / remaining viewpoint_stack size',
                  'save_semantics': 'unchanged: save before densification and Adam; N loops, N-1 optimizer calls',
                  'resume_semantics': 'completed coarse can start fine; no exact mid-stage optimizer resume',
                  'training_hr_decodes': 0, 'training_test_or_dev_image_reads': 0, 'testing_iterations': [],
                  'environment': {'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                                  'host': socket.gethostname(), 'gpu': torch.cuda.get_device_name(),
                                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
                  'seed': args.seed, 'smoke': args.smoke}
        write_json(out / 'config.json', config)
        shutil.copy2(data.path, out / 'manifest_source.json')
        snapshot = out / 'source_snapshot'
        snapshot.mkdir()
        for source in [Path(__file__), Path(__file__).with_name('sr4d_common.py'), root / 'train.py', root / 'train_2.py']:
            shutil.copy2(source, snapshot / source.name)
        if args.coarse_checkpoint:
            import_coarse(args.coarse_checkpoint, out, args.coarse_steps, evidence['manifest_sha256'], hashes)
        stages = ['coarse', 'fine'] if args.stage == 'both' else [args.stage]
        for stage in stages:
            # A stage gets a reproducible independent process-like random start;
            # this does not change the author's within-stage sampling algorithm.
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats()
            module_path = root / ('train.py' if stage == 'coarse' else 'train_2.py')
            module = module_from_file(module_path, f'author_sr4d_{stage}')
            module.Scene = functools.partial(GenericScene, data=data)
            total = args.coarse_steps if stage == 'coarse' else args.fine_steps
            milestones = sorted({x for x in args.save_iterations if 0 < x <= total} | {total})
            write_json(out / 'status.json', {'state': 'running', 'stage': stage, 'iterations': total,
                                           'saved_milestones': milestones, 'model_polling': False})
            stage_started = time.monotonic()
            if stage == 'coarse':
                module.training(dataset, opt, pipe, [], milestones)
            else:
                module.training(dataset, opt, pipe, [], milestones, coarse_iteration=args.coarse_steps)
            torch.cuda.synchronize()
            folder = 'point_cloud' if stage == 'coarse' else 'point_cloud_hr'
            deform_folder = 'deform_lr' if stage == 'coarse' else 'deform_hr'
            outputs = [out / folder / f'iteration_{total}' / 'point_cloud.ply',
                       out / deform_folder / f'iteration_{total}' / f'{deform_folder}.pth']
            if not all(p.is_file() and p.stat().st_size > 0 for p in outputs):
                raise RuntimeError('Original training returned without complete endpoint')
            receipt = {'complete': True, 'stage': stage, 'iterations': total, 'optimizer_calls': total - 1,
                       'seconds': time.monotonic() - stage_started, 'peak_GiB': torch.cuda.max_memory_allocated() / 2 ** 30,
                       'files': {str(p.relative_to(out)): sha256(p) for p in outputs}}
            if stage == 'fine':
                log = json.loads((out / 'timelog/trainlog_stage2.json').read_text())
                receipt['coarse_parameters_unchanged'] = log['coarse_parameters_unchanged']
                if not receipt['coarse_parameters_unchanged']:
                    raise RuntimeError('Coarse changed during fine training')
            write_json(out / f'{stage}_complete.json', receipt)
            del module
            torch.cuda.empty_cache()
        if source_hashes(root) != hashes:
            raise RuntimeError('Training source changed during execution')
        write_json(out / 'status.json', {'state': 'completed', 'stages': stages,
                                       'elapsed_s': time.monotonic() - started})
    except BaseException:
        write_json(out / 'status.json', {'state': 'failed', 'traceback': traceback.format_exc(),
                                       'elapsed_s': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
