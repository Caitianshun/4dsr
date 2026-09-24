#!/usr/bin/env python3
"""CPU-only compact delivery package and local checkpoint index.

Weights are streamed for SHA256 only, never deserialized or included in ZIP.
Original reports, records, models, and training code are never modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
BRANCHES = ('joint', 'ordinary_split', 'bound_split')
ENDPOINTS = (('dev', 1200), ('dev', 6000), ('test', 6000))
REPORT = PROJECT / 'docs/dynamic_sr_motion_bound_refinement_2026-09-23.md'


def read(path, optional=False):
    path = Path(path)
    if optional and not path.is_file():
        return {}
    return json.loads(path.read_text())


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class HashCache:
    def __init__(self):
        self.cache = {}

    def sha(self, path):
        real = Path(path).resolve(strict=True)
        before = real.stat()
        key = (str(real), before.st_size, before.st_mtime_ns)
        if key not in self.cache:
            value = hashlib.sha256()
            with real.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    value.update(block)
            after = real.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    f'File changed while hashing: {path}')
            self.cache[key] = value.hexdigest()
        return self.cache[key]


def completed_batch(argument):
    outer = argument.absolute()
    receipt = read(outer / 'complete.json')
    require(receipt.get('status') == 'completed_and_evaluated', f'Incomplete batch: {outer}')
    batch = outer if (outer / 'joint').is_dir() else outer / 'batch_v1'
    require(read(batch / 'complete.json').get('status') == 'completed_and_evaluated',
            f'Incomplete inner batch: {batch}')
    require(read(batch / 'summary/complete.json').get('status') == 'completed_summary',
            f'Summary incomplete: {batch}')
    for branch in BRANCHES:
        done = read(batch / branch / 'complete.json')
        require(done.get('status') == 'completed' and done.get('parameter_updates') == 6000
                and not done.get('smoke'), f'Training incomplete: {batch / branch}')
        for split, step in ENDPOINTS:
            folder = batch / branch / f'eval_{split}_{step}'
            require(read(folder / 'complete.json').get('status') == 'completed_evaluation',
                    f'Evaluation incomplete: {folder}')
            require((folder / 'metrics.json').is_file(), f'Missing metrics: {folder}')
    return outer, batch


class Package:
    def __init__(self):
        self.files = {}
        self.checkpoints = {}
        self.visited_models = set()
        self.hashes = HashCache()
        self.warnings = []

    def add(self, path, category, required=True):
        path = Path(path).absolute()
        if not path.is_file():
            require(not required, f'Missing package source: {path}')
            return
        require(path.suffix not in {'.pt', '.pth', '.ckpt', '.mp4', '.avi', '.mov'},
                f'Weights/video prohibited in archive: {path}')
        relative = path.relative_to(PROJECT).as_posix()
        record = self.files.setdefault(relative, {'path': path, 'categories': []})
        if category not in record['categories']:
            record['categories'].append(category)

    def records(self, folder, category):
        for name in ('config.json', 'complete.json', 'exposure.json', 'reuse.json', 'training.jsonl'):
            self.add(folder / name, category, required=False)
        for path in (folder / 'sources').glob('*.py'):
            self.add(path, 'run_source_snapshot')

    def model(self, folder, owner, role, batch=None):
        """Index direct checkpoints, reuse aliases, and recursively referenced parents."""
        folder = Path(folder).absolute()
        visit = (str(folder), owner, role)
        if visit in self.visited_models:
            return
        self.visited_models.add(visit)
        config = read(folder / 'config.json', optional=True)
        done = read(folder / 'complete.json', optional=True)
        self.records(folder, 'model_record')
        status = done.get('status') or ('completion_receipt_present_no_status' if done else 'unknown')
        gpu = config.get('gpu') or done.get('gpu')
        gpu_id = config.get('visible_cuda') or done.get('visible_cuda')
        # For new confirmation parents, the enclosing orchestration records the
        # actual training GPU. Never substitute an evaluation GPU or current GPU.
        for event in (folder.parent / 'events' / f'{folder.name}.json',
                      folder.parent / 'events/lr_integrated.json' if folder.name == 'integrated_parent'
                      else folder / '__no_event__'):
            data = read(event, optional=True)
            if data.get('status') == 'completed':
                gpu_id = gpu_id or data.get('gpu')
        outer_done = read(folder.parent / 'complete.json', optional=True)
        actual_train = outer_done.get('gpus', {}).get('train', {})
        if actual_train and (folder.parent / 'native_warmup').is_dir():
            gpu = gpu or actual_train.get('name')
            gpu_id = gpu_id or actual_train.get('uuid')
        checkpoints = sorted(folder.glob('checkpoint*.pt'))
        for path in checkpoints:
            require(path.is_file(), f'Broken checkpoint link: {path}')
            milestone = re.fullmatch(r'checkpoint_(\d+)\.pt', path.name)
            additional = int(milestone[1]) if milestone and role == 'branch' else None
            if path.name == 'checkpoint_final.pt' and role == 'branch':
                additional = done.get('intervention_step', done.get('parameter_updates'))
            if additional is not None and 'scheduler_offset' in config:
                step = config['scheduler_offset'] + additional
                step_source = 'config.scheduler_offset + intervention_step'
            elif milestone:
                # The legacy warmup saves checkpoint_<fine-stage-step>.pt;
                # its final complete.json must not overwrite every milestone.
                step = int(milestone[1])
                step_source = 'checkpoint filename (legacy stage-local step)'
            else:
                step = done.get('step')
                step_source = 'owning completion receipt step'
            entry = self.checkpoints.setdefault(str(path), {
                'path': str(path), 'resolved_path': str(path.resolve()),
                'project_relative_path': path.relative_to(PROJECT).as_posix(),
                'symlink': path.is_symlink(), 'sha256': self.hashes.sha(path),
                'bytes': path.stat().st_size, 'checkpoint_kind': 'final' if 'final' in path.stem else 'milestone',
                'step': step, 'step_source': step_source, 'intervention_step': additional,
                'step_semantics': 'recorded model step; intervention_step is additional branch updates',
                'completion_status': status, 'completion_receipt': str(folder / 'complete.json') if done else None,
                'completion_status_scope': 'owning training run; checkpoint existence and SHA256 checked separately',
                'training_gpu': gpu, 'training_gpu_id': gpu_id,
                'training_gpu_note': 'unknown if null; no GPU identity inferred from evaluation',
                'owners': [],
            })
            ownership = {'name': owner, 'role': role, 'batch': str(batch) if batch else None}
            if ownership not in entry['owners']:
                entry['owners'].append(ownership)
        reused = read(folder / 'reuse.json', optional=True).get('source')
        if reused:
            self.model(Path(reused), owner, role, batch)
        parent = config.get('checkpoint') or done.get('args', {}).get('checkpoint')
        if parent:
            parent = Path(parent)
            require(parent.is_file(), f'Referenced local parent missing: {parent}')
            self.model(parent.parent, parent.parent.name, 'parent_model', batch)
        selection = config.get('selection') or done.get('args', {}).get('selection')
        if selection:
            self.add(Path(selection).parent / 'complete.json', 'selection_receipt')

    def batch(self, outer, batch):
        self.add(outer / 'complete.json', 'batch_completion')
        self.add(batch / 'complete.json', 'batch_completion')
        for path in (batch / 'summary').iterdir():
            if path.is_file() and path.suffix in {'.json', '.md', '.png', '.svg', '.py'}:
                self.add(path, 'batch_summary')
        for branch in BRANCHES:
            folder = batch / branch
            self.model(folder, branch, 'branch', batch)
            for split, step in ENDPOINTS:
                for name in ('complete.json', 'metrics.json'):
                    self.add(folder / f'eval_{split}_{step}' / name, 'evaluation_record')
        for folder in (outer, batch):
            for path in (folder / 'events').glob('*.json'):
                self.add(path, 'execution_receipt')

    def build_zip(self, out, index, batches):
        source_rows = []
        archive = out / 'results_package.zip'
        lookup = {}
        for name, record in self.files.items():
            lookup[str(record['path'])] = name
            lookup.setdefault(str(record['path'].resolve()), name)
        index_bytes = encoded(index)
        (out / 'checkpoint_index.json').write_bytes(index_bytes)
        readme = f'''# 动态场景超分：运动绑定细分交付包

本包收录 {len(batches)} 个已完成批次的固定端点汇总、报告、源码与必要记录。它们是**单种子、60 帧短窗先导，不等于完整标准基准结果**；连续帧不构成独立统计重复，局部感知改善不自动证明真实高频或几何恢复。

优先阅读 [主报告](docs/{REPORT.name}) 与 [Codex 阅读版](docs/{REPORT.stem}.codex.md)。批次汇总位置：

''' + ''.join(f'- [{b.name}: {b.parent.name}]({b.relative_to(PROJECT).as_posix()}/summary/summary.md)\n' for b in batches) + '''
`checkpoint_index.json` 登记全部本轮分支 checkpoint（milestones/final 与复用别名）及所引用父模型。绝对路径是本机定位信息；`resolved_path` 和 SHA256 可识别符号链接/同一权重。未知历史训练 GPU 保留 null，不用评价 GPU 替代。索引只读取文件字节和保存记录，不反序列化模型。

ZIP **不含模型权重、selection.pt、完整训练/预测图像序列、视频、数据集或 CUDA 环境**，不是离线一键训练环境。旧证据仅收录四个既定观察的 ROI、完整对照面板与记录。源码包括当前实验目录、直接公共依赖和训练时保存的源码快照；复现实验优先核对记录中的快照/hash，不把当前源码自动等同于运行时版本。

`source_manifest.json` 逐文件记录原始来源 SHA256 与打包字节 SHA256；`SHA256SUMS` 用于检验包内文件。为便于移机阅读，Markdown 中指向**已收录文件**的本机绝对链接改成相对链接；原始文件未修改，数学内容未改。未收录的历史链接仍指向本机来源。原始 manifest/config 路径不改写。

这是结果材料的 CPU 快照，不增加训练、模型诊断、论文主张或有效性判定。首次预览不含后续确认结果；最终包必须在确认批次与报告均完成后重新生成。
'''
        if self.warnings:
            readme += '\n打包提示：\n\n' + ''.join(f'- {w}\n' for w in self.warnings)
        readme_bytes = readme.encode()
        (out / 'README.md').write_bytes(readme_bytes)
        checksums = []
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for name, record in sorted(self.files.items()):
                path = record['path']
                data = path.read_bytes()
                original_sha = hashlib.sha256(data).hexdigest()
                rewritten = 0
                if path.suffix == '.md':
                    text = data.decode()
                    def replace(match):
                        nonlocal rewritten
                        target = match.group(1)
                        if target not in lookup:
                            return match.group(0)
                        rewritten += 1
                        relative = os.path.relpath(lookup[target], str(Path(name).parent))
                        return '](' + relative + ')'
                    text = re.sub(r'\]\(<?(/[^\n)]+?)>?\)', replace, text)
                    data = text.encode()
                digest = hashlib.sha256(data).hexdigest()
                zf.writestr(name, data)
                source_rows.append({'archive_path': name, 'source_path': str(path),
                    'resolved_source_path': str(path.resolve()), 'source_sha256': original_sha,
                    'archive_sha256': digest, 'archive_bytes': len(data),
                    'markdown_links_rewritten': rewritten, 'categories': record['categories']})
                checksums.append(f'{digest}  {name}\n')
            manifest = {'schema': 1, 'created_utc': index['created_utc'], 'sources': source_rows,
                        'excluded': ['model_weights', 'selection_tensor', 'complete_image_sequences', 'videos'],
                        'warnings': self.warnings}
            manifest_bytes = encoded(manifest)
            (out / 'source_manifest.json').write_bytes(manifest_bytes)
            for name, data in [('checkpoint_index.json', index_bytes), ('README.md', readme_bytes),
                               ('source_manifest.json', manifest_bytes)]:
                zf.writestr(name, data)
                checksums.append(f'{hashlib.sha256(data).hexdigest()}  {name}\n')
            zf.writestr('SHA256SUMS', ''.join(checksums))
        with zipfile.ZipFile(archive) as zf:
            require(zf.testzip() is None, 'ZIP CRC verification failed')
            require(not any(Path(n).suffix in {'.pt', '.pth', '.ckpt', '.mp4', '.avi', '.mov'}
                            for n in zf.namelist()), 'Prohibited payload found')
        return archive, len(source_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-root', type=Path, required=True)
    parser.add_argument('--confirmation-root', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; never overwrite a package')
    args = parser.parse_args()
    roots = [completed_batch(args.first_root)]
    if args.confirmation_root:
        roots.append(completed_batch(args.confirmation_root))
    out = args.out.absolute()
    require(not out.exists(), f'Output already exists; choose a new directory: {out}')
    package = Package()
    for outer, batch in roots:
        package.batch(outer, batch)
    for path in (REPORT, REPORT.with_name(REPORT.stem + '.codex.md')):
        package.add(path, 'main_report')
    math_root = PROJECT / 'docs/assets/codex_math' / REPORT.stem
    math_manifest = read(math_root / 'manifest.json')
    if math_manifest.get('source_sha256') != package.hashes.sha(REPORT):
        package.warnings.append('主报告在公式阅读版生成后有更新；最终交付前请重新运行 scripts/render_codex_math.mjs。')
    for path in math_root.iterdir():
        if path.is_file():
            package.add(path, 'report_math_asset')
    for folder in (Path(__file__).parent, PROJECT / 'experiments/dynamic_sr_20260918',
                   PROJECT / 'experiments/dynamic_sr_scene_residual_20260923'):
        for path in folder.glob('*.py'):
            package.add(path, 'current_experiment_source')
    for path in (PROJECT / 'experiments/dynamic_sr_20260920/resume_control.py',
                 PROJECT / 'scripts/render_codex_math.mjs'):
        package.add(path, 'direct_support_source')
    evidence = args.first_root.absolute().parent / 'evidence_existing'
    for name in ('README.md', 'evidence.json', 'collect_cpu.py', 'gpu_views_v1/README.md',
                 'gpu_views_v1/metrics.json', 'gpu_views_v1/complete.json'):
        package.add(evidence / name, 'existing_evidence')
    for pattern in ('*/roi_*.png', '*/full_comparison.png'):
        for path in (evidence / 'gpu_views_v1').glob(pattern):
            package.add(path, 'fixed_existing_roi')
    index = {'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Specified completed batches, their reused A sources and recursively referenced parent model directories only',
        'batches': [str(batch) for _, batch in roots],
        'checkpoint_count_including_aliases': len(package.checkpoints),
        'unique_resolved_checkpoint_files': len({r['resolved_path'] for r in package.checkpoints.values()}),
        'checkpoints': sorted(package.checkpoints.values(), key=lambda r: r['path']),
        'weights_in_archive': False, 'model_deserialization': False,
        'hash_cache': 'Within-run cache keyed by resolved path, size and mtime_ns; symlink aliases share one hash read'}
    out.mkdir(parents=True, exist_ok=False)
    archive, count = package.build_zip(out, index, [b for _, b in roots])
    receipt = {'status': 'completed_cpu_package', 'archive': str(archive),
        'archive_sha256': package.hashes.sha(archive), 'archive_bytes': archive.stat().st_size,
        'archive_mib': archive.stat().st_size / 1024**2, 'under_20_mib': archive.stat().st_size < 20 * 1024**2,
        'source_files': count, 'checkpoint_count': len(package.checkpoints),
        'checkpoint_index_sha256': package.hashes.sha(out / 'checkpoint_index.json'),
        'confirmation_included': args.confirmation_root is not None, 'warnings': package.warnings,
        'finished_utc': datetime.now(timezone.utc).isoformat()}
    (out / 'complete.json').write_bytes(encoded(receipt))
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
