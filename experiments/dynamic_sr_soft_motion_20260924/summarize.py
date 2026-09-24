#!/usr/bin/env python3
"""Summarize a declared method mapping, keeping train/dev/test and costs distinct.

Input: {"methods":{"A":{"checkpoint":"...6000.pt","train_dir":"...",
"evaluations":{"train_fixed_6000":"...","dev_6000":"...","test_6000":"..."},
"cost":{}},"B":{...},"S":{...}}}. evaldir may instead name a root with
eval_<split>_<step> folders. Existing dev_1200 is optional supplementary data.
This is CPU JSON/file processing; it never loads model tensors or runs training.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

REGIONS = ('full', 'dynamic', 'static')
MAIN = ('train_fixed_6000', 'dev_6000', 'test_6000')
ALLOWED_EVAL = ('motion_bound_eval_v1', 'soft_motion_eval_v1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def require(value, message):
    if not value:
        raise ValueError(message)


def path_from(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_methods(path):
    """Resolve paths only; export can freeze ROIs without reading predictions."""
    path = Path(path).resolve()
    raw = read(path)
    methods = raw.get('methods', raw)
    require(isinstance(methods, dict) and len(methods) >= 2, 'Declare at least two methods')
    result = {}
    for name, item in methods.items():
        require(isinstance(item, dict) and 'checkpoint' in item, f'{name}: checkpoint is required')
        checkpoint = path_from(item['checkpoint'], path.parent)
        train_dir = path_from(item.get('train_dir', str(checkpoint.parent)), path.parent)
        evaluations = item.get('evaluations', {})
        evaldir = path_from(item.get('evaldir', str(train_dir)), path.parent)
        resolved = {key: path_from(value, path.parent) for key, value in evaluations.items()}
        if not evaluations:
            for key in (*MAIN, 'dev_1200', 'test_1200', 'train_fixed_1200'):
                folder = evaldir / f'eval_{key}'
                if folder.exists():
                    resolved[key] = folder
        result[name] = {**item, 'checkpoint': checkpoint, 'train_dir': train_dir,
                        'evaldir': evaldir, 'evaluations': resolved,
                        'label': item.get('label', name), 'role': item.get('role', name)}
    return result


def teacher_identity(config):
    inputs = config.get('teacher_inputs')
    require(isinstance(inputs, list) and inputs, 'Teacher identity list is required for comparison')
    return [(r['camera'], r['frame'], r['sha256']) for r in inputs]


def expected_keys(manifest, split, cameras):
    if split == 'train_fixed':
        return sorted((o['camera_id'], o['frame_index']) for o in manifest['observations']
                      if o['split'] == 'train' and o['camera_id'] in cameras and o['frame_index'] in [0,40,80,118])
    return sorted((o['camera_id'], o['frame_index']) for o in manifest['observations'] if o['split'] == split)


def metric_identity(value):
    caches = {camera: {k:v for k,v in cache.items() if k != 'path'}
              for camera, cache in value['evaluation_caches'].items()}
    return {**{k:value[k] for k in ['manifest_sha256','metric_helpers_sha256','motion_model_sha256',
        'observation_keys','versions','lr_reprojection_protocol','gpu','lpips_device','dynamic_threshold','flow_scale']},
        'evaluation_caches':caches,
        'observed_lr_sha256':[r['observed_lr_sha256'] for r in value['rows']]}


def value_or_none(group, key):
    value = group.get(key)
    require(value is None or isinstance(value, (int,float)) and math.isfinite(value), f'Invalid metric {key}: {value}')
    return value


def summarize(manifest_path, methods_path, out):
    manifest_path, methods_path, out = map(lambda p:Path(p).resolve(), (manifest_path,methods_path,out))
    manifest, methods = read(manifest_path), load_methods(methods_path)
    manifest_hash = sha(manifest_path)
    rows, costs, provenance, identities, draws, paired = [], [], {}, {}, {}, {}
    common = None
    for name, item in methods.items():
        config_path, done_path = item['train_dir']/'config.json', item['train_dir']/'complete.json'
        config, done = read(config_path), read(done_path)
        require(done.get('status') == 'completed' and not done.get('smoke',False), f'{name}: formal training incomplete')
        require(config['manifest_sha256'] == manifest_hash, f'{name}: wrong manifest')
        teacher = teacher_identity(config)
        control = {key:config[key] for key in ['manifest_sha256','parent_sha256','seed','sr_weight',
                   'prior_cameras','prior_subdir','teacher_count','teacher_cameras','scheduler_offset','initial_points']}
        control['teacher_inputs'] = teacher
        require(float(control['sr_weight']) == .1, f'{name}: this batch requires SR weight0.1')
        require(int(config['steps']) >= 6000, f'{name}: insufficient registered training length')
        if common is None:
            common = control
        else:
            require(control == common, f'{name}: parent/seed/teacher/input/schedule mismatch')
        require(len(set(control['teacher_cameras'])) == 4, f'{name}: expected four teacher cameras')
        require(set(MAIN) <= set(item['evaluations']), f'{name}: missing fixed16 train/dev6000/test6000 evaluations')
        parent_sha = config['parent_sha256']
        checkpoint_hash = sha(item['checkpoint'])
        endpoints = {}
        for endpoint, location in item['evaluations'].items():
            split, step_text = endpoint.rsplit('_', 1)
            step = int(step_text)
            require(split in ['train_fixed','dev','test'] and step in [1200,6000], f'Unregistered endpoint: {endpoint}')
            metrics_path = location if location.name == 'metrics.json' else location/'metrics.json'
            receipt_path = metrics_path.parent/'complete.json'
            values, receipt = read(metrics_path), read(receipt_path)
            require(receipt.get('status') == 'completed_evaluation' and receipt.get('parameter_updates') == 0,
                    f'{name}/{endpoint}: evaluation incomplete')
            require(sha(metrics_path) == receipt['metrics_sha256'], f'{name}/{endpoint}: metrics changed')
            require(values['version'] in ALLOWED_EVAL, f'{name}/{endpoint}: unsupported metric implementation')
            require(values['manifest_sha256'] == manifest_hash and values['split'] == split,
                    f'{name}/{endpoint}: evaluation split/manifest mismatch')
            expected_branch={'A':'joint','B':'ordinary_split','C':'bound_split','S':'ordinary_split'}.get(item['role'])
            if expected_branch is not None:
                require(values['branch']==expected_branch,f'{name}: declared role does not match renderer branch')
                require((values.get('soft_motion') is not None)==(item['role']=='S'),
                        f'{name}: declared soft-motion status does not match checkpoint')
            metadata = values['checkpoint_metadata']
            require(metadata['intervention_step'] == step, f'{name}/{endpoint}: wrong checkpoint step')
            require(metadata.get('parent_sha') == parent_sha, f'{name}/{endpoint}: wrong evaluation parent')
            evaluated_path = Path(values['checkpoint'])
            require(sha(evaluated_path) == values['checkpoint_sha256'] == receipt['checkpoint_sha256'],
                    f'{name}/{endpoint}: checkpoint identity mismatch')
            if step == 6000:
                require(values['checkpoint_sha256'] == checkpoint_hash, f'{name}: main endpoints use different checkpoint')
            expected = expected_keys(manifest,split,control['teacher_cameras'])
            actual = [(r['camera_id'],r['frame_index']) for r in values['rows']]
            require(actual == expected and len(actual) == (16 if split=='train_fixed' else 60),
                    f'{name}/{endpoint}: incomplete or wrong observation set')
            require(len(actual) == receipt['observations'], f'{name}/{endpoint}: receipt observation count mismatch')
            require(values['lpips_status'] == 'alex_v0.1_standard_full_and_fixed_mask_spatial_map', f'{name}: LPIPS absent')
            require(values['lr_reprojection_protocol']['source_precision'] == 'float32_raw_render_before_png_quantization',
                    f'{name}: quantized LR reprojection is not comparable')
            identity = metric_identity(values)
            if endpoint in identities:
                require(identity == identities[endpoint], f'{name}/{endpoint}: metric/cache/hardware identity differs')
            else:
                identities[endpoint] = identity
            draw = metadata.get('draw_sha256')
            if draw is None and done.get('parameter_updates') == step:
                draw = done.get('draw_sha256')
            require(draw is not None, f'{name}/{endpoint}: no actual sampling stream hash')
            if step in draws:
                require(draw == draws[step], f'{name}/{endpoint}: sampling stream differs')
            else:
                draws[step] = draw
            for region in REGIONS:
                spatial = values['aggregate'].get(region,{})
                lr = values['lr_reprojection_aggregate'].get(region,{})
                temporal = values.get('temporal_aggregate',{}).get(region,{})
                row = {'method':name,'label':item['label'],'role':item['role'],'split':split,'step':step,'region':region,
                       'psnr':value_or_none(spatial,'psnr_mean'),'ssim':value_or_none(spatial,'ssim_mean'),
                       'lpips':value_or_none(spatial,'lpips_alex_mean' if region=='full' else 'lpips_alex_spatial_mask_mean'),
                       'temporal':value_or_none(temporal,'gt_relative_warp_l1_mean'),
                       'lr_psnr':value_or_none(lr,'psnr_mean'),'lr_mse':value_or_none(lr,'mse_mean'),
                       'lr_l1':value_or_none(lr,'l1_mean')}
                rows.append(row)
            paired[name,endpoint] = values
            endpoints[endpoint] = {'path':str(metrics_path),'sha256':sha(metrics_path),
                                   'evaluation_script_sha256':values['script_sha256'],
                                   'checkpoint_sha256':values['checkpoint_sha256'],
                                   'render_seconds':values['render_seconds'],'elapsed_seconds':values['elapsed_seconds']}
        values=paired[name,'test_6000']
        capacity=values['capacity']
        cost=item.get('cost',{})
        if isinstance(cost,str):
            cost=read(path_from(cost,methods_path.parent))
        require(isinstance(cost,dict),f'{name}: cost must be an object or JSON filename')
        if int(done.get('parameter_updates',0)) != 6000:
            require('train_seconds' in cost, f'{name}: reused early endpoint needs its own training cost, not whole18k cost')
        soft = values.get('soft_motion') or {}
        preparation_total = soft.get('reference_calibration_seconds')
        cache_build = soft.get('cache_build_seconds')
        calibration_only = (max(0.0, preparation_total-cache_build)
                            if isinstance(preparation_total,(int,float)) and isinstance(cache_build,(int,float)) else None)
        costs.append({'method':name,'label':item['label'],'role':item['role'],**capacity,
                      'train_seconds':cost.get('train_seconds',done.get('train_s')),
                      'wall_seconds':cost.get('wall_seconds',done.get('wall_s')),
                      'peak_allocated_gb':cost.get('peak_allocated_gb',done.get('peak_gb')),
                      'checkpoint_bytes':item['checkpoint'].stat().st_size,
                      'reference_preparation_seconds':cost.get('reference_preparation_seconds',cache_build),
                      'reference_cache_bytes':cost.get('reference_cache_bytes',soft.get('cache_bytes')),
                      'calibration_seconds':cost.get('calibration_seconds',calibration_only),
                      'reference_and_calibration_total_seconds':cost.get('reference_and_calibration_total_seconds',preparation_total),
                      'render_seconds_test60':values['render_seconds'],
                      'extra_cost_record':cost,
                      'cost_note':'Missing costs mean unavailable, not zero. S metadata reference_calibration_seconds includes cache construction: calibration-only is total minus cache time, not added twice. Rendering excludes metrics/I/O.'})
        provenance[name]={'checkpoint':str(item['checkpoint']),'checkpoint_sha256':checkpoint_hash,
                          'config':str(config_path),'config_sha256':sha(config_path),
                          'training_receipt':str(done_path),'training_receipt_sha256':sha(done_path),
                          'training_sources':config.get('sources'), 'selection_sha256':config.get('selection_sha256'),
                          'soft_motion':values.get('soft_motion'),'endpoints':endpoints}
    by_role={item['role']:name for name,item in methods.items()}
    if 'B' in by_role and 'S' in by_role:
        b,s=by_role['B'],by_role['S']
        require(provenance[b]['selection_sha256'] is not None and provenance[b]['selection_sha256']==provenance[s]['selection_sha256'],
                'B/S must use the same saved split selection')
        for field in ['active_gaussians','stored_gaussians','trainable_parameters','child_parameters']:
            cb=next(c for c in costs if c['method']==b);cs=next(c for c in costs if c['method']==s)
            require(cb.get(field)==cs.get(field),f'B/S inference capacity differs: {field}')
        require(paired[s,'test_6000'].get('soft_motion') is not None,'S evaluation did not identify soft-motion metadata')
    deltas=[]
    index={(r['method'],r['split'],r['step'],r['region']):r for r in rows}
    if 'S' in by_role:
        for row in [r for r in rows if r['method']==by_role['S']]:
            for baseline in methods:
                key=(baseline,row['split'],row['step'],row['region'])
                if baseline==row['method'] or key not in index:
                    continue
                other=index[key]
                deltas.append({'comparison':f'{row["method"]}-{baseline}','split':row['split'],'step':row['step'],'region':row['region'],
                               **{k:None if row[k] is None or other[k] is None else row[k]-other[k]
                                  for k in ['psnr','ssim','lpips','temporal','lr_psnr','lr_mse','lr_l1']}})
    out.mkdir(parents=True,exist_ok=False)
    result={'status':'completed_descriptive_summary','scene':manifest['scene'],'manifest':str(manifest_path),
            'manifest_sha256':manifest_hash,'methods_mapping':str(methods_path),'methods_mapping_sha256':sha(methods_path),
            'metrics':rows,'deltas':deltas,'costs':costs,'provenance':provenance,'common_control':common,
            'evaluation_identity':identities,'draw_sha256_by_step':draws,'summary_script_sha256':sha(__file__),
            'lpips_audit':'legacy.spatial_metrics consumes the COMPLETE RGB pair. It computes scalar full LPIPS, then spatial=True map; only map values are averaged within fixed masks. No outside-region RGB zeroing. Regional spatial means cannot be assembled into standard full scalar LPIPS.',
            'limitations':['Fixed16 training diagnostics are separate from full60 held-out-view development comparisons.',
                           'Current scenes/views have informed development; held out from gradients does not mean untouched by method design.',
                           'One seed, synthetic known x4 degradation and60-frame short windows; no significance or full benchmark claim.',
                           'Sources may differ across training/evaluation wrappers; actual identities retained and numeric helper/cache/hardware controls checked.',
                           'No best-step selection; available1200 results supplementary,6000 remains the fixed main endpoint.',
                           'No automated decision to expand experiments is made by this summary.'],
            'finished_utc':datetime.now(timezone.utc).isoformat()}
    write(out/'summary.json',result)
    for filename,data in [('metrics.csv',rows),('deltas.csv',deltas),('costs.csv',costs)]:
        if data:
            with (out/filename).open('w',newline='') as stream:
                fields=list(dict.fromkeys(key for row in data for key in row))
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(data)
    lines=[f'# {manifest["scene"]}：软运动约束固定端点', '',
           '训练固定16张与 dev/test 全60帧分开报告。6000为固定主端点；已有1200仅作补充。所有指标取浮点渲染，区域为时间变化评价代理，不是语义分割。', '',
           '区域 LPIPS 先对完整 RGB 图像计算 spatial map，再在区域内平均；区域外像素未置零。它与标准全图标量 LPIPS 不同，不能按区域比例拼成全图分数。', '',
           '| 划分/步数 | 方法 | 区域 | PSNR | SSIM | LPIPS | 时序 L1 | LR PSNR | LR L1 |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    fmt=lambda x:'—' if x is None else f'{x:.6f}'
    for r in rows:
        lines.append(f'| {r["split"]}/{r["step"]} | {r["label"]} | {r["region"]} | '+' | '.join(fmt(r[k]) for k in ['psnr','ssim','lpips','temporal','lr_psnr','lr_l1'])+' |')
    lines += ['', '| 方法 | 活跃点数 | 可训练参数 | 训练秒 | 参考缓存+标定秒 | 参考缓存字节 | 60帧渲染秒 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in costs:
        lines.append(f'| {c["label"]} | {c["active_gaussians"]} | {c["trainable_parameters"]} | '
                     + ' | '.join(fmt(c.get(k)) for k in ['train_seconds','reference_and_calibration_total_seconds','reference_cache_bytes','render_seconds_test60'])+' |')
    lines += ['', '完整差值、点数/参数/检查点体积与训练、缓存、标定、推理成本见 CSV/JSON。缺失成本记为未提供，不当作零；单次渲染计时不包含指标和 I/O。', '',
              '当前场景与留出视角均已参与开发分析；本表不作显著性、普遍有效性或真实高频恢复的自动结论。', '']
    (out/'summary.md').write_text('\n'.join(lines))
    write(out/'complete.json',{'status':'completed_summary','summary_sha256':sha(out/'summary.json'),'cpu_only':True,'parameter_updates':0})
    print(json.dumps({'out':str(out),'methods':list(methods),'metric_rows':len(rows)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--methods',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.manifest,args.methods,args.out)
