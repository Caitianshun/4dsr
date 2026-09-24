#!/usr/bin/env python3
"""Summarize U/W/G/L, coverage B4, historical methods and post-SR separately.

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
ALLOWED_EVAL = ('motion_bound_eval_v1', 'soft_motion_eval_v1', 'detail_supervision_eval_v1')


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
    require(isinstance(methods, dict) and len(methods) >= 1, 'Declare at least one method')
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
                        'label': item.get('label', name), 'role': item.get('role', name),
                        'group': item.get('group', 'main' if item.get('role',name) in ['U','W','G','L'] else
                                         'postprocess' if item.get('role',name) in ['B4_post','postprocess'] else
                                         'coverage_reference' if item.get('role',name) in ['B','B4'] else 'historical')}
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
    return {**{k:value[k] for k in ['manifest_sha256','metric_helpers_sha256',
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
    rows, costs, provenance, identities, draws, paired, h_rows = [], [], {}, {}, {}, {}, []
    common, main_control, detail_identity, teacher_hashes = None, None, None, {}
    main_methods = [n for n,i in methods.items() if i['role'] in ['U','W','G','L']]
    if main_methods:
        require({methods[n]['role'] for n in main_methods} == {'U','W','G','L'}, 'Main summary waits for all U/W/G/L')
    original_cameras = ['cam02','cam04','cam08','cam12'] if 'MeetRoom' in manifest.get('dataset_family','') else ['cam02','cam06','cam12','cam18']
    for name, item in methods.items():
        config_path, done_path = item['train_dir']/'config.json', item['train_dir']/'complete.json'
        config, done = read(config_path), read(done_path)
        require(done.get('status') == 'completed' and not done.get('smoke',False), f'{name}: formal training incomplete')
        require(config['manifest_sha256'] == manifest_hash, f'{name}: wrong manifest')
        teacher = teacher_identity(config)
        control = {key:config[key] for key in ['manifest_sha256','parent_sha256','seed','scheduler_offset','initial_points']}
        require(int(config['steps']) >= 6000, f'{name}: insufficient registered training length')
        if common is None:
            common = control
        else:
            require(control == common, f'{name}: parent/seed/input/schedule mismatch')
        for camera,frame,digest in teacher:
            key = (camera,frame)
            require(key not in teacher_hashes or teacher_hashes[key] == digest, 'Shared original teacher identity changed')
            teacher_hashes[key] = digest
        if item['role'] in ['U','W','G','L']:
            require(config.get('method') == item['role'], f'{name}: training method label mismatch')
            require(float(config['sr_weight']) == (.2 if item['role']=='W' else .1), f'{name}: wrong registered RGB weight')
            mc = {key:config[key] for key in ['prior_cameras','prior_subdir','teacher_count','teacher_cameras','selection_sha256']}
            mc['teacher_inputs'] = teacher
            mc['initial_identity'] = config['initial_identity']
            if main_control is None: main_control = mc
            else: require(mc == main_control, 'U/W/G/L teacher pool or split initialization differs')
            require(set(config['teacher_cameras']) == set(manifest['splits']['train']), 'U/W/G/L must cover exactly the legal training cameras')
        required = ('dev_6000','test_6000') if item['group']=='postprocess' else MAIN
        require(set(required) <= set(item['evaluations']), f'{name}: missing registered main evaluations')
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
            expected_branch={'A':'joint','B':'ordinary_split','B4':'ordinary_split','C':'bound_split','S':'ordinary_split','U':'ordinary_split','W':'ordinary_split','G':'ordinary_split','L':'ordinary_split','B4_post':'ordinary_split','postprocess':'ordinary_split'}.get(item['role'])
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
            if item['role'] in ['U','W','G','L']:
                dm = values.get('detail_supervision')
                require(dm is not None and dm['method'] == item['role'], f'{name}: detail metadata absent/wrong')
                require(values.get('postprocess') is None, 'U/W/G/L must not use post-SR inference')
            require((values.get('postprocess') is not None) == (item['group']=='postprocess'), f'{name}: post-SR grouping mismatch')
            expected = expected_keys(manifest,split,original_cameras)
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
            if values['version'] == 'detail_supervision_eval_v1':
                di = {k:values[k] for k in ['detail_operator','detail_operator_sha256','roi_protocol_sha256','detail_protocol']}
                if detail_identity is None: detail_identity = di
                else: require(di == detail_identity, f'{name}: H operator or fixed ROI protocol differs')
                for reference, key in [('HR','h_hr_aggregate'),('train_teacher','h_teacher_aggregate')]:
                    for region, metrics in values[key].items():
                        h_rows.append({'method':name,'group':item['group'],'split':split,'step':step,
                                       'reference':reference,'region':region,'h_l1':value_or_none(metrics,'l1_mean'),
                                       'h_mse':value_or_none(metrics,'mse_mean')})
            elif item['role'] in ['U','W','G','L','B4']:
                raise ValueError(f'{name}: current main/B4/post reference requires new H evaluation')
            draw = metadata.get('draw_sha256')
            if draw is None and done.get('parameter_updates') == step:
                draw = done.get('draw_sha256')
            require(draw is not None, f'{name}/{endpoint}: no actual sampling stream hash')
            draw_group = 'UWGL' if item['role'] in ['U','W','G','L'] else name
            draw_key = f'{draw_group}/{step}'
            if draw_key in draws: require(draw == draws[draw_key], f'{name}/{endpoint}: within-group sampling stream differs')
            else: draws[draw_key] = draw
            if item['role'] in ['U','W','G','L']:
                schedule = values['detail_supervision'].get('schedule_sha') or values['detail_supervision'].get('schedule_sha256')
                require(schedule is not None, f'{name}: fixed UWGL schedule identity missing')
                schedule_key = 'UWGL/schedule'
                if schedule_key in draws: require(schedule == draws[schedule_key], 'U/W/G/L schedule mismatch')
                else: draws[schedule_key] = schedule
            for region in REGIONS:
                spatial = values['aggregate'].get(region,{})
                lr = values['lr_reprojection_aggregate'].get(region,{})
                temporal = values.get('temporal_aggregate',{}).get(region,{})
                row = {'method':name,'label':item['label'],'role':item['role'],'group':item['group'],'split':split,'step':step,'region':region,
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
        if item['role'] in ['G','L']:
            require(config['geometry_protocol']['basis']['kind']==item['role'],'Basis branch mismatch')
            require(values['motion_model_sha256']==sha(Path(__file__).with_name('geometry_model.py')),'Wrong geometry evaluator')
        else:
            require(values['motion_model_sha256']==sha(Path(__file__).resolve().parents[1]/'dynamic_sr_motion_bound_20260923/motion_model.py'),'Wrong baseline renderer')
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
        detail = values.get('detail_supervision') or {}
        calibration_only = detail.get('calibration_seconds', calibration_only)
        costs.append({'method':name,'label':item['label'],'role':item['role'],'group':item['group'],**capacity,
                      'training_gpu':config.get('gpu'),'training_torch':config.get('torch'),'evaluation_gpu':values['gpu'],
                      'postprocess':values.get('postprocess_cost'),
                      'gradient_audit_seconds':done.get('gradient_audit_seconds'),
                      'training_reused_for_postprocess':item['group']=='postprocess',
                      'additional_training_seconds':0 if item['group']=='postprocess' else None,
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
                          'soft_motion':values.get('soft_motion'),'detail_supervision':values.get('detail_supervision'),'endpoints':endpoints}
    by_role={item['role']:name for name,item in methods.items()}
    if main_methods:
        cc={c['method']:c for c in costs}
        for field in ['active_gaussians','stored_gaussians','child_parameters']:
            require(len({cc[n][field] for n in main_methods})==1,'Fixed Gaussian/child capacity changed')
        require(cc['G']['trainable_parameters']==cc['L']['trainable_parameters'],'G/L parameter count mismatch')
        require(cc['G']['trainable_parameters']>cc['U']['trainable_parameters'],'Residual capacity not counted')
        for n in main_methods:
            require(cc[n]['training_gpu']==cc['U']['training_gpu'],'Main comparison training hardware mismatch')
    deltas=[]
    index={(r['method'],r['split'],r['step'],r['region']):r for r in rows}
    pairs = [('G','U'),('L','U'),('L','G'),('G','W'),('L','W'),('G','B4'),('L','B4'),('U','B4'),('W','B4')]
    for role, baseline_role in pairs:
        if role not in by_role or baseline_role not in by_role: continue
        name, baseline = by_role[role], by_role[baseline_role]
        for row in [r for r in rows if r['method']==name]:
            key=(baseline,row['split'],row['step'],row['region'])
            if key not in index: continue
            other=index[key]
            deltas.append({'comparison':f'{name}-{baseline}','group':methods[name]['group'],
                           'split':row['split'],'step':row['step'],'region':row['region'],
                           **{k:None if row[k] is None or other[k] is None else row[k]-other[k]
                              for k in ['psnr','ssim','lpips','temporal','lr_psnr','lr_mse','lr_l1']}})
    out.mkdir(parents=True,exist_ok=False)
    result={'status':'completed_descriptive_summary','scene':manifest['scene'],'manifest':str(manifest_path),
            'manifest_sha256':manifest_hash,'methods_mapping':str(methods_path),'methods_mapping_sha256':sha(methods_path),
            'metrics':rows,'h_metrics':h_rows,'detail_identity':detail_identity,'deltas':deltas,'costs':costs,'provenance':provenance,'common_control':common,
            'main_teacher_control':main_control,'evaluation_identity':identities,'draw_sha256_by_step':draws,'summary_script_sha256':sha(__file__),
            'lpips_audit':'legacy.spatial_metrics consumes the COMPLETE RGB pair. It computes scalar full LPIPS, then spatial=True map; only map values are averaged within fixed masks. No outside-region RGB zeroing. Regional spatial means cannot be assembled into standard full scalar LPIPS.',
            'limitations':['Fixed16 training diagnostics are separate from full60 held-out-view development comparisons.',
                           'Current scenes/views have informed development; held out from gradients does not mean untouched by method design.',
                           'One seed, synthetic known x4 degradation and60-frame short windows; no significance or full benchmark claim.',
                           'Sources may differ across training/evaluation wrappers; actual identities retained and numeric helper/cache/hardware controls checked.',
                           'No best-step selection; available1200 results supplementary,6000 remains the fixed main endpoint.',
                           'No automated decision to expand experiments is made by this summary.'],
            'finished_utc':datetime.now(timezone.utc).isoformat()}
    write(out/'summary.json',result)
    for filename,data in [('metrics.csv',rows),('deltas.csv',deltas),('costs.csv',costs),('h_metrics.csv',h_rows)]:
        if data:
            with (out/filename).open('w',newline='') as stream:
                fields=list(dict.fromkeys(key for row in data for key in row))
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(data)
    lines=[f'# {manifest["scene"]}：子点中心时间残差验证', '',
           '固定6000步；原四相机16张训练诊断、dev/test各60帧分列。G/L沿用U的教师、采样和监督，新增世界中心时间表达；G/L参数量相同，均高于U。W为强监督对照，B4为四教师覆盖历史参照。', '',
           'PSNR/SSIM越高越好；LPIPS、H误差、相对GT时序L1、LR误差越低越好。区域LPIPS为完整RGB空间图固定区域均值，不能拼成标准全图LPIPS。', '']
    fmt=lambda x:'—' if x is None else f'{x:.6f}'
    for group in ['main','coverage_reference','historical','postprocess']:
        selected=[r for r in rows if r['group']==group]
        if not selected: continue
        lines += [f'## {group}', '', '| 划分/步数 | 方法 | 区域 | PSNR | SSIM | LPIPS | 时序L1 | LR PSNR |',
                  '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for r in selected:
            lines.append(f'| {r["split"]}/{r["step"]} | {r["label"]} | {r["region"]} | '+' | '.join(fmt(r[k]) for k in ['psnr','ssim','lpips','temporal','lr_psnr'])+' |')
        lines += ['']
    lines += ['## 有符号降采样残差误差', '', 'H在完整浮点图像上计算，再取固定区域。只在合法训练观察对教师评价；cam01没有新增局部框。H误差不是纹理真实性证明。', '',
              '| 划分 | 方法 | 参照 | 区域 | H L1 |', '| --- | --- | --- | --- | ---: |']
    for r in h_rows:
        lines.append(f'| {r["split"]}/{r["step"]} | {r["method"]} | {r["reference"]} | {r["region"]} | {fmt(r["h_l1"])} |')
    lines += ['', '## 实际成本', '', '| 方法 | 训练GPU | 活跃点 | 参数 | 训练秒 | 一次标定秒 | 峰值GB | 60帧3D渲染秒 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in costs:
        lines.append(f'| {c["label"]} | {c["training_gpu"]} | {c["active_gaussians"]} | {c["trainable_parameters"]} | '+
                     ' | '.join(fmt(c.get(k)) for k in ['train_seconds','calibration_seconds','peak_allocated_gb','render_seconds_test60'])+' |')
        if c.get('postprocess'):
            q=c['postprocess']
            lines += ['', f'后处理 {c["label"]}：额外网络张量{q["network_tensor_bytes"]} bytes；额外平均{q["extra_seconds_per_frame"]:.6f}s/帧；实际进程峰值{q["peak_allocated_bytes"]} bytes。网络显存与额外临时峰值详见JSON，不把3D单次渲染时间当完整FPS。', '']
    lines += ['', '训练硬件不同时不把耗时比归为算法开销；新增教师覆盖/缓存成本需另列，不能称相同信息预算。检查点字节含训练状态，不是纯推理权重体积。', '',
              'G/L新增容量比较与U/W/B4强对照具有不同解释范围；本脚本不自动触发扩展实验。单种子、已观察开发视角与60帧合成×4短窗不支持显著性/完整基准/真实纹理结论。', '']
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
