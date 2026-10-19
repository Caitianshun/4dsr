#!/usr/bin/env python3
"""CPU-only consumer of identified evaluation previews; never renders a model.

Example:
  python3 make_render_comparison.py --manifest MANIFEST --methods METHODS.json \
      --out NEW_REVIEW_DIRECTORY

METHODS.json contains {"methods": [{"name": "Wu_matched", "model": "Wu",
"arm": "areaTm", "summary": "/.../summary.json", "previews": "/.../previews"},
{"name": "Multi4D_matched", "model": "Multi4D", "arm": "areaTm",
"summary": "/.../summary.json", "previews": "/.../previews"}]}.
Missing producers remain pending. No substitute prediction is manufactured.
Only numpy and Pillow are imported; no torch, checkpoint loading or CUDA.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FIXED_FRAMES = (0, 150, 299)
CONSECUTIVE_FRAMES = (149, 150, 151)
COLORS = ('#c42030', '#087dba', '#1b8a45', '#a45a00', '#7654a3')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def resolve(value, relative_to=ROOT):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def font(size=18):
    for candidate in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                      '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def safe_name(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Method names must use letters, digits, underscore or hyphen')
    return name


def checked_image(manifest_path, record, kind, expected_size):
    path = resolve(record[kind + '_path'], manifest_path.parent)
    if not path.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError('Reference image escapes manifest directory')
    actual = identity(path)
    if actual['sha256'] != record[kind + '_sha256']:
        raise ValueError('Reference image SHA mismatch: ' + str(path))
    with Image.open(path) as source:
        image = source.convert('RGB')
    if image.size != tuple(expected_size):
        raise ValueError('Reference image resolution mismatch: ' + str(path))
    return image, actual


def load_roi(path, manifest_path, manifest):
    value = read(path)
    if value.get('status') != 'fixed_before_prediction_reads' or value.get('scene') != manifest['scene']:
        raise ValueError('An identified, pre-fixed ROI registry for this scene is required')
    declared = value.get('manifest_sha256')
    if isinstance(value.get('manifest'), dict):
        declared = value['manifest'].get('sha256', declared)
    inherited = declared != sha(manifest_path)
    references = value.get('reference_images', [])
    if inherited and not references:
        raise ValueError('Inherited ROIs require original reference-image hashes')
    lookup = {(r['camera_id'], int(r['frame_index'])): r for r in manifest['observations']}
    for reference in references:
        row = lookup[(reference['camera'], int(reference['frame']))]
        _, actual = checked_image(manifest_path, row, 'hr', manifest['resolutions']['hr'])
        if actual['sha256'] != reference['sha256']:
            raise ValueError('Inherited ROI reference pixels changed')
    width, height = manifest['resolutions']['hr']
    for boxes in value['regions_by_camera_xyxy_exclusive'].values():
        for box in boxes.values():
            if len(box) != 4 or not all(type(x) is int for x in box):
                raise ValueError('ROI coordinates must be integer xyxy-exclusive')
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError('ROI coordinates exceed the HR image')
    return value, dict(source=identity(path), inherited_from_short_window=inherited,
        reference_image_hash_check='passed', interpretation=value.get('interpretation'),
        coordinates='xyxy-exclusive at native HR resolution; never tracked or reselected')


def method_source(entry, manifest_sha):
    name = safe_name(entry['name'])
    summary_path = resolve(entry['summary'])
    previews = resolve(entry.get('previews', str(summary_path.parent / 'previews')))
    result = dict(name=name, label=entry.get('label', name), model=entry['model'],
        arm=entry.get('arm'), summary_path=str(summary_path), previews=str(previews),
        status='pending', frames={}, supervision='HR_oracle_diagnostic_only'
        if entry.get('arm') == 'areaOracle' else 'legal_LR_optional_frozen_teacher')
    if not summary_path.is_file():
        return dict(result, reason='summary_missing')
    result['summary'] = identity(summary_path)
    value = read(summary_path)
    result.update(evaluation_status=value.get('status'), evaluated_frames=value.get('frames'),
        evaluation_gpu=value.get('gpu'), host=value.get('host'),
        training_hardware=value.get('checkpoint_metadata', {}).get('hardware'),
        posthoc_cull=value.get('posthoc_cull'), cull=value.get('cull'),
        render_variant=value.get('render_variant', 'posthoc_cull' if value.get('posthoc_cull') else None),
        declared_checkpoint=value.get('checkpoint'), declared_manifest=value.get('manifest'),
        full300_metrics=value.get('test'))
    declared_manifest = value.get('manifest')
    if not isinstance(declared_manifest, dict) or declared_manifest.get('sha256') != manifest_sha:
        return dict(result, status='failed', reason='summary_manifest_identity_mismatch')
    checkpoint = value.get('checkpoint')
    if not isinstance(checkpoint, dict) or not checkpoint.get('sha256') or not checkpoint.get('path'):
        return dict(result, status='failed', reason='checkpoint_identity_not_recorded')
    checkpoint_path = resolve(checkpoint['path'])
    if not checkpoint_path.is_file():
        return dict(result, reason='checkpoint_file_missing_for_identity_verification')
    actual_checkpoint = identity(checkpoint_path)
    result['actual_checkpoint'] = actual_checkpoint
    if actual_checkpoint['sha256'] != checkpoint['sha256']:
        return dict(result, status='failed', reason='checkpoint_sha_mismatch')
    if value.get('status') == 'failed':
        return dict(result, status='failed', reason='producer_evaluation_failed')
    if value.get('status') != 'completed' or value.get('frames') != 300:
        return dict(result, reason='producer_evaluation_not_completed_full300')
    if entry['model'] == 'Multi4D' and value.get('method') != entry.get('arm'):
        return dict(result, status='failed', reason='Multi4D_summary_arm_mismatch')
    if result['posthoc_cull'] is None:
        return dict(result, status='failed', reason='cull_state_not_recorded')
    return dict(result, status='ready',
        preview_binding='Consumer hashes actual NPZ bytes. Legacy summaries do not cryptographically bind preview hashes; association is explicit supplied path plus evaluator naming/schema, not embedded NPZ camera/checkpoint metadata.')


def prediction(source, camera, frame, size):
    path = Path(source['previews']) / f'{camera}_{frame:04d}.npz'
    if not path.is_file():
        return None, dict(status='pending', path=str(path), reason='raw_npz_missing')
    actual = identity(path)
    with np.load(path, allow_pickle=False) as data:
        if 'rgb_raw' not in data:
            raise ValueError('Preview NPZ lacks rgb_raw: ' + str(path))
        raw = data['rgb_raw']
    width, height = size
    if raw.shape != (3, height, width) or not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
        raise ValueError('Expected finite floating rgb_raw with CHW HR shape: ' + str(path))
    # Identical fixed conversion for every model, independent of cached PNGs.
    rgb = np.rint(np.clip(raw.transpose(1, 2, 0), 0., 1.) * 255.).astype(np.uint8)
    return Image.fromarray(rgb, 'RGB'), dict(status='ready', actual_npz=actual,
        raw_shape=list(raw.shape), raw_dtype=str(raw.dtype),
        minimum=float(raw.min()), maximum=float(raw.max()),
        display_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())


def sheet(panels, title, panel_width=448, crop=False, notes=()):
    if not panels:
        raise ValueError('Cannot create an empty comparison sheet')
    panel_height = round(panels[0][1].height * panel_width / panels[0][1].width)
    top, header, foot, margin = 48, 52, max(36, 24 * len(notes)), 12
    canvas = Image.new('RGB', (margin + len(panels) * (panel_width + margin),
                              top + header + panel_height + foot + margin), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), title, fill='#111111', font=font(19))
    for index, (label, image) in enumerate(panels):
        x = margin + index * (panel_width + margin)
        draw.multiline_text((x, top), label, fill='#111111', font=font(16), spacing=3)
        resample = Image.Resampling.NEAREST if crop else Image.Resampling.BILINEAR
        view = image.resize((panel_width, panel_height), resample)
        canvas.paste(view, (x, top + header))
    for line, note in enumerate(notes):
        draw.text((margin, top + header + panel_height + 8 + 23 * line), note,
                  fill='#444444', font=font(15))
    return canvas


def stack(images):
    canvas = Image.new('RGB', (max(i.width for i in images), sum(i.height for i in images)), '#dddddd')
    y = 0
    for image in images:
        canvas.paste(image, (0, y)); y += image.height
    return canvas


def run(args):
    manifest_path = args.manifest.resolve()
    manifest = read(manifest_path)
    if manifest.get('scale') != 4 or manifest.get('physical_lr_derivation', {}).get('formation') != 'area4':
        raise ValueError('This comparison requires the registered scale4 area4 manifest')
    if manifest.get('splits', {}).get('test') != ['cam00'] or args.camera != 'cam00':
        raise ValueError('Current registered comparison is held-out cam00 only')
    lookup = {(r['camera_id'], int(r['frame_index'])): r for r in manifest['observations']}
    if len(lookup) != len(manifest['observations']):
        raise ValueError('Duplicate camera/frame observation')
    if any((args.camera, f) not in lookup for f in [*args.frames, *args.consecutive_frames]):
        raise ValueError('Requested frame absent from manifest')
    roi_path = args.roi_protocol or ROOT / 'output/dynamic_sr_soft_motion_20260924/roi_registry' / manifest['scene'] / 'roi_protocol.json'
    registry, roi_record = load_roi(roi_path, manifest_path, manifest)
    boxes = registry['regions_by_camera_xyxy_exclusive'][args.camera]
    entries = read(args.methods)['methods']
    if (not entries or len({e['name'] for e in entries}) != len(entries)
            or len({e.get('label', e['name']) for e in entries}) != len(entries)):
        raise ValueError('Methods must contain unique nonempty names and labels')
    sources = [method_source(e, sha(manifest_path)) for e in entries]
    observations = read(args.roi_observations) if args.roi_observations else {}
    if args.roi_observations and observations.get('manifest_sha256') != sha(manifest_path):
        raise ValueError('HR-only ROI content observations must identify this exact manifest')
    args.out.mkdir(parents=True, exist_ok=False)
    frames_dir = args.out / 'frames'; frames_dir.mkdir()
    roi_dir = args.out / 'fixed_rois'; roi_dir.mkdir()
    result = dict(schema='identified_render_comparison_v1', status='incomplete',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scene=manifest['scene'], camera=args.camera, frames=args.frames,
        still_frames_are_consecutive=len(args.frames) > 1 and all(b == a + 1 for a, b in zip(args.frames, args.frames[1:])),
        manifest=identity(manifest_path), methods_input=identity(args.methods),
        consumer=identity(Path(__file__)), methods=sources, roi_protocol=roi_record,
        ROI_semantics='Names describe the old reference frame40. Fixed coordinates are not tracked faces, text masks or semantic correspondence. Each requested frame needs content review.',
        display=dict(predictions='rgb_raw CHW -> HWC; clip[0,1]; numpy.rint(*255); uint8 RGB, identical for all methods',
            HR='verified original RGB pixels', LR='verified original LR, display-only nearest-neighbour x4',
            full_sheet='all native images downsampled by the same fixed Pillow BILINEAR transform',
            ROI_sheet='native fixed crop, nearest-neighbour x2; no adaptive sharpening, exposure, contrast, tone mapping or per-method adjustment'),
        CPU_runtime=dict(numpy=np.__version__, Pillow=Image.__version__, torch_imported='torch' in sys.modules,
            CUDA_or_model_execution=False, font=identity(font().path) if hasattr(font(), 'path') else None),
        required_models=args.required_models, frames_evidence=[], artifacts=[],
        human_blind_review=dict(performed=False, accepted=False, reason='Labeled evidence only; no blind/randomized human assessment has been performed'),
        conclusions=dict(model_winner='undetermined', visual_acceptance='not_assessed',
            caveat='Wu uses recorded posthoc culling; Multi4D uses native branches. These are complete pipeline images, not a capacity-only comparison.'))
    if args.roi_observations:
        result['roi_observations_input'] = identity(args.roi_observations)

    def save(image, path, kind, **extra):
        image.save(path)
        result['artifacts'].append(dict(kind=kind, **identity(path), **extra))

    def gather(frame, record_frames=True):
        row = lookup[(args.camera, frame)]
        hr, hr_id = checked_image(manifest_path, row, 'hr', manifest['resolutions']['hr'])
        lr, lr_id = checked_image(manifest_path, row, 'lr', manifest['resolutions']['lr'])
        panels = [('HR reference', hr), ('LR nearest x4\n(display only)', lr.resize(hr.size, Image.Resampling.NEAREST))]
        frame_record = dict(frame=frame, HR=hr_id, LR=lr_id, methods={}, fixed_rois={})
        for source in sources:
            if source['status'] != 'ready':
                frame_record['methods'][source['name']] = dict(status=source['status'], reason=source.get('reason'))
                continue
            try:
                image, meta = prediction(source, args.camera, frame, manifest['resolutions']['hr'])
            except (ValueError, OSError) as error:
                image, meta = None, dict(status='failed', reason=str(error))
            frame_record['methods'][source['name']] = meta
            if record_frames:
                source['frames'][str(frame)] = meta
            if image is not None:
                cull = 'posthoc cull' if source['posthoc_cull'] else 'native inference'
                oracle = ' | HR ORACLE' if source['supervision'] == 'HR_oracle_diagnostic_only' else ''
                panels.append((source['label'] + '\n' + cull + oracle, image))
        return panels, frame_record

    full_sheets, roi_overview = [], []
    for frame in args.frames:
        panels, frame_record = gather(frame)
        directory = frames_dir / f'{args.camera}_{frame:04d}'; directory.mkdir()
        for label, image in panels:
            name = 'HR' if label == 'HR reference' else 'LR_x4_nearest' if label.startswith('LR nearest') else next(s['name'] for s in sources if label.startswith(s['label'] + '\n'))
            save(image, directory / (name + '.png'), 'native_full_image', frame=frame, method=name)
        current = sheet(panels, f'{manifest["scene"]} | {args.camera} | frame {frame:03d} | labeled evidence; missing methods omitted',
            notes=(f'Selected frames: {args.frames}. Consecutive: {result["still_frames_are_consecutive"]}.', 'Missing methods are omitted; no placeholder prediction. Human blind accepted = false.'))
        save(current, directory / 'contact_sheet.png', 'same_frame_full_sheet', frame=frame)
        full_sheets.append(current)
        for index, (name, box) in enumerate(boxes.items()):
            content = observations.get(args.camera, {}).get(str(frame), {}).get(name,
                dict(status='unverified_current_frame', note='Reference-frame label only; semantic content at this fixed location has not been assessed.'))
            frame_record['fixed_rois'][name] = dict(box=box, content_observation=content)
            cropped = [(label, image.crop(tuple(box))) for label, image in panels]
            current_roi = sheet(cropped, f'{args.camera} frame {frame:03d} | {name} | box={box}', panel_width=336, crop=True,
                notes=('Coordinates fixed from HR frame40; not a tracked semantic region.', 'Current content: ' + content.get('note', content.get('status', 'unverified'))))
            save(current_roi, roi_dir / f'{args.camera}_{frame:04d}_{name}.png', 'same_frame_fixed_roi', frame=frame, roi=name, box=box)
            if name in ('face_reference', 'background_books_reference'):
                roi_overview.append(current_roi)
        # HR overlay is a coordinate aid, never used in prediction or metrics.
        overlay = panels[0][1].copy(); draw = ImageDraw.Draw(overlay)
        for index, (name, box) in enumerate(boxes.items()):
            draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=COLORS[index % len(COLORS)], width=3)
            draw.text((box[0], max(0, box[1] - 22)), name, fill=COLORS[index % len(COLORS)], font=font(18))
        save(overlay, directory / 'HR_fixed_roi_coordinates.png', 'reference_coordinate_overlay', frame=frame)
        result['frames_evidence'].append(frame_record)
    save(stack(full_sheets), args.out / 'full_contact_sheet.png', 'all_fixed_stills_full_sheet')
    if roi_overview:
        save(stack(roi_overview), args.out / 'face_text_candidate_contact_sheet.png', 'fixed_face_text_candidate_overview')

    consecutive = dict(status='pending', frames=args.consecutive_frames, missing=[], frames_evidence=[],
        note='Requires actual cached NPZ for every listed method and every consecutive frame; no interpolation.')
    consecutive_sheets = []
    for frame in args.consecutive_frames:
        panels, record = gather(frame, record_frames=False)
        for source in sources:
            if record['methods'][source['name']]['status'] != 'ready':
                consecutive['missing'].append(dict(method=source['name'], frame=frame, evidence=record['methods'][source['name']]))
        consecutive['frames_evidence'].append(record)
        consecutive_sheets.append(sheet(panels, f'{args.camera} frame {frame:03d} | actual consecutive cache', panel_width=448))
    if not consecutive['missing'] and all(b == a + 1 for a, b in zip(args.consecutive_frames, args.consecutive_frames[1:])):
        save(stack(consecutive_sheets), args.out / 'consecutive_contact_sheet.png', 'actual_consecutive_cached_frames')
        consecutive.update(status='available_for_review', human_accepted=False)
    result['consecutive_review'] = consecutive
    missing_models = sorted(set(args.required_models) - {s['model'] for s in sources if s['status'] == 'ready'})
    result['missing_required_models'] = missing_models
    states = [v['status'] for r in result['frames_evidence'] for v in r['methods'].values()]
    result['still_comparison_status'] = 'failed' if 'failed' in states else 'complete' if not missing_models and states and all(s == 'ready' for s in states) else 'incomplete'
    result['status'] = ('failed' if result['still_comparison_status'] == 'failed' else 'complete'
        if result['still_comparison_status'] == 'complete' and consecutive['status'] == 'available_for_review' else 'incomplete')
    result['still_evidence_status'] = 'available_for_review'
    # This is evidence completeness only. Blind acceptance remains false even with all producers.
    write(args.out / 'comparison.json', result)
    lines = ['# 同帧渲染比较（CPU 读取已有预测）', '',
        f'场景：{manifest["scene"]}。当前状态：`{result["status"]}`。缺少模型：{missing_models}。', '',
        '默认帧 0、150、299 是三个离散时间点，不能用于连续时序判断。',
        f'连续帧缓存状态：`{consecutive["status"]}`；不插帧，不补造结果。', '',
        'HR/LR 由同一 manifest 核验 SHA256。所有预测由实际 NPZ 的 rgb_raw 统一裁剪到 [0,1] 后转 RGB；未使用缓存 PNG、锐化或曝光调整。', '',
        '固定裁剪沿用旧 HR 第40帧登记坐标。名称描述原参考内容，不保证当前帧仍有完整人脸或可辨文字。',
        '图片带方法标签，只是核对材料；尚未完成随机顺序、隐藏方法的人类盲评，accepted=false。', '',
        'Wu 图采用记录中的 posthoc cull，Multi4D 图采用 native branches。两者的完整系统差异不能直接归因于模型容量。', '',
        '来源、checkpoint 实际哈希、manifest、评价 GPU、cull、每帧原 NPZ 哈希、坐标和内容观察见 comparison.json。', '',
        '![整图](full_contact_sheet.png)', '', '![固定人脸和文字候选位置](face_text_candidate_contact_sheet.png)', '']
    (args.out / 'README.md').write_text('\n'.join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--methods', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New directory; never overwrite prior review')
    parser.add_argument('--camera', default='cam00')
    parser.add_argument('--frames', type=int, nargs='+', default=list(FIXED_FRAMES))
    parser.add_argument('--consecutive-frames', type=int, nargs='+', default=list(CONSECUTIVE_FRAMES))
    parser.add_argument('--roi-protocol', type=Path)
    parser.add_argument('--roi-observations', type=Path, help='Optional prior HR-only content observations, camera/frame/ROI keyed JSON')
    parser.add_argument('--required-models', nargs='+', default=['Wu', 'Multi4D'])
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('Select a fresh review directory: ' + str(args.out))
    if not args.frames or len(set(args.frames)) != len(args.frames) or args.frames != sorted(args.frames):
        parser.error('Frames must be distinct and sorted')
    if len(args.consecutive_frames) < 2 or any(b != a + 1 for a, b in zip(args.consecutive_frames, args.consecutive_frames[1:])):
        parser.error('Consecutive frame list must contain at least two truly consecutive frames')
    result = run(args)
    print(json.dumps(dict(status=result['status'], output=str(args.out.resolve()),
        missing_models=result['missing_required_models'], consecutive=result['consecutive_review']['status'],
        human_blind_accepted=False), ensure_ascii=False))
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
