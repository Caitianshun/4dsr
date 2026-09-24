"""CPU-only three-metric comparison from completed, immutable evaluations."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / 'output/dynamic_sr_20260918'
SCENES = ['cook_spinach', 'cut_roasted_beef']
METHODS = {
    'lr_native': '原生 LR → 直接 HR 渲染',
    'postrender_bicubic': 'LR 渲染 → bicubic',
    'postrender_swinir': 'LR 渲染 → SwinIR',
    'lr_integrated': '采样对齐（LR-integrated）',
    'joint': '联合 SR（Joint）',
    'appearance': 'SR 仅更新外观（Appearance）',
    'frozen': '冻结几何与变形（Frozen）',
    'hr_reference': '额外 HR 监督诊断',
    'input_bicubic': '真实测试 LR → bicubic（输入诊断）',
    'input_swinir': '真实测试 LR → SwinIR（输入诊断）',
}
DIAGNOSTICS = {'hr_reference', 'input_bicubic', 'input_swinir'}


def main():
    out = ROOT / 'assessment_pilot_v1'
    result = {'scope': 'cam00; x4; 60-frame means; one seed; HR reference is privileged diagnostic',
              'metric_directions': {'psnr': 'higher', 'ssim': 'higher', 'lpips': 'lower'},
              'variation_lpips_definition': 'official full-image spatial LPIPS map averaged inside a fixed temporal-variation mask; not standard scalar LPIPS or true-motion mask',
              'scenes': {}, 'sources': []}
    for scene in SCENES:
        result['scenes'][scene] = {}
        cache_keys = set()
        for method in METHODS:
            postrender = method.startswith('postrender_')
            input_diagnostic = method.startswith('input_')
            path = ROOT / (f'{scene}_pilot_v1_sr_reference_diagnostic/metrics.json' if input_diagnostic else
                           f'{scene}_pilot_v1_postrender_controls/metrics.json' if postrender
                           else f'{scene}_pilot_v1_{method}/evaluation/metrics.json')
            raw = path.read_bytes()
            source = json.loads(raw)
            cache_keys.add(source['cache_key'])
            data = source['modes'][method.removeprefix('input_')] if input_diagnostic else source['modes'][method.removeprefix('postrender_')] if postrender else source
            assert [r['frame_index'] for r in data['rows']] == list(range(0, 120, 2))
            result['sources'].append({'scene': scene, 'method': method,
                                      'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
            row = {}
            for group in ['full', 'dynamic']:
                fields = data['aggregate'][group]
                row[group] = {'psnr': fields['psnr_mean'], 'ssim': fields['ssim_mean'],
                              'lpips': fields['lpips_alex_mean' if group == 'full' else 'lpips_alex_spatial_mask_mean']}
            result['scenes'][scene][method] = row
        assert len(cache_keys) == 1, 'Do not compare inconsistent evaluation masks/flows'
    texts = []
    for group, title in [('full', '全图质量'), ('dynamic', '时间变化区域质量')]:
        texts += [f'**{title}**', '',
            '| 方法 | 炒菠菜 PSNR ↑ | SSIM ↑ | LPIPS ↓ | 切烤牛肉 PSNR ↑ | SSIM ↑ | LPIPS ↓ |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for method, label in METHODS.items():
            values = []
            for scene in SCENES:
                for metric in ['psnr', 'ssim', 'lpips']:
                    value = result['scenes'][scene][method][group][metric]
                    candidates = [result['scenes'][scene][m][group][metric] for m in METHODS if m not in DIAGNOSTICS]
                    best = min(candidates) if metric == 'lpips' else max(candidates)
                    display = f'{value:.3f}' if metric == 'psnr' else f'{value:.4f}'
                    values.append(f'**{display}**' if method not in DIAGNOSTICS and value == best else display)
            texts.append('| ' + ' | '.join([label] + values) + ' |')
        texts += ['', '粗体只比较前七种新视角重建方案。后三行为额外信息诊断：HR reference 使用训练 HR；真实测试 LR 诊断直接拥有目标视角图像，不参与训练，也不是可与新视角合成公平排名的方法或严格数学上界。' if group == 'full' else
                  '此表 LPIPS 是固定区域的 LPIPS spatial-map 均值，不与全图标准 LPIPS 混称；区域来自时间变化，不是语义或运动真值掩码。', '']
    (out / 'quality_comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    (out / 'quality_table.md').write_text('\n'.join(texts))
    print('\n'.join(texts))


if __name__ == '__main__':
    main()
