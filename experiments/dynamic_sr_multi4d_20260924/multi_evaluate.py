"""Use the unchanged fixed metric loop, with an explicit Multi4D loader/camera.

Only model identity/loading, teacher inventory and render-camera construction
are adapted. All spatial/temporal/LR/ROI aggregation calls remain the existing
unified evaluator. No evaluator inputs are passed to training.
"""
import os
from pathlib import Path
import sys
import types
import torch
from multi_data import camera

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ORIGINAL=ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py'


def main():
    source=ORIGINAL.read_text()
    old="    parser.add_argument('--manifest', type=Path, required=True)"
    assert source.count(old)==1
    source=source.replace(old,old+"\n    parser.add_argument('--teacher-index', type=Path, required=True)")
    source=source.replace('from detail_loss import highpass, OPERATOR','from metric_operator import highpass, OPERATOR')
    source=source.replace("teacher_config = json.loads(config_path.read_text())", "teacher_config = json.loads(args.teacher_index.read_text())\n        teacher_config['teacher_inputs'] = teacher_config['entries']")
    source=source.replace("('joint', 'ordinary_split', 'bound_split')", "('joint', 'ordinary_split', 'bound_split', 'multi4d')")
    source=source.replace("'gaussian_count': len(model.g._xyz)","'gaussian_count': sum(len(g._xyz) for g in model.branches)")
    source=source.replace("'gpu': torch.cuda.get_device_name()", "'adapter_sha256': legacy.file_sha(adapter_path), 'time_parameterization': 'original_frame_index / 120', 'gpu': torch.cuda.get_device_name()")
    source=source.replace('                h_hr = detail_metrics(h_gt)', '                rgb_teacher = None\n                h_hr = detail_metrics(h_gt)')
    source=source.replace('                    h_teacher = detail_metrics(highpass(teacher_tensor, lr_size))',
        '                    h_teacher = detail_metrics(highpass(teacher_tensor, lr_size))\n                    delta_teacher = (raw-teacher_tensor).double()\n                    rgb_teacher = dict(l1=float(delta_teacher.abs().mean()),mse=float(delta_teacher.square().mean()))')
    source=source.replace("'h_hr': h_hr, 'h_teacher': h_teacher or {}", "'rgb_teacher': rgb_teacher, 'raster_visible_points': {k:int(v.sum()) for k,v in model.last_visibility.items()}, 'h_hr': h_hr, 'h_teacher': h_teacher or {}")
    module=types.ModuleType('multi4d_fixed_metric_adapter');module.__file__=str(ORIGINAL)
    exec(compile(source,str(ORIGINAL),'exec'),module.__dict__)
    module.adapter_path=__file__
    module.MOTION=HERE/'multi_model.py'
    module.render_camera=lambda m,o,uid:camera(m,o,torch.zeros((3,m['resolutions']['lr'][1],m['resolutions']['lr'][0])))
    module.main()


if __name__=='__main__':main()
