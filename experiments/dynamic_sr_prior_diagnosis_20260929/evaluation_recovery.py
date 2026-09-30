"""Reuse only complete, identity-matched local camera evaluation subtasks."""
import shutil
from dv_common import *


def recover_camera(canonical, out, camera, split, label, checkpoint_hash, protocol):
    for attempt in sorted(canonical.glob('attempt_*'), reverse=True):
        source=attempt/camera
        if source==out/camera or not (source/'complete.json').is_file():continue
        receipt=read(source/'complete.json');metrics=read(source/'metrics.json')
        checks=[
            receipt['status']=='completed_evaluation',receipt['checkpoint_sha256']==checkpoint_hash,
            receipt['metrics_sha256']==sha(source/'metrics.json'),receipt['observations']==60,
            receipt['split']==split,receipt['method']==label,receipt['parameter_updates']==0,
            metrics['checkpoint_sha256']==checkpoint_hash,metrics['manifest_sha256']==protocol['manifest']['sha256'],
            metrics['roi_protocol_sha256']==protocol['roi']['sha256'],metrics['split']==split,
            metrics['adapter_sha256']==sha(HERE/'evaluate_camera.py'),
            metrics['script_sha256']==sha(ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py'),
            metrics['metric_helpers_sha256']==sha(ROOT/'experiments/dynamic_sr_20260918/evaluate.py'),
            metrics['detail_operator_sha256']==sha(ROOT/'experiments/dynamic_sr_detail_supervision_20260924/detail_loss.py'),
            metrics['visible_cuda']==protocol['evaluation']['physical_gpu'],metrics['postprocess'] is None,
            [r['frame_index'] for r in metrics['rows']]==list(range(0,120,2)),
            all(r['camera_id']==camera for r in metrics['rows']),
            metrics['checkpoint_metadata']['protocol_sha256']==sha(OUT/'protocol.json'),
        ]
        if not all(checks):raise RuntimeError(('Completed evaluation identity mismatch; inspect before reuse',source,checks))
        files={}
        for frame in range(0,120,2):
            f=source/'predictions'/camera/f'{frame:04d}.png'
            files[str(f.relative_to(source))]=sha(f)
        shutil.copytree(source,out/camera)
        return dict(source=str(source.relative_to(ROOT)),receipt=entry(source/'complete.json'),
            metrics=entry(source/'metrics.json'),display_prediction_sha256=files,
            reused_rgb_forwards=60,elapsed_seconds=metrics['elapsed_seconds'])
    return None
