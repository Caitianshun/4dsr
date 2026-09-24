"""Package small local evidence only; checkpoints/full PNG sequences stay indexed."""
import argparse
from pathlib import Path
import json
import zipfile
from summarize import read,write,sha
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_geometry_residual_20260924'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    files=[]
    for scene in ['cook','discussion']:
        assert read(OUT/f'{scene}_finalization_v1/complete.json')['status']=='completed_unified_summary_views_storage'
        files += [OUT/f'{scene}_teacher_fit_v1/teacher_fit.json', OUT/f'{scene}_methods.json',OUT/f'{scene}_summary_v1/summary.md',OUT/f'{scene}_summary_v1/summary.json']
        files += list((OUT/f'{scene}_summary_v1').glob('*.csv'))
        files += [OUT/f'{scene}_views_v1/README.md',OUT/f'{scene}_views_v1/views.json',OUT/f'{scene}_inference_storage_v1/storage.json',OUT/f'{scene}_finalization_v1/checkpoint_index.json']
        files += list((OUT/f'{scene}_views_v1').glob('*_frame*/*reference.png'))
        mapping=read(OUT/f'{scene}_methods.json')['methods']
        for name in ['G','L']:
            folder=Path(mapping[name]['train_dir'])
            files += [folder/'config.json',folder/'complete.json',folder/'training.jsonl',folder/'residual_statistics_1200.json',folder/'residual_statistics_6000.json']
    files += [ROOT/'docs/dynamic_sr_geometry_residual_2026-09-24.md',ROOT/'docs/dynamic_sr_geometry_residual_2026-09-24.codex.md',OUT/'user_plan.md',OUT/'implementation_freeze_v1.json',OUT/'checks_cook_v3/checks.json',OUT/'cts_deployment_v1/checks_discussion.json']
    files += list((ROOT/'docs/assets/codex_math/dynamic_sr_geometry_residual_2026-09-24').glob('*'))
    files += list(Path(__file__).parent.glob('*.py'))
    for name in ['decision.json','checkpoint_index.json','protocol_integrity.json','publication_receipt.json']:
        if (OUT/name).is_file():files.append(OUT/name)
    files=sorted(set(files));rows=[]
    with zipfile.ZipFile(a.out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in files:
            assert file.is_file() and file.suffix not in ['.pt','.mp4']
            relative=str(file.relative_to(ROOT));z.write(file,relative);rows.append(dict(path=relative,sha256=sha(file),bytes=file.stat().st_size))
        z.writestr('EVIDENCE_INDEX.json',json.dumps(dict(files=rows,scope='All registered native ROI panels, metrics, configurations, statistics and source. Full sequences, videos, tensors and model checkpoints remain on the local machine with hashed indices.'),indent=2))
        z.writestr('START_HERE.md','# G/L geometry residual evidence\n\nRead docs/dynamic_sr_geometry_residual_2026-09-24.md first. Per-scene summary.md files contain all U/W/B4/G/L metrics. Every registered native ROI panel is included; full-resolution sequences, videos and checkpoints remain at indexed local paths.\n')
    with zipfile.ZipFile(a.out) as z:assert z.testzip() is None
    write(OUT/'evidence_package.json',dict(status='completed',path=str(a.out.resolve()),sha256=sha(a.out),bytes=a.out.stat().st_size,files=len(files)))

if __name__=='__main__':main()
