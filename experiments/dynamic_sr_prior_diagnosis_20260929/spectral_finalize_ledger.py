#!/usr/bin/env python3
"""Preserve per-observation device attribution and honest resumed cost scope."""
from datetime import datetime, timezone
from spectral_common import *
from audit_spectrum_color import OBS

def main():
    repart=read(OUT/'diagnostic_repartition.json'); cutoff=datetime.fromisoformat(repart['utc']).timestamp()
    g1='GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec';g0='GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'
    rows=[];times=[]
    for model in ['LR6k','HR6k','U6000','r1_J1','r1_Async2']:
        assert read(OUT/model/'complete.json')['observations']==196
        for f in sorted((OUT/model).glob('cam*.json')):
            d=read(f);b=d['budget'];source=read(OBS/model/f.name)
            assert b['float_asset_sha256']==source['npz_sha256']
            assert b['hr_sha256']==source['hr_sha256']
            tm=f.stat().st_mtime;times.append(tm)
            device=g0 if model=='r1_J1' and tm>cutoff else g1
            rows.append(dict(model=model,camera=b['camera'],frame=b['frame'],metric_gpu=device,
                        scalar_metrics_observations=6 if b['split']=='train' else 5,
                        result_path=str(f.relative_to(ROOT)),result_sha256=sha(f),
                        original_completion_utc=datetime.fromtimestamp(tm,timezone.utc).isoformat()))
        cp=OUT/model/'complete.json';d=read(cp)
        d['seconds_scope']='Seconds in this process invocation only, including reused row reads. Do not sum as full diagnostic cost after resume; see diagnostic_hardware_ledger.json.'
        d['metric_device_scope']='This invocation device; cached row device is attributed in diagnostic_hardware_ledger.json.'
        write(cp,d)
    ledger=dict(status='completed',rows=rows,observations=len(rows),scalar_quality_pairs=sum(r['scalar_metrics_observations'] for r in rows),
        new_3D_training_updates=0,new_3D_render_forwards=0,
        hardware={g1:'NVIDIA GeForce RTX 3090',g0:'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'},
        hardware_scope='Different local GPU architectures, same project PyTorch/LPIPS runtime. Only r1_J1 late rows moved after allocation; metric parity differences recorded, formal training endpoints evaluated on GPU1.',
        first_observation_completion_utc=datetime.fromtimestamp(min(times),timezone.utc).isoformat(),
        last_observation_completion_utc=datetime.fromtimestamp(max(times),timezone.utc).isoformat(),
        completion_span_seconds=max(times)-min(times),
        completion_span_scope='Observed wall span between first and last row completion. Excludes initial setup before first row, not a measurement of GPU active time.',
        restart='Own image-diagnostic processes only; completed atomic rows reused, no training interrupted. At most an uncommitted current observation was recomputed.',
        repartition_sha256=sha(OUT/'diagnostic_repartition.json'))
    write(OUT/'diagnostic_hardware_ledger.json',ledger)

if __name__=='__main__':main()
