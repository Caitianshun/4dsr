#!/usr/bin/env python3
"""Diagnostic-only relation of LR-derived demand to genuine HR errors.

Nothing written here is a training map. Full-grid Pearson and weighted error
coverage accompany fixed every-eight-pixel Spearman (cost-bounded statistic).
"""
from spectral_common import *
from audit_spectrum_color import dump_csv, train_observations, OBS
from scipy.stats import spearmanr

def main():
    torch.set_num_threads(4)
    idxpath = ROOT/'output/dynamic_sr_prior_diagnosis_20260929/diagnostic_maps/prior_index.json'
    idx=read(idxpath); maps={(e['camera'],e['frame']):e for e in idx['entries']}
    _, obs, dr=train_observations(); rows=[]; curves=[]
    teachers={ (e['camera'],e['frame']):e for e in read(OUT/'legal_anchored_targets/index.json')['entries'] }
    for o in obs:
        c,f=o['camera_id'],o['frame_index']; e=maps[c,f]; mp=ROOT/e['path']; assert sha(mp)==e['sha256']
        m=np.load(mp); weight=np.repeat(np.repeat(m,4,axis=0),4,axis=1).astype(np.float64)
        g=rgb(dr/o['hr_path']); source={'SwinIR':rgb(ROOT/teachers[c,f]['teacher_path'])}
        for model in ['U6000','r1_J1','r1_Async2']:
            source[model]=np.clip(np.load(OBS/model/f'{c}_{f:04d}.npz')['rgb_raw'].transpose(1,2,0),0,1)
        for label,p in source.items():
            err=np.square(p.astype(np.float64)-g.astype(np.float64)).mean(axis=2)
            corr=float(np.corrcoef(weight.ravel(),err.ravel())[0,1])
            rho=float(spearmanr(weight[::8,::8].ravel(),err[::8,::8].ravel()).statistic)
            rows.append(dict(camera=c,frame=f,model=label,mean_weight=float(weight.mean()),
                             effective_pixel_fraction=float(weight.sum()**2/(weight.size*np.square(weight).sum())),
                             mse=float(err.mean()), weighted_mse_full_denominator=float((weight*err).mean()),
                             weighted_error_share=float((weight*err).sum()/err.sum()),pearson_weight_error=corr,
                             spearman_weight_error_fixed_stride8=rho))
            for threshold in [0.0,0.1,0.25,0.5,0.75,0.9]:
                mask=weight>=threshold
                curves.append(dict(camera=c,frame=f,model=label,weight_threshold=threshold,
                                   area_fraction=float(mask.mean()),error_fraction=float(err[mask].sum()/err.sum())))
        print(f'coverage {c}/{f}',flush=True)
    dump_csv(OUT/'demand_error_coverage.csv',rows);dump_csv(OUT/'demand_coverage_curve.csv',curves)
    agg=[]
    for label in ['SwinIR','U6000','r1_J1','r1_Async2']:
        rr=[r for r in rows if r['model']==label]
        agg.append(dict(model=label,observations=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ['model','camera','frame']}))
    write(OUT/'demand_coverage_summary.json',dict(status='completed',aggregate=agg,map_index_sha256=sha(idxpath),
          input_boundary='Maps frozen from legitimate LR/calibration/model; HR errors read only here. No HR-derived weights or thresholds returned to training.',
          interpretation='Demand measures missing observations, not teacher correctness. Correlation or weighted error share is diagnostic overlap, not reliability proof.',
          upsample='nearest exact repeat4 to HR, as frozen original weighting',
          spearman='fixed coordinate stride8 in each axis; Pearson and coverage use every HR pixel'))

if __name__=='__main__':main()
