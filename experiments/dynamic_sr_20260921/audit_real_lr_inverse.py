"""Independent CPU checks for the completed local inverse diagnostic.

Reads frozen source snapshots and numerical records; changes no experiment.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/dynamic_sr_20260921'
OUT = BASE / 'real_lr_inverse_independent_review'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source(path):
    spec = importlib.util.spec_from_file_location('frozen_local_inverse', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    torch.set_num_threads(1)
    m.cv2.setNumThreads(1)
    rng = np.random.default_rng(9921)
    d = m.aa_matrix()
    h, w = 60, 80
    whole = rng.uniform(0, 1, (h*4, w*4, 3))
    yy, xx = np.mgrid[:h, :w]
    backward = np.stack([.37+.002*(xx-w/2), -.21+.003*(yy-h/2)], axis=-1).astype(np.float32)
    center = (40, 30)
    origin = (center[0]*4-48, center[1]*4-48)
    a, ys, xs, _, _ = m.make_operator(d, backward, np.ones((h,w), bool), origin, center, (h,w))
    flow = m.F.interpolate(torch.from_numpy(backward.astype(np.float64).transpose(2,0,1))[None],
        size=(h*4,w*4), mode='bilinear', align_corners=False)[0].permute(1,2,0).numpy()*4
    Y, X = np.mgrid[:h*4,:w*4]
    sample = np.stack([X,Y], axis=-1)+flow
    grid = 2*(sample+.5)/np.array([w*4,h*4])-1
    warped = m.F.grid_sample(m.ten(whole), torch.from_numpy(grid)[None], mode='bilinear',
        padding_mode='zeros', align_corners=False)
    global_lr = m.F.interpolate(warped, size=(h,w), mode='bicubic', align_corners=False,
        antialias=True)[0].permute(1,2,0).numpy()
    patch = whole[origin[1]:origin[1]+m.N, origin[0]:origin[0]+m.N]
    forward_error = float(np.max(np.abs(a@patch.reshape(-1,3)-global_lr[ys,xs])))
    x = rng.normal(size=a.shape[1])
    y = rng.normal(size=a.shape[0])
    left, right = np.dot(a@x,y), np.dot(x,a.T@y)
    adjoint_error = float(abs(left-right)/max(abs(left),abs(right),1))
    l, reg = m.reg_matrix()
    vec = rng.normal(size=m.N*m.N)
    energy, direct_energy = float(vec@(reg@vec)), float(np.mean((l@vec)**2))
    regularizer_error = abs(energy-direct_energy)/direct_energy
    assert forward_error < 1e-12 and adjoint_error < 1e-12 and regularizer_error < 1e-12
    strict_check = None
    if 'strict_footprint' in m.make_operator.__code__.co_varnames:
        mask = np.ones((h,w), bool)
        mask[28:31, 37:40] = False
        ac, yc, xc, _, _ = m.make_operator(d, backward, mask, origin, center, (h,w), False)
        az, yz, xz, _, _ = m.make_operator(d, backward, mask, origin, center, (h,w), True)
        mapping = {(int(y),int(x)):i for i,(y,x) in enumerate(zip(yc,xc))}
        assert all((int(y),int(x)) in mapping for y,x in zip(yz,xz))
        ids = [mapping[int(y),int(x)] for y,x in zip(yz,xz)]
        difference = (az-ac[ids]).tocsr()
        coefficient_difference = float(np.max(abs(difference.data))) if difference.nnz else 0.
        assert az.shape[0] < ac.shape[0] and coefficient_difference == 0
        strict_check = dict(center_only_rows=ac.shape[0], full_footprint_rows=az.shape[0],
            is_subset=True, retained_operator_coefficient_max_difference=coefficient_difference)
    return dict(source=str(path), source_sha256=sha(path),
        global_nonconstant_backward_warp_then_AA_vs_local_sparse_max_abs=forward_error,
        retained_rows=a.shape[0], adjoint_relative_error=adjoint_error,
        regularizer_energy_relative_error=regularizer_error,
        row_sum_max_abs_error=float(np.max(abs(np.asarray(a.sum(axis=1)).ravel()-1))),
        strict_mask_hole_control=strict_check)


def main():
    assert not torch.cuda.is_available()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ['real_lr_local_inverse_v1','real_lr_local_inverse_strict_footprint_v1']:
        folder = BASE/name
        result = json.loads((folder/'metrics.json').read_text())
        assert result['identity']['source_sha256'] == sha(folder/'source.py')
        before = json.loads((folder/'all_predictions_frozen_before_hr.json').read_text())
        hashes = [sha(folder/f'patch_{i:02d}_solved_before_hr.npz') for i in range(len(result['records']))]
        assert hashes == before['predictions']
        assert sha(folder/'protocol_before_hr.json') == before['protocol_sha256']
        records = result['records']
        delta = {}
        for scene, summary in result['summary'].items():
            delta[scene] = {}
            for lam in ['0.01','0.1','1']:
                multi, single = summary['multiframe_lambda_'+lam], summary['reference_lambda_'+lam]
                delta[scene][lam] = {k:multi[k]-single[k] for k in ['psnr','ssim','high_residual_mse']}
        results[name] = dict(source_checks=check_source(folder/'source.py'),
            prediction_hashes_and_protocol_match_pre_hr_receipt=True,
            max_repeat_solution_difference=max(max(r['repeat_solution_max_abs'].values()) for r in records),
            max_cg_relative_normal_equation_residual=max(z['relative_normal_equation_residual']
                for r in records for lam in r['solver'].values() for channels in lam.values() for z in channels),
            neighbor_valid_rows_min=min(o['accepted_rows'] for r in records for o in r['operators'][1:]),
            neighbor_valid_rows_max=max(o['accepted_rows'] for r in records for o in r['operators'][1:]),
            delta_multiframe_minus_single=delta)
    a, b = [json.loads((BASE/n/'protocol_before_hr.json').read_text()) for n in results]
    assert a['chosen'] == b['chosen'] and a['lambdas'] == b['lambdas'] and a['ridge'] == b['ridge']
    output = dict(cpu_only=True, audit_source_sha256=sha(__file__),
        same_selected_cells_and_regularization_across_versions=True, results=results)
    (OUT/'audit.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
