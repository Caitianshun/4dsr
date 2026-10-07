"""Exact exposure-preserving same-time triplets from registered old anchors.

Only camera poses and frozen train-only parent moment caches enter assignment.
No LR/HR/teacher pixel is opened. 3000-block failure triggers a registered 6000
assignment for BOTH suffixes, never replacement or changing the old anchor.
"""
from __future__ import annotations
import argparse
from collections import Counter
import itertools
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from fp_common import ROOT, OUT, read, write, sha, entry, local

OLD = ROOT / 'output/dynamic_sr_confidence_geometry_20261006'
MANIFEST = ROOT / 'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'
PARENT = OLD / 'cache/parent'
CAL_FRAMES = (0, 40, 80, 118)
POLICY = dict(
    version=1, coarse_stride=8, alpha_positive_threshold=1e-6,
    coarse_overlap='fraction of finite positive parent rays projected in-bounds into finite positive parent rays with depth agreement',
    overlap_depth_tolerance='max(0.05*projected_z,3*sqrt(target_var+source_var))',
    symmetric_overlap='mean of both directed fractions; heuristic frozen model overlap, not true surface correspondence',
    pair_cost='1-symmetric_overlap + 0.20*baseline_distance/median_nonzero_baseline + 0.10*viewpoint_angle/90',
    assignment='minimum-cost b then constrained c; up to 32 deterministic bounded-jitter b retries, c excludes a/b camera',
    retry_jitter=0.025, attempt_limit=32, primary_block_size=3000,
    fallback_block_size=6000, fallback_applies_to_both_suffixes=True,
    calibration_selection='minimum triangular pair-cost same-time distinct-camera triplet with fixed anchor; camera-id tie break',
    sources='camera pose and frozen U6000 coarse moments only', hr_pixels_read=False,
    lr_pixels_read=False, teacher_pixels_read=False, gradient_or_metric_used=False)


def _safe_counter(values):
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def _stats(v):
    a = np.asarray(v, dtype=np.float64)
    return dict(count=int(a.size), mean=float(a.mean()), minimum=float(a.min()),
                q25=float(np.quantile(a, .25)), median=float(np.median(a)),
                q75=float(np.quantile(a, .75)), maximum=float(a.max())) if a.size else dict(count=0)


def _coarse_parent(path, cal):
    with np.load(path, allow_pickle=False) as z:
        # Sampling frozen LR moments, rather than interpolating depth, is used
        # solely for assignment cost. Formal X uses separately exported HR moments.
        depth = z['depth_lr'][4::8, 4::8].astype(np.float64)
        alpha = z['alpha_lr'][4::8, 4::8].astype(np.float64)
        variance = z['raw_variance_lr'][4::8, 4::8].astype(np.float64)
        lr_shape = z['depth_lr'].shape
    yy, xx = np.indices(depth.shape)
    xy = np.stack((xx*8+4, yy*8+4, np.ones(depth.shape)), -1)
    rays = xy @ np.linalg.inv(np.asarray(cal['K_lr'], np.float64)).T
    xyz = rays * np.nan_to_num(depth, nan=0)[..., None]
    world = np.concatenate((xyz, np.ones((*depth.shape, 1))), -1) @ np.asarray(cal['c2w'], np.float64).T
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(alpha) & (alpha > 1e-6)
    return dict(depth=depth, alpha=alpha, variance=variance, world=world, valid=valid, lr_shape=lr_shape)


def _directed_overlap(target, source, source_cal):
    world = target['world']; cp = world @ np.asarray(source_cal['w2c'], np.float64).T
    uvh = cp[..., :3] @ np.asarray(source_cal['K_lr'], np.float64).T
    z = cp[..., 2]
    xy = uvh[..., :2] / np.where(np.isfinite(z) & (z > 1e-9), z, 1)[..., None]
    h, w = source['lr_shape']
    finite = np.isfinite(xy).all(-1) & np.isfinite(z) & (z > 0)
    inside = finite & (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    # Nearest coarse cell suffices for this registered coarse cost only.
    ix = np.rint((np.nan_to_num(xy[..., 0], nan=-1000, posinf=-1000, neginf=-1000)-4)/8).astype(np.int64)
    iy = np.rint((np.nan_to_num(xy[..., 1], nan=-1000, posinf=-1000, neginf=-1000)-4)/8).astype(np.int64)
    ix = ix.clip(0, source['depth'].shape[1]-1); iy = iy.clip(0, source['depth'].shape[0]-1)
    sd, sv = source['depth'][iy, ix], source['variance'][iy, ix]
    tv = target['variance']
    # Unknown variance gets only the relative tolerance, without invalidating a ray.
    variance = np.where(np.isfinite(sv) & (sv >= 0), sv, 0) + np.where(np.isfinite(tv) & (tv >= 0), tv, 0)
    tolerance = np.maximum(.05*np.maximum(z, 0), 3*np.sqrt(variance))
    agreement = np.isfinite(sd) & (np.abs(sd-z) <= tolerance)
    support = target['valid'] & inside & source['valid'][iy, ix] & agreement
    den = int(target['valid'].sum())
    return float(support.sum()/den) if den else 0.0


def prepare_pair_costs(out=OUT/'schedules', manifest=MANIFEST, parent=PARENT):
    out, manifest, parent = Path(out), local(manifest), local(parent)
    out.mkdir(parents=True, exist_ok=True)
    index = read(parent/'index.json'); m = read(manifest)
    if index.get('status') != 'completed' or index['manifest_sha256'] != sha(manifest) or index.get('privileged_train_hr'):
        raise ValueError('Registered completed train-only frozen parent identity required')
    cameras = tuple(m['splits']['train']); frames = tuple(m['frame_indices'])
    if cameras != tuple(f'cam{i:02d}' for i in range(2, 21)) or frames != tuple(range(0,120,2)):
        raise ValueError('Unexpected legal observation boundary')
    registry = {(r['camera'],int(r['frame'])):r for r in index['entries']}
    if set(registry) != set(itertools.product(cameras,frames)):
        raise ValueError('Frozen parent coverage mismatch')
    costs_path = out/'pair_costs.json'
    identity = dict(manifest=entry(manifest), parent_index=entry(parent/'index.json'),
                    parent_sha256=index['parent_sha256'], policy=POLICY, source_sha256=sha(__file__))
    if costs_path.exists():
        saved = read(costs_path)
        if saved['identity'] != identity:
            raise ValueError('Pair cost identity changed; use an independent output directory')
        for r in saved['parent_files']:
            if sha(local(r['path'])) != r['sha256']:
                raise ValueError('Parent byte identity changed')
        return saved
    t0=time.monotonic(); cals=m['cameras']; centres={c:np.asarray(cals[c]['c2w'],float)[:3,3] for c in cameras}
    distances=np.asarray([[np.linalg.norm(centres[a]-centres[b]) for b in cameras] for a in cameras])
    norm=float(np.median(distances[distances>0])); frame_costs={}; files=[]
    for f in frames:
        parents={}
        for c in cameras:
            r=registry[c,f]; p=parent/r['path']
            if sha(p)!=r['sha256']: raise ValueError(f'Frozen parent corruption: {p}')
            files.append(entry(p)); parents[c]=_coarse_parent(p,cals[c])
        points=np.concatenate([v['world'][v['valid'],:3] for v in parents.values()],axis=0)
        scene_centre=np.median(points,axis=0)
        directions=np.asarray([scene_centre-centres[c] for c in cameras]); directions/=np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-12)
        angles=np.rad2deg(np.arccos(np.clip(directions@directions.T,-1,1)))
        directed=np.zeros((len(cameras),len(cameras)),np.float64)
        for i,a in enumerate(cameras):
            for j,b in enumerate(cameras):
                directed[i,j]=1.0 if i==j else _directed_overlap(parents[a],parents[b],cals[b])
        overlap=(directed+directed.T)/2
        cost=1-overlap+.20*distances/norm+.10*angles/90
        np.fill_diagonal(cost,1e6)
        frame_costs[str(f)]=dict(cost=cost.tolist(),symmetric_overlap=overlap.tolist(),
            directed_overlap=directed.tolist(),baseline_distance=distances.tolist(),
            viewpoint_angle_degrees=angles.tolist(),scene_centre=scene_centre.tolist())
    d=dict(status='completed_frozen_pair_costs',identity=identity,cameras=list(cameras),frames=list(frames),
           parent_files=files,baseline_normalizer=norm,by_frame=frame_costs,seconds=time.monotonic()-t0,
           RGB_forwards=0,moment_forwards=0,parameter_updates=0)
    write(costs_path,d);return d


def _assign_frame(indices, keys, frame, pair, seed):
    cams=pair['cameras']; ci={c:i for i,c in enumerate(cams)}
    ids=list(indices); camera_ids=np.asarray([ci[keys[i][0]] for i in ids],np.int64)
    n=len(ids); counts=Counter(camera_ids.tolist())
    if n<3 or max(counts.values())*3>n:
        raise ValueError(f'Frame {frame} has no three-column distinct-camera permutation: n={n}, counts={dict(counts)}')
    matrix=np.asarray(pair['by_frame'][str(frame)]['cost'],float)[camera_ids[:,None],camera_ids[None,:]]
    rng=np.random.default_rng(seed); attempts=[]
    for attempt in range(POLICY['attempt_limit']):
        # First try is the pure minimum; later deterministic finite bounded
        # perturbations provide joint-feasibility backtracking without replacement.
        adjusted=matrix.copy()
        if attempt: adjusted+=rng.uniform(0,POLICY['retry_jitter'],size=matrix.shape)
        adjusted[camera_ids[:,None]==camera_ids[None,:]]=1e9
        _,b=linear_sum_assignment(adjusted)
        forbidden=(camera_ids[:,None]==camera_ids[None,:]) | (camera_ids[b,None]==camera_ids[None,:])
        c_cost=matrix+np.asarray(pair['by_frame'][str(frame)]['cost'],float)[camera_ids[b,None],camera_ids[None,:]]
        c_cost[forbidden]=1e9
        _,c=linear_sum_assignment(c_cost)
        feasible=bool(np.all(~forbidden[np.arange(n),c]))
        attempts.append(dict(attempt=attempt+1,feasible=feasible))
        if feasible:
            assert len(set(b.tolist()))==n and len(set(c.tolist()))==n
            return [[ids[i],ids[int(b[i])],ids[int(c[i])]] for i in range(n)],dict(frame=frame,n=n,attempts=attempts)
    raise ValueError(f'Finite joint assignment backtracking exhausted for frame {frame}')


def _synchronous(old,pair,block_size):
    keys=old['record_keys']; anchors=[r[0] for r in old['rows']]; rows=[None]*len(anchors); registration=[]
    for start in range(0,len(anchors),block_size):
        positions={}
        for k in range(start,min(start+block_size,len(anchors))): positions.setdefault(keys[anchors[k]][1],[]).append(k)
        for f,locs in sorted(positions.items()):
            triplets,trace=_assign_frame([anchors[k] for k in locs],keys,f,pair,old['seed']+start+int(f)*7919)
            for k,r in zip(locs,triplets):rows[k]=r
            registration.append(dict(start=start,**trace))
    return rows,registration


def calibration_triplets(schedule):
    """Return 76 registered index triples; a is camera-major fixed train76."""
    return [list(r['indices']) for r in schedule['calibration_train76']]


def select_rows(schedule,method):
    if method in ('B0','E'):return schedule['random_rows']
    if method in ('Bsync','M','X','MX'):return schedule['rows']
    raise ValueError(f'Unknown registered arm {method}')


def validate(schedule):
    keys=schedule['record_keys'];rows=schedule['rows'];original=schedule['random_rows'];n=len(rows)
    assert n==6000 and len(original)==n
    assert [r[0] for r in rows]==[r[0] for r in original]
    for r in rows:
        assert len({keys[i][0] for i in r})==3
        assert len({keys[i][1] for i in r})==1
    for start in range(0,n,schedule['exposure_block_size']):
        counts=[Counter(r[j] for r in rows[start:start+schedule['exposure_block_size']]) for j in range(3)]
        assert counts[0]==counts[1]==counts[2]
    assert len(schedule['calibration_train76'])==76
    for r in schedule['calibration_train76']:
        a,b,c=r['indices'];assert len({keys[i][0] for i in (a,b,c)})==3
        assert len({keys[i][1] for i in (a,b,c)})==1
    return dict(passed=True,anchor_exactly_reused=True,random_three_columns_exactly_reused=True,
                same_time_distinct_cameras=True,exact_permutation_each_registered_block=True,
                exposure_equal_100_steps_claimed=False,
                balanced_3000_endpoint=schedule['exposure_block_size']==3000)


def _exposure(rows,keys,lr_coefficients):
    lr=np.zeros(len(keys),float);sr=np.zeros(len(keys),float)
    for row in rows:
        for j,w in enumerate(lr_coefficients):lr[row[j]]+=w
        sr[row[1]]+=.05;sr[row[2]]+=.05
    return [dict(camera=c,frame=f,LR_coefficient_exposure=float(lr[i]),SR_coefficient_exposure=float(sr[i])) for i,(c,f) in enumerate(keys)]


def _build(old,pair,rows,registration,block,source):
    keys=old['record_keys'];lookup={tuple(k):i for i,k in enumerate(keys)};cams=pair['cameras'];ci={c:i for i,c in enumerate(cams)}
    calibration=[]
    for a in cams:
        for f in CAL_FRAMES:
            candidates=[]
            cost=np.asarray(pair['by_frame'][str(f)]['cost'],float);ai=ci[a]
            for b,c in itertools.combinations([x for x in cams if x!=a],2):
                score=cost[ai,ci[b]]+cost[ai,ci[c]]+cost[ci[b],ci[c]]
                candidates.append((float(score),b,c))
            score,b,c=min(candidates)
            calibration.append(dict(anchor=[a,f],indices=[lookup[a,f],lookup[b,f],lookup[c,f]],joint_pair_cost=score))
    pairs=Counter();angles=[];overlaps=[];baselines=[]
    for row in rows:
        f=keys[row[0]][1];stats=pair['by_frame'][str(f)]
        for i,j in ((0,1),(0,2),(1,0),(1,2),(2,0),(2,1)):
            a,b=keys[row[i]][0],keys[row[j]][0];ai,bi=ci[a],ci[b]
            pairs[a+'->'+b]+=1;angles.append(stats['viewpoint_angle_degrees'][ai][bi]);overlaps.append(stats['symmetric_overlap'][ai][bi]);baselines.append(stats['baseline_distance'][ai][bi])
    d=dict(schema='dynamic_sr_multiview_footprint_schedule_v1',status='completed',seed=old['seed'],
        record_keys=keys,random_rows=old['rows'],rows=rows,exposure_block_size=block,
        old_schedule=entry(source),pair_costs=entry(OUT/'schedules/pair_costs.json'),policy=POLICY,
        assignment_trace=registration,calibration_train76=calibration,
        calibration_triplets=[r['indices'] for r in calibration],
        directed_camera_pair_frequency=dict(sorted(pairs.items())),
        registered_six_edges=[[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]],
        viewpoint_angle_degrees=_stats(angles),baseline_distance=_stats(baselines),frozen_symmetric_overlap=_stats(overlaps),
        weighted_exposure=dict(B0_E=_exposure(old['rows'],keys,(1,0,0)),Bsync_X=_exposure(rows,keys,(1,0,0)),M_MX=_exposure(rows,keys,(1/3,1/3,1/3))),
        definitions=dict(rows='synchronous a/b/c; select_rows returns original random rows for B0/E',
            calibration='train76 anchor-major; same triangular frozen pair cost as official assignment',
            overlap='coarse frozen-parent heuristic, not independently verified surface truth',
            exposure='exact per registered 3000 or 6000 block; old 100-step claim not applied'))
    d['audit']=validate(d)
    for arm,ex in d['weighted_exposure'].items():
        # Floating 1/3 summation round-off cannot alter intended coefficient exposure.
        expected=d['weighted_exposure']['B0_E']
        assert all(abs(r['LR_coefficient_exposure']-e['LR_coefficient_exposure'])<1e-10 and abs(r['SR_coefficient_exposure']-e['SR_coefficient_exposure'])<1e-10 for r,e in zip(ex,expected)),arm
    return d


def generate_all(out=OUT/'schedules'):
    out=Path(out)
    # Stable API currently binds costs beneath OUT/schedules; explicitly reject
    # an alternative root to avoid recording a wrong portable identity.
    if out.resolve()!=(OUT/'schedules').resolve():raise ValueError('Use registered OUT/schedules directory')
    t0=time.monotonic();pair=prepare_pair_costs(out)
    olds=[read(OLD/f'schedule_{i}.json') for i in (1,2)]
    block=3000;failure=None
    try:generated=[_synchronous(old,pair,block) for old in olds]
    except ValueError as e:
        failure=str(e);block=6000;generated=[_synchronous(old,pair,block) for old in olds]
    outputs=[]
    for i,(old,(rows,trace)) in enumerate(zip(olds,generated),1):
        d=_build(old,pair,rows,trace,block,OLD/f'schedule_{i}.json')
        d['fallback_registration']=dict(triggered=block==6000,primary_failure=failure)
        p=out/f'schedule_{i}.json'
        if p.exists() and read(p)!=d:raise ValueError('Registered schedule changed; do not overwrite')
        write(p,d);outputs.append(entry(p))
    result=dict(status='completed',schedules=outputs,pair_costs=entry(out/'pair_costs.json'),
        common_exposure_block_size=block,primary_3000_failure=failure,
        calibration_triplets_per_suffix=76,seconds=time.monotonic()-t0,
        RGB_forwards=0,moment_forwards=0,parameter_updates=0)
    write(out/'index.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');p.add_argument('--audit',type=Path)
    a=p.parse_args()
    if a.audit:print(validate(read(a.audit)))
    else:print(generate_all())

if __name__=='__main__':main()
