"""Audit numeric pose/video mapping, stream PTS, and source integrity on CPU."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/dynamic_sr/meetroom_raw/discussion'
SOURCE = RAW.parent / 'source'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

if __name__ == '__main__':
    stream_records = []
    for i in range(13):
        p = RAW / f'cam_{i}.mp4'
        command = [str(ROOT / 'scripts/ffprobe_local.sh'), '-v', 'error', '-select_streams', 'v:0',
                   '-show_streams', '-show_frames', '-show_entries',
                   'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:frame=best_effort_timestamp_time',
                   '-of', 'json', str(p)]
        data = json.loads(subprocess.check_output(command))
        (SOURCE / f'ffprobe_cam{i:02d}.json').write_text(json.dumps(data, indent=2) + '\n')
        stream, = data['streams']
        pts = np.array([float(f['best_effort_timestamp_time']) for f in data['frames']])
        assert (stream['width'], stream['height'], len(pts)) == (1280, 720, 300), stream
        assert np.allclose(pts, np.arange(300) / 30, atol=1e-6), pts
        stream_records.append({'camera_id': f'cam{i:02d}', 'raw_filename': p.name,
                               'pose_row_index': i, 'stream': stream, 'frames': len(pts),
                               'pts_max_error_from_i_over_30_s': float(np.max(np.abs(pts-np.arange(300)/30))),
                               'sha256': sha(p)})
    bounds = np.load(RAW / 'poses_bounds.npy')
    centers = bounds[:,:15].reshape(-1,3,5)[:,:3,3]
    centered = centers - centers.mean(axis=0)
    _, singular, rotation = np.linalg.svd(centered)
    flat = centered @ rotation[:2].T
    from scipy.spatial import ConvexHull
    train = list(range(2,13))
    hull = ConvexHull(flat[train])
    inside = lambda idx: bool(np.all(hull.equations[:,:2] @ flat[idx] + hull.equations[:,2] <= 1e-7))
    report = {'dataset': 'MeetRoom discussion', 'official_paper': 'https://arxiv.org/abs/2210.14831',
              'official_repository': 'https://github.com/AlgoHunt/StreamRF',
              'source_code_commit': json.loads((SOURCE/'official_commit.json').read_text())['sha'],
              'stream_records': stream_records,
              'synchronization': {'official_claim': 'external pulse synchronizes camera shutters',
                                  'observed_pts_identical': True,
                                  'limitation': 'PTS equality checks file indexing; physical synchronization follows supplied dataset, not independently measured.'},
              'mapping_evidence': 'Official prepare_dataset.py converts raw cam_N.mp4 to camNN.png; util/llff_dataset.py sorts image filenames; poses_bounds rows follow this numeric camera order.',
              'test_split_evidence': 'Official meetroom_init.json llffhold=100; LLFFDataset chooses index % hold_every == 0 for test: cam00 in 13 cameras.',
              'pilot_split': {'train': [f'cam{i:02d}' for i in train], 'dev': ['cam01'], 'test': ['cam00']},
              'camera_geometry': {'camera_centers_world': centers.tolist(),
                                  'pca_2d_retained_variance': float((singular[:2]**2).sum()/(singular**2).sum()),
                                  'test_cam00_inside_train_camera_pca_hull': inside(0),
                                  'dev_cam01_inside_train_camera_pca_hull': inside(1),
                                  'train_extent_1p1': float(np.max(np.linalg.norm(centers[train]-centers[train].mean(0),axis=1))*1.1)},
              'source_snapshots': {p.name: sha(p) for p in SOURCE.glob('official_*') if p.is_file()},
              'poses_sha256': sha(RAW/'poses_bounds.npy')}
    (RAW/'source_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'audit': str(RAW/'source_audit.json'), 'streams': len(stream_records),
                      'all_pts_match_30fps': True, 'camera_geometry': report['camera_geometry']}, indent=2))
