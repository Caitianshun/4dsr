"""CPU-only full-camera schedule, preserving every old B4 LR/time draw."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['manifest', 'old-b-config', 'out']:
        p.add_argument('--' + key, required=True, type=Path)
    args = p.parse_args()
    m, old = [json.loads(v.read_text()) for v in [args.manifest, args.old_b_config]]
    assert old['manifest_sha256'] == sha(args.manifest) and old['seed'] == 20260923
    assert old['steps'] == 6000 and old['branch'] == 'ordinary_split'
    records = [o for o in m['observations'] if o['split'] == 'train']
    cams = sorted({o['camera_id'] for o in records})
    frames = sorted({o['frame_index'] for o in records})
    lookup = {(o['camera_id'], o['frame_index']): i for i, o in enumerate(records)}
    assert len(lookup) == len(cams) * len(frames) and frames == list(range(0, 120, 2))
    old_cams = set(old['teacher_cameras'])
    old_ids = [i for i, o in enumerate(records) if o['camera_id'] in old_cams]
    assert len(old_cams) == 4 and len(old_ids) == 240
    seed = old['seed']
    lr_rng, sr_rng, cam_rng = [random.Random(seed + n) for n in [177, 211, 313]]
    rows, bag = [], []
    lr_sha, time_sha, draw_sha = [hashlib.sha256() for _ in range(3)]
    for _ in range(6000):
        li, old_si = lr_rng.randrange(len(records)), sr_rng.choice(old_ids)
        if not bag:
            bag = list(cams)
            cam_rng.shuffle(bag)
        camera = bag.pop()
        frame = records[old_si]['frame_index']
        si = lookup[camera, frame]
        rows.append([li, si, old_si])
        lr_sha.update(f'{li}\n'.encode())
        time_sha.update(f'{frame}\n'.encode())
        draw_sha.update(f'{li},{si}\n'.encode())
    counts = Counter(records[si]['camera_id'] for _, si, _ in rows)
    assert max(counts.values()) - min(counts.values()) <= 1
    result = dict(status='prepared_fixed_schedule', scene=m['scene'], manifest_sha256=sha(args.manifest),
                  source_old_B_config=str(args.old_b_config.resolve()), source_old_B_config_sha256=sha(args.old_b_config),
                  seed=seed, steps=6000, train_cameras=cams, frames=frames, original_teacher_cameras=sorted(old_cams),
                  record_keys=[[o['camera_id'], o['frame_index']] for o in records],
                  row_schema=['lr_record_index', 'full_camera_sr_record_index', 'old_B4_sr_record_index'], rows=rows,
                  lr_draw_sha256=lr_sha.hexdigest(), sr_frame_sha256=time_sha.hexdigest(), draw_sha256=draw_sha.hexdigest(),
                  sr_camera_counts=dict(counts), sr_observation_counts=dict(Counter(f'{records[si]["camera_id"]}/{records[si]["frame_index"]}' for _,si,_ in rows)),
                  rule='Exact B4 LR draws and SR times; independently seeded shuffled full-camera cycles, counts differ by at most1',
                  global_rng_used=False, source_sha256=sha(__file__), created_utc=datetime.now(timezone.utc).isoformat())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ['scene','train_cameras','steps','lr_draw_sha256','sr_frame_sha256','draw_sha256']}))


if __name__ == '__main__':
    main()
