"""Z5-bound front end for the frozen teacher; never rewrite source receipts."""
import argparse,itertools
from pathlib import Path
from common import *
import teacher_highres as teacher
def gate(planpath,receipt_path,samples_path=None):
    plan=read(local(planpath));manifest_path=local(plan['producer_manifest']['path']);m=read(manifest_path)
    if sha(manifest_path)!=plan['producer_manifest']['sha256']:raise ValueError('Teacher producer changed')
    if not m.get('full_time_decode_verified') or m['source_sha256']!=sha(HERE/'prepare_colmap_dataset.py'):raise ValueError('Prepared data/source identity differs')
    receipt_path=local(receipt_path);z=read(receipt_path)
    if z.get('Z5_pass') is not True:raise ValueError('Z5 has not passed; full teacher refused')
    samples_path=local(samples_path or z['samples_identity']['path'])
    if sha(samples_path)!=z['samples_identity']['sha256']:raise ValueError('Z5 sample inventory changed')
    s=read(samples_path);cfg_path=local(s['dataset_config']['path']);cfg=read(cfg_path)
    if sha(cfg_path)!=s['dataset_config']['sha256']:raise ValueError('Dataset configuration changed since Z5')
    expected_rung='full' if m['scene']=='dgF_cube' else 'half'
    if s['dataset']!=cfg['id'] or z['dataset']!=cfg['id'] or s['rung']!=expected_rung or z['rung']!=expected_rung or s['start']!=m['original_start']:raise ValueError('Z5 belongs to another dataset/grid/window')
    for ref in [s['source'],z['source']]:
        if sha(local(ref['path']))!=ref['sha256']:raise ValueError('Z5 source code changed; evidence needs reacceptance')
    f=z['frozen']
    if f['network_SHA256']!=teacher.FROZEN['network_sha256'] or f['checkpoint_SHA256']!=teacher.FROZEN['checkpoint_sha256'] or f['tile']!=0 or f['precision']!='float32':raise ValueError('Z5 frozen teacher identity differs')
    train={(r['camera_id'],r['frame_index']):r for r in m['observations'] if r['split']=='train'}
    keys=[(e['camera'],e['frame']) for e in s['entries']]
    if len(keys)!=len(set(keys)) or set(keys)!=set(itertools.product(m['splits']['train'],[15,150,285])) or s.get('test_views_used') is not False:raise ValueError('Z5 train-only population differs')
    for e in s['entries']:
        if e['source_frame']!=m['original_start']+e['frame'] or e['sha256']['lr']!=train[e['camera'],e['frame']]['lr_sha256']:raise ValueError('Z5 LR does not match prepared training input')
    metric_keys=[(r['camera'],r['frame']) for r in z['rows']]
    if len(metric_keys)!=len(set(metric_keys)) or set(metric_keys)!=set(keys):raise ValueError('Z5 metrics are incomplete or duplicated')
    return dict(receipt=entry(receipt_path),samples=entry(samples_path),producer_manifest=entry(manifest_path),source=entry(__file__),passed=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['cost','run','accept']);p.add_argument('--plan',required=True,type=Path);p.add_argument('--z5-receipt',required=True,type=Path);p.add_argument('--z5-samples',type=Path);a=p.parse_args()
    accepted=gate(a.plan,a.z5_receipt,a.z5_samples)
    write(local(a.plan).parent/'screening_acceptance.json',accepted)
    if a.command=='accept':print(teacher.accept(a.plan))
    else:
        teacher.generate(a.plan,a.command=='cost')
        if a.command=='run':print(teacher.accept(a.plan))
if __name__=='__main__':main()
