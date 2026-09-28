"""Uniform local evaluation, run immediately after a checkpoint is returned."""
import argparse
import subprocess
import traceback
import csv
from dv_common import *
from runtime_identity import identity as runtime_identity

def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--label',required=True);ap.add_argument('--run-root',type=Path,default=OUT);a=ap.parse_args();require_run_root(a.run_root);canonical=OUT/'evaluation'/a.label;canonical.mkdir(parents=True,exist_ok=True)
    lock=(canonical/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoint_hash=sha(a.checkpoint)
    if (canonical/'complete.json').exists():
        prior=read(canonical/'complete.json');assert prior['checkpoint_sha256']==checkpoint_hash and prior['protocol_sha256']==sha(OUT/'protocol.json');assert sha(canonical/'endpoint.json')==prior['endpoint_sha256'];return
    attempt=1+len(list(canonical.glob('attempt_*')));out=canonical/f'attempt_{attempt:02d}';out.mkdir(exist_ok=False)
    p=read(OUT/'protocol.json');assert os.environ['CUDA_VISIBLE_DEVICES']==p['evaluation']['physical_gpu'];start=time.time()
    assert runtime_identity()==p['runtime']['evaluation']
    assert a.label in {t['task_id'] for t in p['task_plan']}
    checkpoint_meta=read(a.checkpoint.with_suffix('.json'));assert checkpoint_meta['metadata']['intervention_step']==12000 and checkpoint_meta['metadata']['task_id']==a.label
    for c,asset in p['temporal_caches'].items():
        for e in asset['assets']:bound(e)
    def run(cmd,name):
        with (out/(name+'.log')).open('w') as f:subprocess.run([str(v) for v in cmd],stdout=f,stderr=subprocess.STDOUT,check=True)
    cam={};frame_rows=[]
    for camera,split in [('cam00','test'),('cam01','dev')]:
        dest=out/camera
        run([sys.executable,'-u',HERE/'evaluate_camera.py','--manifest',bound(p['manifest']),'--checkpoint',a.checkpoint,'--out',dest,'--split',split,'--roi-protocol',bound(p['roi']),'--teacher-index',bound(p['teacher']),'--method',a.label,'--no-video'],camera)
        d=read(dest/'metrics.json');assert len(d['rows'])==60
        assert d['evaluation_caches'][camera]['dynamic_mask_sha256']==next(e['sha256'] for e in p['temporal_caches'][camera]['assets'] if e['path'].endswith('dynamic_mask.png'))
        f=d['aggregate']['full'];dyn=d['aggregate']['dynamic'];tmp=d['temporal_aggregate']['dynamic'];cam[camera]=dict(psnr=f['psnr_mean'],ssim=f['ssim_mean'],lpips=f['lpips_alex_mean'],dynamic_lpips=dyn['lpips_alex_spatial_mask_mean'],temporal=tmp['gt_relative_warp_l1_mean'],temporal_full=d['temporal_aggregate']['full']['gt_relative_warp_l1_mean'])
        cam[camera].update(H_HR=sum(r['h_hr']['full']['l1'] for r in d['rows'])/60,lr_l1=sum(r['lr_reprojection']['full']['l1'] for r in d['rows'])/60)
        for r in d['rows']:
            q=r['spatial']['full'];frame_rows.append(dict(endpoint=a.label,camera=camera,frame=r['frame_index'],psnr=q['psnr'],ssim=q['ssim'],lpips=q['lpips_alex'],dynamic_lpips=r['spatial']['dynamic']['lpips_alex_spatial_mask'],temporal=r.get('temporal',{}).get('dynamic',{}).get('gt_relative_warp_l1'),lr_l1=r['lr_reprojection']['full']['l1'],H_HR=r['h_hr']['full']['l1']))
        # Same complete 60-frame preview encoding. Playback rate is illustrative.
        import shutil
        ff=shutil.which('ffmpeg')
        if not ff:
            import imageio_ffmpeg
            ff=imageio_ffmpeg.get_ffmpeg_exe()
        if ff:
            listing=dest/'frames.txt';listing.write_text(''.join("file '"+str((dest/'predictions'/camera/f'{i:04d}.png').resolve())+"'\n" for i in range(0,120,2)))
            run([ff,'-hide_banner','-loglevel','error','-f','concat','-safe','0','-r','15','-i',listing,'-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',dest/'preview60.mp4'],camera+'_video')
    run([sys.executable,'-u',HERE/'train76.py','--checkpoint',a.checkpoint,'--out',out/'train76','--run-root',OUT],'train76')
    d=read(out/'train76/metrics.json')
    for r in d['rows']:frame_rows.append(dict(endpoint=a.label,**r))
    result=dict(cameras=cam,train=d['by_camera'],train76_lr=d['aggregate']['lr_l1'],train76=d['aggregate'],train16=d['train16'])
    csvwrite(out/'metrics_per_frame.csv',frame_rows);camera_rows=[dict(endpoint=a.label,camera=c,**v) for c,v in {**cam,**d['by_camera']}.items()];csvwrite(out/'metrics_per_camera.csv',camera_rows)
    write(out/'endpoint.json',result)
    write(out/'complete.json',dict(repeat=checkpoint_meta['metadata']['repeat'],arm=checkpoint_meta['metadata']['method'],step=12000,training_gpu=p['training']['physical_gpu'],parent_sha256=p['parent']['sha256'],attempt_directory=str(out.relative_to(ROOT)),status='completed_evaluation',checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),protocol_sha256=sha(OUT/'protocol.json'),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],rgb_forwards=196,parameter_updates=0,seconds=time.time()-start,source_sha256={str(q.relative_to(ROOT)):sha(q) for q in [HERE/'evaluate_endpoint.py',HERE/'train76.py',HERE/'evaluate_camera.py',HERE/'evaluation_adapter.py',ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py',ROOT/'experiments/dynamic_sr_controlled_headroom_20260926/evaluate.py',ROOT/'experiments/dynamic_sr_20260918/evaluate.py']},video='All 60 registered frames; h264 CRF18 yuv420p, illustrative 15fps',endpoint_sha256=sha(out/'endpoint.json')))
    import shutil
    for name in ['endpoint.json','metrics_per_frame.csv','metrics_per_camera.csv','complete.json']:shutil.copyfile(out/name,canonical/name)
if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/('evaluation_failure_'+str(os.getpid())+'.json'),dict(error=traceback.format_exc()));raise
