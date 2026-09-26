"""Finalizer exit event -> artifact integrity and report; no scientific verdict."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
from run_segment import ROOT,leases
from finalize import wait_pid


def audit(root):
    import cv2
    final=root/'final_v1';assert leases.read(final/'complete.json')['status']=='artifacts_ready_for_scientific_and_visual_review'
    methods=leases.read(final/'methods.json')['methods'];index=leases.read(root/'checkpoint_index.json')['checkpoints'];evaluations=[]
    assert len(methods)==len(index)==9
    for name,item in methods.items():
        checkpoint=leases.identity(item['checkpoint']);assert checkpoint['sha256']==index[name]['sha256']
        for split,folder in item['evaluations'].items():
            folder=Path(folder);receipt=leases.read(folder/'complete.json');metrics=leases.read(folder/'metrics.json')
            assert receipt['status']=='completed_evaluation' and receipt['checkpoint_sha256']==checkpoint['sha256']
            assert receipt['metrics_sha256']==leases.sha(folder/'metrics.json')
            assert len(metrics['rows'])==(16 if split=='train_fixed' else 60)
            if item['method']=='O':assert metrics['privileged_train_hr'] and 'PRIVILEGED' in metrics['method']
            evaluations.append(dict(method=name,split=split,metrics_sha256=receipt['metrics_sha256'],rows=len(metrics['rows'])))
    views=leases.read(final/'views/complete.json');assert views['status']=='completed_fixed_views'
    assert {v['method'] for v in views['sources']}=={'Z40000','U40000','O_HR_PRIVILEGED40000'}
    videos=[]
    for path in sorted((final/'views/headroom').glob('*.mp4')):
        cap=cv2.VideoCapture(str(path));assert cap.isOpened();count=0;shape=None;fps=cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok,frame=cap.read()
            if not ok:break
            if shape is None:shape=list(frame.shape)
            assert list(frame.shape)==shape;count+=1
        cap.release();assert count==60 and abs(fps-15)<.01
        videos.append(dict(path=str(path),sha256=leases.sha(path),frames=count,fps=fps,shape=shape))
    assert len(videos)==2
    for p in views['artifacts']:assert Path(p).is_file()
    result=dict(status='artifact_integrity_verified_scientific_review_pending',checkpoints=index,evaluations=evaluations,videos=videos,
        actual_visual_review='not yet performed; decoding proves readability only',scientific_complete=False,finished_utc=leases.stamp())
    leases.write(root/'artifact_audit.json',result)
    report=ROOT/'docs/dynamic_sr_controlled_headroom_results_2026-09-26.md';text=report.read_text();lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('状态：'):lines[i]='状态：固定训练、统一评价和自动产物完整性核验完成；科学结论与实际画面抽看待复核。';break
    text='\n'.join(lines).split('<!-- AUTO_FIXED_TABLES -->')[0].rstrip()
    text+='\n\n<!-- AUTO_FIXED_TABLES -->\n\n'+(final/'tables/tables.md').read_text()
    text+='\n\n图像/视频索引：['+str(final/'views/README.md')+']('+str(final/'views/README.md')+')。\n'
    text+='\n逐帧解码已验证 cam00/cam01 各60帧；此项不是实际视觉判断。待抽看记录与一页结论完成前，本轮不标记科学任务已完成。\n'
    report.write_text(text)
    subprocess.run(['node','scripts/render_codex_math.mjs',str(report)],cwd=ROOT,check=True)
    index_path=root/'execution_index.json';state=leases.read(index_path);state.update(status=result['status'],artifact_audit=str(root/'artifact_audit.json'));leases.write(index_path,state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();a.root=a.root.resolve()
    status=dict(status='waiting_finalizer_exit',started_utc=leases.stamp())
    try:
        if a.worker:audit(a.root);return
        leases.write(a.root/'audit_service_status.json',status)
        wait_pid(leases.read(a.root/'finalize_service.json')['pid'],'finalize.py')
        plan=leases.read(a.root/'controller_spec.json')
        subprocess.run([plan['wu_python'],str(Path(__file__).resolve()),'--root',str(a.root),'--worker'],cwd=ROOT,check=True)
        status.update(status='artifact_integrity_verified_scientific_review_pending',finished_utc=leases.stamp())
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc());raise
    finally:
        if not a.worker:leases.write(a.root/'audit_service_status.json',status)


if __name__=='__main__':main()
