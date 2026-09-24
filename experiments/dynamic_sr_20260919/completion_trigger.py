"""Linux pidfd process-exit callbacks, without training-status polling.

Per-model evaluation already runs directly after subprocess.run returns in the
training queues. This listener verifies each whole scene on its queue's exit,
then submits at most one bounded read-only Codex review per outcome.
Local verification/notification never depends on the network review finishing.
The CLI review writes a report; it does not inject a turn into the desktop chat.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,select,shutil,signal,subprocess,sys,tempfile,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_20260919'
SCENES=['cook_spinach','cut_roasted_beef','meetroom_discussion']
CASES=['lr_long','sr_w01','sr_w10','hr_oracle_w10','sr_w10_dense','lr_dense']

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(cond,msg):
    if not cond:raise AssertionError(msg)

def verify_means(ev,name):
    for region in ['full','dynamic','static']:
        for key in ['psnr','ssim','lpips_alex' if region=='full' else 'lpips_alex_spatial_mask']:
            values=[r['spatial'][region][key] for r in ev['rows']]
            require(all(v is not None and math.isfinite(v) for v in values),f'{name}: nonfinite {region}/{key}')
            mean=sum(values)/len(values)
            reported=ev['aggregate'][region][key+'_mean']
            require(abs(mean-reported)<1e-7,f'{name}: mean mismatch {region}/{key}')

def verify_scene(scene):
    """Completed result integrity, independent of whether metrics improved."""
    rows=[];parent_hashes=set();manifest_hashes=set()
    for case in CASES:
        name=scene+'_'+case;run=OUT/name
        c=read(run/'config.json');done=read(run/'complete.json')
        ev=read(OUT/(name+'_evaluation')/'metrics.json')
        sampling=read(OUT/(name+'_sampling')/'sampling.json')
        ck=sha(run/'checkpoint_final.pt')
        require(done['intervention_step']==18000,f'{name}: wrong end step')
        require(ev['checkpoint_sha256']==ck==sampling['checkpoint_sha256'],f'{name}: checkpoint identity mismatch')
        require(ev['manifest_sha256']==c['manifest_sha256']==sampling['manifest_sha256'],f'{name}: manifest mismatch')
        require(ev['manifest_sha256']==sha(ev['manifest']),f'{name}: current manifest changed')
        require(len(ev['rows'])==60 and sum('temporal' in r for r in ev['rows'])==59,f'{name}: incomplete evaluation')
        require(ev['lpips_status'].startswith('alex_v0.1'),f'{name}: LPIPS unavailable')
        verify_means(ev,name)
        for step in [1200,6000,12000,18000]:
            fit=read(run/f'fit_{step}.json')
            require(fit['step']==step and len(fit['rows'])==16,f'{name}: incomplete train fit at {step}')
        exposure=read(run/'exposure.json');expected=0 if c['teacher']=='none' else 18000
        require(sum(exposure.values())==expected,f'{name}: exposure mismatch')
        require(done['points']==ev['gaussian_count'],f'{name}: point count mismatch')
        if c['dense']:require(done['points']<=220000,f'{name}: capacity budget exceeded')
        else:require(done['points']==c['initial_points'],f'{name}: fixed topology changed')
        parent_hashes.add(c['parent_sha256']);manifest_hashes.add(c['manifest_sha256'])
        rows.append(dict(case=case,checkpoint_sha256=ck,points=done['points'],teacher_exposures=expected,
                         information_condition='extra_train_HR_oracle' if c['teacher']=='hr' else 'LR_and_frozen_SR_only'))
    require(len(parent_hashes)==1 and len(manifest_hashes)==1,f'{scene}: controls do not share initial state/data')
    return dict(scene=scene,status='passed',checked_at=now(),checks=rows)

def notify(title,body):
    executable=shutil.which('notify-send')
    if not executable:return dict(status='unavailable')
    try:p=subprocess.run([executable,'--app-name=4DSR',title,body],capture_output=True,text=True,timeout=10)
    except (OSError,subprocess.TimeoutExpired) as e:return dict(status='failed',error=repr(e))
    return dict(status='sent' if p.returncode==0 else 'failed',returncode=p.returncode,stderr=p.stderr[-500:])

def notify_once(eventdir,kind,title,body):
    """At-most-once attempt, persisted before the external desktop call."""
    marker=eventdir/f'{kind}_notification_started.json'
    try:
        with marker.open('x') as f:json.dump(dict(started=now()),f)
    except FileExistsError:return dict(status='already_attempted')
    result=notify(title,body)
    write(eventdir/f'{kind}_notification_result.json',result)
    return result

def summary():
    return subprocess.run(['/usr/bin/python3',str(Path(__file__).with_name('summarize_controls.py')),
        '--root',str(OUT),'--out',str(OUT/'assessment_controls_v1')],cwd=ROOT,capture_output=True,text=True,timeout=120)

def watch_state(path):
    """Subscribe once, handling exit before/during registration without polling."""
    before=read(path)
    if before['status'] in ['complete','failed']:return None
    try:fd=os.pidfd_open(before['pid'])
    except ProcessLookupError:return None
    try:
        after=read(path)
        require(before['pid']==after['pid'],f'Queue PID changed while subscribing: {path}')
        if after['status'] in ['complete','failed']:
            os.close(fd);return None
        return fd
    except BaseException:
        os.close(fd);raise

def run_review_process(cmd,prompt,log,timeout_s):
    """One deadline wait on a pidfd; kill the whole review group on timeout.

    This deadline bounds only the optional model process. Training waits have
    no timeout. No periodic process-state probes or Popen.wait(timeout) loop.
    """
    started=time.monotonic()
    with tempfile.TemporaryFile(mode='w+',encoding='utf-8') as input_file:
        input_file.write(prompt);input_file.seek(0)
        return _run_review_process(cmd,input_file,log,timeout_s,started)

def _run_review_process(cmd,input_file,log,timeout_s,started):
    with subprocess.Popen(cmd,stdin=input_file,stdout=log,stderr=subprocess.STDOUT,
                          text=True,cwd=ROOT,start_new_session=True) as child:
        fd=None;timed_out=False
        try:
            try:fd=os.pidfd_open(child.pid)
            except ProcessLookupError:pass # exited before registration
            if fd is not None:
                ready,_,_=select.select([fd],[],[],max(0,timeout_s-(time.monotonic()-started)))
                if not ready:
                    timed_out=True
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            code=child.wait() # blocking waitpid, not timed polling
        except BaseException:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait();raise
        finally:
            if fd is not None:os.close(fd)
    return dict(returncode=code,timed_out=timed_out,elapsed_s=time.monotonic()-started)

def review(eventdir,kind,timeout_s=600):
    """One model call per all-complete event or first failure; never a loop."""
    marker=eventdir/f'{kind}_review_started.json'
    try:
        with marker.open('x') as f:json.dump(dict(started=now(),timeout_s=timeout_s),f)
    except FileExistsError:return dict(status='already_started')
    report=eventdir/f'{kind}_review.md'
    prompt=f'''用户已授权这次训练结束自动触发的只读研究复核。当前事件：{kind}。
工作目录 /home/cai_tianshun/Project/4dsr。不要启动训练、不要修改任何文件、不要创建自动化或发送消息；只读审查并在最终答复写完整中文报告，调用者将保存它。
先读取 docs/dynamic_sr_control_protocol_2026-09-19.md、docs/dynamic_sr_control_execution_2026-09-19.md、{eventdir}/trigger_state.json、output/dynamic_sr_20260919/assessment_controls_v1/summary.md，以及 docs/dynamic_sr_novelty_update_2026-09-19.md。
若为失败事件，核对对应队列错误和最近日志，准确定位失败及恢复建议；不得拿不完整结果作机制结论。
若全部成功，按原用户任务验证可信细节为什么未充分拟合/保留：比较延训曲线、权重、匹配HRoracle、SR与LR增密、会议室独立域，以及1/2/4倍率采样、PSNR/SSIM/LPIPS、变化区域和时序。必要时读原始JSON核算，目视固定帧图像。说明真实性/可靠性、控制不能排除的原因、资源成本与收敛证据。结合上述文献给一个优先创新候选及可证伪验证，不把简单延训或可靠增密包装全新贡献。区分事实、推断、尚未验证。报告只基于实际文件，不编造完成、指标或因果。'''
    cmd=['/home/cai_tianshun/.local/bin/codex','exec','--ephemeral','--skip-git-repo-check',
         '--sandbox','read-only','-C',str(ROOT),'-o',str(report),'-']
    try:
        with (eventdir/f'{kind}_review.log').open('x') as log:
            execution=run_review_process(cmd,prompt,log,timeout_s)
        result=dict(status='timeout' if execution['timed_out'] else
                    ('complete' if execution['returncode']==0 and report.is_file() and report.stat().st_size else 'failed'),
                    **execution,report=str(report),finished=now())
    except Exception as e:result=dict(status='failed',error=repr(e),report=str(report),finished=now())
    write(eventdir/f'{kind}_review_result.json',result)
    return result

def self_test():
    # Child waits on stdin. Parent subscribes before releasing it: deterministic
    # simulation of process exit with no sleeps or periodic checks.
    results=[]
    for code in [0,7]:
        child=subprocess.Popen([sys.executable,'-c',f'import sys;sys.stdin.read();sys.exit({code})'],stdin=subprocess.PIPE)
        fd=os.pidfd_open(child.pid);start=time.monotonic();child.stdin.close()
        ready,_,_=select.select([fd],[],[])
        actual=child.wait();os.close(fd)
        require(bool(ready) and actual==code,'pidfd exit event failed')
        results.append(dict(exit_code=actual,callback_latency_s=time.monotonic()-start))
    ev=read(ROOT/'output/dynamic_sr_20260918/cook_spinach_pilot_v1_joint/evaluation/metrics.json')
    verify_means(ev,'historical schema verification')
    ev['aggregate']['full']['psnr_mean']+=.1
    try:verify_means(ev,'intentional corruption')
    except AssertionError:rejected=True
    else:rejected=False
    require(rejected,'metric corruption was not rejected')
    return dict(status='passed',cases=results,actual_metric_schema='passed',intentional_bad_mean_rejected=rejected,
                mechanism='pidfd + blocking select; no timeout/polling')

def listen(eventdir,review_timeout_s=600):
    """One listener for one experiment attempt; event evidence is never reset."""
    eventdir=Path(eventdir);eventdir.mkdir(parents=True,exist_ok=True)
    claim=eventdir/'listener_started.json'
    with claim.open('x') as f:json.dump(dict(pid=os.getpid(),started=now(),script_sha256=sha(__file__)),f)
    status=dict(status='listening',started=now(),mechanism='kernel pidfd exit events',scenes={},pid=os.getpid())
    state=eventdir/'trigger_state.json';write(state,status)
    fds={};immediate=[];subscription_errors={};reviews={}
    # Worker never mutates trigger_state.json. Its own result file is atomic;
    # only this event-loop thread writes the shared listener state.
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='4dsr-review') as executor:
        def submit_review(kind):
            if kind not in reviews:reviews[kind]=executor.submit(review,eventdir,kind,review_timeout_s)
        def fail_notify(body):
            if 'failure_notification' not in status:
                status['failure_notification']=notify_once(eventdir,'failure','4DSR 实验检查发现异常',body)
                write(state,status)
        try:
            for scene in SCENES:
                try:fd=watch_state(OUT/f'{scene}_queue.json')
                except Exception as e:subscription_errors[scene]=repr(e);fd=None
                if fd is None:immediate.append(scene)
                else:fds[fd]=scene
            while fds or immediate:
                scenes=immediate;immediate=[]
                if not scenes:
                    ready,_,_=select.select(list(fds),[],[]) # no timeout: only exit wakes us
                    for fd in ready:scenes.append(fds.pop(fd));os.close(fd)
                for scene in scenes:
                    try:
                        require(scene not in subscription_errors,f'Subscription failed: {subscription_errors.get(scene)}')
                        q=read(OUT/f'{scene}_queue.json')
                        require(q['status']=='complete',f'Queue exited in state {q["status"]}: {q.get("error")}')
                        result=verify_scene(scene)
                    except Exception as e:result=dict(scene=scene,status='failed',error=repr(e),checked_at=now())
                    status['scenes'][scene]=result;write(eventdir/f'{scene}_verification.json',result);write(state,status)
                    if result['status']=='failed':
                        fail_notify(f'{scene} 已退出或无法订阅，错误已保存；其余队列继续等待结束事件。')
                    try:
                        s=summary();callback=dict(returncode=s.returncode,stderr=s.stderr[-2000:],finished=now())
                    except Exception as e:callback=dict(status='failed',error=repr(e),finished=now())
                    write(eventdir/f'{scene}_summary_callback.json',callback)
                    if result['status']=='failed':submit_review('failure')
            allpass=len(status['scenes'])==len(SCENES) and all(v['status']=='passed' for v in status['scenes'].values())
            if allpass:
                try:
                    trajectory=OUT/'generalization_trajectory/state.json'
                    fd=watch_state(trajectory)
                    if fd is not None:
                        status['status']='waiting_trajectory';write(state,status)
                        try:select.select([fd],[],[])
                        finally:os.close(fd)
                    q=read(trajectory)
                    require(q['status']=='complete' and len(q['completed'])==18,'Held-out trajectory incomplete')
                    diag=read(trajectory.parent/'metrics.json')
                    require(diag['no_checkpoint_selection'] is True,'Trajectory protocol changed')
                    status['trajectory']=dict(status='passed',checked_at=now(),path=str(trajectory.parent/'metrics.json'))
                except Exception as e:
                    allpass=False;status['trajectory']=dict(status='failed',error=repr(e));write(state,status)
                    fail_notify('训练结果已保存，但补充检查点评测不完整；请查看 trigger_state.json。')
                    submit_review('failure')
            status.update(status='verified' if allpass else 'failed',finished=now());write(state,status)
            if allpass:
                # User-visible local completion is independent of model/network.
                status['notification']=notify_once(eventdir,'completed','4DSR 训练与评测已完成',
                    '本地完整性核验已通过；结果已保存。文字复核在后台执行，超时不会影响本次完成结果。')
                write(state,status);submit_review('completed')
        finally:
            for fd in fds:os.close(fd)
        # Only now can waiting for the optional worker delay listener shutdown:
        # all queue/trajectory exits and local notifications are already handled.
        status['reviews']={kind:future.result() for kind,future in reviews.items()}
        if 'completed' in status['reviews']:status['review']=status['reviews']['completed']
        status['review_phase_finished']=now();write(state,status)
    return status

def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true')
    p.add_argument('--review-timeout',type=float,default=600,help='Optional CLI review deadline in seconds (default 600)')
    p.add_argument('--event-dir',type=Path,default=OUT/'completion_events',help='Fresh directory for this listener attempt')
    a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()));return
    if not math.isfinite(a.review_timeout) or a.review_timeout<=0:p.error('--review-timeout must be finite and positive')
    listen(a.event_dir,a.review_timeout)

if __name__=='__main__':main()
