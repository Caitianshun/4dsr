"""One process-exit-triggered, bounded research review of the new controls.

Numerical evaluation/integrity checks already run in run_followup.py. This
optional textual review cannot delay or invalidate their completion. It writes
a local report; it does not inject a message into the desktop conversation.
"""
import argparse
import json
import os
from pathlib import Path
import select
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_20260919'))
from completion_trigger import now, read, run_review_process, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manager-pid', type=int, required=True)
    p.add_argument('--review-timeout', type=float, default=600)
    a = p.parse_args()
    if a.review_timeout <= 0:
        p.error('Review timeout must be positive')
    out = ROOT / 'output/dynamic_sr_20260920/continuation/completion_event'
    out.mkdir(exist_ok=False)
    state = out / 'state.json'
    status = dict(status='waiting_manager_exit', pid=os.getpid(), manager_pid=a.manager_pid,
                  started=now(), mechanism='pidfd + blocking select, no timeout or polling')
    write(state, status)
    try:
        try:
            fd = os.pidfd_open(a.manager_pid)
        except ProcessLookupError:
            fd = None
        if fd is not None:
            try:
                select.select([fd], [], [])
            finally:
                os.close(fd)
        manager = read(out.parent / 'state.json')
        status.update(status='reviewing', event_received=now(), manager_status=manager['status'])
        write(state, status)
        report = out / 'completed_review.md'
        prompt = f'''这是用户已授权的训练完成事件只读研究复核；经理进程已退出，记录状态为 {manager['status']}。
工作目录 /home/cai_tianshun/Project/4dsr。只读，不启动训练、不修改文件、不创建自动化、不发送消息。最终中文答复由调用者保存为报告。
先读 docs/dynamic_sr_next_validation_protocol_2026-09-20.md、docs/dynamic_sr_final_findings_2026-09-20.md、output/dynamic_sr_20260920/continuation/state.json。成功时读 output/dynamic_sr_20260920/output_swap/summary.md、contrasts.json，以及 continuation/summary.md、summary.json 和四份 verification.json；失败时只诊断已记录故障及日志，不编造完整结果。
完成状态下，回答输出交换与四组固定12k续训实际支持或否定什么。比较全图、变化区、非变化区PSNR/SSIM/LPIPS，训练HR/SR拟合、LR闭环、时序、资源和新joint与旧终点的恢复差异。固定支撑分支必须核对 frozen_verification.passed=true、Adam/采样/父模型一致。必要时看固定0040图。区分事实与推断，不把几何/透明度输出的变化等同真实几何错误，也不把SH系数交换当成独立因果分解。说明何种后期G-C协适配得到支持，以及冻结方法是否真的跨域有效，是否以训练拟合下降换泛化。
这些cam00已用于开发，不是独立测试。输出交换不是方法；冻结是强简单基线，不自动称CVPR创新。结合当前投稿时限，建议是否有必要复制牛肉的两个分支、还缺何种决定性证据。只提最小后续，不无目的扩数据。不要在没有证据时断言运动—细节冲突或投影不确定性就是主因。报告用普通Markdown可读形式，不依赖侧栏不支持的LaTeX渲染。'''
        command = ['/home/cai_tianshun/.local/bin/codex', 'exec', '--ephemeral', '--skip-git-repo-check',
                   '--sandbox', 'read-only', '-C', str(ROOT), '-o', str(report), '-']
        with (out / 'review.log').open('x') as log:
            result = run_review_process(command, prompt, log, a.review_timeout)
        valid = result['returncode'] == 0 and report.is_file() and report.stat().st_size > 0
        status.update(status='complete' if valid else 'review_failed', finished=now(),
                      report=str(report), execution=result)
    except Exception as e:
        status.update(status='failed', finished=now(), error=repr(e))
    write(state, status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
