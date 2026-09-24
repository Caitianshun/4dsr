#!/usr/bin/env python3
"""Create exact trajectory tables/figures from evaluation-only JSON records."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_20260921'
KEYS=['psnr','ssim','lpips_alex']
GROUPS=['teacher_camera_train','lr_only_camera_train','novel_cam00']
LABELS=['Train: prior-camera stratum','Train: other-camera stratum','Unseen camera 00']


def read(p):return json.loads(p.read_text())
def metric(v,group,mode):
    if group=='all_selected_train':
        rows=[r[mode] for r in v['rows'] if r['group']!='novel_cam00']
        return {k:float(np.mean([r[k] for r in rows])) for k in KEYS}
    return v['groups'][group][mode]


def main():
    a=read(OUT/'resolution_trajectory/metrics.json')
    b=read(OUT/'native_resolution_trajectory/metrics.json')
    dst=OUT/'resolution_trajectory'
    deltas={}
    lines=['# 分辨率轨迹逐项表','',
        '所有当前表格固定四帧 0/40/80/118；训练两层分别4相机、3相机，未参与拟合层为cam00。分数是逐图平均；不同分辨率的LPIPS绝对值不可直接横比。','']
    for mode in ['hr','lr_integrated']:
        lines += [f'## 相同父检查点分支：{mode}，6k → 18k','',
            '| 场景 | 分支 | 观察层 | PSNR起点 | PSNR终点 | ΔPSNR | ΔSSIM | ΔLPIPS |',
            '|---|---|---|---:|---:|---:|---:|---:|']
        for scene,cases in a['scenes'].items():
            for case,steps in cases.items():
                for group in GROUPS+['all_selected_train']:
                    start,end=[metric(steps[s],group,mode) for s in ['6000','18000']]
                    d={k:end[k]-start[k] for k in KEYS}
                    deltas[f'{scene}/{case}/{group}/{mode}']=d
                    lines.append(f'| {scene} | {case} | {group} | {start["psnr"]:.6f} | {end["psnr"]:.6f} | {d["psnr"]:+.6f} | {d["ssim"]:+.6f} | {d["lpips_alex"]:+.6f} |')
        lines += ['']
    lines += ['## 从初始化的原生 LR / 完整 HR：2k → 6k','',
        '这里步数为fine阶段，另有1000次coarse。不能与以上18k分支拼接。native_lr看lr_native；full_hr看hr。','',
        '| 场景 | 分支/评价分辨率 | 观察层 | PSNR起点 | PSNR终点 | ΔPSNR | ΔSSIM | ΔLPIPS |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for scene,cases in b['scenes'].items():
        for case,steps in cases.items():
            mode='lr_native' if case=='native_lr' else 'hr'
            for group in GROUPS+['all_selected_train']:
                start,end=[metric(steps[s],group,mode) for s in ['2000','6000']]
                d={k:end[k]-start[k] for k in KEYS}
                deltas[f'native/{scene}/{case}/{group}/{mode}']=d
                lines.append(f'| {scene} | {case}/{mode} | {group} | {start["psnr"]:.6f} | {end["psnr"]:.6f} | {d["psnr"]:+.6f} | {d["ssim"]:+.6f} | {d["lpips_alex"]:+.6f} |')
    lines += ['','## 原生分支后段：4k → 6k（所有四分支固定同一窗口）','',
        '| 场景 | 分支/评价分辨率 | 观察层 | PSNR起点 | PSNR终点 | ΔPSNR | ΔSSIM | ΔLPIPS |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for scene,cases in b['scenes'].items():
        for case,steps in cases.items():
            mode='lr_native' if case=='native_lr' else 'hr'
            for group in GROUPS+['all_selected_train']:
                start,end=[metric(steps[s],group,mode) for s in ['4000','6000']]
                d={k:end[k]-start[k] for k in KEYS}
                deltas[f'native_late/{scene}/{case}/{group}/{mode}']=d
                lines.append(f'| {scene} | {case}/{mode} | {group} | {start["psnr"]:.6f} | {end["psnr"]:.6f} | {d["psnr"]:+.6f} | {d["ssim"]:+.6f} | {d["lpips_alex"]:+.6f} |')
    lines += ['','## 原生 LR / HR 的容量轨迹','', '| 场景 | 分支 | 2k点数 | 4k点数 | 6k点数 |','|---|---|---:|---:|---:|']
    for scene,cases in b['scenes'].items():
        for case,steps in cases.items():
            lines.append('| '+scene+' | '+case+' | '+' | '.join(str(steps[s]['point_count']) for s in ['2000','4000','6000'])+' |')
    lines += ['','## 全部逐点评价','', '| 协议 | 场景 | 分支 | 步数 | 评价模式 | 观察层 | PSNR | SSIM | LPIPS |','|---|---|---|---:|---|---|---:|---:|---:|']
    for protocol,result in [('matched_parent',a),('from_init',b)]:
        for scene,cases in result['scenes'].items():
            for case,steps in cases.items():
                modes=['hr','lr_integrated'] if protocol=='matched_parent' else ['hr','lr_integrated','lr_native']
                for step,v in steps.items():
                    for mode in modes:
                        for group in GROUPS:
                            m=metric(v,group,mode)
                            lines.append(f'| {protocol} | {scene} | {case} | {step} | {mode} | {group} | {m["psnr"]:.6f} | {m["ssim"]:.6f} | {m["lpips_alex"]:.6f} |')
    (dst/'tables.md').write_text('\n'.join(lines)+'\n')
    (dst/'deltas.json').write_text(json.dumps(deltas,indent=2)+'\n')
    for mode in ['hr','lr_integrated']:
        fig,axes=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
        for col,(scene,cases) in enumerate(a['scenes'].items()):
            for row,(case,steps) in enumerate(cases.items()):
                ax=axes[row,col]
                x=[int(s) for s in steps]
                for group,label in zip(GROUPS,LABELS):
                    ax.plot(x,[metric(v,group,mode)['psnr'] for v in steps.values()],marker='o',label=label)
                ax.set_title(f'{scene}\n{case}');ax.set_ylabel('PSNR (dB)');ax.set_xlabel('additional updates')
                ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3)
        fig.suptitle(f'Matched parent: {mode}; same four frames per camera')
        fig.savefig(dst/f'curves_{mode}.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for col,(scene,cases) in enumerate(b['scenes'].items()):
        for row,(case,steps) in enumerate(cases.items()):
            mode='lr_native' if case=='native_lr' else 'hr'
            ax=axes[row,col]
            for group,label in zip(GROUPS,LABELS):
                ax.plot([int(s) for s in steps],[metric(v,group,mode)['psnr'] for v in steps.values()],marker='o',label=label)
            ax.set_title(f'{scene}\n{case}, evaluated at training resolution')
            ax.set_xlabel('fine-stage updates (+1000 coarse)');ax.set_ylabel('PSNR (dB)');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle('From-initialization native LR versus full HR; not the 18k protocol')
    fig.savefig(dst/'curves_native_protocol.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(tables=str(dst/'tables.md'),deltas=deltas),ensure_ascii=False))


if __name__=='__main__':main()
