"""Generate factual tables for the human-authored decision report from verified JSON."""
from pathlib import Path
from summarize import read
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_geometry_residual_20260924'


def main():
    lines=['# G/L 完整结果表（自动提取，无自动胜负判断）','']
    for scene,label in [('cook','菠菜'),('discussion','会议室')]:
        summary=read(OUT/f'{scene}_summary_v1/summary.json');rows=summary['metrics']
        idx={(r['method'],r['split'],r['step'],r['region']):r for r in rows}
        lines += [f'## {label}：固定 6000 步','', '| 视角 | 方法 | 全图 PSNR | SSIM | 全图 LPIPS | 变化区 LPIPS | 其余区 LPIPS | 变化区时序 L1×1000 | LR 回投 PSNR |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for split,cam in [('test','cam00'),('dev','cam01')]:
            for method in ['U','W','B4','G','L']:
                f=idx[method,split,6000,'full'];d=idx[method,split,6000,'dynamic'];s=idx[method,split,6000,'static']
                lines += [f'| {cam} | {method} | {f["psnr"]:.5f} | {f["ssim"]:.6f} | {f["lpips"]:.6f} | {d["lpips"]:.6f} | {s["lpips"]:.6f} | {d["temporal"]*1000:.5f} | {f["lr_psnr"]:.5f} |']
        lines += ['','| 比较 | 视角 | Δ全图 PSNR | 全图 LPIPS 变化% | 变化区 LPIPS 变化% | 变化区时序误差变化% |','| --- | --- | ---: | ---: | ---: | ---: |']
        for a,b in [('G','U'),('L','U'),('L','G'),('G','W'),('L','W'),('G','B4'),('L','B4')]:
            for split,cam in [('test','cam00'),('dev','cam01')]:
                af=idx[a,split,6000,'full'];bf=idx[b,split,6000,'full'];ad=idx[a,split,6000,'dynamic'];bd=idx[b,split,6000,'dynamic']
                lines += [f'| {a}/{b} | {cam} | {af["psnr"]-bf["psnr"]:+.5f} | {100*(af["lpips"]/bf["lpips"]-1):+.3f} | {100*(ad["lpips"]/bd["lpips"]-1):+.3f} | {100*(ad["temporal"]/bd["temporal"]-1):+.3f} |']
        storage={r['method']:r for r in read(OUT/f'{scene}_inference_storage_v1/storage.json')['methods']}
        lines += ['','| 方法 | 训练秒 | 进程峰值 GB | 渲染 ms/帧 | 可训练标量 | 推理张量 MB | 序列化推理档 MB |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for c in summary['costs']:
            s=storage[c['method']]
            lines += [f'| {c["method"]} | {c["train_seconds"]:.3f} | {c["peak_allocated_gb"]:.3f} | {1000*c["render_seconds_test60"]/60:.4f} | {c["trainable_parameters"]} | {s["inference_tensor_bytes"]/1e6:.4f} | {s["inference_archive_bytes"]/1e6:.4f} |']
        fit=read(OUT/f'{scene}_teacher_fit_v1/teacher_fit.json')['methods']
        h={(r['method'],r['split'],r['reference'],r['region']):r for r in summary['h_metrics'] if r['step']==6000}
        lines += ['','| 方法 | train16 教师 RGB L1 | 对 U 变化% | train16 教师 H L1 | train16 对 HR PSNR | train16 对 HR LPIPS |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for method in ['U','W','B4','G','L']:
            f=fit[method]['l1_mean'];tr=idx[method,'train_fixed',6000,'full']
            lines += [f'| {method} | {f:.7f} | {100*(f/fit["U"]["l1_mean"]-1):+.3f} | {h[method,"train_fixed","train_teacher","full"]["h_l1"]:.7f} | {tr["psnr"]:.5f} | {tr["lpips"]:.6f} |']
        lines += ['','| 方法/步数 | 静态位移中位数 / P95 | 时变 RMS 中位数 / P95 | 静态/父轴中位数 / P95 | 时变/父轴中位数 / P95 |','| --- | ---: | ---: | ---: | ---: |']
        mapping=read(OUT/f'{scene}_methods.json')['methods']
        for method in ['G','L']:
            for step in [1200,6000]:
                s=read(Path(mapping[method]['train_dir'])/f'residual_statistics_{step}.json')
                vals=[f'{s[k]["median"]:.6f} / {s[k]["p95"]:.6f}' for k in ['static_world','temporal_world','static_parent_scale','temporal_parent_scale']]
                lines += [f'| {method}/{step} | '+' | '.join(vals)+' |']
        lines += ['']
    lines += ['变化百分比为误差本身的相对变化，负数更好。世界坐标未假定为米；静态位移及时间 RMS 不等于几何或轨迹真实。进程峰值包含端点诊断，旧 U/W 的额外梯度诊断与本轮不同；不能据此宣布显存优化。推理档排除了 Adam/RNG，属于存储核算档，不替代已验证的完整检查点加载器。','']
    (OUT/'result_tables.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
