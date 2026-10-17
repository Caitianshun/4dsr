"""超分段最终步拓扑修复入口（2026-10-11，4dsr-ee）。只新增本文件，不修改任何已有入口。

问题：cullretrain / modsr / csc 的超分段在最后一步（cursor = stop）仍按日程做致密化与剪枝（topology），随后直接保存检查点；
最后一次克隆/拆分出的高斯没有经过任何优化。4DGaussians 原作者的 train.py 在同一迭代中先保存、后致密化。
10-11 扫描 165 个超分段运行：检查点在最后 24 个训练行上重算的 LR L1 与训练记录之比（最终步损伤比），
与训练视角 LR 闭环偏差的相关为 −0.84；比值 ≥ 1.2 的 7 个运行 cam00 平均低 1.56 dB。LR 段不受影响（第 10000 步后不再致密化）。

用法：srfix.py --mode {fix,dual} <cullretrain|modsr|csc> sr <该入口原参数...>
  fix ：第 stop 步跳过 topology，其余不变（含该步的优化器更新）；主检查点即修复版。
  dual：主检查点保持原行为；在第 stop 步 topology 之前另存模型状态，运行结束后与主检查点的 plan/affine 合并为
        checkpoint_<stop>_pretopo.pt，用于同一训练轨迹上“原行为 vs 去掉最终拓扑”的配对比较
        （二者另差该步对形变网络的一次更新：原行为中拓扑替换了高斯参数，只有形变网络在该步得到更新）。
运行结束在输出目录写 srfix.json（模式、入口、源码 SHA、命中的步）。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CULL = ROOT / 'experiments/dynamic_sr_cull_retrain_20261009'
CSC = ROOT / 'experiments/dynamic_sr_csc_20261010'
for p in (CULL, HERE, CSC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cullretrain as cr  # noqa: E402  (sets up the runtime paths and imports torch)
import torch  # noqa: E402

STATE = dict(mode=None, stop=None, out=None, hits=[])
_original_topology = cr.topology


def topology(g, opt, stage, iteration, extent, out, statistics, prune_gate, counts):
    if stage == 'fine' and iteration == STATE['stop'] and Path(out).resolve() == STATE['out']:
        STATE['hits'].append(int(iteration))
        if STATE['mode'] == 'fix':
            counts['final_step_topology_skipped'] = counts.get('final_step_topology_skipped', 0) + 1
            return None
        torch.save(dict(model=g.capture(), deformation_accum=g._deformation_accum),
                   STATE['out'] / f'pretopo_model_{iteration:05d}.pt')
    return _original_topology(g, opt, stage, iteration, extent, out, statistics, prune_gate, counts)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('fix', 'dual'), required=True)
    p.add_argument('entry', choices=('cullretrain', 'modsr', 'csc', 'refs'))
    p.add_argument('rest', nargs=argparse.REMAINDER)
    a = p.parse_args()
    rest = list(a.rest)
    # refs.py takes its own options before the subcommand (e.g. --teacher-source hr --hr-root ... sr ...)
    k = rest.index('sr') if 'sr' in rest else -1
    if k < 0 or (k > 0 and a.entry != 'refs'):
        raise SystemExit('srfix only wraps the sr subcommand')
    sr_args = rest[k:]
    if '--holdout-cameras' in sr_args and '--holdout-refill' not in sr_args:
        raise SystemExit('holdout without refill changes stop; not supported')
    out = Path(sr_args[sr_args.index('--out') + 1])
    stop = int(sr_args[sr_args.index('--stop') + 1]) if '--stop' in sr_args else cr.SR_STEPS
    STATE.update(mode=a.mode, stop=stop, out=(out if out.is_absolute() else ROOT / out).resolve())
    cr.topology = topology
    if a.entry == 'cullretrain':
        module = cr
    elif a.entry == 'modsr':
        import modsr as module
    elif a.entry == 'refs':
        import refs as module
    else:
        import csc as module
    if getattr(module, 'cr', cr) is not cr:
        raise SystemExit('entry imported a different cullretrain module')
    module.main(rest)
    if STATE['hits'] != [stop]:
        raise SystemExit(f'final-step hook fired {STATE["hits"]}, expected [{stop}]')
    record = dict(mode=a.mode, entry=a.entry, argv=rest, stop=stop, hits=STATE['hits'],
                  sources={str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, cr.__file__, module.__file__)})
    if a.mode == 'dual':
        main_ck = STATE['out'] / f'checkpoint_{stop:05d}.pt'
        pre = STATE['out'] / f'pretopo_model_{stop:05d}.pt'
        payload = torch.load(main_ck, map_location='cpu', weights_only=False)
        state = torch.load(pre, map_location='cpu', weights_only=False)
        payload['model'] = state['model']
        payload['deformation_accum'] = state['deformation_accum']
        payload['parameter_gradients'] = None
        payload['srfix'] = dict(variant='pretopo', note='model state before the final-step topology; plan and affine from the main checkpoint')
        merged = STATE['out'] / f'checkpoint_{stop:05d}_pretopo.pt'
        tmp = merged.with_name(merged.name + '.tmp')
        torch.save(payload, tmp)
        tmp.replace(merged)
        record.update(main_checkpoint=dict(path=str(main_ck.relative_to(ROOT)), sha256=sha(main_ck)),
                      pretopo_checkpoint=dict(path=str(merged.relative_to(ROOT)), sha256=sha(merged)))
        pre.unlink()
    (STATE['out'] / 'srfix.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(srfix=a.mode, hits=STATE['hits'])), flush=True)


if __name__ == '__main__':
    main()
