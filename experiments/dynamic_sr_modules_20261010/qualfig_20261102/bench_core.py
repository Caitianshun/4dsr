"""Shared timing loop for the render-speed benchmark (Claude 4dsr-ac, 2026-11-02).

An adapter renders cam00 frame i of a loaded model at the HR test resolution and exposes
  frames             list of frame indices (0..299)
  deform(i)          per-frame time query of the dynamic representation -> state (method specific)
  extra(i, state)    optional per-frame step after the query (ours: P+C residuals); identity otherwise
  raster(i, state)   the method's own rasterizer at HR -> image tensor [3,H,W] on the GPU
Everything that is not per frame (model load, post-hoc cull, cameras, residual arrays) happens before timing.
Timed per frame: deform + extra + raster + clamp. Not timed: loading, PNG encoding, metrics, host copies.
Protocol: 30 warm-up frames; 2 latency passes over all frames (synchronize before and after every frame);
1 stage pass (synchronize between stages); 3 throughput passes (one synchronize per pass, frames back to back).
"""
import json
import statistics
import time


def summary(values_ms):
    v = sorted(values_ms)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return dict(n=len(v), mean=statistics.fmean(v), median=statistics.median(v), p10=q(0.1), p90=q(0.9), min=v[0], max=v[-1])


def run(adapter, torch, warmup=30, latency_passes=2, throughput_passes=3):
    frames = list(adapter.frames)
    sync = torch.cuda.synchronize
    clock = time.perf_counter

    def render(i):
        st = adapter.deform(i)
        st = adapter.extra(i, st)
        return adapter.raster(i, st).clamp(0, 1)

    import os
    load_before = os.getloadavg()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    with torch.no_grad():
        for i in frames[:warmup]:
            render(i)
        sync()
        lat = []
        for _ in range(latency_passes):
            for i in frames:
                sync()
                t0 = clock()
                render(i)
                sync()
                lat.append(1000.0 * (clock() - t0))
        stages = dict(deform=[], extra=[], raster=[])
        for i in frames:
            sync()
            t0 = clock()
            st = adapter.deform(i)
            sync()
            t1 = clock()
            st = adapter.extra(i, st)
            sync()
            t2 = clock()
            adapter.raster(i, st).clamp(0, 1)
            sync()
            t3 = clock()
            stages['deform'].append(1000.0 * (t1 - t0))
            stages['extra'].append(1000.0 * (t2 - t1))
            stages['raster'].append(1000.0 * (t3 - t2))
        tput = []
        for _ in range(throughput_passes):
            sync()
            t0 = clock()
            for i in frames:
                render(i)
            sync()
            tput.append(len(frames) / (clock() - t0))
    latency = summary(lat)
    return dict(frames=len(frames), warmup=warmup, latency_ms=latency, fps_from_mean_latency=1000.0 / latency['mean'],
                fps_throughput=statistics.median(tput), fps_throughput_passes=tput,
                stage_ms={k: summary(v) for k, v in stages.items()},
                peak_memory_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                render_extra_memory_mib=(torch.cuda.max_memory_allocated() - base_alloc) / 2 ** 20,
                host_loadavg_before=list(load_before), host_loadavg_after=list(os.getloadavg()))


def environment(torch, sys):
    import platform
    props = torch.cuda.get_device_properties(0)
    return dict(gpu=torch.cuda.get_device_name(0), gpu_total_mib=props.total_memory / 2 ** 20, capability=list(torch.cuda.get_device_capability(0)),
                torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), python=sys.executable,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark, host=platform.node(), cpu=platform.processor() or platform.machine())


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(record, indent=1) + '\n')
    tmp.replace(path)
