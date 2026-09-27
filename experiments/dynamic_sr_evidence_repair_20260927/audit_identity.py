"""Snapshot model, named optimizer, samplers, source and actual extension identity."""
import difflib
import subprocess
import time
from context import *

def main():
    p=paths();m=load_manifest(p['manifest']);protocol=read(OUT/'protocol.json');torch.set_num_threads(4)
    for key in ['manifest','teacher','schedule','lr_curve']:assert sha256(p[key])==protocol[key]['sha256']
    sources={};checkpoints={};ms=read(PRIOR/'methods.json')['methods'];old=read(ROOT/'output/dynamic_sr_view_recovery_20260926/methods.json')['methods']
    inputs={'U6000':p['start'],**{k:Path(ms[k]['checkpoint']) for k in ['shared_12000','shared_17550','C_joint_18000']}}
    for k,v in old.items():
        if 'C_joint' in k or k=='C':inputs['previous_'+k]=Path(v['checkpoint'])
    dest=OUT/'identity_v1';dest.mkdir(exist_ok=False)
    for name,path in inputs.items():
        model=motion.load_model(path,m,restore_rng=True);ck=model.checkpoint
        resume.assert_state_equal(ck['model'][12],model.g.optimizer.state_dict());resume.assert_state_equal(ck['motion_refinement']['child_optimizer'],model.child_optimizer.state_dict())
        assert digest(rng())==digest(ck['rng'])
        opt=optimizer_named(model)
        optrows={n:{**{k:v for k,v in q.items() if k!='state'},'state':{k:dict(shape=list(v.shape),sha256=digest(v),value=float(v) if v.numel()==1 else None) if torch.is_tensor(v) else v for k,v in q['state'].items()}} for n,q in opt.items()}
        checkpoints[name]=dict(path=str(path),sha256=sha256(path),metadata=ck['metadata'],model_optimizer_sha256=support.initial_identity(model),
            topology=shared.topology(model),samplers_sha256=digest(ck['samplers']),rng_sha256=digest(ck['rng']),optimizer=optrows,
            hidden=vars(model.h),optim=vars(model.o),restored_exact=True)
        del model,opt
    for f in HERE.glob('*.py'):sources[str(f)]=sha256(f)
    for f in imports().values():sources[f]=sha256(f)
    import diff_gaussian_rasterization as ext
    import diff_gaussian_rasterization._C as binary
    for f in [Path(ext.__file__),Path(binary.__file__)]:sources[str(f)]=sha256(f)
    extroot=Path(ext.__file__).resolve().parents[1]
    for f in extroot.rglob('*'):
        if f.is_file() and f.suffix in ['.cu','.h','.cpp','.py'] and 'build/' not in str(f):sources[str(f)]=sha256(f)
    # Captured old/new training and rendering sources are compared without overwriting them.
    groups={}
    for label,folder in [('old',ROOT/'output/dynamic_sr_view_recovery_20260926'),('new',PRIOR)]:
        candidates=list(folder.glob('**/train/config.json'))
        for c in candidates:
            cfg=read(c)
            if cfg.get('method')!='C_joint':continue
            for item in cfg.get('sources',[]):
                source=Path(item.get('snapshot',''))
                if source.is_file():groups.setdefault(Path(item['path']).name,{})[label]=source
            break
    diffs=[]
    for name,v in groups.items():
        if set(v)=={'old','new'}:
            diff=''.join(difflib.unified_diff(v['old'].read_text().splitlines(True),v['new'].read_text().splitlines(True),fromfile=str(v['old']),tofile=str(v['new'])))
            file=dest/(name+'.diff');file.write_text(diff);diffs.append(dict(name=name,old_sha=sha256(v['old']),new_sha=sha256(v['new']),different=bool(diff),path=str(file)))
    env=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),visible_cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),
        nvidia_smi=subprocess.check_output(['nvidia-smi','-q'],text=True),extension=str(binary.__file__),extension_sha256=sha256(binary.__file__),
        upstream=str(UPSTREAM),upstream_head=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),modules=imports())
    write_json(OUT/'environment.json',env);write_json(OUT/'source_manifest.json',dict(sha256=sources,diffs=diffs))
    write_json(OUT/'checkpoint_manifest.json',dict(checkpoints=checkpoints))
    write_json(dest/'complete.json',dict(status='passed',checkpoints=list(checkpoints),modules=imports(),completed_unix=time.time()))

if __name__=='__main__':main()
