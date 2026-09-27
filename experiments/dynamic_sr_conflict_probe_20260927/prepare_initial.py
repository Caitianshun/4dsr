"""Zero-update state identity and exact legacy forward bridge, training images only."""
import argparse
import shutil
import subprocess
from probe_common import *
from parameter_groups import geometry_hash,configure

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--protocol',type=Path,required=True);a=pa.parse_args();p=load_protocol(a.protocol);out=Path(p['_root']);assert p['b'] is not None
    torch.set_num_threads(4);image_reads,blocked=guard_images(p);tick=time.time();m,data,masks=load_data(p);assert sha(p['start'])==p['start_sha256']
    source=motion.load_model(p['start'],m,restore_rng=True);model=fixed.bake(source,p['time']);assert len(model.xyz)==132972 and model.degree==3
    parity=[]
    with torch.no_grad():
        for c,r in data.items():
            old=motion.render_model(source,r['camera'])['render'];new=render(model,r['camera']);parity.append(dict(camera=c,maxabs=float((new-old).abs().max())))
    assert all(x['maxabs']==0 for x in parity)
    configure(model,p);path=out/'initial_baked.pt';assert not path.exists();torch.save(dict(baked=fixed.state(model),metadata=dict(parent=p['start'],parent_sha256=p['start_sha256'],frame=40,parameter_updates=0)),path)
    initial=dict(path=str(path),sha256=sha(path),model_hash=model_hash(model),geometry_hash=geometry_hash(model),base_count=model.base_count,points=len(model.xyz),degree=model.degree,trainable_scalars=sum(v.numel() for v in model.parameters() if v.requires_grad),parity=parity,parameter_updates=0,seconds=time.time()-tick,counts=COUNTS,image_reads=image_reads,blocked=blocked)
    write(out/'initial_state.json',initial)
    import diff_gaussian_rasterization as ext
    binary=Path(ext._C.__file__);oldenv=read(ROOT/'output/dynamic_sr_evidence_repair_20260927/environment.json')
    env=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),extension=str(binary),extension_sha256=sha(binary),matches_previous_extension=sha(binary)==oldenv['extension_sha256'],modules=legacy.imports(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,default_dtype=str(torch.get_default_dtype()),software_reference=p['source_commit'],nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True))
    assert env['matches_previous_extension'];write(out/'environment.json',env);write(out/'source_manifest.json',sources());print(json.dumps({k:v for k,v in initial.items() if k in ['points','degree','trainable_scalars','seconds','geometry_hash']}))
if __name__=='__main__':main()
