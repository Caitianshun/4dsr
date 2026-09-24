"""Fixed train16 raw-render RGB-to-teacher fitting diagnostic; no updates/HR reads."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from geometry_model import load_model,render_model
from summarize import read,write,sha
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_detail_supervision_20260924'))
# This shared module constructs cameras using calibration, time and a zero RGB placeholder.
import importlib.util
spec=importlib.util.spec_from_file_location('teacher_fit_shared_camera',ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py')
evalmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(evalmod)
from n3dv_data import load_manifest
from common import image_tensor


def main():
    p=argparse.ArgumentParser()
    for k in ['manifest','methods','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    m=load_manifest(a.manifest);mapping=read(a.methods)['methods'];result={}
    for method in ['U','W','B4','G','L']:
        item=mapping[method];evaluation=read(Path(item['evaluations']['train_fixed_6000'])/'metrics.json')
        assert evaluation['checkpoint_sha256']==sha(item['checkpoint'])
        config=read(Path(item['train_dir'])/'config.json');teacher_hashes={(x['camera'],x['frame']):x['sha256'] for x in config['teacher_inputs']}
        observations={(o['camera_id'],o['frame_index']):o for o in m['observations'] if o['split']=='train'}
        model=load_model(item['checkpoint'],m);model.g._deformation.eval();rows=[]
        with torch.inference_mode():
            for i,key in enumerate(evaluation['observation_keys']):
                pair=key['camera_id'],key['frame_index'];o=observations[pair]
                assert pair[0] not in ['cam00','cam01'] and pair in teacher_hashes
                teacher_path=Path(m['_root'])/'sr_swinir_x4'/pair[0]/Path(o['lr_path']).name
                assert sha(teacher_path)==teacher_hashes[pair]
                teacher=image_tensor(teacher_path);raw=render_model(model,evalmod.render_camera(m,o,i))['render']
                assert raw.shape==teacher.shape and torch.isfinite(raw).all()
                delta=(raw-teacher).double()
                mse=float(delta.square().mean());l1=float(delta.abs().mean())
                rows.append(dict(camera=pair[0],frame=pair[1],teacher_sha256=teacher_hashes[pair],l1=l1,mse=mse))
        assert len(rows)==16
        result[method]=dict(rows=rows,l1_mean=sum(x['l1'] for x in rows)/16,mse_mean=sum(x['mse'] for x in rows)/16,
                            checkpoint_sha256=evaluation['checkpoint_sha256'])
        del model
    write(a.out/'teacher_fit.json',dict(status='completed_raw_train16_teacher_fit',scene=m['scene'],methods=result,
        manifest_sha256=sha(a.manifest),methods_sha256=sha(a.methods),script_sha256=sha(__file__),
        gpu=torch.cuda.get_device_name(),parameter_updates=0,
        protocol='Unquantized unclamped render versus original uint8 frozen train teacher/255, equal-frame RGB L1/MSE; same fixed16. No HR or held-out teacher. This measures imitation, not detail truth.'))

if __name__=='__main__':main()
