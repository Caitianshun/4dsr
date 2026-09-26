"""Export U6000 legal moment evidence or privileged U/O observable comparisons."""
import argparse
import os
import time
import numpy as np
from shared import *
from gradient_policy import effective_state
from moment_renderer import render_moments
from utils.sh_utils import eval_sh


def main():
    p=argparse.ArgumentParser()
    for k in ['manifest','checkpoint','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--oracle',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            s=os.fsdecode(args[0]);assert not (s.endswith('.png') and '/hr/' in s),'No HR image input needed for attribute export'
    sys.addaudithook(guard)
    m=load_manifest(a.manifest);ev=evaluator();model=load_model(a.checkpoint,m);other=load_model(a.oracle,m) if a.oracle else None
    if other:assert topology(model)==topology(other),'Point correspondence not established'
    cameras=['cam02','cam06','cam12','cam18'] if other else m['splits']['train']
    observations=[o for o in m['observations'] if o['camera_id'] in cameras and o['frame_index'] in [0,40,80,118]]
    rows=[];started=time.time()
    @torch.no_grad()
    def export(mod,cam):
        s=effective_state(mod,cam.time);mom=render_moments(cam,s['xyz'],s['cov'],s['opacity'],(252,336))
        result={k:mom[k].cpu().numpy() for k in ['alpha','expected_z','variance_z']}
        if other:
            result['rgb']=torch.nn.functional.interpolate(render_model(mod,cam)['render'][None],size=(252,336),mode='bicubic',align_corners=False,antialias=True)[0].cpu().numpy()
        return result,s
    with torch.inference_mode():
        for o in observations:
            cam=ev.render_camera(m,o,0);data,s=export(model,cam)
            if other:
                od,other_state=export(other,cam);wrongcam=ev.render_camera(m,o,0);wrongcam.time=((o['frame_index']+40)%120)/300
                wrong,_=export(other,wrongcam)
                vals={prefix+'_'+k:v for prefix,d in [('U',data),('O',od),('O_wrong_time',wrong)] for k,v in d.items()}
                eig=torch.linalg.eigvalsh(s['cov']).clamp_min(1e-12);oeig=torch.linalg.eigvalsh(other_state['cov']).clamp_min(1e-12)
                dirs=torch.nn.functional.normalize(s['xyz']-cam.camera_center.cuda(),dim=-1);odirs=torch.nn.functional.normalize(other_state['xyz']-cam.camera_center.cuda(),dim=-1)
                color=(eval_sh(model.g.active_sh_degree,s['sh'].transpose(1,2),dirs)+.5).clamp_min(0)
                ocolor=(eval_sh(other.g.active_sh_degree,other_state['sh'].transpose(1,2),odirs)+.5).clamp_min(0)
                attrs=dict(cov_log_eigen_abs_median=float((eig.log()-oeig.log()).abs().median()),view_color_abs_median=float((color-ocolor).abs().median()),effective_center_distance_median=float((s['xyz']-other_state['xyz']).norm(dim=-1).median()),point_stat_scope='same-ID whole stored topology; NOT visibility-filtered truth')
            else:vals=data;attrs={}
            path=a.out/f"{o['camera_id']}_{o['frame_index']:04d}.npz";np.savez_compressed(path,**vals)
            rows.append(dict(camera=o['camera_id'],frame=o['frame_index'],path=path.name,sha256=sha256(path),attributes=attrs))
    write_json(a.out/'complete.json',dict(status='completed',rows=rows,checkpoint_sha256=sha256(a.checkpoint),oracle_sha256=sha256(a.oracle) if a.oracle else None,topology=topology(model),mode='privileged_UO' if other else 'legal_U6000_moments',gpu=torch.cuda.get_device_name(),seconds=time.time()-started,no_new_training=True,independent_seed_reference_available=False,interpretation='O is same-start privileged supervision continuation, not an independently converged HR upper bound. Rendered moments share opacity/geometry ambiguity. Parent IDs were not treated as base-row indices.'))


if __name__=='__main__':main()
