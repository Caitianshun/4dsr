"""Incremental current generic-SR source/input sync; verify every remote hash."""
import argparse,subprocess,shlex
from cg_common import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');a=ap.parse_args()
    if a.verify:
        inventory=read(OUT/'deployment_files.json')
        for e in inventory['files']:bound(e)
        shared,policy=setup()
        import torch
        assert str(torch.__version__)=='2.7.1+cu128'
        write(OUT/'remote_verification.json',dict(status='passed',file_count=len(inventory['files']),torch=str(torch.__version__),source_identity=source_identity()))
        return
    p=read(OUT/'protocol.json');items={}
    def add(e):items[e['path']]=e
    for e in p['training_files']:add(e)
    for key in ['parent','manifest','teacher','old_schedule','lr_curve','roi']:add(p[key])
    for e in p['schedules'].values():add(e)
    for folder in ['dynamic_sr_confidence_geometry_20261006','dynamic_sr_prior_guidance_20260927',
      'dynamic_sr_detail_supervision_20260924','dynamic_sr_motion_bound_20260923','dynamic_sr_20260918','dynamic_sr_20260920']:
        for path in (ROOT/'experiments'/folder).glob('*.py'):add(entry(path))
    for path in [OUT/'protocol.json',OUT/'calibration.json']:
        if path.exists():add(entry(path))
    # Later cache synchronization remains scoped to this experiment's frozen files.
    for folder in ['geometry','confidence_view']:
        directory=OUT/'cache'/folder
        if directory.exists():
            for path in directory.rglob('*'):
                if path.is_file() and path.suffix in ['.json','.npz']:add(entry(path))
    write(OUT/'deployment_files.json',dict(files=list(items.values()),scope='only current generic source, legal train LR/SR, parent, protocol and frozen legal caches'))
    filelist=OUT/'deployment_files.txt';filelist.write_text('\n'.join(sorted(items))+f'\n{(OUT/"deployment_files.json").relative_to(ROOT)}\n')
    remote='/home/ubuntu/3DGS/4dsr';host='a100-train'
    subprocess.run(['rsync','-az','--files-from='+str(filelist),str(ROOT)+'/',host+':'+remote+'/'],check=True)
    command=f'cd {remote} && source activate_a100.sh && python experiments/{HERE.name}/deploy.py --verify'
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',host,command],check=True)
    subprocess.run(['rsync','-az',host+':'+remote+'/output/'+HERE.name+'/remote_verification.json',str(OUT)+'/'],check=True)
    print(json.dumps(read(OUT/'remote_verification.json')))
if __name__=='__main__':main()
