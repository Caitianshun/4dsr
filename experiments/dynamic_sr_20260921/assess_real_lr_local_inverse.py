"""Post-hoc audit and frozen-teacher completion, never tunes the inverse solver."""
from pathlib import Path
import importlib.util,json,hashlib,time
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('inverse',Path(__file__).with_name('real_lr_local_inverse.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
BASE=ROOT/'output/dynamic_sr_20260921'
OUT=BASE/'real_lr_local_inverse_assessment_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=False);(OUT/'source.py').write_text(Path(__file__).read_text())
    inputs={};results={};panels={}
    for version,directory in [('center_valid','real_lr_local_inverse_v1'),('whole_support_valid','real_lr_local_inverse_strict_footprint_v1')]:
        folder=BASE/directory;mp=folder/'metrics.json';d=json.loads(mp.read_text());inputs[str(mp)]=sha(mp)
        assert json.loads((folder/'complete.json').read_text())['metrics_sha256']==sha(mp)
        assert sha(folder/'source.py')==d['identity']['source_sha256']
        for inp,h in d['input_sha256'].items():assert sha(inp)==h
        for i,r in enumerate(d['rows']):
            pp=folder/f'patch_{i:02d}_solved_before_hr.npz';assert sha(pp)==d['records'][i]['prediction_sha256_before_hr']
            stored=np.load(folder/f'patch_{i:02d}_evaluation.npz');c=r['cell'];gt=stored['hr_evaluation_only'];x,y=c['x_lr']*4,c['y_lr']*4
            if c['scene']=='meetroom_discussion' and c['camera']=='cam06':
                t=BASE/f'local_inverse_teacher_v1/cam06_{c["anchor"]:04d}.png';inputs[str(t)]=sha(t);teacher=mod.rgb(t)[y-48:y+48,x-48:x+48]
            else:
                teacher=stored['swinir_existing']
            r['methods']['swinir_all']=mod.measure(teacher,gt)
            if version=='whole_support_valid':panels[i]=(dict(stored),teacher)
        scenes={}
        for scene in mod.SCENES:
            rr=[r for r in d['rows'] if r['cell']['scene']==scene];rec=[r for r in d['records'] if r['cell']['scene']==scene]
            names=[m for m in rr[0]['methods'] if m!='swinir_existing']
            means={m:{k:float(np.mean([r['methods'][m][k] for r in rr])) for k in ['psnr','ssim','mse','high_residual_mse','high_residual_error_ratio','clip_fraction']} for m in names}
            comparisons={}
            for lam in mod.LAMBDAS:
                m=f'multiframe_lambda_{lam:g}';s=f'reference_lambda_{lam:g}'
                dd={k:np.array([r['methods'][m][k]-r['methods'][s][k] for r in rr]) for k in ['psnr','ssim','high_residual_mse']}
                comparisons[str(lam)]={k:dict(mean=float(v.mean()),positive_count=int((v>0).sum()),negative_count=int((v<0).sum()),values=v.tolist()) for k,v in dd.items()}
            ref=[r['reference_hr_forward_only_evaluation'][0]['rmse'] for r in rr];neighbors=[e['rmse'] for r in rr for e in r['reference_hr_forward_only_evaluation'][1:]]
            scenes[scene]=dict(n=8,means=means,multiframe_minus_same_solver_reference=comparisons,actual_lr_rows=[r['total_actual_rows'] for r in rec],actual_frame_counts=[r['unique_actual_frames'] for r in rec],reference_hr_forward_rmse=dict(reference_median=float(np.median(ref)),neighbor_median=float(np.median(neighbors)),reference_range=[min(ref),max(ref)],neighbor_range=[min(neighbors),max(neighbors)]))
        results[version]=dict(source_directory=str(folder),elapsed_seconds=d['identity']['elapsed_seconds'],scenes=scenes,rows=d['rows'],repeat_solution_max_abs=max(v for r in d['records'] for v in r['repeat_solution_max_abs'].values()),operator_parity_max_abs=max(o['parity_max_abs'] for r in d['records'] for o in r['operators']),cg_max_relative_residual=max(v['relative_normal_equation_residual'] for r in d['records'] for lam in r['solver'].values() for controls in lam.values() for v in controls))
    # A panel for each domain with all fixed locations, no cherry-picked crops.
    for scene in mod.SCENES:
        rows=[r for r in results['whole_support_valid']['rows'] if r['cell']['scene']==scene]
        canvas=Image.new('RGB',(5*192,8*216),'white');draw=ImageDraw.Draw(canvas)
        for ri,r in enumerate(rows):
            stored,teacher=panels[r['index']];c=r['cell']
            ims=[('HR evaluation',stored['hr_evaluation_only']),('Bicubic',stored['bicubic']),('Single inverse .1',stored['reference_lambda_0.1']),('Multi inverse .1',stored['multiframe_lambda_0.1']),('Frozen SwinIR',teacher)]
            for j,(name,im) in enumerate(ims):
                draw.text((j*192+2,ri*216),name,fill='black');canvas.paste(Image.fromarray(np.round(np.clip(im[24:72,24:72],0,1)*255).astype(np.uint8)).resize((192,192),Image.Resampling.NEAREST),(j*192,ri*216+18))
            draw.text((0,ri*216+202),f'{c["camera"]} t={c["anchor"]} LR({c["x_lr"]},{c["y_lr"]})',fill='black')
        canvas.save(OUT/f'{scene}_all_fixed_patches.png')
    payload=dict(inputs=inputs,source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started,results=results,lpips='omitted: 48x48 crop smaller than useful Alex/VGG context; no perceptual full-frame conclusion')
    (OUT/'metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2));(OUT/'complete.json').write_text(json.dumps(dict(metrics_sha256=sha(OUT/'metrics.json'),source_sha256=sha(__file__)),indent=2))
    print(json.dumps({v:{s:x['means'] for s,x in d['scenes'].items()} for v,d in results.items()},indent=2))
if __name__=='__main__':main()
