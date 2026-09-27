"""Register the parent selected from the completed index, LR-only ROIs and budgets."""
from common_cov import *
from PIL import Image,ImageDraw
from collections import Counter

def main():
    assert not (OUT/'protocol.json').exists()
    old=ROOT/'output/dynamic_sr_conflict_probe_20260927';idx=read(old/'checkpoint_index.json');execution=read(old/'execution_index.json');assert idx['status']=='completed' and execution['status']=='completed'
    candidates=[r for r in idx['checkpoints'] if r['arm']=='AB' and r['step']==600 and r['repeat_id']==1];assert len(candidates)==1
    parent=candidates[0];assert sha(parent['path'])==parent['sha256']
    prev=read(old/'protocol.json')
    for item in prev['training_files']+list(prev['masks'].values())+[prev['manifest'],prev['teacher_index']]:assert sha(item['path'])==item['sha256']
    assert sha(prev['start'])==prev['start_sha256']
    rois={'cam02':{'upper_body':[178,133,260,190],'interaction':[166,177,245,245]},'cam03':{'upper_body':[166,134,244,190],'interaction':[158,177,232,245]}}
    definition=dict(registered_unix=time.time(),source='Only real LR cam02/cam03 frame40 viewed, before any new model training or HR pixels',coordinate_convention='LR xyxy left/top inclusive right/bottom exclusive; HR = LR * 4',rois=rois,reasons={'upper_body':'Visible apron/shirt torso with shoulders and arm context','interaction':'Both hands, utensil and pan contents with table context'},not_cross_view_correspondences=True)
    write(OUT/'roi_definition.json',definition)
    previews=[]
    for c,items in rois.items():
        entry=next(r for r in prev['training_files'] if r['camera']==c and r['reference']=='LR');im=Image.open(entry['path']).convert('RGB');draw=ImageDraw.Draw(im)
        for k,(name,xy) in enumerate(items.items()):
            x0,y0,x1,y1=xy;assert (x1-x0)*4>=64 and (y1-y0)*4>=64;draw.rectangle((x0,y0,x1-1,y1-1),outline=['cyan','yellow'][k],width=1);draw.text((x0,y0-11),name,fill=['cyan','yellow'][k])
        path=OUT/f'{c}_lr_roi.png';im.save(path);previews.append(dict(path=str(path),sha256=sha(path),LR_sha256=entry['sha256']))
    sequences={}
    for repeat,seed in [(1,2026092702),(2,2026092703)]:
        rng=random.Random(seed);seq=[]
        while len(seq)<600:
            cycle=[f'cam{i:02d}' for i in range(2,21)];rng.shuffle(cycle);seq+=cycle
        seq=seq[:600];assert set(Counter(seq).values())=={31,32};sequences[str(repeat)]=seq
    p={k:copy.deepcopy(prev[k]) for k in ['manifest','teacher_index','training_files','masks','rates','operator','physical_gpu']}
    p.update(schema=1,registered_unix=time.time(),stage='A',frame=40,time=40/300,a='cam02',b='cam03',parent=parent,parent_index=dict(path=str(old/'checkpoint_index.json'),sha256=sha(old/'checkpoint_index.json')),parent_execution=dict(path=str(old/'execution_index.json'),sha256=sha(old/'execution_index.json')),dynamic_parent=dict(path=prev['start'],sha256=prev['start_sha256']),previous_protocol=dict(path=str(old/'protocol.json'),sha256=sha(old/'protocol.json')),adopted_plan_sha256=sha(OUT/'adopted_plan.codex.md'),source_commit='765a7330a028ca342fc93c2066dd317476121f75',rois=rois,roi_definition=dict(path=str(OUT/'roi_definition.json'),sha256=sha(OUT/'roi_definition.json')),roi_previews=previews,sequences=sequences,seeds={'1':2026092702,'2':2026092703},order={'1':['C','S'],'2':['S','C']},active={'C':['sh_dc','sh_rest'],'S':['sh_dc','sh_rest','logscale','quaternion']},steps=600,save_steps=[0,300,600],optimizer_initial='empty state for every active group in both arms; no AB momentum',loss='L1(D(R_lr),LR)+0.05*L1(R_cam02,T_cam02)+0.05*L1(R_cam03,T_cam03)',thresholds=dict(min_denominator=1e-6,person_H_gain=.05,anchor_H_harm=.02,LR_harm=.02,remaining_H_harm=.02,roi_LPIPS_harm=.02,roi_PSNR_loss=.10,dev_PSNR_loss=.20,dev_LPIPS_harm=.05),budget=dict(A_formal=2400,B_formal=36000,engineering_max=32,total_updates_max=38432,total_formal_RGB_max=79200),dynamic_repeat=dict(LR_seed=2026092711,SR_seed=2026092712,preserve_parent_prefix_validation=True),information_boundary='Train-only decision frozen before new HR/dev pixel reads; ROI evaluation only',HR_ROI_quality='Crop clamped float prediction and HR; same legacy SSIM 11x11 and LPIPS Alex; H computed full image then cropped')
    write(OUT/'protocol.json',p);write(OUT/'initial_state.json',dict(**parent,source='AB600 baked effective state, no U6000 rebake'))
    write(OUT/'execution_index.json',dict(status='registered',protocol_sha256=sha(OUT/'protocol.json'),created_unix=time.time(),parent=parent,updates=0))
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.json'),parent=parent,roi_sha256=sha(OUT/'roi_definition.json')),indent=2))
if __name__=='__main__':main()
