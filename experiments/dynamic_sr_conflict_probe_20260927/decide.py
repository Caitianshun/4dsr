"""Predeclared train-only gates. This entry never imports image/evaluation code."""
import argparse
import json
from pathlib import Path
read=lambda p:json.loads(Path(p).read_text())

def decide(protocol,root,repeat_id):
    p=read(protocol);t=p['thresholds'];initial=read(root/'initial_train_eval/complete.json');assert not initial['privileged'];u=initial['values'];runs={}
    for arm in p['arms']:
        j=read(root/f'repeat{repeat_id}'/arm/'eval_train_600/complete.json');assert not j['privileged'];assert j['step']==600 and j['repeat_id']==repeat_id;runs[arm]=j['values']
    aa,bb=p['a'],p['b'];gates={};quantities={}
    def rate(name,num,den):
        value=num/den if den>t['min_denominator'] else None;quantities[name]=dict(numerator=num,denominator=den,relative=value,denominator_valid=value is not None);return value
    ea=lambda arm:runs[arm][aa]['M_H'];eb=lambda arm:runs[arm][bb]['M_H']
    ga=rate('A_half_own_gain',u[aa]['M_H']-ea('A_half'),u[aa]['M_H']);gb=rate('B_half_own_gain',u[bb]['M_H']-eb('B_half'),u[bb]['M_H'])
    gates['single_teacher_reachability']=dict(passed=ga is not None and gb is not None and ga>=t['own_H_gain'] and gb>=t['own_H_gain'],a_gain=ga,b_gain=gb,threshold=t['own_H_gain'])
    harm=rate('AB_vs_A_half_a_harm',ea('AB')-ea('A_half'),ea('A_half'));benefit=rate('AB_vs_A_half_b_gain',eb('A_half')-eb('AB'),eb('A_half'))
    gates['joint_tradeoff']=dict(passed=harm is not None and benefit is not None and harm>=t['cross_H'] and benefit>=t['cross_H'],a_harm=harm,b_gain=benefit,threshold=t['cross_H'])
    dose=rate('A_full_vs_AB_a_gain',ea('AB')-ea('A_full'),ea('AB'))
    gates['same_source_dose_control']=dict(passed=ea('A_full')<=ea('A_half') and dose is not None and dose>=t['cross_H'],A_full_a=ea('A_full'),A_half_a=ea('A_half'),AB_a=ea('AB'),A_full_vs_AB_gain=dose,threshold=t['cross_H'])
    mean=lambda x:sum(x[c]['LR'] for c in p['train_cameras'])/19;uLR=mean(u);lrs={arm:mean(v) for arm,v in runs.items()};harms={arm:rate(arm+'_LR_harm',v-uLR,uLR) for arm,v in lrs.items()};jointLR=rate('AB_vs_A_half_LR_harm',lrs['AB']-lrs['A_half'],lrs['A_half'])
    gates['LR_fidelity']=dict(passed=all(v is not None and v<=t['lr_harm'] for v in harms.values()) and jointLR is not None and jointLR<=t['lr_harm'],initial=uLR,branch_values=lrs,relative_harm=harms,AB_vs_A_half=jointLR,threshold=t['lr_harm'])
    init=read(root/'updates_initial/complete.json');final=read(root/f'updates_AB_repeat{repeat_id}/complete.json')
    gates['conditional_SH_response']=dict(passed=init['conditional_SH_signal'] or final['conditional_SH_signal'],initial=init['conditional_SH_checks'],AB600=final['conditional_SH_checks'],rule='For both epsilon1 and0.25, a masked H rises and b masked H falls beyond predeclared floor and single-repeat/zero envelope')
    return dict(status='round_decided_train_only',repeat_id=repeat_id,passed=all(x['passed'] for x in gates.values()),gates=gates,quantities=quantities,raw_anchor_values={arm:{c:v[c] for c in [aa,bb]} for arm,v in runs.items()},mask_pixels={c:p['masks'][c]['pixels_hr'] for c in [aa,bb]},read_HR=False,read_dev_scores=False,unrun_reason=[k for k,v in gates.items() if not v['passed']])

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--protocol',type=Path,required=True);pa.add_argument('--repeat-id',type=int,required=True);pa.add_argument('--out',type=Path,required=True);a=pa.parse_args();result=decide(a.protocol,a.protocol.resolve().parent,a.repeat_id);a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(dict(repeat_id=a.repeat_id,passed=result['passed'],failed=result['unrun_reason'])))
if __name__=='__main__':main()
