"""Protocol-driven quality decision. Teacher diagnostics never enter acceptance."""
import math
from dv_common import read,sha,write

def get(x,path):
    for k in path.split('.'):x=x[k]
    return x

def measure(candidate,reference,rule,protocol):
    try:
        c=float(get(candidate,rule['path']));r=float(get(reference,rule['path']))
        if not math.isfinite(c) or not math.isfinite(r):raise ValueError('nonfinite')
        absolute=c-r
        v=absolute
        if rule['change']=='relative':
            if r<protocol['minimum_denominator']:raise ValueError('denominator below protocol minimum')
            v=absolute/r
        t=rule['threshold'];tol=protocol['comparison_roundoff_tolerance']
        passed=(v<=t or abs(v-t)<=tol) if rule['operator']=='le' else (v>=t or abs(v-t)<=tol)
        return dict(rule,reference_role=rule['reference'],candidate=c,reference=r,denominator=r,absolute_delta=absolute,change_value=v,passed=passed,determinate=True)
    except (KeyError,TypeError,ValueError) as e:return dict(**rule,passed=False,determinate=False,error=str(e))

def compare(candidate,reference,u,protocol,include_u=True):
    modes={k:[measure(candidate,reference,r,protocol) for r in rules] for k,rules in protocol['benefit_modes'].items()}
    gates=[measure(candidate,reference,r,protocol) for r in protocol['guardrails_vs_reference']]
    if include_u:gates.extend(measure(candidate,u,r,protocol) for r in protocol['guardrails_vs_U6000'])
    passed_modes=[k for k,rr in modes.items() if all(x['passed'] for x in rr)]
    observations=[measure(candidate,reference,r,protocol) for r in protocol['observation_lines']]
    return dict(modes=passed_modes,mode_checks=modes,guardrails=gates,all_guardrails=all(x['passed'] for x in gates),eligible=bool(passed_modes) and all(x['passed'] for x in gates),observation_lines=observations,quality_tradeoff=any(x['determinate'] and not x['passed'] for x in observations))

def decide_set(endpoints,p):
    reference=p['comparison']['reference'];u=endpoints[p['comparison']['early_reference']]
    candidates={c:compare(endpoints[c],endpoints[reference],u,p) for c in p['comparison']['candidates']}
    a,b=p['comparison']['extra_path']
    extra=compare(endpoints[a],endpoints[b],u,p,False)
    eligible=[k for k,v in candidates.items() if v['eligible']]
    return dict(status='frozen',eligible_candidates=eligible,repeat_allowed=bool(eligible),candidates=candidates,modes_vs_J={k:v['modes'] for k,v in candidates.items()},S_vs_A=extra,modes_SA=extra['modes'],effective_rules=p)

def finalize(first,second,p):
    confirmed={}
    if second:
        for c in first['eligible_candidates']:
            x,y=first['candidates'][c],second['candidates'][c];common=sorted(set(x['modes'])&set(y['modes']))
            if y['eligible'] and common:confirmed[c]=dict(modes=common,quality_tradeoff=x['quality_tradeoff'] or y['quality_tradeoff'])
    a,b=p['comparison']['extra_path'];extra_common=[]
    if a in confirmed and b in confirmed:
        extra_common=sorted(set(first['modes_SA'])&set(second['modes_SA']))
        extra_ok=bool(extra_common) and first['S_vs_A']['all_guardrails'] and second['S_vs_A']['all_guardrails']
        selected=a if extra_ok else b
    elif confirmed:selected=next(iter(confirmed))
    else:selected=p['comparison']['fallback']
    return dict(status='completed',classification='repeated_support' if confirmed else 'unconfirmed',confirmed=confirmed,selected=selected,modes_SA_common=extra_common,scope='Shared-parent, different-suffix repeat; development views, not independent training or statistical significance.')
