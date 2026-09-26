"""Guard stop precedence and jointly required quality gates."""
from copy import deepcopy
from summarize import decide

def main():
    early={c:dict(psnr=30.,ssim=.9,lpips=.12,dynamic_lpips=.15,static_lpips=.11,temporal=.01,lr_psnr=33.,lr_l1=.01) for c in ['cam00','cam01']}
    c=deepcopy(early);c['cam00']['psnr']-=.3
    f=deepcopy(c);f['cam00']['psnr']+=.15
    assert decide({},c,f,early,18000)['matched_rule']==2
    f['cam01']['dynamic_lpips']*=1.07
    assert decide({},c,f,early,18000)['matched_rule']==1
    c=deepcopy(early);f=deepcopy(c)
    assert decide({},c,f,early,18000)['matched_rule']==3
    c['cam00']['psnr']-=.3;f=deepcopy(c)
    assert decide({},c,f,early,18000)['matched_rule']==4
    f['cam00']['psnr']+=.15;f['cam01']['lpips']+=.01
    assert not decide({},c,f,early,18000)['extend']
    print('five endpoint gate cases passed')

if __name__=='__main__':main()
