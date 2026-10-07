"""Pre-registered content-aware permutations, independent of training RNG."""
import random,hashlib
from collections import Counter
from cg_common import read
def plain(v):
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    return v
def validate_parent(old,ck):
    keys=old['record_keys'];ids=[i for i,(c,f) in enumerate(keys) if c in old['original_teacher_cameras']]
    lr,sr,cam=[random.Random(old['seed']+n) for n in [177,211,313]]
    lookup={tuple(k):i for i,k in enumerate(keys)};cams=sorted(set(c for c,f in keys));bag=[]
    draw,ld,st=[hashlib.sha256() for _ in range(3)];ex=Counter()
    for k in range(6000):
        li=lr.randrange(len(keys));oi=sr.choice(ids)
        if not bag:bag=list(cams);cam.shuffle(bag)
        si=lookup[bag.pop(),keys[oi][1]]
        assert old['rows'][k]==[li,si,oi]
        draw.update(f'{li},{si}\n'.encode());ld.update(f'{li}\n'.encode());st.update(f'{keys[si][1]}\n'.encode())
        ex[f'{keys[si][0]}/{keys[si][1]}']+=1
    for k,v in [('lr',lr.getstate()),('sr',sr.getstate()),('additional_exposure',dict(ex))]:
        assert plain(v)==plain(ck['samplers'][k]),k
    for k,v in [('draw_sha256',draw.hexdigest()),('lr_draw_sha256',ld.hexdigest()),('sr_frame_sha256',st.hexdigest())]:
        assert v==ck['metadata'][k],k
    return dict(passed=True,native_prefix_steps=6000,new_suffix_is_native_continuation=False)
def generate(keys,seed):
    rng=random.Random(seed);base=[rng.randrange(len(keys)) for _ in range(6000)]
    rows=[];attempts=[]
    for start in range(0,6000,100):
        src=base[start:start+100];perms=[]
        for slot in range(2):
            best=None;bestn=101
            for attempt in range(50):
                p=list(range(100));rng.shuffle(p)
                same=sum(keys[src[i]]==keys[src[j]] for i,j in enumerate(p))
                if same<bestn:best,bestn=p[:],same
                if same==0:break
            perms.append(best);attempts.append(dict(window=start//100,slot=slot,attempts=attempt+1,remaining_same_observation=bestn))
        rows += [[src[i],src[perms[0][i]],src[perms[1][i]]] for i in range(100)]
    for a,b in [(0,3000),(0,6000)]+[(i,i+100) for i in range(0,6000,100)]:
        cs=[Counter(r[j] for r in rows[a:b]) for j in range(3)]
        assert cs[0]==cs[1]==cs[2]
    stats={}
    for j in [1,2]:
        stats[str(j)]={k:sum(test(keys[r[0]],keys[r[j]]) for r in rows)/6000 for k,test in
          [('same_observation',lambda a,b:a==b),('same_camera',lambda a,b:a[0]==b[0]),('same_time',lambda a,b:a[1]==b[1])]}
    return dict(seed=seed,record_keys=keys,rows=rows,window=100,attempt_limit=50,attempts=attempts,
      audit=dict(passed=True,exposure_equal_each_window_and_3000_and_6000=True,statistics=stats),
      definition='Rows LR, permutedSR_A, permutedSR_B; B2 coefficients .05+.05, others .1')
def slots(method,row):
    if method in ['P_joint','P_xyz','P_SH']:return []
    if method=='Jperm':return [(row[1],.1)]
    if method=='B2perm':return [(row[1],.05),(row[2],.05)]
    return [(row[0],.1)]
