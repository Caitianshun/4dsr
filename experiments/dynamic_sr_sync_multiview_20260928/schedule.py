"""Original-prefix verification plus explicitly seeded three-RNG suffix."""
import random
import hashlib
from collections import Counter
from dv_common import read,sha

def plain(x):
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    return x

class Schedule:
    def __init__(self,old,manifest,mode):
        self.old,self.manifest,self.mode=old,manifest,mode
        assert mode in ['original_continuation','independent_suffix']
        self.keys=old['record_keys'];self.lookup={tuple(k):i for i,k in enumerate(self.keys)}
        self.old_ids=[i for i,(c,f) in enumerate(self.keys) if c in old['original_teacher_cameras']]
        self.cams=sorted(set(c for c,f in self.keys));self.rows=[];self.cursor=0;self.exposure=Counter();self.lr_exposure=Counter()
        self.draw,self.lr_draw,self.sr_times=[hashlib.sha256() for _ in range(3)]
        self.reset(old['seed'])
    def reset(self,seed):
        self.seed=seed;self.lr_rng,self.sr_rng,self.cam_rng=[random.Random(seed+n) for n in [177,211,313]];self.bag=[]
    def advance(self):
        if self.cursor==6000 and self.mode=='independent_suffix':self.reset(2026092801)
        li=self.lr_rng.randrange(len(self.keys));oi=self.sr_rng.choice(self.old_ids)
        if not self.bag:self.bag=list(self.cams);self.cam_rng.shuffle(self.bag)
        si=self.lookup[self.bag.pop(),self.keys[oi][1]];row=[li,si,oi]
        if self.cursor<6000 or self.mode=='original_continuation':assert row==self.old['rows'][self.cursor],self.cursor
        assert row==self.manifest['rows'][self.cursor],('registered schedule differs',self.cursor)
        self.cursor+=1;self.rows.append(row)
        self.draw.update(f'{li},{si}\n'.encode());self.lr_draw.update(f'{li}\n'.encode());self.sr_times.update(f'{self.keys[si][1]}\n'.encode())
        self.exposure[f'{self.keys[si][0]}/{self.keys[si][1]}']+=1;self.lr_exposure[f'{self.keys[li][0]}/{self.keys[li][1]}']+=1
        return li,si
    def state(self):return dict(mode=self.mode,cursor=self.cursor,lr=self.lr_rng.getstate(),sr=self.sr_rng.getstate(),camera=self.cam_rng.getstate(),camera_bag=list(self.bag),seed=self.seed,additional_exposure=dict(self.exposure),lr_exposure=dict(self.lr_exposure),draw_sha256=self.draw.hexdigest(),lr_draw_sha256=self.lr_draw.hexdigest(),sr_frame_sha256=self.sr_times.hexdigest())
    def validate_parent(self,ck):
        assert self.cursor==6000
        st=self.state()
        for k in ['lr','sr','additional_exposure']:assert plain(st[k])==plain(ck['samplers'][k]),k
        for k in ['draw_sha256','lr_draw_sha256','sr_frame_sha256']:assert st[k]==ck['metadata'][k],k

def generate(old,mode):
    keys=old['record_keys'];rows=list(old['rows'][:6000])
    if mode=='original_continuation':rows+=old['rows'][6000:12000]
    else:
        lr,sr,cam=[random.Random(2026092801+n) for n in [177,211,313]];bag=[]
        cams=sorted(set(c for c,f in keys));lookup={tuple(k):i for i,k in enumerate(keys)};ids=[i for i,(c,f) in enumerate(keys) if c in old['original_teacher_cameras']]
        for _ in range(6000):
            li=lr.randrange(len(keys));oi=sr.choice(ids)
            if not bag:bag=list(cams);cam.shuffle(bag)
            rows.append([li,lookup[bag.pop(),keys[oi][1]],oi])
    result=dict(mode=mode,seed=old['seed'] if mode=='original_continuation' else 2026092801,offsets=[177,211,313],rows=rows,record_keys=keys,row_schema=old['row_schema'],identities=[dict(step=i+1,lr=keys[li],sr=keys[si],old_B4_sr=keys[oi]) for i,(li,si,oi) in enumerate(rows)],states={})
    obj=Schedule(old,result,mode)
    for i in range(12000):
        obj.advance()
        if obj.cursor in [6000,9000,12000]:result['states'][str(obj.cursor)]=plain(obj.state())
    result['exposure']={}
    for label,column in [('lr',0),('sr',1)]:
        counts=Counter(tuple(keys[r[column]]) for r in rows[6000:])
        allcounts=[counts[tuple(k)] for k in keys]
        result['exposure'][label]=dict(by_camera=dict(Counter({c:sum(v for (cc,f),v in counts.items() if cc==c) for c in obj.cams})),by_time={str(f):sum(v for (c,ff),v in counts.items() if ff==f) for f in range(0,120,2)},by_observation={f'{c}/{f}':counts[c,f] for c,f in keys},minimum=min(allcounts),maximum=max(allcounts),zero_exposure=sum(v==0 for v in allcounts),total=sum(allcounts))
    return result
