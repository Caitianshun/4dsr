"""Frozen target provider. No 3D state, optimizer, HR or evaluation-cache access."""
from dv_common import *

ARM_TO_MODE={'Repeat7':'repeat7','Video7':'video7'}

class TemporalPrior:
    def __init__(self,mode,index_entry,records,shape,manifest,maximum=64):
        from training_support import TeacherCache
        assert mode in ['single_image','repeat7','video7']
        self.mode=mode;self.index=read(bound(index_entry));self.index_entry=index_entry
        entries=self.index['entries'];by={(e['camera'],e['frame']):e for e in entries}
        keys=[(r['camera_id'],r['frame_index']) for r in records]
        assert len(by)==len(entries)==len(keys)==1140 and set(by)==set(keys)
        if mode!='single_image':assert self.index['status']=='completed' and self.index['mode']==mode
        self.paths={}
        for i,key in enumerate(keys):
            e=by[key]
            if mode=='single_image':path=Path(manifest['_root'])/e['relative_path']
            else:
                assert e['mode']==mode and e['shape']==list(shape) and e['center_index']==3
                assert (e['sources'][3]['camera'],e['sources'][3]['frame'])==key
                path=bound(e)
            self.paths[i]=path
        self.cache=TeacherCache(self.paths,shape,maximum)

    def get(self,index):
        value=self.cache.get(index)
        assert not value.requires_grad
        return value

    def identity(self):return dict(mode=self.mode,index=self.index_entry,quantization=self.index.get('identity',{}).get('quantization','historical SwinIR PNG'))
