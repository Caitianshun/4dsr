"""Single-time independent attributes; same point order, original activation."""
from types import SimpleNamespace
from context import *

class Baked(torch.nn.Module):
    def __init__(self,values,base_count,degree):
        super().__init__()
        for name,value in values.items():self.register_parameter(name,torch.nn.Parameter(value.detach().clone()))
        self.base_count=int(base_count);self.degree=int(degree)

def bake(model,time):
    g,c=model.g,model.children
    def deform(x,s,r,a,h):return g._deformation(x,s,r,a,h,torch.full((len(x),1),time,device=x.device,dtype=x.dtype))
    with torch.no_grad():
        a=deform(g._xyz,g._scaling,g._rotation,g._opacity,g.get_features);b=deform(c.xyz(),c.logscale,c.quaternion,c.opacity,c.features())
        v=[torch.cat((x,y)).detach() for x,y in zip(a,b)]
    return Baked(dict(xyz=v[0],logscale=v[1],quaternion=v[2],opacity=v[3],sh_dc=v[4][:,:1],sh_rest=v[4][:,1:]),len(g._xyz),g.active_sh_degree)

def render_baked(model,camera):
    n=model.base_count
    cov=torch.cat((motion.covariance(model.logscale[:n],model.quaternion[:n]),motion.covariance(model.logscale[n:],model.quaternion[n:])),0)
    return motion.rasterize(SimpleNamespace(active_sh_degree=model.degree),camera,model.xyz,cov,model.opacity.sigmoid(),torch.cat((model.sh_dc,model.sh_rest),1))

def state(model):
    return dict(values={n:cpu(v) for n,v in model.named_parameters()},base_count=model.base_count,degree=model.degree)

def load_baked(path):
    ck=torch.load(path,map_location='cuda',weights_only=False);x=ck['baked'];return Baked(x['values'],x['base_count'],x['degree'])
