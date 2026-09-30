"""Source-image edge location and LR-only cross-scale gradient diagnostics."""
from image_prior_common import *

def magnitude(im):
    gray=im.mean(2);gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8;return np.sqrt(gx*gx+gy*gy),np.stack([gx,gy],-1)

def run():
    m,obs,keys=setup();index=read(OUT/'texture_input_index.json');rows=[];start=time.time()
    for entry in index:
        c,f=entry['camera'],entry['frame'];hr=imfloat(image(obs,c,f,'hr'));lr=imfloat(image(obs,c,f));gm,gg=magnitude(hr);ge=gm>=np.quantile(gm,.9);gd=cv2.distanceTransform((~ge).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);sources={'B':imfloat(cv2.resize(image(obs,c,f),(1344,1008),interpolation=cv2.INTER_CUBIC))}
        for b in ['SwinIR','Video7']:
            p=Path(entry[b+'_path']);assert sha(p)==entry[b+'_sha256'];sources[b]=imfloat(cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB))
        for b,im in sources.items():
            am,ag=magnitude(im);ae=am>=np.quantile(am,.9);ad=cv2.distanceTransform((~ae).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);p=float((gd[ae]<=2).mean());r=float((ad[ge]<=2).mean());rows.append(dict(prior='texture_edge_location',camera=c,frame=f,branch=b,top_decile_edge_F1_2HRpx=2*p*r/max(p+r,1e-12),symmetric_edge_distance_HRpx=float((gd[ae].mean()+ad[ge].mean())/2)))
        lmag,lg=magnitude(lr);small=cv2.resize(lr,(168,126),interpolation=cv2.INTER_AREA);up=cv2.resize(small,(336,252),interpolation=cv2.INTER_CUBIC);umag,ug=magnitude(up);edge=lmag>=np.quantile(lmag,.75);cos=(lg*ug).sum(-1)/np.maximum(lmag*umag,1e-9);rows.append(dict(prior='LR_internal_crossscale',camera=c,frame=f,branch='native_LR',gradient_direction_cosine=float(cos[edge].mean()),gradient_energy_retained=float((umag[edge]**2).sum()/max((lmag[edge]**2).sum(),1e-12))))
    csvwrite(OUT/'texture_edge_localization.csv',rows);summary=dict(status='completed',rows=len(rows),seconds=time.time()-start,definition='Edge location: top10% Sobel RGB-mean gradient magnitude positions independently in source andHR, matching tolerance2HRpx, symmetric Euclidean distance. Edge ranking is location-only; original gradient energy intexture_comparison must be read alongside. LR crossscale: area down2, cubic up2, direction on originalLR top25% gradients. No HR-trained labels.',aggregate={b:{key:stats([r[key] for r in rows if r['branch']==b]) for key in ['top_decile_edge_F1_2HRpx','symmetric_edge_distance_HRpx']} for b in ['B','SwinIR','Video7']},LR_crossscale={key:stats([r[key] for r in rows if r['prior']=='LR_internal_crossscale']) for key in ['gradient_direction_cosine','gradient_energy_retained']},source_sha256=sha(__file__))
    write(OUT/'texture_edges_summary.json',summary)
    s=read(OUT/'summary.json');s['texture']['edge_localization']=summary;write(OUT/'summary.json',s)
    with (OUT/'image_prior_audit.csv').open() as f:old=[r for r in csv.DictReader(f) if r['prior'] not in ['texture_edge_location','LR_internal_crossscale']]
    csvwrite(OUT/'image_prior_audit.csv',old+rows);print('texture_edges_complete',len(rows),time.time()-start,flush=True)
if __name__=='__main__':run()
