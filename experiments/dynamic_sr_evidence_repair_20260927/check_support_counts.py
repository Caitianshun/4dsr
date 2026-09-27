"""Audit retained-track counts and make direct-pixel support version explicit."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from audit_depth_evidence import OUT,read,write,project

def main():
 p=read(OUT/'protocol.json');m=read(p['manifest']['path']);tracks=read(OUT/'support_v1/tracks.json');full=np.load(OUT/'support_v1/full_pair_support.npz');counts=[];direct={}
 for c in p['train_cameras']:
  xy,z=project(full['xyz'],m['cameras'][c])
  for f in p['frames']:
   ts=[t for t in tracks if t['frame']==f and c in t['projections']];a=np.array([t['projections'][c]['projected_xy'] for t in ts]).reshape(-1,2);d=np.array([t['projections'][c]['z'] for t in ts]);u=np.array([t['projections'][c]['z_uncertainty'] if t['projections'][c]['z_uncertainty'] is not None else np.inf for t in ts]);ok=np.isfinite(a).all(1)&(a[:,0]>=16)&(a[:,0]<320)&(a[:,1]>=16)&(a[:,1]<236)&(d>0);a,d,u=a[ok],d[ok],u[ok]
   used=set();npairs=0
   for i,j in cKDTree(a).query_pairs(40):
    diff=abs(d[i]-d[j]);distance=np.linalg.norm(a[i]-a[j])
    if distance>=4 and diff/max(d[i],d[j])>=.03 and diff>u[i]+u[j]:used.update([i,j]);npairs+=1
   counts.append(dict(camera=c,frame=f,pre_filter_tracks=len(a),participating_filtered_tracks=len(used),filtered_pairs=npairs))
   src=np.array([t['projections'][c]['projected_xy'] for t in ts if c in t['source_pair']]).reshape(-1,2)
   okpair=full['passes_spatial_filter']&(full['frames']==f)&(full['camera_pair']==int(c[3:])).any(1)&(z>0)&np.isfinite(xy).all(1)&(xy[:,0]>=16)&(xy[:,0]<320)&(xy[:,1]>=16)&(xy[:,1]<236)
   def cover(points):
    v=points[np.isfinite(points).all(1)&(points[:,0]>=0)&(points[:,0]<336)&(points[:,1]>=0)&(points[:,1]<252)];ij=np.floor(v).astype(int);return len(set(map(tuple,ij)))/(252*336)
   direct[c,f]={'pairs':cover(xy[okpair]),'sources':cover(src),'third':cover(a)}
 eligible=[r for r in counts if r['filtered_pairs']>=30 and r['pre_filter_tracks']>=10];assert all(r['participating_filtered_tracks']>=10 for r in eligible)
 write(OUT/'rank_support_check.json',dict(status='passed',minimum_participating_tracks=min(r['participating_filtered_tracks'] for r in eligible),eligible_observations=len(eligible),rows=counts))
 path=OUT/'support_funnel.csv';backup=OUT/'depth_evidence_v1/support_funnel_original.csv'
 if not backup.exists():shutil.copyfile(path,backup)
 with backup.open() as f:rows=list(csv.DictReader(f))
 for r in rows:
  v=direct[r['camera'],int(r['frame'])];kind='sources' if r['version']=='dedup_sources_area' else 'third' if r['version']=='third_verified_area' else 'pairs'
  r['third_verified_direct_support_pixel_fraction']=r['direct_support_pixel_fraction'];r['direct_support_pixel_fraction']=v[kind];r['direct_support_version']=kind
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write(OUT/'support_funnel_correction.json',dict(status='completed',change='Original direct_support_pixel_fraction used third-verified pixels for every version. Corrected to each version own points and retained explicit third-verified column. Selected block fractions, W, ordering and every gate unchanged.',original=str(backup)))
if __name__=='__main__':main()
