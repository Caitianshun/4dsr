"""Hash pinned upstream checkouts or relocatable overlay package files; no network."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REMOTE=Path('/home/cts/Project/4DSR/tasks/static_sr_wu_20261022')
REPOS=('splatsure','splatsure-rasterizer','iesrgs','sequence_matters','threedsr','stablesr','taming-transformers','clip')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def tracked_paths(base):
 """Include author-tracked artifacts even when their names resemble caches."""
 result=subprocess.run(['git','-C',str(base),'ls-files','--stage','--cached','--recurse-submodules','-z'],
                       check=True,capture_output=True)
 paths=set();gitlinks=[]
 for record in result.stdout.split(b'\0'):
  if not record:continue
  metadata,raw=record.split(b'\t',1);mode,commit,stage=metadata.split()
  path=base/os.fsdecode(raw)
  if stage!=b'0':raise ValueError('Unmerged Git index cannot be transported: '+str(path))
  if mode==b'160000':
   if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
    raise ValueError('Uninitialized Git link must remain an existing empty directory: '+str(path))
   gitlinks.append(dict(path=path,gitlink_commit=commit.decode('ascii')));continue
  if not path.is_file() and not path.is_symlink():raise FileNotFoundError('Missing Git-tracked file: '+str(path))
  paths.add(path)
 return paths,gitlinks
def inventory_paths(base,upstream=False):
 """Untracked/overlay files retain the original filter; tracked files override it."""
 tracked,gitlinks=tracked_paths(base) if upstream else (set(),[])
 selected=set(tracked)
 for parent,dirs,names in os.walk(base,followlinks=False):
  dirs[:]=[n for n in dirs if n not in ('__pycache__','build','dist')]
  for name in names:
   if name.endswith(('.pyc','.lock')) or name in ('reuse_wu_torch.pth','taming_pinned_namespace.pth'):continue
   selected.add(Path(parent)/name)
 return sorted(selected),tracked,gitlinks
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['upstream','overlays'],required=True);p.add_argument('--out',required=True);p.add_argument('--remote-root',default=str(REMOTE));a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);dest=Path(a.remote_root);files=[];links=[];empty_dirs=[]
 roots=[ROOT/'tmp/static_sr_wu_20261022/upstream'/n for n in REPOS] if a.stage=='upstream' else [ROOT/'output/static_sr_wu_20261022/runtime'/n/'lib/python3.10/site-packages' for n in ('stablesr_env','sequence_env')]
 for base in roots:
  paths,tracked,gitlinks=inventory_paths(base,a.stage=='upstream')
  for entry in gitlinks:
   rel=str(entry['path'].relative_to(ROOT))
   empty_dirs.append(dict(path=rel,gitlink_commit=entry['gitlink_commit'],destination=str(dest/rel)))
  for path in paths:
    rel=str(path.relative_to(ROOT))
    if path.is_symlink():
     links.append(dict(path=rel,target=os.readlink(path),git_tracked=path in tracked));continue
    if not path.is_file():continue
    files.append(dict(path=rel,bytes=path.stat().st_size,sha256=sha(path),destination=str(dest/rel),git_tracked=path in tracked))
 # Keep symlink materialization explicit: existing transport supports files only.
 record=dict(schema='static_sr_cts_transport_inventory_v1',stage=a.stage,source_root=str(ROOT),destination_root=str(dest),files=sorted(files,key=lambda x:x['path']),symlinks=links,empty_dirs=empty_dirs,total_bytes=sum(v['bytes'] for v in files),remote_checked=False,GPU_calls=0,
    transport_blockers=['Symlink materialization is unsupported by sync_assets.py; do not omit these members'] if links else [],
    instructions='Preserve all Git-tracked files including author pyc/build/binary assets, relative .git files and Git metadata. Preserve uninitialized Git links as empty directories at their registered commits; never fetch submodules. Symlink inventories require separate reviewed support and are rejected by sync_assets.py.' if a.stage=='upstream' else 'Only site-packages are transported. bootstrap_remote.py creates new venv launchers and CTS-only .pth; never copy local bin/pyvenv.cfg.')
 (out/'inventory.json').write_text(json.dumps(record,indent=2)+'\n');(out/'files.txt').write_text(''.join(v['path']+'\n' for v in record['files']))
 print(json.dumps(dict(stage=a.stage,files=len(files),symlinks=len(links),empty_dirs=len(empty_dirs),bytes=record['total_bytes'],out=str(out))))
if __name__=='__main__':main()
