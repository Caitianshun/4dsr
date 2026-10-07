"""Local CPU contracts with 15-byte fixtures; never read actual transfer blobs."""
import ast,copy,hashlib,json,os,time
from pathlib import Path
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Use new CUDA_VISIBLE_DEVICES empty CPU process')
import torch
assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
import preparation_transfer_reuse as helper
from fp_common import ROOT,HERE,OUT,entry,write,read,sha


def main():
    started=time.monotonic();target=OUT/'operator_checks/preparation_transfer_reuse'/str(time.time_ns());target.mkdir(parents=True)
    # Actual manifest shape/scalars only. No real .npz/.pt/.py transfer is read.
    real=read(OUT/'completion/incoming/preparation_1791383090008639557/manifest.json')
    proto=target/'protocol.json';proto.write_bytes(b'123456789abcdef');protocol=dict(**entry(proto),bytes=15)
    rootreg=entry(OUT/'advance/root_registration_v1.json');workspace=real['workspace'];incoming=target/'incoming';incoming.mkdir()
    source=incoming/'preparation_1';source.mkdir();good=b'abcdefghijklmno';passed=[]
    def item(path,body=good,role='artifact'):return dict(path=path,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),role=role)
    names={name:str((target/name).relative_to(ROOT)) for name in ('good.npz','point.pt','source.py','bad.npz','partial.npz','symlink.npz','mutable.json')}
    entries=[dict(path=protocol['path'],bytes=15,sha256=protocol['sha256'],role='artifact')]+[item(names[n],role='source' if n=='source.py' else 'artifact') for n in names]
    def manifest(stage,rows=entries):
        value=copy.deepcopy(real);value['entries']=copy.deepcopy(rows);value['source_identities']=[dict(path=names['source.py'],sha256=hashlib.sha256(good).hexdigest(),role='source')]
        write(stage/'manifest.json',value);return value
    manifest(source)
    def put(relative,body):
        p=source/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body);return p
    oldgood=put(Path('files')/Path(names['good.npz']).parent/'.completion-rsync-partial'/Path(names['good.npz']).name,good)
    put(Path('files')/names['point.pt'],good);put(Path('files')/Path(names['source.py']).parent/'.~tmp~'/Path(names['source.py']).name,good)
    put(Path('files')/names['bad.npz'],b'XXXXXXXXXXXXXXX');put(Path('files')/names['partial.npz'],good[:7]);put(Path('files')/names['mutable.json'],good)
    sym=source/'files'/names['symlink.npz'];sym.parent.mkdir(parents=True,exist_ok=True);sym.symlink_to(oldgood)
    original=oldgood.read_bytes();original_inode=oldgood.stat().st_ino;original_mode=oldgood.stat().st_mode
    reuse=helper.Reuse(target/'cache',workspace,protocol,rootreg);reuse.build(source)
    assert set(reuse.index)=={names['good.npz'],names['point.pt'],names['source.py']};passed.append('only_complete_SHA_matching_npz_pt_py_cached_including_both_rsync_partial_locations')
    assert oldgood.read_bytes()==original and oldgood.stat().st_ino==original_inode and oldgood.stat().st_mode==original_mode;passed.append('old_stage_bytes_inode_permissions_retained_no_move_delete_or_chmod')
    blob=reuse.blobs/reuse.index[names['good.npz']]['blob'];assert blob.stat().st_ino!=original_inode and blob.stat().st_mode&0o222==0;passed.append('independent_readonly_cache_copy_not_original_stage_hardlink')
    stage=incoming/'preparation_2';stage.mkdir();manifest(stage);receipt=read(reuse.seed(stage));dest=stage/'files'/names['good.npz']
    assert dest.read_bytes()==good and dest.stat().st_ino==blob.stat().st_ino and len([r for r in receipt['rows'] if r['status']=='seeded_verified_immutable_hardlink'])==3;passed.append('exact_new_manifest_path_bytes_SHA_seeded_by_hardlink')
    assert not (stage/'files'/names['mutable.json']).exists() and not receipt['preparation_gate_published'];passed.append('mutable_JSON_logs_and_completion_gates_not_seeded_or_published')
    assert reuse.seed(stage) is None;passed.append('duplicate_manifest_event_idempotent_no_repeat_seed')
    conflict=incoming/'preparation_3';conflict.mkdir();manifest(conflict);existing=conflict/'files'/names['good.npz'];existing.parent.mkdir(parents=True,exist_ok=True);existing.write_bytes(b'preserve-me')
    r=read(reuse.seed(conflict));assert existing.read_bytes()==b'preserve-me' and any(x['status']=='skipped_existing_destination_never_overwritten' for x in r['rows']);passed.append('existing_destination_never_overwritten_even_if_wrong_bytes')
    linkstage=incoming/'preparation_4';linkstage.mkdir();manifest(linkstage);link=linkstage/'files'/names['good.npz'];link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(oldgood);reuse.seed(linkstage)
    assert link.is_symlink() and link.resolve()==oldgood;passed.append('existing_destination_symlink_preserved_never_followed_or_overwritten')
    outside=target/'outside';outside.mkdir();ancestor=incoming/'preparation_5';ancestor.mkdir();manifest(ancestor);(ancestor/'files').symlink_to(outside,target_is_directory=True);r=read(reuse.seed(ancestor))
    assert not list(outside.iterdir()) and all(x['status']=='not_seeded_error_saved' for x in r['rows']);passed.append('symlink_destination_parent_refused_using_openat_O_NOFOLLOW')
    def refuses(value,label):
        s=incoming/label;s.mkdir();write(s/'manifest.json',value)
        try:reuse.seed(s)
        except ValueError:return
        raise AssertionError('Bad manifest accepted')
    v=manifest(incoming/'preparation_2');v['workspace']='/different/workspace';refuses(v,'preparation_6');passed.append('different_workspace_manifest_refused')
    v=copy.deepcopy(v);v['workspace']=workspace;v['purpose']='task';refuses(v,'preparation_7');passed.append('task_manifest_cannot_seed_preparation')
    v=manifest(stage);v['entries'][0]['sha256']='0'*64;refuses(v,'preparation_8');passed.append('registered_protocol_identity_mismatch_refused')
    v=manifest(stage);v['entries'].append(copy.deepcopy(v['entries'][1]));refuses(v,'preparation_9');passed.append('duplicate_manifest_relative_path_refused')
    v=manifest(stage);v['entries'][1]['path']='../escape.npz';refuses(v,'preparation_10');passed.append('path_traversal_refused')
    changed=incoming/'preparation_11';changed.mkdir();v=manifest(changed);v['entries'][1]['sha256']='f'*64;write(changed/'manifest.json',v);r=read(reuse.seed(changed))
    assert not (changed/'files'/names['good.npz']).exists();passed.append('same_path_different_SHA_never_reused')
    corrupt=incoming/'preparation_12';corrupt.mkdir();manifest(corrupt);os.chmod(blob,0o644);blob.write_bytes(b'XXXXXXXXXXXXXXX');os.chmod(blob,0o444);r=read(reuse.seed(corrupt))
    assert not (corrupt/'files'/names['good.npz']).exists() and all(x['status']=='not_seeded_error_saved' for x in r['rows']);passed.append('cache_content_mutation_refused_before_any_destination_link')
    watched=target/'watched';watched.mkdir();events=helper.StageEvents(watched)
    try:
        s=watched/'preparation_13';s.mkdir();assert ('stage',s) in events.next();events.stage(s)
        temp=s/'manifest.json.tmp';temp.write_text('{}');temp.rename(s/'manifest.json');assert ('manifest',s) in events.next()
    finally:events.close()
    assert events.fd is None;passed.append('real_local_inotify_IN_CREATE_directory_and_MOVED_TO_manifest_no_polling_then_close')
    tree=ast.parse(Path(helper.__file__).read_text());calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'run','Popen','sleep','bundle','request','dispatch','step','backward'};passed.append('source_has_no_SSH_service_GPU_timed_poll_model_or_gate_calls')
    assert not torch.cuda.is_initialized()
    frozen=target/'source_snapshot';frozen.mkdir()
    for p in (Path(helper.__file__),Path(__file__)):(frozen/p.name).write_bytes(p.read_bytes())
    receipt=target/'receipt.json';write(receipt,dict(status='passed_preparation_transfer_reuse_CPU_contracts',count=len(passed),checks=passed,source=entry(Path(helper.__file__)),checks_source=entry(Path(__file__)),real_manifest_metadata_only=entry(OUT/'completion/incoming/preparation_1791383090008639557/manifest.json'),actual_transfer_payload_bytes_read=0,fixture_bytes_per_complete_file=15,CUDA_VISIBLE_DEVICES='',cuda_initialized_before=False,cuda_initialized_after=False,SSH_calls=0,GPU_calls=0,service_calls=0,actual_preparation_seeded=False,preparation_gate_published=False,wall_seconds=time.monotonic()-started));print(json.dumps(dict(source=entry(Path(helper.__file__)),checks_source=entry(Path(__file__)),receipt=entry(receipt),count=len(passed)),indent=2))

if __name__=='__main__':main()
