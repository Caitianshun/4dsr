"""CPU fixture tests only: no SSH, mount, package download, CUDA or training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile

import remote_storage as s
import storage_supervisor as w


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    checks = []
    def passed(name):
        checks.append(dict(name=name, passed=True))
    def rejects(name, fun):
        try:
            fun()
        except (ValueError, RuntimeError, FileExistsError, FileNotFoundError):
            passed(name)
        else:
            raise AssertionError('Expected rejection: ' + name)
    d = hashlib.sha256(b'pixels').hexdigest()
    files = {s.PREFIX + f'hr/cam{c:02d}/{f:04d}.png': dict(bytes=6, sha256=d) for c in range(1, 33) for f in range(300)}
    files[s.PREFIX + 'hr_full_storage.json'] = dict(bytes=1, sha256=d)
    short = s.validate_inventory(files)
    assert len(short) == 9601
    passed('exact32traincameras300frames_and_original_index')
    for name, key in [('cam00_test_view', 'hr/cam00/0000.png'), ('extra_frame', 'hr/cam01/0300.png'), ('failure_receipt', 'failure.json'), ('video', 'video.mp4'), ('escape', '../elsewhere'), ('absolute', '/tmp/elsewhere')]:
        bad = copy.deepcopy(files); bad[s.PREFIX + key] = dict(bytes=6, sha256=d)
        rejects('reject_' + name, lambda b=bad: s.validate_inventory(b))
    bad = copy.deepcopy(files); bad['data/dynamic_sr/highres_20261011/prepared/l3H_long360/lr/cam01/0000.png'] = dict(bytes=6, sha256=d)
    rejects('reject_LR_or_other_prefix', lambda: s.validate_inventory(bad))
    bad = copy.deepcopy(files); bad.pop(next(iter(bad)))
    rejects('reject_missing_full_HR_frame', lambda: s.validate_inventory(bad))
    for field, value in [('bytes', 0), ('bytes', -1), ('bytes', 1.5), ('bytes', True), ('sha256', '0'), ('sha256', 'g'*64)]:
        bad = copy.deepcopy(files); bad[next(iter(bad))][field] = value
        rejects('reject_member_' + field + '_' + str(value), lambda b=bad: s.validate_inventory(b))
    for value in ('../other', 'HAS_CAPS', 'spaces are unsafe', 'x'*65, ''):
        rejects('reject_run_id_' + repr(value), lambda v=value: s.paths(v))
    with tempfile.TemporaryDirectory(prefix='readonly_HR_CPU_') as temp:
        base = Path(temp)
        s.ROOT = base/'root'; s.DRIVE = base/'Game Drive'; s.EXTERNAL = s.DRIVE/'4dsr-highres-20261011/hr_readonly_inputs'
        s.ROOT.mkdir(); s.DRIVE.mkdir()
        run_id = 'fixture_run'
        source, target, out = s.paths(run_id)
        source.mkdir(parents=True); target.mkdir(parents=True); out.mkdir(parents=True)
        storage = dict(schema='highres_hr_full_storage_v1', LR_roundtrip_all_frames_SHA_equal=True, GPU_calls=0,
                       cameras={f'cam{c:02d}':dict(frames=300,hr_sha256={str(f):d for f in range(300)}) for c in range(1,33)})
        for rel in short:
            dest = source/rel; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(json.dumps(storage).encode() if rel == 'hr_full_storage.json' else b'pixels')
            short[rel] = dict(bytes=dest.stat().st_size, sha256=s.digest(dest))
        full = {s.PREFIX+k:v for k,v in short.items()}
        verified = s.verify_tree(source, short)
        assert verified['files'] == 9601
        passed('streamed_allfile_SHA_matches_original_HR_index')
        original = (source/'hr_full_storage.json').read_bytes()
        fixture = source/'hr/cam01/0000.png'
        fixture.write_bytes(b'change')
        rejects('reject_different_same_size_pixels', lambda: s.verify_tree(source, short))
        assert fixture.read_bytes() == b'change'; passed('conflicting_pixels_preserved')
        fixture.write_bytes(b'pixels')
        extra=source/'extra.txt'; extra.write_bytes(b'preserved')
        rejects('reject_extra_source_member', lambda: s.verify_tree(source, short)); extra.unlink()
        fixture.unlink(); fixture.symlink_to(base/'outside')
        rejects('reject_source_file_symlink', lambda: s.verify_tree(source, short)); fixture.unlink(); fixture.write_bytes(b'pixels')
        for field, value in [('LR_roundtrip_all_frames_SHA_equal',False),('GPU_calls',1),('schema','other')]:
            bad=copy.deepcopy(storage);bad[field]=value;(source/'hr_full_storage.json').write_text(json.dumps(bad))
            changed=dict(short);changed['hr_full_storage.json']=dict(bytes=(source/'hr_full_storage.json').stat().st_size,sha256=s.digest(source/'hr_full_storage.json'))
            rejects('reject_HR_index_'+field, lambda m=changed: s.verify_tree(source,m))
        (source/'hr_full_storage.json').write_bytes(original)
        bad=copy.deepcopy(storage);bad['cameras']['cam01']['hr_sha256']['0']='0'*64
        (source/'hr_full_storage.json').write_text(json.dumps(bad))
        changed=dict(short);changed['hr_full_storage.json']=dict(bytes=(source/'hr_full_storage.json').stat().st_size,sha256=s.digest(source/'hr_full_storage.json'))
        rejects('reject_index_pixel_SHA_disagreement',lambda:s.verify_tree(source,changed))
        (source/'hr_full_storage.json').write_bytes(original)
        real_disk = s.shutil.disk_usage
        s.shutil.disk_usage=lambda path:SimpleNamespace(free=70*10**9)
        assert s.guard(base,10*10**9)['reserve_bytes']==60*10**9;passed('exact60GB_reserve_boundary')
        rejects('reject_external_reserve_shortfall',lambda:s.guard(base,10*10**9+1))
        s.shutil.disk_usage=real_disk
        original_context=s.context
        s.context=lambda rid:(source,target,out,{},dict(pinned=True))
        rejects('reject_existing_owned_run_no_retry',lambda:s.apply('prepare',run_id,full))
        # Tests below stub subprocesses and /proc only; no actual service/mount.
        other_run='new_run';other_source,other_target,other_out=s.paths(other_run)
        s.context=lambda rid:(other_source,other_target,other_out,{},dict(pinned=True))
        (target/'occupied').write_bytes(b'protected')
        rejects('reject_nonempty_canonical_target',lambda:s.apply('prepare',other_run,full))
        assert (target/'occupied').read_bytes()==b'protected';passed('occupied_target_preserved')
        (target/'occupied').unlink()
        original_mountinfo=s.mountinfo
        s.mountinfo=lambda p:[dict(fstype='ext4',source='other',options=['rw'],super_options=[])]
        rejects('reject_existing_foreign_mount',lambda:s.apply('prepare',other_run,full))
        row=dict(fstype='fuse.bindfs',source=str(source),options=['ro'],super_options=['user_id='+str(os.getuid())])
        s.mountinfo=lambda p:[row]
        assert s.readonly_mount(target,source)==row;passed('accept_exact_readonly_currentuser_FUSE_identity')
        for field,value in [('fstype','ext4'),('source','/other'),('options',['rw']),('super_options',['allow_other'])]:
            bad=dict(row);bad[field]=value;s.mountinfo=lambda p,b=bad:[b]
            rejects('reject_mount_'+field,lambda:s.readonly_mount(target,source))
        s.mountinfo=lambda p:[row,row]
        rejects('reject_stacked_mounts',lambda:s.readonly_mount(target,source))
        s.mountinfo=original_mountinfo;s.context=original_context
        # Existing service names block dispatch; source and target stay intact.
        s.context=lambda rid:(source,target,out,{},dict(pinned=True))
        actual_capture=s.capture
        calls_service=[]
        def occupied_unit(argv):
            calls_service.append(argv)
            return dict(returncode=0,stdout='loaded\n',stderr='')
        s.capture=occupied_unit
        rejects('reject_existing_service_before_systemd_run',lambda:s.apply('mount',run_id,full))
        assert all('systemd-run' not in argv for argv in calls_service) and not (out/'mount_started.json').exists()
        passed('no_dispatch_or_receipt_overwrite_on_unit_conflict')
        s.capture=actual_capture;s.context=original_context
        # /proc fixtures validate exact service ownership without a live mount.
        original_binary,original_binary_sha=s.BINARY,s.BINARY_SHA
        s.BINARY=base/'bindfs_fixture';s.BINARY.write_bytes(b'pinned fake executable');s.BINARY_SHA=s.digest(s.BINARY)
        procroot=base/'proc';proc=procroot/'4242';proc.mkdir(parents=True)
        stat='4242 (bindfs fixture) '+' '.join(['0']*19+['123456'])+'\n'
        (proc/'stat').write_text(stat)
        expected=[str(s.BINARY),'--no-allow-other','-r','-f','-o','fsname='+str(source)+',subtype=bindfs',str(source),str(target)]
        (proc/'cmdline').write_bytes(b'\0'.join(os.fsencode(v) for v in expected)+b'\0')
        (proc/'exe').symlink_to(s.BINARY)
        state=dict(returncode=0,stdout='ActiveState=active\nSubState=running\nMainPID=4242\nExecStart=fake-display-only\nResult=success\n',stderr='')
        identity=s.service_identity(state,source,target,procroot)
        assert identity['argv']==expected and identity['start_ticks']==123456;passed('exact_owned_PID_arguments_binary_SHA_and_start_identity')
        for old,new in [('MainPID=4242','MainPID=0'),('MainPID=4242','MainPID=9999'),('ActiveState=active','ActiveState=failed'),('SubState=running','SubState=dead')]:
            bad=dict(state,stdout=state['stdout'].replace(old,new))
            rejects('reject_service_'+new,lambda b=bad:s.service_identity(b,source,target,procroot))
        bad=dict(state,stdout=state['stdout']+'MainPID=4242\n')
        rejects('reject_duplicate_service_PID',lambda:s.service_identity(bad,source,target,procroot))
        for argv in (expected+['--allow-other'],[x for x in expected if x!='-r'],[*expected[:-1],'/other/target']):
            (proc/'cmdline').write_bytes(b'\0'.join(os.fsencode(v) for v in argv)+b'\0')
            rejects('reject_changed_process_arguments_'+str(len(argv)),lambda:s.service_identity(state,source,target,procroot))
        (proc/'cmdline').write_bytes(b'\0'.join(os.fsencode(v) for v in expected)+b'\0')
        s.BINARY.write_bytes(b'modified executable')
        rejects('reject_running_binary_SHA_drift',lambda:s.service_identity(state,source,target,procroot))
        s.BINARY.write_bytes(b'pinned fake executable')
        actual_getuid=s.os.getuid;s.os.getuid=lambda:actual_getuid()+1
        rejects('reject_foreign_process_UID',lambda:s.service_identity(state,source,target,procroot))
        s.os.getuid=actual_getuid
        from unittest.mock import patch
        real_read=Path.read_text;seen=[]
        def shifting_read(path,*args,**kwargs):
            if path==proc/'stat':
                seen.append(True)
                return stat if len(seen)==1 else stat.replace('123456','123457')
            return real_read(path,*args,**kwargs)
        with patch.object(Path,'read_text',shifting_read):
            rejects('reject_reused_PID_during_identity_check',lambda:s.service_identity(state,source,target,procroot))
        s.BINARY,s.BINARY_SHA=original_binary,original_binary_sha
        # Source pin executes only the expected helper; mutated bytes fail first.
        probe=base/'root/helper_fixture.py';probe.write_text('import json,sys; print(json.dumps(dict(ok=True, argv=sys.argv)))\n')
        pin=s.digest(probe)
        cmd=[sys.executable,'-B','-c',w.PINNED_HELPER,str(probe),pin,str(base/'root'),'--mode','verify']
        result=subprocess.run(cmd,capture_output=True,text=True)
        assert result.returncode==0 and json.loads(result.stdout)['argv'][1:]==['--mode','verify']
        passed('helper_SHA_verified_before_exact_argument_dispatch')
        probe.write_text('raise RuntimeError("changed helper must not execute")\n')
        result=subprocess.run(cmd,capture_output=True,text=True)
        assert result.returncode!=0 and 'Frozen remote storage helper changed' in result.stderr and 'changed helper must not execute' not in result.stderr
        passed('remote_helper_drift_refused_before_execution')
        # Ensure wrapper delegates scientific validation and guards; no SSH.
        old_validate=w.parent.validate_spec
        calls=[]
        w.parent.validate_spec=lambda spec:calls.append(spec)
        w.PROJECT=w.DEPLOY.parents[1]
        pins={str((w.DEPLOY/n).relative_to(w.PROJECT)):w.parent.sha(w.DEPLOY/n) for n in w.DEPLOY_FILES}
        cfg=dict(schema=w.STORAGE_SCHEMA,external_parent=str(s.EXTERNAL),canonical_prefix=s.PREFIX,binary=str(s.BINARY),binary_sha256=s.BINARY_SHA,package=str(s.PACKAGE),package_sha256=s.PACKAGE_SHA,external_reserve_bytes=s.RESERVE)
        spec=dict(dataset='long360',condition='l3H_long360',host='a100-train',storage_recipe=cfg,freeze=dict(inputs=pins))
        w.validate_storage_spec(spec);assert len(calls)==1;passed('wrapper_calls_original_scientific_spec_validator')
        for field,value in [('dataset','deskgames_cube'),('condition','dgH_cube'),('host','cts')]:
            bad=copy.deepcopy(spec);bad[field]=value
            rejects('reject_wrapper_'+field,lambda b=bad:w.validate_storage_spec(b))
        for field,value in [('external_reserve_bytes',0),('canonical_prefix','other/'),('binary_sha256','0'*64),('external_parent','/tmp/other')]:
            bad=copy.deepcopy(spec);bad['storage_recipe'][field]=value
            rejects('reject_storage_spec_'+field,lambda b=bad:w.validate_storage_spec(b))
        for route in ('hoi-5090','other','cts -o ProxyCommand=evil',True):
            bad=copy.deepcopy(spec);bad['storage_recipe']['ssh_jump']=route
            rejects('reject_unknown_jump_'+str(route),lambda b=bad:w.validate_storage_spec(b))
        routed=copy.deepcopy(spec);routed['storage_recipe']['ssh_jump']='cts'
        w.validate_storage_spec(routed);passed('accept_explicit_temporary_cts_jump_same_A100_host')
        basic=dict(routed,out='output/dynamic_sr_highres_20261011/orchestration/fixture')
        supervisor=w.StorageSupervisor(basic)
        assert supervisor.ssh[-3:]==['-J','cts','a100-train']
        passed('temporary_cts_jump_is_inherited_by_original_SSH_methods')
        transfer=w.external_rsync_argv(supervisor.ssh,'a100-train','/owned/local','/media/ubuntu/Game Drive/owned/fixture',base/'members.txt')
        assert '--protect-args' in transfer and transfer[-1]=='a100-train:/media/ubuntu/Game Drive/owned/fixture/' and "'" not in transfer[-1]
        passed('rsync_space_path_protected_arguments_no_literal_shell_quotes')
        rejects('reject_external_transfer_wrong_host',lambda:w.external_rsync_argv(supervisor.ssh,'cts','/owned/local','/owned/remote',base/'members.txt'))
        bad=copy.deepcopy(spec);bad['freeze']['inputs'].pop(next(iter(pins)))
        rejects('reject_unfrozen_deployment_source',lambda:w.validate_storage_spec(bad))
        w.parent.validate_spec=old_validate
    assert 'torch' not in sys.modules
    passed('no_torch_import_or_CUDA_initialization')
    report=dict(status='passed_CPU_no_SSH_mount_GPU_or_training',count=len(checks),checks=checks,
                sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in w.DEPLOY_FILES})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=report['status'],checks=len(checks),out=str(args.out))))


if __name__=='__main__':
    main()
