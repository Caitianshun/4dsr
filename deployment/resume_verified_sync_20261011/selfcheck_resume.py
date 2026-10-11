"""CPU fixture recovery checks: no SSH, CUDA, training or data preparation."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import resume_sync as r


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    checks=[]
    def passed(name):checks.append(dict(name=name,passed=True))
    def rejects(name,callback):
        try:callback()
        except (ValueError,RuntimeError,FileExistsError,FileNotFoundError,BlockingIOError):passed(name)
        else:raise AssertionError('Expected rejection: '+name)
    original_root=r.parent.ROOT
    with tempfile.TemporaryDirectory(prefix='resume_sync_CPU_') as temp:
        root=Path(temp);old=root/r.OLD_REL;old.mkdir(parents=True);(old/'events').mkdir();(old/'logs').mkdir()
        data=root/'data/dynamic_sr/highres_20261011';out=root/'output/dynamic_sr_highres_20261011'
        def put(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
        def ent(p):return dict(path=str(Path(p).relative_to(root)),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
        spec=dict(schema=r.parent.SCHEMA,dataset='deskgames_cube',condition='dgH_cube',host='cts',remote_root='/home/cts/Project/4DSR/workspaces/highres_20261011',remote_python='/home/cts/Project/4DSR/.venv/bin/python',seed=r.parent.SEED,rung='half',run_id=r.OLD_ID,out=r.OLD_REL,local_python=sys.executable,workers=4,cpu_threads=2,reserve_bytes=60*10**9,remote_future_gb=65,gpus=[dict(index=0,uuid='GPU-fd2728bf-7bb9-4ce0-fc1b-6ab34ae0a910')],expansion_policy='fixed old policy',supervisor=None,freeze=dict(namespace_python={},inputs={},z1_path='dummy'))
        put(old/'spec.json',spec)
        failure=dict(status='failed_preserved_no_retry',stage='verified_additive_sync',remote_started=False,error='Additive synchronization failed, no retry')
        failure_path=put(old/'failure_fixture.json',failure)
        (old/'logs/sync.log').write_text('not responding\nBroken pipe\nrsync error in socket IO (code 10)\n')
        f=root/'frozen.txt';f.write_bytes(b'original frozen bytes')
        files={'frozen.txt':dict(bytes=f.stat().st_size,sha256=ent(f)['sha256'])}
        freeze=dict(namespace_python={'fixture.py':'0'*64},inputs={'frozen.txt':ent(f)['sha256']},z1_path='dummy')
        inv=dict(schema='highres_additive_stream_sync_v1',host='cts',destination=spec['remote_root'],future_bytes=65*10**9,files=files,freeze=freeze)
        table=put(out/'pipelines'/f'tasks_dgH_cube_seed{r.parent.SEED}_{r.OLD_ID}.json',dict(approved=True))
        freeze['inputs'][str(table.relative_to(root))]=ent(table)['sha256']
        put(old/'sync_inputs.json',inv);(old/'sync_missing.txt').write_text('frozen.txt\n')
        put(old/'screening_freeze.json',dict(freeze,inputs={'frozen.txt':ent(f)['sha256']}));put(old/'Z5_gate.json',dict(status='passed_Z5'))
        names=['wait_binding','supervisor_exited','started','returned','input_freeze','started','returned','input_freeze','started','returned','input_freeze','started','returned','started','returned','input_freeze','started','returned','input_freeze','input_freeze','sync_inventory','remote_preflight','sync_preflight_accepted']
        events=[dict(stage='fixture') for name in names]
        for stage,begin,end in [('prepare_all_video_half',3,4),('initialize_seed07',6,7),('expand_privileged_HR',9,10),('register_LR',12,13),('plan_teacher',14,15),('make_screening_tasks',17,18)]:events[begin-1]['stage']=stage;events[end-1].update(stage=stage,exit_code=0)
        events[19]['frozen_inputs']=freeze['inputs'];events[21]['exit_code']=0
        events[22]=dict(status='preflight_accepted',missing=['frozen.txt'],reused_count=0,disk=dict(reserve_bytes=60*10**9,needed_bytes=65*10**9+f.stat().st_size))
        event_paths=[put(old/'events'/f'{i:03d}_{n}.json',e) for i,(n,e) in enumerate(zip(names,events),1)]
        cams=[f'cam{i:02d}' for i in range(1,21)]
        put(data/'prepared/dgH_cube/manifest.json',dict(full_time_decode_verified=True,splits=dict(train=cams),decode_audit=[dict(actual_decoded_count=6000,packet_count=6000,selected_frame_count=300)]*21))
        put(data/'prepared/dgH_cube/complete.json',dict(GPU_calls=0))
        put(data/'hr_full/dgH_cube/hr_full_storage.json',dict(LR_roundtrip_all_frames_SHA_equal=True,cameras={c:dict(frames=300) for c in cams}))
        with patch.object(r.parent,'ROOT',root),patch.object(r.parent,'DATA',data),patch.object(r.parent,'OUT',out),patch.object(r.parent,'entry',ent),patch.object(r.parent,'local',lambda p:Path(p) if Path(p).is_absolute() else root/p),patch.object(r.parent,'validate_spec',lambda s:None),patch.object(r.parent,'check_freeze',lambda s:None),patch.object(r.parent,'freeze_screening',lambda s:(s['freeze'],dict(status='passed_Z5'))),patch.object(r.runner,'validate_table',lambda t:None):
            bound=r.validate_checkpoint(old,full_hash=True);assert bound['local_input_check']['full_SHA_checked'];passed('positive_fixture_exact23events_finalfreeze_and_local_SHA')
            original_failure=failure_path.read_bytes()
            for field,value in [('remote_started',True),('stage','remote_exit_event_wait'),('status','completed')]:
                bad=dict(failure);bad[field]=value;put(failure_path,bad)
                rejects('reject_failure_'+field,lambda:r.validate_checkpoint(old))
            failure_path.write_bytes(original_failure)
            event_paths[-1].rename(old/'events/024_startup_dispatch_attempted.json')
            rejects('reject_any_post_sync_dispatch_event',lambda:r.validate_checkpoint(old))
            (old/'events/024_startup_dispatch_attempted.json').rename(event_paths[-1])
            for idx in (3,6,9,12,14,17):
                before=event_paths[idx].read_bytes();bad=dict(events[idx],exit_code=1);put(event_paths[idx],bad)
                rejects('reject_failed_CPU_stage_'+str(idx),lambda:r.validate_checkpoint(old));event_paths[idx].write_bytes(before)
            before=event_paths[19].read_bytes();put(event_paths[19],dict(frozen_inputs={'lost_input':'0'*64}))
            rejects('reject_initial_or_changed_dynamic_freeze',lambda:r.validate_checkpoint(old));event_paths[19].write_bytes(before)
            before=event_paths[-1].read_bytes()
            for field,value in [('reused_count',1),('missing',['frozen.txt','frozen.txt']),('status','sync_completed')]:
                bad=copy.deepcopy(events[-1]);bad[field]=value;put(event_paths[-1],bad)
                rejects('reject_preflight_'+field,lambda:r.validate_checkpoint(old))
            event_paths[-1].write_bytes(before)
            oldlist=(old/'sync_missing.txt').read_bytes();(old/'sync_missing.txt').write_text('other.txt\n')
            rejects('reject_changed_transfer_list',lambda:r.validate_checkpoint(old));(old/'sync_missing.txt').write_bytes(oldlist)
            for field,value in [('host','a100-train'),('future_bytes',0),('destination','/tmp/shared')]:
                bad=dict(inv);bad[field]=value
                rejects('reject_inventory_'+field,lambda b=bad:r.validate_inventory(b,spec))
            before=f.read_bytes();f.write_bytes(b'changed same size!!!!!')
            rejects('reject_local_size_or_SHA_drift',lambda:r.validate_inventory(inv,spec,True));f.write_bytes(before)
            wrong=dict(inv,files={'../outside':dict(bytes=0,sha256='0'*64)})
            rejects('reject_escaping_sync_member',lambda:r.validate_inventory(wrong,spec))
            audit=put(out/'analysis/operator_remote_audit.json',dict(status='fixture_scoped_absence_accepted',host='cts'))
            recovery=copy.deepcopy(spec);recovery.update(run_id='fresh_spec_fixture',out=str((out/'orchestration/fresh_spec_fixture').relative_to(root)),supervisor=None)
            recovery['sync_recovery']=dict(schema=r.RECIPE_SCHEMA,old_out=r.OLD_REL,inventory=ent(old/'sync_inputs.json'),table=ent(table),failure=ent(failure_path),operator_remote_audit=ent(audit))
            pins=dict(freeze['inputs'],**r.evidence(old));pins.update({str((r.DEPLOY/n).relative_to(r.PROJECT)):r.parent.sha(r.DEPLOY/n) for n in r.DEPLOY_FILES});pins[str(audit.relative_to(root))]=ent(audit)['sha256']
            recovery['freeze']=dict(freeze,inputs=pins)
            r.validate_recovery_spec(recovery);passed('fresh_spec_retains_exact_science_inventory_evidence_and_sources')
            for field,value in [('seed',20261008),('rung','full'),('remote_future_gb',0),('reserve_bytes',0),('host','a100-train'),('gpus',[dict(index=0,uuid='GPU-other')])]:
                bad=copy.deepcopy(recovery);bad[field]=value
                rejects('reject_science_or_hardware_drift_'+field,lambda b=bad:r.validate_recovery_spec(b))
            bad=copy.deepcopy(recovery);bad['freeze']['inputs'][next(iter(pins))]='0'*64
            rejects('reject_recovery_frozen_source_or_evidence_drift',lambda:r.validate_recovery_spec(bad))
            bad=copy.deepcopy(recovery);bad['sync_recovery']['operator_remote_audit']['sha256']='0'*64
            rejects('reject_operator_audit_ref_drift',lambda:r.validate_recovery_spec(bad))
            bad=copy.deepcopy(recovery);bad['sync_recovery']['table']['sha256']='0'*64
            rejects('reject_old_table_ref_drift',lambda:r.validate_recovery_spec(bad))
            # Control flow test: only inherited sync/startup/runner/pull/summary.
            recipe=dict(schema=r.RECIPE_SCHEMA,old_out=r.OLD_REL)
            new=dict(spec,run_id='fixture_recovery',out=str((out/'orchestration/fixture_recovery').relative_to(root)),sync_recovery=recipe)
            sequence=[]
            class NoNetwork(r.SyncRecovery):
                def frozen(self):pass
                def no_launch_audit(self,label):sequence.append(label)
                def sync(self,inventory):assert inventory==inv;sequence.append('sync_original_inventory')
                def startup(self):sequence.append('startup_original')
                def runner(self,table):assert table==bound['table'];sequence.append('runner_original_table')
                def pull(self):sequence.append('pull_original')
                def summarize(self):sequence.append('summary_original')
            with patch.object(r,'validate_recovery_spec',lambda s:bound):NoNetwork(new).run()
            assert sequence==['before_sync_no_old_dispatch','sync_original_inventory','before_startup_no_old_dispatch','startup_original','runner_original_table','pull_original','summary_original'];passed('resume_starts_at_sync_no_CPU_gate_or_table_remake')
            rejects('fresh_recovery_out_prevents_second_dispatch',lambda:NoNetwork(new).run())
            rejects('CPU_method_is_explicitly_forbidden',lambda:r.SyncRecovery(new).cpu('prepare','unused',[]))
            assert failure_path.read_bytes()==original_failure;passed('original_failure_bytes_preserved')
            # Exercise the actual inherited summarize() body, including its
            # self.cpu() dispatch. Only base CPU execution is intercepted.
            summary_spec=dict(new,run_id='fixture_summary',out=str((out/'orchestration/fixture_summary').relative_to(root)))
            class ActualSummary(r.SyncRecovery):
                def frozen(self):pass
                def summarize_prior_fit(self,runs):return fitting
            summary=ActualSummary(summary_spec)
            summary.out.mkdir(parents=True)
            summary.pipeline_rel=str((out/'pipelines/fixture_summary').relative_to(root))
            put(root/summary.pipeline_rel/'complete.json',dict(status='awaiting_review'))
            put(summary.out/'startup_acceptance.json',dict(status='diagnostic_fixture'))
            put(summary.out/'checkpoint_index.json',dict(status='returned_fixture'))
            fitting=put(summary.out/'train_view_prior_fit_summary.json',dict(status='fixture_prior_fit',arms={a:dict(observations=60) for a in ('LRparent','SRrecon','HRref','base')}))
            zero=out/'zero_checks/deskgames_cube'
            put(zero/'window_selection.json',dict(selected_start=1200))
            put(zero/'gpu_screen_half/z5_half.json',dict(Z5_pass=True))
            runs=out/f'runs/dgH_cube_seed{r.parent.SEED}'
            for a in ('LRparent','lam0','base','oracleHR','SRrecon','HRref'):
                checkpoint=runs/a/'checkpoint.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(('fixture_'+a).encode())
                put(checkpoint.parent/'run.json',dict(seed=r.parent.SEED,scene=r.CONDITION))
                evaluation=runs/('eval_LRparent17k' if a=='LRparent' else 'eval_'+a)
                put(evaluation/'summary.json',dict(status='completed',frames=300))
                put(evaluation/'run.json',dict(checkpoint=ent(checkpoint)))
            dispatched=[]
            def base_analysis(obj,name,script,argv,estimated_gb=0):
                dispatched.append(dict(name=name,script=script,argv=list(map(str,argv)),estimated_gb=estimated_gb))
                put(obj.out/'headroom.json',dict(status='fixture_CPU_analysis_complete'))
            with patch.object(r.parent.Supervisor,'cpu',base_analysis):
                summary.summarize()
            assert ActualSummary.summarize is r.parent.Supervisor.summarize
            assert len(dispatched)==1 and dispatched[0]==dict(name='headroom',script='headroom.py',argv=['--spec',str(summary.out/'headroom_spec.json'),'--out',str(summary.out/'headroom.json')],estimated_gb=0)
            passed('actual_inherited_summarize_dispatches_only_exact_headroom_CPU_call')
            complete=json.loads((summary.out/'complete.json').read_text())
            assert complete['status']=='completed_screening_awaiting_review' and complete['headroom']==ent(summary.out/'headroom.json')
            passed('actual_inherited_summarize_writes_complete_receipt')
            allowed=['--spec',summary.out/'headroom_spec.json','--out',summary.out/'headroom.json']
            with patch.object(r.parent.Supervisor,'cpu',base_analysis):
                for stage in ('prepare_all_video_half','initialize_seed07','expand_privileged_HR','register_LR','plan_teacher','make_screening_tasks'):
                    rejects('reject_CPU_remake_stage_'+stage,lambda s=stage:summary.cpu(s,'headroom.py',allowed))
                rejects('reject_arbitrary_CPU_script',lambda:summary.cpu('headroom','prepare_colmap_dataset.py',allowed))
                rejects('reject_headroom_unowned_result_path',lambda:summary.cpu('headroom','headroom.py',['--spec',allowed[1],'--out',root/'foreign.json']))
                rejects('reject_headroom_extra_argument',lambda:summary.cpu('headroom','headroom.py',[*allowed,'--force']))
                rejects('reject_headroom_changed_CPU_budget',lambda:summary.cpu('headroom','headroom.py',allowed,estimated_gb=1))
                summary.stage='created'
                rejects('reject_headroom_before_final_summary_stage',lambda:summary.cpu('headroom','headroom.py',allowed))
                summary.stage='headroom_analysis'
                hpath=summary.out/'headroom_spec.json';original_hspec=hpath.read_bytes();hspec=json.loads(original_hspec)
                for field,value in [('seed',20261008),('condition','l3H_long360'),('arms',{})]:
                    put(hpath,dict(hspec,**{field:value}))
                    rejects('reject_headroom_identity_drift_'+field,lambda:summary.cpu('headroom','headroom.py',allowed))
                drift=copy.deepcopy(hspec);drift['protocol']['budgets']['base']=17000;put(hpath,drift)
                rejects('reject_headroom_endpoint_budget_drift',lambda:summary.cpu('headroom','headroom.py',allowed))
                hpath.write_bytes(original_hspec)
            assert len(dispatched)==1;passed('all_invalid_CPU_requests_stop_before_base_execution')
            new=dict(new,run_id='fixture_fail',out=str((out/'orchestration/fixture_fail').relative_to(root)))
            sequence=[]
            class FailSync(NoNetwork):
                def sync(self,inventory):sequence.append('sync_failed');raise RuntimeError('network fixture')
            with patch.object(r,'validate_recovery_spec',lambda s:bound):rejects('failed_sync_stops_before_any_GPU_method',lambda:FailSync(new).run())
            assert sequence==['before_sync_no_old_dispatch','sync_failed'];passed('no_automatic_retry_after_recovery_sync_failure')
            assert list((root/new['out']).glob('failure_*.json'));passed('fresh_recovery_failure_is_recorded')
        # Run the literal read-only audit against a local fake systemctl and
        # temporary filesystem. No host connection or real service operation.
        fakebin=root/'fakebin';fakebin.mkdir()
        fake=fakebin/'systemctl';marker=root/'foreign_unit'
        fake.write_text('#!'+sys.executable+'\nfrom pathlib import Path\np=Path('+repr(str(marker))+')\nprint("LoadState=loaded\\nActiveState=active\\nMainPID=123\\nInvocationID=other" if p.exists() else "LoadState=not-found\\nActiveState=inactive\\nMainPID=0\\nInvocationID=")\n');fake.chmod(0o755)
        payload=dict(root=str(root),units=['fixture_old_GPU.service'],paths=['output/fixture_GPU_output'],uuid='GPU-fixture')
        def audit_local():return subprocess.run([sys.executable,'-B','-c',r.NO_LAUNCH_AUDIT],input=json.dumps(payload),capture_output=True,text=True,env=dict(os.environ,PATH=str(fakebin)+os.pathsep+os.environ['PATH']))
        p=audit_local();assert p.returncode==0 and json.loads(p.stdout)['status']=='scoped_old_GPU_dispatch_absence_verified';passed('literal_remote_audit_accepts_local_scoped_absence_fixture')
        artifact=root/'output/fixture_GPU_output';artifact.mkdir();p=audit_local();assert p.returncode!=0 and artifact.is_dir();passed('audit_rejects_and_preserves_any_old_GPU_artifact');artifact.rmdir()
        claim=root/'output/dynamic_sr_highres_20261011/claims/GPU-fixture.lock';claim.parent.mkdir(parents=True);claim.write_text('foreign ownership')
        p=audit_local();assert p.returncode!=0 and claim.read_text()=='foreign ownership';passed('audit_rejects_and_preserves_foreign_GPU_claim');claim.unlink()
        marker.write_bytes(b'foreign unit');p=audit_local();assert p.returncode!=0 and marker.read_bytes()==b'foreign unit';passed('audit_rejects_live_or_unknown_old_GPU_unit');marker.unlink()
    # Validate source confinement of the audit via compilation only; never run
    # systemctl/SSH. Actual scoped audit is executed only by authorized --spec.
    compile(r.NO_LAUNCH_AUDIT,'no_launch_audit','exec');passed('remote_readonly_audit_compiles_without_network')
    assert 'torch' not in sys.modules;passed('no_torch_import_or_GPU_initialization')
    report=dict(status='passed_CPU_no_SSH_GPU_training_or_data_preparation',count=len(checks),checks=checks,sources={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in r.DEPLOY_FILES},scope='fixture metadata/provenance drift and execution order; source-only tests; full real corpus audited separately')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=report['status'],checks=len(checks),out=str(args.out))))


if __name__=='__main__':main()
