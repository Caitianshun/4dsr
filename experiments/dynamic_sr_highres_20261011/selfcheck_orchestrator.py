"""CPU-only negative gate, freeze, transfer and exit-event checks. Never SSH/GPU."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import local_orchestrator as orch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    checks = []
    def passed(name): checks.append(dict(name=name, passed=True))
    def rejects(name, callback):
        try: callback()
        except (ValueError, FileExistsError, RuntimeError): passed(name)
        else: raise AssertionError('Expected rejection: ' + name)
    with tempfile.TemporaryDirectory(prefix='highres_orchestrator_cpu_') as tmp:
        root = Path(tmp)
        orch.ROOT = root; orch.HERE = root/'experiments/dynamic_sr_highres_20261011'
        orch.OUT = root/'output/dynamic_sr_highres_20261011'; orch.DATA = root/'data/dynamic_sr/highres_20261011'
        orch.local = lambda p: Path(p).resolve() if Path(p).is_absolute() else (root/Path(p)).resolve()
        orch.HERE.mkdir(parents=True)
        zero = orch.OUT/'zero_checks/deskgames_cube'; zero.mkdir(parents=True)
        (orch.HERE/'zero_checks.py').write_text('fixture frozen inference\n')
        (orch.HERE/'local_orchestrator.py').write_text('fixture frozen supervisor\n')
        cfg = orch.HERE/'datasets/deskgames_cube.json'; cfg.parent.mkdir()
        cfg.write_text(json.dumps(dict(id='deskgames_cube')))
        z1 = zero/'z1_half.json'; window = zero/'window_selection.json'
        z1.write_text(json.dumps(dict(passed=True,rung='half',reprojection=dict(median_px=.3,p90_px=.8),NCC=dict(median=.85,wrong_mapping_median=.01))))
        window.write_text(json.dumps(dict(selected_start=300)))
        def ent(p):
            p=Path(p); p=p if p.is_absolute() else root/p
            return dict(path=str(p.relative_to(root)),sha256=orch.sha(p))
        orch.entry = ent
        freeze=dict(namespace_python=orch.namespace_inventory(),inputs={str(p.relative_to(root)):orch.sha(p) for p in (z1,window,cfg)},z1_path=str(z1.relative_to(root)))
        b=orch.BINDINGS['deskgames_cube']
        spec=dict(schema=orch.SCHEMA,dataset='deskgames_cube',**{k:b[k] for k in ('condition','host','remote_root','remote_python')},seed=orch.SEED,rung='half',run_id='fixture_seed07',out=str((orch.OUT/'orchestration/fixture_seed07').relative_to(root)),local_python=sys.executable,workers=1,cpu_threads=1,reserve_bytes=60*10**9,remote_future_gb=65,gpus=[dict(index=0,uuid='GPU-00000000-0000-0000-0000-000000000001')],supervisor=None,freeze=freeze)
        orch.validate_spec(spec); passed('valid_half_seed07_spec_no_GPU')
        for field,value in [('seed',20261008),('rung','full'),('host','hoi-5090'),('remote_root','/tmp/shared'),('reserve_bytes',0)]:
            bad=copy.deepcopy(spec);bad[field]=value
            rejects('reject_unapproved_'+field,lambda:orch.validate_spec(bad))
        (orch.HERE/'added.py').write_text('namespace drift\n')
        rejects('reject_added_namespace_source',lambda:orch.check_freeze(freeze))
        (orch.HERE/'added.py').unlink()
        before=z1.read_bytes();z1.write_bytes(before+b'\n')
        rejects('reject_changed_Z1_bytes',lambda:orch.check_freeze(freeze));z1.write_bytes(before)
        missing,receipt=orch.freeze_screening(spec)
        assert missing is None and 'missing' in receipt['status'];passed('missing_Z5_rejects_without_training')
        network=root/'third_party/swinir_frozen_20261010/network_swinir.py';network.parent.mkdir(parents=True);network.write_bytes(b'network fixture')
        weight=network.with_name('001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth');weight.write_bytes(b'weight fixture')
        orch.NETWORK_SHA=orch.sha(network);orch.CHECKPOINT_SHA=orch.sha(weight)
        samples=zero/'samples_half/samples.json';samples.parent.mkdir()
        rows=[]
        for c in range(1,21):
            for frame in (15,150,285):
                paths={k:f'cam{c:02d}_{frame:04d}_{k}.png' for k in ('lr','hr','companion')}
                for p in paths.values():(samples.parent/p).write_bytes(b'pixels fixture')
                rows.append(dict(camera=f'cam{c:02d}',frame=frame,source_frame=300+frame,paths=paths,sha256={k:orch.sha(samples.parent/v) for k,v in paths.items()}))
        samples.write_text(json.dumps(dict(dataset='deskgames_cube',rung='half',start=300,entries=rows,dataset_config=ent(cfg),source=ent(orch.HERE/'zero_checks.py'),test_views_used=False,teacher_cache=False)))
        result=zero/'gpu_screen_half/z5_half.json';result.parent.mkdir()
        z=dict(dataset='deskgames_cube',rung='half',source=ent(orch.HERE/'zero_checks.py'),samples_identity=dict(path='/unchanged/remote/path/samples.json',sha256=orch.sha(samples)),frozen=dict(network_SHA256=orch.NETWORK_SHA,checkpoint_SHA256=orch.CHECKPOINT_SHA,tile=0,precision='float32'),rows=[dict(camera=r['camera'],frame=r['frame']) for r in rows],aggregate=dict(bicubic_PSNR=33.,SwinIR_PSNR=34.,band_PSNR_gain=dict(mid=1.2,high=.2)),Z5_pass=False)
        complete=result.parent/'complete.json'
        def save_result(z,status):
            result.write_text(json.dumps(z));complete.write_text(json.dumps(dict(status=status,result=ent(result))))
        save_result(z,'completed_rejected_Z5')
        frozen,rejected=orch.freeze_screening(spec);assert frozen is None and rejected['status']=='rejected_Z5_no_training';passed('single_frequency_gain_insufficient')
        class NoWork(orch.Supervisor):
            def cpu(self,*args,**kwargs): raise AssertionError('CPU preparation unexpectedly ran after reject')
            def remote(self,*args,**kwargs): raise AssertionError('SSH unexpectedly ran after reject')
        NoWork(spec).run();assert (root/spec['out']/'reject.json').is_file();passed('rejected_gate_run_writes_receipt_without_CPU_SSH_training')
        z['aggregate']['band_PSNR_gain']['high']=1.1;z['Z5_pass']=True
        save_result(z,'completed_passed_Z5')
        frozen,accepted=orch.freeze_screening(spec);assert frozen is not None and accepted['status']=='passed_Z5';passed('two_frequency_gate_pass_binds_receipt_and_samples')
        assert json.loads(result.read_text())['samples_identity']['path']=='/unchanged/remote/path/samples.json';passed('original_absolute_source_record_preserved')
        z['Z5_pass']=False;save_result(z,'completed_passed_Z5')
        rejects('numeric_status_gate_disagreement',lambda:orch.freeze_screening(spec))
        z['Z5_pass']=True;z['rows'].pop();save_result(z,'completed_passed_Z5')
        rejects('missing_metric_row',lambda:orch.freeze_screening(spec))
        (root/'symlink').symlink_to(root/'third_party',target_is_directory=True)
        rejects('symlink_ancestor_rejected',lambda:orch.safe_local_member('symlink/file'))
        rejects('escaping_member_rejected',lambda:orch.safe_local_member('../outside'))
        transfer=root/'transfer';transfer.mkdir();file=transfer/'same.bin';file.write_bytes(b'x'*2000001)
        manifest=dict(files={'same.bin':dict(bytes=file.stat().st_size,sha256=orch.sha(file)),'missing.bin':dict(bytes=1,sha256=orch.sha(file))},future_bytes=0)
        def helper(mode,inventory):
            return subprocess.run([sys.executable,'-B','-c',orch.REMOTE_HELPER,str(transfer),mode],input=json.dumps(inventory),text=True,capture_output=True)
        p=helper('preflight',manifest);assert p.returncode==0 and json.loads(p.stdout)['missing']==['missing.bin'];passed('stream_sync_preflight_reuses_identical_and_lists_missing')
        p=helper('verify',manifest);assert p.returncode!=0;passed('post_sync_missing_rejected')
        manifest['files']['same.bin']['sha256']='0'*64
        p=helper('preflight',manifest);assert p.returncode!=0 and file.read_bytes()==b'x'*2000001;passed('differing_remote_file_rejected_and_preserved')
        inventory=helper('inventory',dict(paths=['same.bin']));assert inventory.returncode==0;passed('stream_result_inventory')
        fits=orch.OUT/'runs/fixture'
        expected=[[f'cam{c:02d}',f] for c in range(1,21) for f in (15,150,285)]
        for arm,psnr in [('LRparent17k',20),('SRrecon',22),('HRref',25),('base',23)]:
            d=fits/('prior_fit_'+arm);d.mkdir(parents=True)
            columns={'scope':'train_prior_fit_diagnostic'}
            for prefix,value in [('render_raw_vs_HR',psnr),('teacher_vs_HR',26),('bicubic_vs_HR',21),('render_raw_vs_teacher',28)]:
                columns.update({prefix+'_psnr':value,prefix+'_ssim':.8,prefix+'_lpips':.2,prefix+'_mse':.01})
            rows=[dict(columns,camera=c,frame=f) for c,f in expected]
            table=d/'metrics_per_observation.csv'
            with table.open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            metrics=d/'prior_fit.json';metrics.write_text('{}')
            index=d/'observation_index.json';index.write_text('{}')
            complete=dict(status='completed_train_view_prior_fit',seed=orch.SEED,scene='dgH_cube',heldout_pixels_used=False,parameter_updates=0,population_keys=expected,metrics=ent(metrics),csv=ent(table),observation_index=ent(index))
            (d/'prior_fit_complete.json').write_text(json.dumps(complete))
        supervisor=orch.Supervisor(spec)
        summary=supervisor.summarize_prior_fit(fits);value=json.loads(summary.read_text())
        assert value['observations']==60 and value['comparisons']['image_teacher_minus_SRrecon_render']['metrics']['psnr']['improvement_mean']==4 and value['comparisons']['HRref_minus_SRrecon_same_recipe']['metrics']['psnr']['improvement_mean']==3
        passed('prior_fit_identical_population_and_diagnostic_differences')
        prior= fits/'prior_fit_base/prior_fit_complete.json'
        value=json.loads(prior.read_text());value['parameter_updates']=1;prior.write_text(json.dumps(value))
        rejects('prior_fit_parameter_updates_refused',lambda:supervisor.summarize_prior_fit(fits))
        import gpu_startup as startup
        from types import SimpleNamespace
        a=SimpleNamespace(condition='l3H_long360',seed=20261008,gpu=1,uuid='GPU-test',out='output/dynamic_sr_highres_20261011/smoke/fixture',z5_receipt='unused',z5_samples='unused')
        rejects('GPU_startup_rejects_second_seed_before_query',lambda:startup.validate(a))
        a.seed=orch.SEED;a.gpu=0
        rejects('GPU_startup_rejects_wrong_A100_card_before_query',lambda:startup.validate(a))
        # Exercise the real collection path: old-output generator source and
        # external metric helpers must be transferred without old result trees.
        condition=spec['condition'];ready=orch.DATA/'prepared'/condition/'manifest.json';ready.parent.mkdir(parents=True);ready.write_text('{}')
        hrindex=orch.DATA/'hr_full'/condition/'hr_full_storage.json';hrindex.parent.mkdir(parents=True);hrindex.write_text('{}')
        lrconfig=orch.OUT/'registry/full_LR_prefixes'/condition/f'seed_{orch.SEED}/config.json';lrconfig.parent.mkdir(parents=True);lrconfig.write_text('{}')
        table=orch.OUT/'pipelines/tasks_fixture.json';table.parent.mkdir(parents=True);table.write_text('{}')
        generator=root/'output/dynamic_sr_20260918/cook_spinach_pilot_v1_lr_integrated/source_snapshot/06_generate_prior.py';generator.parent.mkdir(parents=True);generator.write_text('frozen generator fixture\n')
        oldresult=generator.parent.parent/'do_not_copy.pt';oldresult.write_bytes(b'old result protected')
        planpath=orch.OUT/'registry/teachers'/condition/'plan.json';planpath.parent.mkdir(parents=True)
        plan=dict(generator=ent(generator),network=ent(network),checkpoint=ent(weight),frozen_recipe=dict(generator_sha256=orch.sha(generator),network_sha256=orch.sha(network),checkpoint_sha256=orch.sha(weight)))
        planpath.write_text(json.dumps(plan))
        for relative in orch.METRIC_SOURCES:
            p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('frozen external metric fixture\n')
        for name in orch.DEPENDENCIES:
            d=root/'experiments'/name;d.mkdir(parents=True);(d/'fixture.py').write_text('frozen generic dependency\n')
        sync=supervisor.collect_inputs(condition,table)
        assert str(generator.relative_to(root)) in sync['files'] and all(r in sync['files'] for r in orch.METRIC_SOURCES) and str(oldresult.relative_to(root)) not in sync['files']
        passed('required_generator_and_metric_source_closure_without_old_results')
        plan['generator']['sha256']='0'*64;planpath.write_text(json.dumps(plan))
        rejects('changed_generator_pin_refused_before_transfer',lambda:supervisor.collect_inputs(condition,table))
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.2)'])
        identity=orch.process_identity(child.pid);begin=time.monotonic();exitrec=orch.wait_local_exit(identity);child.wait()
        assert exitrec['event_wait'] and time.monotonic()-begin>.05;passed('owned_PID_exit_event_wait')
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)'])
        try:
            identity=orch.process_identity(child.pid);identity['start_ticks']+=1
            rejects('reused_PID_identity_rejected',lambda:orch.wait_local_exit(identity))
        finally: child.terminate();child.wait()
    report=dict(status='passed_CPU_no_SSH_GPU_or_training',checks=checks,count=len(checks),source=str(Path(orch.__file__).resolve()),source_sha256=orch.sha(orch.__file__),scope='gate rejection, drift, additive streaming transfer, exact process-exit identity; fixture inputs only')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=report['status'],checks=len(checks),out=str(args.out))))
if __name__=='__main__':main()
