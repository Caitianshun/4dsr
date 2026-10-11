"""CPU acceptance of the screening DAG and refusal paths; launches no job."""
from __future__ import annotations
import copy,csv,json,math,tempfile
from pathlib import Path
from unittest import mock
from common import *
import make_screening_tasks as table
import persistent_runner as runner
import teacher_screened as screened

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=True)+'\n');return path

def main():
    checks=[]
    def accepts(name,fn):
        fn();checks.append(dict(name=name,result='accepted_expected'))
    def refuses(name,fn):
        try:fn()
        except (ValueError,RuntimeError,OSError,KeyError,TypeError):
            checks.append(dict(name=name,result='refused_expected'));return
        raise AssertionError(f'Unsafe case accepted: {name}')
    tasks=table.build('l3H_long360',20261007,'receipt.json','samples.json')
    accepts('six_endpoint_DAG',lambda:runner.validate(tasks))
    assert len(tasks)==28
    names={t['name'] for t in tasks}
    for arm in ['LRparent17k','lam0','base','oracleHR','SRrecon','HRref']:
        assert f'l3H_long360_seed20261007_eval_{arm}' in names
        assert f'l3H_long360_seed20261007_extras_{arm}' in names
        extra=next(t for t in tasks if t['name']==f'l3H_long360_seed20261007_extras_{arm}')
        assert 'l3H_long360_seed20261007_register_sr' in extra['deps']
    for arm in ['LRparent17k','SRrecon','HRref','base']:
        assert f'l3H_long360_seed20261007_prior_fit_{arm}' in names
    complete_table=dict(schema='highres_screening_event_DAG_v1',condition='l3H_long360',seed=20261007,tasks=tasks,budgets=dict(LRparent17k=17000,lam0_base_oracle=23000,SRrecon_HRref=17000))
    accepts('exact_approved_builder_and_budgets',lambda:runner.validate_table(complete_table))
    wrong=copy.deepcopy(complete_table);wrong['budgets']['SRrecon_HRref']=23000
    refuses('budget_change_before_launch',lambda:runner.validate_table(wrong))
    wrong=copy.deepcopy(complete_table);wrong['tasks']=[t for t in wrong['tasks'] if 'prior_fit' not in t['name']]
    refuses('omitted_required_endpoint_diagnostics',lambda:runner.validate_table(wrong))
    for kind,change in [
        ('duplicate_names',lambda x:x.append(copy.deepcopy(x[0]))),
        ('unsafe_name',lambda x:x[0].update(name='../escaped')),
        ('unresolved_dependency',lambda x:x[0]['deps'].append('missing')),
        ('cycle',lambda x:x[0]['deps'].append(x[1]['name'])),
        ('output_escape',lambda x:x[0].update(done='output/other/result.json')),
        ('unapproved_script',lambda x:x[0].update(script='../../train.py')),
        ('wrong_device',lambda x:x[0].update(device='cpu')),
        ('shortened_training',lambda x:next(t for t in x if t['script']=='entry.py')['argv'].extend(['--stop','2'])),
    ]:
        invalid=copy.deepcopy(tasks);change(invalid)
        refuses(kind,lambda x=invalid:runner.validate(x))
    mkdir(OUT/'selfchecks')
    with tempfile.TemporaryDirectory(prefix='runner_fixture_',dir=OUT/'selfchecks') as td:
        folder=Path(td);producer=folder/'manifest.json';plan=folder/'plan.json';samples=folder/'samples.json';z5=folder/'z5.json'
        cfg=HERE/'datasets/deskgames_cube.json';camids=['cam01','cam02'];frames=[15,150,285]
        m=dict(scene='dgH_cube',full_time_decode_verified=True,source_sha256=sha(HERE/'prepare_colmap_dataset.py'),original_start=600,splits=dict(train=camids),observations=[dict(camera_id=c,frame_index=f,split='train',lr_sha256=f'fixture_{c}_{f}') for c in camids for f in frames])
        dump(producer,m);p=dict(producer_manifest=entry(producer));dump(plan,p)
        s=dict(dataset=read(cfg)['id'],rung='half',start=600,dataset_config=entry(cfg),source=entry(HERE/'zero_checks.py'),test_views_used=False,entries=[dict(camera=c,frame=f,source_frame=600+f,sha256=dict(lr=f'fixture_{c}_{f}')) for c in camids for f in frames])
        dump(samples,s)
        z=dict(Z5_pass=True,dataset=s['dataset'],rung='half',samples_identity=entry(samples),source=entry(HERE/'zero_checks.py'),frozen=dict(network_SHA256=screened.teacher.FROZEN['network_sha256'],checkpoint_SHA256=screened.teacher.FROZEN['checkpoint_sha256'],tile=0,precision='float32'),rows=[dict(camera=c,frame=f) for c in camids for f in frames])
        dump(z5,z)
        accepts('Z5_all_training_cameras_three_frames',lambda:screened.gate(plan,z5,samples))
        mutations=[
            ('Z5_false','z',lambda q:q.update(Z5_pass=False)),
            ('wrong_window','s',lambda q:q.update(start=601)),
            ('wrong_grid','z',lambda q:q.update(rung='full')),
            ('wrong_source_frame','s',lambda q:q['entries'][0].update(source_frame=999)),
            ('LR_hash_differs','s',lambda q:q['entries'][0]['sha256'].update(lr='wrong')),
            ('Z5_source_changed','z',lambda q:q['source'].update(sha256='bad')),
            ('samples_source_changed','s',lambda q:q['source'].update(sha256='bad')),
            ('frozen_checkpoint_changed','z',lambda q:q['frozen'].update(checkpoint_SHA256='bad')),
            ('tiled_teacher','z',lambda q:q['frozen'].update(tile=128)),
            ('mixed_precision','z',lambda q:q['frozen'].update(precision='float16')),
            ('test_view_leakage','s',lambda q:q.update(test_views_used=True)),
            ('missing_sample','s',lambda q:q['entries'].pop()),
            ('duplicate_sample','s',lambda q:q['entries'].append(copy.deepcopy(q['entries'][0]))),
            ('missing_metric','z',lambda q:q['rows'].pop()),
            ('duplicate_metric','z',lambda q:q['rows'].append(copy.deepcopy(q['rows'][0]))),
        ]
        for name,which,mutate in mutations:
            sm,zm=copy.deepcopy(s),copy.deepcopy(z);mutate(sm if which=='s' else zm)
            dump(samples,sm);zm['samples_identity']=entry(samples);dump(z5,zm)
            refuses(name,lambda:screened.gate(plan,z5,samples))
        dump(samples,s);dump(z5,z)
        dump(producer,dict(m,full_time_decode_verified=False));dump(plan,dict(p,producer_manifest=entry(producer)))
        refuses('unverified_full_decode',lambda:screened.gate(plan,z5,samples))
        dump(producer,m);dump(plan,p)
        points=folder/'points.npz';points.write_bytes(b'CPU synthetic point fixture')
        consumer=folder/'manifest_train_ready.json';prepared_z1=folder/'z1_half.json';window=folder/'window_selection.json'
        initialized=dict(m,initialization=dict(seed=20261007,train_only_lr=True,npz_path='points.npz',npz_sha256=sha(points)),original_full_manifest=entry(producer))
        dump(consumer,initialized);dump(prepared_z1,dict(passed=True,rung='half'));dump(window,dict(selected_start=600))
        prepared=dict(manifest=entry(producer),z1=entry(prepared_z1),window=entry(window));dump(folder/'complete.json',prepared)
        dump(folder/'initialization_seed_20261007/complete.json',dict(fixture=True))
        accepts('prepared_Z1_window_initialization_identity',lambda:runner.prepared_inputs('dgH_cube',20261007,consumer))
        dump(prepared_z1,dict(passed=False,rung='half'));dump(folder/'complete.json',dict(prepared,z1=entry(prepared_z1)))
        refuses('prepared_Z1_failed',lambda:runner.prepared_inputs('dgH_cube',20261007,consumer))
        dump(prepared_z1,dict(passed=True,rung='full'));dump(folder/'complete.json',dict(prepared,z1=entry(prepared_z1)))
        refuses('prepared_Z1_another_grid',lambda:runner.prepared_inputs('dgH_cube',20261007,consumer))
        dump(prepared_z1,dict(passed=True,rung='half'));dump(window,dict(selected_start=601));dump(folder/'complete.json',dict(prepared,window=entry(window)))
        refuses('prepared_window_different',lambda:runner.prepared_inputs('dgH_cube',20261007,consumer))
        dump(window,dict(selected_start=600));dump(folder/'complete.json',prepared);points.write_bytes(b'changed')
        refuses('prepared_initial_points_corrupted',lambda:runner.prepared_inputs('dgH_cube',20261007,consumer))
        points.write_bytes(b'CPU synthetic point fixture')
        # The checkpoint contains synthetic bytes, not a trainable model.
        modeldir=folder/'model';modeldir.mkdir();cp=modeldir/'checkpoint.pt';cp.write_bytes(b'CPU validation fixture only')
        task=dict(name='fixture',script='entry.py',deps=[],device='gpu',done=str(modeldir/'complete.json'),argv=['lr','--out',str(modeldir)])
        dump(modeldir/'dispatch.json',dict(argv=task['argv'],entry=entry(HERE/'entry.py')))
        complete=dict(status='completed',seconds=1.0,checkpoint=entry(cp),state=dict(coarse_iteration=3000,fine_iteration=14000))
        dump(task['done'],complete)
        accepts('formal_LR_complete_and_checkpoint_SHA',lambda:runner.check_completion(task))
        for name,edit in [('wrong_LR_budget',lambda q:q['state'].update(fine_iteration=10)),('failed_training_status',lambda q:q.update(status='failed')),('nonfinite_training_time',lambda q:q.update(seconds=math.nan)),('wrong_checkpoint_hash',lambda q:q['checkpoint'].update(sha256='bad'))]:
            broken=copy.deepcopy(complete);edit(broken);dump(task['done'],broken)
            refuses(name,lambda:runner.check_completion(task))
        dump(task['done'],complete);dump(modeldir/'smoke_override.json',dict(diagnostic_only=True))
        refuses('smoke_parent',lambda:runner.check_completion(task));(modeldir/'smoke_override.json').unlink()
        sr=copy.deepcopy(task);sr['argv']=['sr','--out',str(modeldir)]
        dump(modeldir/'dispatch.json',dict(argv=sr['argv'],entry=entry(HERE/'entry.py')))
        dump(sr['done'],dict(status='completed',seconds=1,checkpoint=entry(cp),cursor=6000))
        accepts('SR_6000_complete',lambda:runner.check_completion(sr))
        dump(sr['done'],dict(status='completed',seconds=1,checkpoint=entry(cp),cursor=5))
        refuses('SR_short_budget',lambda:runner.check_completion(sr))
        evaluation=copy.deepcopy(task);evaluation.update(done=str(modeldir/'summary.json'),argv=['eval','--out',str(modeldir)])
        dump(modeldir/'dispatch.json',dict(argv=evaluation['argv'],entry=entry(HERE/'entry.py')))
        metrics=dict(status='completed',frames=300,test=dict(psnr=27.0,ssim=.86,lpips=.16))
        dump(evaluation['done'],metrics)
        def frames_csv(indices):
            with (modeldir/'metrics_per_observation.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=['scope','camera','frame']);w.writeheader();w.writerows(dict(scope='test',camera='cam00',frame=i) for i in indices)
        frames_csv(range(300));accepts('full_300_unique_frame_evaluation',lambda:runner.check_completion(evaluation))
        frames_csv(range(299));refuses('missing_evaluation_frame',lambda:runner.check_completion(evaluation))
        frames_csv([*range(299),298]);refuses('duplicate_evaluation_frame',lambda:runner.check_completion(evaluation))
        frames_csv(range(300));csvpath=modeldir/'metrics_per_observation.csv';csvpath.write_text(csvpath.read_text().replace('cam00','cam01',1))
        refuses('wrong_main_camera',lambda:runner.check_completion(evaluation))
        frames_csv(range(300));bad=copy.deepcopy(metrics);bad['test']['lpips']=math.nan;dump(evaluation['done'],bad)
        refuses('nonfinite_evaluation_metric',lambda:runner.check_completion(evaluation))
        actual=ROOT/'output/dynamic_sr_modules_20261010/rep2_3090/eval/Cook07_gateaff_lam0p2_dssim0p2_posthoc_cull'
        if actual.is_dir():
            dump(evaluation['done'],read(actual/'summary.json'))
            csvpath.write_bytes((actual/'metrics_per_observation.csv').read_bytes())
            accepts('existing_Cook07_actual_300_endpoint_schema',lambda:runner.check_completion(evaluation))
        prior=folder/'prior';prior.mkdir();baseline=folder/'baseline';baseline.mkdir();hr=folder/'hr';hr.mkdir()
        for path in [baseline/'config.json',hr/'hr_full_storage.json',prior/'teacher_index.json',prior/'teacher_acceptance.json',prior/'prior_fit.json',prior/'observation_index.json']:
            dump(path,dict(fixture=True))
        pcsv=prior/'metrics_per_observation.csv'
        def prior_csv(keys):
            with pcsv.open('w') as f:
                w=csv.DictWriter(f,fieldnames=['camera','frame']);w.writeheader();w.writerows(dict(camera=c,frame=i) for c,i in keys)
        keys=[(c,i) for c in camids for i in frames];prior_csv(keys)
        prior_task=dict(name='prior_fixture',script='prior_fit.py',deps=[],device='gpu',done=str(prior/'prior_fit_complete.json'),argv=['--checkpoint',str(cp),'--baseline-sr',str(baseline),'--z5-samples',str(samples),'--hr-root',str(hr),'--out',str(prior),'--cull'])
        prior_value=dict(schema='highres_train_prior_fit_complete_v1',status='completed_train_view_prior_fit',training=False,optimizer_calls=0,parameter_updates=0,heldout_pixels_used=False,observations=len(keys),sampled_frames=frames,population_keys=keys,train_cameras=camids,
                         source=entry(HERE/'prior_fit.py'),checkpoint=entry(cp),manifest=entry(producer),baseline_sr=entry(baseline/'config.json'),teacher_index=entry(prior/'teacher_index.json'),teacher_acceptance=entry(prior/'teacher_acceptance.json'),hr_storage=entry(hr/'hr_full_storage.json'),z5_samples=entry(samples),metrics=entry(prior/'prior_fit.json'),csv=entry(pcsv),observation_index=entry(prior/'observation_index.json'))
        dump(prior_task['done'],prior_value);accepts('complete_train_only_prior_fit_all_cameras',lambda:runner.check_completion(prior_task))
        for name,edit in [('prior_fit_parameter_updates',lambda q:q.update(parameter_updates=1)),('prior_fit_heldout_leakage',lambda q:q.update(heldout_pixels_used=True)),('prior_fit_missing_population',lambda q:q['population_keys'].pop()),('prior_fit_changed_source',lambda q:q['source'].update(sha256='bad'))]:
            broken=copy.deepcopy(prior_value);edit(broken);dump(prior_task['done'],broken)
            refuses(name,lambda:runner.check_completion(prior_task))
        dump(prior_task['done'],prior_value)
        wrong_prior=copy.deepcopy(prior_task);wrong_prior['argv'][1]=str(modeldir/'other_checkpoint.pt')
        refuses('prior_fit_another_checkpoint_command',lambda:runner.check_completion(wrong_prior))
        prior_csv(keys[:-1]);broken=copy.deepcopy(prior_value);broken['csv']=entry(pcsv);dump(prior_task['done'],broken)
        refuses('prior_fit_incomplete_CSV_population',lambda:runner.check_completion(prior_task))
        prior_csv(keys);dump(prior_task['done'],prior_value)
        # Mock child processes test return-code handling without invoking Python/GPU.
        supervisor=folder/'supervisor';(supervisor/'logs').mkdir(parents=True);(supervisor/'events').mkdir()
        failed=dict(name='mock_nonzero',script='entry.py',argv=['lr'],done=str(folder/'absent.json'))
        child=mock.Mock(pid=123,wait=mock.Mock(return_value=17))
        with mock.patch.object(runner.subprocess,'Popen',return_value=child) as popen:
            refuses('nonzero_child_exit',lambda:runner.execute(failed,None,supervisor,[],[]));assert popen.call_count==1 and child.wait.call_count==1
        missing=dict(failed,name='mock_no_artifact');child=mock.Mock(pid=124,wait=mock.Mock(return_value=0))
        with mock.patch.object(runner.subprocess,'Popen',return_value=child):
            refuses('zero_exit_without_completion_artifact',lambda:runner.execute(missing,None,supervisor,[],[]))
        frozen=[entry(cp)];cp.write_bytes(b'corrupted')
        with mock.patch.object(runner.subprocess,'Popen') as popen:
            refuses('code_or_input_change_before_launch',lambda:runner.execute(dict(failed,name='mock_frozen'),None,supervisor,frozen,[]));assert not popen.called
        with mock.patch.object(runner,'disk_guard',side_effect=RuntimeError('reserve breached')),mock.patch.object(runner.subprocess,'Popen') as popen:
            refuses('disk_reserve_breached_before_child_launch',lambda:runner.execute(dict(failed,name='mock_disk'),None,supervisor,[],[]));assert not popen.called
        refuses('new_Python_file_in_frozen_namespace',lambda:runner.check_frozen_identity([],['different.py']))
        accepts('unchanged_full_Python_inventory',lambda:runner.check_frozen_identity([],sorted(q.name for q in HERE.glob('*.py'))))
        gpu=dict(uuid='GPU-fixture',index='0',name='fixture')
        resource=mock.Mock(stdout='GPU-fixture, 442, 40960, 0\n')
        result=mock.Mock(stdout='GPU-fixture, 11, /usr/bin/foreign_training, 12000\n')
        with mock.patch.object(runner.subprocess,'run',side_effect=[resource,result]):
            refuses('foreign_compute_process',lambda:runner.check_gpu_idle(gpu,[]))
        result=mock.Mock(stdout='GPU-fixture, 12, /usr/libexec/gnome-remote-desktop-daemon, 420\n')
        with mock.patch.object(runner.subprocess,'run',side_effect=[resource,result]):
            accepts('explicit_allowed_graphics_process',lambda:runner.check_gpu_idle(gpu,['gnome-remote-desktop-daemon']))
        for name,text in [('high_GPU_utilization','GPU-fixture, 442, 40960, 85\n'),('high_GPU_memory','GPU-fixture, 2048, 40960, 0\n')]:
            with mock.patch.object(runner.subprocess,'run',return_value=mock.Mock(stdout=text)):
                refuses(name,lambda:runner.check_gpu_idle(gpu,['gnome-remote-desktop-daemon']))
        with mock.patch.object(runner.subprocess,'run',side_effect=[resource,mock.Mock(stdout='GPU-fixture, 12, /usr/libexec/gnome-remote-desktop-daemon, 800\n')]):
            refuses('allowed_graphics_above_600MiB',lambda:runner.check_gpu_idle(gpu,['gnome-remote-desktop-daemon']))
        claims=folder/'claims';claims.mkdir();lock=claims/'GPU-fixture.lock';lock.write_text('foreign screen claim')
        def try_claim():
            with runner.gpu_claims([gpu],claims,[]):pass
        with mock.patch.object(runner,'check_gpu_idle'):
            refuses('existing_screen_claim_exclusively_protected',try_claim)
        assert lock.read_text()=='foreign screen claim';lock.unlink()
        with mock.patch.object(runner,'check_gpu_idle',side_effect=RuntimeError('busy')):
            refuses('claim_released_on_idle_check_failure',try_claim)
        assert not lock.exists()
        with mock.patch.object(runner,'check_gpu_idle'):
            with runner.gpu_claims([gpu],claims,[]):
                assert read(lock)['PID']==os.getpid()
            assert not lock.exists()
        checks.append(dict(name='claim_released_after_success',result='accepted_expected'))
    sources=[entry(HERE/name) for name in ['persistent_runner.py','teacher_screened.py','make_screening_tasks.py','selfcheck_runner.py','prior_fit.py']]
    sources.extend(entry(q) for q in [*runner.METRIC_SOURCE_PATHS,screened.teacher.GENERATOR])
    import hashlib
    identity=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:12];out=HERE/f'selfcheck_runner_{identity}.json'
    write(out,dict(status='passed_CPU_acceptance',checks=checks,sources=sources,tasks=28,endpoints=6,real_subprocess_launches=0,CUDA_calls=0,GPU_queries=0,fixture_policy='synthetic JSON and checkpoint bytes removed after test; no real training or inference'))
    print(json.dumps(dict(status='passed_CPU_acceptance',checks=len(checks),receipt=str(out),real_subprocess_launches=0,CUDA_calls=0)))
if __name__=='__main__':main()
