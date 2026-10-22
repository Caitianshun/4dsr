"""Local CPU tests only, including real Linux inotify/pidfd event waits."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('continue_acceptance',Path(__file__).with_name('continue_acceptance.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.patch=patch.object(c,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.here=patch.object(c,'HERE',self.root/'deployment');self.here.start();self.addCleanup(self.here.stop)

    def save(self,path,value):
        file=self.root/path;file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(value));return file

    def watch(self,path,timeout=2,code=None):
        return subprocess.run([sys.executable,'-c',code or c.WATCH_CODE],
            input=json.dumps(dict(path=str(path),timeout=timeout))+'\n',text=True,capture_output=True,timeout=timeout+2)

    def test_inotify_accepts_status_that_already_exists(self):
        path=self.save('status.json',{'status':'completed'})
        result=self.watch(path);self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'completed')

    def test_real_inotify_follows_new_parent_and_close_write_event(self):
        path=self.root/'new/deeper/status.json'
        timer=threading.Timer(.1,lambda:self.save('new/deeper/status.json',{'status':'failed','error':'CPU failure'}))
        timer.start()
        try:result=self.watch(path)
        finally:timer.join()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'failed')

    def test_inotify_timeout_is_terminal_failure_not_polling_retry(self):
        result=self.watch(self.root/'absent.json',.05)
        self.assertNotEqual(result.returncode,0);self.assertIn('terminal event wait expired',result.stderr)

    def test_inotify_directory_created_between_parent_search_and_add_watch(self):
        # Deterministically inject the exact race: directory creation precedes
        # add_watch on its old ancestor; terminal write follows 80 ms later.
        code=c.WATCH_CODE.replace('import ctypes,errno,json,os,select,struct,sys,time',
                                  'import ctypes,errno,json,os,select,struct,sys,time,threading')
        code=code.replace('   wd=libc.inotify_add_watch',
            "   if not p.parent.exists():\n    p.parent.mkdir(parents=True)\n    threading.Timer(.08,lambda:p.write_text(json.dumps({'status':'completed'}))).start()\n   wd=libc.inotify_add_watch",1)
        before=time.monotonic();result=self.watch(self.root/'race/deeper/status.json',2,code)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertLess(time.monotonic()-before,.8,'Watcher missed the event and waited for its timeout')

    def test_inotify_rearms_after_watched_parent_is_moved(self):
        parent=self.root/'watched';parent.mkdir();path=parent/'status.json'
        def replace():
            parent.rename(self.root/'old_parent');parent.mkdir()
            threading.Timer(.08,lambda:path.write_text(json.dumps({'status':'completed'}))).start()
        timer=threading.Timer(.1,replace);timer.start();before=time.monotonic()
        try:result=self.watch(path,2)
        finally:timer.join()
        self.assertEqual(result.returncode,0,result.stderr);self.assertLess(time.monotonic()-before,.8)

    def transfer_plan(self,pid=1,ticks=2):
        self.save('verified.json',dict(status='verified',inventory_sha256='expected',files=10,missing=[]))
        return dict(event_timeout_seconds=2,transfer=dict(pid=pid,start_ticks=ticks,boot_id=c.boot_id(),
                    verified='verified.json',inventory={'sha256':'expected'},expected_files=10))

    def test_real_pidfd_waits_exact_child_exit(self):
        child=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read()'],stdin=subprocess.PIPE)
        try:
            plan=self.transfer_plan(child.pid,c.process_ticks(child.pid))
            timer=threading.Timer(.1,child.stdin.close);timer.start()
            result=c.wait_transfer(plan);timer.join()
            self.assertEqual(result['process'],'pidfd_exit_observed')
        finally:
            if child.stdin and not child.stdin.closed:child.stdin.close()
            child.wait(timeout=3)

    def test_exited_pid_needs_verified_and_reused_pid_always_refused(self):
        plan=self.transfer_plan()
        with patch.object(c,'process_ticks',return_value=None):
            self.assertEqual(c.wait_transfer(plan)['process'],'already_exited')
            (self.root/'verified.json').unlink()
            with self.assertRaises(FileNotFoundError):c.wait_transfer(plan)
        self.transfer_plan()
        with patch.object(c,'process_ticks',return_value=999):
            with self.assertRaisesRegex(ValueError,'reused'):c.wait_transfer(plan)

    def test_reuse_during_pidfd_open_and_boot_change_refuse(self):
        plan=self.transfer_plan();fd=os.open(self.root/'verified.json',os.O_RDONLY)
        with patch.object(c,'process_ticks',side_effect=[2,999]),patch.object(c,'pidfd_open',return_value=fd):
            with self.assertRaisesRegex(ValueError,'during pidfd'):c.wait_transfer(plan)
        with patch.object(c,'boot_id',return_value='different'):
            with self.assertRaisesRegex(ValueError,'rebooted'):c.wait_transfer(plan)

    def test_transfer_receipt_bound_to_exact_inventory_and_count(self):
        plan=self.transfer_plan()
        self.save('verified.json',dict(status='verified',inventory_sha256='foreign',files=10,missing=[]))
        with self.assertRaisesRegex(ValueError,'another inventory'):c.verified_transfer(plan)

    def minimal_plan(self,name='plan.json'):
        source=self.save('frozen.json',{'immutable':True})
        return self.save(name,dict(schema=c.SCHEMA,host='cts',remote_root=c.REMOTE,no_replay=True,
            formal_training_allowed=False,frozen_files=[c.identity(source)]))

    def fake_operations(self,calls,fail=None,change=False):
        root=self.root
        class Ops:
            def __init__(self,*args):pass
            def action(self,name):
                calls.append(name)
                if name==fail:raise RuntimeError('fixture failure '+name)
                if change and name=='cpu_suite':(root/'frozen.json').write_text('changed')
                return {'observed':name}
            def cpu_suite(self):return self.action('cpu_suite')
            def job(self,name):return self.action(name)
            def transfer(self):return self.action('transfer')
            def bootstrap(self):return self.action('bootstrap')
        return Ops

    def test_stage_order_and_plan_cannot_be_replayed_into_new_output(self):
        path=self.minimal_plan();calls=[]
        record=c.execute(path,'run',self.fake_operations(calls))
        self.assertEqual(calls,['cpu_suite','sequence','transfer','bootstrap','diffusion'])
        self.assertEqual(record['status'],'acceptance_continuation_completed')
        self.assertFalse(record['formal_training_started'])
        with self.assertRaises(FileExistsError):c.execute(path,'second_run',self.fake_operations(calls))
        self.assertEqual(len(calls),5)
        self.assertEqual(c.read('second_run/failure.json')['status'],'failed')

    def test_failure_at_each_stage_stops_all_dependent_actions(self):
        stages=['cpu_suite','sequence','transfer','bootstrap','diffusion']
        for index,stage in enumerate(stages):
            path=self.minimal_plan('plan_'+stage+'.json');calls=[]
            with self.assertRaisesRegex(RuntimeError,'fixture failure'):
                c.execute(path,'run_'+stage,self.fake_operations(calls,fail=stage))
            self.assertEqual(calls,stages[:index+1])
            self.assertEqual(c.read('run_'+stage+'/status.json')['status'],'failed')

    def test_source_changed_while_waiting_stops_before_job_submission(self):
        path=self.minimal_plan();calls=[]
        with self.assertRaisesRegex(ValueError,'Frozen bytes changed'):
            c.execute(path,'run',self.fake_operations(calls,change=True))
        self.assertEqual(calls,['cpu_suite'])

    def job_fixture(self,methods=('sequence_matters',)):
        self.save(c.TEACHER,{'wrapper':True})
        commands=[];returns=[]
        for method in methods:
            for scene,size in [('Cook07',[64,32]),('Discussion07',[48,24])]:
                manifest=self.save('data/'+scene+'.json',{'resolutions':{'hr':size}})
                self.save('output/static_sr_wu_20261022/prepared_final/'+scene+'/splatsure.json',
                          {'manifest':c.identity(manifest)})
                out='output/static_sr_wu_20261022/cts_acceptance/'+method+'_'+scene
                commands.append(dict(argv=['{ROOT}/env/python','-u','{ROOT}/'+c.TEACHER,
                    '--method',method,'--records','{ROOT}/output/static_sr_wu_20261022/cts_prepared/'+scene+'/training_records.json',
                    '--frame','0','--out','{ROOT}/'+out]))
                returns.append(out)
        return self.save('job.json',dict(method='real_teacher_GPU_acceptance_only',training_only_inputs=True,
            no_proposed_modules=True,commands=commands,return_paths=returns,required_files=[c.identity(c.TEACHER)],
            job_id='fixture_job',gpu_uuid='GPU-fixture',out='output/job'))

    def test_job_plan_derives_dimensions_and_expected_receipts_from_frozen_inputs(self):
        path=self.job_fixture();spec=c.job_spec(path,['sequence_matters'])
        self.assertEqual([r['hr_wh'] for r in spec['receipts']],[[64,32],[48,24]])
        self.assertEqual(len(spec['receipts']),2)
        value=c.read(path);value['commands'][0]['argv'][2]='{ROOT}/experiments/train_stage.py';self.save('job.json',value)
        with self.assertRaisesRegex(ValueError,'non-teacher'):c.job_spec(path,['sequence_matters'])

    def teacher_fixture(self,method='threedsr'):
        expected=dict(method=method,records='output/records.json',hr_wh=[64,32])
        files=[dict(path=c.TEACHER,sha256='wrapper')]
        names=['sequence_matters.py'] if method=='sequence_matters' else ['threedsr.py','splatsure_stablesr.py']
        files+=[dict(path='experiments/static_sr_wu_20261022/methods/'+name,sha256=name) for name in names]
        receipt=dict(schema='static_sr_wu_teacher_acceptance_fixture_v1',status='fixture_passed',gpu_executed=True,
            teacher_generated=True,method=method,hardware={'gpu_uuid':'GPU-fixture'},wrapper={'sha256':'wrapper'},
            registered_records={'path':c.REMOTE+'/output/records.json'},adapter_sources=files[1:])
        if method=='threedsr':receipt['rounds']=[dict(round_index=i,outputs=[{'size_wh':[64,32]}]) for i in range(4)]
        else:receipt['outputs']=[{'size_wh':[64,32]}]
        return receipt,expected,dict(gpu_uuid='GPU-fixture'),files

    def test_teacher_gate_requires_four_actual_rounds_and_exact_source_hardware(self):
        value,expected,job,files=self.teacher_fixture();c.validate_teacher_receipt(value,expected,job,files)
        for change in ({'status':'failed'},{'gpu_executed':False},{'rounds':value['rounds'][:3]},
                       {'hardware':{'gpu_uuid':'wrong'}},{'adapter_sources':[]}):
            with self.assertRaises(ValueError):c.validate_teacher_receipt(dict(value,**change),expected,job,files)

    def test_teacher_gpu_uuid_optional_prefix_preserves_hardware_identity(self):
        value,expected,job,files=self.teacher_fixture('sequence_matters')
        uuid='fd2728bf-7bb9-4ce0-fc1b-6ab34ae0a910';job['gpu_uuid']='GPU-'+uuid
        for spelling in (uuid,uuid.upper(),'GPU-'+uuid):
            value['hardware']['gpu_uuid']=spelling;c.validate_teacher_receipt(value,expected,job,files)
        for spelling in ('fd2728bf-7bb9-4ce0-fc1b-6ab34ae0a911','',None):
            value['hardware']['gpu_uuid']=spelling
            with self.assertRaises(ValueError):c.validate_teacher_receipt(value,expected,job,files)

    def reuse_fixture(self):
        path=self.job_fixture();job=c.read(path)
        adapter='experiments/static_sr_wu_20261022/methods/sequence_matters.py'
        self.save(adapter,{'official_bridge':True});job['required_files'].append(c.identity(adapter))
        self.save('weights/psrt.pth',{'real_fixture':'psrt'});self.save('weights/spynet.pth',{'real_fixture':'spynet'})
        for index,command in enumerate(job['commands']):
            command['name']='fixture_'+str(index)
            command['argv']+=['--sequence-plan','{ROOT}/plans/'+str(index)+'.json',
                '--psrt-checkpoint','{ROOT}/weights/psrt.pth','--spynet-checkpoint','{ROOT}/weights/spynet.pth']
        self.save('job.json',job);item=c.job_spec(path,['sequence_matters'])
        for index,expected in enumerate(item['receipts']):
            value,_,_,_=self.teacher_fixture('sequence_matters')
            value['wrapper']=c.identity(c.TEACHER);value['adapter_sources']=[c.identity(adapter)]
            directory=Path(expected['path']).parent
            png=self.save(directory/'teacher/cam01/0000.png',{'pixels':'original'})
            value['outputs']=[dict(c.identity(png),key='cam01/0000',size_wh=expected['hr_wh'])]
            value['selected_keys']=['cam01/0000']
            for name in ('fixture_records','teacher_receipt'):
                value[name]=c.identity(self.save(directory/(name+'.json'),{'fixture':name}))
            value['registered_records']=c.identity(self.save(expected['records'],{'registered':True}))
            value['original_sequence_plan']=c.identity(self.save('plans/'+str(index)+'.json',{'ALS':True}))
            value['model_identities']=[c.identity('weights/'+name+'.pth') for name in ('psrt','spynet')]
            self.save(expected['path'],value)
        directory='deployment/dispatch/fixture_job/'
        terminal=dict(state='ended',outcome='success',training_exit_code=0,job_id=item['job_id'])
        self.save(directory+'intent.json',dict(host='cts',job=item['job']['path'],job_sha256=item['job']['sha256']))
        self.save(directory+'remote_terminal.json',terminal)
        self.save(directory+'collected.json',dict(status='collected',remote_terminal=terminal,missing_paths=[],transfer_errors=[],
            returned_paths=[{'path':p,'kind':'directory'} for p in job['return_paths']],local_SHA_verification=False))
        self.save(item['out']+'/complete.json',dict(status='completed',job_id=item['job_id'],job_sha256=item['job']['sha256'],
            gpu_uuid=item['gpu_uuid'],stages=[dict(name=x['name'],returncode=0,argv=[a.replace('{ROOT}',c.REMOTE) for a in x['argv']]) for x in job['commands']]))
        previous=self.save('old/plan.json',dict(schema=c.SCHEMA,host='cts',remote_root=c.REMOTE,sequence=item,frozen_files=item['frozen_files']))
        self.save('old/plan.claimed.json',dict(launchplan=c.identity(previous),out=str(self.root/'old/run')))
        self.save('old/run/status.json',dict(status='failed',current_stage='sequence',launchplan=c.identity(previous)))
        return previous,item

    def test_explicit_reuse_checks_existing_pngs_and_skips_all_submission_and_ssh(self):
        prior,item=self.reuse_fixture();reuse=c.sequence_completion(prior,item)
        self.assertEqual(len(reuse['outputs']),2);self.assertTrue(reuse['local_output_SHA_verified'])
        plan=dict(sequence=dict(item,reuse_completion=reuse),frozen_files=item['frozen_files'])
        out=self.root/'recovery';out.mkdir();ops=c.Operations(plan,out)
        before=(self.root/'old/run/status.json').read_bytes()
        with patch.object(ops,'query',side_effect=AssertionError('reuse attempted SSH')),\
             patch.object(c.subprocess,'run',side_effect=AssertionError('reuse attempted submission')):
            result=ops.job('sequence')
        self.assertTrue(result['reused_completion']);self.assertEqual(result['GPU_submissions'],0)
        self.assertEqual((self.root/'old/run/status.json').read_bytes(),before)
        self.assertFalse((out/'sequence_submit.log').exists())

    def test_reuse_rejects_foreign_job_incomplete_collection_and_output_tampering(self):
        prior,item=self.reuse_fixture()
        for path,key,value in [('output/job/complete.json','job_sha256','different'),
                ('deployment/dispatch/fixture_job/remote_terminal.json','job_id','different'),
                ('deployment/dispatch/fixture_job/collected.json','transfer_errors',[{'error':'missing'}]),
                ('old/run/status.json','current_stage','diffusion')]:
            original=c.read(path);self.save(path,dict(original,**{key:value}))
            with self.assertRaises(ValueError):c.sequence_completion(prior,item)
            self.save(path,original)
        reuse=c.sequence_completion(prior,item)
        self.save(reuse['outputs'][0]['path'],{'pixels':'tampered'})
        with self.assertRaisesRegex(ValueError,'Frozen bytes changed'):c.sequence_completion(prior,item)
        out=self.root/'recovery';out.mkdir()
        ops=c.Operations(dict(sequence=dict(item,reuse_completion=reuse)),out)
        with patch.object(ops,'query',side_effect=AssertionError('attempted SSH')):
            with self.assertRaisesRegex(ValueError,'Frozen bytes changed'):ops.job('sequence')

    def test_reuse_rejects_changed_source_or_job_and_default_still_refuses_dispatch(self):
        prior,item=self.reuse_fixture();out=self.root/'default';out.mkdir()
        ops=c.Operations(dict(sequence=item),out)
        with patch.object(ops,'query',return_value={'status':'verified'}),\
             patch.object(c.subprocess,'run',side_effect=AssertionError('default replayed job')):
            with self.assertRaisesRegex(FileExistsError,'never replay'):ops.job('sequence')
        changed=copy.deepcopy(item);changed['job_id']='other'
        with self.assertRaisesRegex(ValueError,'original frozen job'):c.sequence_completion(prior,changed)
        self.save(c.TEACHER,{'changed':'wrapper'})
        with self.assertRaisesRegex(ValueError,'Frozen bytes changed'):c.sequence_completion(prior,item)

    def test_cpu_suite_success_is_required_before_psrt(self):
        out=self.root/'out';out.mkdir()
        ops=c.Operations(dict(cpu_status='output/status.json',event_timeout_seconds=2),out)
        with patch.object(ops,'query',return_value={'status':'failed','GPU_calls':0,'stages':[]}):
            with self.assertRaisesRegex(RuntimeError,'did not complete'):ops.cpu_suite()

    def test_submit_arguments_and_receipts_are_checked_without_actual_dispatch(self):
        path=self.job_fixture();job=c.job_spec(path,['sequence_matters'])
        frozen=job['frozen_files'];plan=dict(sequence=dict(job,timeout_seconds=7200),frozen_files=frozen,local_python=sys.executable)
        out=self.root/'out';out.mkdir();ops=c.Operations(plan,out);seen=[]
        def run(argv,**kwargs):
            seen.append(argv)
            self.save('deployment/dispatch/fixture_job/remote_terminal.json',{'outcome':'success'})
            self.save('output/job/complete.json',{'status':'completed','job_id':'fixture_job'})
            for item in job['receipts']:self.save(item['path'],{'returned':True})
            return SimpleNamespace(returncode=0)
        with patch.object(ops,'query',return_value={'status':'verified'}),patch.object(c.subprocess,'run',side_effect=run),patch.object(c,'validate_teacher_receipt') as gate:
            result=ops.job('sequence')
        self.assertEqual(gate.call_count,2);self.assertEqual(len(result['receipts']),2)
        self.assertEqual(seen[0][-4:],['--memory-mib','18000','--timeout-seconds','7200'])
        self.assertIn('--gpu-index',seen[0]);self.assertNotIn('--reattach',seen[0]);self.assertNotIn('--submit-only',seen[0])

    def test_plan_freezes_inputs_and_pid_identity_without_remote_execution(self):
        sequence=self.job_fixture();value=c.read(sequence);value['job_id']='sequence';self.save('sequence.json',value)
        diffusion=self.job_fixture(('stablesr','threedsr'));value=c.read(diffusion);value['job_id']='diffusion';self.save('diffusion.json',value)
        for path in (c.SUBMIT,c.BOOTSTRAP,c.TEMPLATE,'coordinator.py'):self.save(path,{'frozen':path})
        inventory=self.save('inventory.json',{'files':[]})
        self.save('transfer/intent.json',dict(root=c.REMOTE,inventory='inventory.json',inventory_sha256=c.sha(inventory),files=10))
        args=SimpleNamespace(sequence_job='sequence.json',diffusion_job='diffusion.json',transfer_dir='transfer',
                             transfer_pid=1234,transfer_start_ticks=919543,event_timeout_seconds=3600)
        with patch.object(c,'__file__',str(self.root/'coordinator.py')),patch.object(c,'process_ticks',return_value=919543),\
             patch.object(c.subprocess,'run',side_effect=AssertionError('plan attempted execution')):
            plan=c.build_plan(args)
            prior=self.save('prior_plan.json',{'old_run':'failed_at_sequence'})
            reuse=dict(prior_launchplan=c.identity(prior),evidence_files=[c.identity(prior)])
            args.reuse_sequence_completion='prior_plan.json'
            with patch.object(c,'sequence_completion',return_value=reuse) as inspect:
                recovery=c.build_plan(args)
            self.assertEqual(inspect.call_args.args[0],'prior_plan.json')
            self.assertEqual(recovery['sequence']['reuse_completion'],reuse)
            self.assertIn(c.identity(prior),recovery['frozen_files'])
        self.assertEqual(plan['schema'],c.SCHEMA);self.assertEqual(plan['transfer']['start_ticks'],919543)
        self.assertEqual(len(plan['sequence']['receipts']),2);self.assertEqual(len(plan['diffusion']['receipts']),4)
        self.assertEqual(plan['sequence']['timeout_seconds'],7200);self.assertEqual(plan['diffusion']['timeout_seconds'],14400)
        self.assertEqual(plan['remote_calls_during_planning'],0)
        names={x['path'] for x in plan['frozen_files']}
        self.assertTrue({c.SUBMIT,c.BOOTSTRAP,c.TEMPLATE,'sequence.json','diffusion.json','coordinator.py'}<=names)

    def bootstrap_fixture(self,returncode):
        bootstrap_out='output/bootstrap';remote=self.root/'fake_remote';remote.mkdir()
        receipt=dict(status='CPU_overlays_cache_imports_and_requested_registration_passed_GPU_unverified',
            GPU_calls=0,training_started=False,teacher_images_generated=0,script={'sha256':'bootstrap'},
            checks={name:{'returncode':0} for name in ('stablesr','threedsr','sequence','wu_extensions')})
        if returncode:receipt['status']='failed_CPU_bootstrap_GPU_not_run'
        assets=remote/'runtime_assets_cts.json';assets.write_text('{}')
        log=remote/'stablesr_cpu_check.log';log.write_text('CPU-only log')
        receipt['runtime_assets']=dict(path=c.REMOTE+'/'+bootstrap_out+'/runtime_assets_cts.json',sha256=c.sha(assets))
        for value in receipt['checks'].values():
            value['log']=dict(path=c.REMOTE+'/'+bootstrap_out+'/stablesr_cpu_check.log',sha256=c.sha(log))
        files=[]
        for name,content in [('receipt.json',json.dumps(receipt)),('stablesr_cpu_check.log','CPU-only log'),('runtime_assets_cts.json','{}')]:
            source=remote/name;source.write_text(content)
            files.append(dict(path=bootstrap_out+'/'+name,bytes=source.stat().st_size,sha256=c.sha(source)))
        out=self.root/'coord';out.mkdir()
        plan=dict(bootstrap=dict(script={'path':c.BOOTSTRAP,'sha256':'bootstrap'},template={'path':c.TEMPLATE,'sha256':'template'},
            out=bootstrap_out,timeout_seconds=5))
        ops=c.Operations(plan,out);calls=[]
        def query(code,request,timeout):
            return {'files':files} if code==c.SMALL_RECORDS_CODE else {'status':'verified'}
        def run(argv,**kwargs):
            calls.append(argv)
            if argv[0]=='ssh':
                self.assertIn('CUDA_VISIBLE_DEVICES=',argv[-1]);self.assertIn(' apply ',argv[-1])
                return SimpleNamespace(returncode=returncode)
            self.assertEqual(argv[0],'rsync');shutil.copyfile(remote/Path(argv[-2]).name,argv[-1])
            return SimpleNamespace(returncode=0)
        return ops,query,run,calls,files

    def test_bootstrap_failure_still_returns_small_records_with_sha_then_stops(self):
        ops,query,run,calls,files=self.bootstrap_fixture(1)
        with patch.object(ops,'query',side_effect=query),patch.object(c.subprocess,'run',side_effect=run):
            with self.assertRaisesRegex(RuntimeError,'bootstrap failed'):ops.bootstrap()
        self.assertEqual(sum(x[0]=='rsync' for x in calls),3)
        proof=c.read('coord/bootstrap_return_verified.json')
        self.assertTrue(proof['local_SHA_verified']);self.assertEqual(proof['files'],files)

    def test_bootstrap_success_requires_full_four_import_checks_and_return_hashes(self):
        ops,query,run,calls,files=self.bootstrap_fixture(0)
        with patch.object(ops,'query',side_effect=query),patch.object(c.subprocess,'run',side_effect=run):
            result=ops.bootstrap()
        self.assertEqual(c.checked(result['receipt']),self.root/'output/bootstrap/receipt.json')
        for item in files:c.checked(item)

    def test_bootstrap_transfer_error_does_not_hide_later_small_logs(self):
        ops,query,run,calls,files=self.bootstrap_fixture(0)
        def failed_copy(argv,**kwargs):
            if argv[0]=='rsync' and argv[-1].endswith('receipt.json'):
                calls.append(argv);raise subprocess.CalledProcessError(23,argv)
            return run(argv,**kwargs)
        with patch.object(ops,'query',side_effect=query),patch.object(c.subprocess,'run',side_effect=failed_copy):
            with self.assertRaisesRegex(RuntimeError,'return incomplete'):ops.bootstrap()
        self.assertEqual(sum(x[0]=='rsync' for x in calls),3)
        self.assertTrue((self.root/'output/bootstrap/stablesr_cpu_check.log').is_file())
        proof=c.read('coord/bootstrap_return_verified.json')
        self.assertFalse(proof['local_SHA_verified']);self.assertEqual(len(proof['transfer_errors']),1)

    def test_bootstrap_cannot_accept_old_local_receipt_when_listing_is_empty(self):
        ops,query,run,calls,files=self.bootstrap_fixture(0)
        self.save('output/bootstrap/receipt.json',{'status':'old'})
        def empty(code,request,timeout):return {'files':[]} if code==c.SMALL_RECORDS_CODE else query(code,request,timeout)
        with patch.object(ops,'query',side_effect=empty),patch.object(c.subprocess,'run',side_effect=run):
            with self.assertRaisesRegex(RuntimeError,'return incomplete'):ops.bootstrap()


if __name__=='__main__':unittest.main()
