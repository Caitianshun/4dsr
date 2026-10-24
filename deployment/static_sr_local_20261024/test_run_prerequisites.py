import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('local_prerequisites_runner',Path(__file__).with_name('run_prerequisites.py'))
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value))


class PrerequisiteTests(unittest.TestCase):
    def test_author_parents_then_immediate_evaluation_without_teachers_or_reused_weights(self):
        nodes = m.stage_commands('/run','/launch')
        self.assertEqual([n['name'] for n in nodes],['iesrgs_train','iesrgs_evaluate','threedsr_train','threedsr_evaluate'])
        for index,method in ((0,'iesrgs'),(2,'threedsr')):
            train,evaluate = nodes[index:index+2]
            self.assertEqual(train['argv'][:7],[str(m.shared.BASE_PYTHON),'-s','/launch','--mode','base','--',str(m.EXP/'train_stage.py')])
            self.assertEqual(train['argv'][train['argv'].index('--stage')+1],m.STAGES[method])
            self.assertIn(str(m.CONFIGS[method]),train['argv'])
            for forbidden in ('--checkpoint','--prior-bundle','--steps','--wiring-fixture-steps'): self.assertNotIn(forbidden,train['argv'])
            self.assertIn(f'/run/{method}/{m.STAGES[method]}/checkpoint.pt',evaluate['argv'])
            self.assertEqual(train['maximum_file_bytes'],2*m.GiB)
            self.assertEqual(evaluate['argv'][6],str(m.EXP/'render_eval.py'))

    def test_all_remaining_storage_and_checkpoint_failure_cap(self):
        nodes = m.stage_commands('/run','/launch')
        needed = 20*m.GiB+sum(n['additional_bytes'] for n in nodes)
        self.assertEqual(needed,36400332800)
        with self.assertRaises(RuntimeError): m.shared.storage_guard(nodes,0,20*m.GiB,free=needed-1)
        self.assertEqual(m.shared.storage_guard(nodes,2,20*m.GiB,free=needed)['required_bytes'],20*m.GiB+sum(n['additional_bytes'] for n in nodes[2:]))

    def test_each_method_has_its_own_real_wiring_gate(self):
        rows = [dict(scene='Cook07',method=x,status='passed') for x in m.CONFIGS]
        with patch.object(m.shared,'validate_gate',return_value={'results':rows}): m.validate_gate('wiring',{},m.GPU)
        for method in m.CONFIGS:
            bad = copy.deepcopy(rows); next(x for x in bad if x['method']==method)['status']='failed'
            with patch.object(m.shared,'validate_gate',return_value={'results':bad}):
                with self.assertRaisesRegex(ValueError,method): m.validate_gate('wiring',{},m.GPU)

    def test_outer_mode_variable_alone_cannot_replace_actual_launcher(self):
        with patch.object(m.shared,'admission',return_value={'cgroup':'broker'}),patch.dict(os.environ,{'STATIC_SR_LOCAL_RUNTIME_MODE':'base'}):
            actual = '\0'.join(m.outer_argv('/plan'))+'\0'
            with patch.object(Path,'read_bytes',return_value=actual.encode()):
                self.assertEqual(m.admission({},'/plan')['outer_argv'],m.outer_argv('/plan'))
            with patch.object(Path,'read_bytes',return_value=b'python\0run_prerequisites.py\0'):
                with self.assertRaisesRegex(ValueError,'Outer process'): m.admission({},'/plan')

    def test_nonbroker_is_refused_even_with_correct_launcher(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':m.GPU}),patch.object(Path,'read_text',return_value='0::/user.slice/unrelated'):
            with self.assertRaisesRegex(ValueError,'matching local broker'):
                m.admission(dict(host=socket.gethostname(),gpu_uuid=m.GPU,job_id='test'),'/plan')

    def sequence_fixture(self,root):
        run = root/'sequence'; config=root/'sequence.json'; save(config,{'method':'sequence_matters'})
        paths=[run/'teacher_acceptance/acceptance.json',run/'teacher/receipt.json',run/'bundle.json',run/'sr/complete.json',run/'sr/checkpoint.pt',run/'sr/evaluation/metrics.json']
        for p in paths: p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}')
        complete=dict(status='completed_with_evaluation',wiring_fixture=False,stage='sr',steps=30000,training_keys=6000,
            checkpoint=m.identity(paths[4]),config=m.identity(config),host=socket.gethostname(),gpu_uuid=m.GPU)
        save(paths[3],complete)
        log=run/'execution/train.log';log.parent.mkdir(parents=True);log.write_text('finished')
        prior=dict(job_id='sequence-job',host=socket.gethostname(),gpu_uuid=m.GPU,required_files=[],commands=[dict(name='sr')])
        plan=root/'sequence_plan.json';save(plan,prior)
        record=dict(schema='static_sr_local_sequence_completion_v1',status='completed',run_root=str(run),job_id='sequence-job',
            config=m.identity(config),plan=m.identity(plan),broker=dict(host=socket.gethostname(),CUDA_VISIBLE_DEVICES=m.GPU,
            cgroup='0::/claudevpn.slice/claude-local-sequence-job.service'),stages=[dict(name='sr',returncode=0,seconds=1,log=m.identity(log))],
            artifacts=[m.identity(p) for p in paths])
        path=run/'execution/complete.json';save(path,record)
        return run,config,path,record,complete,paths

    def test_complete_sequence_parent_and_original_artifact_bytes_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,config,path,record,complete,paths=self.sequence_fixture(Path(tmp))
            with patch.object(m.sequence,'CONFIG',config),patch.object(m.sequence,'validate_plan',return_value=run),patch.object(m.shared,'verify_metrics',return_value={}):
                m.validate_sequence_completion(path,m.GPU)
                paths[2].write_text('changed bundle')
                with self.assertRaisesRegex(ValueError,'Frozen file differs'):m.validate_sequence_completion(path,m.GPU)

    def test_sequence_fixture_failure_and_wrong_gpu_cannot_admit_prerequisites(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,config,path,record,complete,paths=self.sequence_fixture(Path(tmp))
            with patch.object(m.sequence,'CONFIG',config),patch.object(m.sequence,'validate_plan',return_value=run),patch.object(m.shared,'verify_metrics',return_value={}):
                failed=copy.deepcopy(record);failed['status']='failed';save(path,failed)
                with self.assertRaisesRegex(ValueError,'successful full'):m.validate_sequence_completion(path,m.GPU)
                save(path,record)
                with self.assertRaisesRegex(ValueError,'host/GPU'):m.validate_sequence_completion(path,'GPU-other')
                complete['wiring_fixture']=True;save(paths[3],complete)
                record['artifacts']=[m.identity(p) for p in paths];save(path,record)
                with self.assertRaisesRegex(ValueError,'formal same-host'):m.validate_sequence_completion(path,m.GPU)

    def test_missing_sequence_evaluation_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,config,path,record,complete,paths=self.sequence_fixture(Path(tmp))
            with patch.object(m.sequence,'CONFIG',config),patch.object(m.sequence,'validate_plan',return_value=run),patch.object(m.shared,'verify_metrics',side_effect=ValueError('missing full cam00')):
                with self.assertRaisesRegex(ValueError,'full cam00'):m.validate_sequence_completion(path,m.GPU)

    def plan(self):
        run=m.RUNS/'cpu_contract';launcher=m.RUNTIME/'launch_v2.py'
        return dict(schema=m.SCHEMA,status='CPU_plan_ready_requires_broker',run_id='cpu_contract',job_id=m.shared.job_id('cpu_contract','prereq'),
            gpu_uuid=m.GPU,run_root=str(run),execution_paths=dict(base_launcher=str(launcher)),commands=m.stage_commands(run,launcher),
            storage=dict(reserve_bytes=20*m.GiB),configs={k:{} for k in m.CONFIGS},gates={'wiring':{},'metrics':{}},sequence_completion={},required_files=[])

    def test_fixed_command_config_and_disk_contract_rejects_changes(self):
        p=self.plan()
        with patch.object(m,'identity',return_value={}),patch.object(m,'required_paths',return_value=[]),patch.object(m,'read',return_value={}):
            self.assertEqual(m.validate_plan(p),m.RUNS/'cpu_contract')
            for mutate in (lambda x:x['commands'][0]['argv'].extend(['--wiring-fixture-steps','2']),lambda x:x['commands'].reverse(),
                           lambda x:x['commands'][0].update(maximum_file_bytes=1),lambda x:x['storage'].update(reserve_bytes=0),
                           lambda x:x.update(status='blocked_storage'),lambda x:x.update(gpu_uuid='GPU-other')):
                bad=copy.deepcopy(p);mutate(bad)
                with self.assertRaises(ValueError):m.validate_plan(bad)

    def test_parent_completion_accepts_common_relative_identity_and_rejects_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'cfg.json';save(config,{'method':'iesrgs'})
            parent=root/'run/iesrgs/internal';parent.mkdir(parents=True);checkpoint=parent/'checkpoint.pt';checkpoint.write_bytes(b'checkpoint')
            config_id=dict(path='cfg.json',sha256=m.sha(config))
            result=dict(status='stage_completed',wiring_fixture=False,steps=30000,stage='internal',training_keys=6000,
                prior_bundle=None,training_HR_read=False,heldout_pixels_read=False,host=socket.gethostname(),gpu_uuid=m.GPU,
                config=config_id,checkpoint=m.identity(checkpoint),seconds=10.,peak_bytes=100,gpu_name='CPU synthetic metadata')
            save(parent/'complete.json',result)
            with patch.object(m.shared,'ROOT',root),patch.dict(m.CONFIGS,{'iesrgs':config}):
                self.assertEqual(m.verify_parent(root/'run','iesrgs',m.GPU)['training_seconds'],10.)
                result['wiring_fixture']=True;save(parent/'complete.json',result)
                with self.assertRaisesRegex(ValueError,'formal prerequisite'):m.verify_parent(root/'run','iesrgs',m.GPU)

    def test_parent_completion_refuses_foreign_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'cfg.json';save(config,{})
            parent=root/'run/threedsr/lr';parent.mkdir(parents=True);foreign=root/'foreign.pt';foreign.write_bytes(b'foreign')
            save(parent/'complete.json',dict(status='stage_completed',wiring_fixture=False,steps=30000,stage='lr',training_keys=6000,
                prior_bundle=None,training_HR_read=False,heldout_pixels_read=False,host=socket.gethostname(),gpu_uuid=m.GPU,
                config=m.identity(config),checkpoint=m.identity(foreign),seconds=1.,peak_bytes=10,gpu_name='CPU synthetic metadata'))
            with patch.dict(m.CONFIGS,{'threedsr':config}):
                with self.assertRaisesRegex(ValueError,'formal prerequisite'):m.verify_parent(root/'run','threedsr',m.GPU)

    def execution(self,root,code='raise SystemExit(0)',required=()):
        run=root/'run';path=root/'plan.json';parent=root/'parent.json';save(parent,{})
        nodes=[dict(name=method+'_'+action,method=method,action=action,
            argv=[sys.executable,'-B','-c',code],additional_bytes=1,maximum_file_bytes=4096)
            for method in ('iesrgs','threedsr') for action in ('train','evaluate')]
        save(path,dict(job_id='safe',configs={},gpu_uuid=m.GPU,gates={},sequence_completion=m.identity(parent),
            commands=nodes,required_files=list(required),storage=dict(reserve_bytes=0)))
        return run,path

    def patches(self,run):
        from contextlib import ExitStack
        stack=ExitStack()
        for name,value in [('validate_plan',run),('admission',{}),('check_inputs',{}),('validate_sequence_completion',{})]:
            stack.enter_context(patch.object(m,name,return_value=value))
        return stack

    def test_real_cpu_children_are_serial_and_each_evaluation_precedes_next_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);observed=root/'order'
            run,path=self.execution(root,'from pathlib import Path; p=Path('+repr(str(observed))+'); p.write_text(p.read_text()+"x" if p.exists() else "x")')
            order=[]
            # Generated artifact bytes are fixed independently of a child's order marker.
            artifact=root/'artifact';artifact.write_text('sealed')
            def parent(*args):
                order.append(args[1]+'_train');return dict(completion=m.identity(artifact),checkpoint=m.identity(artifact))
            def metrics(path,*args):
                order.append(Path(path).parent.parent.name+'_evaluate');return m.identity(artifact)
            with self.patches(run),patch.object(m,'verify_parent',side_effect=parent),patch.object(m.shared,'verify_metrics',side_effect=metrics):m.run_plan(path)
            self.assertEqual(order,['iesrgs_train','iesrgs_evaluate','threedsr_train','threedsr_evaluate'])
            self.assertEqual(observed.read_text(),'xxxx')
            self.assertEqual(m.read(run/'execution/complete.json')['status'],'completed')

    def test_failed_metrics_stop_before_threedsr_and_no_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,path=self.execution(Path(tmp))
            with self.patches(run),patch.object(m,'verify_parent',return_value=dict(completion=m.identity(path),checkpoint=m.identity(path))),patch.object(m.shared,'verify_metrics',side_effect=ValueError('cam00 incomplete')):
                with self.assertRaisesRegex(ValueError,'cam00 incomplete'):m.run_plan(path)
                self.assertFalse((run/'execution/threedsr_train.log').exists())
                self.assertFalse((run/'execution/complete.json').exists())
                with self.assertRaises(FileExistsError):m.run_plan(path)

    def test_child_failure_preserves_log_and_stops_following_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,path=self.execution(Path(tmp),'print("bounded failure evidence");raise SystemExit(7)')
            with self.patches(run):
                with self.assertRaisesRegex(RuntimeError,'code 7'):m.run_plan(path)
            self.assertFalse((run/'execution/iesrgs_evaluate.log').exists())
            self.assertEqual(m.read(run/'execution/status.json')['status'],'failed')
            self.assertIn('bounded failure evidence',m.read(run/'execution/diagnostics.json')['failure_log']['tail'])

    def test_source_mutation_blocks_next_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.write_text('frozen')
            run,path=self.execution(root,'from pathlib import Path;Path('+repr(str(source))+').write_text("changed")',[m.identity(source)])
            with self.patches(run),patch.object(m,'verify_parent',return_value=dict(completion={},checkpoint={})):
                with self.assertRaisesRegex(ValueError,'Frozen file differs'):m.run_plan(path)
            self.assertFalse((run/'execution/iesrgs_evaluate.log').exists())

    def test_actual_file_cap_fails_without_truncating_checkpoint_to_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'checkpoint.pt'
            result=m.subprocess.run([sys.executable,'-B','-c','from pathlib import Path;Path('+repr(str(target))+').write_bytes(b"a"*8192)'],
                stdout=m.subprocess.DEVNULL,stderr=m.subprocess.DEVNULL,preexec_fn=lambda:m.shared.file_limit(4096))
            self.assertNotEqual(result.returncode,0)


if __name__=='__main__':unittest.main()
