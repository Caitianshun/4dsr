import importlib.util
import json
import copy
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('local_splat_runner',Path(__file__).with_name('run_splatsure.py'))
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class RunnerTests(unittest.TestCase):
    def test_broker_ids_preserve_distinct_original_names(self):
        names=['discussion07_v1','discussion07-v1','A','a','a'*120,'a'*119+'b']
        ids=[m.job_id(s,'lr') for s in names]
        self.assertEqual(len(ids),len(set(ids)))
        for value in ids:self.assertRegex(value,r'^[a-z][a-z0-9-]{0,63}$')
        self.assertNotEqual(m.job_id('a','lr'),m.job_id('a','full'))
        for bad in ('../x','a/b','','a'*121):
            with self.assertRaises(ValueError):m.job_id(bad,'lr')

    def test_lr_uses_full_training_then_complete_evaluator(self):
        nodes=m.stage_commands('lr','/out','/launch','/teacher','/assets')
        self.assertEqual([n['name'] for n in nodes],['lr_train','lr_evaluate'])
        self.assertIn('lr',nodes[0]['argv']);self.assertIn('--checkpoint',nodes[1]['argv'])
        self.assertTrue(any(x.endswith('/render_eval.py') for x in nodes[1]['argv']))
        self.assertNotIn('--wiring-fixture-steps',str(nodes))
        self.assertNotIn('--steps',str(nodes))
        self.assertEqual(nodes[0]['maximum_file_bytes'],2*m.GiB)

    def test_full_preserves_producers_and_fresh_sr_with_implicit_evaluation(self):
        with patch.object(m,'read',return_value={'manifest':{'path':'/manifest','sha256':'x'}}),patch.object(m,'checked',return_value=Path('/manifest')):
            nodes=m.stage_commands('full','/out','/launch','/teacher','/assets')
        self.assertEqual([x['name'] for x in nodes],['teacher_plan','teacher_generate','contribution','bundle','bundle_gate','sr_train_and_evaluate'])
        self.assertIn('/out/lr/checkpoint.pt',nodes[2]['argv'])
        self.assertIn('--check-only',nodes[-2]['argv'])
        self.assertNotIn('--checkpoint',nodes[-1]['argv'])
        self.assertIn('/out/bundle.json',nodes[-1]['argv'])
        self.assertEqual(nodes[3]['argv'].count('--model'),2)

    def test_remaining_space_is_checked_not_just_next_stage(self):
        nodes=[dict(additional_bytes=10),dict(additional_bytes=20)]
        with self.assertRaises(RuntimeError):m.storage_guard(nodes,0,5,free=34)
        self.assertEqual(m.storage_guard(nodes,0,5,free=35)['required_bytes'],35)
        self.assertEqual(m.storage_guard(nodes,1,5,free=25)['required_bytes'],25)

    def test_mutated_production_command_or_budget_is_rejected(self):
        run=m.RUNS/'unit'
        plan=dict(schema=m.SCHEMA,status='CPU_plan_ready_requires_broker',phase='lr',run_id='unit',
            job_id=m.job_id('unit','lr'),run_root=str(run),execution_paths=dict(base_launcher='/launch',teacher_python='/teacher',runtime_assets='/assets'),
            commands=m.stage_commands('lr',run,'/launch','/teacher','/assets'),storage=dict(reserve_bytes=20*m.GiB),
            required_files=[],gates={'wiring':{},'metrics':{}})
        with patch.object(m,'required_paths',return_value=[]),patch.object(m,'read',return_value={}):
            self.assertEqual(m.validate_plan(plan),run)
            bad=copy.deepcopy(plan);bad['commands'][0]['argv']+=['--wiring-fixture-steps','3']
            with self.assertRaisesRegex(ValueError,'fixed production'):m.validate_plan(bad)
            bad=copy.deepcopy(plan);bad['commands'][0]['maximum_file_bytes']*=2
            with self.assertRaisesRegex(ValueError,'fixed production'):m.validate_plan(bad)
            bad=copy.deepcopy(plan);del bad['gates']['metrics']
            with self.assertRaisesRegex(ValueError,'gates are incomplete'):m.validate_plan(bad)

    def test_shared_wu_source_and_overlay_bytes_are_in_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runtime=root/'runtime';runtime.mkdir();exp=root/'experiments/static';exp.mkdir(parents=True)
            files=['config.json','manifest.json','source.json','base_python',
                'runtime/base_acceptance.json','runtime/base_v2_acceptance.json','runtime/launch_v2.py','runtime/dependency.so',
                'experiments/dynamic_sr_cull_retrain_20261009/rt.py',
                'experiments/dynamic_sr_multiview_footprint_20261007/full_refine_registered_v2.py',
                'experiments/dynamic_sr_20260918/n3dv_data.py','third_party/4dgs_843d5ac_runtime/scene/gaussian_model.py',
                'third_party/4dgs_843d5ac_runtime/simple_knn/_C.so']
            for name in files:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'{}')
            dep=m.identity(runtime/'dependency.so')
            (runtime/'copied_files.json').write_text(json.dumps({'files':[dict(destination=dep['path'],sha256=dep['sha256'],bytes=dep['bytes'])]}))
            (runtime/'git_owner_v2_acceptance.json').write_text(json.dumps(dict(status='different_owner_simulation_passed_no_sudo',launch_sha256=m.sha(runtime/'launch_v2.py'))))
            cfg=dict(manifest=m.identity(root/'manifest.json'),source_configuration=m.identity(root/'source.json'))
            with patch.multiple(m,ROOT=root,RUNTIME=runtime,EXP=exp,CONFIG=root/'config.json',BASE_PYTHON=root/'base_python'):
                result=m.required_paths('lr',cfg,runtime/'launch_v2.py','/teacher','/assets')
                for name in files:self.assertIn(root/name,result)
                (runtime/'dependency.so').write_bytes(b'changed')
                with self.assertRaises(ValueError):m.required_paths('lr',cfg,runtime/'launch_v2.py','/teacher','/assets')

    def test_source_change_between_children_stops_before_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';planfile=root/'plan.json';frozen=root/'frozen.py';frozen.write_text('first')
            nodes=[dict(name='lr_train',argv=[sys.executable,'-B','-c','from pathlib import Path;Path(__import__("sys").argv[1]).write_text("changed")',str(frozen)],additional_bytes=1,maximum_file_bytes=4096),
                   dict(name='lr_evaluate',argv=[sys.executable,'-B','-c','raise SystemExit(0)'],additional_bytes=1,maximum_file_bytes=4096)]
            plan=dict(phase='lr',job_id='safe',run_id='run',run_root=str(run),config={},required_files=[m.identity(frozen)],gates={},gpu_uuid='GPU-test',commands=nodes,storage=dict(reserve_bytes=0))
            planfile.write_text(json.dumps(plan))
            with patch.object(m,'validate_plan',return_value=run),patch.object(m,'admission',return_value={}),patch.object(m,'check_inputs',return_value=({},{})):
                with self.assertRaisesRegex(ValueError,'Frozen file differs'):m.run_plan(planfile)
            self.assertFalse((run/'lr_phase/lr_evaluate.log').exists())
            self.assertEqual(json.loads((run/'lr_phase/status.json').read_text())['status'],'failed')

    def test_real_file_limit_fails_without_truncating_into_valid_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'large'
            cmd=[sys.executable,'-B','-c','from pathlib import Path; Path(__import__("sys").argv[1]).write_bytes(b"x"*8192)',str(path)]
            result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,preexec_fn=lambda:m.file_limit(1024))
            self.assertNotEqual(result.returncode,0)
            self.assertLessEqual(path.stat().st_size,1024)

    def test_sha_identity_rejects_changed_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'asset';p.write_bytes(b'original');ident=m.identity(p);m.checked(ident)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):m.checked(ident)

    def test_cpu_or_foreign_gpu_receipts_rejected_before_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'gate.json'
            gate=dict(schema='static_sr_wu_gpu_acceptance_v1',status='passed_bounded_GPU_wiring_only',GPU_executed=False,gpu_uuid='abc',host=m.socket.gethostname())
            p.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError,'no real GPU'):m.validate_gate('wiring',m.identity(p),'GPU-abc')
            gate['GPU_executed']=True;gate['gpu_uuid']='foreign';p.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError,'wrong GPU'):m.validate_gate('wiring',m.identity(p),'GPU-abc')
            gate['gpu_uuid']='abc';gate['host']='cts';p.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError,'another host'):m.validate_gate('wiring',m.identity(p),'GPU-abc')

    def test_uuid_optional_prefix_is_normalized(self):
        self.assertEqual(m.normalize_uuid('GPU-ABC'),m.normalize_uuid('abc'))

    def test_refuses_nonbroker_execution(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-a'}),patch.object(Path,'read_text',return_value='0::/user.slice/arbitrary.service'):
            with self.assertRaisesRegex(ValueError,'matching local broker'):m.admission(dict(host=m.socket.gethostname(),gpu_uuid='GPU-a',job_id='safe'))

    def test_partial_or_fixture_lr_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'complete.json';p.write_text(json.dumps(dict(schema='static_sr_local_phase_completion_v1',status='failed',phase='lr',run_root=tmp,config={})))
            with self.assertRaisesRegex(ValueError,'successful same-run'):m.verify_lr_completion(p,Path(tmp),{})

    def test_partial_or_nonfinite_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'metrics.json';p.write_text(json.dumps(dict(schema='static_sr_wu_uniform_png_metrics_v1',scene='cook_spinach',camera='cam00',training_view_metrics=False,frames=[])))
            with self.assertRaisesRegex(ValueError,'Full cam00'):m.verify_metrics(p,Path(tmp)/'ck')

    def test_failed_child_keeps_diagnostic_and_cannot_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';planfile=root/'plan.json'
            plan=dict(phase='lr',job_id='safe',run_id='run',run_root=str(run),config={},required_files=[],gates={},
                gpu_uuid='GPU-test',commands=[dict(name='lr_train',argv=[sys.executable,'-B','-c','print("real failure evidence");raise SystemExit(9)'],additional_bytes=1,maximum_file_bytes=4096)],storage=dict(reserve_bytes=0))
            planfile.write_text(json.dumps(plan))
            with patch.object(m,'validate_plan',return_value=run),patch.object(m,'admission',return_value={}),patch.object(m,'check_inputs',return_value=({},{})):
                with self.assertRaisesRegex(RuntimeError,'code 9'):m.run_plan(planfile)
                status=json.loads((run/'lr_phase/status.json').read_text())
                self.assertEqual(status['status'],'failed')
                self.assertEqual(status['stages'][0]['returncode'],9)
                diagnostics=json.loads((run/'lr_phase/diagnostics.json').read_text())
                self.assertIn('real failure evidence',diagnostics['failure_log']['tail'])
                self.assertFalse((run/'lr_phase/complete.json').exists())
                with self.assertRaises(FileExistsError):m.run_plan(planfile)


if __name__=='__main__':unittest.main()
