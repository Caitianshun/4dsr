"""CPU-only execution/contract tests; no broker service or model is started."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.static_sr_shared_teacher_20261024 import run_splatsure_shared as runner


class SharedPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='shared_pipeline_test_', dir=runner.ROOT/'tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root/'run'
        self.launcher = runner.local.RUNTIME/'launch_v2.py'
        self.profile = dict(base_python=sys.executable, base_launcher=str(self.launcher), launcher_arguments=['--mode','base','--'])

    def stages(self):
        return runner.commands(self.run, self.profile, self.root/'config.json', self.root/'shared.json',
                               self.root/'lr_complete.json', self.root/'old_lr.pt')

    def test_recipe_uses_existing_PNGs_and_parent_only_for_contribution(self):
        stages = self.stages()
        self.assertEqual([s['name'] for s in stages], ['contribution','bundle','bundle_gate','fresh_sr_and_evaluation'])
        self.assertIn(str(self.root/'old_lr.pt'), stages[0]['argv'])
        self.assertIn(str(self.root/'shared.json'), stages[1]['argv'])
        for n in stages[1:]:
            self.assertNotIn(str(self.root/'old_lr.pt'), n['argv'])
        sr = stages[-1]['argv']
        self.assertNotIn('--checkpoint', sr)
        self.assertNotIn('--wiring-fixture-steps', sr)
        self.assertNotIn('--steps', sr)
        self.assertNotIn('stablesr', ' '.join(sr))
        self.assertEqual(sr[sr.index('--stage')+1], 'sr')
        self.assertIn('--check-only', stages[2]['argv'])
        for n in stages:
            self.assertEqual(n['argv'][:7], [sys.executable,'-s','-B',str(self.launcher),'--mode','base','--'])

    def test_plain_remote_interpreter_and_explicit_launcher_prefix(self):
        plain=dict(base_python='/task/.venv/bin/python',base_launcher=None,launcher_arguments=[])
        self.assertEqual(runner.launch(plain,'/task/script.py','--flag'),
                         ['/task/.venv/bin/python','-s','-B','/task/script.py','--flag'])
        self.assertEqual(runner.outer(self.profile,self.root/'plan.json')[-3:],['run','--plan',str(self.root/'plan.json')])

    def test_remote_job_ID_is_bounded_and_normalization_does_not_collide(self):
        names=['cook07_shared','cook07-shared','Cook07_shared','a'*120]
        values=[runner.job_id(name) for name in names]
        self.assertEqual(len(set(values)),len(names))
        for value in values:
            self.assertTrue(re.fullmatch('[a-z][a-z0-9-]{0,63}',value))
            self.assertTrue(value.startswith('ss-shared-'))

    def test_additional_storage_covers_raw_maps_and_one_evaluation_set(self):
        stages = self.stages()
        self.assertGreater(stages[0]['additional_bytes'], 6000*336*252*4)
        self.assertGreater(stages[-1]['additional_bytes'], 2*runner.GiB+300*1344*1008*3)
        self.assertEqual(stages[-1]['maximum_file_bytes'], 2*runner.GiB)
        required = sum(s['additional_bytes'] for s in stages)+20*runner.GiB
        with self.assertRaises(RuntimeError): runner.local.storage_guard(stages,0,20*runner.GiB,free=required-1)
        self.assertEqual(runner.local.storage_guard(stages,0,20*runner.GiB,free=required)['required_bytes'],required)

    def test_fresh_output_rejects_existing_directory_or_symlink_and_unsafe_ID(self):
        with patch.object(runner,'RUNS',self.root):
            self.assertEqual(runner.fresh_run('new_run'),self.root/'new_run')
            (self.root/'old').mkdir()
            (self.root/'link').symlink_to(self.root/'old')
            for name in ('old','link','../escape'):
                with self.subTest(name=name), self.assertRaises((ValueError,FileExistsError)):
                    runner.fresh_run(name)

    def fixture_plan(self, command):
        frozen=self.root/'frozen.py';frozen.write_text('# immutable fixture source\n')
        source=self.root/'plan.json'
        doc=dict(schema=runner.SCHEMA,status='CPU_plan_ready_requires_broker',run_id='new',job_id='CPU-test-only',
            config={'path':'unused','sha256':'0'*64},shared_cache={'path':'unused','sha256':'0'*64},
            lr_completion={'path':'unused','sha256':'0'*64},gates={},
            required_files=[runner.local.identity(frozen)],commands=[dict(name='first',argv=command,
            additional_bytes=1,maximum_file_bytes=runner.GiB)],storage={'reserve_bytes':20*runner.GiB})
        source.write_text(json.dumps(doc));return source,doc,frozen

    def cpu_run_context(self):
        stack=ExitStack()
        evidence=self.root/'synthetic_context.json';evidence.write_text('{}')
        stack.enter_context(patch.object(runner,'admission',return_value={'test':'CPU-only',
            'execution_context':runner.local.identity(evidence),'request':runner.local.identity(evidence)}))
        stack.enter_context(patch.object(runner,'validate_plan',return_value=(self.run,{},{})))
        stack.enter_context(patch.object(runner.local,'storage_guard',return_value={'test':'CPU-only'}))
        stack.enter_context(patch.object(runner,'product_ids',return_value=[]))
        return stack

    def test_real_child_failure_stops_and_records_diagnostics_without_replay(self):
        source,doc,_=self.fixture_plan([sys.executable,'-c','print("CPU failure evidence"); raise SystemExit(7)'])
        marker=self.root/'must_not_run'
        doc['commands'].append(dict(name='second',argv=[sys.executable,'-c',f'open({str(marker)!r},"w").write("bad")'],
            additional_bytes=1,maximum_file_bytes=runner.GiB));source.write_text(json.dumps(doc))
        with self.cpu_run_context(),self.assertRaises(RuntimeError):runner.run_plan(source)
        self.assertFalse(marker.exists())
        r=json.loads((self.run/'pipeline/status.json').read_text())
        self.assertEqual(r['status'],'failed_preserved_no_replay')
        self.assertEqual(len(r['stages']),1)
        self.assertEqual(r['stages'][0]['returncode'],7)
        d=json.loads((self.run/'pipeline/diagnostics.json').read_text())
        self.assertIn('CPU failure evidence',d['failure_log']['tail'])
        with self.cpu_run_context(),self.assertRaises(FileExistsError):runner.run_plan(source)

    def test_frozen_source_rechecked_before_second_child(self):
        source,doc,frozen=self.fixture_plan([sys.executable,'-c','pass'])
        doc['commands'][0]['argv']=[sys.executable,'-c',f'open({str(frozen)!r},"w").write("changed")']
        marker=self.root/'must_not_run'
        doc['commands'].append(dict(name='second',argv=[sys.executable,'-c',f'open({str(marker)!r},"w").write("bad")'],
            additional_bytes=1,maximum_file_bytes=runner.GiB));source.write_text(json.dumps(doc))
        with self.cpu_run_context(),self.assertRaises(ValueError):runner.run_plan(source)
        self.assertFalse(marker.exists())
        self.assertFalse((self.run/'pipeline/complete.json').exists())

    def test_admission_failure_precedes_validation_and_output(self):
        source,_,_=self.fixture_plan([sys.executable,'-c','pass'])
        with patch.object(runner,'admission',side_effect=ValueError('not broker')),patch.object(runner,'validate_plan') as validate:
            with self.assertRaises(ValueError):runner.run_plan(source)
            validate.assert_not_called()
        self.assertFalse(self.run.exists())

    def test_PRO_is_rejected_before_supervisor_or_model_access(self):
        for uuid in (runner.PRO6000,runner.PRO6000.removeprefix('GPU-')):
            with patch.object(runner.remote,'admission') as access:
                with self.assertRaises(ValueError):runner.admission(dict(broker='remote',gpu_uuid=uuid))
                access.assert_not_called()

    def test_remote_request_must_bind_actual_job_project_cwd_and_command(self):
        job_dir=self.root/'.claude-training/jobs/test-job';job_dir.mkdir(parents=True)
        request=dict(project_id='4dsr',operation='run',job_id='test-job',target_job_id='test-job',
            project_root=str(self.root),remote_cwd=str(runner.ROOT),argv=['python','frozen-script'],cuda_visible_devices='1')
        request_path=job_dir/'request.json';request_path.write_text(json.dumps(request))
        plan=dict(broker='remote',gpu_uuid='GPU-remote-test',broker_project_root=str(self.root),job_id='test-job',
                  broker_entry_argv=request['argv'],visible_device='1')
        with patch.object(runner.remote,'admission',return_value={'supervisor_context':{'project_id':'4dsr'}}):
            self.assertEqual(runner.admission(plan)['request']['sha256'],runner.local.sha(request_path))
            for field,new in [('project_id','other'),('job_id','old-job'),('remote_cwd','/different'),('argv',['python','different']),('cuda_visible_devices','0')]:
                with self.subTest(field=field):
                    request_path.write_text(json.dumps(dict(request,**{field:new})))
                    with self.assertRaises(ValueError):runner.admission(plan)

    def test_runtime_profile_requires_current_host_and_complete_frozen_environment(self):
        receipt=self.root/'CPU_imports.json';receipt.write_text('{"CPU_only":true}')
        value=dict(schema='shared_splatsure_remote_runtime_v1',status='CPU_imports_verified',broker='remote',project_id='4dsr',
            root=str(runner.ROOT),host=runner.socket.gethostname(),broker_project_root=str(runner.ROOT),gpu_uuid='GPU-a100-fixture',
            visible_device='1',base_python=sys.executable,base_launcher=None,launcher_arguments=[],
            CPU_acceptance=runner.local.identity(receipt),required_files=[runner.local.identity(receipt),runner.local.identity(sys.executable)])
        path=self.root/'runtime.json';path.write_text(json.dumps(value))
        self.assertEqual(runner.checked_profile(path),value)
        for change in (dict(gpu_uuid=runner.PRO6000),dict(host='foreign'),dict(required_files=[runner.local.identity(receipt)])):
            with self.subTest(change=change):
                path.write_text(json.dumps(dict(value,**change)))
                with self.assertRaises(ValueError):runner.checked_profile(path)

    def test_real_file_failure_cap_is_not_a_truncated_success(self):
        path=self.root/'too_large'
        r=subprocess.run([sys.executable,'-c',f'open({str(path)!r},"wb").write(b"x"*8192)'],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,preexec_fn=lambda:runner.local.file_limit(1024))
        self.assertNotEqual(r.returncode,0)
        self.assertLessEqual(path.stat().st_size,1024)


if __name__=='__main__':unittest.main()
