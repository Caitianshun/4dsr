"""CPU contracts only; all fabricated receipts stay inside temporary directories."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from deployment.static_sr_local_20261024 import run_splatsure_from_shards as m


class ContinuationTests(unittest.TestCase):
    def test_commands_reuse_teachers_and_parent_only_for_maps(self):
        nodes=m.stage_commands(Path('/out'),'/launch','/aggregate.json')
        self.assertEqual([n['name'] for n in nodes],['contribution','bundle','bundle_gate','sr_train_and_evaluate'])
        self.assertIn('/out/lr/checkpoint.pt',nodes[0]['argv'])
        self.assertIn('/aggregate.json',nodes[1]['argv'])
        self.assertEqual(nodes[1]['argv'].count('--model'),2)
        self.assertIn('--check-only',nodes[2]['argv'])
        self.assertNotIn('--checkpoint',nodes[-1]['argv'])
        self.assertNotIn('--steps',str(nodes));self.assertNotIn('wiring-fixture',str(nodes))
        self.assertNotIn('splatsure_stablesr.py',str(nodes))
        self.assertEqual(nodes[-1]['maximum_file_bytes'],2*m.GiB)

    def test_outer_launcher_is_the_broker_registered_python(self):
        argv=m.outer_argv('/launch','/plan')
        self.assertEqual(argv[:6],[str(m.shared.BASE_PYTHON),'-s','/launch','--mode','base','--'])
        self.assertEqual(argv[-3:],['run','--plan','/plan'])

    def test_remaining_budget_and_cap_are_failure_boundaries(self):
        nodes=m.stage_commands('/out','/launch','/aggregate')
        need=sum(n['additional_bytes'] for n in nodes)+20*m.GiB
        with self.assertRaises(RuntimeError):m.shared.storage_guard(nodes,0,20*m.GiB,free=need-1)
        self.assertEqual(m.shared.storage_guard(nodes,0,20*m.GiB,free=need)['required_bytes'],need)
        self.assertLess(need,32*m.GiB)

    def test_existing_failed_partial_or_old_full_phase_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in m.OUTPUTS:
                with self.subTest(name=name):
                    p=root/name;p.touch()
                    with self.assertRaises(FileExistsError):m.assert_fresh(root)
                    p.unlink()
            target=root/'target';target.mkdir();link=root/'link';link.symlink_to(target)
            with self.assertRaises(ValueError):m.assert_fresh(link)

    def test_task_local_operational_identity_rejects_symlink_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inside=root/'task';inside.mkdir();asset=inside/'a';asset.write_bytes(b'a')
            other=root/'outside';other.write_bytes(b'b');link=inside/'link';link.symlink_to(asset)
            with patch.object(m,'ROOT',inside):
                self.assertEqual(m.local_file(m.identity(asset)),asset)
                for p in (other,link):
                    with self.assertRaises(ValueError):m.local_file(m.identity(p))

    def test_producer_relative_identity_matches_absolute_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'config.json';p.write_text('{}')
            with patch.object(m.shared,'ROOT',root):
                self.assertTrue(m.same_file(dict(path='config.json',sha256=m.sha(p)),m.identity(p)))
                p.write_text('{"changed":true}')
                with self.assertRaises(ValueError):m.same_file(dict(path='config.json',sha256='0'*64),m.identity(p))

    def test_aggregate_refuses_fixture_wrong_count_and_schema_before_recompute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'receipt.json'
            b=types.SimpleNamespace(RECIPE='recipe')
            good=dict(schema='wu_splatsure_stablesr_teacher_v1',status='completed',method='splatsure',recipe='recipe',
                      scene='cook_spinach',training_only=True,parameter_updates=0,HR_images_read=0,heldout_images_read=0,
                      sharding=dict(schema='stablesr_aggregated_shards_v1',target_count=6000,original_receipts_preserved=True,rng_policy=m.shards.RNG))
            bads=[]
            for field,value in [('status','fixture_passed'),('formal_prior_allowed',False),('HR_images_read',1),('recipe','SwinIR')]:
                r=copy.deepcopy(good);r[field]=value;bads.append(r)
            r=copy.deepcopy(good);r['sharding']['target_count']=20;bads.append(r)
            with patch.object(m,'ROOT',root),patch.object(m.shards,'load_bridge',return_value=b),patch.object(m.shards,'verify_aggregate') as verify:
                for r in bads:
                    p.write_text(json.dumps(r))
                    with self.assertRaisesRegex(ValueError,'Complete formal'):m.validate_aggregate(p,{})
                verify.assert_not_called()

    def test_formal_parent_validation_is_required_not_a_path_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'completion.json'
            p.write_text(json.dumps(dict(schema='static_sr_local_phase_completion_v1',status='failed',phase='lr',run_root=str(root),config={})))
            with self.assertRaisesRegex(ValueError,'successful same-run'):m.validate_lr(p,root,{})

    def test_plan_changes_or_missing_closure_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';planpath=root/'plan.json';cfg=root/'config.json';cfg.write_text('{}');aggregate=root/'aggregate.json';aggregate.write_text('{}')
            frozen=root/'frozen.py';frozen.write_text('source')
            p=dict(schema=m.SCHEMA,status='CPU_plan_ready_requires_broker',phase=m.PHASE,run_id='run',
                job_id=m.shared.job_id('run',m.PHASE),run_root=str(run),gpu_uuid=m.PRO6000,config=m.identity(cfg),
                execution_paths=dict(base_launcher='/launch'),aggregate_receipt=m.identity(aggregate),lr_completion=m.identity(cfg),
                commands=m.stage_commands(run,'/launch',aggregate),storage=dict(reserve_bytes=20*m.GiB),
                gates={'wiring':{},'metrics':{}},plan_path=str(planpath),broker_entry_argv=m.outer_argv('/launch',planpath),
                required_files=[m.identity(frozen)])
            # CONFIG participates in fixed argv as well; rebuild under the same patched constant.
            with patch.multiple(m,CONFIG=cfg,RUNS=root),patch.object(m,'required_paths',return_value=[frozen]),patch.object(m,'validate_lr',return_value=[]),patch.object(m,'validate_aggregate',return_value=({},[])):
                p['commands']=m.stage_commands(run,'/launch',aggregate)
                self.assertEqual(m.validate_plan(p,planpath),run)
                for mutate in (lambda x:x['commands'][-1]['argv'].extend(['--wiring-fixture-steps','3']),
                               lambda x:x['commands'][0].update(maximum_file_bytes=3*m.GiB),
                               lambda x:x['storage'].update(reserve_bytes=0),lambda x:x['gates'].pop('metrics'),
                               lambda x:x.update(broker_entry_argv=['python','run']),lambda x:x.update(gpu_uuid='GPU-other'),
                               lambda x:x.update(required_files=[])):
                    q=copy.deepcopy(p);mutate(q)
                    with self.assertRaises(ValueError):m.validate_plan(q,planpath)

    def test_failed_child_records_diagnostic_and_replay_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();planpath=root/'plan.json'
            p=dict(job_id='unit',config={},aggregate_receipt={},lr_completion={},gates={},gpu_uuid=m.PRO6000,required_files=[],
                commands=[dict(name='contribution',argv=[sys.executable,'-B','-c','print("bounded CPU failure");raise SystemExit(7)'],
                               additional_bytes=1,maximum_file_bytes=4096)],storage=dict(reserve_bytes=0))
            planpath.write_text(json.dumps(p))
            with patch.object(m,'validate_plan',return_value=run),patch.object(m.shared,'admission',return_value={}),patch.object(m.shared,'check_inputs',return_value=({},{})):
                with self.assertRaisesRegex(RuntimeError,'code 7'):m.run_plan(planpath)
                receipt=m.read(run/'shards_sr_phase/status.json')
                self.assertEqual(receipt['status'],'failed');self.assertEqual(receipt['stages'][0]['returncode'],7)
                self.assertIn('bounded CPU failure',m.read(run/'shards_sr_phase/diagnostics.json')['failure_log']['tail'])
                self.assertFalse((run/'shards_sr_phase/complete.json').exists())
                with self.assertRaises(FileExistsError):m.run_plan(planpath)

    def test_source_or_generated_product_change_stops_before_next_child(self):
        for change_source in (True,False):
            with self.subTest(source=change_source),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);run=root/'run';run.mkdir();source=root/'source.py';source.write_text('first');planpath=root/'plan.json'
                first=dict(name='contribution',argv=[sys.executable,'-B','-c','pass'],additional_bytes=1,maximum_file_bytes=4096)
                second=dict(first,name='bundle')
                p=dict(job_id='unit',config={},aggregate_receipt={},lr_completion={},gates={},gpu_uuid=m.PRO6000,
                       required_files=[m.identity(source)] if change_source else [],commands=[first,second],storage=dict(reserve_bytes=0))
                planpath.write_text(json.dumps(p))
                def finish(*args):
                    old=m.identity(source);source.write_text('changed');return [] if change_source else [old]
                with patch.object(m,'validate_plan',return_value=run),patch.object(m.shared,'admission',return_value={}),patch.object(m.shared,'check_inputs',return_value=({},{})),patch.object(m,'validate_products',side_effect=finish):
                    with self.assertRaisesRegex(ValueError,'Frozen file differs'):m.run_plan(planpath)
                self.assertFalse((run/'shards_sr_phase/bundle.log').exists())

    def test_nonbroker_execution_and_foreign_gpu_rejected(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=m.PRO6000),patch.object(Path,'read_text',return_value='0::/user.slice/arbitrary.service'):
            with self.assertRaisesRegex(ValueError,'matching local broker'):
                m.shared.admission(dict(host=m.socket.gethostname(),gpu_uuid=m.PRO6000,job_id='unit'))

    def test_module_import_is_cpu_only(self):
        command=[sys.executable,'-B','-c','import sys;from deployment.static_sr_local_20261024 import run_splatsure_from_shards;assert "torch" not in sys.modules']
        result=subprocess.run(command,cwd=m.ROOT,capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
