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

SPEC=importlib.util.spec_from_file_location('local_sequence_runner',Path(__file__).with_name('run_sequence.py'))
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
sys.path.insert(0,str(m.EXP))
from methods.sequence_matters import UPSTREAM_COMMIT


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class RunnerTests(unittest.TestCase):
    def test_exact_sequence_teacher_then_fresh_sr_and_builtin_evaluation(self):
        nodes=m.stage_commands('/run','/launcher','/teacher')
        self.assertEqual([x['name'] for x in nodes],['teacher_acceptance','teacher_generate','bundle','bundle_gate','sr_train_and_evaluate'])
        self.assertIn('--frame',nodes[0]['argv']);self.assertIn('0',nodes[0]['argv'])
        self.assertIn('--records',nodes[0]['argv']);self.assertIn('--manifest',nodes[1]['argv'])
        for node in nodes[:2]:
            self.assertIn(str(m.CPU_PLAN),node['argv']);self.assertIn(str(m.PSRT),node['argv']);self.assertIn(str(m.SPYNET),node['argv'])
            self.assertNotIn('--plan-only',node['argv'])
        self.assertEqual(nodes[2]['argv'].count('--model'),2)
        self.assertIn('/run/teacher/receipt.json',nodes[2]['argv'])
        self.assertNotIn('/run/teacher_acceptance/teacher/receipt.json',nodes[2]['argv'])
        self.assertIn('--check-only',nodes[-2]['argv']);self.assertIn('sr',nodes[-1]['argv'])
        for forbidden in ('--checkpoint','--steps','--wiring-fixture-steps'):
            self.assertNotIn(forbidden,nodes[-1]['argv'])

    def test_complete_disk_bound_keeps_twenty_gib_reserve(self):
        nodes=m.stage_commands('/run','/launcher','/teacher')
        total=20*m.GiB+sum(n['additional_bytes'] for n in nodes)
        self.assertEqual(total,61115334656)
        with self.assertRaises(RuntimeError):m.shared.storage_guard(nodes,0,20*m.GiB,free=total-1)
        self.assertEqual(m.shared.storage_guard(nodes,0,20*m.GiB,free=total)['required_bytes'],total)

    def test_broker_identity_and_nonbroker_refusal(self):
        jid=m.shared.job_id('cook07-sequence-v1','sequence')
        self.assertRegex(jid,r'^[a-z][a-z0-9-]{0,63}$')
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-a'}),patch.object(Path,'read_text',return_value='0::/user.slice/other.service'):
            with self.assertRaisesRegex(ValueError,'matching local broker'):
                m.shared.admission(dict(host=socket.gethostname(),gpu_uuid='GPU-a',job_id=jid))

    def test_sequence_wiring_required_in_addition_to_common_gate(self):
        with patch.object(m.shared,'validate_gate',return_value={'results':[dict(scene='Cook07',method='sequence_matters',status='failed')]}):
            with self.assertRaisesRegex(ValueError,'Sequence Matters wiring'):m.validate_gate('wiring',{},'GPU-a')

    def plan(self):
        run=m.RUNS/'cpu_contract'
        return dict(schema=m.SCHEMA,status='CPU_plan_ready_requires_broker',run_id='cpu_contract',job_id=m.shared.job_id('cpu_contract','sequence'),
            run_root=str(run),execution_paths=dict(base_launcher='/launch',teacher_python='/teacher'),
            commands=m.stage_commands(run,'/launch','/teacher'),storage=dict(reserve_bytes=20*m.GiB),gates={'wiring':{},'metrics':{}},
            config={},records={},CPU_plan={},required_files=[])

    def test_commands_and_budget_are_immutable_and_blocked_plan_cannot_run(self):
        plan=self.plan()
        with patch.object(m,'identity',return_value={}),patch.object(m,'required_paths',return_value=[]),patch.object(m,'read',return_value={}):
            self.assertEqual(m.validate_plan(plan),m.RUNS/'cpu_contract')
            for mutate in (lambda p:p['commands'][-1]['argv'].extend(['--wiring-fixture-steps','3']),
                           lambda p:p['commands'][1].update(additional_bytes=1),
                           lambda p:p['storage'].update(reserve_bytes=0),
                           lambda p:p.update(status='blocked_storage')):
                bad=copy.deepcopy(plan);mutate(bad)
                with self.assertRaises(ValueError):m.validate_plan(bad)

    def teacher_fixture(self,root,count=20):
        from PIL import Image
        run=root/'run';out=run/'teacher_acceptance';teacher=out/'teacher';up=root/'upstream'
        (up/'vsr').mkdir(parents=True);(up/'vsr/utils_vsr.py').write_text('source')
        model=root/'psrt.pth';model.write_bytes(b'weights');models={str(model):m.sha(model)}
        rows=[dict(key=f'cam{i:02d}/0000',camera=f'cam{i:02d}',frame=0,time=0.0,lr_sha256='source') for i in range(1,count+1)]
        records=dict(records=rows,heldout_cameras=['cam00'])
        sequences=[dict(indices=list(range(i,min(i+4,count)))) for i in range(0,count,4)]
        group=dict(time=0.0,keys=[r['key'] for r in rows],sequences=sequences,lr_sha256={r['key']:'source' for r in rows})
        plan=dict(groups=[group]);reg=root/'records.json';orig=root/'plan.json'
        save(reg,records);save(orig,plan);save(out/'fixture_records.json',records);save(out/'fixture_plan.json',plan)
        targets={};hashes={};outputs=[]
        for row in rows:
            path=teacher/(row['key']+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(1344,1008),(1,2,3)).save(path)
            targets[row['key']]=str(path);hashes[row['key']]=m.sha(path)
            outputs.append(dict(key=row['key'],**m.identity(path),size_wh=[1344,1008]))
        producer=dict(status='teacher_complete',teacher_generated=True,manifest_sha256=m.canonical(records),
            upstream_commit=UPSTREAM_COMMIT,source_sha256=m.sha(up/'vsr/utils_vsr.py'),weights=models,groups=[group],
            weight_key_audit=dict(missing_parameters=[]),targets=targets,target_sha256=hashes)
        save(teacher/'receipt.json',producer)
        result=dict(schema='static_sr_wu_teacher_acceptance_fixture_v1',status='fixture_passed',method='sequence_matters',formal_prior_allowed=False,
            formal_parent_allowed=False,teacher_generated=True,gpu_executed=True,hostname=socket.gethostname(),hardware=dict(gpu_uuid='abc'),
            wrapper=m.identity(m.EXP/'teacher_acceptance.py'),registered_records=m.identity(reg),original_sequence_plan=m.identity(orig),
            fixture_records=m.identity(out/'fixture_records.json'),teacher_receipt=m.identity(teacher/'receipt.json'),
            adapter_sources=[m.identity(m.EXP/'methods/sequence_matters.py')],model_identities=[m.identity(model)],
            registered_manifest_canonical_sha256=m.canonical(records),fixture_manifest_canonical_sha256=m.canonical(records),
            selected_keys=group['keys'],outputs=outputs,forward_calls=[dict(keys=[group['keys'][i] for i in s['indices']],
                input_shape=[1,len(s['indices']),3,252,336],output_shape=[1,len(s['indices']),3,1008,1344],finite_tensors_checked=1,elapsed_seconds=1.0) for s in sequences])
        save(out/'acceptance.json',result)
        context=patch.multiple(m,UPSTREAM=up,MODELS=models,RECORDS=reg,CPU_PLAN=orig)
        return context,run,records,plan,producer,result

    def test_genuine_producer_bytes_and_fixture_order_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            context,run,records,plan,producer,result=self.teacher_fixture(Path(tmp))
            with context:
                m.validate_fixture(run,'GPU-abc',records,plan)
                changed=copy.deepcopy(result);changed['forward_calls'][0]['keys'].reverse()
                save(run/'teacher_acceptance/acceptance.json',changed)
                with self.assertRaisesRegex(ValueError,'input order'):m.validate_fixture(run,'GPU-abc',records,plan)
                changed=copy.deepcopy(result);changed['forward_calls'][0]['output_shape'][-1]=1343
                save(run/'teacher_acceptance/acceptance.json',changed)
                with self.assertRaisesRegex(ValueError,'dimensions'):m.validate_fixture(run,'GPU-abc',records,plan)
                changed=copy.deepcopy(result);changed['forward_calls'][0]['elapsed_seconds']=float('nan')
                save(run/'teacher_acceptance/acceptance.json',changed)
                with self.assertRaises(ValueError):m.validate_fixture(run,'GPU-abc',records,plan)

    def test_fixture_cannot_replace_formal_and_tampered_targets_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            context,run,records,plan,producer,result=self.teacher_fixture(Path(tmp),1)
            path=run/'teacher_acceptance/teacher/receipt.json';output=path.parent
            with context:
                m.validate_teacher(path,records,plan,output)
                bad=copy.deepcopy(producer);bad['status']='fixture_passed';save(path,bad)
                with self.assertRaisesRegex(ValueError,'incomplete'):m.validate_teacher(path,records,plan,output)
                bad=copy.deepcopy(producer);bad['weight_key_audit']['missing_parameters']=['net.weight'];save(path,bad)
                with self.assertRaisesRegex(ValueError,'completely loaded'):m.validate_teacher(path,records,plan,output)
                bad=copy.deepcopy(producer);bad['targets']['cam00/0000']=next(iter(bad['targets'].values()));save(path,bad)
                with self.assertRaisesRegex(ValueError,'held-out'):m.validate_teacher(path,records,plan,output)
                save(path,producer);Path(next(iter(producer['targets'].values()))).write_bytes(b'tampered')
                with self.assertRaises(ValueError):m.validate_teacher(path,records,plan,output)

    def test_partial_fixture_or_foreign_hardware_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            context,run,records,plan,producer,result=self.teacher_fixture(Path(tmp),1)
            with context:
                with self.assertRaisesRegex(ValueError,'frame-zero'):m.validate_fixture(run,'GPU-abc',records,plan)
                with self.assertRaisesRegex(ValueError,'different host/GPU'):m.validate_fixture(run,'GPU-other',records,plan)

    def execution(self,root,nodes,required=()):
        run=root/'run';path=root/'plan.json'
        save(path,dict(job_id='safe',config={},required_files=list(required),gates={},gpu_uuid='GPU-a',commands=nodes,storage=dict(reserve_bytes=0)))
        return run,path

    def test_source_change_stops_before_second_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.py';source.write_text('original')
            nodes=[dict(name='cpu_first',argv=[sys.executable,'-B','-c','from pathlib import Path;Path(__import__("sys").argv[1]).write_text("changed")',str(source)],additional_bytes=1,maximum_file_bytes=4096),
                   dict(name='cpu_second',argv=[sys.executable,'-B','-c','raise SystemExit(0)'],additional_bytes=1,maximum_file_bytes=4096)]
            run,path=self.execution(root,nodes,[m.identity(source)])
            with patch.object(m,'validate_plan',return_value=run),patch.object(m.shared,'admission',return_value={}),patch.object(m,'check_inputs',return_value=({},{},{},{})):
                with self.assertRaisesRegex(ValueError,'Frozen file differs'):m.run_plan(path)
            self.assertFalse((run/'execution/cpu_second.log').exists());self.assertEqual(m.read(run/'execution/status.json')['status'],'failed')

    def test_failed_fixture_stops_before_formal_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);nodes=[dict(name=n,argv=[sys.executable,'-B','-c','raise SystemExit(0)'],additional_bytes=1,maximum_file_bytes=4096) for n in ('teacher_acceptance','teacher_generate')]
            run,path=self.execution(root,nodes)
            with patch.object(m,'validate_plan',return_value=run),patch.object(m.shared,'admission',return_value={}),patch.object(m,'check_inputs',return_value=({},{},{},{})),patch.object(m,'validate_fixture',side_effect=ValueError('fixture failed')):
                with self.assertRaisesRegex(ValueError,'fixture failed'):m.run_plan(path)
            self.assertFalse((run/'execution/teacher_generate.log').exists())

    def test_failed_formal_receipt_stops_before_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);nodes=[dict(name=n,argv=[sys.executable,'-B','-c','raise SystemExit(0)'],additional_bytes=1,maximum_file_bytes=4096) for n in ('teacher_generate','bundle')]
            run,path=self.execution(root,nodes)
            with patch.object(m,'validate_plan',return_value=run),patch.object(m.shared,'admission',return_value={}),patch.object(m,'check_inputs',return_value=({},{},{},{})),patch.object(m,'validate_teacher',side_effect=ValueError('incomplete teacher')):
                with self.assertRaisesRegex(ValueError,'incomplete teacher'):m.run_plan(path)
            self.assertFalse((run/'execution/bundle.log').exists())

    def test_failure_keeps_log_evidence_and_refuses_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);nodes=[dict(name='cpu_failure',argv=[sys.executable,'-B','-c','print("failure evidence");raise SystemExit(9)'],additional_bytes=1,maximum_file_bytes=4096)]
            run,path=self.execution(root,nodes)
            with patch.object(m,'validate_plan',return_value=run),patch.object(m.shared,'admission',return_value={}),patch.object(m,'check_inputs',return_value=({},{},{},{})):
                with self.assertRaisesRegex(RuntimeError,'code 9'):m.run_plan(path)
                self.assertEqual(m.read(run/'execution/status.json')['status'],'failed')
                self.assertIn('failure evidence',m.read(run/'execution/diagnostics.json')['failure_log']['tail'])
                self.assertFalse((run/'execution/complete.json').exists())
                with self.assertRaises(FileExistsError):m.run_plan(path)


if __name__=='__main__':unittest.main()
