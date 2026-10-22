"""Synthetic CPU evidence only; no GPU, submission, or real experiment output."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('prefix_test_planner', Path(__file__).with_name('plan_scene.py'))
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)
q = p.scene_prefix


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.root_patch = patch.object(p, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def put(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value) if not isinstance(value, str) else value)
        return p.identity(target)

    def fixture(self, terminal=True):
        manifest = self.put('input/manifest.json', dict(scene='meetroom_discussion', observations=[
            dict(camera_id='cam01', frame_index=0, time=0., split='train', lr_sha256='one_lr')]))
        source = self.put('source.py', 'frozen CPU fixture source')
        weight = self.put('weight.pt', 'fixed generic fixture weight')
        configs = {m:self.put('prepared/Discussion07/'+m+'.json',
                             dict(scene='meetroom_discussion', method=m, manifest=manifest)) for m in p.METHODS}
        run, nodes, _, _ = p.build_dag('Discussion07', 'unit', 'prepared', 'als', 'assets', sys.executable,
                                      '/unused/stable', '/unused/sequence', 'npz_byte_shuffle')
        for n in nodes:
            n['argv'] = [x.replace('__REGISTERED_MANIFEST__', '{ROOT}/input/manifest.json') for x in n['argv']]
        plan = dict(schema='static_sr_scene_execution_plan_v1', status='executable_after_resource_admission',
                    phase='ie_prepare', scene='Discussion07', run_id='unit', run_root=run,
                    configs=configs, LR_identity=manifest, prefix_dag=nodes[:4], dag=nodes[:4],
                    gpu_uuid='GPU-unit', base_python=sys.executable, env={}, reserve_bytes=0,
                    required_files=[source, manifest, *configs.values()],
                    storage={n['id']:dict(known_payload_lower_bound_bytes=0, additional_bytes=1000000,
                                         maximum_file_bytes=1000000) for n in nodes[:4]})
        args = SimpleNamespace(out='cts_job_specs/prefix', base_python=sys.executable,
                               shared_project_root='/home/cts/Project/4DSR')
        p.materialize(args, plan)
        plan_path = self.root/'cts_job_specs/prefix/plan.json'
        job = p.read('cts_job_specs/prefix/job.json')
        log = run+'/job_logs_ie_prepare'
        job_id = self.put(log+'/job.json', job)
        self.put(run+'/iesrgs/internal_train/checkpoint.pt', 'synthetic CPU checkpoint')
        ck = p.identity(run+'/iesrgs/internal_train/checkpoint.pt')
        self.put(run+'/iesrgs/internal_train/complete.json', dict(status='stage_completed', wiring_fixture=False,
                 stage='internal', steps=30000, config=configs['iesrgs'], checkpoint=ck))
        entry = dict(camera='cam01', frame=0, time=0., lr_sha256='one_lr')
        ext = {k:self.put(run+'/iesrgs/external/cam01/0000_'+k+'.npz', k)
               for k in ('external_rgb','external_inverse_depth')}
        internal = {k:self.put(run+'/iesrgs/internal_cache/cam01/0000_'+k+'.npz', k)
                    for k in ('internal_rgb','internal_depth')}
        base = dict(schema='static_sr_wu_prior_bundle_v1', manifest=manifest, scene='meetroom_discussion',
                    training_only=True, producer=[source], model_identities=[weight])
        self.put(run+'/iesrgs/external/index.json', dict(base, kind='iesrgs_external', entries=[dict(entry, **ext)]))
        self.put(run+'/iesrgs/internal_cache/index.json', dict(base, kind='iesrgs_internal',
                  internal_teacher=dict(checkpoint=ck), entries=[dict(entry, **internal)]))
        self.put(run+'/iesrgs/bundle.json', dict(base, kind='iesrgs_complete', internal_teacher=dict(checkpoint=ck),
                                               entries=[dict(entry, **ext, **internal)]))
        for node in plan['dag']:
            self.put(run+'/prefix_stage_receipts/'+node['id']+'.json', dict(argv=node['argv'], returncode=0,
                     maximum_file_bytes=plan['storage'][node['id']]['maximum_file_bytes']))
        stages = [dict(name=c['name'], argv=[x.replace('{ROOT}', str(self.root)) for x in c['argv']], returncode=0)
                  for c in job['commands']]
        state = dict(status='running', scene='Discussion07', gpu_uuid='GPU-unit', job_id=job['job_id'],
                     job_sha256=job_id['sha256'], stages=copy.deepcopy(stages))
        state['stages'][-1].pop('returncode')
        self.put(log+'/status.json', state)
        q.seal(p, plan_path)
        if terminal:
            state.update(status='completed', stages=stages)
            self.put(log+'/status.json', state)
            self.put(log+'/complete.json', state)
        return plan, run+'/ie_prefix_completion.json', plan_path

    def test_exact_complete_prefix_seals_and_reuses_without_any_child_process(self):
        plan, receipt, _ = self.fixture()
        with patch.object(q.subprocess, 'run', side_effect=AssertionError('No process when validating')):
            result = q.validate(p, receipt, plan)
        self.assertEqual(result['reused_stages'], list(q.STAGES))
        self.assertGreater(result['retained_bytes'], 0)
        self.assertFalse(p.read(receipt)['full_scene_completed'])

    def test_missing_terminal_and_failed_stage_cannot_resume(self):
        plan, receipt, _ = self.fixture(terminal=False)
        with self.assertRaises(FileNotFoundError): q.validate(p, receipt, plan)
        log = plan['run_root']+'/job_logs_ie_prepare'
        state = p.read(log+'/status.json')
        state['status'] = 'completed'
        state['stages'][-1]['returncode'] = 1
        self.put(log+'/status.json', state); self.put(log+'/complete.json', state)
        with self.assertRaisesRegex(ValueError, 'incomplete'): q.validate(p, receipt, plan)

    def test_changed_artifact_or_source_is_rejected(self):
        plan, receipt, _ = self.fixture()
        target = plan['run_root']+'/iesrgs/external/cam01/0000_external_rgb.npz'
        original = (self.root/target).read_bytes()
        (self.root/target).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifacts changed'): q.validate(p, receipt, plan)
        (self.root/target).write_bytes(original)
        (self.root/'source.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Identity changed'): q.validate(p, receipt, plan)

    def test_scene_run_config_command_and_partial_remaining_outputs_are_rejected(self):
        plan, receipt, _ = self.fixture()
        for key, value in [('scene','Cook07'), ('run_id','another'), ('configs',{}), ('prefix_dag',[])]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'different plan'):
                q.validate(p,receipt,dict(plan,**{key:value}))
        (self.root/plan['run_root']/'splatsure').mkdir()
        with self.assertRaisesRegex(ValueError,'beyond the sealed prefix'): q.validate(p,receipt,plan)

    def test_link_and_traversal_outputs_are_rejected(self):
        plan, receipt, _ = self.fixture()
        (self.root/plan['run_root']/'iesrgs/external/foreign').symlink_to(self.root/'weight.pt')
        with self.assertRaisesRegex(ValueError,'symlink'): q.validate(p,receipt,plan)
        with self.assertRaisesRegex(ValueError,'outside'):
            q.safe_output(p,plan['run_root']+'/../foreign',plan['run_root'])

    def test_terminal_argv_and_successful_stage_receipt_are_bound(self):
        plan, receipt, _ = self.fixture()
        log = plan['run_root']+'/job_logs_ie_prepare'
        state = p.read(log+'/complete.json'); state['stages'][1]['argv'].append('--changed')
        self.put(log+'/complete.json',state);self.put(log+'/status.json',state)
        with self.assertRaisesRegex(ValueError,'executed command'):q.validate(p,receipt,plan)

    def test_prefix_job_has_exact_four_commands_no_final_evaluation_or_teacher_execution(self):
        plan, _, plan_path = self.fixture()
        job = p.read(plan_path.with_name('job.json'))
        self.assertEqual(job['commands'],q.expected_commands(p,plan,plan_path))
        self.assertEqual(len(job['commands']),9)
        self.assertEqual(job['out'],plan['run_root']+'/job_logs_ie_prepare')
        self.assertFalse(any('checkpoint.pt' in x or x.endswith('.npz') for x in job['return_paths']))
        self.assertNotIn('collect_compact_results',[x['name'] for x in job['commands']])

    def test_prefix_requires_all_six_real_acceptance_categories_like_full(self):
        # Mock only external evidence validation; exercise real planner gate
        # selection. These tiny records are never written as operational plans.
        for source in (p.SELF,q.SELF,'deployment/static_sr_cts_20261022/run_job.py'):
            self.put(source,'source fixture')
        row=dict(camera_id='cam01',frame_index=0,time=0.,split='train',lr_sha256='one_lr')
        manifest=dict(scene='meetroom_discussion',frame_indices=list(range(300)),
                      splits={'test':['cam00'],'train':['cam01']},resolutions={'lr':[320,180],'hr':[1280,720]},observations=[row])
        mid=self.put('input/manifest.json',manifest)
        source=self.put('source_configuration.json',{})
        for method in p.METHODS:
            lr,sr=(0,30000) if method=='sequence_matters' else (30000,20000 if method=='threedsr' else 30000)
            self.put('prepared/Discussion07/'+method+'.json',dict(method=method,scene='meetroom_discussion',manifest=mid,
                source_configuration=source,teacher_variant='stablesr',specification=dict(lr_steps=lr,sr_steps=sr)))
        self.put('prepared/Discussion07/training_records.json',{})
        for path in ('base_python','stable_python','sequence_python','als.json','assets.json'):
            self.put(path,{})
        receipts={kind:self.put('acceptance/'+kind+'.json',dict(method=kind))['path']
                  for kind in ('wiring','external','metrics','stablesr','sequence_matters','threedsr')}
        args=SimpleNamespace(phase='ie_prepare',completed_ie_prefix=None,prepared='prepared',scene='Discussion07',
            run_id='gate_check',sequence_plan='als.json',runtime_assets='assets.json',base_python='base_python',
            stable_python='stable_python',sequence_python='sequence_python',iesrgs_cache_format='npz_byte_shuffle',
            gpu_uuid='GPU-unit',wiring_receipt=receipts['wiring'],external_receipt=receipts['external'],
            metric_receipt=receipts['metrics'],teacher_receipt=[receipts[k] for k in ('stablesr','sequence_matters','threedsr')],
            budget=None,reserve_bytes=20*(1<<30))
        common=SimpleNamespace(training_manifest=lambda *a,**kw:(manifest,[row]),validate_records_bundle=lambda *a:None)
        sequence=SimpleNamespace(validate_sequence_plan=lambda *a:None)
        with patch.dict(sys.modules,{'common':common,'methods.sequence_matters':sequence}), \
             patch.object(p,'validate_acceptance'),patch.object(p,'check_disk',return_value={'passed':True}):
            ready=p.create_plan(args)
            self.assertEqual(ready['status'],'executable_after_resource_admission')
            self.assertEqual([n['id'] for n in ready['dag']],list(q.STAGES))
            self.assertEqual({r['kind'] for r in ready['accepted_receipts']},set(receipts))
            for missing in receipts:
                altered=copy.deepcopy(args)
                if missing in ('wiring','external','metrics'):
                    setattr(altered,{'wiring':'wiring_receipt','external':'external_receipt','metrics':'metric_receipt'}[missing],None)
                else:altered.teacher_receipt.remove(receipts[missing])
                with self.subTest(missing=missing):
                    blocked=p.create_plan(altered)
                    self.assertEqual(blocked['status'],'blocked')
                    self.assertTrue(any(b.startswith(missing+':') for b in blocked['blockers']))

    def test_full_continuation_skips_all_four_scientific_stages_and_keeps_barriers(self):
        plan, receipt, _ = self.fixture()
        reused = q.validate(p,receipt,plan)
        _,nodes,_,_=p.build_dag('Discussion07','unit','prepared','als','assets',sys.executable,'/stable','/seq','npz_byte_shuffle')
        full = dict(plan,phase='full',dag=nodes[4:],completed_ie_prefix=reused,
                    storage={n['id']:dict(known_payload_lower_bound_bytes=0,additional_bytes=1) for n in nodes[4:]})
        args=SimpleNamespace(out='cts_job_specs/full',base_python=sys.executable,shared_project_root='/home/cts/Project/4DSR')
        p.materialize(args,full)
        job=p.read('cts_job_specs/full/job.json')
        names=[x['name'] for x in job['commands']]
        self.assertEqual(names[0],'verify_completed_ie_prefix')
        self.assertTrue(set(q.STAGES).isdisjoint(names))
        self.assertIn('iesrgs_bundle_gate',names)
        self.assertEqual(names[-1],'collect_compact_results')

    def test_prefix_raw_budget_does_not_assume_compression_or_old_checkpoint_size(self):
        for source in (p.SELF,q.SELF):self.put(source,'source bound in CPU fixture')
        nodes=[dict(id=s) for s in q.STAGES]
        budget=q.prefix_budget(p,nodes,3600,[1280,720])
        stages=budget['stages']
        self.assertGreater(stages['ie_external']['additional_bytes'],53084160000)
        self.assertEqual(stages['ie_internal_train']['additional_bytes'],33*(1<<30))
        self.assertLess(sum(s['additional_bytes'] for s in stages.values())+20*(1<<30),213*(1<<30))

    def test_file_limit_really_fails_oversized_cpu_write_and_records_exit(self):
        run='output/static_sr_wu_20261022/cts_scene_runs/Discussion07/cpu_limit'
        plan=dict(phase='ie_prepare',status='executable_after_resource_admission',run_root=run,dag=[dict(id='ie_external',argv=[sys.executable,'-c',
             "from pathlib import Path; Path('too_large').write_bytes(b'x'*4096)"])],
             storage={'ie_external':{'additional_bytes':4096,'maximum_file_bytes':1024}})
        self.put('limit_plan.json',plan)
        with patch.object(p,'guard'):
            with self.assertRaisesRegex(RuntimeError,'scientific command failed'):
                q.run_stage(p,'limit_plan.json','ie_external')
        self.assertLessEqual((self.root/'too_large').stat().st_size,1024)
        receipt=p.read(run+'/prefix_stage_receipts/ie_external.json')
        self.assertNotEqual(receipt['returncode'],0)
        self.assertEqual(receipt['maximum_file_bytes'],1024)

    def test_full_optional_cap_keeps_scientific_argv_exact_and_defaults_uncapped(self):
        run,nodes,_,_=p.build_dag('Discussion07','optional_cap','prepared','als','assets',sys.executable,'/stable','/seq')
        selected=[next(n for n in nodes if n['id']=='iesrgs_sr')]
        original=copy.deepcopy(selected[0]['argv'])
        for capped in (False,True):
            with self.subTest(capped=capped):
                entry=dict(known_payload_lower_bound_bytes=0,additional_bytes=3*(1<<30))
                if capped:entry['maximum_file_bytes']=2*(1<<30)
                plan=dict(status='executable_after_resource_admission',phase='full',scene='Discussion07',run_id='optional_cap',
                          run_root=run,gpu_uuid='GPU-unit',env={},required_files=[],dag=selected,storage={'iesrgs_sr':entry})
                out='spec_cap_'+str(capped)
                p.materialize(SimpleNamespace(out=out,base_python=sys.executable,shared_project_root='/home/cts/Project/4DSR'),plan)
                actual=p.read(out+'/job.json')['commands'][1]['argv']
                if capped:self.assertEqual(actual[2],'stage-run')
                else:self.assertEqual(actual,original)
                self.assertEqual(p.read(out+'/plan.json')['dag'][0]['argv'],original)
                with patch.object(p,'guard'),patch.object(q.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as execute:
                    if capped:
                        q.run_stage(p,out+'/plan.json','iesrgs_sr')
                        self.assertEqual(execute.call_args.args[0],[x.replace('{ROOT}',str(self.root)) for x in original])
                    else:
                        with self.assertRaisesRegex(ValueError,'positive file limit'):q.run_stage(p,out+'/plan.json','iesrgs_sr')
                        execute.assert_not_called()

    def test_invalid_file_limit_never_becomes_executable(self):
        nodes=[dict(id='training',storage_kind='checkpoint_training')]
        for value in (0,-1,True,1.5,'1024'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'positive integer'):
                p.storage_estimates(nodes,1,[1,1],[4,4],dict(stages={'training':dict(maximum_file_bytes=value)}))

    def test_job_spec_inside_execution_run_is_rejected_before_creation(self):
        run='output/static_sr_wu_20261022/cts_scene_runs/Discussion07/no_overwrite'
        with self.assertRaisesRegex(ValueError,'outside the execution run'):
            p.materialize(SimpleNamespace(out=run+'/job_logs'),dict(run_root=run))
        self.assertFalse((self.root/run).exists())


if __name__=='__main__':unittest.main()
