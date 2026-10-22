"""CPU contracts only; synthetic receipt fixtures are never operational evidence."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('plan_scene',Path(__file__).with_name('plan_scene.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class ScenePlanTests(unittest.TestCase):
    def test_lossless_storage_is_explicit_and_never_assumes_a_ratio(self):
        for format in ('npz_deflate','npz_byte_shuffle'):
            with self.subTest(format=format):
                _,nodes,_,_=p.build_dag('Discussion07','unit_only','prepared','plans/receipt.json','assets.json',
                    '/base/python','/stable/python','/sequence/python',iesrgs_cache_format=format)
                by={n['id']:n for n in nodes}
                sizes=p.storage_estimates(nodes,3600,[320,180],[1280,720],None)
                for name in ('ie_external','ie_internal_cache'):
                    argv=by[name]['argv']
                    self.assertEqual(argv.count('--cache-format'),1)
                    self.assertEqual(argv[argv.index('--cache-format')+1],format)
                    self.assertEqual(by[name]['storage_kind'],'ie_lossless_arrays')
                    self.assertIsNone(sizes[name]['additional_bytes'])
                    self.assertEqual(sizes[name]['uncompressed_float32_payload_bytes'],53084160000)
                    self.assertFalse(p.check_disk(sizes[name]['additional_bytes'],0,p.ROOT)['passed'])

    def test_byte_shuffle_cli_is_opt_in_and_blocked_budget_stays_blocked(self):
        plan=dict(status='blocked',scene='Discussion07',blockers=['Unknown stage storage budgets'])
        args=['plan_scene.py','plan','--run-id','unit_only','--out','output/unit_only',
              '--gpu-uuid','GPU-test','--base-python','/base/python']
        for extra,format in (([],'npy'),(['--iesrgs-cache-format','npz_byte_shuffle'],'npz_byte_shuffle')):
            with self.subTest(format=format),patch.object(p.sys,'argv',args+extra), \
                 patch.object(p,'create_plan',return_value=plan) as create, \
                 patch.object(p,'materialize') as materialize,patch('builtins.print'):
                with self.assertRaises(SystemExit) as stopped:p.main()
                self.assertEqual(stopped.exception.code,2)
                self.assertEqual(create.call_args.args[0].iesrgs_cache_format,format)
                self.assertEqual(materialize.call_args.args[1]['status'],'blocked')

    def test_byte_shuffle_still_requires_explicit_hashed_budget_evidence(self):
        _,nodes,_,_=p.build_dag('Discussion07','unit_only','prepared','plans/receipt.json','assets.json',
            '/base/python','/stable/python','/sequence/python',iesrgs_cache_format='npz_byte_shuffle')
        for name in ('ie_external','ie_internal_cache'):
            budget={'stages':{name:{'additional_bytes':100,'basis':'measured','evidence':[]}}}
            with self.subTest(stage=name),self.assertRaisesRegex(ValueError,'hashed evidence'):
                p.storage_estimates(nodes,3600,[320,180],[1280,720],budget)
        with self.assertRaisesRegex(ValueError,'Unsupported IE lossless'):
            p.build_dag('Discussion07','unit_only','prepared','plans/receipt.json','assets.json',
                '/base/python','/stable/python','/sequence/python',iesrgs_cache_format='unregistered_codec')

    def dag(self):
        return p.build_dag('Discussion07','unit_only','prepared','plans/receipt.json','assets.json',
                           '/base/python','/stable/python','/sequence/python')

    def test_source_interfaces_full_schedules_and_barrier(self):
        run,nodes,paths,gates=self.dag();by={n['id']:n for n in nodes}
        self.assertEqual(len(nodes),len(by))
        for name in ('iesrgs_sr','splatsure_sr','sequence_matters_sr','threedsr_coupled'):
            self.assertTrue(set(gates)<=set(by[name]['depends_on']))
        for node in nodes:
            self.assertNotIn('--wiring-fixture-steps',node['argv'])
        for name,method in [('splat_lr','splatsure'),('ie_internal_train','iesrgs'),('threedsr_lr','threedsr')]:
            self.assertIn('/'+method+'.json', ' '.join(by[name]['argv']))
        self.assertTrue(by['ie_merge']['argv'][-1].endswith('/bundle.json'))
        self.assertIn('/base/python',by['threedsr_coupled']['argv'])
        self.assertIn('/stable/python',by['threedsr_coupled']['argv'])
        self.assertIn('/sequence/python',by['sequence_teacher']['argv'])
        self.assertIn('--sequence-plan',by['sequence_teacher']['argv'])
        self.assertNotIn('render_eval.py',' '.join(x for n in nodes for x in n['argv']))

    def test_every_generated_cli_flag_exists_in_current_production_parser(self):
        _,nodes,_,_=self.dag()
        for node in nodes:
            source=p.resolve(node['argv'][2])
            tree=ast.parse(source.read_text())
            flags={arg.value for call in ast.walk(tree) if isinstance(call,ast.Call)
                   and isinstance(call.func,ast.Attribute) and call.func.attr=='add_argument'
                   for arg in call.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str)
                   and arg.value.startswith('--')}
            supplied={value for value in node['argv'][3:] if value.startswith('--')}
            self.assertLessEqual(supplied,flags,node['id'])

    def test_unknown_storage_never_becomes_zero_or_ready(self):
        _,nodes,_,_=self.dag()
        estimates=p.storage_estimates(nodes,3600,[320,180],[1280,720],None)
        self.assertIsNone(estimates['threedsr_coupled']['additional_bytes'])
        self.assertIsNone(estimates['splat_teacher']['additional_bytes'])
        exact=sum(estimates[k]['known_payload_lower_bound_bytes'] for k in ('ie_external','ie_internal_cache'))
        self.assertEqual(exact,106170163200)
        budget={'stages':{'ie_external':{'additional_bytes':1,'basis':'measured','evidence':[{}]}}}
        with self.assertRaisesRegex(ValueError,'below known payload'):
            p.storage_estimates(nodes,3600,[320,180],[1280,720],budget)

    def test_budget_requires_evidence_and_real_disk_gate(self):
        _,nodes,_,_=self.dag()
        budget={'stages':{'splat_teacher':{'additional_bytes':100,'basis':'measured','evidence':[]}}}
        with self.assertRaisesRegex(ValueError,'hashed evidence'):
            p.storage_estimates(nodes,3600,[320,180],[1280,720],budget)
        with patch.object(p.shutil,'disk_usage',return_value=SimpleNamespace(free=100)):
            self.assertFalse(p.check_disk(None,0,p.ROOT)['passed'])
            self.assertFalse(p.check_disk(90,20,p.ROOT)['passed'])
            self.assertTrue(p.check_disk(80,20,p.ROOT)['passed'])

    def test_completed_teacher_receipt_required_same_hardware_and_size(self):
        item={'path':'unit_fixture','sha256':'synthetic'}
        receipt=dict(schema='static_sr_wu_teacher_acceptance_fixture_v1',status='fixture_passed',
            gpu_executed=True,hardware={'gpu_uuid':'GPU-test'},method='stablesr',teacher_generated=True,
            outputs=[{'size_wh':[1280,720]}],wrapper=item,registered_records=item,
            adapter_sources=[item],model_identities=[item],runtime_assets_registration=item)
        p.validate_acceptance('stablesr',receipt,'test',[1280,720],check=lambda _:None)
        for change in ({'status':'plan_only'},{'gpu_executed':False},{'method':'threedsr'},
                       {'teacher_generated':False},{'hardware':{'gpu_uuid':'GPU-other'}},
                       {'outputs':[{'size_wh':[1344,1008]}]}):
            with self.assertRaises(ValueError):
                p.validate_acceptance('stablesr',dict(receipt,**change),'test',[1280,720],check=lambda _:None)
        def mismatch(_):raise ValueError('bytes changed')
        with self.assertRaisesRegex(ValueError,'bytes changed'):
            p.validate_acceptance('stablesr',receipt,'test',[1280,720],check=mismatch)

    def test_all_four_wiring_methods_and_diffusion_rounds_required(self):
        item={'path':'fixture','sha256':'synthetic'}
        receipt=dict(schema='static_sr_wu_gpu_acceptance_v1',status='passed_bounded_GPU_wiring_only',
            GPU_executed=True,gpu_uuid='GPU-test',all_extension_modules_coexisted=True,source=item,
            source_snapshot={'one':item},scenes=[{'short':'Discussion07','resolutions':{'hr':[1280,720]}}],
            results=[{'scene':'Discussion07','method':m,'status':'passed'} for m in p.METHODS],extensions={})
        p.validate_acceptance('wiring',receipt,'test',[1280,720],check=lambda _:None)
        receipt['results'].pop()
        with self.assertRaisesRegex(ValueError,'All four'):
            p.validate_acceptance('wiring',receipt,'test',[1280,720],check=lambda _:None)
        teacher=dict(schema='static_sr_wu_teacher_acceptance_fixture_v1',status='fixture_passed',
            gpu_executed=True,hardware={'gpu_uuid':'test'},method='threedsr',teacher_generated=True,
            outputs=[{'size_wh':[1280,720]}],rounds=[{},{}])
        with self.assertRaisesRegex(ValueError,'four actual'):
            p.validate_acceptance('threedsr',teacher,'test',[1280,720],check=lambda _:None)

    def test_threedsr_uses_real_per_round_output_receipt_contract(self):
        item={'path':'fixture','sha256':'synthetic'}
        # teacher_acceptance.run_four_round_fixture emits these fields per
        # transition and deliberately has no top-level outputs field.
        receipt=dict(schema='static_sr_wu_teacher_acceptance_fixture_v1',status='fixture_passed',
            gpu_executed=True,hardware={'gpu_uuid':'GPU-test'},method='threedsr',teacher_generated=True,
            rounds=[dict(round_index=i,outputs=[dict(item,key='cam01/0000',size_wh=[1280,720])],
                         latent=item,teacher_receipt=item) for i in range(4)],
            wrapper=item,registered_records=item,adapter_sources=[item],model_identities=[item],
            runtime_assets_registration=item)
        self.assertNotIn('outputs',receipt)
        p.validate_acceptance('threedsr',receipt,'test',[1280,720],check=lambda _:None)
        broken=copy.deepcopy(receipt);broken['rounds'][2]['outputs']=[]
        with self.assertRaisesRegex(ValueError,'every diffusion transition'):
            p.validate_acceptance('threedsr',broken,'test',[1280,720],check=lambda _:None)
        broken=copy.deepcopy(receipt);broken['rounds'][2]['outputs'][0]['size_wh']=[640,360]
        with self.assertRaisesRegex(ValueError,'dimensions'):
            p.validate_acceptance('threedsr',broken,'test',[1280,720],check=lambda _:None)
        broken=copy.deepcopy(receipt);broken['rounds'][2]['round_index']=1
        with self.assertRaisesRegex(ValueError,'four actual'):
            p.validate_acceptance('threedsr',broken,'test',[1280,720],check=lambda _:None)

    def test_cleanup_scoped_and_return_not_claimed_verified(self):
        run,_,paths,_=self.dag();contract=p.cleanup_contract(run,paths,['cam01','cam02'])
        self.assertFalse(contract['execute']);self.assertFalse(contract['checkpoint_retention'])
        self.assertTrue(all(Path(x).is_relative_to(run) for x in contract['allowlist']))
        self.assertTrue(all('/data/' not in x and '/weight/' not in x for x in contract['allowlist']))
        self.assertTrue(any('SHA256' in x for x in contract['gates']))
        self.assertIn(paths['threedsr']+'/coupled/latents',contract['allowlist'])
        self.assertNotIn(run,contract['allowlist'])
        self.assertNotIn(paths['splatsure']+'/teacher/inputs',contract['allowlist'])
        self.assertNotIn(paths['splatsure']+'/teacher/execution_workdir',contract['allowlist'])
        self.assertTrue(all(not x.endswith('_runtime') for x in contract['allowlist']))
        self.assertIn(paths['splatsure']+'/teacher/generated',contract['allowlist'])

    def test_blocked_materialization_never_writes_executable_job(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'ROOT',Path(tmp)):
            args=SimpleNamespace(out='plan');plan={'status':'blocked','scene':'Discussion07','dag':[], 'storage':{}}
            p.materialize(args,plan)
            self.assertTrue((Path(tmp)/'plan/plan.json').is_file())
            self.assertFalse((Path(tmp)/'plan/job.json').exists())

    def test_executable_job_keeps_guards_and_returns_only_compact_results(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'ROOT',Path(tmp)):
            run,nodes,_,_=self.dag()
            args=SimpleNamespace(out='plan',base_python='/base/python',shared_project_root='/home/cts/Project/4DSR')
            plan=dict(status='executable_after_resource_admission',scene='Discussion07',run_id='unit_only',run_root=run,
                gpu_uuid='GPU-test',env={},required_files=[],dag=nodes,
                storage={n['id']:{'known_payload_lower_bound_bytes':0} for n in nodes})
            p.materialize(args,plan)
            job=json.loads((Path(tmp)/'plan/job.json').read_text())
            self.assertEqual(len(job['commands']),2*len(nodes)+1)
            self.assertEqual(job['commands'][0]['name'],'ie_external_disk_guard')
            self.assertEqual(job['commands'][-1]['name'],'collect_compact_results')
            self.assertTrue(all(x.endswith('/compact') or Path(x).suffix=='.json' for x in job['return_paths']))
            self.assertNotIn(run,job['return_paths'])

    def test_partial_or_nonfinite_metrics_cannot_unlock_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'ROOT',Path(tmp)):
            configs={m:{'path':m+'.json','sha256':'synthetic'} for m in p.METHODS}
            plan=dict(scene_name='meetroom_discussion',final_outputs={m:m for m in p.METHODS},configs=configs)
            values=dict(PSNR=20.,SSIM=.8,LPIPS=.2)
            checkpoint={'path':'not_transferred','sha256':'bound_identity'}
            for m in p.METHODS:
                directory=Path(tmp)/m;directory.mkdir();(directory/'evaluation').mkdir()
                (directory/'complete.json').write_text(json.dumps(dict(status='completed_with_evaluation',wiring_fixture=False,
                    config=configs[m],checkpoint=checkpoint,steps=30000,rounds=[dict(round=i,checkpoint=checkpoint) for i in range(4)])))
                metric=dict(schema='static_sr_wu_uniform_png_metrics_v1',scene=plan['scene_name'],camera='cam00',
                    metrics=values,frames=[dict(frame=i,**values) for i in range(300)],checkpoint=checkpoint)
                (directory/'evaluation/metrics.json').write_text(json.dumps(metric))
            self.assertEqual(set(p.verify_final_metrics(plan)),set(p.METHODS))
            broken=Path(tmp)/'splatsure/evaluation/metrics.json'
            data=json.loads(broken.read_text());data['frames'].pop();broken.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'coverage'):p.verify_final_metrics(plan)
            data['frames'].append(dict(frame=299,PSNR=float('nan'),SSIM=.8,LPIPS=.2));broken.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'nonfinite'):p.verify_final_metrics(plan)

    def test_formal_result_rejects_foreign_config_or_checkpoint(self):
        plan=dict(scene_name='meetroom_discussion',configs={'splatsure':{'path':'splat.json','sha256':'correct'}},
                  final_outputs={'splatsure':'run'})
        complete=dict(status='completed_with_evaluation',wiring_fixture=False,steps=30000,
                      config={'path':'other.json','sha256':'other'},checkpoint={'path':'x','sha256':'one'})
        metric=dict(checkpoint={'path':'x','sha256':'two'})
        with patch.object(p,'read',side_effect=[complete,metric]):
            with self.assertRaisesRegex(ValueError,'another registered'):p.verify_final_metrics(plan)
        complete['config']=plan['configs']['splatsure']
        with patch.object(p,'read',side_effect=[complete,metric]):
            with self.assertRaisesRegex(ValueError,'checkpoint differs'):p.verify_final_metrics(plan)

    def test_compact_copies_bound_input_metadata_but_not_running_job_status(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'ROOT',Path(tmp)):
            root=Path(tmp);run=root/'run';(run/'job_logs').mkdir(parents=True)
            (run/'job_logs/status.json').write_text('{"status":"running"}')
            (run/'job_logs/finished_stage.log').write_text('finished\n')
            (run/'large_cache.npy').write_bytes(b'not copied')
            metadata=[]
            for role in ('manifest','training_records',*('source_configuration:'+m for m in p.METHODS),
                         'sequence_plan','runtime_assets'):
                file=root/'inputs'/(role.replace(':','_')+'.json');file.parent.mkdir(exist_ok=True)
                file.write_text(json.dumps({'role':role,'split':['cam01'],'lr_sha256':'source_identity'}))
                metadata.append(dict(role=role,**p.identity(file)))
            configs={}
            for method in p.METHODS:
                file=root/(method+'.json');file.write_text('{}');configs[method]=p.identity(file)
            plan=dict(status='executable_after_resource_admission',scene='Discussion07',run_id='unit',
                run_root='run',reserve_bytes=0,configs=configs,compact_metadata=metadata)
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan))
            with patch.object(p,'verify_final_metrics',return_value={}),patch.object(p,'check_disk',return_value={'passed':True}):
                p.collect(plan_path)
            result=json.loads((run/'compact/index.json').read_text())
            originals={x['original_path'] for x in result['files']}
            self.assertTrue({x['path'] for x in metadata}<=originals)
            self.assertNotIn('run/job_logs/status.json',originals)
            self.assertNotIn('run/large_cache.npy',originals)
            self.assertIn('run/job_logs/finished_stage.log',originals)
            self.assertEqual(result['input_metadata'],metadata)
            self.assertFalse(result['local_return_verified']);self.assertFalse(result['cleanup_allowed'])
            for item in result['files']:
                self.assertEqual(p.sha(run/'compact'/item['path']),item['sha256'])

    def test_compact_rejects_changed_metadata_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'ROOT',Path(tmp)):
            root=Path(tmp);(root/'run').mkdir();file=root/'input.json';file.write_text('{}')
            item=p.identity(file)
            metadata=[dict(role=role,**item) for role in
                ('manifest','training_records',*('source_configuration:'+m for m in p.METHODS))]
            file.write_text('{"changed":true}')
            plan=dict(status='executable_after_resource_admission',run_root='run',reserve_bytes=0,
                      compact_metadata=metadata,configs={})
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan))
            with patch.object(p,'verify_final_metrics',return_value={}),patch.object(p,'check_disk',return_value={'passed':True}):
                with self.assertRaisesRegex(ValueError,'Identity changed'):p.collect(plan_path)
            self.assertFalse((root/'run/compact').exists())


if __name__=='__main__':unittest.main()
