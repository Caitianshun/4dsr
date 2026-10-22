"""CPU entry-point integration contracts; all learned/GPU execution is mocked.

Tests assert observable production boundaries, not model quality or CUDA validity.
"""
from __future__ import annotations
import contextlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import common
import bundle_priors
import run_threedsr
import train_stage
import render_eval


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        init = self.root / "initial.npz"
        init.write_bytes(b"CPU contract initializer identity; never loaded as a model")
        observations = []
        for frame in (0, 1):
            for camera in ("cam01", "cam02", "cam03"):
                path = self.root / "lr" / camera / f"{frame:04d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((16, 16, 3), 50+frame, dtype=np.uint8)).save(path)
                observations.append(dict(camera_id=camera, frame_index=frame, time=frame/300., split="train",
                    lr_path=str(path.relative_to(self.root)), lr_sha256=common.sha(path)))
        cameras = {camera: {"c2w": [[1,0,0,i+1],[0,1,0,0],[0,0,1,1],[0,0,0,1]]}
                   for i, camera in enumerate(["cam00", "cam01", "cam02", "cam03"])}
        self.manifest = dict(schema="n3dv_dynamic_sr_pilot_v1", scene="fixture_scene", scale=4,
            physical_lr_derivation={"formation":"area4"}, splits={"train":["cam01","cam02","cam03"],"test":["cam00"]},
            frame_indices=[0,1], observations=observations, cameras=cameras,
            initialization=dict(train_only_lr=True,npz_path=init.name,npz_sha256=common.sha(init)))
        self.manifest_file = self.write("manifest.json", self.manifest)
        self.m, self.rows = common.training_manifest(self.manifest_file, verify_pixels=True)
        self.records = dict(schema="static_sr_wu_train_records_v1", manifest=common.identity(self.manifest_file),
                            heldout_cameras=["cam00"], records=common.records(self.m, self.rows))
        self.records_file = self.write("records.json", self.records)
        self.source = self.write("source.json", {"configuration":{}})
        self.config = dict(schema=common.SCHEMA,method="sequence_matters",scene="fixture_scene",seed=7,
                           manifest=common.identity(self.manifest_file),source_configuration=common.identity(self.source),
                           wu_configuration={},degradation="area4",specification={"lr_steps":30000,"sr_steps":30000})
        self.cfg_file = self.write("config.json", self.config)
        self.weights = []
        for name in ("teacher_a.pt","teacher_b.pt"):
            p=self.root/name;p.write_bytes(name.encode());self.weights.append(p)
        self.targets = {}
        for record in self.records["records"]:
            p=self.root/"teacher"/(record["key"]+".png");p.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.full((64,64,3),120,dtype=np.uint8)).save(p)
            self.targets[record["key"]]=str(p)
        self.receipt = dict(status="teacher_complete",targets=self.targets,
            target_sha256={key:common.sha(path) for key,path in self.targets.items()},
            weights={str(p):common.sha(p) for p in self.weights},
            groups=[dict(time=0,lr_sha256={r['key']:r['lr_sha256'] for r in self.records['records']})])
        self.receipt_file = self.write("teacher_receipt.json", self.receipt)

    def write(self, name, payload):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload));return path

    def runtime_assets_fixture(self):
        from methods import splatsure_stablesr as helper
        cache=self.root/'runtime_cache'
        hf=cache/'hf'
        torch_home=cache/'torch'
        repo=hf/helper.OPENCLIP_CACHE_REPO
        paths={'openclip':repo/'snapshots'/helper.OPENCLIP_REVISION/'open_clip_pytorch_model.bin',
               'vgg16':torch_home/'hub/checkpoints/vgg16-397923af.pth', 'lpips':cache/'lpips/vgg.pth'}
        for role,path in paths.items():
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('CPU cache fixture '+role).encode())
        (repo/'refs').mkdir();(repo/'refs/main').write_text(helper.OPENCLIP_REVISION)
        files=[dict(role=role,path=str(path),sha256=common.sha(path)) for role,path in paths.items()]
        assets=dict(files=files,cache_mappings=copy.deepcopy(files),
                    cache_environment={'TORCH_HOME':str(torch_home),'HUGGINGFACE_HUB_CACHE':str(hf)})
        (self.root/'upstream/configs').mkdir(parents=True)
        return self.write('runtime_assets.json',assets),{item['role']:item['sha256'] for item in files}

    def bundle(self, models=None, receipt=None, name="bundle.json"):
        args=["bundle_priors.py","--config",str(self.cfg_file),"--sr-receipt",str(receipt or self.receipt_file),
              "--output",str(self.root/name)]
        for model in (self.weights if models is None else models):args += ["--model",str(model)]
        with mock.patch.object(sys,"argv",args):bundle_priors.main()
        return self.root/name

    def test_sequence_receipt_bundles_and_train_entry_accepts_exact_inputs(self):
        path=self.bundle()
        _,rows,priors=train_stage.preflight(self.config,"sr",path)
        self.assertEqual(len(priors),len(rows))
        self.assertEqual({(r['camera_id'],r['frame_index']) for r in rows},set(priors))
        self.assertEqual(common.read(path)['method'],'sequence_matters')

    def test_bundle_rejects_plan_only_changed_targets_or_wrong_model_identity(self):
        plan=copy.deepcopy(self.receipt);plan['status']='plan_only'
        with self.assertRaises(ValueError):self.bundle(receipt=self.write('plan.json',plan),name='plan_bundle.json')
        unrelated=self.root/'unrelated.pt';unrelated.write_bytes(b'unrelated model')
        with self.assertRaises(ValueError):self.bundle(models=[unrelated],name='wrong_model.json')
        # Rehashing mutated targets when making the bundle must not launder a stale producer receipt.
        next(iter(map(Path,self.targets.values()))).write_bytes(b'changed-after-production')
        with self.assertRaises(ValueError):self.bundle(name='changed_target.json')

    def test_train_entry_rejects_method_swapped_prior_bundle(self):
        path=self.bundle()
        payload=common.read(path);payload['method']='threedsr';path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):train_stage.preflight(self.config,'sr',path)

    def driver(self, *, bad_parent=False, altered_records=False, fail_teacher_round=None, receipt_mutation=None,
               runtime_assets=False):
        import numpy as np
        import random
        import torch
        from methods.sequence_matters import source_objects
        rng_helpers=source_objects(common.ROOT/'experiments/dynamic_sr_multiview_footprint_20261007/full_prefix_registered.py',
                                   ['rng_state','restore_rng'],{'random':random})
        config=dict(self.config,method='threedsr',specification={'lr_steps':30000,'sr_steps':20000})
        cfg=self.write('three_config.json',config)
        records=copy.deepcopy(self.records)
        if altered_records:records['records'][0]['time']=0.77
        record_file=self.write('three_records.json',records)
        parent=self.root/'lr_parent.pt'
        torch.save({'adaptation_config':config,'metadata':{'stage':'lr','iterations':3 if bad_parent else 30000,
                    'optimizer_updates':3 if bad_parent else 29999}},parent)
        events=[]
        active=[]
        previous=[common.identity(parent)]
        sampling_instances=[]
        sampling_end_states=[]
        training_draws=[]
        fake_cuda_state=[239]
        real_manual_seed=torch.manual_seed
        def seed_torch_once(seed):
            events.append(('seed_training',seed))
            return real_manual_seed(seed)
        def cuda_seed(seed):fake_cuda_state[0]=seed%251
        def cuda_get():return [torch.tensor([fake_cuda_state[0]],dtype=torch.uint8)]
        def cuda_set(states):fake_cuda_state[0]=int(states[0][0])
        def interfere_with_global_rng():
            random.random();np.random.random();torch.rand(())
            fake_cuda_state[0]=(fake_cuda_state[0]+13)%251
        def capture_rng(torch_module,numpy_module):
            events.append(('capture_training_rng',))
            return rng_helpers['rng_state'](torch_module,numpy_module)
        def restore_rng(value,torch_module,numpy_module):
            events.append(('restore_training_rng',))
            rng_helpers['restore_rng'](value,torch_module,numpy_module)
        def training_setup(_):
            events.append(('adam_reset',));interfere_with_global_rng()
        class FakeBackend:
            def __init__(inner,cfg,checkpoint):
                self.assertEqual(checkpoint,previous[0])
                self.assertFalse(active)
                active.append(inner)
                events.append(('restore',checkpoint['path']))
                interfere_with_global_rng()  # Model initialization must not perturb the training stream.
                inner.opt=object()
                inner.runtime={'status':'CPU contract fixture; no GPU initialized'}
                inner.g=SimpleNamespace(training_setup=training_setup)
                inner.prefix=SimpleNamespace(rng_state=capture_rng,restore_rng=restore_rng)
        backend_module=ModuleType('backend');backend_module.WuBackend=FakeBackend
        def teacher(command,check,**worker_options):
            self.assertEqual(command[0],str(teacher_launcher.absolute()),'venv launcher must not be resolved to its base Python')
            self.assertTrue(check)
            self.assertFalse(active,'Teacher must run after the Wu backend is released')
            r=int(command[command.index('--round-index')+1])
            if r:
                self.assertIn('--rendered-paths',command)
                rendered=common.read(command[command.index('--rendered-paths')+1])
                self.assertEqual(set(rendered),set(self.targets))
            else:self.assertNotIn('--rendered-paths',command)
            events.append(('teacher',r))
            interfere_with_global_rng()  # Stronger than a real subprocess, whose RNG is isolated.
            if r == fail_teacher_round:
                raise subprocess.CalledProcessError(1,command)
            out=Path(command[command.index('--output')+1]);out.mkdir()
            if runtime_assets:
                self.assertIn('--runtime-assets',command)
                self.assertTrue(Path(command[command.index('--runtime-assets')+1]).is_absolute())
                work=Path(worker_options['cwd'])
                self.assertTrue(work.is_dir())
                self.assertEqual(work.parent,out.parent/(out.name+'_runtime'))
                self.assertFalse(work.is_relative_to(out),'Cache cwd must not occupy fresh target output')
                self.assertEqual(worker_options['env']['HF_HUB_OFFLINE'],'1')
                self.assertEqual(worker_options['env']['TRANSFORMERS_OFFLINE'],'1')
                self.assertEqual(worker_options['env']['TORCH_HOME'],str(self.root/'runtime_cache/torch'))
                self.assertTrue((work/'taming/modules/autoencoder/lpips/vgg.pth').is_file())
            else:self.assertFalse(worker_options)
            from methods.threedsr import UPSTREAM_COMMIT
            receipt={'status':'diffusion_round_complete','round_index':r,'targets':self.targets,
                     'upstream_commit':UPSTREAM_COMMIT,
                     'weight_loads':[{'checkpoint':str(p),'sha256':common.sha(p),'missing_keys':[],
                                      'unexpected_keys':[]} for p in self.weights],
                     'records':[dict(key=rec['key'],scene=rec['scene'],camera=rec['camera'],time=rec['time'],
                         lr_sha256=rec['lr_sha256'],target_sha256=common.sha(self.targets[rec['key']]),
                         render_sha256=common.sha(rendered[rec['key']]) if r else None)
                         for rec in records['records']]}
            if runtime_assets:
                receipt['runtime_assets']={'registration':{'path':str(assets_file),'sha256':common.sha(assets_file)},
                    'execution_workdir':str(work),'offline_cache':True}
            if receipt_mutation == 'target_bytes' and r == 0:
                Path(next(iter(self.targets.values()))).write_bytes(b'replaced after producer hashed target')
            elif receipt_mutation == 'weight_hash' and r == 0:
                receipt['weight_loads'][0]['sha256']='0'*64
            elif receipt_mutation == 'weight_path' and r == 0:
                receipt['weight_loads'][0]['checkpoint']=str(self.source)
                receipt['weight_loads'][0]['sha256']=common.sha(self.source)
            elif receipt_mutation == 'duplicate_observation' and r == 0:
                receipt['records'].append(copy.deepcopy(receipt['records'][0]))
            elif receipt_mutation == 'feedback_hash' and r == 1:
                receipt['records'][0]['render_sha256']='0'*64
            elif receipt_mutation == 'runtime_identity' and r == 0:
                receipt['runtime_assets']['registration']['sha256']='0'*64
            (out/'receipt.json').write_text(json.dumps(receipt))
            return SimpleNamespace(returncode=0)
        def optimize(backend,cfg,rows,stage,priors,steps,out,clock_offset,sampling_rng):
            self.assertEqual(stage,'sr');self.assertEqual(steps,5000);self.assertEqual(clock_offset,0)
            self.assertEqual(set(priors),{(o['camera_id'],o['frame_index']) for o in self.rows})
            if sampling_instances:
                self.assertIs(sampling_rng,sampling_instances[0])
                self.assertEqual(sampling_rng.getstate(),sampling_end_states[-1])
            sampling_instances.append(sampling_rng)
            sampling_rng.random()
            sampling_end_states.append(sampling_rng.getstate())
            events.append(('optimize',len([e for e in events if e[0]=='teacher'])-1))
            training_draws.append((random.random(),float(np.random.random()),float(torch.rand(())),fake_cuda_state[0]))
            fake_cuda_state[0]=(fake_cuda_state[0]+1)%251
            p=out/'checkpoint.pt';p.write_bytes(str(out).encode());previous[0]=common.identity(p)
            return previous[0]
        def render(backend,out):
            events.append(('render',));active.clear()
            interfere_with_global_rng()
            return dict(self.targets)
        def evaluate(*args):events.append(('evaluate',));active.clear()
        teacher_launcher=self.root/'teacher_env/bin/python'
        teacher_launcher.parent.mkdir(parents=True)
        teacher_launcher.symlink_to(sys.executable)
        args=['run_threedsr.py','--teacher-python',str(teacher_launcher),'--config',str(cfg),'--records',str(record_file),'--lr-checkpoint',str(parent),
              '--upstream',str(self.root/'upstream'),'--stable-checkpoint',str(self.weights[0]),
              '--vqgan-checkpoint',str(self.weights[1]),'--out',str(self.root/'driver')]
        if runtime_assets:
            assets_file,asset_hashes=self.runtime_assets_fixture()
            args+=['--runtime-assets',str(assets_file)]
        with contextlib.ExitStack() as stack:
            if runtime_assets:
                from methods import splatsure_stablesr as helper
                stack.enter_context(mock.patch.dict(helper.RUNTIME_SHA256,asset_hashes,clear=True))
                stack.enter_context(mock.patch.object(helper,'source_check',side_effect=AssertionError('Splat source audit called')))
            stack.enter_context(mock.patch.object(sys,'argv',args))
            stack.enter_context(mock.patch.dict(sys.modules,{'backend':backend_module}))
            stack.enter_context(mock.patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'contract-fixture'}))
            stack.enter_context(mock.patch.object(run_threedsr.subprocess,'run',side_effect=teacher))
            stack.enter_context(mock.patch.object(train_stage,'optimize',side_effect=optimize))
            stack.enter_context(mock.patch.object(render_eval,'render_training',side_effect=render))
            stack.enter_context(mock.patch.object(render_eval,'evaluate',side_effect=evaluate))
            stack.enter_context(mock.patch.object(torch.cuda,'empty_cache'))
            stack.enter_context(mock.patch.object(torch.cuda,'get_device_name',return_value='CPU mock GPU'))
            stack.enter_context(mock.patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(uuid='GPU-contract')))
            stack.enter_context(mock.patch.object(torch.cuda,'synchronize',side_effect=lambda:events.append(('synchronize',))))
            stack.enter_context(mock.patch.object(torch.cuda,'reset_peak_memory_stats',side_effect=lambda:events.append(('reset_peak',))))
            stack.enter_context(mock.patch.object(torch.cuda,'max_memory_allocated',return_value=123456))
            stack.enter_context(mock.patch.object(torch.cuda,'max_memory_reserved',return_value=234567))
            stack.enter_context(mock.patch.object(torch,'manual_seed',side_effect=seed_torch_once))
            stack.enter_context(mock.patch.object(torch.cuda,'is_initialized',return_value=True))
            stack.enter_context(mock.patch.object(torch.cuda,'manual_seed_all',side_effect=cuda_seed))
            stack.enter_context(mock.patch.object(torch.cuda,'get_rng_state_all',side_effect=cuda_get))
            stack.enter_context(mock.patch.object(torch.cuda,'set_rng_state_all',side_effect=cuda_set))
            run_threedsr.main()
        # Independent continuous generators represent four uninterrupted training
        # steps. Backend/teacher/render draws above must leave these unchanged.
        expected_python=random.Random(config['seed'])
        expected_numpy=np.random.RandomState(config['seed'])
        expected_torch=torch.Generator().manual_seed(config['seed'])
        expected=[(expected_python.random(),float(expected_numpy.random_sample()),
                   float(torch.rand((),generator=expected_torch)),(config['seed']+i)%251) for i in range(4)]
        self.assertEqual(training_draws,expected)
        return events,common.read(self.root/'driver/complete.json')

    def test_real_four_round_driver_preserves_barriers_and_resets_adam_once(self):
        events,receipt=self.driver()
        self.assertEqual([e[1] for e in events if e[0]=='teacher'],[0,1,2,3])
        self.assertEqual([e[1] for e in events if e[0]=='optimize'],[0,1,2,3])
        self.assertEqual(sum(e[0]=='adam_reset' for e in events),1)
        self.assertEqual(sum(e[0]=='render' for e in events),3)
        self.assertEqual(events[-1],('evaluate',))
        self.assertEqual(receipt['status'],'completed_with_evaluation')
        self.assertEqual(len(receipt['rounds']),4)
        self.assertIn('single Python Random',receipt['sampling_rng_policy'])

    def test_driver_seeds_training_once_and_restores_all_rng_after_backend_reconstruction(self):
        events,receipt=self.driver()
        self.assertEqual([e for e in events if e[0]=='seed_training'],[('seed_training',self.config['seed'])])
        seed_index=next(i for i,e in enumerate(events) if e[0]=='seed_training')
        self.assertEqual(events[seed_index-1],('adam_reset',))
        self.assertEqual(sum(e[0]=='capture_training_rng' for e in events),4)
        self.assertEqual(sum(e[0]=='restore_training_rng' for e in events),3)
        for index,event in enumerate(events):
            if event[0]=='restore_training_rng':
                self.assertEqual(events[index-1][0],'restore')
                self.assertEqual(events[index+1:index+3],[('synchronize',),('reset_peak',)])
            if event[0]=='capture_training_rng':
                self.assertEqual(events[index-1],('synchronize',))
        self.assertIn('after first SR backend setup',receipt['training_rng_policy'])

    def test_driver_records_hardware_and_synchronized_training_cost_separate_from_teacher(self):
        import torch
        events,receipt=self.driver()
        hardware=receipt['hardware']
        self.assertTrue(hardware['hostname'])
        self.assertEqual(hardware['CUDA_VISIBLE_DEVICES'],'contract-fixture')
        self.assertEqual(hardware['torch_version'],torch.__version__)
        self.assertEqual(hardware['gpu_name'],'CPU mock GPU')
        self.assertEqual(hardware['uuid'],'GPU-contract')
        self.assertEqual(common.read(self.root/'driver/hardware.json'),hardware)
        self.assertIn('runtime',hardware)
        self.assertEqual(sum(e[0]=='reset_peak' for e in events),4)
        self.assertEqual(sum(e[0]=='synchronize' for e in events),8)
        for index,event in enumerate(events):
            if event[0]=='optimize':
                self.assertEqual(events[index-2:index],[('synchronize',),('reset_peak',)])
                self.assertEqual(events[index+1],('synchronize',))
        for row in receipt['rounds']:
            self.assertGreaterEqual(row['teacher_seconds'],0)
            self.assertGreaterEqual(row['training_seconds'],0)
            self.assertEqual(row['seconds'],row['training_seconds'])
            self.assertEqual(row['peak_cuda_allocated_bytes'],123456)
            self.assertEqual(row['peak_cuda_reserved_bytes'],234567)
            self.assertEqual(row['teacher_timing_scope'],'subprocess wall time')

    def test_driver_consumes_runtime_assets_with_isolated_worker_cwd_and_offline_environment(self):
        events,receipt=self.driver(runtime_assets=True)
        self.assertEqual([e[1] for e in events if e[0]=='teacher'],[0,1,2,3])
        self.assertEqual(receipt['runtime_assets'],common.identity(self.root/'runtime_assets.json'))

    def test_driver_rejects_runtime_assets_identity_not_used_by_teacher(self):
        with self.assertRaisesRegex(ValueError,'runtime cache receipt'):
            self.driver(runtime_assets=True,receipt_mutation='runtime_identity')
        self.assertFalse((self.root/'driver/reconstruction_0').exists())

    def test_standalone_diffusion_worker_resolves_inputs_before_using_cache_workdir(self):
        from methods import threedsr, splatsure_stablesr as helper
        assets_file,asset_hashes=self.runtime_assets_fixture()
        output=self.root/'standalone_targets'
        expected_work=self.root/'standalone_targets_runtime/execution_workdir'
        outer=self
        class FakeBridge:
            def __init__(inner,upstream,stable,vqgan,state,device):
                outer.assertEqual(Path.cwd(),expected_work)
                outer.assertTrue(all(Path(path).is_absolute() for path in (upstream,stable,vqgan,state)))
                outer.assertEqual(Path(stable),outer.weights[0])
                outer.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')
            def generate_targets(inner,manifest,round_index,target,rendered):
                outer.assertEqual(Path(target),output)
                outer.assertFalse(output.exists(),'Workdir creation must preserve fresh output requirement')
                outer.assertTrue(all(Path(r['lr_path']).is_absolute() for r in manifest['records']))
                outer.assertEqual(inner.runtime_assets_context['registration']['sha256'],common.sha(assets_file))
                output.mkdir()
                return outer.targets
        args=['threedsr.py','--manifest','records.json','--upstream','upstream',
              '--stable-checkpoint','teacher_a.pt','--vqgan-checkpoint','teacher_b.pt','--state-root','latents',
              '--output','standalone_targets','--round-index','0','--runtime-assets','runtime_assets.json']
        old_cwd=Path.cwd()
        try:
            os.chdir(self.root)
            with mock.patch.object(sys,'argv',args),mock.patch.dict(os.environ), \
                    mock.patch.dict(helper.RUNTIME_SHA256,asset_hashes,clear=True), \
                    mock.patch.object(helper,'source_check',side_effect=AssertionError('Splat source audit called')), \
                    mock.patch.object(threedsr,'StableSRTurboBridge',FakeBridge):
                threedsr.main()
        finally:os.chdir(old_cwd)

    def test_diffusion_runtime_assets_reject_replaced_cached_model_before_model_construction(self):
        from methods import threedsr, splatsure_stablesr as helper
        assets_file,asset_hashes=self.runtime_assets_fixture()
        assets=common.read(assets_file)
        Path(assets['files'][0]['path']).write_bytes(b'replaced model')
        with mock.patch.dict(helper.RUNTIME_SHA256,asset_hashes,clear=True):
            with self.assertRaisesRegex(ValueError,'cached model'):
                threedsr.prepare_runtime_assets(assets_file,self.root/'upstream',self.root/'bad_runtime')
        self.assertFalse((self.root/'bad_runtime').exists())

    def test_real_driver_rejects_short_lr_parent(self):
        with self.assertRaises(ValueError):self.driver(bad_parent=True)

    def test_real_driver_rejects_record_mutation_with_unchanged_manifest_label(self):
        with self.assertRaises(ValueError):self.driver(altered_records=True)

    def test_teacher_failure_stops_next_optimization_and_records_failure(self):
        with self.assertRaises(subprocess.CalledProcessError):self.driver(fail_teacher_round=1)
        failure=common.read(self.root/'driver/failure.json')
        self.assertEqual(failure['status'],'failed')
        self.assertEqual(len(failure['rounds']),1)
        self.assertFalse((self.root/'driver/reconstruction_1').exists())
        self.assertFalse((self.root/'driver/complete.json').exists())

    def assert_rejected_before_reconstruction(self, mutation, completed_rounds=0):
        with self.assertRaises(ValueError):self.driver(receipt_mutation=mutation)
        failure=common.read(self.root/'driver/failure.json')
        self.assertEqual(failure['status'],'failed')
        self.assertEqual(len(failure['rounds']),completed_rounds)
        self.assertFalse((self.root/f'driver/reconstruction_{completed_rounds}').exists())

    def test_driver_rejects_target_replaced_after_generation(self):
        self.assert_rejected_before_reconstruction('target_bytes')

    def test_driver_rejects_wrong_loaded_weight_hash(self):
        self.assert_rejected_before_reconstruction('weight_hash')

    def test_driver_rejects_unrelated_loaded_weight_path(self):
        self.assert_rejected_before_reconstruction('weight_path')

    def test_driver_rejects_duplicate_teacher_observation(self):
        self.assert_rejected_before_reconstruction('duplicate_observation')

    def test_driver_rejects_wrong_previous_shared_render_feedback(self):
        self.assert_rejected_before_reconstruction('feedback_hash',completed_rounds=1)

    def test_source_schedule_boundary_densification(self):
        import torch
        events=[]
        gaussians=SimpleNamespace(max_radii2D=torch.zeros(1),
            add_densification_stats=lambda *_:events.append('stats'),
            densify=lambda *_:events.append('densify'),prune=lambda *_:events.append('prune'),
            reset_opacity=lambda:events.append('reset'))
        stats=lambda *_:(torch.ones(1),torch.ones(1,dtype=torch.bool),torch.zeros(1,3))
        backend=SimpleNamespace(g=gaussians,torch=torch,prefix=SimpleNamespace(combine_batch_statistics=stats),
            config={'method':'threedsr','cameras_extent':1,'wu_configuration':{'ModelParams':{'_white_background':False}}})
        for iteration in (500,600,2400,2500,3000):
            before=len(events);train_stage.topology(backend,[{}],iteration,5000,self.root,'sr')
            new=events[before:]
            self.assertEqual('densify' in new,iteration in (600,2400))
            self.assertNotIn('reset',new)
        backend.config['method']='sequence_matters'
        before=len(events);train_stage.topology(backend,[{}],3000,30000,self.root,'sr')
        self.assertIn('reset',events[before:])
        before=len(events);train_stage.topology(backend,[{}],15000,30000,self.root,'sr')
        self.assertEqual(len(events),before)

    def test_optimize_preserves_source_last_step_and_round_local_clock(self):
        import torch
        parameter=torch.nn.Parameter(torch.tensor(1.))
        optimizer=torch.optim.SGD([parameter],lr=.1)
        clocks=[];metadata={}
        gaussians=SimpleNamespace(optimizer=optimizer,get_xyz=torch.zeros(1,3),
            update_learning_rate=clocks.append,oneupSHdegree=lambda:None)
        backend=SimpleNamespace(g=gaussians)
        def save(path,meta):metadata.update(meta);return {'path':str(path),'sha256':'contract'}
        backend.save=save
        config=dict(self.config,method='threedsr')
        out=self.root/'optimizer';out.mkdir()
        with mock.patch.object(train_stage,'batch_loss',side_effect=lambda *args:(parameter.square(),[])),\
             mock.patch.object(train_stage,'topology'):
            train_stage.optimize(backend,config,self.rows,'sr',{},2,out,clock_offset=4998)
        self.assertEqual(clocks,[4999,5000])
        self.assertEqual(metadata['optimizer_updates'],1)
        self.assertAlmostEqual(float(parameter.detach()),.8,places=6)

    def test_two_round_camera_sampling_matches_author_continuous_rng_and_checkpoint_state(self):
        import ast
        import random
        import torch
        source=common.ROOT/'tmp/static_sr_wu_20261022/upstream/threedsr/train_3dsr.py'
        function=next(n for n in ast.parse(source.read_text()).body
                      if isinstance(n,ast.FunctionDef) and n.name=='training_with_iters')
        loop=next(n for n in function.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
                  and n.target.id=='iteration')
        # Independently execute the author's actual pool refill, random pop,
        # and (disabled by default) high-resolution gate, excluding rendering.
        begin=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.If)
                   and isinstance(n.test,ast.UnaryOp) and isinstance(n.test.operand,ast.Name)
                   and n.test.operand.id=='viewpoint_stack')
        source_selection=compile(ast.Module(body=loop.body[begin:begin+3],type_ignores=[]),str(source),'exec')
        rows=[dict(camera_id=f'cam{i+1:02d}',frame_index=0,time=0.) for i in range(17)]
        reference_rng=random.Random(7)
        namespace={'scene':SimpleNamespace(getTrainCameras=lambda:rows.copy()),'randint':reference_rng.randint,
                   'random':reference_rng,'dataset':SimpleNamespace(sample_more_highres=False)}
        expected=[]
        for _ in range(2):
            namespace['viewpoint_stack']=None
            observed=[]
            for _ in range(5):
                exec(source_selection,namespace)
                observed.append(namespace['viewpoint_cam']['camera_id'])
            expected.append(observed)
        self.assertNotEqual(set(expected[0]),set(expected[1]))
        parameter=torch.nn.Parameter(torch.tensor(1.))
        optimizer=torch.optim.SGD([parameter],lr=.01)
        metadata=[];actual=[[],[]];clocks=[]
        backend=SimpleNamespace(g=SimpleNamespace(optimizer=optimizer,get_xyz=torch.zeros(1,3),
            update_learning_rate=clocks.append,oneupSHdegree=lambda:None))
        def save(path,meta):metadata.append(meta);return {'path':str(path),'sha256':'contract'}
        backend.save=save
        shared_rng=random.Random(7)
        for round_index in range(2):
            out=self.root/f'rng_round_{round_index}';out.mkdir()
            def loss(_backend,batch,*_):
                actual[round_index].append(batch[0]['camera_id'])
                return parameter.square(),[]
            with mock.patch.object(train_stage,'batch_loss',side_effect=loss),mock.patch.object(train_stage,'topology'):
                train_stage.optimize(backend,dict(self.config,method='threedsr',seed=7),rows,'sr',{},5,out,
                                     sampling_rng=shared_rng)
        self.assertEqual(actual,expected)
        self.assertEqual(clocks,[1,2,3,4,5]*2)
        self.assertEqual(metadata[0]['sampling']['python_rng_state_end'],metadata[1]['sampling']['python_rng_state_start'])
        self.assertEqual(metadata[1]['sampling']['python_rng_state_end'],reference_rng.getstate())
        restored=random.Random();restored.setstate(metadata[0]['sampling']['python_rng_state_end'])
        namespace.update(randint=restored.randint,random=restored,viewpoint_stack=None)
        replay=[]
        for _ in range(5):
            exec(source_selection,namespace);replay.append(namespace['viewpoint_cam']['camera_id'])
        self.assertEqual(replay,actual[1])

    def test_training_and_eval_use_method_renderer_with_same_time_train_references(self):
        import torch
        from methods import threedsr_renderer,iesrgs_renderer
        rows=self.rows
        def camera(obs,res='hr',allow_heldout=False):
            return SimpleNamespace(camera_id=obs['camera_id'],time=obs['time'],frame=obs['frame_index'],res=res,
                                   allow_heldout=allow_heldout)
        backend=SimpleNamespace(torch=torch,observations=rows,config={},g=SimpleNamespace(active_sh_degree=1),
            camera=camera,effective_state=lambda t:{'t':t},background=torch.zeros(3),
            lr=lambda o:torch.zeros(3,16,16),regularizer=lambda:torch.tensor(0.))
        observed=[]
        def render(cam,state,bg,degree,refs,**kwargs):
            observed.append((cam,refs,kwargs))
            self.assertEqual({ref.time for ref in refs},{cam.time})
            self.assertNotIn('cam00',{ref.camera_id for ref in refs})
            return {'render':torch.zeros(3,16 if cam.res=='lr' else 64,16 if cam.res=='lr' else 64)}
        for method,module,stage in [('threedsr',threedsr_renderer,'lr'),('iesrgs',iesrgs_renderer,'internal')]:
            backend.config=dict(self.config,method=method)
            with mock.patch.object(module,'render_state',side_effect=render):
                loss,_=train_stage.batch_loss(backend,[rows[0]],stage,None,backend.config)
                self.assertEqual(observed[-1][0].res,'lr')
                held=dict(rows[0],camera_id='cam00',split='test')
                result=render_eval.render_observation(backend,held,allow_heldout=True)
                self.assertEqual(observed[-1][0].camera_id,'cam00')
                self.assertEqual(observed[-1][0].res,'hr')
                self.assertTrue(observed[-1][0].allow_heldout)
                self.assertEqual(result.shape,(3,64,64))
        with self.assertRaises(ValueError):
            train_stage.batch_loss(backend,[rows[0],rows[-1]],'internal',None,backend.config)

    def test_documented_cli_options_parse_without_cuda(self):
        commands=[('train_stage.py',['--config','--stage','--prior-bundle','--wiring-fixture-steps','--check-only']),
                  ('prepare_priors.py',['--kind','--checkpoint','--rasterizer-source','--internal-index']),
                  ('bundle_priors.py',['--sr-receipt','--weights-receipt','--model','--output']),
                  ('run_threedsr.py',['--records','--lr-checkpoint','--stable-checkpoint','--vqgan-checkpoint']),
                  ('methods/sequence_matters.py',['--manifest','--psrt-checkpoint','--spynet-checkpoint','--plan-only'])]
        for filename,options in commands:
            result=subprocess.run([sys.executable,str(BASE/filename),'--help'],capture_output=True,text=True,check=True)
            for option in options:self.assertIn(option,result.stdout)


if __name__=='__main__':unittest.main()
