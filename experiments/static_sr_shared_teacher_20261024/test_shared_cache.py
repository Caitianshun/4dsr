"""CPU boundary tests with synthetic files; these are not teacher/GPU results."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from PIL import Image

try:
    from . import shared_cache as shared
    from . import build_splatsure_bundle as splat
except ImportError:
    import shared_cache as shared
    import build_splatsure_bundle as splat


class CacheContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='shared_teacher_test_', dir=shared.ROOT / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = self.root / 'scene'
        self.scene.mkdir()
        self.manifest = self.scene / 'manifest.json'
        self.index_path = self.scene / 'index.json'
        self.summary_path = self.root / 'summary.json'
        self.cache_path = self.root / 'shared.json'
        self.init = self.scene / 'init.npz'
        self.init.write_bytes(b'test initializer identity, never loaded by a model')
        self.model = self.root / 'areaft.pth'
        self.model.write_bytes(b'CPU test weight identity, no inference')
        self.base_model = self.root / 'base.pth'
        self.base_model.write_bytes(b'CPU test base model identity')
        self.ft_code = self.root / 'historical_training.py'
        self.ft_code.write_text('# independently recorded historical training source\n')
        self.data_index = self.root / 'external_data.json'
        self.save(self.data_index, dict(purpose='area-matched SwinIR fine-tuning (non-evaluation sources)',
            frames={'hr/external/0.png':dict(source='external', video='external/video.mp4')}))
        self.summary = dict(status='completed', weights=shared.identity(self.model),
            base_weights=shared.identity(self.base_model), data_index=shared.identity(self.data_index),
            train_frames=1, val_frames=1, degradation='round(4x4 box mean), 8-bit sRGB', code=shared.identity(self.ft_code))
        self.save(self.summary_path, self.summary)
        observations=[];entries=[]
        for camera in ['cam00','cam01','cam02','cam03']:
            for frame in [0, 1]:
                # Deliberately absent held-out LR/HR assets prove they are not read.
                lr = self.scene / f'lr/{camera}/{frame:04d}.png'
                lr_sha = '0'*64
                if camera != 'cam00':
                    lr.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (4,3), (24+frame,80,180)).save(lr)
                    lr_sha=shared.sha(lr)
                    sr=self.scene/f'{shared.DEFAULT_TAG}/{camera}/{frame:04d}.png'
                    sr.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB',(16,12),(24+frame,80,180)).save(sr)
                    entries.append(dict(camera=camera,frame=frame,relative_path=str(sr.relative_to(self.scene)),
                        sha256=shared.sha(sr),lr_sha256=lr_sha,lr_relative_path=str(lr.relative_to(self.scene))))
                observations.append(dict(camera_id=camera,frame_index=frame,time=frame/300,split='test' if camera=='cam00' else 'train',
                    lr_path=f'lr/{camera}/{frame:04d}.png',lr_sha256=lr_sha,hr_path=f'hr/{camera}/{frame:04d}.png',hr_sha256='0'*64))
        self.m=dict(schema='n3dv_dynamic_sr_pilot_v1',scene='synthetic_boundary_scene',scale=4,
            physical_lr_derivation={'formation':'area4'},splits={'train':['cam01','cam02','cam03'],'test':['cam00'],'dev':[]},
            frame_indices=[0,1],resolutions={'lr':[4,3],'hr':[16,12]},observations=observations,
            initialization={'train_only_lr':True,'npz_path':'init.npz','npz_sha256':shared.sha(self.init)})
        self.save(self.manifest,self.m)
        self.index=dict(schema=1,status='completed_teacher_inventory',scene=self.m['scene'],
            manifest=shared.identity(self.manifest)['path'],manifest_sha256=shared.sha(self.manifest),oracle=False,
            seed_independent_frozen_prior=True,train_cameras=['cam01','cam02','cam03'],frame_indices=[0,1],entries=entries,
            teacher_config=dict(kind='physlr_teacher',tag=shared.DEFAULT_TAG,teacher='swinir_area_finetuned_x4',scale=4,
                weights=shared.identity(self.model),code=shared.identity(shared.ROOT/'experiments/dynamic_sr_physlr_20261016/teacher_physlr3.py')))
        self.save(self.index_path,self.index)

    def save(self,p,v):
        Path(p).write_text(json.dumps(v,indent=2)+'\n')

    def imported(self):
        return shared.import_cache(self.manifest,self.index_path,finetune_summary=self.summary_path)

    def cached(self):
        value=self.imported();shared.write_new(self.cache_path,value);return value

    def test_real_inventory_contract_preserves_bytes_and_has_no_heldout_reads(self):
        before=shared.sha(self.index_path);value=self.cached()
        self.assertEqual(len(value['entries']),6)
        self.assertEqual(shared.verify_cache(self.cache_path,self.manifest),value)
        self.assertEqual(before,shared.sha(self.index_path))
        self.assertFalse((self.scene/'lr/cam00').exists())
        self.assertEqual(value['entries'][3]['time'],1/300)
        self.assertFalse(value['inference_executed']);self.assertFalse(value['original_method_receipt'])
        self.assertEqual(value['decode']['tensor'],'float32_CHW_div255')
        self.assertFalse(value['decode']['w4']);self.assertFalse(value['decode']['colour_correction'])

    def test_cache_relabel_or_time_mutation_is_not_a_valid_import(self):
        self.cached();value=shared.read(self.cache_path);value['entries'][0]['time']=.5
        self.save(self.cache_path,value)
        with self.assertRaises(ValueError):shared.verify_cache(self.cache_path)

    def test_heldout_duplicate_and_missing_inventory_entries_are_rejected(self):
        for change in ('heldout','duplicate','missing'):
            with self.subTest(change=change):
                value=copy.deepcopy(self.index)
                if change=='heldout':value['entries'][0]['camera']='cam00'
                elif change=='duplicate':value['entries'].append(value['entries'][0])
                else:value['entries'].pop()
                self.save(self.index_path,value)
                with self.assertRaises(ValueError):self.imported()

    def test_cross_scene_manifest_or_wrong_LR_identity_is_rejected(self):
        for change in ('scene','manifest','lr'):
            with self.subTest(change=change):
                value=copy.deepcopy(self.index)
                if change=='scene':value['scene']='different_scene'
                elif change=='manifest':value['manifest_sha256']='0'*64
                else:value['entries'][0]['lr_sha256']='0'*64
                self.save(self.index_path,value)
                with self.assertRaises(ValueError):self.imported()

    def test_byte_mutation_of_LR_or_SR_is_rejected(self):
        for field in ['lr_relative_path','relative_path']:
            with self.subTest(field=field):
                p=self.scene/self.index['entries'][0][field];old=p.read_bytes();p.write_bytes(old+b'changed')
                with self.assertRaises(ValueError):self.imported()
                p.write_bytes(old)

    def test_wrong_size_or_non_RGB_PNG_rejected_even_with_new_recorded_hash(self):
        for mode,size in [('RGB',(8,6)),('RGBA',(16,12)),('L',(16,12))]:
            with self.subTest(mode=mode,size=size):
                p=self.scene/self.index['entries'][0]['relative_path'];Image.new(mode,size).save(p)
                value=copy.deepcopy(self.index);value['entries'][0]['sha256']=shared.sha(p);self.save(self.index_path,value)
                with self.assertRaises(ValueError):self.imported()

    def test_path_substitution_traversal_and_symlink_rejected(self):
        for target in ['../outside.png','sr_swinir_x4_areaft/cam02/0000.png']:
            value=copy.deepcopy(self.index);value['entries'][0]['relative_path']=target;self.save(self.index_path,value)
            with self.assertRaises(ValueError):self.imported()
        self.save(self.index_path,self.index)
        p=self.scene/self.index['entries'][0]['relative_path'];other=self.root/'same.png';p.rename(other);p.symlink_to(other)
        with self.assertRaises(ValueError):self.imported()

    def test_producer_and_model_cannot_be_replaced_silently(self):
        self.model.write_bytes(b'new weights')
        with self.assertRaises(ValueError):self.imported()
        self.model.write_bytes(b'CPU test weight identity, no inference')
        fake=self.root/'unknown.py';fake.write_text('# corrected-image producer\n')
        self.index['teacher_config']['code']=shared.identity(fake);self.save(self.index_path,self.index)
        with self.assertRaises(ValueError):self.imported()

    def test_historical_training_source_difference_is_disclosed_not_rewritten(self):
        old=copy.deepcopy(self.summary['code']);self.ft_code.write_text('# later independent source update\n')
        value=self.imported()['model_provenance']
        self.assertEqual(value['historical_training_code'],old)
        self.assertFalse(value['historical_training_code_currently_matches'])
        self.assertEqual(shared.read(self.summary_path)['code'],old)

    def test_index_time_or_noninteger_key_rejected(self):
        for field,value in [('time',.8),('frame',True)]:
            changed=copy.deepcopy(self.index);changed['entries'][0][field]=value;self.save(self.index_path,changed)
            with self.assertRaises(ValueError):self.imported()

    def test_output_refuses_overwrite_and_duplicate_JSON_keys(self):
        self.cached()
        with self.assertRaises(FileExistsError):shared.write_new(self.cache_path,{})
        self.index_path.write_text('{"schema":1,"schema":1}')
        with self.assertRaises(ValueError):self.imported()


class SplatBundleTests(unittest.TestCase):
    """Synthetic checkpoint/maps exercise the real legacy preflight interface."""
    save = CacheContractTests.save
    imported = CacheContractTests.imported
    cached = CacheContractTests.cached

    def setUp(self):
        CacheContractTests.setUp(self);self.cached()
        source=self.root/'source_plan.json';self.save(source,{'configuration':{'test_only':'no model constructed'}})
        self.base_config=self.root/'base_config.json'
        self.cfg=dict(schema='static_sr_wu_adaptation_v1',method='splatsure',scene=self.m['scene'],
            label='old main teacher label',teacher_variant='stablesr',manifest=shared._common().identity(self.manifest),
            source_configuration=shared._common().identity(source),wu_configuration={'test_only':'no model constructed'},
            initialization='fresh_scene_local_train_LR_points',degradation='area4',information={'train_only':True},
            training={'steps':30000,'batch_views':1,'stage':'fine'},specification={'lr_steps':30000,'sr_steps':30000},seed=7)
        self.save(self.base_config,self.cfg)
        self.config_path=self.root/'shared_config.json';self.save(self.config_path,splat.build_config(self.base_config,self.cache_path))
        import torch
        self.checkpoint=self.root/'lr.pt'
        torch.save(dict(schema='static_sr_wu_checkpoint_v1',adaptation_config=self.cfg,
                        metadata={'stage':'lr','iterations':30000,'optimizer_updates':29999},model=[]),self.checkpoint)
        self.lr_completion=self.root/'lr_complete.json'
        self.complete=dict(status='stage_completed',stage='lr',steps=30000,wiring_fixture=False,
            prior_bundle=None,training_HR_read=False,heldout_pixels_read=False,training_keys=6,
            config=shared.identity(self.base_config),checkpoint=shared.identity(self.checkpoint))
        self.save(self.lr_completion,self.complete)
        binary=self.root/'test_only_binary.so';binary.write_bytes(b'CPU fixture identity; never loaded')
        self.weight_dir=self.root/'weight_cache';(self.weight_dir/'maps').mkdir(parents=True)
        records=[]
        for e in shared.read(self.cache_path)['entries']:
            name=f"maps/{e['camera']}_{e['frame']:06d}.pt";p=self.weight_dir/name
            torch.save(torch.ones(3,4,dtype=torch.float32)*.5,p)
            records.append(dict(camera_id=e['camera'],frame_index=e['frame'],time=e['time'],path=name,sha256=shared.sha(p),shape=[3,4]))
        self.weights=self.weight_dir/'receipt.json'
        self.weight_document=dict(schema='wu_splatsure_weights_v1',status='prepared',scene=self.m['scene'],
            manifest_sha256=shared.sha(self.manifest),manifest_source=str(self.manifest),training_only=True,
            time_policy='same_time_train_cameras_only',requested_frame_indices=None,parameter_updates=0,HR_image_reads=0,
            upstream_commit=splat.UPSTREAM,rasterizer_commit=splat.RASTERIZER,config=splat.MAP_CONFIG,
            rasterizer_settings={'antialiasing':False,'num_contributions':10},
            provenance=dict(parent_checkpoint=shared.identity(self.checkpoint),parent_checkpoint_sha256=shared.sha(self.checkpoint),
                source_config=shared.identity(self.base_config),backend=shared.identity(splat.EXP/'prepare_priors.py')),
            producer_files=[shared.identity(splat.EXP/'methods/splatsure_prepare.py'),shared.identity(splat.EXP/'methods/splatsure.py')],
            rasterizer_binary_identities=[shared.identity(binary)],records=records,render_calls=18)
        self.save(self.weights,self.weight_document)

    def bundle(self):return splat.build_bundle(self.config_path,self.cache_path,self.weights,self.lr_completion)

    def test_teacher_only_config_and_real_frozen_preflight_acceptance(self):
        result=self.bundle();p=self.root/'bundle.json';self.save(p,result)
        new=shared.read(self.config_path)
        changed={k for k in self.cfg if self.cfg[k]!=new[k]}
        self.assertEqual(changed,{'teacher_variant','label','information'})
        self.assertEqual(result['manifest'],self.cfg['manifest'])
        self.assertEqual(result['teacher_variant'],shared.VARIANT)
        self.assertFalse(result['LR_parent_reuse']['original_checkpoint_or_metadata_rewritten'])
        sys.path.insert(0,str(splat.EXP))
        try:
            spec=importlib.util.spec_from_file_location('shared_test_frozen_train_stage',splat.EXP/'train_stage.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            m,rows,priors=module.preflight(new,'sr',p)
            self.assertEqual(len(priors),6)
            actual=module.load_tensor(result['entries'][0]['sr'],device='cpu')
            self.assertEqual(tuple(actual.shape),(3,12,16))
            self.assertAlmostEqual(float(actual[0,0,0]),24/255,places=7)
        finally:sys.path.remove(str(splat.EXP))

    def test_changes_to_Wu_schedule_or_existing_information_fail(self):
        for part in ['training','wu_configuration','information']:
            original=shared.read(self.config_path);changed=copy.deepcopy(original);changed[part]['unauthorized']=1
            self.save(self.config_path,changed)
            with self.assertRaises(ValueError):self.bundle()
            self.save(self.config_path,original)

    def test_wiring_or_failed_LR_cannot_supply_formal_maps(self):
        for change in [dict(wiring_fixture=True),dict(status='failed'),dict(steps=10)]:
            self.save(self.lr_completion,dict(self.complete,**change))
            with self.assertRaises(ValueError):self.bundle()

    def test_parent_payload_must_really_have_30k_LR_metadata(self):
        import torch
        payload=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        payload['metadata']['optimizer_updates']=1;torch.save(payload,self.checkpoint)
        self.complete['checkpoint']=shared.identity(self.checkpoint);self.save(self.lr_completion,self.complete)
        self.weight_document['provenance']['parent_checkpoint']=shared.identity(self.checkpoint)
        self.weight_document['provenance']['parent_checkpoint_sha256']=shared.sha(self.checkpoint)
        self.save(self.weights,self.weight_document)
        with self.assertRaises(ValueError):self.bundle()

    def test_incomplete_wrong_time_or_wrong_parent_contributions_rejected(self):
        for kind in ['missing','time','parent']:
            value=copy.deepcopy(self.weight_document)
            if kind=='missing':value['records'].pop()
            elif kind=='time':value['records'][0]['time']=.7
            else:value['provenance']['parent_checkpoint_sha256']='0'*64
            self.save(self.weights,value)
            with self.assertRaises(ValueError):self.bundle()

    def test_contribution_values_and_shapes_are_checked_beyond_hashes(self):
        import torch
        for value in [torch.ones(2,2),torch.full((3,4),float('nan')),torch.full((3,4),1.1)]:
            doc=copy.deepcopy(self.weight_document);row=doc['records'][0];p=self.weight_dir/row['path']
            torch.save(value,p);row['sha256']=shared.sha(p);self.save(self.weights,doc)
            with self.assertRaises(ValueError):self.bundle()

    def test_portable_parent_bundle_retains_original_receipt_and_mapping(self):
        imported = self.root/'portable.json'
        value = splat.import_lr_parent(self.lr_completion,self.checkpoint,self.base_config,
            self.config_path,str(self.lr_completion))
        self.save(imported,value)
        result=splat.build_bundle(self.config_path,self.cache_path,self.weights,imported)
        provenance=result['LR_parent_reuse']['provenance']
        self.assertEqual(provenance['mode'],'explicit_portable_import')
        self.assertEqual(provenance['original_completion'],shared.identity(self.lr_completion))
        self.assertEqual(provenance['origin_checkpoint'],self.complete['checkpoint'])
        self.assertEqual(provenance['imported_checkpoint'],shared.identity(self.checkpoint))
        self.assertEqual(shared.read(self.lr_completion),self.complete)


class PortableLRTests(unittest.TestCase):
    """Two real temporary host roots, immutable original JSON/checkpoint copies."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='portable_lr_test_',dir=shared.ROOT/'tmp')
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.origin=root/'origin_host';self.origin.mkdir()
        self.remote=root/'remote_host';self.remote.mkdir()
        self.source=self.remote/'source.json';self.save(self.source,{'configuration':{'geometry':'fixture only'}})
        scene=self.remote/'scene';scene.mkdir();(scene/'init.npz').write_bytes(b'initializer identity')
        rows=[]
        for camera in ('cam01','cam02','cam03'):
            lr=scene/f'lr/{camera}/0000.png';lr.parent.mkdir(parents=True)
            Image.new('RGB',(4,3),(16,32,64)).save(lr)
            rows.append(dict(camera_id=camera,frame_index=0,time=0.0,split='train',
                lr_path=f'lr/{camera}/0000.png',lr_sha256=shared.sha(lr),hr_path=f'hr/{camera}/0000.png',hr_sha256='0'*64))
        manifest=dict(schema='n3dv_dynamic_sr_pilot_v1',scene='portable_scene',scale=4,
            physical_lr_derivation={'formation':'area4'},splits={'train':['cam01','cam02','cam03'],'test':['cam00'],'dev':[]},
            frame_indices=[0],resolutions={'lr':[4,3],'hr':[16,12]},observations=rows,
            initialization={'train_only_lr':True,'npz_path':'init.npz','npz_sha256':shared.sha(scene/'init.npz')})
        self.manifest=scene/'manifest.json';self.save(self.manifest,manifest)
        self.cfg=dict(schema='static_sr_wu_adaptation_v1',method='splatsure',scene='portable_scene',
            teacher_variant='stablesr',label='original',manifest=shared.identity(self.manifest,self.remote),
            source_configuration=shared.identity(self.source,self.remote),wu_configuration={'geometry':'fixture only'},
            initialization='fresh_scene_local_train_LR_points',degradation='area4',information={},training={'steps':30000})
        self.old_config=self.origin/'config.json';self.save(self.old_config,self.cfg)
        self.config=self.remote/'original_config.json';shutil.copyfile(self.old_config,self.config)
        self.target=self.remote/'shared_config.json';self.save(self.target,dict(self.cfg,teacher_variant=shared.VARIANT,label='shared'))
        import torch
        self.old_checkpoint=self.origin/'checkpoint.pt'
        torch.save(dict(schema='static_sr_wu_checkpoint_v1',adaptation_config=self.cfg,
            metadata={'stage':'lr','iterations':30000,'optimizer_updates':29999},model=[]),self.old_checkpoint)
        self.checkpoint=self.remote/'copied_checkpoint.pt';shutil.copyfile(self.old_checkpoint,self.checkpoint)
        self.complete=dict(status='stage_completed',stage='lr',steps=30000,wiring_fixture=False,
            prior_bundle=None,training_HR_read=False,heldout_pixels_read=False,training_keys=3,
            config=dict(path=str(self.old_config),sha256=shared.sha(self.old_config)),
            checkpoint=dict(path=str(self.old_checkpoint),sha256=shared.sha(self.old_checkpoint),bytes=self.old_checkpoint.stat().st_size))
        self.old_complete=self.origin/'complete.json';self.save(self.old_complete,self.complete)
        self.copied_complete=self.remote/'original_complete.json';shutil.copyfile(self.old_complete,self.copied_complete)
        self.portable=self.remote/'import.json'
        producer=self.remote/Path(splat.__file__).relative_to(shared.ROOT)
        producer.parent.mkdir(parents=True);shutil.copyfile(splat.__file__,producer)

    def save(self,p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n')

    def imported(self):
        return splat.import_lr_parent(self.copied_complete,self.checkpoint,self.config,self.target,
                                      str(self.old_complete),project_root=self.remote)

    def verify(self):
        return splat.verify_lr_parent(self.portable,shared.read(self.target,self.remote),project_root=self.remote)

    def test_foreign_absolute_paths_are_provenance_only_and_originals_are_preserved(self):
        before={p:shared.sha(p) for p in (self.old_checkpoint,self.old_complete,self.copied_complete,self.checkpoint,self.old_config,self.config)}
        self.assertFalse(self.old_checkpoint.is_relative_to(self.remote))
        with self.assertRaises(ValueError):
            splat.verify_lr_parent(self.copied_complete,shared.read(self.target,self.remote),project_root=self.remote)
        value=self.imported();self.save(self.portable,value)
        parent,record=self.verify()
        self.assertEqual(parent,shared.identity(self.checkpoint,self.remote))
        self.assertFalse(Path(parent['path']).is_absolute())
        self.assertEqual(record,self.complete)
        self.assertEqual(value['origin_checkpoint'],self.complete['checkpoint'])
        self.assertEqual(value['original_completion'],shared.identity(self.copied_complete,self.remote))
        self.assertEqual(value['origin_completion']['path'],str(self.old_complete))
        self.assertEqual(before,{p:shared.sha(p) for p in before})
        # The source host is absent; verification still opens only the copied files.
        self.old_checkpoint.rename(self.origin/'source_temporarily_unavailable.pt')
        self.old_config.rename(self.origin/'source_config_temporarily_unavailable.json')
        self.verify()

    def test_import_rejects_checkpoint_SHA_or_original_size_mismatch(self):
        old=self.checkpoint.read_bytes();self.checkpoint.write_bytes(old+b'changed')
        with self.assertRaises(ValueError):self.imported()
        self.checkpoint.write_bytes(old)
        bad=copy.deepcopy(self.complete);bad['checkpoint']['bytes']+=1;self.save(self.copied_complete,bad)
        with self.assertRaises(ValueError):self.imported()

    def test_import_rejects_other_scene_config_and_short_payload(self):
        target=shared.read(self.target,self.remote);target['scene']='other_scene';self.save(self.target,target)
        with self.assertRaises(ValueError):self.imported()
        target['scene']='portable_scene';self.save(self.target,target)
        import torch
        payload=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        payload['metadata']['optimizer_updates']=9;torch.save(payload,self.checkpoint)
        changed=copy.deepcopy(self.complete)
        changed['checkpoint']=dict(path=str(self.old_checkpoint),sha256=shared.sha(self.checkpoint),bytes=self.checkpoint.stat().st_size)
        self.save(self.copied_complete,changed)
        with self.assertRaises(ValueError):self.imported()

    def test_portable_mapping_cannot_change_origin_SHA_path_or_scene(self):
        original=self.imported()
        for kind in ('origin','checkpoint','scene','completion'):
            value=copy.deepcopy(original)
            if kind=='origin':value['origin_checkpoint']['path']='/different/original.pt'
            elif kind=='checkpoint':value['imported_checkpoint']['sha256']='0'*64
            elif kind=='scene':value['scene']='other_scene'
            else:value['origin_completion']['sha256']='0'*64
            self.save(self.portable,value)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.verify()

    def test_other_scene_checkpoint_payload_is_rejected_even_with_matching_file_hash(self):
        import torch
        payload=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        payload['adaptation_config']['scene']='a_different_scene'
        torch.save(payload,self.checkpoint)
        changed=copy.deepcopy(self.complete)
        changed['checkpoint']=dict(path=str(self.old_checkpoint),sha256=shared.sha(self.checkpoint),bytes=self.checkpoint.stat().st_size)
        self.save(self.copied_complete,changed)
        with self.assertRaises(ValueError):self.imported()

    def test_copied_completion_and_config_are_frozen_dependencies(self):
        self.save(self.portable,self.imported())
        paths=splat.lr_dependency_paths(self.portable,project_root=self.remote)
        self.assertEqual(set(paths),{self.portable,self.copied_complete,self.config,self.checkpoint})
        self.copied_complete.write_bytes(self.copied_complete.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.verify()


if __name__=='__main__':unittest.main()
