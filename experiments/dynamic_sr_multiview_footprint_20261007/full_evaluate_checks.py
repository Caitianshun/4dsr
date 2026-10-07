"""CPU-only contracts; launch with CUDA_VISIBLE_DEVICES='' in a new process."""
from __future__ import annotations
import ast,copy,json,math,os,sys,time
from pathlib import Path
from types import SimpleNamespace

if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('New CPU subprocess must have CUDA_VISIBLE_DEVICES empty')
import torch
if torch.cuda.is_initialized():raise RuntimeError('CUDA was initialized before CPU contracts')
torch.set_num_threads(1)
import numpy as np
import cv2
cv2.setNumThreads(1)
import full_evaluate as ev
import full_refine as native
from fp_common import ROOT,HERE,OUT,read,write,sha,entry

MANIFEST=ROOT/'data/dynamic_sr/full_time_20261007/prepared/cook_spinach/manifest_train_ready_seed20261007.json'
CHECKPOINT=OUT/'full_LR_prefixes/cook_spinach/seed_20261007/checkpoint_fine_14000.pt'


def run():
    started=time.monotonic();target=OUT/'operator_checks/full_evaluate'/str(time.time_ns());target.mkdir(parents=True)
    checks=[]
    def check(name,fn):
        fn();assert not torch.cuda.is_initialized();checks.append(name)
    def rejects(fn):
        try:fn()
        except (ValueError,KeyError,FileNotFoundError):return
        raise AssertionError('Invalid identity/contract was accepted')
    manifest,m,data,test=ev.validate_full_manifest(MANIFEST)
    def exact_full():
        assert len(test)==300 and {r['camera_id'] for r in test}=={'cam00'}
        assert m['splits']['dev']==[] and 'cam01' in m['splits']['train']
        assert all(r['time']==r['frame_index']/300 for r in test)
        assert m['resolutions']==dict(hr=[1344,1008],lr=[336,252])
    check('real_full300_split_time_corrected_camera_metadata',exact_full)
    protocol=ev.register(MANIFEST,target/'protocol.json')
    check('CPU_register_no_prediction_or_reference_decode',lambda: (ev.validate_protocol(protocol,m,data),ev.verify_sources(protocol),assert_equal(protocol['HR_quality_inspected_during_registration'],False)))
    check('old_fixed_ROI_reference_bytes_match_full_crop',lambda: assert_equal((protocol['ROI']['frames'],len(protocol['ROI']['reference_images'])),([40,80],2)))
    check('testLR_forbidden_as_training_teacher_diagnostic',lambda: rejects(lambda:ev.register(MANIFEST,target/'bad_train.json',train_keys=[['cam00',1]],teacher_index=target/'missing.json')))
    held=copy.deepcopy(m);held['role']='held_confirmation';p=copy.deepcopy(protocol);p['role']='held_confirmation'
    check('held_confirmation_without_actual_selection_refused',lambda:rejects(lambda:ev.validate_protocol(p,held,data)))
    unknown=copy.deepcopy(m);unknown['role']='unknown';p2=copy.deepcopy(protocol);p2['role']='unknown'
    check('unknown_data_role_refused',lambda:rejects(lambda:ev.validate_protocol(p2,unknown,data)))
    bad=copy.deepcopy(protocol);bad['train_teacher_diagnostics']['keys']=[['cam00',1]]
    check('post_registration_test_diagnostic_injection_refused',lambda:rejects(lambda:ev.validate_protocol(bad,m,data)))
    bad=copy.deepcopy(protocol);bad['test_keys']=bad['test_keys'][:-1]
    check('299_frame_protocol_refused',lambda:rejects(lambda:ev.validate_protocol(bad,m,data)))
    bad=copy.deepcopy(protocol);bad['sources'][str(Path(ev.__file__).relative_to(ROOT))]='0'*64
    check('source_mutation_refused',lambda:rejects(lambda:ev.verify_sources(bad)))
    image=ev.camera_item(m,test[0],torch)
    def synthetic_camera():
        assert image['image'].device.type=='cpu' and image['image'].shape==(3,252,336)
        assert not image['image'].any() and np.array_equal(image['K'],np.array(m['cameras']['cam00']['K_lr']))
        assert not {'hr_path','lr_path','hr_image','lr_image'} & set(image)
    check('camera_pose_from_synthetic_CPU_payload_no_test_pixels',synthetic_camera)
    identity,plan=ev.validate_checkpoint(CHECKPOINT,data['manifest'])
    check('actual_17000_parent_sidecar_one_Adam_eight_groups',lambda:assert_equal((identity['schema'],identity['metadata']['completed_iterations'],[g['name'] for g in identity['metadata']['audit']['optimizer_groups']]),(ev.PARENT_SCHEMA,17000,list(ev.GROUPS))))
    check('actual_parent_all13_author_runtime_files_exact_SHA',lambda:assert_equal(set(ev.validate_author_runtime(plan,Path('/home/cai_tianshun/Project/4dgs'))),set(ev.prefix.RUNTIME_FILES)))
    wrongplan=copy.deepcopy(plan);wrongplan['runtime_source_sha256'].pop('utils/general_utils.py')
    check('incomplete_author_runtime_refused',lambda:rejects(lambda:ev.validate_author_runtime(wrongplan,Path('/home/cai_tianshun/Project/4dgs'))))
    # Sidecar-only fixture tests the future SR producer's schema/field contract.
    # It is never loaded or classified as an actually trained checkpoint.
    fake=target/'future_SR_sidecar_fixture';fake.mkdir();fakecp=fake/'checkpoint_000001.pt';fakecp.write_bytes(b'CPU-sidecar-contract-not-a-model')
    srplan=dict(schema=ev.SR_SCHEMA,status='registered_ready_full_native_SR_refinement',data=data,author_commit=ev.prefix.PIN,representation=ev.REPRESENTATION,schedule=dict(parameter_updates=6000),parent=dict(checkpoint=identity['checkpoint']))
    srmeta=dict(plan_sha256=ev.prefix.digest(srplan),representation=ev.REPRESENTATION,audit=identity['metadata']['audit'],cursor=1,density_statistical_boundary_applied=True)
    write(fake/'config.json',srplan);write(fakecp.with_suffix('.json'),dict(**entry(fakecp),schema=ev.SR_SCHEMA,metadata=srmeta))
    check('future_SR_schedule_parameter_updates_field_compatible',lambda:assert_equal(ev.validate_checkpoint(fakecp,data['manifest'])[0]['schema'],ev.SR_SCHEMA))
    srmeta['density_statistical_boundary_applied']=False;write(fakecp.with_suffix('.json'),dict(**entry(fakecp),schema=ev.SR_SCHEMA,metadata=srmeta))
    check('future_SR_missing_one_time_boundary_proof_refused',lambda:rejects(lambda:ev.validate_checkpoint(fakecp,data['manifest'])))
    write(fakecp.with_suffix('.json'),dict(**entry(fakecp),schema='short_2Adam_checkpoint',metadata=srmeta))
    check('short_twoAdam_sidecar_refused_before_tensors',lambda:rejects(lambda:ev.validate_checkpoint(fakecp,data['manifest'])))
    # map_location CPU is a hard tensor-deserialization boundary. Do not call
    # native restore here: its parent has real CUDA RNG states for later GPU use.
    payload=native.checkpoint_payload(CHECKPOINT,torch,plan,ev.PARENT_SCHEMA,'cpu')
    check('actual_parent121MB_native14tuple_CPU_load',lambda:ev.validate_payload(payload,identity,plan))
    def all_cpu(value):
        if torch.is_tensor(value):assert value.device.type=='cpu'
        elif isinstance(value,dict):
            for v in value.values():all_cpu(v)
        elif isinstance(value,(tuple,list)):
            for v in value:all_cpu(v)
    check('all_saved_model_Adam_buffers_grads_RNG_tensors_CPU',lambda:all_cpu(payload))
    bad=dict(payload);bad['schema']='dynamic_sr_short_two_Adam'
    check('short_custom_payload_refused',lambda:rejects(lambda:ev.validate_payload(bad,identity,plan)))
    bad=dict(payload);bad['model']=payload['model'][:-1]
    check('13tuple_native_conversion_refused',lambda:rejects(lambda:ev.validate_payload(bad,identity,plan)))
    bad=dict(payload);bad['model']=list(payload['model']);bad['model'][12]=dict(param_groups=[dict(name='xyz'),dict(name='children')])
    check('two_Adam_children_payload_refused',lambda:rejects(lambda:ev.validate_payload(bad,identity,plan)))
    bad=dict(payload);bad.pop('parameter_gradients')
    check('missing_saved_gradients_refused',lambda:rejects(lambda:ev.validate_payload(bad,identity,plan)))
    raw=torch.linspace(-.6,1.6,3*64*64).reshape(3,64,64)
    legacy,closure,roi,fft=ev.metric_helpers();pred=legacy.image_array(raw);gt=np.full_like(pred,.47);empty=np.zeros((64,64),bool)
    check('clamped_float32_pred_not_PNG_quantized',lambda: (assert_equal(pred.dtype,np.float32),assert_true(np.any(np.abs(pred*255-np.rint(pred*255))>1e-5))))
    def no_shave():
        q=legacy.spatial_metrics(pred,gt,empty)['full'];assert q['mse']==float(((pred-gt)**2).mean(axis=2).mean())
        assert q['ssim']==float(legacy.ssim_map_rgb(pred,gt)[5:-5,5:-5].mean())
    check('PSNR_float32_full_no_shave_SSIM_outer5',no_shave)
    import torch.nn.functional as F
    observed=F.interpolate(raw[None],size=(16,16),mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1).permute(1,2,0).numpy()
    def closure_order():
        value=closure(raw,observed,empty)['full'];assert value['mse']==0 and value['l1']==0
        wrong=F.interpolate(raw.clamp(0,1)[None],size=(16,16),mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1).permute(1,2,0).numpy()
        assert np.max(np.abs(wrong-observed))>1e-4
    check('LR_closure_filters_raw_before_clamp_float64_error',closure_order)
    check('FFT_absolute_band_energy_Parseval',lambda:assert_true(abs(sum(fft.decompose(pred,gt)[b+'_mse'] for b in ('low','mid','high'))-fft.decompose(pred,gt)['mse'])<1e-11))
    zero=fft.decompose(gt,gt)
    check('perfect_prediction_FFT_shares_undefined_None',lambda:assert_equal(ev.spectrum_summary([zero])['shares_of_total_MSE'],dict(low=None,mid=None,high=None)))
    def equal_psnr():
        rows=[dict(mse=.0001,psnr=40.),dict(mse=.04,psnr=legacy.metric_psnr(.04))];v=ev.aggregate_flat(rows,('mse','psnr'))['equal_observation_means']
        assert abs(v['psnr']-legacy.metric_psnr(v['mse']))>5
    check('equal_frame_PSNR_separate_from_pooled_MSE_PSNR',equal_psnr)
    # This is a real cached AlexNet/v0.1 CPU LPIPS forward, not a raster mock.
    import lpips
    metric=lpips.LPIPS(net='alex',version='0.1',spatial=False,pnet_rand=False,pretrained=True).eval().requires_grad_(False)
    metric_hash=native.tree_digest(metric.state_dict());q=legacy.spatial_metrics(pred,gt,empty,metric,'cpu')['full'];r=roi(legacy,pred,gt,{'fixed':[12,12,44,44]},metric,'cpu')['fixed']
    check('real_CPU_scalar_LPIPS_and_full_context_spatial_ROI',lambda:(assert_true(math.isfinite(q['lpips_alex']) and math.isfinite(r['lpips_alex_spatial_mask'])),assert_equal(metric.spatial,False),assert_equal(native.tree_digest(metric.state_dict()),metric_hash)))
    runtime=ev.metric_runtime(metric,lpips,torch,native)
    check('actual_LPIPS_weight_source_loaded_state_SHA256_recorded',lambda:assert_equal(runtime['LPIPS']['loaded_state_SHA256'],metric_hash))
    def state_digest():
        original=native.tree_digest(dict(model=payload['model'],accum=payload['deformation_accum'],gradients=payload['parameter_gradients'],rng=payload['rng']))
        value=payload['model'][1];old=value[0,0].item()
        with torch.no_grad():value[0,0]=old+1.
        changed=native.tree_digest(dict(model=payload['model'],accum=payload['deformation_accum'],gradients=payload['parameter_gradients'],rng=payload['rng']))
        with torch.no_grad():value[0,0]=old
        assert changed!=original and original==native.tree_digest(dict(model=payload['model'],accum=payload['deformation_accum'],gradients=payload['parameter_gradients'],rng=payload['rng']))
    check('actual_native_model_Adam_buffers_grads_RNG_digest_detects_mutation',state_digest)
    def static_no_updates():
        tree=ast.parse(Path(ev.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        assert not any(n.func.attr in ('step','zero_grad','backward','densify_and_prune','prune','grow','reset_opacity','initialize_SR_statistics') for n in calls)
        assert any(isinstance(n,ast.Attribute) and n.attr=='no_grad' for n in ast.walk(fn))
    check('evaluation_no_Adam_backward_topology_or_density_reset_calls',static_no_updates)
    assert not torch.cuda.is_initialized()
    frozen=target/'source_snapshot';frozen.mkdir()
    for p in (Path(ev.__file__),Path(__file__)):(frozen/p.name).write_bytes(p.read_bytes())
    result=dict(status='passed_full_native_evaluation_CPU_contracts',checks=checks,count=len(checks),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized_before=False,cuda_initialized_after=torch.cuda.is_initialized(),native_raster_calls=0,GPU_forwards=0,Adam_calls=0,backward_calls=0,HR_images_decoded=0,LR_images_decoded=0,actual_parent=identity,sources=ev.sources(),checks_source=entry(Path(__file__)),metric_runtime=runtime,wall_seconds=time.monotonic()-started,protocol=entry(target/'protocol.json'),scope='CPU metadata/cached real LPIPS/float fixture/native checkpoint deserialization; no native GPU raster or300-frame image-quality result')
    write(target/'receipt.json',result);print(json.dumps(dict(status=result['status'],count=len(checks),receipt=entry(target/'receipt.json'),source=entry(Path(ev.__file__)),checks_source=entry(Path(__file__)),CUDA_initialized=False,seconds=result['wall_seconds']),indent=2))
    return result


def assert_equal(a,b):assert a==b,(a,b)
def assert_true(a):assert a
if __name__=='__main__':run()
