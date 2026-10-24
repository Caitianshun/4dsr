"""CPU-only contract tests; synthetic pixels never become formal priors."""
from __future__ import annotations
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

from deployment.static_sr_local_20261024 import stablesr_shards as s


class ShardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.remote = self.root/'remote'; self.remote.mkdir()
        self.returned = self.root/'returned'; self.returned.mkdir()
        from PIL import Image
        self.manifest = self.remote/'manifest.json'; self.manifest.write_text('{}')
        self.rows = []
        for i in range(4):
            lr = self.remote/f'lr{i}.png'; Image.new('RGB', (2,2), (i,0,0)).save(lr)
            self.rows.append(dict(camera='cam01', frame=i, time=i/300, split='train', lr_path=str(lr),
                lr_sha256=s.sha(lr), staged_name=f'cam01_{i:06d}.png', expected_sr_shape=[8,8]))
        self.m = dict(scene='cook_spinach')
        self.bridge = types.SimpleNamespace(
            RECIPE='official_recipe', STABLESR_COMMIT='pinned', WEIGHTS={}, RUNTIME_SHA256={},
            OPENCLIP_CACHE_REPO='repo',OPENCLIP_REVISION='revision',
            command=lambda p: [p['python'],str(Path(p['source_root'])/'author.py'),'--ddpm_steps','200',
                '--input_size','512','--seed',str(p['seed']),'--outdir',str(Path(p['output_root'])/'generated')],
            png_shape=s.load_bridge(Path(__file__).resolve().parents[2]).png_shape)
        for role in ('diffusion','cfw'):
            w=self.remote/(role+'.ckpt'); w.write_text(role); self.bridge.WEIGHTS[role]=(w.name,s.sha(w))
        self.roster_path = self.remote/'roster.json'
        with patch.object(s, 'load_bridge', return_value=self.bridge), patch.object(s, 'full_targets', return_value=(self.m,self.rows)):
            self.roster = s.partition(self.root, self.manifest, self.roster_path, 2)
        self.mappings = [str(self.remote)+'='+str(self.remote)]
        self.root_bridge = self.root/s.BRIDGE; self.root_bridge.parent.mkdir(parents=True); self.root_bridge.write_text('bridge')

    def fixture(self):
        from PIL import Image
        source=self.remote/'author.py'; source.write_text('author')
        ref=self.remote/'hf/repo/refs/main'; ref.parent.mkdir(parents=True); ref.write_text('revision')
        runtime=self.remote/'runtime.json'; runtime.write_text(json.dumps(dict(files=[],cache_mappings=[],cache_environment=dict(HUGGINGFACE_HUB_CACHE=str(self.remote/'hf')))))
        template=dict(schema='wu_splatsure_stablesr_teacher_v1',status='planned_runtime_unverified',recipe=self.bridge.RECIPE,
            scene='cook_spinach',manifest=s.identity(self.manifest),upstream_commit='pinned',source_root=str(self.remote),
            output_root=str(self.remote/'template'),execution_workdir=str(self.remote/'template/execution_workdir'),
            python=sys.executable,seed=42,producer=[s.identity(source)],
            model_identities=[dict(s.identity(self.remote/v[0]),role=role) for role,v in self.bridge.WEIGHTS.items()],
            runtime_assets=s.read(runtime),runtime_assets_identity=s.identity(runtime),targets=self.rows,
            training_only=True,parameter_updates=0,HR_images_read=0,heldout_images_read=0)
        template['command']=self.bridge.command(template)
        tp=self.remote/'template.json'; s.write_new(tp,template)
        nodes=[]; shard_plans=[]
        for sh in self.roster['shards']:
            out=self.remote/sh['id']; out.mkdir(); (out/'generated').mkdir()
            p=copy.deepcopy(template); p.update(output_root=str(out),execution_workdir=str(out/'execution_workdir'),
                targets=self.rows[sh['start']:sh['stop']],shard=dict(id=sh['id'],keys=sh['keys'],roster_sha256=s.sha(self.roster_path)))
            p['command']=self.bridge.command(p); pp=out/'plan.json'; s.write_new(pp,p)
            nodes.append(dict(id=sh['id'],keys=sh['keys'],plan=s.identity(pp)))
            shard_plans.append((sh,out,p,pp))
        job=dict(schema='stablesr_shard_job_v1',status='CPU_plan_ready_requires_broker',rng_policy=s.RNG,
            manifest=s.identity(self.manifest),roster=s.identity(self.roster_path),wrapper=s.identity(s.__file__),
            required_files=[s.identity(source)],template=s.identity(tp),nodes=nodes,visible_device='GPU-test',gpu_uuid='GPU-test',host='testhost',
            broker='local',job_id='test-job',output_root=str(self.remote))
        jp=self.remote/'job_plan.json'; s.write_new(jp,job)
        seals=[]
        for sh,out,p,pp in shard_plans:
            e=dict(status='completed',returncode=0,plan=s.identity(pp),command=p['command'],execution_workdir=p['execution_workdir'],
                source_root=p['source_root'],runtime_assets_identity=p['runtime_assets_identity'],cuda_visible_devices='GPU-test')
            ep=out/'execution.json'; s.write_new(ep,e)
            r=copy.deepcopy(p); r.update(status='completed',execution=s.identity(ep),plan=s.identity(pp))
            for row in r['targets']:
                image=out/'generated'/row['staged_name']; Image.new('RGB',(8,8),(row['frame'],3,7)).save(image)
                row['sr']=s.identity(image)
            rp=out/'receipt.json'; s.write_new(rp,r); log=out/'inference.log'; log.write_text('completed')
            seal=dict(schema='stablesr_shard_completion_v1',status='completed',shard_id=sh['id'],keys=sh['keys'],rng_policy=s.RNG,
                roster=s.identity(self.roster_path),manifest=s.identity(self.manifest),job_plan=s.identity(jp),plan=s.identity(pp),
                receipt=s.identity(rp),execution=s.identity(ep),files=[x['sr'] for x in r['targets']],inference_log=s.identity(log),
                broker=dict(gpu_uuid='GPU-test',host='testhost',CUDA_VISIBLE_DEVICES='GPU-test',cgroup='0::/claudevpn.slice/claude-local-test-job.service'))
            sp=out/'completion.json'; s.write_new(sp,seal); seals.append(sp)
        return seals

    def aggregate(self,seals,**kwargs):
        with patch.object(s,'load_bridge',return_value=self.bridge), patch.object(s,'full_targets',return_value=(self.m,self.rows)):
            return s.aggregate(self.root,self.manifest,self.roster_path,seals,kwargs.get('mappings',self.mappings),self.root/'aggregate.json')

    def test_partition_exact_and_rng_disclosed(self):
        self.assertEqual([k for sh in self.roster['shards'] for k in sh['keys']],[s.key(r) for r in self.rows])
        self.assertIn('author_outer_tile_seed42_reset_preserved',self.roster['rng_policy'])
        with patch.object(s,'full_targets',return_value=(self.m,self.rows)):
            s.verify_roster(self.roster,self.bridge,self.manifest)
            altered=copy.deepcopy(self.roster); altered['shards'][0]['keys'][0]='cam00/0000'
            with self.assertRaises(ValueError): s.verify_roster(altered,self.bridge,self.manifest)

    def test_full_protocol_rejects_fixture(self):
        b=types.SimpleNamespace(training_targets=lambda p:(dict(scene='cook_spinach'),self.rows))
        with self.assertRaises(ValueError): s.full_targets(b,self.manifest)

    def test_aggregate_complete_rebinds_without_mutating_originals(self):
        seals=self.fixture(); before={p:s.sha(p) for p in self.remote.rglob('*.json')}
        import shutil
        shutil.copytree(self.remote,self.returned/'copy')
        returned_seals=[self.returned/'copy'/p.relative_to(self.remote) for p in seals]
        result=self.aggregate(returned_seals,mappings=[str(self.remote)+'='+str(self.returned/'copy')])
        self.assertEqual(len(result['targets']),4)
        self.assertTrue(all(Path(r['sr']['path']).is_relative_to(self.returned) for r in result['targets']))
        self.assertEqual(before,{p:s.sha(p) for p in before})
        self.assertTrue(result['sharding']['original_receipts_preserved'])
        self.assertEqual({x['role'] for x in result['model_identities']},{'diffusion','cfw'})

    def test_aggregate_missing_and_duplicate_shards_rejected(self):
        seals=self.fixture()
        for selected in (seals[:1],seals+[seals[0]]):
            with self.subTest(selected=selected), self.assertRaises(ValueError): self.aggregate(selected)

    def test_failed_execution_rejected_even_with_rehashed_seal(self):
        seals=self.fixture(); seal=s.read(seals[0]); ep=Path(seal['execution']['path']); e=s.read(ep); e['returncode']=1
        ep.write_text(json.dumps(e)); seal['execution']=s.identity(ep)
        rp=Path(seal['receipt']['path']); r=s.read(rp); r['execution']=s.identity(ep); rp.write_text(json.dumps(r)); seal['receipt']=s.identity(rp)
        seals[0].write_text(json.dumps(seal))
        with self.assertRaisesRegex(ValueError,'successful'): self.aggregate(seals)

    def test_tampered_png_rejected(self):
        seals=self.fixture(); png=Path(s.read(seals[0])['files'][0]['path']); png.write_bytes(png.read_bytes()+b'corruption')
        with self.assertRaisesRegex(ValueError,'SHA'): self.aggregate(seals)

    def test_png_from_outside_original_generated_directory_rejected(self):
        seals=self.fixture(); seal=s.read(seals[0]); rp=Path(seal['receipt']['path']); r=s.read(rp)
        old=Path(r['targets'][0]['sr']['path']); foreign=self.remote/'foreign.png'; foreign.write_bytes(old.read_bytes())
        r['targets'][0]['sr']=s.identity(foreign); rp.write_text(json.dumps(r)); seal['receipt']=s.identity(rp)
        seal['files']=[x['sr'] for x in r['targets']]; seals[0].write_text(json.dumps(seal))
        with self.assertRaisesRegex(ValueError,'generated directory'): self.aggregate(seals)

    def test_host_specific_json_explicit_mapping_and_exact_bytes(self):
        seals=self.fixture(); host_json=self.remote/'host_environment.json'; host_json.write_text('{"host":"remote","path":"/remote/private"}\n')
        jp=self.remote/'job_plan.json'; job=s.read(jp); job['required_files'].append(s.identity(host_json)); jp.write_text(json.dumps(job))
        for sp in seals:
            seal=s.read(sp); seal['job_plan']=s.identity(jp); sp.write_text(json.dumps(seal))
        copy=self.returned/'original_host_environment.json'; copy.write_bytes(host_json.read_bytes())
        result=self.aggregate(seals,mappings=self.mappings+[str(host_json)+'='+str(copy)])
        self.assertEqual(result['status'],'completed'); self.assertEqual(copy.read_bytes(),host_json.read_bytes())

    def test_exact_offline_ref_no_newline(self):
        seals=self.fixture(); (self.remote/'hf/repo/refs/main').write_text('revision\n')
        with self.assertRaisesRegex(ValueError,'revision ref'): self.aggregate(seals)

    def test_remote_original_context_bytes_required_in_aggregate(self):
        job=dict(broker='remote',job_id='job',gpu_uuid='GPU-123',host='remote',visible_device='0',
            output_root=str(self.remote),root='/project/task',broker_project_root='/project')
        context=dict(version=1,job_id='job',project_root='/project',selected_gpu_uuids=['GPU-123'],supervisor_pid=123,startticks=456,boot_id='boot')
        cp=self.remote/'broker_execution_context.json'; cp.write_text(json.dumps(context))
        seal=dict(broker_context=s.identity(cp),broker=dict(gpu_uuid='GPU-123',host='remote',CUDA_VISIBLE_DEVICES='0',cgroup='0::/session',
            supervisor_context=context,execution_context=dict(path='/project/.claude-training/jobs/job/execution_context.json',sha256=s.sha(cp))))
        mapper=s.Mapper(self.mappings); s.verify_broker_provenance(seal,job,mapper)
        bad=copy.deepcopy(seal); bad['broker']['supervisor_context']['startticks']=789
        with self.assertRaises(ValueError): s.verify_broker_provenance(bad,job,mapper)
        cp.write_text(json.dumps(dict(context,job_id='other')))
        with self.assertRaises(ValueError): s.verify_broker_provenance(seal,job,mapper)

    def test_wrong_roster_manifest_or_recipe_rejected(self):
        seals=self.fixture(); seal=s.read(seals[0]); seal['manifest']['sha256']='0'*64; seals[0].write_text(json.dumps(seal))
        with self.assertRaises(ValueError): self.aggregate(seals)

    def test_mapper_longest_prefix_and_traversal_symlink_rejection(self):
        special=self.root/'special'; special.mkdir()
        m=s.Mapper(['/remote='+str(self.remote),'/remote/out='+str(special)])
        self.assertEqual(m.path('/remote/out/a'),special/'a')
        with self.assertRaises(ValueError): m.path('/remote/../secret')
        (self.remote/'link').symlink_to(special)
        with self.assertRaises(ValueError): m.path('/remote/link/a')
        with self.assertRaises(ValueError): s.Mapper(['/same='+str(special),'/same='+str(self.remote)])

    def test_storage_reserves_all_remaining_shards(self):
        nodes=[dict(additional_bytes=100),dict(additional_bytes=200)]
        with patch.object(s.shutil,'disk_usage',return_value=types.SimpleNamespace(free=s.RESERVE+299)):
            with self.assertRaises(ValueError): s.storage_guard(self.root,nodes)
        with patch.object(s.shutil,'disk_usage',return_value=types.SimpleNamespace(free=s.RESERVE+300)):
            self.assertEqual(s.storage_guard(self.root,nodes)['required_bytes'],s.RESERVE+300)

    def test_file_failure_limit_enforced_in_subprocess(self):
        code='from deployment.static_sr_local_20261024 import stablesr_shards as s; s.FILE_CAP=1024; s.enforce_file_cap(1024); f=open('+repr(str(self.root/'bounded'))+',"wb",buffering=0); f.write(b"x"*2048); f.write(b"x")'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True)
        self.assertNotEqual(result.returncode,0); self.assertLessEqual((self.root/'bounded').stat().st_size,1024)

    def test_run_no_replay_and_failure_stops_before_second_shard(self):
        jp=self.root/'run.json'; jp.write_text('{}')
        out=self.root/'job'; out.mkdir(); nodes=[]
        for i in range(2):
            d=out/f'shard{i}'; d.mkdir(); pp=d/'plan.json'; pp.write_text(json.dumps(dict(python=sys.executable)))
            nodes.append(dict(id=f'shard{i}',plan=s.identity(pp),additional_bytes=1))
        p=dict(output_root=str(out),nodes=nodes,broker='local')
        with patch.object(s,'verify_job',return_value=(p,None)),patch.object(s,'admission',return_value={}),patch.object(s,'storage_guard'),patch.object(s.subprocess,'run',return_value=types.SimpleNamespace(returncode=7)) as run:
            with self.assertRaises(RuntimeError): s.run(jp)
            self.assertEqual(run.call_count,1); self.assertEqual(s.read(out/'failure.json')['failed_shard'],'shard0')
            with self.assertRaises(FileExistsError): s.run(jp)
            self.assertEqual(run.call_count,1)

    def test_local_admission_exact_job_and_uuid(self):
        p=dict(host='test',visible_device='GPU-123',gpu_uuid='GPU-123',broker='local',job_id='shard-job')
        with patch.object(s.socket,'gethostname',return_value='test'),patch.dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-123'),patch.object(Path,'read_text',return_value='0::/claudevpn.slice/claude-local-shard-job.service'):
            s.admission(p)
            p['job_id']='other'
            with self.assertRaises(ValueError): s.admission(p)

    def test_remote_supervisor_ancestry_ticks_boot_gpu_and_cgroup(self):
        base=self.root/'project'; jobdir=base/'.claude-training/jobs/shard-job'; jobdir.mkdir(parents=True)
        context=dict(version=1,job_id='shard-job',project_id='4dsr',project_root=str(base),boot_id='boot',
            supervisor_pid=100,startticks=123,selected_gpu_uuids=['GPU-123'])
        cp=jobdir/'execution_context.json'; cp.write_text(json.dumps(context))
        p=dict(host='test',visible_device='0',gpu_uuid='GPU-123',broker='remote',job_id='shard-job',root=str(base/'task'),broker_project_root=str(base))
        original=Path.read_text
        def read_text(path,*a,**kw):
            if str(path)=='/proc/sys/kernel/random/boot_id': return 'boot\n'
            if str(path) in ('/proc/self/cgroup','/proc/100/cgroup'): return '0::/ssh-session\n'
            return original(path,*a,**kw)
        with patch.object(s.socket,'gethostname',return_value='test'),patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0',CLAUDE_REMOTE_JOB_DIR=str(jobdir)),patch.object(Path,'read_text',read_text),patch.object(s.os,'getpid',return_value=102),patch.object(s,'process_info',side_effect=lambda pid:{100:(1,123),102:(101,999),101:(100,456)}[pid]):
            proof=s.admission(p); self.assertEqual(proof['supervisor_context'],context)
            context['startticks']=124; cp.write_text(json.dumps(context))
            with self.assertRaisesRegex(ValueError,'process identity'): s.admission(p)
            context['startticks']=123; context['selected_gpu_uuids']=['GPU-other']; cp.write_text(json.dumps(context))
            with self.assertRaisesRegex(ValueError,'context'): s.admission(p)

    def test_fixture_uses_actual_nested_hardware_schema(self):
        f=self.remote/'one'; f.write_text('artifact'); item=s.identity(f)
        models=[s.identity(self.remote/w[0]) for w in self.bridge.WEIGHTS.values()]
        assets=self.remote/'assets'; assets.write_text('{}')
        records=self.remote/'records.json'; records.write_text(json.dumps(dict(schema='static_sr_wu_train_records_v1',heldout_cameras=['cam00'],manifest=s.identity(self.manifest))))
        r=dict(schema='static_sr_wu_teacher_acceptance_fixture_v1',status='fixture_passed',method='stablesr',gpu_executed=True,
            teacher_generated=True,hardware=dict(gpu_uuid='abcdef'),hostname='test',upstream_commit='pinned',wrapper=item,
            registered_records=s.identity(records),adapter_sources=[s.identity(self.root_bridge)],model_identities=models,
            runtime_assets_registration=dict(s.identity(assets),bytes=2),outputs=[dict(item,size_wh=[1344,1008])])
        receipt=self.remote/'fixture.json'; receipt.write_text(json.dumps(r))
        p=dict(gpu_uuid='GPU-abcdef',host='test',root=str(self.root),runtime_assets_identity=s.identity(assets),manifest=s.identity(self.manifest))
        s.gate(receipt,p,self.bridge)
        r['gpu_executed']=False; receipt.write_text(json.dumps(r))
        with self.assertRaises(ValueError): s.gate(receipt,p,self.bridge)


if __name__ == '__main__': unittest.main()
