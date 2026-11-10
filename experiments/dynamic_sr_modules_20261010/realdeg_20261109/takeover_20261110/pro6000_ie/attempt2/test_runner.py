#!/usr/bin/python3
"""CPU-only contract tests; no real service, CUDA, approval or scientific output."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pro_runner_under_test',HERE / 'runner.py')
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2)+'\n')
    p.chmod(0o644)


class ContextFixture:
    def __init__(self,d):
        self.base = Path(d)
        self.plan = V.read(V.PLAN)
        self.expected = self.plan['preflight_manifest']
        self.job = self.base / self.expected['job_id']
        self.frozen = self.job / 'manifest.json'
        save(self.frozen,dict(self.expected,root_added_guard=True))
        self.digest()
        self.current = os.getpid()
        cg = '/claudevpn.slice/claude-local-' + self.expected['job_id'] + '.service'
        self.runner = dict(pid=100,start_ticks=1234,cgroup='0::'+cg,netns='net:[123]')
        self.child = dict(pid=self.current,start_ticks=5678,cgroup='0::'+cg,netns='net:[123]')
        self.status = dict(job_id=self.expected['job_id'],project_id='4dsr',gpu_uuids=[V.GPU],
            state='running',command_started=True,runner=self.runner,child=self.child)
        self.identities = {100:dict(pid=100,ppid=1,state='S',start_ticks=1234,cgroup=cg,netns='net:[123]'),
            self.current:dict(pid=self.current,ppid=100,state='R',start_ticks=5678,cgroup=cg,netns='net:[123]')}
        self.write_status()

    def digest(self):
        save(self.job / 'submission.json',dict(manifest_sha256=V.sha(self.frozen)))

    def write_status(self):
        save(self.job / 'run/status.json',self.status)

    def call(self):
        real_lstat = Path.lstat
        def frozen_root_metadata(p,*args,**kwargs):
            info = real_lstat(p,*args,**kwargs)
            if p == self.frozen:
                fields = list(info);fields[4] = 0
                return os.stat_result(fields)
            return info
        with patch.object(Path,'lstat',frozen_root_metadata),patch.object(V.sys,'executable',V.PYTHON),\
             patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':V.GPU}),\
             patch.object(V,'proc_identity',side_effect=lambda pid,proc:self.identities[pid]):
            return V.execution_context(self.plan,self.expected,service_base=self.base)


class Tests(unittest.TestCase):
    def test_cpu_plan_and_only_eight_original_IE_scope(self):
        plan = V.read(V.PLAN)
        V.validate_plan(plan)
        self.assertNotIn('torch',sys.modules)
        for kind in ('gpu','environment','reorder','scientific_budget','service_argv','source_identity'):
            changed = copy.deepcopy(plan)
            if kind == 'gpu':changed['gpu_uuid'] = 'foreign'
            elif kind == 'environment':changed['env_id'] = 'sr4d'
            elif kind == 'reorder':changed['jobs'].reverse()
            elif kind == 'scientific_budget':changed['jobs'][0]['service_manifest']['timeout_seconds'] += 1
            elif kind == 'service_argv':changed['jobs'][0]['service_manifest']['argv'].append('--skip-checks')
            else:changed['jobs'][0]['original_request_value']['env_id'] = 'mml'
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):V.validate_plan(changed)

    def test_normalized_root_manifest_and_live_ancestor_identity(self):
        with tempfile.TemporaryDirectory() as d:
            f = ContextFixture(d)
            result = f.call()
            self.assertEqual(result['job_id'],'rd4-p6-ie-preflight-02')
            self.assertNotEqual(V.sha(f.frozen),V.sha(V.PLAN))

    def test_reject_changed_root_bytes_stale_or_foreign_service_context(self):
        for kind in ('bytes','gpu','source_field','runner_ticks','child_pid','current_netns','not_running','nonancestor'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                f = ContextFixture(d)
                if kind == 'bytes':
                    with f.frozen.open('a') as stream:stream.write('\n')
                elif kind in ('gpu','source_field'):
                    manifest = V.read(f.frozen)
                    if kind == 'gpu':manifest['gpu_uuids'] = ['foreign']
                    else:manifest['timeout_seconds'] += 1
                    save(f.frozen,manifest);f.digest()
                elif kind == 'runner_ticks':f.status['runner']['start_ticks'] += 1
                elif kind == 'child_pid':f.status['child']['pid'] += 1
                elif kind == 'current_netns':f.identities[f.current]['netns'] = 'net:[foreign]'
                elif kind == 'not_running':f.status['state'] = 'pending'
                else:f.identities[f.current]['ppid'] = 1
                f.write_status()
                with self.assertRaises(RuntimeError):f.call()

    def test_ownership_activation_cannot_be_omitted(self):
        plan = V.read(V.PLAN)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):V.authorize(plan,plan['jobs'][0],Path(d))

    def test_sandbox_only_own_job_visible_uses_exact_project_preflight_mirrors(self):
        plan = V.read(V.PLAN)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            here = root / Path(plan['runner_path']).parent
            runner = here / 'runner.py'
            runner.parent.mkdir(parents=True)
            runner.write_text('# fixture frozen runner\n')
            prepared = here / 'prepared_plan.json'
            save(prepared,plan)
            def reference(p):return dict(path=str(p),sha256=V.sha(p))
            resource = root / plan['resource_gate_reassignment']['path']
            resource.parent.mkdir(parents=True,exist_ok=True)
            resource.write_bytes((V.ROOT/plan['resource_gate_reassignment']['path']).read_bytes())
            resource_ref = reference(resource)
            jobs = []
            for row in plan['jobs']:
                manifest = root / 'manifests' / (row['service_job_id']+'.json')
                save(manifest,row['service_manifest'])
                jobs.append(dict(original_job_id=row['original_job_id'],service_job_id=row['service_job_id'],
                    original_request=row['original_request'],original_lane=row['lane'],config=row['config'],
                    service_manifest=reference(manifest)))
            delegation_path = root / plan['authorization']['delegation_path']
            delegation = dict(schema='rd4_pro6000_split_owner_delegation_v1',
                runner=dict(path=plan['runner_path'],sha256=V.sha(runner)),
                prepared_plan=dict(path=prepared.relative_to(root).as_posix(),sha256=V.sha(prepared)),
                jobs=jobs,resource_gate_reassignment=resource_ref)
            save(delegation_path,delegation)
            partition_path = root / plan['authorization']['partition_path']
            save(partition_path,dict(resource_gate_reassignment=resource_ref))
            proofs = {}
            expected = plan['preflight_manifest']
            mirrors = root / plan['authorization']['preflight_mirror_directory']
            root_manifest = mirrors / 'root_manifest.json'
            save(root_manifest,dict(expected,root_added_guard=True))
            values = dict(manifest=expected,root_manifest=V.read(root_manifest),
                root_submission=dict(manifest_sha256=V.sha(root_manifest)),
                actual_status=dict(job_id=expected['job_id'],state='ended',reason='completed',exit_code=0),
                native_report=dict(status='passed',optimizer_steps=0,runner_sha256=V.sha(runner),
                    prepared_plan_sha256=V.sha(prepared),fixed_context=dict(root_manifest_sha256=V.sha(root_manifest))))
            for key,value in values.items():
                p = mirrors / (key+'.json');save(p,value);proofs[key]=reference(p)
            activation = dict(schema='rd4_pro6000_split_activation_v2',status='active',
                old_v6_waiting_controller_stopped=True,original_jobs_empty=True,local_mode_excludes_ie=True,
                pro_mode_owns_original_ids=list(V.IDS),delegation_sha256=V.sha(delegation_path),
                partition_sha256=V.sha(partition_path),original_plan_sha256=plan['original_owner_plan']['sha256'],
                v6_controller_sha256=plan['predecessor_controller']['sha256'],resource_gate_reassignment=resource_ref,
                preflight={k:dict(path="/var/lib/claude-training/jobs/only-own-job-hidden-prior/"+k+".json",sha256=v["sha256"]) for k,v in proofs.items()},gpu_visible_preflight=proofs)
            activation_path = root / plan['authorization']['activation_path']
            save(activation_path,activation)
            actual_read, actual_sha = V.read, V.sha
            def sandbox_read(p):
                if Path(p).is_relative_to('/var/lib/claude-training/jobs/only-own-job-hidden-prior'):
                    raise FileNotFoundError('Prior fixed-service job hidden inside GPU unit')
                return actual_read(p)
            def sandbox_sha(p):
                if Path(p).is_relative_to('/var/lib/claude-training/jobs/only-own-job-hidden-prior'):
                    raise FileNotFoundError('Prior fixed-service job hidden inside GPU unit')
                return actual_sha(p)
            with patch.object(V,'ROOT',root),patch.object(V,'HERE',here),patch.object(V,'PLAN',prepared),\
                 patch.object(V,'read',side_effect=sandbox_read),patch.object(V,'sha',side_effect=sandbox_sha):
                V.authorize(plan,plan['jobs'][0],root)
                for key,value in [('old_v6_waiting_controller_stopped',False),('original_jobs_empty',False),
                                  ('local_mode_excludes_ie',False),('pro_mode_owns_original_ids',list(V.IDS[:-1]))]:
                    changed = copy.deepcopy(activation);changed[key]=value;save(activation_path,changed)
                    with self.subTest(key=key),self.assertRaises(RuntimeError):V.authorize(plan,plan['jobs'][0],root)
                save(activation_path,activation)
                corrupted = copy.deepcopy(activation)
                corrupted['preflight']['root_submission']['sha256'] = '0'*64
                save(activation_path,corrupted)
                with self.assertRaisesRegex(RuntimeError,'mirror or root binding'):
                    V.authorize(plan,plan['jobs'][0],root)
                save(activation_path,activation)
                runner.write_text('# changed after native preflight\n')
                with self.assertRaisesRegex(RuntimeError,'final runner/plan'):V.authorize(plan,plan['jobs'][0],root)

    def test_host_adapter_changes_only_in_memory_GPU_and_preserves_original_lane(self):
        plan = V.read(V.PLAN)
        row = plan['jobs'][0]
        old_argv = sys.argv[:]
        observed = []
        module = SimpleNamespace(LOCAL_GPU=plan['original_gpu_uuid'])
        module.main = lambda:observed.append((module.LOCAL_GPU,sys.argv[:]))
        with tempfile.TemporaryDirectory() as d,patch.object(V,'ROOT',Path(d)),\
             patch.object(V,'load',return_value=module):
            try:
                V.execute_lane(plan,row)
                self.assertEqual(observed,[(V.GPU,[str(V.BASE/'run_lane.py'),'--lane',row['lane']['path'],'--execute'])])
                (Path(d)/row['output_root']).mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError,'never adopt'):V.execute_lane(plan,row)
            finally:sys.argv = old_argv


if __name__ == '__main__':
    unittest.main(verbosity=2)
