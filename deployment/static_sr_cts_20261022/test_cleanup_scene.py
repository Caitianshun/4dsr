"""Destructive branches operate exclusively on disposable synthetic temp files."""
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('cleanup_scene_tested',Path(__file__).with_name('cleanup_scene.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


class CleanupContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.top=Path(self.tmp.name)
        self.root=self.top/'4DSR/tasks/static_sr_wu_20261022';self.root.mkdir(parents=True)
        self.root_patch=patch.object(c,'ROOT',self.root);self.root_patch.start()
        self.machine=patch.object(c,'machine_id',return_value='b'*64);self.machine.start()
        self.host=patch.object(c.socket,'gethostname',return_value='unit-remote');self.host.start()
        self.proc=patch.object(c,'find_writers',return_value=[]);self.proc.start()
        shared=self.root.parent.parent/'deployment/cts_20260921';shared.mkdir(parents=True)
        for name in ('project_access.lock','gpu0.lock'):(shared/name).touch()
        self.runrel='output/static_sr_wu_20261022/cts_scene_runs/Discussion07/unit_only'
        self.run=self.root/self.runrel;self.run.mkdir(parents=True)
        self.planpath=self.root/'plans/plan.json'
        self.plan=dict(schema='static_sr_scene_execution_plan_v1',status='executable_after_resource_admission',root=str(self.root),
            scene='Discussion07',scene_name='meetroom_discussion',run_id='unit_only',run_root=self.runrel,
            gpu_uuid='GPU-unit',train_cameras=['cam01'],configs={},final_outputs={},cleanup={'allowlist':[]})
        candidates=[];checkpoint=dict(path=self.runrel+'/unique_checkpoint.pt',sha256='a'*64)
        values=dict(PSNR=20.,SSIM=.8,LPIPS=.2)
        for method in c.METHODS:
            p=self.runrel+'/'+method;directory=p+('/coupled' if method=='threedsr' else '/sr')
            configpath=self.root/'configs'/f'{method}.json';save(configpath,dict(method=method,scene='meetroom_discussion'))
            self.plan['configs'][method]=dict(path=str(configpath.relative_to(self.root)),sha256=c.sha(configpath))
            self.plan['final_outputs'][method]=directory
            completion=dict(status='completed_with_evaluation',wiring_fixture=False,config=self.plan['configs'][method],steps=30000,checkpoint=checkpoint,
                rounds=[dict(round=i,checkpoint=checkpoint) for i in range(4)])
            metrics=dict(schema='static_sr_wu_uniform_png_metrics_v1',scene='meetroom_discussion',camera='cam00',checkpoint=checkpoint,
                metrics=values,frames=[dict(frame=i,**values) for i in range(300)])
            save(self.root/directory/'complete.json',completion);save(self.root/directory/'evaluation/metrics.json',metrics)
            if method=='splatsure':candidates += [p+'/teacher/generated',p+'/contribution/maps',p+'/lr/checkpoint.pt']
            if method=='iesrgs':candidates += [p+'/internal_train/checkpoint.pt',p+'/external/cam01',p+'/internal_cache/cam01']
            if method=='sequence_matters':candidates += [p+'/teacher/cam01']
            if method!='threedsr':candidates += [p+'/sr/checkpoint.pt',p+'/sr/evaluation/renders']
            else:
                candidates += [p+'/lr/checkpoint.pt',p+'/coupled/latents',p+'/coupled/evaluation/renders']
                for i in range(4):
                    candidates += [p+f'/coupled/teacher_{i}/cam01',p+f'/coupled/reconstruction_{i}/checkpoint.pt']
                    if i<3:candidates += [p+f'/coupled/reconstruction_{i}/feedback']
        self.plan['cleanup']['allowlist']=candidates;save(self.planpath,self.plan)
        for value in candidates:
            path=self.root/value
            if path.suffix=='.pt':path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'unit checkpoint')
            else:path.mkdir(parents=True,exist_ok=True);(path/'unit.npy').write_bytes(b'unit array')
        job=dict(job_id='unit-job',scene='Discussion07',out=self.runrel+'/job_logs',gpu_uuid='GPU-unit',
            required_files=[dict(path='plans/plan.json',sha256=c.sha(self.planpath))],commands=[dict(name='collect_compact_results')])
        save(self.run/'job_logs/job.json',job)
        completed=dict(schema='static_sr_cts_execution_v1',status='completed',scene='Discussion07',job_id='unit-job',
            job_sha256=c.sha(self.run/'job_logs/job.json'),stages=[dict(name='collect_compact_results',returncode=0)])
        save(self.run/'job_logs/status.json',completed);save(self.run/'job_logs/complete.json',completed)
        self.broker_path=self.root.parent.parent/'.claude-training/jobs/unit-job/status.json'
        save(self.broker_path,dict(job_id='unit-job',state='ended',outcome='success',
            task_identity=dict(pid=999999999,startticks=1,boot_id='unit-only-different-boot')))
        self.compact=self.run/'compact';self.compact.mkdir();files=[]
        sources=[self.planpath,*[self.root/x['path'] for x in self.plan['configs'].values()]]
        sources += [self.root/d/name for d in self.plan['final_outputs'].values() for name in ('complete.json','evaluation/metrics.json')]
        for source in sources:
            original=str(source.relative_to(self.root));target=self.compact/'records'/original
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            files.append(dict(path=str(target.relative_to(self.compact)),bytes=target.stat().st_size,sha256=c.sha(target),original_path=original))
        save(self.compact/'index.json',dict(schema='static_sr_scene_compact_results_v1',status='remote_collected_not_return_verified',
            scene='Discussion07',run_id='unit_only',files=files,bytes=sum(x['bytes'] for x in files)))
        self.local=self.top/'local_return/compact';shutil.copytree(self.compact,self.local)
        local_jobs=self.top/'local_return/job';shutil.copytree(self.run/'job_logs',local_jobs)
        shutil.copyfile(self.broker_path,local_jobs/'broker_status.json')
        self.returned=self.root/'receipts/local_return.json'
        args=SimpleNamespace(compact=str(self.local),expected_index_sha256=c.sha(self.compact/'index.json'),
            plan=str(self.local/'records/plans/plan.json'),job=str(local_jobs/'job.json'),job_status=str(local_jobs/'status.json'),
            job_complete=str(local_jobs/'complete.json'),broker_status=str(local_jobs/'broker_status.json'),out=str(self.top/'local_return/verified.json'))
        self.verify_args=args
        with patch.object(c,'machine_id',return_value='a'*64),patch.object(c.socket,'gethostname',return_value='unit-local'):
            c.verify_return(args)
        self.returned.parent.mkdir(parents=True);shutil.copyfile(args.out,self.returned)
        self.freeze=self.run/'cleanup_records/freeze.json'

    def tearDown(self):
        self.proc.stop();self.host.stop();self.machine.stop();self.root_patch.stop();self.tmp.cleanup()

    def args(self,apply=False,out=None):
        return SimpleNamespace(plan=str(self.planpath),return_receipt=str(self.returned),apply=apply,
            freeze=str(self.freeze) if apply else None,out=str(out or (self.run/'cleanup_records/applied.json' if apply else self.freeze)))

    def test_end_to_end_dry_run_then_apply_only_synthetic_outputs(self):
        result=c.cleanup(self.args());self.assertEqual(result['status'],'dry_run_frozen_no_deletion')
        self.assertTrue(all((self.root/p).exists() for p in self.plan['cleanup']['allowlist']))
        applied=c.cleanup(self.args(True));self.assertEqual(applied['status'],'applied_verified')
        self.assertEqual(applied['deleted_files'],27)
        self.assertTrue(all(not (self.root/p).exists() for p in self.plan['cleanup']['allowlist']))
        self.assertTrue((self.compact/'index.json').is_file());self.assertTrue((self.run/'job_logs/complete.json').is_file())
        self.assertTrue(all((self.root/d/'evaluation/metrics.json').is_file() for d in self.plan['final_outputs'].values()))

    def test_return_verification_rejects_tampered_local_bytes(self):
        target=self.local/'records/configs/splatsure.json';target.write_bytes(target.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Returned bytes mismatch'):c.verify_return(self.verify_args)

    def test_scope_and_shared_inputs_and_symlinks_are_rejected(self):
        for illegal in ('../data','/tmp/elsewhere',self.runrel+'/splatsure/teacher/inputs','weight/shared.pt',
                        self.runrel.replace('Discussion07','Cook07')+'/splatsure/sr/checkpoint.pt'):
            plan=copy.deepcopy(self.plan);plan['cleanup']['allowlist'][0]=illegal
            with self.assertRaises(ValueError):c.legal_candidates(plan)
        with self.assertRaisesRegex(ValueError,'task snapshot'):c.snapshot_scope(self.plan,self.top/'4DSR')
        directory=self.root/self.plan['cleanup']['allowlist'][0]
        (directory/'link.npy').symlink_to(self.root/'configs/splatsure.json')
        with self.assertRaisesRegex(ValueError,'Symlink'):c.cleanup(self.args())
        self.assertFalse(self.freeze.exists())

    def test_hardlinked_shared_file_refused(self):
        candidate=self.root/self.plan['cleanup']['allowlist'][0]/'unit.npy'
        os.link(candidate,self.root/'shared.npy')
        with self.assertRaisesRegex(ValueError,'Shared/hardlinked'):c.cleanup(self.args())
        self.assertTrue(candidate.exists());self.assertTrue((self.root/'shared.npy').exists())

    def test_running_or_failed_job_refused(self):
        path=self.run/'job_logs/status.json';value=json.loads(path.read_text());value['status']='running';save(path,value)
        with self.assertRaisesRegex(ValueError,'running, failed'):c.cleanup(self.args())
        self.assertFalse(self.freeze.exists())

    def test_admission_lock_prevents_cleanup_of_active_job(self):
        lock=self.root.parent.parent/'deployment/cts_20260921/gpu0.lock'
        with lock.open('r+') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):c.cleanup(self.args())
        self.assertFalse(self.freeze.exists())

    def test_partial_metrics_and_same_host_return_refused(self):
        with patch.object(c,'machine_id',return_value='a'*64):
            with self.assertRaisesRegex(ValueError,'execution host'):c.cleanup(self.args())
        path=self.root/self.plan['final_outputs']['splatsure']/'evaluation/metrics.json'
        value=json.loads(path.read_text());value['frames'].pop();save(path,value)
        with self.assertRaisesRegex(ValueError,'Incomplete/foreign evaluation'):c.cleanup(self.args())

    def test_replaced_inode_with_identical_bytes_refused_before_any_delete(self):
        c.cleanup(self.args());candidate=self.root/self.plan['cleanup']['allowlist'][0]/'unit.npy'
        old=candidate.stat();data=candidate.read_bytes();replacement=candidate.with_suffix('.new');replacement.write_bytes(data)
        os.utime(replacement,ns=(old.st_atime_ns,old.st_mtime_ns));os.replace(replacement,candidate)
        with self.assertRaisesRegex(ValueError,'Frozen identity changed'):c.cleanup(self.args(True))
        self.assertTrue(all((self.root/p).exists() for p in self.plan['cleanup']['allowlist']))
        self.assertFalse(list((self.run/'cleanup_records').glob('quarantine_*')))

    def test_content_change_with_restored_mtime_refused(self):
        c.cleanup(self.args());candidate=self.root/self.plan['cleanup']['allowlist'][0]/'unit.npy';old=candidate.stat()
        candidate.write_bytes(b'other bytes');os.utime(candidate,ns=(old.st_atime_ns,old.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'Frozen identity changed'):c.cleanup(self.args(True))
        self.assertTrue(candidate.exists())

    def test_rename_race_preserves_unexpected_object_in_quarantine(self):
        c.cleanup(self.args());original=c.rename_noreplace;called=[]
        def inject(source,destination,source_fd,destination_fd):
            if not called:
                called.append(True)
                fd=os.open(source,c.DIR_FLAGS,dir_fd=source_fd)
                try:
                    leaf=os.open('unit.npy',os.O_WRONLY|os.O_TRUNC,dir_fd=fd)
                    os.write(leaf,b'raced contents');os.close(leaf)
                finally:os.close(fd)
            return original(source,destination,source_fd,destination_fd)
        with patch.object(c,'rename_noreplace',side_effect=inject):
            with self.assertRaisesRegex(ValueError,'Post-rename identity'):c.cleanup(self.args(True))
        receipt=json.loads((self.run/'cleanup_records/applied.json').read_text())
        self.assertEqual(receipt['status'],'failed_preserve_quarantine');self.assertFalse(receipt['deleted_candidates'])
        quarantined=self.root/receipt['staged'][0]['quarantine']/'unit.npy'
        self.assertEqual(quarantined.read_bytes(),b'raced contents')

    def test_active_writer_check_is_required(self):
        with patch.object(c,'find_writers',return_value=[{'pid':123,'path':'unit'}]):
            with self.assertRaisesRegex(RuntimeError,'Active writer'):c.cleanup(self.args())

    def test_broker_running_or_live_task_refuses_cleanup(self):
        broker=json.loads(self.broker_path.read_text());broker['state']='running';save(self.broker_path,broker)
        with self.assertRaisesRegex(ValueError,'broker has not'):c.cleanup(self.args())
        broker['state']='ended';save(self.broker_path,broker)
        with patch.object(c,'task_still_alive',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'still running'):c.cleanup(self.args())

    def test_npz_allowed_only_in_fixed_ie_camera_directories(self):
        directory=self.run/'iesrgs/external/cam01';(directory/'unit.npy').rename(directory/'unit.npz')
        result=c.cleanup(self.args())
        entry=next(x for x in result['candidates'] if x['path'].endswith('iesrgs/external/cam01'))
        self.assertTrue(entry['allow_npz'])
        c.cleanup(self.args(True));self.assertFalse(directory.exists())

    def test_npz_in_other_method_directory_refused(self):
        directory=self.run/'splatsure/teacher/generated';(directory/'unit.npy').rename(directory/'unit.npz')
        with self.assertRaisesRegex(ValueError,'Unlisted/non-large'):c.cleanup(self.args())


if __name__=='__main__':unittest.main()
