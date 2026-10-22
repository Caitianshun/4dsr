"""CPU/local-filesystem contracts: never SSH, fetch, or CUDA."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


t=module('transport_inventory')
s=module('sync_assets')


class TransportInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.repo=self.root/'tmp/static_sr_wu_20261022/upstream/repo';self.repo.mkdir(parents=True)
        self.git('init','-q')

    def git(self,*args):
        return subprocess.run(['git','-C',str(self.repo),*args],check=True,capture_output=True,text=True)

    def file(self,name,tracked=False):
        path=self.repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'author file\0')
        if tracked:self.git('add','-f','--',name)
        return path

    def test_tracked_pyc_build_binary_lock_and_pth_override_filters(self):
        names=['pkg/__pycache__/author.pyc','build/lib/author.so','dist/model.bin',
               'author.lock','reuse_wu_torch.pth','taming_pinned_namespace.pth','source.py']
        for name in names:self.file(name,True)
        excluded=['pkg/__pycache__/generated.pyc','build/generated.so','dist/generated.bin',
                  'generated.pyc','generated.lock']
        for name in excluded:self.file(name)
        ordinary=self.file('untracked_notes.txt')
        paths,tracked,empty=t.inventory_paths(self.repo,True)
        self.assertEqual(tracked,{self.repo/name for name in names})
        self.assertTrue(tracked<=set(paths));self.assertIn(ordinary,paths)
        self.assertTrue(all(self.repo/name not in paths for name in excluded));self.assertEqual(empty,[])
        self.assertIn(self.repo/'.git/index',paths)

    def test_overlay_filters_remain_unchanged(self):
        for name in ('x.py','x.pyc','build/lib.so','reuse_wu_torch.pth'):self.file(name,True)
        paths,tracked,empty=t.inventory_paths(self.repo,False)
        self.assertIn(self.repo/'x.py',paths)
        self.assertTrue(all(self.repo/name not in paths for name in ('x.pyc','build/lib.so','reuse_wu_torch.pth')))
        self.assertEqual(tracked,set());self.assertEqual(empty,[])

    def test_missing_tracked_file_is_not_silently_omitted(self):
        path=self.file('build/required.so',True);path.unlink()
        with self.assertRaisesRegex(FileNotFoundError,'Git-tracked'):t.inventory_paths(self.repo,True)

    def test_uninitialized_gitlink_is_preserved_as_empty_directory_with_commit(self):
        commit='a'*40;path=self.repo/'submodules/author';path.mkdir(parents=True)
        self.git('update-index','--add','--cacheinfo','160000,'+commit+',submodules/author')
        paths,tracked,empty=t.inventory_paths(self.repo,True)
        self.assertEqual(empty,[dict(path=path,gitlink_commit=commit)])
        self.assertNotIn(path,paths);self.assertNotIn(path,tracked)
        self.assertEqual(list(path.iterdir()),[])

    def test_main_serializes_every_tracked_asset_and_empty_gitlink(self):
        binary=self.file('build/author.so',True)
        link=self.repo/'submodules/unused';link.mkdir(parents=True)
        self.git('update-index','--add','--cacheinfo','160000,'+'a'*40+',submodules/unused')
        out=self.root/'inventory'
        with patch.object(t,'ROOT',self.root),patch.object(t,'REPOS',('repo',)),patch.object(sys,'argv',
                ['transport_inventory','--stage','upstream','--out',str(out),'--remote-root','/test/snapshot']):
            t.main()
        result=json.loads((out/'inventory.json').read_text())
        entry=next(v for v in result['files'] if v['path']==str(binary.relative_to(self.root)))
        self.assertTrue(entry['git_tracked']);self.assertEqual(entry['sha256'],t.sha(binary))
        self.assertEqual(result['empty_dirs'],[dict(path=str(link.relative_to(self.root)),
            gitlink_commit='a'*40,destination='/test/snapshot/'+str(link.relative_to(self.root)))])
        self.assertEqual(result['transport_blockers'],[])
        self.assertIn(entry['path'],(out/'files.txt').read_text().splitlines())

    def test_tracked_symlink_is_listed_even_under_filtered_directory(self):
        target=self.file('source.py',True);link=self.repo/'build/source_link';link.parent.mkdir()
        link.symlink_to('../source.py');self.git('add','-f','build/source_link')
        paths,tracked,_=t.inventory_paths(self.repo,True)
        self.assertIn(link,paths);self.assertIn(link,tracked);self.assertTrue(link.is_symlink())

    def test_sync_rejects_symlink_inventory_before_contact(self):
        inventory=self.root/'inventory.json';inventory.write_text(json.dumps({'symlinks':[{'path':'link','target':'x'}],'files':[]}))
        with patch.object(sys,'argv',['sync_assets','--inventory',str(inventory),'--out',str(self.root/'out')]),patch.object(s,'remote') as remote:
            with self.assertRaisesRegex(ValueError,'Symlink inventories'):s.main()
            remote.assert_not_called()

    def remote_fixture(self,phase,entry):
        # Execute the literal remote verifier on isolated local temp dirs. This
        # exercises mkdir/conflict/verification without opening SSH or a socket.
        request=dict(root=str(self.root/'destination'),shared=str(self.root/'shared'),
                     files=[],empty_dirs=[entry],reserve_bytes=0,phase=phase)
        return subprocess.run([sys.executable,'-c',s.REMOTE_CODE],input=json.dumps(request),
                              capture_output=True,text=True)

    def test_sync_only_mkdir_empty_gitlinks_then_verifies_commit_record(self):
        entry=dict(path='upstream/submodules/author',gitlink_commit='a'*40)
        first=self.remote_fixture('prepare',entry)
        self.assertEqual(first.returncode,0,first.stderr)
        folder=self.root/'destination'/entry['path'];self.assertTrue(folder.is_dir());self.assertEqual(list(folder.iterdir()),[])
        result=json.loads(first.stdout);self.assertEqual(result['created_empty_dirs'],[entry['path']])
        self.assertEqual(result['empty_gitlinks'],[entry])
        verify=self.remote_fixture('verify',entry);self.assertEqual(verify.returncode,0,verify.stderr)
        (folder/'foreign').write_text('preserve')
        conflict=self.remote_fixture('prepare',entry)
        self.assertNotEqual(conflict.returncode,0);self.assertIn('conflicts',conflict.stderr)
        self.assertEqual((folder/'foreign').read_text(),'preserve')

    def test_sync_verify_does_not_create_a_missing_gitlink(self):
        entry=dict(path='upstream/submodules/author',gitlink_commit='a'*40)
        result=self.remote_fixture('verify',entry)
        self.assertNotEqual(result.returncode,0);self.assertIn('Incomplete deployment',result.stderr)
        self.assertFalse((self.root/'destination'/entry['path']).exists())


if __name__=='__main__':unittest.main()
