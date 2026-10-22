"""CPU-only relocation and non-overwrite contract checks; no CUDA or network."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('cts_bootstrap_tested',Path(__file__).with_name('bootstrap_remote.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class BootstrapContracts(unittest.TestCase):
 def test_equal_retry_and_conflict_preserves_original(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'data/ref';self.assertTrue(b.write_exact(p,'abc'));self.assertFalse(b.write_exact(p,'abc'))
   with self.assertRaises(FileExistsError):b.write_exact(p,'abc\n')
   self.assertEqual(p.read_text(),'abc')
 def test_alias_verifies_hash_and_rejects_collision(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);src=root/'source';dst=root/'alias';src.write_bytes(b'official');b.alias_file(src,dst,b.sha(src));self.assertEqual(src.read_bytes(),dst.read_bytes())
   bad=root/'bad';bad.write_bytes(b'other')
   with self.assertRaises(FileExistsError):b.alias_file(src,bad,b.sha(src))
   self.assertEqual(bad.read_bytes(),b'other')
 def test_overlay_pth_uses_destination_and_preserves_packages(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);runtime=root/'runtime';env=runtime/'stablesr_env';site=env/'lib/python3.10/site-packages';site.mkdir(parents=True);(site/'sitecustomize.py').write_text('# exact alias\n');(env/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
   with patch.object(b,'ROOT',root),patch.object(b,'RUNTIME',runtime):b.overlay('/unused/base/python',{'purelib':'/remote/base/lib/python3.10/site-packages'},'stablesr')
   self.assertEqual((site/'reuse_wu_torch.pth').read_text(),'/remote/base/lib/python3.10/site-packages\n')
   self.assertIn("run_name='static_sr_torchvision_compat'",(site/'static_sr_torchvision_compat.pth').read_text())
   self.assertIn(str(site/'sitecustomize.py'),(site/'static_sr_torchvision_compat.pth').read_text())
   self.assertEqual((site/'taming_pinned_namespace.pth').read_text(),str(root/'tmp/static_sr_wu_20261022/upstream/taming-transformers')+'\n')
   self.assertEqual((site/'sitecustomize.py').read_text(),'# exact alias\n')
 def test_assets_keep_original_provenance_and_exact_ref(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);weight=root/'weight/static_sr_wu_20261022/stablesr';out=root/'out';out.mkdir()
   paths={'openclip':weight/'open_clip_pytorch_model.bin','vgg16':weight/'runtime_cache/torch/hub/checkpoints/vgg16-397923af.pth','lpips':weight/'runtime_cache/lpips/vgg.pth'}
   for role,p in paths.items():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(role.encode())
   checks={role:b.sha(p) for role,p in paths.items()};old={role:{'path':'/original/'+role,'sha256':checks[role]} for role in checks}
   template={'files':[dict(old[r],role=r) for r in checks],'cache_mappings':[dict(old[r],role=r,original_source=dict(old[r]),storage='hardlink') for r in checks]};source=root/'original.json';source.write_text(json.dumps(template));before=source.read_bytes()
   fake=SimpleNamespace(RUNTIME_SHA256=checks,OPENCLIP_CACHE_REPO='repo',OPENCLIP_REVISION='0123456',validate_runtime_assets=lambda x:None)
   with patch.object(b,'ROOT',root),patch.object(b,'load_bridge',return_value=fake):dest=b.runtime_assets(source,out)
   value=json.loads(dest.read_text());self.assertEqual(source.read_bytes(),before)
   self.assertEqual((weight/'cts_runtime_cache/hf/hub/repo/refs/main').read_text(),'0123456')
   self.assertEqual({x['role']:x['original_source'] for x in value['cache_mappings']},old)
   self.assertTrue(all(str(root) in x['path'] for x in value['files']))
if __name__=='__main__':unittest.main()
