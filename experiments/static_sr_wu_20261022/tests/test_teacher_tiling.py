"""CPU shape contracts using both pinned author splitters, never learned inference."""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from methods import splatsure_stablesr as stable, threedsr
from methods.sequence_matters import source_objects, sha256


class TeacherTileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.sources = {
            'stablesr': ROOT/'tmp/static_sr_wu_20261022/upstream/stablesr/scripts/util_image.py',
            'threedsr': ROOT/'tmp/static_sr_wu_20261022/upstream/threedsr/utils/util_image.py',
        }
        cls.splitters = {key: source_objects(path, ['ImageSpliterTh'], {'torch': torch})['ImageSpliterTh']
                         for key, path in cls.sources.items()}

    def author_stablesr_pad(self, working_shape):
        """Execute the real author's isolated pad AST, with CPU tensors only."""
        import torch.nn.functional as F
        path = stable.DEFAULT_SOURCE/stable.SCRIPT
        nodes = [node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.If)
                 and isinstance(node.test, ast.UnaryOp) and 'ori_h % 32' in ast.unparse(node.test)]
        self.assertEqual(len(nodes), 1)
        namespace = dict(F=F, im_lq_bs=self.torch.ones(1, 3, *working_shape),
                         ori_h=working_shape[0], ori_w=working_shape[1])
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        return namespace['im_lq_bs']

    def test_default_author_splitter_reproduces_both_domain_failures(self):
        # Source default genuinely fails: the short image axis truncates a
        # supposedly square patch, but the Gaussian is still1280x1280.
        for shape in ([336, 252], [320, 180]):
            geometry = stable.stablesr_geometry(shape)
            image = self.author_stablesr_pad(geometry['working_shape_hw'])
            self.assertEqual(list(image.shape[-2:]), geometry['padded_shape_hw'])
            splitter = self.splitters['stablesr'](image, 1280, 1000)
            patch, region = next(splitter)
            self.assertNotEqual(patch.shape[-2], 1280)
            with self.assertRaises(RuntimeError):
                splitter.update_gaussian(patch, region)

    def test_actual_author_splitters_cover_cook_discussion_with_square_tiles(self):
        torch = self.torch
        cases = [([336, 252], [1024, 1376], 1024, 992),
                 ([320, 180], [736, 1312], 736, 704)]
        for shape, padded, tile, stride in cases:
            geometry = stable.stablesr_geometry(shape)
            self.assertEqual((geometry['padded_shape_hw'], geometry['tile'], geometry['stride']),
                             (padded, tile, stride))
            self.assertGreaterEqual(tile, 512)
            self.assertLess(stride, tile)
            image = self.author_stablesr_pad(geometry['working_shape_hw'])
            # A spatially varying image makes the stitching check meaningful.
            image *= torch.linspace(.1, .9, padded[1]).view(1, 1, 1, -1)
            for name, splitter_type in self.splitters.items():
                with self.subTest(shape=shape, source=name):
                    splitter = splitter_type(image, tile, stride)
                    count = 0
                    for patch, region in splitter:
                        self.assertEqual(tuple(patch.shape), (1, 3, tile, tile))
                        self.assertEqual(tuple(splitter.weight.shape), tuple(patch.shape))
                        splitter.update_gaussian(patch, region)
                        count += 1
                    result = splitter.gather()
                    self.assertEqual(count, 2)
                    self.assertTrue(bool(torch.isfinite(result).all()))
                    self.assertTrue(bool((splitter.pixel_count > 0).all()))
                    torch.testing.assert_close(result, image)
                    h, w = geometry['working_shape_hw']
                    self.assertEqual(list(result[:, :, :h, :w].shape[-2:]), [shape[1]*4, shape[0]*4])

    def test_command_keeps_inner_512_ddpm200_and_binds_geometry(self):
        for shape, tile, stride in [([336, 252], '1024', '992'), ([320, 180], '736', '704')]:
            plan = dict(python='/python', source_root='/source', output_root='/new', seed=42,
                        input_shape_wh=shape, tiling=stable.stablesr_geometry(shape),
                        model_identities=[{'role': k, 'path': '/weights/'+k} for k in ('diffusion', 'cfw')])
            args = stable.command(plan)
            for flag, expected in [('--vqgantile_size', tile), ('--vqgantile_stride', stride),
                                   ('--input_size', '512'), ('--tile_overlap', '32'),
                                   ('--ddpm_steps', '200'), ('--upscale', '4')]:
                self.assertEqual(args[args.index(flag)+1], expected)
            plan['tiling']['stride'] = 1
            with self.assertRaisesRegex(ValueError, 'tiling differs'):
                stable.command(plan)

    def test_default_no_chop_boundary_and_large_square_keep_original_settings(self):
        self.assertFalse(stable.outer_tile_geometry([736, 1280])['tiled'])
        self.assertFalse(stable.outer_tile_geometry([1280, 1280])['tiled'])
        for shape in ([736, 1280], [1280, 1280], [1536, 2048]):
            result = stable.outer_tile_geometry(shape)
            self.assertEqual((result['tile'], result['stride']), (1280, 1000))
            self.assertFalse(result['adapted'])

    def test_acceptance_uses_the_production_command_and_records_actual_geometry(self):
        import teacher_acceptance as acceptance
        from PIL import Image
        torch = self.torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source.png'
            Image.new('RGB', (336,252)).save(source)
            fixture = {'records': [dict(key='cam01/0000', lr_path=str(source), lr_sha256=sha256(source))]}
            args = SimpleNamespace(upstream='/official', seed=42, stable_checkpoint='/weights/main',
                                   vqgan_checkpoint='/weights/vq', runtime_assets='/assets')
            module = ModuleType('cpu_author_entry_contract')
            module.load_model_from_config = lambda *a, **k: torch.nn.Linear(1,1)
            module.adaptive_instance_normalization = lambda image, _: image
            def main():
                self.assertEqual(sys.argv[sys.argv.index('--vqgantile_size')+1], '1024')
                self.assertEqual(sys.argv[sys.argv.index('--vqgantile_stride')+1], '992')
                module.load_model_from_config(None, args.stable_checkpoint)
                module.load_model_from_config(None, args.vqgan_checkpoint)
                module.adaptive_instance_normalization(torch.zeros(1), torch.zeros(1))
                generated = root/'teacher/generated'
                generated.mkdir()
                Image.new('RGB', (1344,1008)).save(generated/'fixture.png')
            module.main = main
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
            receipt = {}
            with mock.patch.object(acceptance, 'runtime_context', side_effect=lambda *a: nullcontext({})), \
                 mock.patch.object(acceptance.importlib.util, 'spec_from_file_location', return_value=spec), \
                 mock.patch.object(acceptance.importlib.util, 'module_from_spec', return_value=module), \
                 mock.patch.object(torch.cuda, 'synchronize'):
                acceptance.run_stablesr(args, fixture, root, torch, receipt)
            self.assertEqual(receipt['tiling'], stable.stablesr_geometry([336,252]))
            self.assertEqual(len(receipt['outputs']),1)

    def test_threedsr_author_already_handles_rectangular_default_patch(self):
        torch = self.torch
        image = torch.linspace(.1, .9, 1344).view(1, 1, 1, -1).expand(1, 3, 1024, -1).contiguous()
        splitter = self.splitters['threedsr'](image, 1280, 1000)
        regions = []
        for patch, region in splitter:
            self.assertEqual(tuple(patch.shape[-2:]), (1024, 1280))
            self.assertEqual(tuple(splitter.weight.shape[-2:]), (1280, 1280))
            splitter.update_gaussian(patch, region)
            regions.append(region)
        self.assertEqual(regions, [(0, 1280, 0, 1280), (0, 1280, 64, 1344)])
        torch.testing.assert_close(splitter.gather(), image)

    def test_threedsr_real_constructor_binds_config_for_author_gaussian_weights(self):
        """Run production __init__; replace network allocation, never config use.

        The isolated Gaussian method is the untouched author AST that failed
        on GPU. CPU network stand-ins avoid allocating or evaluating a teacher.
        """
        torch = self.torch
        upstream = self.sources['threedsr'].parents[1]
        author = upstream/'third_parties/ldm/models/diffusion/ddpm.py'
        classes = [node for node in ast.parse(author.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == 'LatentDiffusionSRTextWT']
        self.assertEqual(len(classes), 1)
        methods = [node for node in classes[0].body
                   if isinstance(node, ast.FunctionDef) and node.name == '_gaussian_weights']
        self.assertEqual(len(methods), 1)
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(author), 'exec'), namespace)

        class Config(dict):
            def __getattr__(self, key): return self[key]

        class Main(torch.nn.Module):
            _gaussian_weights = namespace['_gaussian_weights']
            def __init__(self, channels):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(channels))
                self.schedule_calls = []
                self.devices = []
            def to(self, device):
                self.devices.append(device)
                return self
            def register_schedule(self, given_betas, timesteps, **kwargs):
                self.schedule_calls.append((timesteps, kwargs))
                self.betas = (torch.full((timesteps,), .001) if given_betas is None
                              else torch.as_tensor(given_betas).clone())
                self.alphas_cumprod = torch.cumprod(1 - self.betas, 0)
                self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
                self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()

        class VQ(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.decoder = SimpleNamespace(fusion_w=None)
            def to(self, device): return self

        factory = ModuleType('cpu_threedsr_constructor_factory')
        factory.Main, factory.VQ = Main, VQ
        main_config = Config(model=Config(target=factory.__name__+'.Main', params=Config(channels=4)),
                             extra_configuration='must preserve the full config object')
        vq_config = Config(model=Config(target=factory.__name__+'.VQ', params=Config()))
        config_paths = []
        def load_config(path):
            config_paths.append(Path(path).relative_to(upstream).as_posix())
            return main_config if 'stableSRNew' in str(path) else vq_config
        omega = ModuleType('omegaconf')
        omega.OmegaConf = SimpleNamespace(load=load_config)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            main_weight, vq_weight = root/'main.pt', root/'vq.pt'
            torch.save({'state_dict': {'weight': torch.ones(4)}}, main_weight)
            torch.save({'state_dict': {'weight': torch.ones(1)}}, vq_weight)
            with mock.patch.dict(sys.modules, {factory.__name__: factory, 'omegaconf': omega}), \
                 mock.patch.object(sys, 'path', list(sys.path)), \
                 mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
                bridge = threedsr.StableSRTurboBridge(
                    upstream, main_weight, vq_weight, root/'state', device='cuda:0')
                self.assertIs(bridge.model.configs, main_config)
                self.assertFalse(hasattr(bridge.vq_model, 'configs'))
                weights = bridge.model._gaussian_weights(64, 64, 2)
                self.assertEqual(tuple(weights.shape), (2, 4, 64, 64))
                self.assertEqual(weights.device.type, 'cpu')
                self.assertTrue(bool(torch.isfinite(weights).all()))
                # Prove this is the failing real dependency, rather than a
                # stand-in Gaussian that silently assumes four channels.
                del bridge.model.configs
                with self.assertRaisesRegex(AttributeError, 'configs'):
                    bridge.model._gaussian_weights(64, 64, 2)
                bridge.model.configs = main_config
            self.assertEqual(config_paths, ['configs/stableSRNew/v2-finetune_text_T_512.yaml',
                                           'configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml'])
            self.assertEqual([call[0] for call in bridge.model.schedule_calls], [1000, 4])
            self.assertEqual(bridge.timesteps, [0, 333, 666, 999])
            self.assertEqual(bridge.model.ori_timesteps, bridge.timesteps)
            self.assertEqual(bridge.model.num_timesteps, 1000)
            self.assertEqual(bridge.vq_model.decoder.fusion_w, .5)
            self.assertEqual(bridge.model.devices, ['cuda:0', 'cuda:0'])
            self.assertTrue(all(not p.requires_grad for model in (bridge.model, bridge.vq_model)
                                for p in model.parameters()))
            self.assertTrue(all(not entry['missing_keys'] and not entry['unexpected_keys']
                                for entry in bridge.weight_loads))
        self.assertFalse(torch.cuda.is_initialized())

    def fake_bridge(self, root):
        """Keep real preprocessing/state/splitter/IO; replace only learned ops."""
        torch = self.torch
        import torch.nn.functional as F
        class Model:
            scale_factor = 1.
            def ema_scope(self): return nullcontext()
            def encode_first_stage(self, image):
                return F.avg_pool2d(image.mean(1, keepdim=True).repeat(1, 4, 1, 1), 8)
            def get_first_stage_encoding(self, image): return image
            def cond_stage_model(self, text): return text
            def q_sample_respace(self, **kwargs): return kwargs['x_start'].clone()
            def sample_canvas_one_iter(inner, **kwargs):
                self.assertEqual((kwargs['tile_size'], kwargs['tile_overlap'], kwargs['timesteps'],
                                  kwargs['time_replace'], kwargs['batch_size_sample']), (64, 32, 4, 4, 2))
                if kwargs['return_x0']:
                    return kwargs['x_T'], kwargs['struct_cond']
                self.assertEqual(kwargs['x_T'].shape, kwargs['x0_input'].shape)
                return kwargs['x_T'] + .125
        class VQ:
            def encode(self, image): return None, image.shape[-2:]
            def decode(self, latent, shape): return F.interpolate(latent[:, :3], size=shape, mode='nearest')
        bridge = object.__new__(threedsr.StableSRTurboBridge)
        bridge.model, bridge.vq_model = Model(), VQ()
        bridge.state_root, bridge.device, bridge.seed = root/'state', 'cpu', 42
        bridge.weight_loads = [{'sha256': 'CPU_mock_main'}, {'sha256': 'CPU_mock_vq'}]
        bridge.timesteps = [0, 333, 666, 999]
        bridge.original_sqrt_alpha = bridge.original_sqrt_one_minus_alpha = None
        bridge.colorfix = lambda image, _: image
        bridge.splitter = self.splitters['threedsr']
        return bridge

    def test_real_threedsr_bridge_four_round_regions_latent_shapes_and_outputs(self):
        torch = self.torch
        from PIL import Image
        for name, shape, padded, patch_count in [('Cook', (336, 252), [1024, 1344], 2),
                                                  ('Discussion', (320, 180), [736, 1280], 1)]:
            with self.subTest(scene=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                image = root/'lr.png'
                Image.new('RGB', shape, (30, 80, 190)).save(image)
                record = dict(key='cam01/0000', camera='cam01', scene=name, frame=0, time=0.,
                              split='train', lr_path=str(image), lr_sha256=sha256(image))
                manifest = {'records': [record], 'heldout_cameras': ['cam00']}
                bridge = self.fake_bridge(root)
                feedback, previous = None, None
                with mock.patch.object(torch, 'autocast', side_effect=lambda *a, **k: nullcontext()), \
                     mock.patch.object(torch.cuda, 'manual_seed_all'):
                    for index in range(4):
                        output = root/f'round{index}'
                        targets = bridge.generate_targets(manifest, index, output, feedback)
                        state = torch.load(bridge.state_root/'cam01/0000.pt', weights_only=True)
                        geometry = state['tiling']
                        self.assertEqual(geometry['padded_shape_hw'], padded)
                        self.assertEqual((geometry['tile'], geometry['stride']), (1280, 1000))
                        self.assertFalse(geometry['adapted'])
                        self.assertEqual(len(state['patches']), patch_count)
                        self.assertEqual(geometry['tiled'], name == 'Cook')
                        self.assertEqual(state['round_index'], index)
                        for key, patch in state['patches'].items():
                            top, bottom, left, right = patch['region']
                            # Author region can extend beyond the short edge;
                            # the actual image/latent is naturally truncated.
                            self.assertEqual(tuple(patch['x_t'].shape),
                                (1, 4, (min(bottom,padded[0])-top)//8, (min(right,padded[1])-left)//8))
                            self.assertTrue(bool(torch.isfinite(patch['x_t']).all()))
                            if previous:
                                self.assertEqual(patch['region'], previous['patches'][key]['region'])
                                torch.testing.assert_close(patch['x_t'], previous['patches'][key]['x_t']+.125)
                        if previous: self.assertEqual(geometry, previous['tiling'])
                        receipt = json.loads((output/'receipt.json').read_text())
                        self.assertEqual(receipt['records'][0]['tiling'], geometry)
                        with Image.open(targets[record['key']]) as result:
                            self.assertEqual(result.size, (shape[0]*4, shape[1]*4))
                        previous, feedback = state, targets
                    # A changed recorded geometry must fail, even if patch keys
                    # and model hashes still look valid.
                    state['round_index'] = 2
                    state['tiling']['stride'] -= 1
                    torch.save(state, bridge.state_root/'cam01/0000.pt')
                    with self.assertRaisesRegex(ValueError, 'Tiling geometry changed'):
                        bridge.generate_targets(manifest, 3, root/'tampered', feedback)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
