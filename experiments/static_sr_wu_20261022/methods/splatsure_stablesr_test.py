"""CPU-only checks of the official StableSR bridge contract."""
import unittest
from pathlib import Path
import tempfile
from .splatsure_stablesr import command,png_shape,WEIGHTS,RECIPE

class StableSRBridgeTests(unittest.TestCase):
    def test_main_teacher_command_is_standard_stablesr_not_turbo(self):
        value=dict(python='/env/bin/python',source_root='/official/StableSR',output_root='/new/cache',seed=42,input_shape_wh=[336,252],
                   model_identities=[dict(role=k,path='/weights/'+v[0]) for k,v in WEIGHTS.items()])
        args=command(value)
        self.assertEqual(args[args.index('--ddpm_steps')+1],'200')
        self.assertEqual(args[args.index('--ckpt')+1],'/weights/stablesr_000117.ckpt')
        self.assertEqual(args[args.index('--init-img')+1],'/new/cache/inputs')
        self.assertEqual(args[args.index('--upscale')+1],'4')
        self.assertEqual(args[args.index('--colorfix_type')+1],'adain')
        self.assertIn('explicit_adaptation',RECIPE)

    def test_png_grid_validation(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rgb.png'
            Image.new('RGB',(16,12)).save(path)
            self.assertEqual(png_shape(path),[16,12])
            path=Path(tmp)/'gray.png'
            Image.new('L',(16,12)).save(path)
            with self.assertRaises(ValueError):png_shape(path)

if __name__=='__main__':unittest.main()
