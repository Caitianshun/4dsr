"""CPU-only frozen-teacher padding and recipe checks; no real inference."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from .splatsure_teacher import infer, sha, DEFAULT_NETWORK, DEFAULT_CHECKPOINT, NETWORK_SHA256, CHECKPOINT_SHA256


class TeacherTests(unittest.TestCase):
    def test_whole_image_official_mirror_pad_crop_and_rgb(self):
        class Model:
            def __call__(self, tensor):
                self.shape = tensor.shape
                self.tensor = tensor
                assert not torch.is_grad_enabled()
                return F.interpolate(tensor, scale_factor=4, mode='nearest')
        model=Model()
        image=np.arange(8*16*3,dtype=np.uint8).reshape(8,16,3)
        result=infer(model,image,device='cpu')
        self.assertEqual(tuple(model.shape),(1,3,16,24))
        expected=np.repeat(np.repeat(image,4,axis=0),4,axis=1)
        np.testing.assert_array_equal(result,expected)
        torch.testing.assert_close(model.tensor[0,:,:8,16:],model.tensor[0,:,:8,8:16].flip([-1]))

    def test_tiled_uniform_average_matches_whole_for_local_model(self):
        class Model:
            def __call__(self,tensor):
                return F.interpolate(tensor,scale_factor=4,mode='nearest')
        image=np.random.default_rng(4).integers(0,256,(16,24,3),dtype=np.uint8)
        np.testing.assert_array_equal(infer(Model(),image,device='cpu'),
                                      infer(Model(),image,device='cpu',tile=16,overlap=8))

    def test_available_generic_prior_identities(self):
        if not DEFAULT_NETWORK.exists() or not DEFAULT_CHECKPOINT.exists():
            self.skipTest('optional local generic SwinIR source/weight assets absent')
        self.assertEqual(sha(DEFAULT_NETWORK),NETWORK_SHA256)
        self.assertEqual(sha(DEFAULT_CHECKPOINT),CHECKPOINT_SHA256)

if __name__=='__main__':unittest.main()
