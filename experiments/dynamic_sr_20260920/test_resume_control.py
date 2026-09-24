"""CPU checks for continuation sampling and Adam freeze semantics."""
import collections
import importlib.util
from pathlib import Path
import random
import unittest

SPEC = importlib.util.spec_from_file_location("resume_control", Path(__file__).with_name("resume_control.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SamplerTest(unittest.TestCase):
    def setUp(self):
        self.records = [dict(camera_id=f"cam{i//4:02}", frame_index=2*(i%4)) for i in range(12)]
        self.ids, self.seed, self.prefix, self.total = [4, 5, 6, 7, 8, 9, 10, 11], 20260919, 6000, 18000
        lr, sr = random.Random(self.seed+177), random.Random(self.seed+211)
        self.draws, self.exposure = [], collections.Counter()
        for _ in range(self.total):
            pair = lr.randrange(len(self.records)), sr.choice(self.ids)
            self.draws.append(pair)
            self.exposure[MODULE.exposure_key(self.records[pair[1]])] += 1

    def test_exact_prefix_and_continuation(self):
        lr, sr, prefix, checksum = MODULE.rebuild_samplers(
            self.seed, self.records, self.ids, self.prefix, self.total, self.exposure)
        expected_prefix = collections.Counter(MODULE.exposure_key(self.records[si]) for _, si in self.draws[:self.prefix])
        self.assertEqual(prefix, expected_prefix)
        actual = [(lr.randrange(len(self.records)), sr.choice(self.ids)) for _ in range(self.total-self.prefix)]
        self.assertEqual(actual, self.draws[self.prefix:])
        self.assertEqual(len(checksum), 64)

    def test_wrong_exposure_fails(self):
        self.exposure["cam01/0"] += 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE.rebuild_samplers(self.seed, self.records, self.ids, self.prefix, self.total, self.exposure)

    def test_changed_record_order_fails(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE.rebuild_samplers(self.seed, list(reversed(self.records)), self.ids,
                                   self.prefix, self.total, self.exposure)


class AdamFreezeTest(unittest.TestCase):
    def test_none_gradient_preserves_parameter_moments_and_step(self):
        import copy
        import torch
        support = torch.nn.Parameter(torch.tensor([1., 2.]))
        color = torch.nn.Parameter(torch.tensor([3., 4.]))
        optimizer = torch.optim.Adam([support, color], lr=.1)
        ((support + color)**2).sum().backward()
        optimizer.step()
        old_support = support.detach().clone()
        old_color = color.detach().clone()
        old_support_state = copy.deepcopy(optimizer.state[support])
        roles = MODULE.apply_parameter_mode({"support": support, "color": color}, [color], "appearance_only")
        self.assertFalse(roles["support"]["trainable"])
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            ((support + color)**2).sum().backward()
            self.assertIsNone(support.grad)
            optimizer.step()
        self.assertTrue(torch.equal(old_support, support))
        self.assertFalse(torch.equal(old_color, color))
        MODULE.assert_state_equal(old_support_state, optimizer.state[support])

    def test_joint_preserves_upstream_frozen_constants(self):
        import torch
        a = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        b = torch.nn.Parameter(torch.ones(1))
        roles = MODULE.apply_parameter_mode({"a": a, "b": b}, [b], "joint")
        self.assertFalse(roles["a"]["trainable"])
        self.assertFalse(roles["a"]["original_trainable"])
        self.assertTrue(roles["b"]["trainable"])
        self.assertFalse(a.requires_grad)
        self.assertTrue(b.requires_grad)


if __name__ == "__main__":
    unittest.main()
