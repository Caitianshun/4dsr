"""Run numerical and boundary checks without creating a CUDA context."""
import argparse
import json
import sys
import unittest
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt');a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    suite=unittest.TestSuite()
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    names=['experiments.static_sr_wu_20261022.methods.'+n for n in
           ('splatsure_test','splatsure_teacher_test','splatsure_stablesr_test','iesrgs_cpu_tests')]
    for name in names:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    suite.addTests(unittest.defaultTestLoader.discover(str(Path(__file__).parent/'tests')))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),cuda_initialized=torch.cuda.is_initialized(),
        GPU_forward_backward_calls=0)
    if receipt['cuda_initialized']:
        raise RuntimeError('CPU suite unexpectedly initialized CUDA')
    if a.receipt:
        from common import write_new
        write_new(a.receipt,receipt)
    print(json.dumps(receipt))
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__=='__main__':main()
