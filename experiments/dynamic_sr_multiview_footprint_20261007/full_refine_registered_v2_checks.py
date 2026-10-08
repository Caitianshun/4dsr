"""CPU-only selection checks for the v2 native full-SR registrar."""
from __future__ import annotations
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import full_refine_registered_v2 as sr

ROOT = HERE.parents[1]
SELECTION = ROOT / 'output/dynamic_sr_multiview_footprint_20261007/summary_recovery_20261007/registered_full_development_selection.json'

def main():
    value = json.loads(SELECTION.read_text())
    sr.validate_selection(value, 'B0')
    sr.validate_selection(value, 'Bsync')
    bad = dict(value, selected_candidates=['B0'])
    try:
        sr.validate_selection(bad, 'B0')
    except ValueError:
        pass
    else:
        raise AssertionError('Baseline cannot also be listed as candidate')
    too_many = dict(value, selected_candidates=['Bsync', 'M', 'X'])
    try:
        sr.validate_selection(too_many, 'Bsync')
    except ValueError:
        pass
    else:
        raise AssertionError('More than two candidates must remain refused')
    print(json.dumps({'status':'passed_v2_registered_B0_Bsync_selection_CPU_contract',
                      'selection_path':str(SELECTION),
                      'selected_candidates':value['selected_candidates'],
                      'baseline':value['baseline_method'],
                      'CUDA_initialized':False,'formal_updates':0},ensure_ascii=False))

if __name__ == '__main__': main()
