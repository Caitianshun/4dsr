"""Unchanged fixed detail evaluation protocol, geometry-aware model loader."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
DETAIL=ROOT/'experiments/dynamic_sr_detail_supervision_20260924'
sys.path.insert(0,str(DETAIL))
spec=importlib.util.spec_from_file_location('geometry_shared_evaluation',DETAIL/'evaluate.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
shared.MOTION=Path(__file__).with_name('geometry_model.py')
if __name__=='__main__': shared.main()
