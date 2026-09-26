"""Delegate all RGB/region/GT-flow metrics to the accepted original evaluator."""
from pathlib import Path
import runpy

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    runpy.run_path(str(root/'experiments/dynamic_sr_controlled_headroom_20260926/evaluate.py'),run_name='__main__')
