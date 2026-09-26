"""Reuse exact native ROI/video helpers with all preregistered recovery endpoints."""
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/dynamic_sr_multi4d_20260924/export_views.py'

def main():
    sys.path.insert(0,str(SOURCE.parent));s=SOURCE.read_text()
    start=s.index("    groups={'teacher':");end=s.index('    artifacts=[]',start)
    s=s[:start]+"    groups={'recovery18000':['U6000','C_joint18000','F_app18000'], 'history':['U6000','U18000','U40000']}\n    if 'F_app40000' in methods: groups['recovery40000']=['U6000','C_joint40000','F_app40000']\n"+s[end:]
    s=s.replace('# Multi4D / Wu 固定图像与视频','# U6000 与配对续训：固定原生 ROI 和两开发相机完整视频')
    module=types.ModuleType('recovery_fixed_views');module.__file__=str(SOURCE)
    exec(compile(s,str(SOURCE),'exec'),module.__dict__);module.main()

if __name__=='__main__':main()
