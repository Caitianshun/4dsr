"""Reuse exact native ROI/video helpers with all preregistered recovery endpoints."""
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/dynamic_sr_multi4d_20260924/export_views.py'

def main():
    sys.path.insert(0,str(SOURCE.parent));s=SOURCE.read_text()
    start=s.index("    groups={'teacher':");end=s.index('    artifacts=[]',start)
    s=s[:start]+"    groups={'late18000':['U6000','C_joint_18000','P_late_18000','A_late_18000'], 'early12000':['U6000','shared_12000','P_early_12000']}\n    if 'P_early_18000' in methods: groups['early18000']=['U6000','C_joint_18000','P_early_18000']\n"+s[end:]
    s=s.replace('# Multi4D / Wu 固定图像与视频','# 先验梯度路由：固定原生 ROI 和两开发相机完整视频')
    module=types.ModuleType('recovery_fixed_views');module.__file__=str(SOURCE)
    exec(compile(s,str(SOURCE),'exec'),module.__dict__);module.main()

if __name__=='__main__':main()
