"""Use established ROI coordinates and video helpers, with explicit O privilege."""
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/dynamic_sr_multi4d_20260924/export_views.py'


def main():
    sys.path.insert(0,str(SOURCE.parent));s=SOURCE.read_text()
    start=s.index("    groups={'teacher':");end=s.index('    artifacts=[]',start)
    s=s[:start]+"    groups={'headroom':['Z40000','U40000','O_HR_PRIVILEGED40000']}\n"+s[end:]
    s=s.replace('# Multi4D / Wu 固定图像与视频','# Z/U/O 固定图像与视频：O 为真实训练 HR 特权诊断')
    s=s.replace("f'{group} {camera} frame{frame}; display resized'", "f'{group} {camera} frame{frame}; O uses PRIVILEGED train HR; display resized'")
    s=s.replace("f'{group} {camera} frame{frame}; matched time'", "f'{group} {camera} frame{frame}; O uses PRIVILEGED train HR; matched time'")
    module=types.ModuleType('controlled_headroom_fixed_views');module.__file__=str(SOURCE)
    exec(compile(s,str(SOURCE),'exec'),module.__dict__);module.main()


if __name__=='__main__':main()
