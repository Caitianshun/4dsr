"""Original fixed evaluator plus train16 RGB-to-teacher and privileged labels."""
from pathlib import Path
import types
import sys

ROOT=Path(__file__).resolve().parents[2]
ORIGINAL=ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py'
sys.path.insert(0,str(ORIGINAL.parent))


def main():
    s=ORIGINAL.read_text();needle="    parser.add_argument('--manifest', type=Path, required=True)";assert s.count(needle)==1
    s=s.replace(needle,needle+"\n    parser.add_argument('--teacher-index', type=Path, required=True)")
    s=s.replace('teacher_config = json.loads(config_path.read_text())',"teacher_config = json.loads(args.teacher_index.read_text())\n        teacher_config['teacher_inputs'] = teacher_config['entries']")
    s=s.replace('                h_hr = detail_metrics(h_gt)','                rgb_teacher = None\n                h_hr = detail_metrics(h_gt)')
    s=s.replace('                    h_teacher = detail_metrics(highpass(teacher_tensor, lr_size))',
        '                    h_teacher = detail_metrics(highpass(teacher_tensor, lr_size))\n                    rgb_teacher = float((raw-teacher_tensor).double().abs().mean())')
    s=s.replace("'h_hr': h_hr, 'h_teacher': h_teacher or {}","'rgb_teacher_l1': rgb_teacher, 'h_hr': h_hr, 'h_teacher': h_teacher or {}")
    s=s.replace("'checkpoint_metadata': model.checkpoint['metadata']","'privileged_train_hr': bool(model.checkpoint['metadata'].get('privileged_train_hr',False)), 'adapter_sha256': legacy.file_sha(adapter_path), 'checkpoint_metadata': model.checkpoint['metadata']")
    module=types.ModuleType('controlled_headroom_fixed_evaluator');module.__file__=str(ORIGINAL)
    exec(compile(s,str(ORIGINAL),'exec'),module.__dict__);module.adapter_path=__file__;module.main()


if __name__=='__main__':main()
