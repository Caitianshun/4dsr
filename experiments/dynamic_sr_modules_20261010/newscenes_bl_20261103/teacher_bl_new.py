"""Run teacher_cache_new unchanged, redirecting only its allowed output root.

Same pinned teacher_physlr3 algorithm, frozen areaft weights and batch6.
For later owner-dispatched A100 inference; not executed in this P2 task.
"""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import CODE, OUT, ROOT, load


def main():
    original = ROOT / "experiments/dynamic_sr_modules_20261010/newscenes_reg_20261103"
    sys.path.insert(0, str(original))
    import common_new
    module = load("_original_newscene_teacher_cache", original / "teacher_cache_new.py")
    # Two bindings of the same output constant; no source transformation,
    # algorithm/weights/batch changes or write access to the old namespace.
    previous = common_new.OUT, module.OUT
    common_new.OUT = module.OUT = OUT
    try:
        module.main()
    finally:
        common_new.OUT, module.OUT = previous


if __name__ == "__main__":
    main()
