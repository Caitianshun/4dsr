"""Path-only adaptation of teacher_physlr3.py; output stays in the new namespace.

The original parser/weights/model/padding/batching/rounding/index algorithm is
executed unchanged. Four exact source substitutions add an output-root argument
and express output image paths relative to the original manifest root. --inspect
checks this adaptation on CPU; generation requires Claude's torch environment.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_new import *

SOURCE = PHYS / "teacher_physlr3.py"
SOURCE_SHA = "dde031195a6db7f4b64eaa5c921788a9f8f181e7d047363a37052243836acde6"


def adapted_source():
    require(sha(SOURCE) == SOURCE_SHA, "Pinned teacher producer changed")
    original = SOURCE.read_text()
    replacements = [
        ('    a = p.parse_args()\n',
         '    p.add_argument("--out-root", required=True)\n    a = p.parse_args()\n', 1),
        ('out = root / TAG / cam / f"{f:04d}.png"',
         'out = ROOT / a.out_root / TAG / cam / f"{f:04d}.png"', 1),
        ('str(out.relative_to(root))', 'os.path.relpath(out, root)', 2),
        ('out = root / f"teacher_index_{TAG}_seed{seed}.json"',
         'out = ROOT / a.out_root / f"teacher_index_{TAG}_seed{seed}.json"', 1),
    ]
    text = original
    for before, after, count in replacements:
        require(text.count(before) == count, "Teacher path adaptation no longer matches exact source")
        text = text.replace(before, after)
    ast.parse(text)
    # Invert only those substitutions to prove no unrelated source edit.
    restored = text
    for before, after, _ in reversed(replacements):
        restored = restored.replace(after, before)
    require(restored == original, "Teacher algorithm changed")
    return text, dict(
        original=entry(SOURCE), wrapper=entry(__file__),
        adapted_source_sha256=__import__("hashlib").sha256(text.encode()).hexdigest(),
        substitutions=[dict(before=a, after=b, occurrences=n) for a, b, n in replacements],
        algorithm_roundtrip_byte_identical=True, torch_imported=False)


def main():
    args = sys.argv[1:]
    text, record = adapted_source()
    if args == ["--inspect"]:
        print(__import__("json").dumps(record, ensure_ascii=False))
        return
    require("--out-root" in args, "--out-root required")
    out = output_path(args[args.index("--out-root") + 1])
    require(out.resolve().is_relative_to((OUT / "teacher_cache").resolve()), "Teacher cache namespace required")
    mp = local(args[args.index("--manifest") + 1])
    m = read(mp)
    require(m["scene"] in NEW and mp == PREP / f"{m['scene']}_area4/manifest_train_ready_seed{SEED}.json",
            "Only original area4 train-ready manifest allowed")
    require("--batch" in args and args[args.index("--batch")+1] == "6"
            and "--weights" in args and local(args[args.index("--weights")+1]) ==
            ROOT / "output/dynamic_sr_physlr_20261016/teacher_ft/swinir_area_ft_v1",
            "Only the historical batch6 frozen areaft teacher recipe is supported")
    # No pre-existing outputs may be rewritten, even by the original teacher's
    # idempotent resume path.
    require(not out.exists(), "Teacher output must be fresh; use a new output revision")
    out.mkdir(parents=True)
    env = {"__file__": str(SOURCE), "__name__": "_newscene_teacher_physlr3_path_adapter"}
    exec(compile(text, str(SOURCE), "exec"), env)
    env["main"]()
    record.update(torch_imported=True, GPU_calls="teacher forwards; owner-dispatched only",
                  manifest=entry(mp), output_root=rel(out))
    write_new(out / "output_path_adapter.json", record)


if __name__ == "__main__":
    main()
