"""CPU preparation of a native-LR Multi4D worker; never edits upstream.

The reused runner supplies state capture and logging. The generated upstream
patch restores all RGB/depth/mask losses to the author's native pixel grid and
preserves the author's skipped final fine-stage Adam step. It has no SR arm.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
AREA = ROOT / "experiments/dynamic_sr_multi4d_area_20261017"
COMMIT = "c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def module(path):
    spec = importlib.util.spec_from_file_location("audited_multi_area_builder", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def native_source(original):
    source, audit = module(AREA / "build_adapter.py").adapt(original)
    source = source.replace("                _lr_render = lr_project(_render, gt_image_tensor)\n", "")
    source = source.replace("torch.nn.functional.avg_pool2d((_region > 1 - vis_thresh).float(), 4, 4)", "_region > 1 - vis_thresh")
    source = source.replace("_lr_render", "_render")
    source = source.replace("((_render - gt_image_tensor).abs() * _mask).mean()", "l1_loss(_render * _mask, gt_image_tensor * _mask)")
    source = source.replace("torch.nn.functional.interpolate(gt_image_tensor, size=depth_scaled.shape[-2:], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)", "gt_image_tensor.permute(0, 2, 3, 1)")
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "lr_project":
            arg = node.args[0]
            edits.append((offsets[node.lineno - 1] + node.col_offset,
                          offsets[node.end_lineno - 1] + node.end_col_offset,
                          source[offsets[arg.lineno - 1] + arg.col_offset:offsets[arg.end_lineno - 1] + arg.end_col_offset]))
    assert len(edits) == 15, len(edits)
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    source = source[:source.index("\n\ndef lr_project(")]
    assert source.count("if iteration <= final_iter:") == 3
    source = source.replace("if iteration <= final_iter:", "if iteration < optimization_params.iterations:")
    assert "lr_project(" not in source and "avg_pool2d(" not in source
    compile(source, "native_adapted_train.py", "exec")
    actual_terms = {node.args[2].value: ast.unparse(node.args[3])
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "adapter_context"
                    and node.func.attr == "native_term"}
    for term in audit["native_loss_terms"]:
        term["expression"] = actual_terms[term["label"]]
    audit.update(observation="Native LR RGB compared directly with original LR; no resize, SR or area pooling",
                 region_mask="Native LR threshold; native regional L1 and SSIM",
                 depth_guidance="Original LR target guides native LR depth regularizer",
                 final_update_repair="None: original iteration < optimization_params.iterations retained",
                 target_rgb_calls=[], parent_builder=str((AREA / "build_adapter.py").relative_to(ROOT)))
    return source, audit


def main():
    upstream = ROOT / "output/dynamic_sr_multi4d_area_20261017/runtime_v2/Multi4D-c483e82"
    original = subprocess.check_output(["git", "-C", str(upstream), "show", COMMIT + ":train.py"], text=True)
    source, audit = native_source(original)
    out = HERE / "multi4d_native"
    write(out / "adapter/adapted_train.py", source)
    write(out / "adapter/train_native_lr.patch", "".join(difflib.unified_diff(original.splitlines(True), source.splitlines(True), fromfile="upstream/train.py", tofile="native_lr/train.py")))
    identity = dict(audit, schema="multi4d_native_lr_adapter_v1", commit=COMMIT,
                    original_sha256=sha(original.encode()), adapted_sha256=sha(source.encode()),
                    builder_sha256=sha(Path(__file__).read_bytes()))
    write(out / "adapter/identity.json", json.dumps(identity, indent=2) + "\n")
    data = (AREA / "data.py").read_text()
    # Only the camera-construction/projection-check functions select a grid.
    start, end = data.index("def camera("), data.index("class ManifestScene:")
    camera = data[start:end].replace('["resolutions"]["hr"]', '["resolutions"]["lr"]').replace('["K_hr"]', '["K_lr"]')
    data = data[:start] + camera + data[end:]
    data = data.replace('manifest = json.loads(path.read_text())', 'manifest = json.loads(path.read_text())\n    if manifest.get("physical_lr_derivation"):\n        raise ValueError("Original LR is required; area4 re-formation is forbidden")', 1)
    write(out / "data.py", data)
    train = (AREA / "train.py").read_text()
    train = train.replace("PROJECT = HERE.parent.parent", "PROJECT = Path(__file__).resolve().parents[5]")
    train = train.replace('LEGACY = HERE.parent / "dynamic_sr_multi4d_20260924"', 'LEGACY = PROJECT / "experiments/dynamic_sr_multi4d_20260924"')
    train = train.replace('METHODS = ("coarse", "areaL0", "areaTm", "areaOracle")', 'METHODS = ("coarse", "nativeLR")')
    train = train.replace('HERE / "build_adapter.py"', 'HERE.parent / "prepare_multi_native.py"')
    train = train.replace('multi4d_area4_adapter_v1', 'multi4d_native_lr_adapter_v1').replace('multi4d_area4_run_v1', 'multi4d_native_lr_run_v1')
    train = train.replace('observation="area4 avg_pool2d(4,4).clamp(0,1), identical to current Wu"', 'observation="native original LR RGB; direct native LR losses"')
    train = train.replace('final_update_repair="three native optimizers execute at the inclusive endpoint"', 'final_update_repair="None; released author final fine Adam skip preserved"')
    train = train.replace('optimizer_updates_per_branch=step)', 'optimizer_updates_per_branch=min(step, opt.iterations - 1), completed_loop_iterations=step)')
    write(out / "train.py", train)
    write(out / "eval_hr_data.py", (AREA / "data.py").read_text())
    evaluate = (AREA / "evaluate.py").read_text()
    evaluate = evaluate.replace("ROOT = HERE.parents[1]", "ROOT = Path(__file__).resolve().parents[5]")
    evaluate = evaluate.replace("HERE / 'data.py'", "HERE / 'eval_hr_data.py'")
    evaluate = evaluate.replace("if manifest.get('physical_lr_derivation', {}).get('formation') != 'area4':\n        raise ValueError('A registered area4 derived manifest is required')", "if manifest.get('physical_lr_derivation'):\n        raise ValueError('Original registered LR manifest required; no area4 re-formation')")
    evaluate = evaluate.replace("F.avg_pool2d(raw.float()[None], 4)[0].clamp(0, 1)", "F.interpolate(raw.float()[None], size=(manifest['resolutions']['lr'][1], manifest['resolutions']['lr'][0]), mode='bicubic', align_corners=False, antialias=True)[0].clamp(0, 1)")
    write(out / "evaluate.py", evaluate)
    for path in (out / "train.py", out / "data.py", out / "evaluate.py", out / "eval_hr_data.py"):
        compile(path.read_text(), str(path), "exec")
    print(json.dumps(dict(status="CPU_source_generated_not_CUDA_validated", output=str(out),
                          identity=str(out / "adapter/identity.json")), indent=2))


if __name__ == "__main__":
    main()
