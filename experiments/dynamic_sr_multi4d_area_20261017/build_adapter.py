"""Generate the area4 observation adapter for pinned, repaired Multi4D.

The original training loop remains third-party code in an experiment artifact.
Every changed RGB call, bookkeeping hook and native-loss contribution is listed
in identity.json; the unified patch is reviewable without importing CUDA code.
"""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = "c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0"
FACTOR = 4


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _apply_edits(source, edits):
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def _offsets(source):
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def adapt(source):
    """Return generated source and an audit of the controlled modifications."""
    def replace(before, after, count=1):
        nonlocal source
        if source.count(before) != count:
            raise ValueError((before, source.count(before), count))
        source = source.replace(before, after)

    start = source.index("    # ---- Checkpoint Resume Logic ----")
    end = source.index("    bg_color =", start)
    source = source[:start] + """    first_iter = adapter_context.stage_start(stage, foreground_gaussians,
        background_gaussians, transient_gaussian, optimization_params)
""" + source[end:]
    start = source.index("        idx = 0\n        viewpoint_cams = []")
    end = source.index("        # ---- Batch Rendering Setup ----", start)
    source = source[:start] + """        viewpoint_cams = [train_cams[i] for i in adapter_context.batch(stage, iteration)]
""" + source[end:]
    replace("viewpoint_cam.original_image.float().cuda() / 255",
            "viewpoint_cam.original_image.cuda()", 2)

    # Threshold at HR exactly as upstream, then average each 4x4 membership
    # footprint. The same LR mask weights both regional RGB loss terms.
    replace("""                _mask = _region > 1 - vis_thresh
                _l1 = l1_loss(_render * _mask, gt_image_tensor * _mask)
                loss += _lam_ssim * ((1.0 - ssim_raw(_render, gt_image_tensor)) * _mask).mean()
                loss += _l1""", """                _mask = torch.nn.functional.avg_pool2d((_region > 1 - vis_thresh).float(), 4, 4)
                _lr_render = lr_project(_render, gt_image_tensor)
                _l1 = ((_lr_render - gt_image_tensor).abs() * _mask).mean()
                loss += _lam_ssim * ((1.0 - ssim_raw(_lr_render, gt_image_tensor)) * _mask).mean()
                loss += _l1""")

    # Depth TV stays at its original HR spatial support. Its edge-guidance
    # image contains LR information only (fixed bilinear expansion).
    replace("gt_image_tensor.permute(0, 2, 3, 1)",
            "torch.nn.functional.interpolate(gt_image_tensor, size=depth_scaled.shape[-2:], "
            "mode='bilinear', align_corners=False).permute(0, 2, 3, 1)")
    replace("""        loss.backward()

        # Emergency restart on NaN loss
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)""", """        native_loss = loss
        teacher_term = adapter_context.teacher(stage, viewpoint_cams,
            image_tensor_first if in_phase3 else image_tensor_hybrid_full_batch)
        loss = native_loss + teacher_term
        adapter_context.before_backward(stage, iteration, locals())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Nonfinite loss: {stage} {iteration}')
        loss.backward()
        adapter_context.after_backward(stage, iteration, foreground_gaussians,
            background_gaussians, transient_gaussian, teacher_term)
""")
    # Native debug/evaluation assumes HR target images. Evaluation is a separate
    # fixed-endpoint process here and cannot select a checkpoint during training.
    start = source.index("            # ---- Logging and Saving ----")
    end = source.index("            # FG->TR organic transfer", start)
    source = source[:start] + source[end:]
    # The released strict '<' skips the final Adam update. A checkpoint called
    # completed_updates=20000 must contain all 20000 actual updates.
    replace("if iteration < optimization_params.iterations:",
            "if iteration <= final_iter:", 3)
    start = source.index("            # ---- Checkpoint Saving ----")
    end = source.index("\n\ndef training(", start)
    source = source[:start] + """            adapter_context.after_update(stage, iteration, loss, Ll1, teacher_term,
                foreground_gaussians, background_gaussians, transient_gaussian,
                optimization_params, in_phase3)
""" + source[end:]

    # Project only RGB-to-observation comparisons. Prediction-prediction
    # separation/diversity and geometry regularizers retain native HR behavior.
    offsets = _offsets(source)
    edits, mapped = [], []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"l1_loss", "ssim", "ssim_raw", "psnr"}
                and len(node.args) >= 2
                and "gt_image_tensor" in ast.unparse(node.args[1])
                and ast.unparse(node.args[0]) != "_lr_render"):
            arg = node.args[0]
            begin = offsets[arg.lineno - 1] + arg.col_offset
            end = offsets[arg.end_lineno - 1] + arg.end_col_offset
            mapped.append(dict(function=node.func.id,
                               prediction=ast.unparse(arg),
                               target=ast.unparse(node.args[1])))
            edits.append((begin, end, "lr_project(" + source[begin:end] + ", "
                          + ast.unparse(node.args[1]) + ")"))
    if len(mapped) != 15:
        raise ValueError(("Pinned RGB call count changed", mapped))
    source = _apply_edits(source, edits)

    # Capture every native contribution without per-term CUDA synchronization.
    # Detached scalars are transferred together only at logging checkpoints.
    offsets = _offsets(source)
    edits, terms = [], []
    loop = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "scene_reconstruction")
    for node in ast.walk(loop):
        is_initial = (isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "loss" for t in node.targets))
        is_add = (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                  and node.target.id == "loss")
        if not (is_initial or is_add):
            continue
        value = node.value
        expression = ast.unparse(value)
        if expression == "native_loss + teacher_term":
            continue
        label = "native_%02d" % len(terms)
        terms.append(dict(label=label, operation="assign" if is_initial else "add",
                          expression=expression))
        begin = offsets[value.lineno - 1] + value.col_offset
        end = offsets[value.end_lineno - 1] + value.end_col_offset
        edits.append((begin, end, "adapter_context.native_term(stage, iteration, "
                      + repr(label) + ", (" + source[begin:end] + "))"))
    source = _apply_edits(source, edits)
    source += '''\n\ndef lr_project(prediction, target):
    """The current Wu area4 operator, including its [0,1] range clamp."""
    expected = (target.shape[-2] * 4, target.shape[-1] * 4)
    if tuple(prediction.shape[-2:]) != expected:
        raise ValueError(f'area4 shape mismatch: {prediction.shape} vs {target.shape}')
    return torch.nn.functional.avg_pool2d(prediction, kernel_size=4, stride=4).clamp(0, 1)
'''
    compile(source, "adapted_train.py", "exec")
    audit = dict(target_rgb_calls=mapped, native_loss_terms=terms,
                 observation="HR RGB -> avg_pool2d(kernel_size=4,stride=4).clamp(0,1), identical to current Wu area4",
                 region_mask="HR threshold -> exact avg_pool2d(4,4), shared by regional L1 and SSIM",
                 depth_guidance="LR bilinear expansion only; depth regularizer remains HR",
                 final_update_repair="All three optimizers execute iteration <= final_iter",
                 evaluation="Native debug/validation side effects replaced by explicit fixed-endpoint evaluation")
    return source, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise ValueError((head, COMMIT))
    original = subprocess.check_output(["git", "-C", str(args.upstream), "show", COMMIT + ":train.py"], text=True)
    source, audit = adapt(original)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "adapted_train.py").write_text(source)
    patch = "".join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
                                         fromfile="upstream/train.py", tofile="adapted_train.py"))
    (args.out / "train_area4.patch").write_text(patch)
    identity = dict(schema="multi4d_area4_adapter_v1", commit=COMMIT,
                    original_sha256=_sha_bytes(original.encode()),
                    adapted_sha256=_sha_bytes(source.encode()),
                    builder_sha256=_sha_bytes(Path(__file__).read_bytes()), **audit)
    (args.out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
