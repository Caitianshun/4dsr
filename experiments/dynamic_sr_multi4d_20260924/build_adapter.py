"""Generate a reviewable LR patch against the pinned official training loop.

Upstream is never vendored as original project code. The generated source and
unified diff live with the experiment, with the original commit and file hashes.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = 'c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0'


def adapt(source):
    def replace(old, new, count=1):
        nonlocal source
        assert source.count(old)==count, (old,source.count(old),count)
        source=source.replace(old,new)
    replace('    # ---- Checkpoint Resume Logic ----', '    # ---- Checkpoint Resume Logic ----')
    start=source.index('    # ---- Checkpoint Resume Logic ----')
    end=source.index('    bg_color =',start)
    source=source[:start]+'''    first_iter = adapter_context.stage_start(stage, foreground_gaussians,
        background_gaussians, transient_gaussian, optimization_params)
'''+source[end:]
    start=source.index('        idx = 0\n        viewpoint_cams = []')
    end=source.index('        # ---- Batch Rendering Setup ----',start)
    source=source[:start]+'''        viewpoint_cams = [train_cams[i] for i in adapter_context.batch(stage, iteration)]
'''+source[end:]
    replace('viewpoint_cam.original_image.float().cuda() / 255', 'viewpoint_cam.original_image.cuda()',2)
    replace('''                _mask = _region > 1 - vis_thresh
                _l1 = l1_loss(_render * _mask, gt_image_tensor * _mask)
                loss += _lam_ssim * ((1.0 - ssim_raw(_render, gt_image_tensor)) * _mask).mean()
                loss += _l1''','''                _mask = torch.nn.functional.avg_pool2d((_region > 1 - vis_thresh).float(), 4, 4)
                _lr_render = lr_project(_render, gt_image_tensor)
                _l1 = ((_lr_render - gt_image_tensor).abs() * _mask).mean()
                loss += _lam_ssim * ((1.0 - ssim_raw(_lr_render, gt_image_tensor)) * _mask).mean()
                loss += _l1''')
    replace('gt_image_tensor.permute(0, 2, 3, 1)',
        "torch.nn.functional.interpolate(gt_image_tensor, size=depth_scaled.shape[-2:], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)")
    replace('''        loss.backward()

        # Emergency restart on NaN loss
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)''','''        teacher_term = adapter_context.teacher(stage, viewpoint_cams,
            image_tensor_first if in_phase3 else image_tensor_hybrid_full_batch)
        loss = loss + teacher_term
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Nonfinite loss: {stage} {iteration}')
        loss.backward()
        adapter_context.after_backward(stage, iteration, foreground_gaussians,
            background_gaussians, transient_gaussian, teacher_term)
''')
    start=source.index('            # ---- Logging and Saving ----')
    end=source.index('            # FG->TR organic transfer',start)
    source=source[:start]+source[end:]
    replace('if iteration < optimization_params.iterations:', 'if iteration <= final_iter:',3)
    start=source.index('            # ---- Checkpoint Saving ----')
    end=source.index('\n\ndef training(',start)
    source=source[:start]+'''            adapter_context.after_update(stage, iteration, loss, Ll1, teacher_term,
                foreground_gaussians, background_gaussians, transient_gaussian,
                optimization_params, in_phase3)
'''+source[end:]
    # Only RGB-target calls are projected; prediction-prediction losses stay HR.
    tree=ast.parse(source)
    mapped=[]
    class Targets(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if (isinstance(node.func,ast.Name) and node.func.id in {'l1_loss','ssim','ssim_raw','psnr'}
                and len(node.args)>=2 and 'gt_image_tensor' in ast.unparse(node.args[1])
                and ast.unparse(node.args[0])!='_lr_render'):
                mapped.append(dict(function=node.func.id,prediction=ast.unparse(node.args[0]),target=ast.unparse(node.args[1])))
                node.args[0]=ast.Call(func=ast.Name(id='lr_project',ctx=ast.Load()),args=[node.args[0],node.args[1]],keywords=[])
            return node
    Targets().visit(tree); ast.fix_missing_locations(tree)
    assert len(mapped)==15, mapped
    # Keep comments and exact native lines: apply only the AST-located call args.
    original=ast.parse(source); edits=[]
    lines=source.splitlines(keepends=True); offsets=[0]
    for line in lines: offsets.append(offsets[-1]+len(line))
    for node in ast.walk(original):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
            and node.func.id in {'l1_loss','ssim','ssim_raw','psnr'} and len(node.args)>=2
            and 'gt_image_tensor' in ast.unparse(node.args[1]) and ast.unparse(node.args[0])!='_lr_render'):
            a=node.args[0];i=offsets[a.lineno-1]+a.col_offset;j=offsets[a.end_lineno-1]+a.end_col_offset
            edits.append((i,j,'lr_project('+source[i:j]+', '+ast.unparse(node.args[1])+')'))
    for i,j,repl in sorted(edits,reverse=True): source=source[:i]+repl+source[j:]
    source+='''\n\ndef lr_project(prediction, target):
    return torch.nn.functional.interpolate(prediction, size=target.shape[-2:],
        mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
'''
    compile(source,'adapted_train.py','exec')
    return source,mapped


def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();head=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip();assert head==COMMIT
    original=subprocess.check_output(['git','-C',str(a.upstream),'show',COMMIT+':train.py'],text=True)
    source,mapped=adapt(original);a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'adapted_train.py').write_text(source)
    (a.out/'train_lr.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='upstream/train.py',tofile='adapted_train.py')))
    (a.out/'identity.json').write_text(json.dumps(dict(commit=COMMIT,original_sha256=hashlib.sha256(original.encode()).hexdigest(),adapted_sha256=hashlib.sha256(source.encode()).hexdigest(),target_rgb_calls=mapped),indent=2)+'\n')


if __name__=='__main__':main()
