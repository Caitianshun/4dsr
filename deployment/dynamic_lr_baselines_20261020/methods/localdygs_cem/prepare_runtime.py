#!/usr/bin/env python3
"""Copy immutable upstream, apply explicit I/O patches, and emit a patch receipt.

No initialization, loss, model, densification, or training schedule is changed.
This does not make either implementation ready before its native COLMAP and
environment acceptance have completed.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one exact source match, got {text.count(old)}: {old[:90]}")
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["localdygs", "cem"], required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    a = p.parse_args()
    if a.runtime.exists():
        raise SystemExit(f"Refusing existing runtime: {a.runtime}")
    commit = subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip()
    shutil.copytree(a.upstream, a.runtime, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "build", "*.egg-info"))
    shutil.copy2(Path(__file__).with_name("protocol_interface.py"), a.runtime / "protocol_interface.py")
    changes = {}
    if a.method == "localdygs":
        relative = "scene/neural_3D_dataset_NDC.py"
        target = a.runtime / relative
        before = target.read_text()
        after = replace_once(before, """        self.img_wh = (
            int(1352 / downsample),
            int(1014 / downsample),
        )  # According to the neural 3D paper, the default resolution is 1024x768
""", """        # Data interface: preserve prepared LR pixels, without SR/interpolation.
        image_roots = sorted(Path(datadir).glob('cam[0-9][0-9]/images'))
        image_roots = [x for x in image_roots if x.parent.name != 'cam00']
        if not image_roots:
            raise RuntimeError('No training LR image directories')
        first = sorted(image_roots[0].glob('*.png'))[0]
        with Image.open(first) as image:
            self.img_wh = image.size
        # Evaluation-only override changes camera pixel grid, never training.
        evaluation_wh = os.environ.get('LOCALDYGS_EVAL_WH')
        if split == 'test' and evaluation_wh:
            self.img_wh = tuple(map(int, evaluation_wh.split('x')))
""")
        after = replace_once(after, "from PIL import Image\n", "from PIL import Image\nfrom pathlib import Path\n")
        after = replace_once(after, "        image_roots = sorted(Path(datadir).glob('cam[0-9][0-9]/images'))\n", "        image_roots = sorted(Path(datadir).glob('cam[0-9][0-9]/images'))\n        if os.environ.get('DYNAMIC_LR_PROTOCOL_MANIFEST'):\n            from protocol_interface import observations\n            source, _, records, _ = observations('train')\n            image_roots = [(source.parent / records[0]['lr_path']).parent]\n")
        after = replace_once(after, "        focal = focal / self.downsample\n", "        self.downsample = W / self.img_wh[0]\n        focal = focal / self.downsample\n")
        after = replace_once(after, '        # Read poses and video file paths.\n', '        if os.environ.get("DYNAMIC_LR_PROTOCOL_MANIFEST"):\n            from protocol_interface import init_localdygs_dataset\n            init_localdygs_dataset(self)\n            return\n        # Read poses and video file paths.\n')
        after = replace_once(after, '        img = Image.open(self.image_paths[index])\n', '        if os.environ.get("DYNAMIC_LR_PROTOCOL_MANIFEST") and self.split == "test":\n            return torch.zeros(3, self.img_wh[1], self.img_wh[0]), self.image_poses[index], self.image_times[index]\n        img = Image.open(self.image_paths[index])\n')
        changes[relative] = (before, after)
        relative = "scene/spacetime_dataset.py"
        before = (a.runtime / relative).read_text()
        after = replace_once(before, '            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None, ', '            camera = Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None, ')
        after = replace_once(after, '                              mask=mask)\n', '                              mask=mask)\n            if hasattr(self.dataset, "protocol_intrinsics"):\n                from protocol_interface import apply_localdygs_projection\n                camera = apply_localdygs_projection(camera, self.dataset.protocol_intrinsics[index])\n            return camera\n')
        changes[relative] = (before, after)
        relative = "render.py"
        target = a.runtime / relative
        before = target.read_text()
        after = replace_once(before, "os.environ['CUDA_VISIBLE_DEVICES'] = '2'\n", "# GPU selection is supplied by the scheduler through CUDA_VISIBLE_DEVICES.\n")
        changes[relative] = (before, after)
    else:
        # Official CLI changes CUDA_VISIBLE_DEVICES after parsing --gpu. Keep it
        # unchanged: pass the real assigned device index using --gpu.
        relative = "scene/dataset_readers.py"
        before = (a.runtime / relative).read_text()
        old = '    cam_infos_unsorted = readN3VCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=path, near=near, far=far, \n                                        startime=args.start_timestamp, endtime=args.end_timestamp)\n'
        new = '    if os.environ.get("DYNAMIC_LR_PROTOCOL_MANIFEST"):\n        from protocol_interface import cem_camera_infos\n        cam_infos_unsorted = cem_camera_infos(CameraInfo2)\n    else:\n        cam_infos_unsorted = readN3VCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=path, near=near, far=far,\n                                          startime=args.start_timestamp, endtime=args.end_timestamp)\n'
        after = replace_once(before, old, new)
        changes[relative] = (before, after)
        relative = "scene/__init__.py"
        before = (a.runtime / relative).read_text()
        after = replace_once(before, '    ImageFile.LOAD_TRUNCATED_IMAGES = True\n', '    if str(path).startswith("protocol-zero://"):\n        return torch.zeros(3, resolution[1], resolution[0])\n    ImageFile.LOAD_TRUNCATED_IMAGES = True\n')
        changes[relative] = (before, after)
    # The author reporting function reads held-out pixels and may initialize
    # LPIPS/VGG. Bypass only its read-only test branch, never training losses,
    # gradients, sampling, optimization, or checkpoint/save intervals.
    relative = "train.py"
    before = (a.runtime / relative).read_text()
    after = replace_once(before, '    if iteration in testing_iterations:\n', '    if iteration in testing_iterations and not os.environ.get("DYNAMIC_LR_PROTOCOL_MANIFEST"):\n')
    if a.method == "localdygs":
        after = replace_once(after, "lpips_fn = lpips.LPIPS(net='vgg').to('cuda')\n", "lpips_fn = None if os.environ.get('DYNAMIC_LR_PROTOCOL_MANIFEST') else lpips.LPIPS(net='vgg').to('cuda')\n")
    changes[relative] = (before, after)
    patch = []
    receipts = []
    for relative, (before, after) in changes.items():
        target = a.runtime / relative
        target.write_text(after)
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        receipts.append({"path": relative, "upstream_sha256": sha(a.upstream / relative), "runtime_sha256": sha(target)})
    (a.runtime / "protocol_interface.patch").write_text("".join(patch))
    (a.runtime / "protocol_interface_receipt.json").write_text(json.dumps({"method": a.method, "upstream_commit": commit, "interface_sha256": sha(a.runtime / "protocol_interface.py"), "changes": receipts, "training_executed": False, "patch_scope": "exact calibrated LR/HR camera interface; heldout metadata/zero placeholder; read-only test logger bypass; scheduler GPU selection", "heldout_pixels_read_by_training": False, "native_readonly_test_evaluation_disabled": True, "core_losses_sampling_optimization_schedule_unchanged": True}, indent=2) + "\n")
    if a.method == "cem":
        for scene in ["cook_spinach", "cut_roasted_beef"]:
            for stage in [1, 2]:
                source = a.upstream / f"configs/N3V_s{stage}/{scene}.json"
                config = json.loads(source.read_text())
                config["resolution"] = 1  # The prepared images and COLMAP intrinsics already are LR.
                output = a.runtime / f"configs/N3V_s{stage}/{scene}_prepared_lr.json"
                output.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"runtime": str(a.runtime), "method": a.method, "upstream_commit": commit, "patched_files": list(changes)}, indent=2))


if __name__ == "__main__":
    main()
