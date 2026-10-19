"""Build a small auditable STG observation adapter; never edit upstream files."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

COMMIT = "427abfc58309a4a5213843dd673fb22c4529306c"
EXPECTED = {
    "train.py": "2e38ed71ff2e292983adde7d26a3dd0f62dbcf8c8db560b092c227032d1f9523",
    "helper_model.py": "97994d47edf46baff32939f02641a9988ee0d6f860d75212a677aa620c0ff80a",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Pinned STG patch anchor changed: {before[:100]!r}")
    return text.replace(before, after, 1)


def build(upstream, out):
    upstream, out = Path(upstream).resolve(), Path(out).resolve()
    originals = {}
    for name, expected in EXPECTED.items():
        if sha(upstream / name) != expected:
            raise ValueError(f"Unreviewed upstream bytes: {name}")
        originals[name] = (upstream / name).read_text()
    source = originals["train.py"]
    source = replace_once(source, 'rdpip="v2"):', 'rdpip="v2", bridge=None):')
    source = replace_once(source, "    with open(os.path.join(args.model_path, \"cfg_args\"), 'w') as cfg_log_f:",
                          "    if bridge is None:\n        raise ValueError('area4 bridge is required')\n    with open(os.path.join(args.model_path, \"cfg_args\"), 'w') as cfg_log_f:")
    source = replace_once(source, "    for iteration in range(first_iter, opt.iterations + 1):        ",
                          "    for iteration in range(first_iter, opt.iterations + 1):\n        bridge.begin_iteration(iteration)")
    source = replace_once(source, "                if opt.gtmask: # for training with undistorted immerisve image, masking black pixels in undistorted image. ",
                          "                image_hr = image\n                image = bridge.observe(image_hr)\n                if image.shape != gt_image.shape:\n                    raise ValueError('area4 render/LR shape mismatch')\n                if opt.gtmask: # for training with undistorted immerisve image, masking black pixels in undistorted image. ")
    source = replace_once(source, "                if flagems == 1:\n                    if viewpoint_cam.image_name not in lossdiect:",
                          "                loss = bridge.add_teacher_loss(loss, image_hr, viewpoint_cam, ssim)\n                if flagems == 1:\n                    if viewpoint_cam.image_name not in lossdiect:")
    source = replace_once(source, "            # guided sampling step\n",
                          "            # EMS scores use the same observed area-LR domain as reconstruction.\n            image_lr, gt_image_lr = image.detach(), gt_image.detach()\n            image, gt_image = bridge.ems_images(image_lr, gt_image_lr)\n            # Repeat each LR cell to its HR 4x4 footprint. Native HR depth,\n            # projection, 16x16 patch size and EMS topology controller stay intact.\n            # guided sampling step\n")
    source = replace_once(source, "                ssimcurrent = ssim(image.detach(), gt_image.detach()).item()",
                          "                ssimcurrent = ssim(image_lr, gt_image_lr).item()")
    source = replace_once(source, '                    image = render_pkg["render"] * errormask',
                          '                    image = image * errormask')
    source = replace_once(source, '\n\n\nif __name__ == "__main__":',
                          '\n            bridge.end_iteration(iteration, gaussians, loss, opt)\n\n\nif __name__ == "__main__":')
    helper = replace_once(originals["helper_model.py"], "from mmcv.ops import knn",
                          "def knn(*args, **kwargs):\n    # Optional native point thinning is unused by full/preprocesspoints=0.\n    # Keep its original implementation when explicitly requested.\n    from mmcv.ops import knn as native_knn\n    return native_knn(*args, **kwargs)")
    generated = {"adapted_train.py": source, "helper_model.py": helper}
    for name, text in generated.items():
        ast.parse(text, filename=name)
    out.mkdir(parents=True, exist_ok=True)
    patch_text = ""
    for name, text in generated.items():
        origin = "train.py" if name == "adapted_train.py" else name
        target = out / name
        if target.exists() and target.read_text() != text:
            raise FileExistsError(f"Refusing to overwrite different adapter: {target}")
        target.write_text(text)
        patch_text += "".join(difflib.unified_diff(originals[origin].splitlines(True), text.splitlines(True),
                                                fromfile=f"upstream/{origin}", tofile=f"adapter/{name}"))
    (out / "upstream.patch").write_text(patch_text)
    identity = dict(schema="stg_area4_adapter_v1", upstream_commit=COMMIT,
                    original_sha256=EXPECTED, generated_sha256={name: sha(out / name) for name in generated},
                    patch_sha256=sha(out / "upstream.patch"),
                    changes=["LR reconstruction and EMS quality decisions use area4(render HR)",
                             "optional HR teacher on exactly the native batch cameras, weight 0.2",
                             "EMS LR RGB residual repeats to HR grid; native HR depth and 16x16 patches retained",
                             "unused MMCV KNN import becomes lazy; full preprocesspoints=0 unchanged",
                             "per-iteration identity/quality/resource logging"],
                    preserved=["native batch2 same-timestamp sampling and averaged gradients",
                               "native representation, rasterizer, optimizers and densification/EMS controller",
                               "pre-update saving; endpoint performs forward/backward but no optimizer update"],
                    resumable=False)
    (out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.upstream, args.out), indent=2))
