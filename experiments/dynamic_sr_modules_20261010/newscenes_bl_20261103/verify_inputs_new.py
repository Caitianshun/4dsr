"""Owner CPU acceptance of actual synchronized inputs, without Torch."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import *


def verify(host, scene):
    from PIL import Image
    mp, m, rp, _, _ = validate_metadata(scene)
    files = []
    for row in m["observations"]:
        levels = ["lr"] if row["split"] == "train" else []
        if row["split"] == "test" or (row["split"] == "train" and row["frame_index"] % 10 == 0):
            levels.append("hr")
        for level in levels:
            p = mp.parent / row[level + "_path"]
            require(sha(p) == row[level + "_sha256"], "Input PNG SHA mismatch: " + rel(p))
            with Image.open(p) as im:
                require(im.format == "PNG" and im.mode == "RGB"
                        and list(im.size) == m["resolutions"][level], "PNG contract mismatch")
                im.verify()
            files.append(dict(path=rel(p), sha256=row[level + "_sha256"], bytes=p.stat().st_size))
    if host == "a100":
        ip = mp.parent / m["initialization"]["npz_path"]
        require(sha(ip) == m["initialization"]["npz_sha256"], "Shared initialization SHA mismatch")
        with np.load(ip, allow_pickle=False) as z:
            points, colors = z["points"], z["colors"]
            require(points.shape == colors.shape == (m["initialization"]["point_count"], 3)
                    and np.isfinite(points).all() and np.isfinite(colors).all(), "Invalid shared point cloud")
        files.append(dict(path=rel(ip), sha256=sha(ip), bytes=ip.stat().st_size))
    original = m["physical_lr_derivation"]["original_manifest"]
    require(sha(original["path"]) == original["sha256"], "Scorer's original manifest SHA mismatch")
    result = dict(schema="newscene_synchronized_inputs_v1", status="passed", host=host, scene=scene,
                  manifest=identity(mp), registry=identity(rp), files=files,
                  score_HR_verified=True, training_HR_used_for_updates=False,
                  heldout_LR_read=False, torch_imported="torch" in sys.modules)
    require(not result["torch_imported"], "CPU input acceptance imported Torch")
    write_new(OUT / f"preflight/{host}/{scene}/inputs.json", result)
    print(__import__("json").dumps(dict(status="passed", scene=scene, host=host, files=len(files),
                                      bytes=sum(f["bytes"] for f in files))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", choices=REMOTE, required=True)
    p.add_argument("--scene", choices=SCENES, required=True)
    a = p.parse_args()
    verify(a.host, a.scene)


if __name__ == "__main__":
    main()
