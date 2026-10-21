"""Verify native endpoints and index identities after training and rendering."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--level", choices=["lr", "hr"], required=True)
    a = p.parse_args()
    manifest_hash = sha(a.manifest)
    stages = []
    for n, endpoint in [(1, 40000), (2, 50000)]:
        path = a.run / f"stage{n}.json"
        state = json.loads(path.read_text())
        assert state["status"] == "completed" and state["train_level"] == a.level
        assert state["endpoint"] == endpoint and state["manifest"]["sha256"] == manifest_hash
        assert not state["heldout_pixels_used"] and state["native_schedule_preserved"]
        ck = a.run / "model" / f"chkpnt{endpoint}.pth"
        assert sha(ck) == state["final_checkpoint"]["sha256"]
        stages.append(dict(path=str(path), sha256=sha(path), checkpoint=str(ck), checkpoint_sha256=sha(ck),
                           elapsed_seconds=state["elapsed_seconds"], peak_reserved=state.get("torch_peak_reserved_bytes")))
    rendering = json.loads((a.run / "renders_hr/render_receipt.json").read_text())
    assert rendering["frames"] == 300 and rendering["finite_before_clip_quantization_passed_frames"] == 300
    for entry in rendering["records"]:
        assert sha(a.run / "renders_hr" / entry["path"]) == entry["sha256"]
    index = dict(status="native_endpoints_and_300_renders_verified", train_level=a.level,
                 manifest_sha256=manifest_hash, stages=stages, heldout_used_for_training=False,
                 metrics_status="shared_A100_complete" if (a.run / "metrics/summary.json").exists() else "shared_A100_evaluation_pending",
                 renders_receipt_sha256=sha(a.run / "renders_hr/render_receipt.json"))
    if (a.run / "metrics/summary.json").exists():
        index["metrics"] = json.loads((a.run / "metrics/summary.json").read_text())
    (a.run / "result_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(dict(status=index["status"], metrics_status=index["metrics_status"])))


if __name__ == "__main__":
    main()
