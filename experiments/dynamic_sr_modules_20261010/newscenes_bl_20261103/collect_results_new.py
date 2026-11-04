"""Validate scored outputs and list actual result files for later download.

Reads only the new result tree; does not transfer, submit jobs or delete PNGs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import *


def collect(method, scene):
    _, m, _, _, _ = validate_metadata(scene)
    base = OUT / method / scene
    selected = []
    for scope in ("test", "train"):
        render, score = base / ("renders_" + scope), base / ("scores_" + scope)
        result, index = read(score / "summary.json"), read(render / "render_index.json")
        count = 300 if scope == "test" else 30 * len(m["splits"]["train"])
        require(result["status"] == "completed" and result["method"] == METHOD_NAMES[method]
                and result["scene"] == scene and result["manifest_sha256"] == sha(manifest_path(scene)),
                "Score identity mismatch")
        require(result["scope"] == scope and result["images"] == count
                and result["render_index_sha256"] == sha(render / "render_index.json"),
                "Score count/render-index binding mismatch")
        require(index["manifest_sha256"] == sha(manifest_path(scene)) and index["scope"] == scope,
                "Render identity mismatch")
        expected = {(c, f) for c in m["splits"][scope]
                    for f in (range(300) if scope == "test" else range(0, 300, 10))}
        actual = {(r["camera_id"], r["frame"]) for r in index["records"]}
        require(actual == expected and len(index["records"]) == count, "Render selection mismatch")
        with (score / "per_view.csv").open() as stream:
            metrics = list(csv.DictReader(stream))
        metric_keys = {(r["camera_id"], int(r["frame"])) for r in metrics}
        require(len(metrics) == count and metric_keys == expected, "Incomplete/duplicate per-view metrics")
        for name in ("summary.json", "per_view.csv", "area4_selection.json"):
            selected.append(score / name)
        selected.append(render / "render_index.json")
        for optional in ("render_receipt.json", "cleanup_receipt.json"):
            if (render / optional).is_file():
                selected.append(render / optional)
        for cam in m["splits"][scope]:
            frames = (0, 150, 299) if scope == "test" else (0, 150, 290)
            lookup = {r["path"]: r for r in index["records"]}
            for frame in frames:
                name = f"{cam}/{frame:04d}.png"
                require(sha(render / name) == lookup[name]["render_sha256"], "Sample PNG SHA mismatch")
                selected.append(render / name)
    record = dict(schema="newscene_result_acceptance_v1", method=method, scene=scene,
                  status="passed", files=[dict(path=rel(p), sha256=sha(p), bytes=p.stat().st_size)
                                          for p in selected])
    write_new(base / "result_acceptance.json", record)
    write_new(OUT / "sync/download_actual" / f"{method}_{scene}.json",
              dict(version=1, files=[dict(path=rel(p)) for p in selected + [base / "result_acceptance.json"]]))
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=METHOD_NAMES)
    p.add_argument("--scene", required=True, choices=SCENES)
    a = p.parse_args()
    r = collect(a.method, a.scene)
    print(__import__("json").dumps(dict(status=r["status"], files=len(r["files"]))))


if __name__ == "__main__":
    main()
