"""Copy the five pinned owned helpers into a new private snapshot, CPU only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EXPECTED = {
    "run_author.py": "9bb2415bfd562e7d02c18bc7bdf30e06b45af9ee43f4a1fb6e773672595ff114",
    "render_hr.py": "cea860627b5d7ba779cc7476221d179a276c08ffeec9fcfc2d571d282fe284bf",
    "index_result.py": "b2ec2c31d2a1dccb3aa6423e77c1e1cbbec2e9f08ee2bcc3ba047b1eefdba1c2",
    "check_gpu_environment.py": "21bbfe78f593c38113628e43124f00ed5e6b82bc49c14754e609481b53d4c3e2",
    "audit_transferred_initialization.py": "defb3840e5684da3f7820f2353561ccb87abf47c57e2ad168005c1ee2a1b8373",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE.parent / "localdygs_cem")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    source.relative_to(ROOT)
    output.relative_to(ROOT)
    if output.exists():
        raise SystemExit("Refusing an existing snapshot; never overwrite a frozen execution helper")
    files = []
    payloads = {}
    for name, expected in EXPECTED.items():
        path = source / name
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(("Canonical helper differs from this frozen cohort", name, actual, expected))
        ast.parse(content, filename=str(path))
        payloads[name] = content
        files.append(dict(path=str((output / name).relative_to(ROOT)), sha256=actual,
                          original_source_path=str(path.relative_to(ROOT))))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".CEM-helper-snapshot-", dir=output.parent))
    try:
        for name, content in payloads.items():
            (staging / name).write_bytes(content)
        identity = dict(status="frozen_no_GPU_submission", CPU_only=True,
                        generator=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),
                                       sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), files=files)
        (staging / "frozen_helper_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        # The new name appears only after all pinned files have been written.
        staging.rename(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps(dict(status="frozen_no_GPU_submission", directory=str(output), files=len(files)), indent=2))


if __name__ == "__main__":
    main()
