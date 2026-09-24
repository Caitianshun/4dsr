"""Publish the allowlisted non-HOI SR source from this research workspace.

The private workspace contains datasets and experimental output and is not a Git
checkout. This command copies selected text source files to a separate clone.
Run without --publish to inspect the selected files; --publish commits and pushes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


SOURCE = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = SOURCE.parent / "4dsr-github"
REMOTE_URLS = {
    "git@github.com:Caitianshun/4dsr.git",
    "https://github.com/Caitianshun/4dsr.git",
}
EXPERIMENTS = (
    "dynamic_sr_20260918",
    "dynamic_sr_20260919",
    "dynamic_sr_20260920",
    "dynamic_sr_20260921",
    "dynamic_sr_20260923",
    "dynamic_sr_detail_supervision_20260924",
    "dynamic_sr_geometry_residual_20260924",
    "dynamic_sr_motion_bound_20260923",
    "dynamic_sr_scene_residual_20260923",
    "dynamic_sr_soft_motion_20260924",
    "dynamic_sr_surface_20260923",
    "sr4d_20260922",
)
DEPLOYMENTS = (
    "a100_20260920",
    "cts_20260921",
    "soft_motion_a100_20260924",
    "sr4d_20260922",
)
MAX_SOURCE_BYTES = 512_000
MANIFEST = ".publication-manifest.json"
GITIGNORE = """# Defense in depth; publication also uses a strict source allowlist.
data/
output/
weight/
paper/
tmp/
logs/
newdoc/
src/
configs/
tests/
third_party/
__pycache__/
*.pyc
*.pt
*.pth
*.ckpt
*.safetensors
*.npy
*.npz
*.png
*.jpg
*.jpeg
*.mp4
*.tar
*.tar.gz
*.zip
*.log
*.jsonl
"""
SECRET_MARKERS = (
    "-----BEGIN " + "PRIVATE KEY-----",
    "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
    "github_" + "pat_",
    "gh" + "p_",
)


def git(target: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(target), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def add_file(selected: dict[str, bytes], source_file: Path, public_path: str) -> None:
    if source_file.is_symlink() or not source_file.is_file():
        raise ValueError(f"Not a regular source file: {source_file}")
    content = source_file.read_bytes()
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError(f"Unexpectedly large source file: {source_file}")
    decoded = content.decode("utf-8")
    for marker in SECRET_MARKERS:
        if marker in decoded:
            raise ValueError(f"Possible credential marker in {source_file}")
    if public_path in selected:
        raise ValueError(f"Duplicate publication path: {public_path}")
    selected[public_path] = content


def select_sources() -> dict[str, bytes]:
    selected: dict[str, bytes] = {}
    for name in EXPERIMENTS:
        folder = SOURCE / "experiments" / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.iterdir()):
            if path.is_file() and not path.is_symlink() and path.suffix == ".py":
                add_file(selected, path, path.relative_to(SOURCE).as_posix())
    for name in DEPLOYMENTS:
        folder = SOURCE / "deployment" / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix in {".py", ".sh"} or path.name in {
                "requirements.txt", "upstream_commit.txt"
            }:
                add_file(selected, path, path.relative_to(SOURCE).as_posix())

    # The original SR4D local patch also edits its README with an HOI result
    # reference. Publish only the code hunks used by this non-HOI comparison.
    original_patch = (SOURCE / "deployment/sr4d_20260922/local_source.patch").read_text()
    chunks = re.split(r"(?=^diff --git )", original_patch, flags=re.MULTILINE)
    code_chunks = [part for part in chunks if part.startswith("diff --git ")
                   and not part.startswith("diff --git a/README.md ")
                   and not part.startswith("diff --git a/.gitignore ")]
    if not code_chunks or any("HODome" in part or "hoi_v1" in part for part in code_chunks):
        raise ValueError("SR4D code-only patch requires manual scope review")
    patch_bytes = "".join(code_chunks).encode("utf-8")
    if len(patch_bytes) > MAX_SOURCE_BYTES:
        raise ValueError("SR4D code patch is unexpectedly large")
    selected["deployment/sr4d_20260922/general_sr_source.patch"] = patch_bytes

    add_file(selected, SOURCE / "scripts/summarize_quality_headroom_20260923.py",
             "scripts/summarize_quality_headroom_20260923.py")
    add_file(selected, SOURCE / "scripts/publish_generic_sr.py",
             "scripts/publish_generic_sr.py")
    add_file(selected, SOURCE / "PUBLIC_README.md", "README.md")
    add_file(selected, SOURCE / "PUBLIC_PROJECT_GUIDELINES.md", "PROJECT_GUIDELINES.md")
    selected[".gitignore"] = GITIGNORE.encode("utf-8")
    return selected


def safe_destination(target: Path, relative: str) -> Path:
    candidate = (target / relative).resolve()
    if not candidate.is_relative_to(target.resolve()):
        raise ValueError(f"Path outside publication clone: {relative}")
    return candidate


def publish(target: Path, selected: dict[str, bytes], message: str) -> None:
    if not (target / ".git").is_dir():
        raise ValueError(f"Not a Git clone: {target}")
    if git(target, "remote", "get-url", "origin") not in REMOTE_URLS:
        raise ValueError("Origin is not Caitianshun/4dsr")
    if git(target, "branch", "--show-current") != "main":
        raise ValueError("Publication clone must be on main")
    if git(target, "status", "--porcelain"):
        raise ValueError("Publication clone has uncommitted changes")
    git(target, "fetch", "origin", "main")
    git(target, "pull", "--ff-only", "origin", "main")

    old_manifest_file = target / MANIFEST
    old_paths: set[str] = set()
    if old_manifest_file.exists():
        old_paths = set(json.loads(old_manifest_file.read_text())["sha256"])
    for relative in sorted(old_paths - selected.keys()):
        old_file = safe_destination(target, relative)
        if old_file.is_file():
            old_file.unlink()

    for relative, content in selected.items():
        destination = safe_destination(target, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or destination.read_bytes() != content:
            destination.write_bytes(content)
    manifest_content = json.dumps(
        {"scope": "non-HOI SR code and necessary text configuration",
         "sha256": {name: digest(content) for name, content in sorted(selected.items())}},
        ensure_ascii=False, indent=2,
    ) + "\n"
    old_manifest_file.write_text(manifest_content)

    # Detect files changed in the source workspace while the copy was made.
    if select_sources() != selected:
        raise RuntimeError("Source changed during publication; rerun after it settles")
    git(target, "add", "-A")
    staged = set(git(target, "diff", "--cached", "--name-only").splitlines())
    permitted = set(selected) | old_paths | {MANIFEST}
    unexpected = staged - permitted
    if unexpected:
        git(target, "reset")
        raise ValueError(f"Unexpected staged paths: {sorted(unexpected)}")
    if not staged:
        print("No source changes to publish; remote main is current.")
        return
    git(target, "commit", "-m", message)
    head = git(target, "rev-parse", "HEAD")
    git(target, "push", "origin", "main")
    remote_head = git(target, "ls-remote", "origin", "refs/heads/main").split()[0]
    if remote_head != head:
        raise RuntimeError(f"Push verification failed: local {head}, remote {remote_head}")
    print(f"Published {len(staged)} changed paths at {head}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="copy, commit, push, verify")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--message", default="Publish current non-HOI dynamic SR code")
    args = parser.parse_args()
    selected = select_sources()
    total = sum(len(item) for item in selected.values())
    print(f"Selected {len(selected)} source/configuration files ({total:,} bytes)")
    if not args.publish:
        for name in sorted(selected):
            print(name)
        return
    publish(args.target, selected, args.message)


if __name__ == "__main__":
    main()
