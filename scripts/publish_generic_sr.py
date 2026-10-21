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
    "dynamic_sr_cem_area_20261020",
    "dynamic_sr_20260918",
    "dynamic_sr_20260919",
    "dynamic_sr_20260920",
    "dynamic_sr_20260921",
    "dynamic_sr_20260923",
    "dynamic_sr_detail_supervision_20260924",
    "dynamic_sr_controlled_headroom_20260926",
    "dynamic_sr_view_recovery_20260926",
    "dynamic_sr_prior_guidance_20260927",
    "dynamic_sr_evidence_repair_20260927",
    "dynamic_sr_conflict_probe_20260927",
    "dynamic_sr_covariance_probe_20260927",
    "dynamic_sr_dynamic_validation_20260927",
    "dynamic_sr_attribute_routing_20260928",
    "dynamic_sr_sync_multiview_20260928",
    "dynamic_sr_temporal_prior_20260928",
    "dynamic_sr_prior_diagnosis_20260929",
    "dynamic_sr_same_observation_20260930",
    "dynamic_sr_confidence_geometry_20261006",
    "dynamic_sr_multiview_footprint_20261007",
    "dynamic_sr_cull_retrain_20261009",
    "dynamic_sr_highres_20261011",
    "dynamic_sr_physlr_20261016",
    "dynamic_sr_geoprobe_20261017",
    "dynamic_sr_geometry_residual_20260924",
    "dynamic_sr_motion_bound_20260923",
    "dynamic_sr_multi4d_20260924",
    "dynamic_sr_multi4d_area_20261017",
    "dynamic_sr_stg_area_20261018",
    "dynamic_sr_scene_residual_20260923",
    "dynamic_sr_soft_motion_20260924",
    "dynamic_sr_surface_20260923",
    "sr4d_20260922",
)
DEPLOYMENTS = (
    "dynamic_sr_cem_backbone_20261020",
    "dynamic_lr_baselines_20261020",
    "a100_20260920",
    "cts_20260921",
    "soft_motion_a100_20260924",
    "sr4d_20260922",
    "readonly_hr_a100_20261011",
    "resume_verified_sync_20261011",
    "runtime_env_resume_20261011",
    "complete_startup_eval_20261011",
    "resume_long_storage_20261011",
    "adopt_existing_dag_20261011",
    "resume_existing_completed_dag_20261011",
    "return_completed_dag_20261011",
    "return_terminal_inventory_20261011",
)
EXPERIMENT_INSTRUCTIONS = (
    "deployment/dynamic_sr_cem_backbone_20261020/README.md",
    "experiments/dynamic_sr_cem_area_20261020/README.md",
    "deployment/dynamic_lr_baselines_20261020/README.md",
    "deployment/readonly_hr_a100_20261011/README.md",
    "deployment/resume_verified_sync_20261011/README.md",
    "deployment/runtime_env_resume_20261011/README.md",
    "deployment/complete_startup_eval_20261011/README.md",
    "deployment/resume_long_storage_20261011/README.md",
    "deployment/adopt_existing_dag_20261011/README.md",
    "deployment/resume_existing_completed_dag_20261011/README.md",
    "deployment/return_completed_dag_20261011/README.md",
    "deployment/return_terminal_inventory_20261011/README.md",
    "experiments/dynamic_sr_cull_retrain_20261009/README.md",
    "experiments/dynamic_sr_highres_20261011/README.md",
    "experiments/dynamic_sr_highres_20261011/README_screening_runner.md",
    "experiments/dynamic_sr_highres_20261011/README_local_orchestrator.md",
    "experiments/dynamic_sr_multi4d_area_20261017/README.md",
    "experiments/dynamic_sr_stg_area_20261018/README.md",
    "experiments/dynamic_sr_multiview_footprint_20261007/full_registered_README.md",
    "experiments/dynamic_sr_multiview_footprint_20261007/full_support_prepare_registered_README.md",
    "experiments/dynamic_sr_multiview_footprint_20261007/full_temporal_diagnostics_README.md",
    "experiments/dynamic_sr_multiview_footprint_20261007/full_overlap_schedule_README.md",
    "experiments/dynamic_sr_multiview_footprint_20261007/full_held_confirmation_README_v2.md",
    "experiments/dynamic_sr_multiview_footprint_20261007/full_held_confirmation_README_v3.md",
    "experiments/dynamic_sr_multiview_footprint_20261007/full_held_confirmation_local_RTX3090_README_v1.md",
)
HIGHRES_REQUIRED_TEXT = (
    # Fixed-checkpoint S1 supplementary evaluation tools, 2026-10-21.
    "experiments/dynamic_sr_mvfb_20261020/evaluate_train.py",
    "experiments/dynamic_sr_mvfb_20261020/dump_eval_sequence.py",
    "experiments/dynamic_sr_cem_area_20261020/experiment_config.json",
    "experiments/dynamic_sr_cem_area_20261020/area4_interface_v1.patch",
    "experiments/dynamic_sr_highres_20261011/datasets/deskgames_cube.json",
    "experiments/dynamic_sr_highres_20261011/datasets/long360.json",
    "experiments/dynamic_sr_modules_20261010/modsr.py",
    "experiments/dynamic_sr_modules_20261010/refs.py",
    "experiments/dynamic_sr_native_20261010/register_native.py",
    "experiments/dynamic_sr_native_20261010/teacher_native.py",
    # dependencies of experiments/dynamic_sr_physlr_20261016 (2026-10-17)
    "experiments/dynamic_sr_modules_20261010/srfix.py",
    "experiments/dynamic_sr_compress_20261013/compress.py",
    "experiments/dynamic_sr_compress_20261013/trainview_eval.py",
    # geometry / calibration probes and their evaluation dependencies (2026-10-17)
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/appearance_refit_v2.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/calibsr.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/calibsr_v2.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/geosr.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/group_eval_bands.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/perturbsr.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/register_probe.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_calib_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_calib_remote_v2.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_calib_remote_v3.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_geo_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_perturb_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_refit_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_refit_remote_v2.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_refit_remote_v3.py",
    # area4 SR-stage module kit (session 4dsr-e9, 2026-10-18): unified spatial correction = per-camera warp field (modkit2);
    # the modkit chain, its remote drivers, the evaluation and the diagnostics cited in docs/sr_modules_area4_2026-10-17
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/modkit.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/modkit2.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/modkit4.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/modkit5.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/modkit6.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/modkit8.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/modkit9.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/modkit10.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_modkit_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_modkit2_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_modkit3_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_modkit4_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_modkit5_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_modkit6_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_modkit7_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_modkit8_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_modkit9_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/run_modkit10_remote.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/eval_full.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/vsel_eval.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/depth_diag.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/gauge_probe.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/epipolar_check.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/temporal_diag_g.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/camres_probe.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/camres_probe2.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/camres_probe3.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/testcolour_probe.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/testcolour_probe2.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/testcolour_probe3.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/region_eval.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/tofflr.py",
    "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/tofflr2.py",
    "experiments/dynamic_sr_cull_retrain_20261009/selfens_20261016/group_eval.py",
    "experiments/dynamic_sr_cull_retrain_20261009/variance_20261014/ens_eval_view.py",
    # SR-prior motion distillation (session 4dsr-25, 2026-10-21): teacher-video RAFT flow cache, Gaussian-flow supervision kit
    # on the w4 recipe (modkit13 wraps modkit6), arm runner, two-lane GPU runner, smoke check and the Cut07 lane lists
    "experiments/dynamic_sr_modules_20261010/motion_20261021/flowcache.py",
    "experiments/dynamic_sr_modules_20261010/motion_20261021/modkit13.py",
    "experiments/dynamic_sr_modules_20261010/motion_20261021/run_motion.py",
    "experiments/dynamic_sr_modules_20261010/motion_20261021/lanes.py",
    "experiments/dynamic_sr_modules_20261010/motion_20261021/smoke_check.py",
    "experiments/dynamic_sr_modules_20261010/motion_20261021/jobs/cut3090a_laneA.txt",
    "experiments/dynamic_sr_modules_20261010/motion_20261021/jobs/cut3090a_laneB.txt",
    # third-party notice for experiments/dynamic_sr_physlr_20261016/hat_arch_local.py (derived from XPixelGroup/HAT,
    # Apache-2.0; notice plus the verbatim license text, 2026-10-18)
    "experiments/dynamic_sr_physlr_20261016/THIRD_PARTY_HAT_LICENSE.txt",
)
DYNAMIC_LR_SOURCE_MANIFESTS = (
    "deployment/dynamic_lr_baselines_20261020/methods/wu_multi/public_source_files.json",
    "deployment/dynamic_lr_baselines_20261020/methods/strict4d/public_source_files.json",
    "deployment/dynamic_lr_baselines_20261020/methods/localdygs_cem/public_source_files.json",
    "deployment/dynamic_lr_baselines_20261020/methods/cem_a100_build/public_source_files.json",
    "deployment/dynamic_lr_baselines_20261020/methods/cts_cem_followon/public_source_files.json",
    "deployment/dynamic_lr_baselines_20261020/methods/cts_local_cut_followon/public_source_files.json",
)
FULL_HELD_SOURCE_NAMES = {
    "full_held_local_RTX3090_scope_v1.py",
    "full_held_native_supervisor_local_RTX3090_v1.py",
    "full_held_coarse_manager_local_RTX3090_v1.py",
    "full_held_confirmation_assets_local_RTX3090_v1.py",
    "full_held_operator_registered_local_RTX3090_v1.py",
    "full_held_pair_coordinator_local_RTX3090_v1.py",
    "full_held_coarse_manager.py",
    "full_held_coarse_manager_GPU1_v2.py",
    "full_held_confirmation_assets_v2.py",
    "full_held_confirmation_assets_v4.py",
    "full_held_confirmation_bridge.py",
    "full_held_confirmation_checks.py",
    "full_held_confirmation_contract.py",
    "full_held_confirmation_contract_v2.py",
    "full_held_native_acceptance.py",
    "full_held_native_schedule_materializer.py",
    "full_held_native_supervisor.py",
    "full_held_native_supervisor_GPU1.py",
    "full_held_operator_registered.py",
    "full_held_operator_registered_v2.py",
    "full_held_pair_coordinator.py",
    "full_held_pair_coordinator_v2.py",
    "full_held_parent_overlap.py",
    "full_held_temporal_diagnostics.py",
}
RETAINED_FULL_HELD_PROTOTYPES = {
    "full_held_coarse_manager_GPU1.py",
    "full_held_confirmation_assets.py",
    "full_held_confirmation_assets_v3.py",
}
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
                if name == "dynamic_sr_multiview_footprint_20261007" and path.name.startswith("full_held_"):
                    if path.name in RETAINED_FULL_HELD_PROTOTYPES:
                        continue
                    if path.name not in FULL_HELD_SOURCE_NAMES:
                        raise ValueError(f"Unreviewed held source is not allowlisted: {path.name}")
                add_file(selected, path, path.relative_to(SOURCE).as_posix())
    held_prefix = "experiments/dynamic_sr_multiview_footprint_20261007/"
    missing_held = {held_prefix + name for name in FULL_HELD_SOURCE_NAMES} - selected.keys()
    if missing_held:
        raise ValueError(f"Missing reviewed held sources: {sorted(missing_held)}")
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
    for relative in EXPERIMENT_INSTRUCTIONS:
        add_file(selected, SOURCE / relative, relative)
    for relative in HIGHRES_REQUIRED_TEXT:
        add_file(selected, SOURCE / relative, relative)
    # Method owners review only small source/configuration files. Generated
    # scientific results and complete third-party/runtime copies stay private.
    baseline_prefix = "deployment/dynamic_lr_baselines_20261020/methods/"
    forbidden_parts = {"logs", "jobs", "prepared_original", "__pycache__",
                       "frozen_execution_helpers", "frozen_execution_helpers_v2"}
    for manifest_name in DYNAMIC_LR_SOURCE_MANIFESTS:
        manifest = json.loads((SOURCE / manifest_name).read_text())
        for record in manifest["files"]:
            relative = record["path"]
            path = Path(relative)
            if (not relative.startswith(baseline_prefix) or path.is_absolute()
                    or ".." in path.parts or forbidden_parts.intersection(path.parts)
                    or path.suffix not in {".py", ".sh", ".md", ".patch", ".json", ".txt"}
                    or path.name == "adapted_train.py"):
                raise ValueError(f"Not an approved baseline source path: {relative}")
            if digest((SOURCE / path).read_bytes()) != record["sha256"]:
                raise ValueError(f"Baseline source changed since its owner review: {relative}")
            add_file(selected, SOURCE / path, relative)

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


def publish(target: Path, selected: dict[str, bytes], message: str, prefixes: list[str] | None = None) -> None:
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
    # A scoped publication preserves the already-published bytes outside this
    # task. Other research sessions may be actively editing those local files.
    def in_scope(name: str) -> bool:
        return not prefixes or any(name == p or name.startswith(p.rstrip('/') + '/') for p in prefixes)
    source_selected = {name: content for name, content in selected.items() if in_scope(name)}
    if prefixes:
        if not source_selected:
            raise ValueError("No allowlisted source matches the requested publication scope")
        selected = {name: safe_destination(target, name).read_bytes() for name in old_paths if not in_scope(name)}
        selected.update(source_selected)
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
    if {name: content for name, content in select_sources().items() if in_scope(name)} != source_selected:
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
    parser.add_argument("--include-prefix", action="append", help="Publish only these allowlisted paths; retain remote bytes elsewhere")
    args = parser.parse_args()
    selected = select_sources()
    total = sum(len(item) for item in selected.values())
    print(f"Selected {len(selected)} source/configuration files ({total:,} bytes)")
    if not args.publish:
        for name in sorted(selected):
            print(name)
        return
    publish(args.target, selected, args.message, args.include_prefix)


if __name__ == "__main__":
    main()
