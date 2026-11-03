"""CPU synthetic/fake-SSH contract checks; never stages production HRref assets."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("returned_image_assets.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location("returned_images_CPU_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual_remote_verify = module.remote_verify
    results, manifests = [], []
    with tempfile.TemporaryDirectory(prefix="returned_HRref_images_CPU_QA_") as directory:
        base = Path(directory)
        for scene in module.SCENES:
            local, remote = base / scene / "local", base / scene / "fake_a100"
            local.mkdir(parents=True)
            remote.mkdir()
            module.ROOT = local
            for relative in [module.HERE / "build_wu_eval_job.py", module.HERE / "verify_assets.py", Path("experiments/dynamic_sr_cem_area_20261020/native_stage.py"),
                             Path("experiments/dynamic_sr_cem_area_20261020/render_hr.py"), Path("experiments/dynamic_sr_cem_area_20261020/sr_stage.py")]:
                (local / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, local / relative)
            builder, verifier = module.contracts()
            manifest = Path("data/dynamic_sr/full_time_20261007/prepared") / (scene + "_area4") / "manifest_train_ready_seed20261007.json"
            (local / manifest).parent.mkdir(parents=True)
            shutil.copy2(ROOT / manifest, local / manifest)
            data = json.loads((local / manifest).read_text())
            manifest_hash = builder.sha(local / manifest)
            manifests.append(dict(scene=scene, path=str(manifest), sha256=manifest_hash, actual_workspace_manifest=True))
            test = [row for row in data["observations"] if row["split"] == "test"]
            assert len(test) == 300
            for row in test:
                relative = manifest.parent / row["hr_path"]
                (local / relative).parent.mkdir(parents=True, exist_ok=True)
                os.link(ROOT / relative, local / relative)
            run = module.BATCH / "runs" / (scene + "_native_HRref_v2")
            full = local / run
            (full / "renders_hr").mkdir(parents=True)
            synthetic = full / "renders_hr/0000.png"
            Image.new("RGB", (1344, 1008), (32, 64, 96)).save(synthetic)
            prediction_hash = builder.sha(synthetic)
            for frame in range(1, 300):
                os.link(synthetic, full / f"renders_hr/{frame:04d}.png")
            records = [dict(frame=frame, path=f"{frame:04d}.png", sha256=prediction_hash, raw_path=None, raw_sha256=None) for frame in range(300)]
            source_files = {str(relative): builder.sha(local / relative) for relative in [Path("experiments/dynamic_sr_cem_area_20261020/native_stage.py"),
                            Path("experiments/dynamic_sr_cem_area_20261020/render_hr.py"), Path("experiments/dynamic_sr_cem_area_20261020/sr_stage.py")]}

            def put(name, value):
                target = full / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(value))
                return builder.sha(target)

            job = dict(protocol="CEM_area4_backbone_control_v1", arm="native_HRref", scene=scene, job_id="synthetic-cpu-contract-only", out=str(run),
                       input_formation="area4", heldout_used_for_training=False, commands=[dict(name="synthetic_not_training")],
                       required_files=[dict(path=path, sha256=sha) for path, sha in source_files.items()])
            job_hash = put("job.json", job)
            completion = dict(status="completed", job_id=job["job_id"], job_sha256=job_hash, arm="native_HRref", scene=scene,
                              stages=[dict(returncode=0)], synthetic_CPU_contract_only=True)
            put("status.json", completion)
            put("complete.json", completion)
            stages = []
            for number, endpoint, checkpoint_hash in [(1,40000,"a"*64),(2,50000,"b"*64)]:
                value = dict(status="completed", stage=number, endpoint=endpoint, train_level="hr", train_HR_used=True, heldout_pixels_used=False,
                             native_schedule_preserved=True, parent_added_affine=False, parent_added_warp=False, manifest=dict(sha256=manifest_hash),
                             entry_sha256=source_files["experiments/dynamic_sr_cem_area_20261020/native_stage.py"], author_optimizer_restored=number==2,
                             source=dict(LR_initialization_pointcloud_sha256="c"*64), final_checkpoint=dict(path="SYNTHETIC_NO_MODEL_BYTES",sha256=checkpoint_hash),
                             start_checkpoint=dict(sha256="a"*64) if number==2 else None)
                stage_hash = put(f"stage{number}.json", value)
                stages.append(dict(path=f"/synthetic/run/stage{number}.json",sha256=stage_hash,checkpoint_sha256=checkpoint_hash))
            rendering = dict(status="completed", kind="native", iteration=50000,camera="cam00",scope="heldout_300",scene=scene,
                frames=300,finite_before_clip_quantization_passed_frames=300,frame_indices=list(range(300)),raw_prediction_shape=[3,1008,1344],
                quantization="np.rint(np.clip(CHW_to_HWC_float32,0,1)*255).astype(np.uint8); PIL RGB PNG",manifest=dict(sha256=manifest_hash),
                checkpoint=dict(sha256="b"*64),training_receipt=dict(sha256=builder.sha(full/"stage2.json")),heldout_pixels_read=False,training_pixels_read=False,
                renderer_source_sha256=source_files["experiments/dynamic_sr_cem_area_20261020/render_hr.py"],camera_helper_sha256=source_files["experiments/dynamic_sr_cem_area_20261020/sr_stage.py"],
                colour_affine="identity",warp="identity",records=records)
            render_hash = put("renders_hr/render_receipt.json", rendering)
            put("result_index.json",dict(status="native_endpoints_and_300_renders_verified",train_level="hr",manifest_sha256=manifest_hash,
                heldout_used_for_training=False,stages=stages,renders_receipt_sha256=render_hash))
            # The fake SSH transport hashes actual files in an isolated local
            # directory and reports canonical A100 provenance ONLY inside this
            # temporary synthetic test. No fake receipt escapes this directory.
            calls = dict(probe=0, rsync=0, remote_verify=0, fetch=0)

            def fake_probe(payload):
                calls["probe"] += 1
                rows, missing, conflicts = {}, [], []
                for name, expected in payload["files"].items():
                    p = remote / name
                    if not p.exists():
                        rows[name] = dict(status="missing");missing.append(name)
                    else:
                        actual = builder.sha(p)
                        okay = p.stat().st_size == expected["bytes"] and actual == expected["sha256"]
                        rows[name] = dict(status="verified" if okay else "conflict",sha256=actual,bytes=p.stat().st_size,stable_during_read=True)
                        if not okay:conflicts.append(name)
                return dict(status="probed",root=module.REMOTE_ROOT,files=rows,missing=missing,conflicts=conflicts,
                            reserved_existing=[name for name in payload["must_be_absent"] if (remote/name).exists()])

            def fake_stage(missing):
                calls["rsync"] += 1
                for relative in missing:
                    target = remote / relative
                    assert not target.exists(), "missing-only transfer may not overwrite"
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(local/relative,target)

            def fake_verify(inventory_path, receipt_path, source_sha):
                calls["remote_verify"] += 1
                assert builder.sha(remote/module.HERE/"verify_assets.py") == source_sha
                value = verifier.verify(remote,remote/inventory_path,2)
                value["root"] = module.REMOTE_ROOT
                module.new_json(remote/receipt_path,value)
                return subprocess.CompletedProcess(["FAKE_SSH_CPU_ONLY"],0,json.dumps(dict(status=value["status"])),"")

            def fake_fetch(relative):
                calls["fetch"] += 1
                assert not (local/relative).exists()
                shutil.copy2(remote/relative,local/relative)

            module.remote_probe, module.stage_missing, module.remote_verify, module.fetch_receipt = fake_probe,fake_stage,fake_verify,fake_fetch
            # One actual source HR file is already present and must be retained.
            existing = manifest.parent/test[0]["hr_path"]
            (remote/existing).parent.mkdir(parents=True)
            shutil.copy2(local/existing,remote/existing)
            before = (remote/existing).stat()
            assert module.execute(SimpleNamespace(scene=scene,run=None,stage=False)) == 0
            assert calls == dict(probe=1,rsync=0,remote_verify=0,fetch=0)
            assert module.execute(SimpleNamespace(scene=scene,run=None,stage=True)) == 0
            after = (remote/existing).stat()
            assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
            named = module.paths(scene)
            _,_,receipt,inventory = builder.asset_gate(named["receipt"])
            assert receipt["file_count"] == receipt["verified_count"] == 608
            assert receipt["failure_count"]==0 and receipt["all_listed_file_bytes_rehashed"]
            builder.hrref(SimpleNamespace(scene=scene,run=run),builder.Required(),inventory)
            assert not list(local.rglob("*.pth")) and not list(remote.rglob("*.pth"))
            assert not list(local.rglob("*.pt")) and not list(remote.rglob("*.pt"))
            results.append(dict(scene=scene,test="probe_stage_and_exact_builder_receipt",passed=True,calls=calls.copy(),files=608,existing_inode_mtime_preserved=True,model_bytes=0))
            try:module.execute(SimpleNamespace(scene=scene,run=None,stage=True))
            except AssertionError:results.append(dict(scene=scene,test="second_stage_rejected",passed=True))
            else:raise AssertionError("second stage accepted")
            synthetic.unlink()
            try:module.build_inventory(scene,run)
            except AssertionError:results.append(dict(scene=scene,test="missing_prediction_rejected_before_SSH",passed=True))
            else:raise AssertionError("missing prediction accepted")
            os.link(full/"renders_hr/0001.png",synthetic)
            # Probe-only detection of an existing wrong remote image does not
            # repair it. Replace a TEMPORARY prediction, never a real HR link.
            wrong = remote/run/"renders_hr/0000.png"
            wrong.unlink();wrong.write_bytes(b"synthetic_conflict")
            payload=dict(files={str(run/"renders_hr/0000.png"):dict(sha256=prediction_hash,bytes=synthetic.stat().st_size)},must_be_absent=[])
            observed=fake_probe(payload)
            assert observed["conflicts"]==[str(run/"renders_hr/0000.png")] and wrong.read_bytes()==b"synthetic_conflict"
            results.append(dict(scene=scene,test="remote_conflict_detected_preserved",passed=True))
        # Execute the actual generated remote programs in a local sandbox,
        # replacing only the hostname/root transport. This catches quoting,
        # frozen-verifier CLI and exclusive remote-receipt wiring mistakes.
        transport_root=base/"actual_program_fake_SSH"
        transport_root.mkdir()
        frozen=transport_root/module.HERE/"verify_assets.py"
        frozen.parent.mkdir(parents=True)
        shutil.copy2(ROOT/module.HERE/"verify_assets.py",frozen)
        tiny=transport_root/"tiny.txt";tiny.write_text("synthetic CPU transport only")
        tiny_row=dict(sha256=hashlib.sha256(tiny.read_bytes()).hexdigest(),bytes=tiny.stat().st_size,role="synthetic",scene="synthetic")
        probe_payload=dict(root=str(transport_root),files={"tiny.txt":tiny_row,"absent.txt":tiny_row},must_be_absent=["receipt.json"])
        completed=subprocess.run(["/usr/bin/python3","-c",module.PROBE.replace(module.REMOTE_ROOT,str(transport_root))],
                                 input=json.dumps(probe_payload),capture_output=True,text=True,check=True)
        observed=json.loads(completed.stdout)
        assert observed["missing"]==["absent.txt"] and not observed["conflicts"]
        results.append(dict(test="actual_probe_program_local_fake_SSH",passed=True))
        inv=Path("tiny_inventory.json")
        (transport_root/inv).write_text(json.dumps(dict(schema="CEM_area4_closed_input_inventory_v1",files={"tiny.txt":tiny_row},
                total_bytes=tiny.stat().st_size,scope="synthetic",train_test_boundary="synthetic")))
        original_run=subprocess.run

        def fake_SSH_run(argv,**kwargs):
            assert argv[:4]==["ssh","-o","BatchMode=yes","a100-train"]
            remote_argv=shlex.split(argv[-1])
            assert remote_argv[:2]==["/usr/bin/python3","-c"] and len(remote_argv)==3
            return original_run(["/usr/bin/python3","-c",remote_argv[2].replace(module.REMOTE_ROOT,str(transport_root))],**kwargs)

        module.subprocess.run=fake_SSH_run
        try:
            completed=actual_remote_verify(inv,Path("receipt.json"),hashlib.sha256(frozen.read_bytes()).hexdigest())
        finally:
            module.subprocess.run=original_run
        assert completed.returncode==0,(completed.stdout,completed.stderr)
        receipt=json.loads((transport_root/"receipt.json").read_text())
        assert receipt["status"]=="verified" and receipt["file_count"]==receipt["verified_count"]==1
        assert (transport_root/"receipt.json.stage.lock").is_file()
        results.append(dict(test="actual_verify_program_CLI_exclusive_receipt_local_fake_SSH",passed=True))
    value=dict(schema="CEM_returned_HRref_image_assets_CPU_contract_v1",status="passed",UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        actual_manifests=manifests,tests=results,test_count=len(results),temporary_roots_removed=True,production_SSH_calls=0,production_GPU_calls=0,
        synthetic_images_and_receipts_only_inside_removed_temporary_roots=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:json.dump(value,stream,indent=2);stream.write("\n")
    print(json.dumps(dict(status=value["status"],tests=len(results),source_sha256=value["source_sha256"])))


if __name__=="__main__":
    main()
