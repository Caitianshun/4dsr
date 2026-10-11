# A100 Long360 read-only HR storage

The original 20-file experiment namespace remains frozen. This deployment
wrapper subclasses `local_orchestrator.Supervisor` and overrides only the
storage synchronization step. It does not change Z1/Z5 thresholds, input
pixels, sampling, losses, seeds, training schedules, or the 65 GB future-output
and 60 GB root-filesystem reserve checks.

The wrapper supports only `l3H_long360`, half rung, seed 20261007, A100 GPU0/1.
The external tree contains exactly 9,600 train HR PNGs plus the original
`hr_full_storage.json`. It does not contain source videos, teachers,
checkpoints, environments, logs, or writable training outputs.

After the original local CPU preparation, initialization, HR expansion,
registration and task-table generation succeed, the wrapper:

1. Validates that all 32 train cameras and 300 frames are present in the frozen
   additive synchronization inventory.
2. Synchronizes only the deployment sources into the existing isolated A100
   workspace. An existing different file is rejected and preserved.
3. Creates a fresh owned run directory under
   `/media/ubuntu/Game Drive/4dsr-highres-20261011/hr_readonly_inputs/`.
   The external filesystem must still be the existing writable exFAT mount.
   The transfer retains a 60 GB external reserve.
4. Copies files with `rsync -rlt --no-owner --no-group --no-perms
   --ignore-existing --protect-args`, checks every size/SHA256, rejects extra files and
   symlinks, and checks the original HR storage index.
5. Starts a unique persistent user service running the privately extracted
   bindfs binary with `--no-allow-other -r -f -o fsname=<owned source>,subtype=bindfs`. The canonical input directory
   inside the workspace must be empty, owned, and unmounted.
6. Checks `/proc/self/mountinfo` for the exact `fuse.bindfs` source and `ro`,
   checks the user service and all canonical file hashes, and calls the
   unchanged parent synchronization. The prefilled HR files are now reused;
   all other missing files still undergo the original 65+60 GB guard.

Every mount acceptance also checks the service MainPID, process UID, complete
`/proc/PID/cmdline`, executable path/SHA256, and stable PID start time. Unknown
or drifting processes fail closed. Transfer and full-file hashing durations
are recorded; the wrapper does not infer throughput from its tiny smoke.

No root privileges, apt installation, `/etc/fuse.conf` changes, symlink
redirection, manifest rewrites, or automatic retries are used. The binary is
Ubuntu Jammy bindfs 1.14.7-1, extracted inside the private workspace:

- Package SHA256: `22a1e0f43a6ae348f779c0a18090bda96b0285dd13c5c092d363b42bed57ba80`
- Binary SHA256: `9f0da916b05f19558f4e272d1d75b2107df1e2eaadaa072cd5860576e299fc3a`

The root operator creates a fresh spec from the original supervisor, retains
the exact original namespace inventory and science inputs, adds a `storage_recipe`
object with the constants in `validate_storage_spec`, and adds SHA256 entries
for these four deployment files to `freeze.inputs`. No spec is automatically
created, and a completed or uncertain Z5 run is never restarted.

An optional `storage_recipe.ssh_jump` may be `cts`; all SSH, rsync, original
GPU startup/runner dispatch and result return then use temporary `-J cts`
arguments targeting the same A100 host. `null` or omission uses the normal
`a100-train` route. The wrapper does not change SSH configuration or target
hardware. The root operator records which route was actually accepted.

```sh
python deployment/readonly_hr_a100_20261011/selfcheck_storage.py --out tmp/storage_selfcheck.json
python deployment/readonly_hr_a100_20261011/storage_supervisor.py --spec output/path/to/fresh_spec.json --validate-only
```

The root operator launches `storage_supervisor.py --spec ...` through a fresh
persistent local user service after the original Z5 completion is recovered
and the spec is reviewed. `--validate-only` performs no SSH, GPU calls or data
preparation. Any partial storage result is preserved on failure. The wrapper
returns the mount receipt and service log locally after the parent pipeline
returns, including failure. It keeps its read-only input service alive for
audited re-evaluation; cleanup requires identifying this exact owned unit
and target. It never unmounts another filesystem or stops an existing job.

The initial 4 KiB rootless smoke passed on 2026-10-11 05:56 UTC, including the
kernel `ro` flag, equal source/canonical SHA256, unmount, foreground process
exit, and unchanged parent Game Drive mount. This proves basic access and
cleanup; full-corpus throughput and formal GPU startup remain separate
acceptances. The storage mapping is a deployment record; the original HR
index and source-video records remain byte-for-byte unchanged.

Primary option documentation: <https://bindfs.org/docs/bindfs.1.html>.
