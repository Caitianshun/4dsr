"""Incrementally stage the frozen job files, then verify remote source hashes.

Conflicting remote files are never overwritten. This is CPU/SSH preparation,
not a GPU submission. Image assets have their separate full verification gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REMOTES = {"a100-train": "/home/ubuntu/3DGS/4dsr", "cts": "/home/cts/Project/4DSR"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", choices=REMOTES, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    assert not a.receipt.exists()
    source = a.job.resolve()
    job = json.loads(source.read_text())
    files = list(job["required_files"]) + [dict(path=str(source.relative_to(ROOT)), sha256=sha(source))]
    assert len({x["path"] for x in files}) == len(files)
    for x in files:
        assert (ROOT / x["path"]).resolve().is_relative_to(ROOT) and sha(ROOT / x["path"]) == x["sha256"]
    data = dict(root=REMOTES[a.host], files=files)
    script = '''import hashlib,json,sys
from pathlib import Path
d=json.loads(sys.stdin.readline())
root=Path(d['root']).resolve()
missing=[];conflicts=[]
for row in d['files']:
 p=(root/row['path']).resolve()
 assert p.is_relative_to(root)
 if not p.exists():missing.append(row['path']);continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 if h.hexdigest()!=row['sha256']:conflicts.append(row['path'])
print(json.dumps(dict(missing=missing,conflicts=conflicts,count=len(d['files']))))
'''
    # Feed the remote Python program via -c and JSON exclusively via stdin.
    import shlex
    command = ["ssh", "-o", "BatchMode=yes", a.host, "/usr/bin/python3 -c " + shlex.quote(script)]
    def check():
        result = subprocess.run(command, input=json.dumps(data)+"\n", text=True, capture_output=True, check=True, timeout=600)
        return json.loads(result.stdout)
    before = check()
    assert not before["conflicts"], "Preserve conflicting remote source files: " + repr(before["conflicts"])
    if before["missing"]:
        subprocess.run(["rsync", "-a", "--relative", "--ignore-existing", "--files-from=-", "./",
                        a.host+":"+REMOTES[a.host]+"/"], cwd=ROOT,
                       input="\n".join(before["missing"])+"\n", text=True, check=True, timeout=3600)
    after = check()
    assert not after["missing"] and not after["conflicts"]
    result = dict(status="all_frozen_job_sources_remote_SHA_verified", host=a.host, job_id=job["job_id"],
                  job_sha256=sha(source), verified_count=after["count"], incremental_missing_count=len(before["missing"]),
                  remote_conflicts_overwritten=0, GPU_calls=0)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
