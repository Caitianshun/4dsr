"""CPU verification of a concrete sync batch; never performs a transfer.

The owner supplies prefixes exported from the fixed sync service. This check
does not itself register a prefix or replace the fixed service's acceptance.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import *

MAX_FILES, MAX_TOTAL, MAX_FILE = 20000, 32 * 2**30, 16 * 2**30


def validate_paths(document, prefixes):
    require(document.get("version") == 1 and set(document) == {"version", "files"}, "Sync schema mismatch")
    require(0 < len(document["files"]) <= MAX_FILES, "Invalid sync batch count")
    paths, seen = [], set()
    for entry in document["files"]:
        require(set(entry) == {"path"}, "Only a project-relative path is allowed")
        path = entry["path"]
        p = PurePosixPath(path)
        require(path and not p.is_absolute() and ".." not in p.parts
                and p.as_posix() == path and path not in seen, "Unsafe/duplicate sync path")
        require(any(path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/")
                    for prefix in prefixes), "Unregistered prefix: " + path)
        seen.add(path)
        paths.append(path)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--allowed-prefixes", required=True, help="JSON array from fixed sync service")
    p.add_argument("--catalog", help="Generated sync/catalog.json; verify known source SHAs")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    document, prefixes = read(a.manifest), read(a.allowed_prefixes)
    require(isinstance(prefixes, list) and all(isinstance(s, str) and s and not s.startswith("/")
                                             and ".." not in Path(s).parts for s in prefixes), "Invalid prefix export")
    paths = validate_paths(document, prefixes)
    known = {}
    if a.catalog:
        for row in read(a.catalog)["files"]:
            if row.get("sha256"):
                known[row["path"]] = row["sha256"]
    total, files = 0, []
    for path in paths:
        pth = local(path)
        require(pth.is_file() and not pth.is_symlink(), "Missing/nonregular sync input: " + path)
        size = pth.stat().st_size
        require(size <= MAX_FILE, "Sync single-file size exceeds 16 GiB")
        digest = sha(pth)
        require(path not in known or known[path] == digest, "Sync source SHA differs from preparation")
        total += size
        files.append(dict(path=path, bytes=size, sha256=digest))
    require(total <= MAX_TOTAL, "Sync batch exceeds 32 GiB; split before transfer")
    write_new(a.out, dict(schema="sync_batch_cpu_acceptance_v1", status="passed", source=identity(a.manifest),
                          prefix_export=identity(a.allowed_prefixes), count=len(files), bytes=total, files=files,
                          actual_sync=False, fixed_entry_acceptance_still_required=True))
    print(__import__("json").dumps(dict(status="passed", count=len(files), bytes=total, actual_sync=False)))


if __name__ == "__main__":
    main()
