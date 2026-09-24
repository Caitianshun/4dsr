"""Download official inputs on the target host; verify original source hashes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/dynamic_sr'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def download(url, dest, digest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha(dest) == digest:
        return
    part = dest.with_name(dest.name + '.download')
    subprocess.run(['curl', '-fLsS', '--retry', '3', '--connect-timeout', '30',
                    '--max-time', '3600', url, '-o', str(part)], check=True)
    assert sha(part) == digest, f'Unexpected source hash: {dest}'
    part.replace(dest)
    print(json.dumps(dict(event='verified_download', path=str(dest), bytes=dest.stat().st_size)), flush=True)

def n3dv(scene):
    rec = json.loads((DATA / f'{scene}_source_manifest.json').read_text())
    dest = DATA / 'n3dv_archives' / (scene + '.zip')
    download(rec['source_url'], dest, rec['sha256'])
    raw = DATA / 'n3dv_raw'
    with zipfile.ZipFile(dest) as z:
        for member in z.infolist():
            p = Path(member.filename)
            assert not p.is_absolute() and '..' not in p.parts and p.parts[0] == scene
        z.extractall(raw)
    return scene

def meetroom(row):
    download(row['url'], DATA / 'meetroom_raw/discussion' / row['name'], row['sha256'])
    return row['name']

def main():
    start = time.time()
    rows = json.loads((DATA / 'meetroom_raw/discussion/download_receipts.json').read_text())['files']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(n3dv, s) for s in ['cook_spinach', 'cut_roasted_beef']]
        futures += [pool.submit(meetroom, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            print('complete', future.result(), flush=True)
    result = dict(status='complete', seconds=time.time()-start,
                  note='Original receipts/manifests retained unchanged. Downloads verified against their SHA256.')
    (ROOT / 'deployment/a100_20260920/raw_download_complete.json').write_text(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
