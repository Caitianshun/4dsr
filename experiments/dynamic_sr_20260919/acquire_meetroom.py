"""Fetch official, publicly shared MeetRoom discussion files with receipts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/dynamic_sr/meetroom_raw'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def acquire(row):
    file_id, name = row[0], row[2]
    dest = RAW / 'discussion' / name
    url = 'https://drive.google.com/uc?export=download&id=' + file_id
    started = time.time()
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + '.part')
        subprocess.run(['curl', '-fL', '--retry', '2', '--max-time', '300', '-sS', url, '-o', str(tmp)], check=True)
        if name.endswith('.mp4'):
            with tmp.open('rb') as f:
                assert f.read(12)[4:8] == b'ftyp', f'Not an MP4: {name}'
        else:
            with tmp.open('rb') as f:
                assert f.read(6) == b'\x93NUMPY', f'Not NPY: {name}'
        tmp.replace(dest)
    receipt = {'name': name, 'url': url, 'bytes': dest.stat().st_size, 'sha256': sha(dest),
               'elapsed_s_this_acquisition': time.time() - started}
    print(json.dumps(receipt), flush=True)
    return receipt

if __name__ == '__main__':
    rows = json.loads((RAW / 'source/discussion_listing.json').read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(acquire, rows))
    output = {'official_repository': 'https://github.com/AlgoHunt/StreamRF',
              'official_dataset_link': 'https://drive.google.com/drive/folders/1lNmQ6_ykyKjT6UKy-SnqWoSlI5yjh3l_',
              'sequence_folder': 'https://drive.google.com/drive/folders/1ZT7WU_giCBDezj_q21zlaiFhQOWBY_gD',
              'files': receipts}
    (RAW / 'discussion/download_receipts.json').write_text(json.dumps(output, indent=2) + '\n')
