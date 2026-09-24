#!/usr/bin/env python3
"""Acquire only the registered official MeetRoom vrheadset sequence and prepare it on CPU."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/dynamic_sr/meetroom_raw/vrheadset'
SOURCE = RAW / 'source'
PREPARED = ROOT / 'data/dynamic_sr/meetroom_prepared/vrheadset'
FOLDER = 'https://drive.google.com/drive/folders/17fX5zp-t311K4xtuJU6H1a2WRydKPOvZ'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def acquire(row):
    file_id, name, expected_bytes = row[0], row[2], int(row[13])
    dest = RAW / name
    url = 'https://drive.google.com/uc?export=download&id=' + file_id
    started = time.monotonic()
    preexisting = dest.exists()
    if not preexisting:
        temp = Path(str(dest) + '.part')
        subprocess.run(['curl', '-fL', '--retry', '2', '--max-time', '300', '-sS', url, '-o', str(temp)], check=True)
        if temp.stat().st_size != expected_bytes:
            raise ValueError(f'Download size differs from official listing: {name}')
        temp.replace(dest)
    if dest.stat().st_size != expected_bytes:
        raise ValueError(f'Existing file differs from official listing: {name}')
    with dest.open('rb') as stream:
        header = stream.read(12)
    record = {'name': name, 'path': str(dest), 'url': url, 'file_id': file_id,
              'bytes': dest.stat().st_size, 'official_listing_bytes': expected_bytes,
              'sha256': sha(dest), 'preexisting_preview_download': preexisting,
              'elapsed_seconds_this_acquisition': time.monotonic() - started}
    if name.endswith('.mp4'):
        if header[4:8] != b'ftyp':
            raise ValueError(f'Not an MP4: {name}')
        command = [str(ROOT / 'scripts/ffprobe_local.sh'), '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration',
                   '-of', 'json', str(dest)]
        result = json.loads(subprocess.check_output(command))
        stream, = result['streams']
        if ((stream['width'], stream['height'], int(stream['nb_frames'])) != (1280, 720, 300)
                or Fraction(stream['avg_frame_rate']) != 30 or Fraction(stream['r_frame_rate']) != 30
                or abs(float(stream['duration']) - 10.0) > 1e-6):
            raise ValueError(f'Unexpected stream protocol: {name}: {stream}')
        record['stream'] = stream
        write(SOURCE / (dest.stem + '_ffprobe.json'), result)
    else:
        if header[:6] != b'\x93NUMPY':
            raise ValueError('Not a NumPy calibration file')
        bounds = np.load(dest, allow_pickle=False)
        if bounds.shape != (13, 17) or not np.isfinite(bounds).all():
            raise ValueError(f'Invalid calibration array: {bounds.shape}')
        record['shape'] = list(bounds.shape)
    write(SOURCE / (dest.name + '.receipt.json'), record)
    print(json.dumps({'file': name, 'bytes': record['bytes'], 'sha256': record['sha256']}), flush=True)
    return record


def main():
    started = time.monotonic()
    rows = json.loads((SOURCE / 'official_listing.json').read_text())
    expected_names = {f'cam_{i}.mp4' for i in range(13)} | {'poses_bounds.npy'}
    if len(rows) != 14 or {r[2] for r in rows} != expected_names:
        raise ValueError('Official listing does not contain the exact registered file set')
    if (PREPARED / 'manifest.json').exists():
        raise FileExistsError('Prepared manifest already exists; refusing to overwrite')
    write(SOURCE / 'acquisition_status.json', {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat()})
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            receipts = list(pool.map(acquire, rows))
        write(RAW / 'download_receipts.json', {
            'official_repository': 'https://github.com/AlgoHunt/StreamRF',
            'official_dataset_link': 'https://drive.google.com/drive/folders/1lNmQ6_ykyKjT6UKy-SnqWoSlI5yjh3l_',
            'sequence_folder': FOLDER, 'sequence': 'vrheadset',
            'files': receipts, 'total_bytes': sum(r['bytes'] for r in receipts),
            'listing_sha256': sha(SOURCE / 'official_listing.json'),
            'acquisition_script_sha256': sha(__file__),
            'selection_basis': 'Before training, inspect only training cam02 frames0/60/118: complete normal background and moving person with clothing/device boundaries and plant occlusion. No dev/test quality inspection or metric-based selection.',
            'independence_scope': 'New sequence not used for motion-bound design; same MeetRoom domain and room, not a new independent domain',
            'physical_sync': 'Dataset-author supplied synchronization; stream metadata checks verify file dimensions/frame counts/rates, not independent physical shutter sync',
            'download_elapsed_seconds': time.monotonic() - started})
        command = [sys.executable, str(ROOT / 'experiments/dynamic_sr_20260919/prepare_meetroom.py'),
                   '--raw-dir', str(RAW), '--output', str(PREPARED), '--start-frame', '0',
                   '--num-frames', '60', '--frame-stride', '2', '--min-init-points', '256', '--max-init-points', '60000']
        write(SOURCE / 'preparation_command.json', {'argv': command,
              'script_sha256': sha(ROOT / 'experiments/dynamic_sr_20260919/prepare_meetroom.py'),
              'gpu_used': False, 'old_source_modified': False})
        with (SOURCE / 'preparation.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = PREPARED / 'manifest.json'
        data = json.loads(manifest.read_text())
        if data['scene'] != 'meetroom_vrheadset' or data['source'] != 'https://github.com/AlgoHunt/StreamRF':
            raise ValueError('Prepared source/scene identity mismatch')
        write(SOURCE / 'acquisition_status.json', {'status': 'completed_and_prepared',
              'manifest': str(manifest), 'manifest_sha256': sha(manifest),
              'observations': len(data['observations']), 'initialization_points': data['initialization']['point_count'],
              'elapsed_seconds': time.monotonic() - started, 'gpu_used': False,
              'finished_utc': datetime.now(timezone.utc).isoformat()})
        print((SOURCE / 'acquisition_status.json').read_text(), flush=True)
    except BaseException as error:
        write(SOURCE / 'acquisition_status.json', {'status': 'failed', 'error': repr(error),
              'elapsed_seconds': time.monotonic() - started, 'gpu_used': False})
        raise


if __name__ == '__main__':
    main()
