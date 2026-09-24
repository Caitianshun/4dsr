"""One-shot deployment continuation, awakened by process exits (no polling)."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / 'deployment/a100_20260920'
status = dict(status='waiting', dependencies=[int(x) for x in sys.argv[1:]])
try:
    with selectors.DefaultSelector() as selector:
        for pid in status['dependencies']:
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            selector.register(fd, selectors.EVENT_READ, pid)
        while selector.get_map():
            for key, _ in selector.select():
                print('DEPENDENCY_EXIT', key.data, flush=True)
                selector.unregister(key.fd)
                os.close(key.fd)
    assert json.loads((DEPLOY / 'install_exit.json').read_text())['returncode'] == 0
    assert json.loads((DEPLOY / 'raw_download_complete.json').read_text())['status'] == 'complete'
    status['status'] = 'restoring'
    subprocess.run([str(ROOT / '.venv/bin/python'), str(DEPLOY / 'restore_images.py'),
                    '--allow-regenerate'], cwd=ROOT, check=True)
    status['status'] = 'complete'
except Exception as error:
    status.update(status='failed', error=repr(error))
(DEPLOY / 'image_restore_continuation.json').write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
raise SystemExit(0 if status['status'] == 'complete' else 1)
