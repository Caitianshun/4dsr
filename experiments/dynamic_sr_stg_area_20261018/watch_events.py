"""Read a run's completion events through Linux inotify, without polling.

Run over SSH; the local collector copies completed stage artifacts when a JSON
line arrives. Existing events are replayed once after reconnect. No GPU access.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import sys


def watch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1 failed')
    if libc.inotify_add_watch(fd, os.fsencode(directory), 0x80 | 0x8) < 0:
        raise OSError(ctypes.get_errno(), 'inotify_add_watch failed')
    delivered = set()
    try:
        while True:
            for p in sorted(directory.glob('*.json')):
                if p.name not in delivered:
                    value = json.loads(p.read_text())
                    print(json.dumps(dict(event=p.name, value=value)), flush=True)
                    delivered.add(p.name)
                    if p.name == 'terminal.json':
                        return
            select.select([fd], [], [])
            os.read(fd, 65536)
    finally:
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    watch(parser.parse_args().directory)
