"""Restore installed console entry points for this freshly created CTS venv.

Only installed distribution metadata is used. This avoids accidentally using
system pip because site-packages transfer does not copy old host bin shebangs.
"""
import importlib.metadata
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
assert Path(sys.prefix).resolve() == (root / '.venv').resolve(), sys.prefix
binary = root / '.venv/bin'
created = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata['Name']
    for entry in distribution.entry_points:
        if entry.group != 'console_scripts':
            continue
        assert '/' not in entry.name and entry.name not in {'python','python3','python3.10'}
        target = binary / entry.name
        if target.is_symlink():
            target.unlink()
        content = (f'#!{sys.executable}\n'
                   'import sys\nfrom importlib.metadata import distribution\n'
                   f'entry = next(e for e in distribution({name!r}).entry_points '
                   f'if e.group == "console_scripts" and e.name == {entry.name!r})\n'
                   'if __name__ == "__main__":\n    sys.exit(entry.load()())\n')
        target.write_text(content)
        target.chmod(0o755)
        created.append(dict(name=entry.name,distribution=name,value=entry.value))
(root/'deployment/cts_20260921/verification/console_scripts.json').write_text(json.dumps(created,indent=2)+'\n')
print(f'Restored {len(created)} console entry points under {binary}')
