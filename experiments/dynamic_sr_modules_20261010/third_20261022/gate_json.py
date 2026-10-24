"""冒烟门（Claude 4dsr-46，2026-10-23）：读取 JSON 文件中的若干键，全部等于期望值时退出 0，否则打印并退出 1（lane 中标为 CRITICAL，
失败即写 ABORT，让同一作业的其他 lane 停下）。用法：gate_json.py FILE KEY=VALUE [KEY=VALUE ...]（VALUE 按 JSON 解析，如 true、30）"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
f = ROOT / sys.argv[1]
d = json.loads(f.read_text()) if f.exists() else {}
bad = {}
for kv in sys.argv[2:]:
    k, v = kv.split('=', 1)
    if d.get(k) != json.loads(v):
        bad[k] = dict(want=json.loads(v), got=d.get(k))
print(json.dumps(dict(file=sys.argv[1], ok=not bad, bad=bad), ensure_ascii=False), flush=True)
sys.exit(1 if bad else 0)
