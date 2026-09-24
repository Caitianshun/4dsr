"""Fetch large pinned wheels with curl, avoiding a slow multiplexed installer connection."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import urllib.parse
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'deployment/wheelhouse'
DEST.mkdir(parents=True, exist_ok=True)
BASE = 'https://pypi.tuna.tsinghua.edu.cn/'
PINS = {
    'nvidia-cublas-cu12':'12.8.3.14', 'nvidia-cuda-cupti-cu12':'12.8.57',
    'nvidia-cuda-nvrtc-cu12':'12.8.61', 'nvidia-cuda-runtime-cu12':'12.8.57',
    'nvidia-cudnn-cu12':'9.7.1.26', 'nvidia-cufft-cu12':'11.3.3.41',
    'nvidia-cufile-cu12':'1.13.0.11', 'nvidia-curand-cu12':'10.3.9.55',
    'nvidia-cusolver-cu12':'11.7.2.55', 'nvidia-cusparse-cu12':'12.5.7.53',
    'nvidia-cusparselt-cu12':'0.6.3', 'nvidia-nccl-cu12':'2.26.2',
    'nvidia-nvjitlink-cu12':'12.8.61', 'nvidia-nvtx-cu12':'12.8.55',
    'triton':'3.3.1', 'numpy':'1.26.4', 'pillow':'12.2.0',
    'scipy':'1.15.3', 'opencv-python':'4.10.0.84', 'open3d':'0.19.0',
    'matplotlib':'3.9.4', 'imageio-ffmpeg':'0.6.0'}
TAGS = {tag:i for i,tag in enumerate(sys_tags())}

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.hrefs=[]
    def handle_starttag(self, tag, attrs):
        if tag == 'a' and 'href' in dict(attrs): self.hrefs.append(dict(attrs)['href'])

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def fetch(item):
    name,version=item; index=BASE+'simple/'+name+'/'
    html=subprocess.check_output(['curl','--noproxy','*','-fsSL','--retry','3','--max-time','90',index],text=True)
    parser=Links(); parser.feed(html); choices=[]
    for href in parser.hrefs:
        url=urllib.parse.urljoin(index,href); parsed=urllib.parse.urlsplit(url)
        filename=urllib.parse.unquote(Path(parsed.path).name)
        if not filename.endswith('.whl'):continue
        try: _,v,_,tags=parse_wheel_filename(filename)
        except ValueError:continue
        matching=[TAGS[t] for t in tags if t in TAGS]
        if str(v)!=version or not matching:continue
        assert parsed.scheme=='https' and parsed.hostname=='pypi.tuna.tsinghua.edu.cn'
        expected=urllib.parse.parse_qs(parsed.fragment)['sha256'][0]
        choices.append((min(matching),filename,url.split('#')[0],expected))
    assert choices,(name,version)
    _,filename,url,expected=min(choices)
    dest=DEST/filename
    if not dest.is_file() or sha(dest)!=expected:
        part=dest.with_suffix('.download')
        subprocess.run(['aria2c','--no-conf=true','--all-proxy=',
                        '--max-connection-per-server=8','--split=8','--min-split-size=8M',
                        '--continue=true','--allow-overwrite=true','--auto-file-renaming=false',
                        '--file-allocation=none','--summary-interval=0','--console-log-level=warn',
                        '--download-result=hide','--max-tries=5','--retry-wait=5',
                        '--checksum=sha-256='+expected,'--dir='+str(DEST),
                        '--out='+part.name,url],check=True)
        assert sha(part)==expected,filename
        part.replace(dest)
    row=dict(name=name,version=version,filename=filename,sha256=expected,bytes=dest.stat().st_size,url=url)
    print(json.dumps(row),flush=True)
    return row

with ThreadPoolExecutor(max_workers=4) as pool:
    rows=list(pool.map(fetch,PINS.items()))
(ROOT/'deployment/a100_20260920/prefetched_wheels.json').write_text(json.dumps(rows,indent=2))
