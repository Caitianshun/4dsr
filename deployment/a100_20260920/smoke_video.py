"""Check the configured evaluator video encoder on four unchanged HR frames."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_20260918'))
from evaluate import encode_video

deploy = ROOT / 'deployment/a100_20260920'
source = ROOT / 'data/dynamic_sr/n3dv_prepared/cook_spinach/hr/cam00'
with tempfile.TemporaryDirectory(prefix='video_smoke_', dir=deploy) as temporary:
    temporary = Path(temporary)
    for frame in [0, 2, 4, 6]:
        image = source / f'{frame:04d}.png'
        assert image.is_file(), image
        (temporary / image.name).symlink_to(image)
    output = deploy / 'video_smoke.mp4'
    result = encode_video(temporary, output, 15)
    assert result['returncode'] == 0, result
    assert output.stat().st_size > 0
    import cv2
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        assert image.shape[:2] == (1008, 1344), image.shape
        decoded += 1
    capture.release()
    assert decoded == 4, decoded
    result.update(status='passed', decoded_frames=decoded, bytes=output.stat().st_size,
                  scope='Video path and encoding only; no model quality claim.')
    (deploy / 'video_smoke.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
