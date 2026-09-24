"""Check the configured evaluator video encoder on four unchanged HR frames."""
import json
from pathlib import Path
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_20260918'))


def main():
    verify = ROOT / 'deployment/cts_20260921/verification'
    verify.mkdir(parents=True, exist_ok=True)
    output = verify / 'video_smoke.mp4'
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite existing smoke artifact: {output}')
    with (verify / 'video_smoke.json').open('x') as report:
        result = dict(status='running', project_root=str(ROOT), output=str(output))
        try:
            import cv2
            from evaluate import encode_video
            source = ROOT / 'data/dynamic_sr/n3dv_prepared/cook_spinach/hr/cam00'
            with tempfile.TemporaryDirectory(prefix='video_smoke_', dir=verify) as temporary:
                temporary = Path(temporary)
                for frame in [0, 2, 4, 6]:
                    image = source / f'{frame:04d}.png'
                    assert image.is_file(), image
                    (temporary / image.name).symlink_to(image)
                result.update(encode_video(temporary, output, 15))
                assert result['returncode'] == 0, result
                assert output.stat().st_size > 0
                capture = cv2.VideoCapture(str(output))
                decoded = 0
                try:
                    while True:
                        ok, image = capture.read()
                        if not ok:
                            break
                        assert image.shape[:2] == (1008, 1344), image.shape
                        decoded += 1
                finally:
                    capture.release()
                assert decoded == 4, decoded
                result.update(status='passed', decoded_frames=decoded, bytes=output.stat().st_size,
                              scope='Video path and encoding only; no model quality claim.')
        except BaseException:
            result.update(status='failed', error=traceback.format_exc())
            raise
        finally:
            json.dump(result, report, indent=2)
            report.write('\n')
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
