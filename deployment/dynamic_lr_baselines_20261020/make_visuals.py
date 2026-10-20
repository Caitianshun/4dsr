"""Build fixed-frame full-image and face comparisons from completed renders."""
import argparse
import hashlib
import json
from pathlib import Path

METHOD_ORDER = ('wu', 'yang', '4c4d', 'multi4d', 'localdygs', 'cem')
METHOD_LABELS = {'wu': 'Wu 4DGS', 'yang': 'Yang 4DGS', '4c4d': '4C4D',
                 'multi4d': 'Multi4D', 'localdygs': 'LocalDyGS', 'cem': 'CEM-4DGS'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True, help='JSON list with method, scene, PNG_directory, PNG_pattern')
    parser.add_argument('--rois', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    root = Path(__file__).resolve().parents[2]
    runs = json.loads(Path(args.runs).read_text())
    regions = json.loads(Path(args.rois).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for region in regions['records']:
        scene, frame = region['scene'], region['frame']
        selected = sorted((r for r in runs if r['scene'] == scene), key=lambda r: METHOD_ORDER.index(r['method']))
        assert len(selected) == 6 and len({r['method'] for r in selected}) == 6
        reference = root / region['reference']
        assert sha(reference) == region['reference_sha256']
        sources = [('HR reference', reference)] + [
            (r['method'], Path(r['PNG_directory']) / r.get('PNG_pattern', '{frame:04d}.png').format(frame=frame))
            for r in selected]
        row = dict(scene=scene, frame=frame, camera='cam00', face_xyxy=region['xyxy'], sources=[])
        for mode, columns, size in [('whole', 2, (672, 504)), ('face', 4, (256, 256))]:
            width, height = size
            header, font_size = (40, 30) if mode == 'whole' else (32, 20)
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
            rows = (len(sources) + columns - 1) // columns
            canvas = Image.new('RGB', (columns * width, rows * (height + header)), 'white')
            draw = ImageDraw.Draw(canvas)
            for i, (method, path) in enumerate(sources):
                with Image.open(path) as stream:
                    assert stream.size == (1344, 1008) and stream.mode == 'RGB'
                    image = stream.copy()
                if mode == 'face':
                    image = image.crop(tuple(region['xyxy'])).resize(size, Image.Resampling.NEAREST)
                else:
                    image = image.resize(size, Image.Resampling.LANCZOS)
                x, y = (i % columns) * width, (i // columns) * (height + header)
                draw.text((x + 6, y + 3), METHOD_LABELS.get(method, method), font=font, fill='black')
                canvas.paste(image, (x, y + header))
                if mode == 'whole':
                    row['sources'].append(dict(method=method, path=str(path), sha256=sha(path)))
            destination = out / f'{scene}_cam00_{frame:04d}_{mode}.png'
            canvas.save(destination)
            row[mode] = dict(path=destination.name, sha256=sha(destination))
        records.append(row)
    receipt = dict(status='completed', camera='cam00', records=records,
                   ROI_source_sha256=sha(args.rois), runs_sha256=sha(args.runs),
                   whole_display='half-size Lanczos thumbnails; original full-resolution renders retained',
                   face_display='fixed 128x128 HR crop enlarged 2x with nearest-neighbor; no sharpening')
    (out / 'visual_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
