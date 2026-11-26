import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pipeline16 as p
from review_gallery import safe_image


def main():
    records = json.loads((p.ROOT / 'edge_finish_v1/val_stagea_private.json').read_text())
    r = next(r for r in records if r['alias'] == 'VAL_STAGEA-013')
    selected = {'1.6': r['item']}
    for version, directory in [('1.3', p.crop.PARENT), ('2.0', p.PRIOR_MANIFEST)]:
        summary = json.loads((directory / 'summary.json').read_text())
        for filename in ['val_usable.jsonl', 'val_held.jsonl']:
            file = directory / filename
            if not file.exists():
                continue
            assert p.sha(file) == summary['manifest_sha256'][filename]
            for line in file.open():
                e = json.loads(line)
                if e['key'] == r['key']:
                    selected[version] = e
    assert len(selected) == 3
    canvas = Image.new('RGB', (1408, 516), '#171b22')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    stats = {}
    for col, version in enumerate(['1.3', '1.6', '2.0']):
        e = selected[version]
        assert p.sha(e['gray01_npy']) == e['gray01_sha256']
        gray, valid, safe, pixels = safe_image(e, r['native_hw'])
        draw.text((16 + 464*col, 14), 'VAL A-013 | YOLO BOX x' + version, font=font, fill='white')
        canvas.paste(Image.fromarray(pixels).convert('RGB'), (16 + 464*col, 48))
        stats[version] = {'variant': e['variant'], 'cleanup_status': e.get('cleanup_status', 'original_1p3'),
                          'valid_pixels': int(valid.sum()), 'preview_hidden_valid_pixels': int((valid & ~safe).sum()),
                          'raw_quantiles_valid': np.quantile(gray[valid], [0, .01, .25, .5, .75, .99, 1]).tolist(),
                          'zero_fraction_valid': float((gray[valid] == 0).mean())}
    image = p.ROOT / 'edge_finish_v1/val13_versions.png'
    assert not image.exists()
    canvas.save(image)
    report = {'alias': 'VAL_STAGEA-013', 'versions': stats, 'image_sha256': p.sha(image),
              'actual_array_gray_unchanged': True, 'patient_identity_and_kl_exported': False}
    p.save(p.OUT / 'val13_diagnostic.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
