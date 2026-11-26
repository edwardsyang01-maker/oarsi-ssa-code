import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pipeline16 as p
from review_gallery import safe_image


def main():
    previous = json.loads((p.crop.pilot.OUT / 'selection_private.json').read_text())
    keys = {r['parent448']['key'] for r in previous}
    maps = {'v1p3': {}, 'v1p6': {}, 'v2p0': {}}
    for version, folder in [('v1p3', p.crop.PARENT), ('v2p0', p.PRIOR_MANIFEST)]:
        summary = json.loads((folder / 'summary.json').read_text())
        for split in ['train', 'val']:
            file = folder / (split + '_usable.jsonl')
            assert p.sha(file) == summary['manifest_sha256'][file.name]
            for line in file.open():
                e = json.loads(line)
                if e['key'] in keys:
                    maps[version][e['key']] = e
    native = {}
    for r in previous:
        e = r['parent448']
        file = p.ROOT / 'edge_finish_v1/cases' / (e['case_id'] + '.json')
        result = json.loads(file.read_text())
        item = next(x for x in result['entries'] if x['key'] == e['key'])
        if item['engineering_input_ready']:
            maps['v1p6'][e['key']] = item
        native[e['key']] = result['native_hw']
    common = set.intersection(*(set(v) for v in maps.values()))
    selected = []
    for split in ['train', 'val']:
        rows = [r for r in previous if r['parent448']['split'] == split and r['parent448']['key'] in common]
        rows.sort(key=lambda r: r['sample'])
        assert len(rows) >= 2
        selected += rows[:2]
    out = p.ROOT / 'three_version_preview'
    out.mkdir(exist_ok=False)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    public, private, pages = [], [], []
    for split in ['train', 'val']:
        selected_split = [r for r in selected if r['parent448']['split'] == split]
        canvas = Image.new('RGB', (1408, 1032), '#171b22')
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), f'{split.upper()} | SAME KNEES | 448 x 448 EACH', font=font, fill='white')
        for column, (version, factor) in enumerate([('v1p3', '1.3'), ('v1p6', '1.6'), ('v2p0', '2.0')]):
            draw.text((16 + column * 464, 40), 'YOLO BOX x' + factor, font=font, fill=['#d5dee9', '#9fe5bb', '#ffd28a'][column])
            for n, r in enumerate(selected_split):
                key = r['parent448']['key']
                e = maps[version][key]
                for field in ['patient_id', 'kl', 'split', 'clinical_side']:
                    assert e[field] == maps['v1p3'][key][field]
                for field in ['gray01', 'valid_mask']:
                    assert p.sha(e[field + '_npy']) == e[field + '_sha256']
                _, _, _, pix = safe_image(e, native[key])
                alias = split.upper() + '-' + f'{n + 1:02d}'
                x, y = 16 + column * 464, 75 + n * 478
                draw.text((x, y), alias + ' | ' + e['clinical_side'], font=font, fill='white')
                image = Image.fromarray(pix).convert('RGB')
                canvas.paste(image, (x, y + 24))
                file = alias.lower() + '_' + version + '.png'
                image.save(out / file)
                public.append({'alias': alias, 'split': split, 'side': e['clinical_side'], 'version': factor,
                               'file': file, 'sha256': p.sha(out / file), 'shape': [448, 448]})
                private.append({'alias': alias, 'version': version, 'key': key,
                                'gray01_sha256': e['gray01_sha256'], 'valid_mask_sha256': e['valid_mask_sha256'],
                                'gray01_npy': e['gray01_npy'], 'valid_mask_npy': e['valid_mask_npy']})
        filename = split + '_compare.png'
        canvas.save(out / filename)
        with Image.open(out / filename) as rendered:
            for column, version in enumerate(['v1p3', 'v1p6', 'v2p0']):
                for n in range(2):
                    x, y = 16 + column * 464, 99 + n * 478
                    single = out / (split + '-' + f'{n + 1:02d}' + '_' + version + '.png')
                    with Image.open(single) as im:
                        assert np.array_equal(np.asarray(rendered.crop((x, y, x + 448, y + 448))), np.asarray(im))
        pages.append({'file': filename, 'sha256': p.sha(out / filename)})
    p.save(out / 'public_index.json', {'samples': 4, 'versions': [1.3, 1.6, 2.0], 'records': public,
                                      'pages': pages, 'actual_final_arrays': True, 'gray_mapping_unchanged': True})
    p.save(p.ROOT / 'three_version_preview_private.json', private)
    bundle = p.ROOT / 'three_version_preview.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.iterdir()):
            z.write(file, file.name)
    receipt = {'samples': 4, 'images': 12, 'pages': 2, 'sha256': p.sha(bundle),
               'tiles_verified_pixel_exact': True, 'patient_identity_and_kl_exported': False}
    p.save(p.OUT / 'three_version_preview_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
