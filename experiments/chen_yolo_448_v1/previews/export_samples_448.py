import hashlib
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path('/path/to/drai')
MANIFEST = PROJECT / 'data/manifests/chen_yolo_usable448_v1_20261111'
ROOT = PROJECT / 'experiments/chen_yolo_448_v1/previews/user_samples_20261111'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def select(rows, seed):
    rng = random.Random(seed)
    chosen = {}
    seen = set()
    for side in ['R', 'L']:
        candidates = sorted((r for r in rows if r['clinical_side'] == side), key=lambda r: r['key'])
        rng.shuffle(candidates)
        chosen[side] = []
        for row in candidates:
            if row['patient_id'] in seen:
                continue
            chosen[side].append(row)
            seen.add(row['patient_id'])
            if len(chosen[side]) == 3:
                break
        assert len(chosen[side]) == 3
    return [chosen[side][i] for i in range(3) for side in ['R', 'L']]


def montage(public, split, entries):
    gap, title_height, label_height = 16, 42, 30
    canvas = Image.new('RGB', (3 * 448 + 4 * gap, 2 * (448 + label_height) + 3 * gap + title_height), '#171b22')
    draw = ImageDraw.Draw(canvas)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    draw.text((gap, 12), f'{split.upper()} | six sampled crops | 448 x 448 each', fill='white', font=font)
    for i, entry in enumerate(entries):
        x = gap + (i % 3) * (448 + gap)
        y = title_height + gap + (i // 3) * (448 + label_height + gap)
        side = 'Right (R)' if entry['side'] == 'R' else 'Left (L)'
        draw.text((x, y + 3), f'{entry["sample"]} | {side}', fill='#d5dee9', font=font)
        with Image.open(public / entry['file']) as im:
            assert im.size == (448, 448)
            canvas.paste(im.convert('RGB'), (x, y + label_height))
    target = public / f'{split}_contact.png'
    canvas.save(target)
    with Image.open(target) as check:
        for i, entry in enumerate(entries):
            x = gap + (i % 3) * (448 + gap)
            y = title_height + gap + (i // 3) * (448 + label_height + gap) + label_height
            with Image.open(public / entry['file']) as source:
                assert np.array_equal(np.asarray(check.crop((x, y, x + 448, y + 448))), np.asarray(source.convert('RGB')))
    return {'file': target.name, 'sha256': sha(target), 'size': list(canvas.size)}


def main():
    summary = json.loads((MANIFEST / 'summary.json').read_text())
    assert summary['phase'] == 'completed' and summary['output_shape'] == [1, 448, 448]
    assert summary['all_selected_448_arrays_verified'] and not summary['resized_from_224']
    assert sha(MANIFEST / 'summary.json') == '2bed6726a57313ebc0d39f4bc93ae57e037505b64abb6c4d13a9ed8ab00072d6'
    ROOT.mkdir(parents=True, exist_ok=False)
    public = ROOT / 'public'
    public.mkdir()
    save_json(ROOT / 'authorization.json', {
        'user_request': 'Pick a few 448x448 crops from the training and validation sets so I can see how they look.',
        'scope': '6 single-knee 448 previews each from train and val plus two mosaics, for viewing in the current chat',
        'transfer': 'only PNGs in public and a preview index without identity/KL are copied locally; private mappings stay on the server',
        'raw_dicom_transfer': False,
        'private_manifest_transfer': False,
        'training_or_test_access': False,
    })
    private = []
    report = {'dataset': MANIFEST.name, 'shape_each': [448, 448], 'selection': '3 left and 3 right knees per set from six different patients, fixed random draw, not selected by KL or quality',
              'display': 'the original float32 [0,1] is only multiplied by 255 and rounded to an 8-bit PNG; no extra windowing, contrast enhancement, flipping or resizing',
              'samples': [], 'contacts': {}}
    for split, seed in [('train', 20261111), ('val', 20261112)]:
        manifest = MANIFEST / f'{split}_usable.jsonl'
        assert sha(manifest) == summary['manifest_sha256'][manifest.name]
        rows = [json.loads(line) for line in manifest.open()]
        entries = []
        for number, row in enumerate(select(rows, seed), start=1):
            assert row['split'] == split and row['output_resolution'] == 448
            assert row['image_position'] == ('image_left' if row['clinical_side'] == 'R' else 'image_right')
            assert sha(row['gray01_npy']) == row['gray01_sha256']
            assert sha(row['valid_mask_npy']) == row['valid_mask_sha256']
            array = np.load(row['gray01_npy'], allow_pickle=False)
            mask = np.load(row['valid_mask_npy'], allow_pickle=False)
            assert array.shape == mask.shape == (1, 448, 448)
            assert array.dtype == np.float32 and mask.dtype == np.bool_
            assert np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1
            assert mask.any() and np.all(array[~mask] == 0)
            pixels = np.rint(array[0] * 255).astype(np.uint8)
            alias = f'{split.upper()} {number:02d}'
            filename = f'{split}_{number:02d}.png'
            Image.fromarray(pixels).save(public / filename)
            with Image.open(public / filename) as im:
                assert im.size == (448, 448) and not im.info
                assert np.array_equal(np.asarray(im), pixels)
            entry = {'sample': alias, 'split': split, 'side': row['clinical_side'], 'file': filename, 'sha256': sha(public / filename)}
            entries.append(entry)
            private.append({'sample': alias, 'seed': seed, 'source': row})
        report['samples'].extend(entries)
        report['contacts'][split] = montage(public, split, entries)
    assert len(private) == 12 and len({x['source']['patient_id'] for x in private}) == 12
    save_json(ROOT / 'selection_private.json', private)
    save_json(public / 'public_index.json', report)
    bundle = ROOT / 'preview_bundle.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(public.iterdir()):
            assert path.name == 'public_index.json' or path.suffix == '.png'
            z.write(path, path.name)
    receipt = {'phase': 'completed', 'samples': 12, 'counts': {'train': 6, 'val': 6}, 'single_png_size': [448, 448],
               'contact_tiles_pixel_exact': True, 'source_hashes_verified': True, 'bundle_sha256': sha(bundle),
               'bundle_files': len(list(public.iterdir())), 'selection_sha256_server_only': sha(ROOT / 'selection_private.json')}
    save_json(ROOT / 'receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
