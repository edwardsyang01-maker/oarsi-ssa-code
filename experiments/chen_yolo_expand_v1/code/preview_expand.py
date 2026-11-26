from pathlib import Path
import hashlib
import json
import math
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path('/path/to/drai')
sys.path.insert(0, str(PROJECT / 'experiments/chen_yolo_448_v1/code'))
import export_448 as e

ROOT = PROJECT / 'experiments/chen_yolo_expand_v1'
OUT = ROOT / 'outputs'
PUBLIC = ROOT / 'preview_public'
DATA = PROJECT / 'data/processed/chen_yolo_expand_v1/pilot'
PREVIOUS = PROJECT / 'experiments/chen_yolo_448_v1/previews/user_samples_20261111'
FACTOR = 1.2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    e.p.save(path, value)


def expanded_box(box, hw):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * FACTOR, (y1 - y0) * FACTOR
    requested = [math.floor(cx - w / 2), math.floor(cy - h / 2), math.ceil(cx + w / 2), math.ceil(cy + h / 2)]
    height, width = hw
    actual = [max(0, requested[0]), max(0, requested[1]), min(width, requested[2]), min(height, requested[3])]
    assert actual[0] <= x0 < x1 <= actual[2] and actual[1] <= y0 < y1 <= actual[3]
    return actual, requested != actual


def native_gray(item, row):
    ds, pixels, receipt = e.read_pa_archive(e.p.NAS / row['input_archive_relpath'],
                                          expected_bytes=int(row['archive_bytes']), source_root=e.p.NAS)
    assert receipt['archive_sha256'] == row['archive_sha256']
    assert receipt['native_pixel_sha256'] == row['native_pixel_sha256']
    values, valid, pres = e.presentation_values(pixels, ds)
    variant = item['variant']
    if variant in ['original', 'full_original', 'lower_original']:
        gray, mapping = e.normalize_roi(values, valid)
    else:
        raw, raw_values, _ = e.c.modality_gray(pixels, ds, valid, pres)
        if variant == 'modality_roi_samebox':
            x0, y0, x1, y1 = item['bbox']
            _, mapping = e.normalize_roi(raw_values[y0:y1, x0:x1], valid[y0:y1, x0:x1])
            gray = np.clip((raw_values - mapping['low']) / (mapping['high'] - mapping['low']), 0, 1).astype(np.float32)
            gray[~valid] = 0
        elif variant in ['inv_center_raw', 'inv_full_raw', 'lower_inv_raw']:
            gray = np.where(valid, 1 - raw, 0).astype(np.float32)
            mapping = {'inverted_modality': True}
        elif variant in ['modality_samebox', 'center_modality', 'full_modality', 'lower_raw']:
            gray = raw
            mapping = {'modality_gray_unchanged': True}
        else:
            raise ValueError('unsupported frozen gray variant')
    return gray, valid, receipt, mapping


def crop(gray, valid, bbox, item):
    region = {'image_position': item['image_position'], 'bbox_xyxy_exclusive': bbox,
              'quality_status': 'candidate', 'reject_reasons': []}
    output = e.extract_regions(gray, valid, [region], image_size=448)[0]
    e.check_array(output['image'], output['valid_mask'], 448)
    return output


def preview(path, result, bbox, native_hw):
    t = result['transform']
    xx = bbox[0] + (np.arange(448) - t['offset_x'] + .5) / t['scale_x'] - .5
    yy = bbox[1] + (np.arange(448) - t['offset_y'] + .5) / t['scale_y'] - .5
    h, w = native_hw
    safe = ((xx >= .03 * w) & (xx < .97 * w))[None, :] & ((yy >= .20 * h) & (yy < .88 * h))[:, None]
    hidden = int((result['valid_mask'][0] & ~safe).sum())
    pixels = np.rint(result['image'][0] * 255).astype(np.uint8)
    pixels[~safe] = 0
    Image.fromarray(pixels).save(path)
    with Image.open(path) as check:
        assert check.size == (448, 448) and not check.info and np.array_equal(np.asarray(check), pixels)
    return hidden


def make_pages(records):
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 18) if Path(font_path).exists() else ImageFont.load_default()
    pages = []
    for split in ['train', 'val']:
        selected = [r for r in records if r['split'] == split]
        for start in range(0, 6, 2):
            canvas = Image.new('RGB', (944, 1032), '#171b22')
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 10), f'{split.upper()} | same cases / same gray mapping / 448 x 448', fill='white', font=font)
            draw.text((16, 40), 'CURRENT CROP', fill='#d5dee9', font=font)
            draw.text((480, 40), 'WIDTH +20%, HEIGHT +20%', fill='#9fe5bb', font=font)
            for row_number, r in enumerate(selected[start:start + 2]):
                y = 75 + row_number * 478
                text = r['sample'] + ' | ' + r['side']
                draw.text((16, y), text, fill='white', font=font)
                draw.text((480, y), text, fill='white', font=font)
                for column, version in enumerate(['current', 'expanded']):
                    with Image.open(PUBLIC / r[version + '_file']) as im:
                        canvas.paste(im.convert('RGB'), (16 + column * 464, y + 24))
            name = f'{split}_compare_{start // 2 + 1:02d}.png'
            canvas.save(PUBLIC / name)
            pages.append({'file': name, 'sha256': sha(PUBLIC / name)})
    return pages


def main():
    started = time.time()
    parent = PROJECT / 'data/manifests/chen_yolo_usable448_v1_20261111/summary.json'
    assert sha(parent) == '2bed6726a57313ebc0d39f4bc93ae57e037505b64abb6c4d13a9ed8ab00072d6'
    assert e.identity() == json.loads((e.OUT / 'identity.json').read_text())
    authorization = json.loads((e.p.OUT / 'preview_transfer_authorization_20261111.json').read_text())
    assert authorization['user_answer'] == 'joint-anomaly review and spot-check previews for this round are allowed'
    OUT.mkdir(parents=True, exist_ok=False)
    PUBLIC.mkdir()
    DATA.mkdir(parents=True, exist_ok=False)
    save(OUT / 'authorization.json', {'request': 'Can the crop range be larger?', 'scope_choice': 'Compare first to decide the range (recommended)',
         'scope': 'compare only the previous 12 train/val examples and necessary joint boundaries, reusing this batch\'s anomaly-review and spot-check preview authorization',
         'parent_preview_authorization_sha256': sha(e.p.OUT / 'preview_transfer_authorization_20261111.json'),
         'no_patient_identity_or_kl_transfer': True, 'raw_dicom_transfer': False, 'full_export_authorized': False})
    e.initialize()
    base = {r['case_id']: r for r in json.loads((e.p.OUT / 'selection_private.json').read_text())}
    selected = json.loads((PREVIOUS / 'selection_private.json').read_text())
    assert len(selected) == 12
    public, private = [], []
    for index, sample in enumerate(selected, start=1):
        item = sample['source']
        for field in ['gray01', 'valid_mask']:
            assert sha(item[field + '_npy']) == item[field + '_sha256']
        gray, valid, receipt, mapping = native_gray(item, base[item['case_id']])
        old = crop(gray, valid, item['bbox'], item)
        assert np.array_equal(old['image'], np.load(item['gray01_npy'], allow_pickle=False))
        assert np.array_equal(old['valid_mask'], np.load(item['valid_mask_npy'], allow_pickle=False))
        box, clipped = expanded_box(item['bbox'], valid.shape)
        new = crop(gray, valid, box, item)
        stem = sample['sample'].lower().replace(' ', '_')
        image_path, mask_path = DATA / (stem + '_gray01.npy'), DATA / (stem + '_valid.npy')
        e.p.npy_save(image_path, new['image'])
        e.p.npy_save(mask_path, new['valid_mask'])
        current_file, expanded_file = stem + '_current.png', stem + '_expanded.png'
        hidden_old = preview(PUBLIC / current_file, old, item['bbox'], valid.shape)
        hidden_new = preview(PUBLIC / expanded_file, new, box, valid.shape)
        record = {'sample': sample['sample'], 'split': item['split'], 'side': item['clinical_side'],
                  'current_file': current_file, 'expanded_file': expanded_file,
                  'current_sha256': sha(PUBLIC / current_file), 'expanded_sha256': sha(PUBLIC / expanded_file),
                  'source_boundary_clipped': clipped, 'privacy_hidden_valid_pixels': [hidden_old, hidden_new]}
        public.append(record)
        private.append({**record, 'parent': item, 'bbox': box, 'native_hw': list(valid.shape),
                        'original448_reproduced_exactly': True, 'gray_mapping': mapping,
                        'source_receipt': receipt, 'transform': new['transform'],
                        'gray01_npy': str(image_path), 'gray01_sha256': sha(image_path),
                        'valid_mask_npy': str(mask_path), 'valid_mask_sha256': sha(mask_path),
                        'crosses_image_midline': box[0] < valid.shape[1] / 2 < box[2],
                        'engineering_input_ready': False, 'clinical_quality_approved': False})
        save(OUT / 'status.json', {'phase': 'previewing', 'completed': index, 'target': 12})
    pages = make_pages(public)
    save(OUT / 'selection_private.json', private)
    save(PUBLIC / 'public_index.json', {'factor_width': FACTOR, 'factor_height': FACTOR, 'output_size': [448, 448],
         'samples': public, 'comparison_pages': pages, 'training_arrays_unchanged': True,
         'pixel_display': 'fixed original gray mapping; PNG is 8-bit display only; may contain privacy masking, recorded per image'})
    bundle = ROOT / 'preview_bundle.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(PUBLIC.iterdir()):
            assert f.name == 'public_index.json' or f.suffix == '.png'
            archive.write(f, f.name)
    report = {'phase': 'completed', 'factor_width': FACTOR, 'factor_height': FACTOR, 'output_shape': [1, 448, 448],
              'samples': 12, 'counts': {'train': 6, 'val': 6}, 'parent448_exact_reproduction': 12,
              'source_boundary_clipped_count': sum(r['source_boundary_clipped'] for r in public),
              'preview_privacy_masked_count': sum(any(r['privacy_hidden_valid_pixels']) for r in public),
              'cross_midline_candidates': sum(r['crosses_image_midline'] for r in private),
              'bundle_files': len(list(PUBLIC.iterdir())), 'bundle_sha256': sha(bundle),
              'code_sha256': sha(Path(__file__)), 'elapsed_seconds': time.time() - started,
              'full_export_started': False, 'training_started': False, 'test_accessed': False}
    save(OUT / 'completion.json', report)
    save(OUT / 'status.json', {'phase': 'completed', 'completed': 12, 'target': 12})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
