import json
import math
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path('/path/to/drai')
PREVIOUS = PROJECT / 'experiments/chen_yolo_expand_v1'
sys.path.insert(0, str(PREVIOUS / 'code'))
import preview_expand as base

e, sha, save = base.e, base.sha, base.save
ROOT = PROJECT / 'experiments/chen_yolo_expand2x_v1'
OUT, PUBLIC = ROOT / 'outputs', ROOT / 'preview_public'
DATA = PROJECT / 'data/processed/chen_yolo_expand2x_v1/pilot'
VERSIONS = [('current', 'YOLO BOX x1.30'), ('expand20', 'YOLO BOX approx. x1.56'), ('direct2x', 'YOLO BOX x2.00')]


def direct_box(item, native_hw):
    source_path = Path(item['parent_candidate_result'])
    assert sha(source_path) == item['parent_candidate_result_sha256']
    source = json.loads(source_path.read_text())
    if item['candidate_family'] == 'base':
        entry = next(x for x in source['yolo'] if x['image_position'] == item['image_position'] and x['status'] == 'saved')
    else:
        entry = e.saved(source, item['variant'], item['image_position'])
    assert entry['bbox'] == item['bbox']
    x0, y0, x1, y1 = map(float, entry['unclipped_bbox'])
    height, width = native_hw

    def bound(values):
        return [max(0, min(width, math.floor(values[0]))), max(0, min(height, math.floor(values[1]))),
                max(0, min(width, math.ceil(values[2]))), max(0, min(height, math.ceil(values[3])))]

    assert bound([x0, y0, x1, y1]) == item['bbox']
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    detector_w, detector_h = (x1 - x0) / 1.3, (y1 - y0) / 1.3
    requested = [cx - detector_w, cy - detector_h, cx + detector_w, cy + detector_h]
    box = bound(requested)
    assert box[0] <= item['bbox'][0] and box[1] <= item['bbox'][1]
    assert box[2] >= item['bbox'][2] and box[3] >= item['bbox'][3]
    clipped = requested[0] < 0 or requested[1] < 0 or requested[2] > width or requested[3] > height
    return box, clipped, {'unclipped_1p3_bbox': entry['unclipped_bbox'], 'requested_2p0_bbox': requested,
                          'detector_native_size_wh': [detector_w, detector_h], 'parent_result_sha256': sha(source_path)}


def make_pages(records):
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    pages = []
    for split in ['train', 'val']:
        selected = [r for r in records if r['split'] == split]
        for start in range(0, 6, 2):
            canvas = Image.new('RGB', (1408, 1032), '#171b22')
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 10), f'{split.upper()} | same cases / same gray mapping / 448 x 448 each', fill='white', font=font)
            for column, (version, title) in enumerate(VERSIONS):
                draw.text((16 + column * 464, 40), title, fill=['#d5dee9', '#9fe5bb', '#ffd28a'][column], font=font)
                for row_number, r in enumerate(selected[start:start + 2]):
                    x, y = 16 + column * 464, 75 + row_number * 478
                    draw.text((x, y), r['sample'] + ' | ' + r['side'], fill='white', font=font)
                    with Image.open(PUBLIC / r[version + '_file']) as im:
                        assert im.size == (448, 448)
                        canvas.paste(im.convert('RGB'), (x, y + 24))
            filename = f'{split}_compare_{start // 2 + 1:02d}.png'
            target = PUBLIC / filename
            canvas.save(target)
            with Image.open(target) as rendered:
                for column, (version, _) in enumerate(VERSIONS):
                    for row_number, r in enumerate(selected[start:start + 2]):
                        x, y = 16 + column * 464, 99 + row_number * 478
                        with Image.open(PUBLIC / r[version + '_file']) as im:
                            assert np.array_equal(np.asarray(rendered.crop((x, y, x + 448, y + 448))), np.asarray(im.convert('RGB')))
            pages.append({'file': filename, 'sha256': sha(target)})
    return pages


def main():
    started = time.time()
    assert sha(Path(base.__file__)) == 'f72f89bcc344eed7dbffd694ad377d8c625037739ea99de82a886832ca083bd3'
    assert e.identity() == json.loads((e.OUT / 'identity.json').read_text())
    assert sha(PROJECT / 'data/manifests/chen_yolo_usable448_v1_20261111/summary.json') == '2bed6726a57313ebc0d39f4bc93ae57e037505b64abb6c4d13a9ed8ab00072d6'
    authorization = json.loads((e.p.OUT / 'preview_transfer_authorization_20261111.json').read_text())
    assert authorization['user_answer'] == 'joint-anomaly review and spot-check previews for this round are allowed'
    OUT.mkdir(parents=True, exist_ok=False)
    PUBLIC.mkdir()
    DATA.mkdir(parents=True, exist_ok=False)
    save(OUT / 'authorization.json', {'user_request': 'Show me a comparison with a direct 2.0x expansion',
         'scope': 'the original 12 train/val examples, expanding the original YOLO box width and height directly by 2.0x, keeping 448 output',
         'parent_preview_authorization_sha256': sha(e.p.OUT / 'preview_transfer_authorization_20261111.json'),
         'full_export_started': False, 'no_patient_identity_or_kl_transfer': True, 'raw_dicom_transfer': False})
    e.initialize()
    rows = {r['case_id']: r for r in json.loads((e.p.OUT / 'selection_private.json').read_text())}
    selected = json.loads((PREVIOUS / 'outputs/selection_private.json').read_text())
    assert len(selected) == 12
    public, private = [], []
    for number, prev in enumerate(selected, start=1):
        item = prev['parent']
        gray, valid, source, mapping = base.native_gray(item, rows[item['case_id']])
        old = base.crop(gray, valid, item['bbox'], item)
        prior = base.crop(gray, valid, prev['bbox'], item)
        for result, saved in [(old, item), (prior, prev)]:
            for field, array_name in [('gray01', 'image'), ('valid_mask', 'valid_mask')]:
                assert sha(saved[field + '_npy']) == saved[field + '_sha256']
                assert np.array_equal(result[array_name], np.load(saved[field + '_npy'], allow_pickle=False))
        box, clipped, detection_geometry = direct_box(item, valid.shape)
        result = base.crop(gray, valid, box, item)
        stem = prev['sample'].lower().replace(' ', '_')
        image_path, mask_path = DATA / (stem + '_gray01.npy'), DATA / (stem + '_valid.npy')
        e.p.npy_save(image_path, result['image'])
        e.p.npy_save(mask_path, result['valid_mask'])
        record = {'sample': prev['sample'], 'split': item['split'], 'side': item['clinical_side'], 'source_boundary_clipped': clipped}
        for version, old_version in [('current', 'current'), ('expand20', 'expanded')]:
            source_file = PREVIOUS / 'preview_public' / prev[old_version + '_file']
            assert sha(source_file) == prev[old_version + '_sha256']
            filename = stem + '_' + version + '.png'
            shutil.copyfile(source_file, PUBLIC / filename)
            record[version + '_file'], record[version + '_sha256'] = filename, sha(PUBLIC / filename)
        filename = stem + '_direct2x.png'
        hidden = base.preview(PUBLIC / filename, result, box, valid.shape)
        record.update(direct2x_file=filename, direct2x_sha256=sha(PUBLIC / filename), privacy_hidden_valid_pixels=hidden)
        public.append(record)
        private.append({**record, 'parent448': item, 'bbox': box, 'detection_geometry': detection_geometry,
                        'native_hw': list(valid.shape), 'source_receipt': source, 'gray_mapping': mapping,
                        'transform': result['transform'], 'gray01_npy': str(image_path), 'gray01_sha256': sha(image_path),
                        'valid_mask_npy': str(mask_path), 'valid_mask_sha256': sha(mask_path),
                        'crosses_image_midline': box[0] < valid.shape[1] / 2 < box[2],
                        'engineering_input_ready': False, 'clinical_quality_approved': False})
        save(OUT / 'status.json', {'phase': 'previewing', 'completed': number, 'target': 12})
    pages = make_pages(public)
    save(OUT / 'selection_private.json', private)
    save(PUBLIC / 'public_index.json', {'factors_relative_to_yolo_detection': [1.3, 1.56, 2.0],
         'middle_factor_is_approximate_due_to_prior_integer_box': True, 'output_size': [448, 448],
         'gray_mapping_unchanged': True, 'samples': public, 'comparison_pages': pages})
    bundle = ROOT / 'preview_bundle.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(PUBLIC.iterdir()):
            assert f.name == 'public_index.json' or f.suffix == '.png'
            archive.write(f, f.name)
    report = {'phase': 'completed', 'factor_relative_to_yolo_detection': 2.0, 'samples': 12,
              'counts': {'train': 6, 'val': 6}, 'output_shape': [1, 448, 448],
              'parent448_exact_reproduction': 12, 'previous20_exact_reproduction': 12,
              'original_detection_geometry_verified': 12,
              'source_boundary_clipped_count': sum(x['source_boundary_clipped'] for x in public),
              'preview_privacy_masked_count': sum(x['privacy_hidden_valid_pixels'] > 0 for x in public),
              'cross_midline_candidates': sum(x['crosses_image_midline'] for x in private),
              'comparison_tiles_pixel_exact': True, 'code_sha256': sha(Path(__file__)),
              'previous_code_sha256': sha(Path(base.__file__)), 'bundle_sha256': sha(bundle),
              'bundle_files': len(list(PUBLIC.iterdir())), 'elapsed_seconds': time.time() - started,
              'full_export_started': False, 'training_started': False, 'test_accessed': False}
    save(OUT / 'completion.json', report)
    save(OUT / 'status.json', {'phase': 'completed', 'completed': 12, 'target': 12})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
