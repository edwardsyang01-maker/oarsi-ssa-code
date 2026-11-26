import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import audit_rulers as a
from review_gallery import safe_image


def rank(row, salt):
    return hashlib.sha256((salt + row['key']).encode()).hexdigest()


def choose(records, kind, limit):
    if kind == 'recovered':
        prior = a.PROJECT / 'data/manifests/chen_yolo_usable448_2x_clean_v1_20261112'
        held = set()
        for split in ['train', 'val']:
            for line in (prior / (split + '_held.jsonl')).open():
                held.add(json.loads(line)['key'])
        rows = [r for r in records if r['key'] in held and r['item']['engineering_input_ready']]
        return sorted(rows, key=lambda r: rank(r, 'recovered'))[:limit]
    if kind in ['all_clean', 'all_held']:
        rows = [r for r in records if (r['item']['cleanup_status'] == 'accepted_cleaned' if kind == 'all_clean' else not r['item']['engineering_input_ready'])]
        return sorted(rows, key=lambda r: rank(r, kind))[:limit]
    if kind == 'edge':
        rows = [r for r in records if r['item'].get('edge_extra_mask_pixels', 0) or
                (r['item'].get('baseline_cleanup_status', '').startswith('accepted') and not r['item']['engineering_input_ready'])]
        return sorted(rows, key=lambda r: rank(r, 'edge'))[:limit]
    split = kind
    rows = [r for r in records if r['item']['split'] == split]
    picked = {}
    def add(group, n, salt):
        for r in sorted(group, key=lambda r: rank(r, salt))[:n]:
            picked[r['key']] = r
    cleaned = [r for r in rows if r['item']['cleanup_status'] == 'accepted_cleaned']
    for r in sorted(cleaned, key=lambda r: -r['item']['proposed_mask_pixels'])[:8]:
        picked[r['key']] = r
    add(cleaned, 12, 'clean-random')
    add([r for r in rows if r['item']['cleanup_status'] == 'accepted_unchanged'], 12, 'unchanged')
    groups = defaultdict(list)
    for r in rows:
        item = r['item']
        groups[('variant', item['variant'], item['cleanup_status'])].append(r)
        for reason in item['cleanup_review_reasons']:
            groups[('reason', reason)].append(r)
    for key in sorted(groups):
        add(groups[key], 2, repr(key))
    for r in sorted(rows, key=lambda r: rank(r, 'fill')):
        if len(picked) >= limit:
            break
        picked[r['key']] = r
    return sorted(picked.values(), key=lambda r: rank(r, 'final'))[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='edge_finish_v1')
    parser.add_argument('--name', required=True)
    parser.add_argument('--kind', choices=['edge', 'train', 'val', 'all_clean', 'all_held', 'recovered'], required=True)
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--exclude-bundle', action='append', default=[])
    args = parser.parse_args()
    root = a.ROOT / args.stage
    records = []
    for file in sorted((root / 'cases').glob('*.json')):
        case = json.loads(file.read_text())
        for item in case['entries']:
            records.append({'key': item['key'], 'item': item, 'native_hw': case['native_hw'],
                            'result_path': str(file), 'result_sha256': a.sha(file)})
    excluded = set()
    for relative in args.exclude_bundle:
        for r in json.loads((a.ROOT / (relative + '_private.json')).read_text()):
            excluded.add(r['key'])
    selected = choose([r for r in records if r['key'] not in excluded], args.kind, args.limit)
    assert selected
    out = root / args.name
    out.mkdir(exist_ok=False)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    public, private, panels = [], [], []
    for n, r in enumerate(selected, 1):
        item = r['item']
        alias = args.name.upper() + '-' + f'{n:03d}'
        assert a.sha(item['gray01_npy']) == item['gray01_sha256']
        assert a.sha(item['valid_mask_npy']) == item['valid_mask_sha256']
        old = {**item, 'gray01_npy': item['raw_gray01_npy'], 'valid_mask_npy': item['raw_valid_mask_npy']}
        _, _, safe, before = safe_image(old, r['native_hw'])
        _, _, _, after = safe_image(item, r['native_hw'])
        rgb = np.repeat(before[:, :, None], 3, axis=2)
        if item['cleanup_mask_npy']:
            mask = np.load(item['cleanup_mask_npy'], allow_pickle=False)[0]
            boundary = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool) & safe
            rgb[boundary] = [255, 110, 50]
        state = 'CLEAN' if item['cleanup_status'] == 'accepted_cleaned' else ('KEEP' if item['engineering_input_ready'] else 'HOLD')
        panel = Image.new('RGB', (912, 480), '#171b22')
        draw = ImageDraw.Draw(panel)
        draw.text((4, 3), f"{alias} {item['split']} {item['clinical_side']} {state} | before / after", fill='white', font=font)
        panel.paste(Image.fromarray(rgb), (4, 28))
        panel.paste(Image.fromarray(after).convert('RGB'), (460, 28))
        panels.append(panel)
        public.append({'alias': alias, 'split': item['split'], 'side': item['clinical_side'],
                       'status': item['cleanup_status'], 'extra_mask_pixels': item.get('edge_extra_mask_pixels', 0),
                       'mask_pixels': item['proposed_mask_pixels'], 'reasons': item['cleanup_review_reasons']})
        private.append({'alias': alias, **r})
    pages = []
    for start in range(0, len(panels), 6):
        subset = panels[start:start + 6]
        canvas = Image.new('RGB', (1824, 480 * ((len(subset) + 1) // 2)), '#171b22')
        for j, panel in enumerate(subset):
            canvas.paste(panel, ((j % 2) * 912, (j // 2) * 480))
        name = f'page_{start // 6 + 1:02d}.png'
        canvas.save(out / name)
        pages.append({'file': name, 'sha256': a.sha(out / name), 'aliases': [r['alias'] for r in public[start:start + 6]]})
    a.save(out / 'public_index.json', {'records': public, 'pages': pages, 'actual_arrays': True})
    a.save(root / (args.name + '_private.json'), private)
    bundle = root / (args.name + '.zip')
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            z.write(f, f.name)
    receipt = {'images': len(public), 'pages': len(pages), 'stage': args.stage, 'kind': args.kind,
               'counts': dict(Counter(r['status'] for r in public)), 'sha256': a.sha(bundle),
               'source_cases_at_export': len(list((root / 'cases').glob('*.json'))),
               'patient_identity_and_kl_exported': False, 'raw_dicom_exported': False}
    a.save(root / (args.name + '_receipt.json'), receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
