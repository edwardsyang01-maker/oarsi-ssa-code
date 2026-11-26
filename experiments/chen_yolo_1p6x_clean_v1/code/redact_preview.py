import json
import zipfile
from PIL import Image, ImageDraw
import audit_rulers as a


def main():
    stage = a.ROOT / 'edge_finish_v1'
    name, alias = 'pilot_train', 'PILOT_TRAIN-012'
    out = stage / name
    public = json.loads((out / 'public_index.json').read_text())
    record = next(r for r in public['records'] if r['alias'] == alias)
    page = next(p for p in public['pages'] if alias in p['aliases'])
    index = page['aliases'].index(alias)
    x, y = (index % 2) * 912, (index // 2) * 480
    rect = [0, 400, 180, 448]
    path = out / page['file']
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    draw = ImageDraw.Draw(image)
    for offset in [4, 460]:
        draw.rectangle([x + offset + rect[0], y + 28 + rect[1], x + offset + rect[2] - 1, y + 28 + rect[3] - 1], fill='black')
    image.save(path)
    record['preview_privacy_rectangles'] = [rect]
    page['sha256'] = a.sha(path)
    a.save(out / 'public_index.json', public)
    bundle = stage / (name + '.zip')
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.iterdir()):
            z.write(file, file.name)
    receipt_path = stage / (name + '_receipt.json')
    receipt = json.loads(receipt_path.read_text())
    receipt.update(sha256=a.sha(bundle), preview_text_region_redacted=True, medical_arrays_unchanged=True)
    a.save(receipt_path, receipt)
    print(json.dumps({'bundle': name, 'sha256': receipt['sha256'], 'medical_arrays_unchanged': True}))


if __name__ == '__main__':
    main()
