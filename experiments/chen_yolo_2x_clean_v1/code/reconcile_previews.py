import json
from pathlib import Path
import audit_rulers as a


def main():
    assert (a.ROOT / 'edge_finish_v1/completion.json').exists()
    current = {}
    indices = {}
    n = 0
    for file in sorted((a.ROOT / 'edge_finish_v1/cases').glob('*.json')):
        case = json.loads(file.read_text())
        for e in case['entries']:
            n += 1
            current[e['key']] = e
            indices[e['key']] = n
    bundles = ['clean_v1/train_check_a', 'clean_v1/val_check_a', 'edge_finish_v1/edge_check_a']
    result = []
    changed_indices = []
    for bundle in bundles:
        rows = json.loads((a.ROOT / (bundle + '_private.json')).read_text())
        changed = []
        status_changed = []
        for r in rows:
            old = r['item']
            final = current[r['key']]
            if any(old[k] != final[k] for k in ['gray01_sha256', 'valid_mask_sha256']):
                changed.append(r['alias'])
                changed_indices.append(indices[r['key']])
            if old['cleanup_status'] != final['cleanup_status']:
                status_changed.append({'alias': r['alias'], 'old': old['cleanup_status'], 'final': final['cleanup_status']})
        result.append({'bundle': bundle, 'images': len(rows), 'changed_array_aliases': changed,
                       'status_changes': status_changed, 'all_arrays_same': not changed})
    receipt = {'bundles': result, 'changed_final_indices': sorted(set(changed_indices)),
               'all_final_array_hashes_match_viewed_previews': not changed_indices,
               'final_completion_sha256': a.sha(a.ROOT / 'edge_finish_v1/completion.json')}
    a.save(a.ROOT / 'release_v1/preview_reconciliation.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
