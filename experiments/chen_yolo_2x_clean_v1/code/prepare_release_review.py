from collections import Counter
import json
import audit_rulers as a


def main():
    root = a.ROOT
    output = root / 'release_v1'
    verification = output / 'verification.json'
    report = json.loads(verification.read_text())
    assert report['all_arrays_and_source_hashes_verified'] and report['all_modified_pixels_verified']
    assert report['verification_code_sha256'] == a.sha(root / 'code/finalize_clean.py')
    reconcile = json.loads((output / 'preview_reconciliation.json').read_text())
    assert reconcile['all_final_array_hashes_match_viewed_previews']
    names = ['train_check_a', 'val_check_a', 'edge_check_a', 'edge_check_b', 'edge_check_c']
    bundles = []
    unique = set()
    edge_keys = set()
    reviewed_counts = Counter()
    review_digests = {}
    hold_count = 0
    for name in names:
        receipt_path = root / 'code' / (name + '_review.json')
        review = json.loads(receipt_path.read_text())
        stage = root / review['relative_stage']
        assert a.sha(stage / (name + '.zip')) == review['sha256']
        public = json.loads((stage / name / 'public_index.json').read_text())
        private = json.loads((stage / (name + '_private.json')).read_text())
        assert len(public['records']) == len(private) == review['images_viewed']
        assert review['pages_actually_viewed'] == list(range(1, len(public['pages']) + 1))
        for page in public['pages']:
            assert a.sha(stage / name / page['file']) == page['sha256']
        aliases = {r['alias'] for r in private}
        assert set(review['hold_aliases']) <= aliases
        for r in private:
            case = json.loads((root / 'edge_finish_v1/cases' / (r['item']['case_id'] + '.json')).read_text())
            e = next(x for x in case['entries'] if x['key'] == r['key'])
            for field in ['gray01_sha256', 'valid_mask_sha256', 'cleanup_status']:
                assert e[field] == r['item'][field], 'viewed_preview_not_final_output'
            if r['key'] not in unique:
                reviewed_counts[e['split']] += 1
            unique.add(r['key'])
            if name.startswith('edge_'):
                assert r['key'] not in edge_keys
                edge_keys.add(r['key'])
        hold_count += len(review['hold_aliases'])
        review_digests[name] = a.sha(receipt_path)
        bundles.append({k: review[k] for k in [
            'relative_stage', 'name', 'sha256', 'images_viewed', 'pages_actually_viewed', 'hold_aliases',
        ]})
    expected_edges = set()
    for file in (root / 'edge_finish_v1/cases').glob('*.json'):
        for e in json.loads(file.read_text())['entries']:
            if e['edge_extra_mask_pixels'] or (e['baseline_cleanup_status'].startswith('accepted') and not e['engineering_input_ready']):
                expected_edges.add(e['key'])
    assert edge_keys == expected_edges
    receipt = {
        'approved_for_engineering_manifest': True,
        'verification_sha256': a.sha(verification),
        'all_exported_pages_actually_viewed': True,
        'preview_scope': 'only the five listed final actual-output preview bundles; not every dataset image',
        'preview_bundles': bundles, 'review_record_sha256': review_digests,
        'unique_final_images_visually_reviewed': len(unique),
        'visually_reviewed_split_counts': dict(reviewed_counts),
        'all_edge_supplement_changed_or_newly_held_images_visually_reviewed': True,
        'edge_supplement_images_visually_reviewed': len(edge_keys),
        'additional_visual_hold_images': hold_count,
        'additional_visual_hold_reason': 'one peripheral removal strip affected a metallic-like short line; original preserved and excluded',
        'preview_reconciliation_sha256': a.sha(output / 'preview_reconciliation.json'),
        'all_images_automatically_audited': True,
        'full_manual_review_completed': False, 'clinical_quality_approved': False,
        'training_started': False, 'test_accessed': False,
        'notes': 'Raw arrays and all masks remain preserved. Final usable means engineering preprocessing criteria only.',
    }
    assert not (output / 'final_visual_review.json').exists()
    a.save(output / 'final_visual_review.json', receipt)
    print(json.dumps({k: receipt[k] for k in [
        'unique_final_images_visually_reviewed', 'visually_reviewed_split_counts',
        'edge_supplement_images_visually_reviewed', 'additional_visual_hold_images',
    ]}), flush=True)


if __name__ == '__main__':
    main()
