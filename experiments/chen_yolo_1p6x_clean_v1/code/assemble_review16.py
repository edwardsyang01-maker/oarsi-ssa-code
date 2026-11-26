import json
import pipeline16 as p


def main():
    report = json.loads((p.OUT / 'automatic_completion.json').read_text())
    assert report['all_images_automatically_audited'] and report['all_modified_pixels_verified']
    names = ['pilot_visual_review.json', 'recovered_a_review.json', 'stagea_review.json',
             'edge_stagea_review.json', 'train_stageb_review.json', 'recovered_b_review.json',
             'final_batch_review.json']
    bundles, sources = [], {}
    for name in names:
        path = p.OUT / name
        d = json.loads(path.read_text())
        assert not d['full_manual_review_completed'] and not d['clinical_quality_approved']
        if 'preview_bundles' in d:
            bundles.extend(d['preview_bundles'])
        else:
            bundles.append({k: d[k] for k in ['stage', 'name', 'sha256', 'images_viewed', 'pages_actually_viewed', 'hold_aliases']})
        sources[name] = p.sha(path)
    assert len(bundles) == len({b['name'] for b in bundles}) == 12
    assert sum(b['images_viewed'] for b in bundles) == 382
    holds = [alias for b in bundles for alias in b['hold_aliases']]
    assert holds == ['TRAIN_STAGEB-027']
    receipt = {'approved_for_engineering_manifest': True,
               'automatic_completion_sha256': p.sha(p.OUT / 'automatic_completion.json'),
               'full_manual_review_completed': False, 'clinical_quality_approved': False,
               'preview_bundles': bundles, 'source_review_receipt_sha256': sources,
               'images_actually_viewed_before_private_deduplication': 382,
               'additional_visual_hold_aliases': holds,
               'observations': [
                   'All 382 1.6x before/after pairs on 67 pages in 12 bundles were viewed; the publisher re-checks every preview against the final arrays and all edge supplement samples.',
                   'TRAIN_STAGEB-027: the cleaning boundary is close to the lateral femoral margin; quarantined for uncertainty, without claiming confirmed bone-margin damage.',
                   'The original automatic quarantine is kept and not relaxed by viewing; full automatic checks are not per-image manual or clinical quality approval.'
               ]}
    target = p.OUT / 'final_visual_review.json'
    assert not target.exists()
    p.save(target, receipt)
    print(json.dumps({'review_ready': True, 'viewed_images': 382, 'extra_visual_holds': len(holds),
                      'sha256': p.sha(target)}), flush=True)


if __name__ == '__main__':
    main()
