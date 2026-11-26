import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pipeline as p


def verify(smoke=False):
    run_id = p.verify_identity()
    selection = json.loads((p.OUT / 'selection_private.json').read_text())
    if smoke:
        chosen = set(json.loads((p.OUT / 'smoke_selection_private.json').read_text()))
        selection = [r for r in selection if r['case_id'] in chosen]
    registry = {r['patient_id']: r['split'] for r in p.read_csv(p.SPLIT)}
    count, states, first, first_row = Counter(), Counter(), None, None
    train_patients, val_patients = set(), set()
    sample_keys = set()
    for row in selection:
        if not p.check_checkpoint(row, run_id):
            raise RuntimeError('Missing required case checkpoint: ' + row['case_id'])
        result = json.loads((p.DATA / row['case_id'] / 'result.json').read_text())
        assert registry[row['patient_id']] == row['split'] in ['train', 'val']
        (train_patients if row['split']=='train' else val_patients).add(row['patient_id'])
        states[result['status']] += 1
        assert result['training_ready'] is False and result['clinical_quality_approved'] is False
        if result['status'] == 'completed':
            assert result['source_verified'] is True
            assert result['source_receipt']['archive_sha256'] == row['archive_sha256']
            assert result['source_receipt']['native_pixel_sha256'] == row['native_pixel_sha256']
        for method in ['yolo', 'fallback']:
            positions = [x['image_position'] for x in result[method]]
            assert len(positions) == len(set(positions)) <= 2
            for roi in result[method]:
                assert roi['clinical_side'] == p.SIDES[roi['image_position']]
                assert roi['kl'] == row['labels'][roi['clinical_side']]
                assert roi['training_ready'] is False and roi['clinical_quality_approved'] is False
                if roi['status'] != 'saved':
                    continue
                key = (row['film_id'], method, roi['clinical_side'])
                assert key not in sample_keys
                sample_keys.add(key)
                image = np.load(roi['gray01_npy'], allow_pickle=False)
                valid = np.load(roi['valid_mask_npy'], allow_pickle=False)
                assert image.shape == valid.shape == (1, 224, 224)
                assert image.dtype == np.float32 and valid.dtype == bool
                assert valid.any() and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
                assert Path(roi['gray01_npy']).is_relative_to(p.DATA / row['case_id'])
                count[(row['split'], method)] += 1
                if first is None and method == 'yolo':
                    first, first_row = result, row
    assert not train_patients & val_patients
    summary = json.loads((p.OUT / 'summary.json').read_text())
    assert not summary['prior304_regression_mismatches']
    if not smoke:
        assert summary['all_candidates_complete'] and summary['prior304_regression_checked'] == 304
    elif states['case_error']:
        raise RuntimeError('Smoke cases include a processing failure')
    receipt = {
        'run_id': run_id, 'scope': 'smoke' if smoke else 'all_trainval',
        'cases_checked': len(selection), 'case_statuses': dict(states),
        'saved_rois_checked': {split: {method: count[(split,method)] for method in ['yolo','fallback']}
                               for split in ['train','val']},
        'all_output_hashes_match': True, 'shape_dtype_range_mask_checked': True,
        'patient_split_disjoint': True, 'label_association_matches_position_policy': True,
        'individual_anatomical_side_verified': False,
        'prior_regression_mismatches': summary['prior304_regression_mismatches'],
        'clinical_quality_approved': False, 'training_ready': False,
        'test_images_or_labels_opened': False, 'detector_retraining': False,
    }
    if smoke:
        if first is None:
            raise RuntimeError('Smoke produced no saved YOLO crop')
        temp = Path(tempfile.mkdtemp(prefix='resume_probe_', dir=p.ROOT))
        original_data = p.DATA
        try:
            bad = copy.deepcopy(first)
            roi = next(x for x in bad['yolo'] if x['status']=='saved')
            altered = np.load(roi['gray01_npy'], allow_pickle=False).copy()
            altered.flat[0] = .1234567 if altered.flat[0] != np.float32(.1234567) else .7654321
            changed = temp / 'corrupt.npy'
            np.save(changed, altered, allow_pickle=False)
            roi['gray01_npy'] = str(changed)
            p.save(temp / first_row['case_id'] / 'result.json', bad)
            p.DATA = temp
            rejected = False
            try:
                p.check_checkpoint(first_row, run_id)
            except RuntimeError as exc:
                rejected = 'hash mismatch' in str(exc)
            assert rejected
        finally:
            p.DATA = original_data
            shutil.rmtree(temp)
        command = [sys.executable, str(p.ROOT/'code/pipeline.py'), 'run', '--smoke', '--workers', '2']
        denied = subprocess.run(command, capture_output=True, text=True)
        assert denied.returncode != 0 and 'Existing results require explicit --resume' in denied.stderr
        before = {r['case_id']:p.sha(p.DATA/r['case_id']/'result.json') for r in selection}
        subprocess.run(command + ['--resume'], check=True, capture_output=True, text=True)
        after = {r['case_id']:p.sha(p.DATA/r['case_id']/'result.json') for r in selection}
        assert before == after
        receipt.update(corrupt_checkpoint_rejected=True, implicit_resume_rejected=True,
                       explicit_resume_preserves_completed_files=True)
    receipt['passed'] = True
    p.save(p.OUT / ('preflight.json' if smoke else 'verification.json'), receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    verify(args.smoke)


if __name__ == '__main__':
    main()
