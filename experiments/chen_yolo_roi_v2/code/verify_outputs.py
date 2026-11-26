import copy
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pipeline as p


def main():
    run_id = p.verify_identity()
    selection = json.loads((p.OUT / 'selection_private.json').read_text())
    total = 0
    first_saved = None
    first_row = None
    for row in selection:
        assert p.check_checkpoint(row, run_id), row['case_id']
        result = json.loads((p.DATA / row['case_id'] / 'result.json').read_text())
        assert result['split'] == 'train'
        assert result['training_ready'] is False and result['clinical_quality_approved'] is False
        for roi in result['yolo'] + result['fallback']:
            assert roi['clinical_side'] == p.SIDES[roi['image_position']]
            assert roi['kl'] == row['labels'][roi['clinical_side']]
            assert roi['training_ready'] is False and roi['clinical_quality_approved'] is False
            if roi['status'] != 'saved':
                continue
            image = np.load(roi['gray01_npy'], allow_pickle=False)
            valid = np.load(roi['valid_mask_npy'], allow_pickle=False)
            assert image.shape == valid.shape == (1, 224, 224)
            assert image.dtype == np.float32 and valid.dtype == bool
            assert valid.any() and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
            total += 1
            if first_saved is None:
                first_saved, first_row = result, row
    if first_saved is None:
        raise RuntimeError('No output to validate')
    temp = Path(tempfile.mkdtemp(prefix='resume_corruption_check_', dir=p.ROOT))
    original_data = p.DATA
    corruption_rejected = False
    try:
        bad = copy.deepcopy(first_saved)
        roi = next(x for x in bad['yolo'] if x['status'] == 'saved')
        altered = np.load(roi['gray01_npy'], allow_pickle=False).copy()
        altered.flat[0] = np.float32(.1234567 if altered.flat[0] != np.float32(.1234567) else .7654321)
        corrupt_path = temp / 'corrupted.npy'
        np.save(corrupt_path, altered, allow_pickle=False)
        roi['gray01_npy'] = str(corrupt_path)
        p.save(temp / first_row['case_id'] / 'result.json', bad)
        p.DATA = temp
        try:
            p.check_checkpoint(first_row, run_id)
        except RuntimeError as exc:
            corruption_rejected = 'hash mismatch' in str(exc)
        assert corruption_rejected
    finally:
        p.DATA = original_data
        shutil.rmtree(temp)
    summary = json.loads((p.OUT / 'summary.json').read_text())
    assert summary['regression48_bbox_mismatches'] == []
    assert len(selection) == len({r['patient_id'] for r in selection}) == 304
    p.save(p.OUT / 'verification.json', {
        'run_id': run_id, 'cases_checked': len(selection), 'saved_rois_checked': total,
        'all_output_hashes_match': True, 'shape_dtype_range_mask_checked': True,
        'same_side_label_mapping_checked': True, 'train_only_unique_patients': True,
        'corrupt_checkpoint_rejected': corruption_rejected, 'pilot48_boxes_identical': True,
        'clinical_quality_approved': False, 'training_ready': False,
    })
    print((p.OUT / 'verification.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
