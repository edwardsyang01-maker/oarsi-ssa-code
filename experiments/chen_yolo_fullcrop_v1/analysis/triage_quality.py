from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import pipeline as p

GEOMETRY = {'expanded_box_clipped', 'crosses_image_midline', 'detector_frame_edge',
            'multiple_detections_same_side', 'bilateral_boxes_overlap', 'low_valid_fraction'}
GRAY = {'high_saturation_fraction', 'low_saturation_fraction', 'low_gray_dynamic_range'}
KNOWN = GEOMETRY | GRAY | {'detector_center_padding'}


def padding_structure_ok(result, roi):
    height, width = result['native_hw']
    x0,y0,x1,y1 = roi['bbox']
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return False
    if roi['gray_metrics']['valid_fraction'] < p.CONFIG['qc']['min_valid_fraction']:
        return False
    for kind in ['gray01','valid_mask']:
        if p.sha(Path(roi[kind+'_npy'])) != roi[kind+'_sha256']:
            return False
    image = np.load(roi['gray01_npy'],allow_pickle=False)
    mask = np.load(roi['valid_mask_npy'],allow_pickle=False)
    return (image.shape == mask.shape == (1,224,224)
            and image.dtype == np.float32 and mask.dtype == bool and bool(mask.any())
            and bool(np.isfinite(image).all()) and float(image.min()) >= 0
            and float(image.max()) <= 1)


def main():
    run_id = p.verify_identity()
    selection = json.loads((p.OUT/'selection_private.json').read_text())
    by_case = {r['case_id']:r for r in selection}
    paths = sorted(p.DATA.glob('*/result.json'))
    queue, source_receipts = [], []
    counts = {s:Counter() for s in ['train','val']}
    flags = {s:Counter() for s in counts}
    actions = {
        'technical_failure':'check decoding, source and output integrity; cannot be fixed by changing the box alone',
        'missing_detection':'check the original film and the independent old-method candidate; re-locate the crop when the source allows',
        'geometry_review':'check the joint space, medial/lateral bone margins and femoral/tibial coverage; re-crop if needed',
        'gray_review':'compare with the original film and display processing to judge whether detail is preserved; do not exclude on saturation fraction alone',
        'padding_only_information':'file and box checks passed; keep the padding hint and spot-check coverage, no automatic re-crop required',
        'other_review':'unresolved flags or structural anomalies; keep for review, cannot be passed automatically',
        'routine_sample':'no automatic flag; include in the routine spot check stratified by device and visit',
    }
    for path in paths:
        result = json.loads(path.read_text())
        row = by_case[result['case_id']]
        assert result['run_id'] == run_id and result['split'] == row['split']
        source_receipts.append({'case_id':row['case_id'],'result_sha256':p.sha(path)})
        for pos, side in p.SIDES.items():
            roi = next((x for x in result['yolo'] if x['image_position']==pos),None)
            found_flags = set(roi['flags']) if roi else set()
            structure_checked = False
            if result['status'] != 'completed' or (roi and roi['status']!='saved'):
                group = 'technical_failure'
            elif roi is None:
                group = 'missing_detection'
            else:
                assert roi['clinical_side']==side and roi['kl']==row['labels'][side]
                assert not roi['training_ready'] and not roi['clinical_quality_approved']
                if found_flags - KNOWN:
                    group = 'other_review'
                elif found_flags & GEOMETRY:
                    group = 'geometry_review'
                elif found_flags & GRAY:
                    group = 'gray_review'
                elif found_flags == {'detector_center_padding'}:
                    structure_checked = padding_structure_ok(result,roi)
                    group = 'padding_only_information' if structure_checked else 'other_review'
                else:
                    assert not found_flags
                    group = 'routine_sample'
            counts[row['split']][group] += 1
            flags[row['split']].update(found_flags)
            queue.append({
                'case_id':row['case_id'],'split':row['split'],'modality':row['modality'],
                'visit':row['visit'],'image_position':pos,'clinical_side':side,
                'primary_review_group':group,'action':actions[group],
                'original_flags':';'.join(sorted(found_flags)),
                'has_gray_flag':bool(found_flags & GRAY),'has_geometry_flag':bool(found_flags & GEOMETRY),
                'padding_structure_checked':structure_checked,
                'yolo_status':roi['status'] if roi else 'missing',
                'review_panel':str(p.OUT/'review_server_only'/(row['case_id']+'.png')),
                'gray01_npy':roi.get('gray01_npy','') if roi else '',
                'original_result':str(path),'clinical_quality_approved':False,
                'auto_excluded':False,'review_decision':'','reviewer':'','review_time':'','notes':'',
            })
    assert len(queue)==2*len(paths)
    assert sum(sum(v.values()) for v in counts.values())==len(queue)
    p.write_csv(p.OUT/'quality_triage_private.csv',queue,list(queue[0]))
    priority = [r for r in queue if r['primary_review_group'] not in ['routine_sample','padding_only_information']]
    if priority:
        p.write_csv(p.OUT/'quality_priority_review_private.csv',priority,list(queue[0]))
    p.save(p.OUT/'quality_triage_source_receipts_private.json',source_receipts)
    summary = {
        'run_id':run_id,'snapshot_utc':datetime.now(timezone.utc).isoformat(),
        'analysis_code_sha256':p.sha(Path(__file__)),'processed_films':len(paths),
        'target_films':len(selection),'is_partial_snapshot':len(paths)!=len(selection),
        'roi_positions_accounted':len(queue),'groups':{s:dict(c) for s,c in counts.items()},
        'original_flag_counts':{s:dict(c) for s,c in flags.items()},
        'priority_review_rows':len(priority),'original_files_or_flags_modified':False,
        'automatic_exclusions':0,'automatic_clinical_approvals':0,
        'side_policy':{'image_left':'R','image_right':'L','header_not_used_for_side':True},
        'kl_labels_or_classifier_predictions_used_to_triage':False,
        'interpretation':'Padding-only group passed structural checks, not anatomical/clinical review; all original flags remain',
    }
    p.save(p.OUT/'quality_triage_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
