from collections import Counter
import json
from pathlib import Path
import pipeline as p


def main():
    status = json.loads((p.OUT / 'status.json').read_text())
    if status['phase'] != 'completed' or status['completed'] != 304:
        raise RuntimeError('Wait for all 304 committed results')
    summary = json.loads((p.OUT / 'summary.json').read_text())
    verification = json.loads((p.OUT / 'verification.json').read_text())
    exposure = json.loads((p.OUT / 'patient_exposure_private.json').read_text())
    registry = {r['patient_id']: r['split'] for r in exposure}
    exposure_by_id = {r['patient_id']: r['author_public_group'] for r in exposure}
    selection = json.loads((p.OUT / 'selection_private.json').read_text())
    records = [json.loads((p.DATA / r['case_id'] / 'result.json').read_text()) for r in selection]
    new = [r for r in records if r['sample_group'] == 'random256']
    detail = []
    for dimension in ['modality', 'detector_public_group']:
        keys = sorted({r['modality'] if dimension == 'modality' else exposure_by_id[r['patient_id']] for r in new})
        for key in keys:
            subset = [r for r in new if (r['modality'] if dimension == 'modality' else exposure_by_id[r['patient_id']]) == key]
            detail.append({'dimension': dimension, 'value': key, 'films': len(subset),
                           'old_rois': sum(len(r['old']) for r in subset),
                           'yolo_boxes': sum(len(r['yolo']) for r in subset),
                           'yolo_two_box_films': sum(len(r['yolo']) == 2 for r in subset),
                           'roi_review_flags': sum(bool(x['flags']) for r in subset for x in r['yolo'])})
    same = yolo_only = old_only = 0
    for r in new:
        old = {x['image_position'] for x in r['old']}
        yolo = {x['image_position'] for x in r['yolo']}
        same += len(old & yolo)
        yolo_only += len(yolo - old)
        old_only += len(old - yolo)
    detail_summary = {'random256_strata': detail, 'random256_common_sides': same,
                      'random256_yolo_only_sides': yolo_only, 'random256_old_only_sides': old_only}
    p.save(p.OUT / 'stratified_summary.json', detail_summary)
    inventory = json.loads((p.OUT / 'public_detector_directory.json').read_text())
    plan_rows = []
    for member in inventory['files']:
        patient = Path(member).stem
        if patient in registry:
            plan_rows.append({'patient_id': patient, 'split': registry[patient], 'public_h5_member': member,
                              'contents_validated': False, 'test_sealed': registry[patient] == 'test'})
    if len({r['patient_id'] for r in plan_rows}) != len(plan_rows):
        raise RuntimeError('Public plan has repeated patient files')
    p.save(p.OUT / 'clean_detector_split_plan_private.json', plan_rows)
    plan = {'candidate_patients_by_existing_project_split': dict(Counter(r['split'] for r in plan_rows)),
            'public_directory_sha256': p.sha(p.OUT / 'public_detector_directory.json'),
            'existing_patient_split_preserved': True, 'h5_contents_validated': False,
            'new_training_started': False, 'training_authorized': False,
            'weights_policy': 'Do not use OAI-trained Chen checkpoint as initialization when claiming no evaluation-patient exposure; use eligible generic initialization or random initialization.',
            'training_targets': 'knee bounding boxes only; no KL targets for localizer',
            'next_prerequisites': ['download and validate train/val annotations on server data directory',
                                   'freeze detection-only implementation and evaluate resume with synthetic inputs',
                                   'specify final configuration and measured budget, then obtain explicit training confirmation'],
            'test_policy': 'filename/ID membership audit only; do not load test H5 or tune using test data',
            'candidate_counts_are_not_ready_training_counts': True}
    p.save(p.OUT / 'clean_detector_next_stage_plan.json', plan)
    overlap = json.loads((p.OUT / 'author_patient_overlap_summary.json').read_text())
    prepared = json.loads((p.OUT / 'preparation_summary.json').read_text())
    counts = summary['groups']
    lines = ['# YOLO crop stage 2 results (2026-11-11)', '',
             'CPU cropping and engineering acceptance are complete for 304 training patients: the original 48 regression cases plus 256 new patients drawn with a fixed seed, one eligible PA film per patient. No new classifier or localizer training was started.', '',
             '## Current conclusion', '',
             'YOLO can serve as a batch localization candidate, but an automatic detection is not an acceptable crop. This stage has no manual reference boxes, formal reader approval or downstream KL results. The author public training/model-selection patients overlap with this project evaluation patients; a formal patient-independent study must first settle the data boundary of the localizer.', '',
             '## Crops on the same films', '',
             '|Sample|Films|Old-method ROIs|YOLO boxes|Films with both knees detected by YOLO|YOLO ROIs saved|ROIs flagged for review|Missed knees|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, label in [('regression48', 'Original 48 (failure-enriched)'), ('random256', 'New 256 (random patients)')]:
        g = counts[group]
        lines.append(f"|{label}|{g['films']}|{g['old_valid_rois']}|{g['yolo_boxes']}|{g['yolo_two_box_films']}|{g['yolo_saved_float_rois']}|{g['yolo_rois_for_review']}|{g['missing_yolo_sides']}|")
    g = counts['random256']
    lines += ['', f'In the new 256 cases, knee sides detected by both methods: {same}; extra YOLO detections: {yolo_only}; old method only: {old_only}. These are output/technical retention counts, not accuracy supported by ground truth.', '',
              'The random sample is drawn by patient and cannot be read as the overall detection rate over all visit images. Patients in the author training directory are also in this engineering sample, so it is not an independent detector test set.', '',
              '## Fixed pipeline', '',
              '- Native DICOM read-only; per-archive CRC and SHA, native pixel fingerprints and frozen lists are checked.',
              '- The frozen display gray and whole-film 0.5/99.5 percentiles are reused; the gray policy of earlier experiments is not changed silently.',
              '- Resample to 0.14 mm/px, center 2048x2560, then shrink to 256x320; detector input is 0-255 float32 with mean 0.431 and std 0.0872, without an extra division by 255.',
              '- Original author weights fixed: score>=0.12, NMS IoU=0.3, box width/height expanded 1.3x and mapped back to the native image for cropping. NMS uses the fixed source value, not the 0.7 stated in the paper.',
              '- Output [1,224,224] float32 with a bool valid-pixel mask; aspect-preserving letterbox; no flipping. Image left -> right knee R, image right -> left knee L.',
              '- When YOLO misses a box, only a separate geometric fallback candidate is saved and not mixed into the main list. Reasons, sources, coordinates and fingerprints are saved per case.',
              '- All training_ready=false and clinical_quality_approved=false; no automatic flag does not mean manual approval.', '',
              '## Gray levels and manual review', '',
              'High saturation >=10%, low saturation >=25%, valid pixels <95%, P95-P05 <0.15, as well as clipped boxes, midline crossing, padding and duplicate boxes, are engineering review triggers, not clinically validated exclusion thresholds. Continuous metrics are kept for review; no patient is removed automatically because of these flags.', '',
              'In C39, 44.70% of the original pixels exceed the upper bound of the first window after rescale and become maximum brightness; C10 and C38 also have high-end clipping from windowing. This does not mean the original radiograph was overexposed, and a different box cannot restore already clipped gray levels.', '',
              'The review book is at outputs/review_server_only/index.html on the server; the 608-row human_review_template.csv has no pre-filled conclusions. Check the joint space, medial/lateral margins, femoral/tibial coverage, gray detail and left/right linkage. A qualified reader must record name/qualification, time, accept/re-crop/exclude and the reason; results are saved separately and must not overwrite the template as if they were historical records. No new case images or patient lists were copied to the Mac.', '',
              '## Localizer patient overlap', '',
              '|Scope in this project|Patients|In author train H5|In author val H5|Not in author train/val|',
              '|---|---:|---:|---:|---:|']
    for split, label in [('train', 'Current training image cohort'), ('val', 'Current validation image cohort')]:
        d = overlap['prepared_cohort_overlap'][split]
        lines.append(f"|{label}|{d['patients']}|{d['author_train']}|{d['author_val']}|{d['outside_author_train_and_val']}|")
    test = overlap['registry_overlap']['test']
    lines += [f"|Sealed test patient registry|{test['patients']}|{test['author_train']}|{test['author_val']}|{test['patients']-test['author_train']-test['author_val']}|", '',
              'The evidence is the publicly released detection H5 split and this project patient registry; the full epoch-470 training log was not available, so directory membership is not equated with the actual training record. This is not a confirmed leak of test KL labels, but full-pipeline patient independence cannot be claimed from the classifier patient split alone. Test images and disease labels were not opened.', '',
              'New localizer candidate splits were generated along the existing patient boundary of this project: 3304 train, 406 val and 420 test (test stays sealed; only directory IDs were checked). These are counts of candidate public annotation files, not yet fully downloaded or verified. Removing the evaluation-patient contact of the current weights cannot be done by further fine-tuning the same author weights; an initialization that never saw these patients must be trained along this project boundary. New training must first freeze the implementation, configuration and budget, pass recovery acceptance and get explicit confirmation.', '',
              '## Engineering acceptance and full scale', '',
              f"Read back {verification['saved_rois_checked']} saved ROIs, checking SHA, shape/dtype/value range/mask and same-side labels; the boxes of both methods for the original 48 cases equal the previous version exactly. An explicit resume added only the new cases and left existing files unchanged; a deliberately corrupted temporary copy was rejected by checkpoint verification.", '',
              f"The full source list is frozen: train {prepared['cohort_films']['train']} films/{prepared['cohort_patients']['train']} patients, val {prepared['cohort_films']['val']} films/{prepared['cohort_patients']['val']} patients, with disjoint patients. Theoretical maximum is 34266 train knees and 4262 val knees; full YOLO cropping and QC have not been run, so these are not final usable train/val counts.", '',
              'This stage processed only 304 train patients. Formal training lists, existing models, patient splits and earlier results were not changed.', '',
              '## Traceable sources and files', '',
              '- [Chen author source](https://github.com/PingjunChen/GradingKneeOA), fixed commit dc19c2462111c3c5b27066e738fc11ddf57a6407.',
              '- [Author public data v1](https://data.mendeley.com/datasets/56rmx5bjcr/1); the detection H5 split was checked through the ZIP central directory.',
              '- [Original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9531250/). This project does not reproduce every original DICOM gray detail.',
              f'- Server experiment: `{p.ROOT}`.', f'- Server float ROI: `{p.DATA}`.',
              '- outputs/identity.json fixes the run identity; summary.json and stratified_summary.json hold statistics; verification.json and resume_preflight.json hold acceptance checks; author_patient_overlap_summary.json is the overlap audit; gray_window_audit.json is the gray evidence; clean_detector_next_stage_plan.json is the data-boundary plan for the next stage.', '']
    (p.ROOT / 'STAGE_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'details': detail_summary, 'clean_detector_plan_counts': plan['candidate_patients_by_existing_project_split'],
                      'report': str(p.ROOT / 'STAGE_REPORT.md')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
