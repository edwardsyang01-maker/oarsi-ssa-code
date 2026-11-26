import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
import warnings

import cv2
import numpy as np
import crop_full as crop
import audit_rulers as audit
import ruler_refinement as ref
import proposal_v1 as proposal
import clean_stage as clean
import edge_finish as edge

ROOT = crop.ROOT
OUT = crop.OUT
PRIOR = crop.PROJECT / 'experiments/chen_yolo_2x_clean_v1'
PRIOR_MANIFEST = crop.PROJECT / 'data/manifests/chen_yolo_usable448_2x_clean_v1_20261112'
CODE_FILES = ['pipeline16.py', 'geometry16.py', 'crop_full.py', 'audit_rulers.py',
              'ruler_refinement.py', 'proposal_v1.py', 'clean_policy.py', 'clean_stage.py', 'edge_finish.py']
sha, save = audit.sha, audit.save


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity():
    previous = json.loads((PRIOR / 'release_v1/completion.json').read_text())
    assert previous['phase'] == 'completed' and previous['all_modified_pixels_verified']
    assert sha(PRIOR / 'release_v1/completion.json') == 'ebecc4b5f6835a7b20c5d5d749f6dc4da274ee52cd1a4b004d6b7f312e6d9d74'
    value = {
        'code_sha256': {n: sha(ROOT / 'code' / n) for n in CODE_FILES},
        'crop_identity': crop.identity(),
        'prior_2x_completion_sha256': sha(PRIOR / 'release_v1/completion.json'),
        'parent_1p3_summary_sha256': sha(crop.PARENT / 'summary.json'),
        'policy': {'expansion': 1.6, 'size': 448, 'gray': 'frozen original mapping',
                   'side': 'image_left=R,image_right=L; no flip', 'source': 'native DICOM',
                   'protect_original_1p3_roi': True,
                   'prior_context': 'same-film confirmed ruler geometry and accepted safe masks; no disease labels',
                   'mask_fill': 0, 'originals_retained': True, 'test_accessed': False, 'training_started': False},
    }
    value['identity'] = digest(value)
    return value


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'identity.json').exists()
    ident = identity()
    old = {}
    prior_report = json.loads((PRIOR / 'release_v1/completion.json').read_text())
    for name, expected in prior_report['manifest_sha256'].items():
        path = PRIOR_MANIFEST / name
        assert sha(path) == expected
        for line in path.open():
            e = json.loads(line)
            assert e['key'] not in old
            old[e['key']] = {k: e[k] for k in ['key', 'engineering_input_ready', 'cleanup_status', 'cleanup_review_reasons']}
    tasks = crop.tasks()
    chosen = set(crop.pilot_cases(tasks))
    strata = defaultdict(list)
    for task in tasks:
        case = task['row']['case_id']
        path = PRIOR / 'edge_finish_v1/cases' / (case + '.json')
        task['prior_case_path'], task['prior_case_sha256'] = str(path), sha(path)
        task['prior_final'] = {e['key']: old[e['key']] for e in task['items']}
        task['identity'] = ident['crop_identity']['identity']
        task['pipeline_identity'] = ident['identity']
        for e in task['items']:
            prior = old[e['key']]
            strata[(e['split'], prior['cleanup_status'])].append(case)
            if e['variant'].startswith('lower_') or prior['cleanup_status'] == 'held_final_visual_review' or 'manual_uncertain_fragment' in prior['cleanup_review_reasons']:
                chosen.add(case)
    for cases in strata.values():
        chosen.update(sorted(set(cases), key=lambda s: hashlib.sha256(('1p6pilot' + s).encode()).hexdigest())[:3])
    assert len(old) == sum(len(t['items']) for t in tasks) == 38520
    save(OUT / 'tasks_private.json', tasks)
    save(OUT / 'pilot_cases.json', sorted(chosen))
    save(OUT / 'identity.json', ident)
    save(OUT / 'input_snapshot.json', {'tasks_sha256': sha(OUT / 'tasks_private.json'), 'pilot_cases_sha256': sha(OUT / 'pilot_cases.json')})
    save(OUT / 'authorization.json', {
        'user_request': 'Make another 1.6x version, so there are three versions: 1.3, 1.6 and 2.0',
        'scope': 'native 1.6x 448 version of the existing train/val; reuse ruler cleaning and anomaly quarantine; keep 1.3/2.0; no training',
        'parent_preview_authorization_sha256': sha(crop.e.p.OUT / 'preview_transfer_authorization_20261111.json'),
        'preview_scope': 'necessary joint-anomaly/spot-check previews of the same authorized queue; no identity, KL list or raw DICOM exported',
    })
    print(json.dumps({'prepared': True, 'cases': len(tasks), 'knees': 38520, 'pilot_cases': len(chosen), 'identity': ident['identity']}), flush=True)


def project_mask(mask, source, target):
    yy, xx = np.mgrid[:448, :448]
    nx, ny = ref.native_point(target, xx, yy)
    sx, sy = ref.image_point(source, nx, ny)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    result = np.zeros((448, 448), bool)
    for dx, dy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        x, y = x0 + dx, y0 + dy
        inside = (x >= 0) & (x < 448) & (y >= 0) & (y < 448)
        result[inside] |= mask[y[inside], x[inside]]
    return result


def projected_points_visible(points, source, target, valid, mask):
    remaining = 0
    for point in points:
        nx, ny = ref.native_point(source, point['x'], point['y'])
        x, y = ref.image_point(target, nx, ny)
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < 448 and 0 <= yi < 448 and valid[yi, xi] and not mask[yi, xi]:
            remaining += 1
    return remaining


def process(task):
    started = time.time()
    case, ident = task['row']['case_id'], task['pipeline_identity']
    cv2.setNumThreads(1)
    prior_path = Path(task['prior_case_path'])
    assert sha(prior_path) == task['prior_case_sha256']
    previous = json.loads(prior_path.read_text())
    pp_path = Path(previous['proposal_result_path'])
    assert sha(pp_path) == previous['proposal_result_sha256']
    prior_proposal = json.loads(pp_path.read_text())
    prior_items = {e['key']: e for e in previous['entries']}
    raw_path = crop.DATA / case / 'result.json'
    if raw_path.exists():
        raw = crop.verify_result(task, task['identity'])
    else:
        result = crop.process(task)
        assert result['status'] == 'completed', 'native_crop_failed_see_private_case_record'
        raw = crop.verify_result(task, task['identity'])
    initial = {'case_id': case, 'source_result_sha256': sha(raw_path), 'native_hw': raw['native_hw'], 'entries': []}
    for item in raw['entries']:
        g = np.load(item['gray01_npy'], allow_pickle=False)[0]
        v = np.load(item['valid_mask_npy'], allow_pickle=False)[0]
        detected = audit.detect(g, v)
        lines = [{**ln, 'classification': audit.classify_line(ln, item)} for ln in detected['lines']]
        prior_entry = next(e for e in prior_proposal['entries'] if e['key'] == item['key'])
        for line in prior_entry['lines']:
            assert line['evidence']['high_confidence']
            mapped = ref.project(line, prior_items[item['key']], item)
            mapped['frozen_2x_confirmed_context'] = {'proposal_sha256': sha(pp_path), 'source_count': line['count'], 'source_step': line['step']}
            lines.append(mapped)
        initial['entries'].append({'key': item['key'], 'lines': lines, 'gray_metrics': detected['gray_metrics'],
                                   'coverage': audit.coverage_metrics(item, v), 'same_film_2x_context': True})
    audit_path = ROOT / 'audit_v2/cases' / (case + '.json')
    save(audit_path, initial)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        proposed = proposal.evaluate(raw, initial)
    proposal_folder = ROOT / 'proposals_v1/cases' / case
    for entry in proposed:
        item = next(e for e in raw['entries'] if e['key'] == entry['key'])
        previous_item = prior_items[item['key']]
        final_prior = task['prior_final'][item['key']]
        pp_entry = next(e for e in prior_proposal['entries'] if e['key'] == item['key'])
        v = np.load(item['valid_mask_npy'], allow_pickle=False)[0]
        mask = entry.pop('mask')
        inherited_pixels = 0
        if final_prior['engineering_input_ready'] and previous_item['cleanup_mask_npy']:
            assert sha(previous_item['cleanup_mask_npy']) == previous_item['cleanup_mask_sha256']
            old_mask = np.load(previous_item['cleanup_mask_npy'], allow_pickle=False)[0]
            projected = project_mask(old_mask, previous_item, item) & v
            inherited_pixels = int((projected & ~mask).sum())
            mask |= projected
        risks = list(entry['review_flags'])
        unresolved_prior = 0
        for line in pp_entry['unresolved_lines']:
            if projected_points_visible(line['points'], previous_item, item, v, mask) >= 2:
                unresolved_prior += 1
        for point in previous_item.get('edge_single_fragments', []):
            if projected_points_visible([point], previous_item, item, v, mask):
                unresolved_prior += 1
        if unresolved_prior:
            risks.append('visible_prior_2x_uncertain_candidate')
        if final_prior['cleanup_status'] == 'held_final_visual_review' or 'manual_uncertain_fragment' in final_prior['cleanup_review_reasons']:
            risks.append('prior_visual_uncertainty_needs_review')
        overlap = clean.protection(mask, item)
        if overlap:
            risks.append('mask_touches_original_joint_roi')
        if mask.mean() > .25:
            risks.append('large_mask')
        coverage = next(e for e in initial['entries'] if e['key'] == item['key'])['coverage']
        assert coverage['contains_entire_parent_crop']
        if not coverage['core_inside_output'] or coverage['core_valid_fraction'] < .995:
            risks.append('joint_core_missing_pixels')
        entry['review_flags'] = sorted(set(risks))
        entry['metrics'].update(mask_pixels=int(mask.sum()), mask_fraction=float(mask.mean()), parent_1p3_overlap_pixels=overlap)
        entry['inherited_2x_mask_pixels'] = inherited_pixels
        entry['visible_prior_2x_uncertain_candidates'] = unresolved_prior
        entry['proposed_mask_npy'], entry['proposed_mask_sha256'] = clean.atomic_array(proposal_folder / (item['clinical_side'] + '_proposed_mask.npy'), mask[None])
    proposal_path = proposal_folder / 'result.json'
    save(proposal_path, {'case_id': case, 'source_result_path': str(raw_path), 'source_result_sha256': sha(raw_path),
                         'audit_sha256': sha(audit_path), 'prior_2x_case_sha256': task['prior_case_sha256'], 'entries': proposed})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        clean.worker((str(proposal_path), 'clean_v1', ident, {}))
        edge.worker((str(ROOT / 'clean_v1/cases' / (case + '.json')), 'edge_finish_v1', ident))
    final_path = ROOT / 'edge_finish_v1/cases' / (case + '.json')
    final = json.loads(final_path.read_text())
    stats = clean.counts_for(final['entries'])
    stats['edge_extra_images'] = sum(e['edge_extra_mask_pixels'] > 0 for e in final['entries'])
    receipt = {'case_id': case, 'identity': ident, 'raw_result_sha256': sha(raw_path),
               'final_result_sha256': sha(final_path), 'counts': stats, 'seconds': time.time() - started}
    save(ROOT / 'processed_cases' / (case + '.json'), receipt)
    return stats


def verify_case(task):
    case = task['row']['case_id']
    receipt = json.loads((ROOT / 'processed_cases' / (case + '.json')).read_text())
    assert receipt['identity'] == task['pipeline_identity']
    raw = crop.verify_result(task, task['identity'])
    raw_path = crop.DATA / case / 'result.json'
    final_path = ROOT / 'edge_finish_v1/cases' / (case + '.json')
    assert sha(raw_path) == receipt['raw_result_sha256'] and sha(final_path) == receipt['final_result_sha256']
    final = json.loads(final_path.read_text())
    assert sha(final['proposal_result_path']) == final['proposal_result_sha256']
    assert sha(final['baseline_cleanup_result_path']) == final['baseline_cleanup_result_sha256']
    parent = {e['key']: e for e in task['items']}
    raw_items = {e['key']: e for e in raw['entries']}
    assert set(parent) == set(raw_items) == {e['key'] for e in final['entries']}
    for e in final['entries']:
        for key in ['key', 'split', 'patient_id', 'clinical_side', 'image_position', 'kl', 'variant']:
            assert e[key] == parent[e['key']][key]
        r = raw_items[e['key']]
        assert e['bbox'] == r['bbox'] and e['transform'] == r['transform']
        b = e['detection_geometry']['unclipped_1p3_bbox']
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        dw, dh = (b[2] - b[0]) / 1.3, (b[3] - b[1]) / 1.3
        expected = [cx - .8 * dw, cy - .8 * dh, cx + .8 * dw, cy + .8 * dh]
        assert np.allclose(e['detection_geometry']['requested_1p6_bbox'], expected, rtol=0, atol=1e-8)
        h, w = raw['native_hw']
        box = [max(0, min(w, math.floor(expected[0]))), max(0, min(h, math.floor(expected[1]))),
               max(0, min(w, math.ceil(expected[2]))), max(0, min(h, math.ceil(expected[3])))]
        assert box == e['bbox']
        for field in ['gray01', 'valid_mask']:
            assert sha(e[field + '_npy']) == e[field + '_sha256']
        gray, valid = np.load(e['gray01_npy'], allow_pickle=False), np.load(e['valid_mask_npy'], allow_pickle=False)
        crop.e.check_array(gray, valid, 448)
        assert (gray[~valid] == 0).all()
        if e['ruler_removal_applied']:
            assert sha(e['cleanup_mask_npy']) == e['cleanup_mask_sha256']
            mask = np.load(e['cleanup_mask_npy'], allow_pickle=False)
            original = np.load(r['gray01_npy'], allow_pickle=False)
            original_valid = np.load(r['valid_mask_npy'], allow_pickle=False)
            assert mask.shape == gray.shape and mask.dtype == bool and int(mask.sum()) == e['proposed_mask_pixels']
            yy, xx = np.mgrid[:448, :448]
            nx, ny = ref.native_point(e, xx, yy)
            protected = (nx >= b[0]) & (nx <= b[2]) & (ny >= b[1]) & (ny <= b[3])
            assert not (mask[0] & protected).any()
            assert np.array_equal(gray[~mask], original[~mask]) and (gray[mask] == 0).all()
            assert np.array_equal(valid, original_valid & ~mask)
        else:
            assert e['gray01_sha256'] == r['gray01_sha256'] and e['valid_mask_sha256'] == r['valid_mask_sha256']
        if e['engineering_input_ready']:
            assert not e['cleanup_review_reasons'] and not e['postclean_high_confidence_rulers']
    return receipt['counts']


def bounded(tasks, worker, workers, progress):
    counts, done = Counter(), 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=crop.initialize, max_tasks_per_child=20) as pool:
        pending = iter(tasks)
        inflight = set()
        for _ in range(workers):
            t = next(pending, None)
            if t is not None:
                inflight.add(pool.submit(worker, t))
        while inflight:
            ready, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in ready:
                inflight.remove(future)
                counts.update(future.result())
                done += 1
                t = next(pending, None)
                if t is not None:
                    inflight.add(pool.submit(worker, t))
                if done % 10 == 0 or done == len(tasks):
                    progress(done, counts)
    return counts


def summary(tasks, ident):
    directory = ROOT / 'candidate'
    assert not directory.exists()
    directory.mkdir()
    entries, hashes, all_patients, seen = [], defaultdict(set), defaultdict(set), set()
    for t in tasks:
        path = ROOT / 'edge_finish_v1/cases' / (t['row']['case_id'] + '.json')
        raw_path = crop.DATA / t['row']['case_id'] / 'result.json'
        for e in json.loads(path.read_text())['entries']:
            assert e['key'] not in seen
            seen.add(e['key'])
            e = {**e, 'source_result': str(raw_path), 'source_result_sha256': sha(raw_path),
                 'final_cleanup_result': str(path), 'final_cleanup_result_sha256': sha(path)}
            entries.append(e)
            all_patients[e['split']].add(e['patient_id'])
            if e['engineering_input_ready']:
                hashes[e['gray01_sha256']].add(e['split'])
    assert len(entries) == 38520 and not all_patients['train'] & all_patients['val']
    duplicated = {h for h, splits in hashes.items() if len(splits) > 1}
    counts, patients, labels = defaultdict(Counter), defaultdict(set), defaultdict(Counter)
    handles = {(s, yes): (directory / (s + ('_usable.jsonl' if yes else '_held.jsonl'))).open('w')
               for s in ['train', 'val'] for yes in [True, False]}
    try:
        for e in entries:
            if e['gray01_sha256'] in duplicated:
                e['engineering_input_ready'] = False
                e['cleanup_status'] = 'held_cross_split_duplicate'
                e['cleanup_review_reasons'] += ['cross_split_identical_output']
            s, ready = e['split'], e['engineering_input_ready']
            counts[s]['total'] += 1
            counts[s]['usable' if ready else 'held'] += 1
            counts[s][e['cleanup_status']] += 1
            if ready:
                patients[s].add(e['patient_id'])
                labels[s][str(e['kl'])] += 1
            e['eligibility'] = 'usable_technical_qa' if ready else 'held_excluded_from_training'
            handles[(s, ready)].write(json.dumps(e, ensure_ascii=False) + '\n')
    finally:
        for f in handles.values():
            f.close()
    assert {s: counts[s]['total'] for s in counts} == crop.EXPECTED
    report = {
        'phase': 'automated_processing_completed_pending_visual_review', 'identity': ident['identity'],
        'counts': dict(counts), 'usable_patient_counts': {s: len(v) for s, v in patients.items()},
        'usable_kl_counts': dict(labels), 'cases': len(tasks), 'output_shape': [1, 448, 448], 'detector_expansion': 1.6,
        'native_dicom_recrop': True, 'parent_1p3_exact_reproduction_all': True,
        'all_images_automatically_audited': True, 'all_modified_pixels_verified': True,
        'patient_split_disjoint': True, 'cross_split_duplicate_hashes': len(duplicated),
        'clinical_quality_approved': False, 'full_manual_review_completed': False,
        'training_started': False, 'test_accessed': False, 'raw_and_masks_retained': True,
        'side_policy': 'image_left=R,image_right=L; no flip',
        'candidate_manifest_sha256': {f.name: sha(f) for f in sorted(directory.glob('*.jsonl'))},
        'identity_file_sha256': sha(OUT / 'identity.json'),
    }
    save(OUT / 'automatic_completion.json', report)
    return report


def run(mode, resume):
    ident = identity()
    assert json.loads((OUT / 'identity.json').read_text()) == ident
    inputs = json.loads((OUT / 'input_snapshot.json').read_text())
    assert sha(OUT / 'tasks_private.json') == inputs['tasks_sha256']
    tasks = json.loads((OUT / 'tasks_private.json').read_text())
    if mode == 'pilot':
        chosen = set(json.loads((OUT / 'pilot_cases.json').read_text()))
        tasks = [t for t in tasks if t['row']['case_id'] in chosen]
    else:
        pilot = json.loads((OUT / 'pilot_completion.json').read_text())
        visual = json.loads((OUT / 'pilot_visual_review.json').read_text())
        assert pilot['passed'] and pilot['identity'] == ident['identity'] and visual['approved_for_full_processing']
        assert visual['pilot_completion_sha256'] == sha(OUT / 'pilot_completion.json')
    assert not (OUT / ('pilot_completion.json' if mode == 'pilot' else 'automatic_completion.json')).exists()
    existing, pending = [], []
    for t in tasks:
        f = ROOT / 'processed_cases' / (t['row']['case_id'] + '.json')
        if f.exists():
            assert resume, 'existing_checkpoints_require_explicit_resume'
            existing.append(t)
        else:
            pending.append(t)
    start = time.time()
    status_path = OUT / ('pilot_status.json' if mode == 'pilot' else 'status.json')

    def status(phase, done, counts):
        save(status_path, {'phase': phase, 'mode': mode, 'cases': done, 'target_cases': len(tasks),
                           'counts': dict(counts), 'elapsed_seconds': time.time() - start, 'updated_unix': time.time()})

    status('processing', len(existing), {})
    bounded(pending, process, 2 if mode == 'pilot' else 6,
            lambda n, c: status('processing', len(existing) + n, c))
    counts = bounded(tasks, verify_case, 2 if mode == 'pilot' else 4,
                     lambda n, c: status('independent_verification', n, c))
    if mode == 'pilot':
        report = {'passed': True, 'identity': ident['identity'], 'cases': len(tasks),
                  'counts': dict(counts), 'all_arrays_and_geometry_verified': True,
                  'elapsed_seconds': time.time() - start}
        save(OUT / 'pilot_completion.json', report)
    else:
        report = summary(tasks, ident)
    status('pilot_verified_pending_visual_review' if mode == 'pilot' else report['phase'], len(tasks), counts)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'pilot', 'full'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.mode == 'prepare':
        prepare()
    else:
        run(args.mode, args.resume)


if __name__ == '__main__':
    main()
