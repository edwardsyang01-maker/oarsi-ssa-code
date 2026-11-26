#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
import json
import time
from pathlib import Path

ROOT = Path('/path/to/drai')
B = ROOT / 'outputs/downstream/oarsi_patchattn_1p6'
T = B / 'test_eval_v1'
MANIFEST = ROOT / 'data/manifests/oarsi_test_eval_v1_20261117'
RUNS = [(17, 'P0', 'source_v1', 'attempt01/train_P0'),
        (23, 'P0', 'source_v3_seed23', 'attempt03_seed23/train_P0'),
        (29, 'P0', 'source_v3_seed29', 'attempt03_seed29/train_P0'),
        (17, 'P3', 'source_v2', 'attempt02_lambda_v2/train_P3'),
        (23, 'P3', 'source_v3_seed23', 'attempt03_seed23/train_P3'),
        (29, 'P3', 'source_v3_seed29', 'attempt03_seed29/train_P3')]
SECONDARY = [(17, 'P1', 'source_v2', 'attempt02_lambda_v2/train_P1'),
             (17, 'P2', 'source_v2', 'attempt02_lambda_v2/train_P2'),
             (17, 'P4', 'source_v2', 'attempt02_lambda_v2/train_P4')]
FINAL_TAG = 's010080_e21_b000'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint(run, tag):
    root = B / run / 'checkpoints'
    commit = root / f'{tag}.commit.json'
    receipt = json.loads(commit.read_text())
    directory = root / receipt['tag']
    assert receipt['tag'] == tag and sha(directory / 'manifest.json') == receipt['manifest_sha256']
    manifest = json.loads((directory / 'manifest.json').read_text())
    total = 0
    for x in manifest['files']:
        p = directory / x['path']
        assert p.stat().st_size == x['bytes'] and sha(p) == x['sha256'], f'CHECKPOINT_FILE_MISMATCH {run} {tag} {x["path"]}'
        total += x['bytes']
    return {'tag': tag, 'commit_receipt_sha256': sha(commit), 'manifest_sha256': receipt['manifest_sha256'],
            'files': len(manifest['files']), 'bytes': total, 'all_files_verified': True,
            'binding_digest': hashlib.sha256(json.dumps(receipt['binding'], sort_keys=True).encode()).hexdigest()}


def main():
    out = T / 'T3/t3_lock_material.json'
    assert not out.exists(), 'refusing to overwrite'
    started = time.time()
    protocol = T / 'T1/test_protocol_draft_v2.json'
    auth = json.load(open(T / 'T1/authorization_T1.json'))
    assert sha(protocol) == auth['approved_file_sha256']
    gate = json.load(open(T / 'prep/val_replay/gate.json'))
    assert gate['passed'] is True
    lock = json.load(open(T / 'prep/test/image_lock.json'))
    assert sha(T / 'prep/test/test_images_locked_private.jsonl') == lock['locked_rows_sha256']
    summary = json.load(open(MANIFEST / 'summary.json'))
    for name, digest in summary['manifest_sha256'].items():
        assert sha(MANIFEST / name) == digest, name
    models = []
    for role, runs in (('primary_selected', RUNS), ('sensitivity_epoch20_end', RUNS), ('secondary_selected', SECONDARY)):
        for seed, group, source, run in runs:
            completed = json.load(open(B / run / 'completed.json'))
            tag = FINAL_TAG if role == 'sensitivity_epoch20_end' else completed['best_checkpoint']
            epoch = 20 if role == 'sensitivity_epoch20_end' else completed['best_epoch']
            models.append({'role': role, 'seed': seed, 'group': group, 'source': f'{B.relative_to(ROOT)}/{source}',
                           'run': f'{B.relative_to(ROOT)}/{run}', 'validation_epoch': epoch,
                           'completed_json_sha256': sha(B / run / 'completed.json'),
                           'source_files_sha256': {p.name: sha(p) for p in sorted((B / source).glob('*.py'))},
                           **checkpoint(run, tag)})
    code = {str(p.relative_to(ROOT)): sha(p) for d in ('prep_code_v1', 'inference_v1', 'code')
            for p in sorted((T / d).glob('*.py'))}
    material = {
        'schema': 'drai_wp6_T3_lock_material_v1', 'status': 'awaiting_user_confirmation', 'approved': False,
        'protocol_sha256': auth['approved_file_sha256'], 'authorization_T1_sha256': sha(T / 'T1/authorization_T1.json'),
        'val_replay_gate_sha256': sha(T / 'prep/val_replay/gate.json'),
        'val_replay_gate': {k: gate[k] for k in ('gate1_preclean_1p6_identical', 'gate2_final_disposition_and_array', 'passed')},
        'patient_split_sha256': sha(ROOT / 'data/manifests/patient_split_v1.csv'),
        'image_lock_sha256': sha(T / 'prep/test/image_lock.json'), 'locked_rows_sha256': lock['locked_rows_sha256'],
        'test_manifest_sha256': summary['manifest_sha256'], 'test_summary_sha256': sha(MANIFEST / 'summary.json'),
        'test_counts': {'knees': summary['knees'], 'patients_usable': summary['patients_usable'],
                        'usable_kl_counts': summary['usable_kl_counts'], 'film_fate': lock['film_fate']},
        'models': models, 'model_count': len(models), 'code_sha256': code,
        'inference_plan': 'Each checkpoint is first replayed on the validation set (compared with val_epochNN.npz from training: max absolute difference of the five class probabilities <= 1e-6 and identical argmax); '
                          'once all pass, each model runs inference on the usable test list exactly once, without augmentation or flip inference.',
        'seconds': time.time() - started, 'created': time.strftime('%F %T %Z'),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(material, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({'models': len(models), 'checkpoint_bytes': sum(m['bytes'] for m in models),
                      'all_verified': all(m['all_files_verified'] for m in models), 'sha256': sha(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
