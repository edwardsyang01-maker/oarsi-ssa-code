#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('/path/to/drai')
REL = 'outputs/downstream/oarsi_patchattn_1p6'
B = ROOT/REL
X = B/'wp8_x2x3'
CODE = ('control.py', 'core.py', 'runner.py', 'runtime.py', 'test_core.py')
DOCS = ('FROZEN_PLAN.md', 'FROZEN_PLAN_AMENDMENT_v2.md', 'FROZEN_PLAN_AMENDMENT_v3_seeds.md')
AMEND = 'FROZEN_PLAN_AMENDMENT_v5_x2_x3.md'
LOCK_REL = REL+'/attempt02_lambda_v2/lambda_lock.json'
LOCK_SHA = 'd584ba13ef726d1d3375248706f7c47fcbed207f2e911aa0923c48e9cdf8d5c0'
LAMBDA = {'P2': 0.16741287014234046, 'P3': 0.2949566469066166}
TEMPLATE_LAUNCH = B/'launch_v4_x1_seed23'
RUN_QUEUE_SHA = 'b8b0d832'
QUEUES = {
    'x2_lam050': {'seed': 17, 'group': 'P3', 'multiplier': 0.5},
    'x2_lam200': {'seed': 17, 'group': 'P3', 'multiplier': 2.0},
    'x3_seed23': {'seed': 23, 'group': 'P4', 'multiplier': None},
    'x3_seed29': {'seed': 29, 'group': 'P4', 'multiplier': None},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, text):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(text)


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def check_code_identity():
    dirs = [B/d for d in ('source_v3_seed23', 'source_v3_seed29', 'source_v4_x1_seed23', 'source_v4_x1_seed29')]
    for name in CODE + DOCS:
        assert len({sha(d/name) for d in dirs}) == 1, 'CODE_NOT_IDENTICAL:'+name
    assert sha(ROOT/LOCK_REL) == LOCK_SHA
    assert json.loads((ROOT/LOCK_REL).read_text())['lambda'] == LAMBDA
    assert sha(TEMPLATE_LAUNCH/'run_queue.py').startswith(RUN_QUEUE_SHA)
    return {name: sha(dirs[0]/name) for name in CODE + DOCS}


def x2_lock(tag, multiplier):
    lam = dict(LAMBDA)
    lam['P3'] = LAMBDA['P3']*multiplier
    assert lam['P3']/multiplier == LAMBDA['P3']
    value = {'lambda': lam, 'multiplier_P3': multiplier, 'seed': 17,
             'derived_from': {'path': LOCK_REL, 'sha256': LOCK_SHA, 'lambda': LAMBDA},
             'note': 'WP8 X2 lambda sensitivity: P3 lambda = %s x seed17 source_v2 locked value; P2 kept unchanged and not trained; '
                     'no recalibration (%s)' % (multiplier, AMEND)}
    path = X/f'lambda_lock_{tag}.json'
    write_new(path, json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    return path


def make_spec(tag, q, lock_path):
    template = json.loads((B/('source_v3_seed%d' % (23 if q['seed'] == 17 else q['seed']))/'spec.json').read_text())
    spec = copy.deepcopy(template)
    assert spec['run_order'] == ['P0', 'P3'] and spec['lambda_fixed']['sha256'] == LOCK_SHA
    spec['version'] = 'oarsi-patchattn-1p6-v5-'+tag.replace('_', '-')+('-seed17' if q['multiplier'] else '')
    spec['seed'] = q['seed']
    spec['output'] = REL+'/attempt05_'+tag
    spec['run_order'] = [q['group']]
    spec['frozen_plan_amendment_v5'] = AMEND
    spec['same_training_code_as'] = ['source_v3_seed23', 'source_v3_seed29', 'source_v4_x1_seed23', 'source_v4_x1_seed29']
    if q['multiplier'] is None:
        spec['single_seed_note'] = ('WP8 X3 authorized by user 2026-11-17 ("add WP8"; X3 selected; "start training X2 and X3 directly once their preflights pass"): '
                                    'P4 seed extension after seeing validation and test results; descriptive only')
        spec['seed17_reference'] = {'P0': 'attempt01/train_P0 (source_v1)', 'P3': 'attempt02_lambda_v2/train_P3 (source_v2)',
                                    'P4': 'attempt02_lambda_v2/train_P4 (source_v2)'}
    else:
        spec['single_seed_note'] = ('WP8 X2 authorized by user 2026-11-17 ("add WP8"; X2 selected; "start training X2 and X3 directly once their preflights pass"): '
                                    'P3 lambda sensitivity x%s, seed17, post hoc; descriptive only' % q['multiplier'])
        lock = json.loads(lock_path.read_text())
        spec['lambda_fixed'] = {'path': str(lock_path.relative_to(ROOT)), 'sha256': sha(lock_path), 'lambda': lock['lambda'],
                                'note': lock['note']}
        spec['x2_lambda_multiplier'] = q['multiplier']
        spec['seed17_reference'] = {'P0': 'attempt01/train_P0 (source_v1)',
                                    'P3_lambda_x1': 'attempt02_lambda_v2/train_P3 (source_v2)'}
    return spec


RUN_QUEUE_EDITS = [
    ('2026-11-17 the user said "add WP8" and selected X1: this v4 queue uses source_v4_x1_seedNN according to its directory name (launch_v4_x1_seedNN) (training code byte-identical to source_v3_seedNN) and trains P1 -> P2 for that seed;\nlambda reuses the value locked for seed17 in source_v2 (spec.lambda_fixed), not recalibrated.',
     '2026-11-17 the user said "add WP8", selected X2 and X3, then said "start training X2 and X3 directly once their preflights pass, no need to ask me again": this v5 queue uses source_v5_TAG according to its directory name (launch_v5_TAG)\n(training code byte-identical to source_v3_seedNN and source_v4_x1_seedNN) and trains one group only: X2 is P3 of seed17 (lambda from spec.lambda_fixed, i.e. 0.5x or 2x the seed17 locked value),\nX3 is P4 of seed23/29 (lambda reuses the seed17 locked value). No recalibration.'),
    ("SEED_TAG = HERE.name.rsplit('_', 1)[-1]          # e.g. 'seed23'\nassert HERE.name == 'launch_v4_x1_'+SEED_TAG and SEED_TAG in ('seed23', 'seed29')\nSOURCE = BASE/('source_v4_x1_'+SEED_TAG)\nSPEC = json.loads((SOURCE/'spec.json').read_text())\nassert 'seed%d' % SPEC['seed'] == SEED_TAG",
     "TAG = HERE.name[len('launch_v5_'):]                # e.g. 'x2_lam050'\nEXPECTED_SEED = {'x2_lam050': 17, 'x2_lam200': 17, 'x3_seed23': 23, 'x3_seed29': 29}\nassert HERE.name == 'launch_v5_'+TAG and TAG in EXPECTED_SEED\nSOURCE = BASE/('source_v5_'+TAG)\nSPEC = json.loads((SOURCE/'spec.json').read_text())\nassert SPEC['seed'] == EXPECTED_SEED[TAG] and len(SPEC['run_order']) == 1"),
    ('two groups for this seed; 80 GiB still reserved.', 'one group in this queue; 80 GiB still reserved.'),
]

MAKE_AUTH = '''"""Authorization for the v5 (WP8 X2, X3) queue: on 2026-11-17 the user said "add WP8", selected X2 and X3, then said "start training X2 and X3 directly once their preflights pass, no need to ask me again".

Usage (from the drai project root):python -B outputs/downstream/oarsi_patchattn_1p6/launch_v5_TAG/make_authorization.py --groups GROUP
Writes the authorization only when all preflights of this queue pass and the lambda lock file is unchanged; does not start training or read test.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_queue as q

ROOT, SOURCE, SPEC = q.ROOT, q.SOURCE, q.SPEC


def main():
    """Write the authorization only when all preflights pass and lambda is locked; refuse to overwrite an existing one."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', nargs='+', required=True, choices=SPEC['run_order'])
    groups = [g for g in SPEC['run_order'] if g in set(parser.parse_args().groups)]
    q.require(Path.cwd().resolve() == ROOT, 'RUN_FROM_DATA_DISK_PROJECT')
    target = HERE/'authorization.json'
    q.require(not target.exists(), 'AUTHORIZATION_ALREADY_EXISTS')
    output = ROOT/SPEC['output']
    q.require(q.read(output/'controller_status.json')['state'] == 'preflight_completed', 'PREFLIGHT_NOT_COMPLETED')
    q.require(not (SOURCE/'training_authorization.json').exists(), 'UNEXPECTED_EXISTING_GROUP_AUTHORIZATION')
    fixed = SPEC['lambda_fixed']
    q.require(q.sha(ROOT/fixed['path']) == fixed['sha256'], 'FIXED_LAMBDA_LOCK_CHANGED')
    lock = q.read(ROOT/fixed['path'])
    q.require(lock['lambda'] == fixed['lambda'], 'FIXED_LAMBDA_VALUE_MISMATCH')
    protected = {}

    def add(path):
        """Add a file to freeze (path relative to the project root)."""
        protected[str(Path(path).relative_to(ROOT))] = q.sha(path)

    for p in sorted(SOURCE.iterdir()):
        if p.is_file() and p.name != 'training_authorization.json':
            add(p)
    base = ROOT/SPEC['base_source']
    for p in sorted(base.glob('*.py'))+[base/'deepspeed_config.json', base/'plan.json']:
        add(p)
    plan = q.read(base/'plan.json')
    for name in ('model.safetensors', 'config.json', 'preprocessor_config.json'):
        add(ROOT/plan['weights_directory']/name)
    for key in ('train_manifest', 'val_manifest', 'train_oarsi', 'val_oarsi'):
        add(ROOT/SPEC[key])
    add(ROOT/fixed['path']); add(output/'controller_status.json')
    if 'derived_from' in lock:
        add(ROOT/lock['derived_from']['path'])
        q.require(q.sha(ROOT/lock['derived_from']['path']) == lock['derived_from']['sha256'], 'SOURCE_LAMBDA_LOCK_CHANGED')
    for g in SPEC['run_order']:
        for r in (0, 1):
            add(output/f'preflight_{g}'/f'resume_comparison_rank{r}.json')
            q.require(q.read(output/f'preflight_{g}'/f'resume_comparison_rank{r}.json')['exact'], 'PREFLIGHT_NOT_EXACT_'+g)
    result = {'authorized': True, 'user_instruction': 'add WP8 (X2 and X3 selected); start training X2 and X3 directly once their preflights pass, no need to ask me again',
              'user_instruction_context': '2026-11-17: in a side session the user selected WP8 X2 (P3 with lambda x0.5 and x2, seed17) and X3 (P4 with added seeds 23 and 29); '
                                          'in this session the user authorized training directly after the preflights pass. This authorization is written automatically by the waiting service after X1 once the preflights of this queue pass',
              'lambda_policy': fixed.get('note', 'fixed from seed17 source_v2 lock; not recalibrated per seed')+' (FROZEN_PLAN_AMENDMENT_v5_x2_x3.md)',
              'authorized_at_utc': q.utc(),
              'scope': 'OARSI auxiliary + patch attention pooling on 1.6x cleaned 448 images; frozen plan v1 + amendments v2, v3 and v5 (WP8 X2/X3); '
                       + q.TAG + ', seed'+str(SPEC['seed'])+' only in this queue; groups '+','.join(groups)+', 20 epochs / 10080 updates each; no test.',
              'groups': groups, 'seed': SPEC['seed'], 'epochs': 20, 'steps_per_group': SPEC['total_steps'],
              'lambda': lock['lambda'], 'bindings': {g: q.binding(g) for g in groups},
              'protected_sha256': dict(sorted(protected.items())), 'queue_script_sha256': q.sha(HERE/'run_queue.py'),
              'gates': SPEC['gates'], 'selection': SPEC['selection'], 'exploratory': True,
              'test_permitted': False, 'automatic_training_retry': False,
              'other_work': 'test already used in WP6 and not accessed'}
    q.write(target, result)
    print(json.dumps({'authorization_sha256': q.sha(target), 'protected_files': len(protected), 'lambda': lock['lambda']}))


if __name__ == '__main__':
    main()
'''


def main():
    assert Path.cwd().resolve() == ROOT
    assert X.is_dir() and (X/AMEND).is_file()
    for tag in QUEUES:
        for d in (B/('source_v5_'+tag), B/('launch_v5_'+tag), B/('attempt05_'+tag)):
            assert not d.exists(), 'ALREADY_EXISTS:'+d.name
    code = check_code_identity()
    rq = (TEMPLATE_LAUNCH/'run_queue.py').read_text()
    for old, new in RUN_QUEUE_EDITS:
        rq = replace_once(rq, old, new)
    report = {'code_sha256': code, 'amendment_sha256': sha(X/AMEND), 'queues': {}}
    for tag, q in QUEUES.items():
        lock_path = x2_lock(tag, q['multiplier']) if q['multiplier'] else None
        src = B/('source_v5_'+tag); src.mkdir()
        for name in CODE + DOCS:
            shutil.copy2(B/'source_v3_seed23'/name, src/name)
        shutil.copy2(X/AMEND, src/AMEND)
        spec = make_spec(tag, q, lock_path)
        write_new(src/'spec.json', json.dumps(spec, indent=2, ensure_ascii=False)+'\n')
        launch = B/('launch_v5_'+tag); launch.mkdir()
        write_new(launch/'run_queue.py', rq)
        write_new(launch/'make_authorization.py', MAKE_AUTH)
        report['queues'][tag] = {'seed': spec['seed'], 'group': q['group'], 'lambda_used': spec['lambda_fixed']['lambda'][spec['lambda_calibration_groups'][q['group']]],
                                 'output': spec['output'], 'spec_sha256': sha(src/'spec.json'),
                                 'lock_sha256': None if lock_path is None else sha(lock_path),
                                 'run_queue_sha256': sha(launch/'run_queue.py'), 'make_authorization_sha256': sha(launch/'make_authorization.py')}
        for name in CODE + DOCS:
            assert sha(src/name) == code[name]
    write_new(X/'prep_report.json', json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
