import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import time

import crop_full as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'pilot', 'full'])
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    c.OUT.mkdir(parents=True, exist_ok=True)
    lock = (c.OUT / 'crop.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = c.identity()
    frozen = c.OUT / 'identity.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == identity
    else:
        assert a.mode == 'prepare'
        c.save(frozen, identity)
    all_tasks = c.tasks()
    key = identity['identity']
    for task in all_tasks:
        task['identity'] = key
    if a.mode == 'prepare':
        selected = c.pilot_cases(all_tasks)
        c.save(c.OUT / 'pilot_cases.json', selected)
        c.save(c.OUT / 'authorization.json', {'user_request': 'Use 2x: crop first, then review all images, and remove any ruler that appears',
               'scope': 'all train/val native 2.0 crops, full review and traceable ruler masks; no training, no test access',
               'parent_preview_authorization_sha256': c.e.p.sha(c.e.p.OUT / 'preview_transfer_authorization_20261111.json'),
               'preview_scope': 'joint-anomaly review of the same batch and necessary full-review previews, joint region and minimal local context only; identity/KL not exported'})
        print(json.dumps({'prepared': True, 'cases': len(all_tasks), 'knees': sum(len(t['items']) for t in all_tasks), 'pilot_cases': len(selected), 'identity': key}), flush=True)
        return
    assert not (c.OUT / 'raw_completion.json').exists(), 'the original 2.0 crops are complete and cannot be restarted'
    if a.mode == 'pilot':
        selected = set(json.loads((c.OUT / 'pilot_cases.json').read_text()))
        tasks = [t for t in all_tasks if t['row']['case_id'] in selected]
    else:
        pilot = json.loads((c.OUT / 'pilot_completion.json').read_text())
        assert pilot['passed'] and pilot['identity'] == key
        tasks = all_tasks
    workers = 2 if a.mode == 'pilot' else 6
    done, pending, failed = {}, [], []
    started = time.time()
    for task in tasks:
        path = c.DATA / task['row']['case_id'] / 'result.json'
        if path.exists():
            assert a.resume, 'a checkpoint exists; --resume is required'
            result = c.verify_result(task, key)
            done[result['case_id']] = result['status']
        else:
            pending.append(task)

    def status(phase):
        value = {'phase': phase, 'mode': a.mode, 'completed_cases': len(done), 'target_cases': len(tasks),
                 'failed_cases': len(failed), 'workers': workers, 'elapsed_seconds': time.time() - started, 'updated_unix': time.time()}
        c.save(c.OUT / ('pilot_status.json' if a.mode == 'pilot' else 'status.json'), value)
        return value

    status('running')
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=c.initialize, max_tasks_per_child=10) as pool:
        iterator = iter(pending)
        inflight = {}
        for _ in range(workers):
            task = next(iterator, None)
            if task is not None:
                inflight[pool.submit(c.process, task)] = task['row']['case_id']
        while inflight:
            ready, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in ready:
                case = inflight.pop(future)
                result = future.result()
                if result['status'] == 'completed':
                    done[case] = 'completed'
                else:
                    failed.append(case)
                task = next(iterator, None)
                if task is not None:
                    inflight[pool.submit(c.process, task)] = task['row']['case_id']
                if (len(done) + len(failed)) % 10 == 0:
                    status('running')
                if (len(done) + len(failed)) % 200 == 0:
                    print(json.dumps(status('running')), flush=True)
    status('generated')
    assert not failed, 'failed cases exist; cannot enter review cleaning or freeze the final data'
    if a.mode == 'pilot':
        for task in tasks:
            c.verify_result(task, key)
        reference = json.loads((c.pilot.OUT / 'selection_private.json').read_text())
        for r in reference:
            parent = r['parent448']
            result = json.loads((c.DATA / parent['case_id'] / 'result.json').read_text())
            item = next(x for x in result['entries'] if x['key'] == parent['key'])
            for field in ['gray01', 'valid_mask']:
                assert item[field + '_sha256'] == r[field + '_sha256']
            assert item['bbox'] == r['bbox'] and item['transform'] == r['transform']
        report = {'passed': True, 'identity': key, 'cases': len(tasks), 'knees': sum(len(t['items']) for t in tasks),
                  'existing12_direct2x_byte_exact': True, 'elapsed_seconds': time.time() - started,
                  'methods': dict(Counter(x['candidate_family']+'/'+x['variant'] for t in tasks for x in t['items']))}
        c.save(c.OUT / 'pilot_completion.json', report)
        print(json.dumps(report), flush=True)
    else:
        status('verifying_raw2x')
        report = c.finalize(tasks, key)
        print(json.dumps({'phase': report['phase'], 'counts': report['counts']}), flush=True)
    status('completed')


if __name__ == '__main__':
    main()
