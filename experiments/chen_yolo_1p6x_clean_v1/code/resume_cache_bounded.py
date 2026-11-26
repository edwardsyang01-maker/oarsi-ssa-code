import gc
import json
import os
from pathlib import Path
import sys
import time

import pipeline16 as p

ORIGINAL_BOUNDED = p.bounded


def release_case_cache(task):
    assert hasattr(os, 'posix_fadvise')
    case = task['row']['case_id']
    files = {p.crop.e.p.NAS / task['row']['input_archive_relpath']}
    for e in task['items']:
        files.update(Path(e[k]) for k in ['gray01_npy', 'valid_mask_npy'])
    for folder in [p.crop.DATA / case, p.clean.DATA / 'clean_v1' / case,
                   p.clean.DATA / 'edge_finish_v1' / case, p.ROOT / 'proposals_v1/cases' / case]:
        files.update(folder.glob('*.npy'))
    for file in files:
        if not file.is_file():
            continue
        fd = os.open(file, os.O_RDONLY)
        try:
            if file.is_relative_to(p.clean.DATA) or file.is_relative_to(p.ROOT):
                os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    gc.collect()


def process(task):
    result = p.process(task)
    release_case_cache(task)
    return result


def verify(task):
    result = p.verify_case(task)
    release_case_cache(task)
    return result


def bounded(tasks, worker, workers, progress):
    fn, count = (process, 12) if worker is p.process else (verify, 4)
    return ORIGINAL_BOUNDED(tasks, fn, count, progress)


def main():
    assert sys.argv[1:] == ['full', '--resume']
    config = {'runner_sha256': p.sha(__file__), 'algorithm_identity_sha256': p.sha(p.OUT / 'identity.json'),
              'processing_workers': 12, 'verification_workers': 4,
              'reason': 'systemd-oomd: parent memory pressure 93.10% > 50% for >20s; initial scope peak 20951412736',
              'cache_policy': 'fsync new case arrays then per-file POSIX_FADV_DONTNEED; no data deletion or global cache change'}
    path = p.OUT / 'resume_cache_runtime_identity.json'
    if path.exists():
        assert json.loads(path.read_text()) == config
    else:
        p.save(path, config)
    assert p.identity() == json.loads((p.OUT / 'identity.json').read_text())
    snapshot = json.loads((p.OUT / 'input_snapshot.json').read_text())
    assert p.sha(p.OUT / 'tasks_private.json') == snapshot['tasks_sha256']
    tasks = json.loads((p.OUT / 'tasks_private.json').read_text())
    done = [t for t in tasks if (p.ROOT / 'processed_cases' / (t['row']['case_id'] + '.json')).exists()]
    started = time.time()
    def progress(n, counts):
        p.save(p.OUT / 'recovery_status.json', {'phase': 'rechecking_committed_cases', 'cases': n,
               'target_cases': len(done), 'counts': dict(counts), 'updated_unix': time.time()})
    progress(0, {})
    counts = ORIGINAL_BOUNDED(done, verify, 4, progress)
    p.save(p.OUT / 'recovery_verification.json', {'phase': 'passed', 'cases': len(done), 'counts': dict(counts),
           'all_committed_arrays_verified': True, 'algorithm_identity_unchanged': True,
           'elapsed_seconds': time.time() - started, 'runtime_identity_sha256': p.sha(path)})
    del tasks, done
    gc.collect()
    p.bounded = bounded
    p.main()


if __name__ == '__main__':
    main()
