import json
import sys
import time

import pipeline16 as p

ORIGINAL_BOUNDED = p.bounded


def bounded(tasks, worker, workers, progress):
    actual = 16 if worker is p.process else 6
    return ORIGINAL_BOUNDED(tasks, worker, actual, progress)


def main():
    assert sys.argv[1:] == ['full', '--resume']
    identity = {'runner_sha256': p.sha(__file__), 'algorithm_identity_sha256': p.sha(p.OUT / 'identity.json'),
                'processing_workers': 16, 'verification_workers': 6, 'submitted_unix': time.time()}
    path = p.OUT / 'full_runtime_identity.json'
    if path.exists():
        previous = json.loads(path.read_text())
        assert all(previous[k] == identity[k] for k in identity if k != 'submitted_unix')
    else:
        p.save(path, identity)
    p.bounded = bounded
    p.main()


if __name__ == '__main__':
    main()
