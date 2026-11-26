import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import pipeline as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[2,4,8,12,16], default=16)
    args = parser.parse_args()
    run_id = p.verify_identity()
    preflight = json.loads((p.OUT/'preflight.json').read_text())
    if not preflight.get('passed') or preflight.get('run_id') != run_id:
        raise RuntimeError('Matching smoke and resume evidence is required')
    if (p.OUT/'completion.json').exists():
        raise RuntimeError('Completed full crop job must not be restarted')
    start = time.time()
    status = {'run_id':run_id,'pid':os.getpid(),'started_unix':start,
              'workers':args.workers,'device':'cpu','detector_retraining':False,
              'new_classification_training':False}
    try:
        p.save(p.OUT/'job_status.json',{**status,'phase':'cropping'})
        p.run(resume=True,limit=0,workers=args.workers)
        p.save(p.OUT/'job_status.json',{**status,'phase':'verifying'})
        subprocess.run([sys.executable,str(p.ROOT/'code/verify_outputs.py')],check=True)
        p.save(p.OUT/'job_status.json',{**status,'phase':'building_review'})
        subprocess.run([sys.executable,str(p.ROOT/'code/make_review.py')],check=True)
        summary = json.loads((p.OUT/'summary.json').read_text())
        verification = json.loads((p.OUT/'verification.json').read_text())
        gallery = json.loads((p.OUT/'review_gallery_receipt.json').read_text())
        if not summary['all_candidates_complete'] or not verification['passed']:
            raise RuntimeError('Final completeness or integrity gate failed')
        if gallery['records'] != summary['processed_films']:
            raise RuntimeError('Review inventory is incomplete')
        result = {**status,'phase':'completed','elapsed_seconds':time.time()-start,
                  'processed_films':summary['processed_films'],
                  'summary_sha256':p.sha(p.OUT/'summary.json'),
                  'verification_sha256':p.sha(p.OUT/'verification.json'),
                  'review_gallery_sha256':gallery['gallery_sha256'],
                  'clinical_quality_approved':False,'training_ready':False,
                  'test_images_or_labels_opened':False}
        p.save(p.OUT/'completion.json',result)
        p.save(p.OUT/'job_status.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as exc:
        p.save(p.OUT/'job_status.json',{**status,'phase':'failed','elapsed_seconds':time.time()-start,
                                     'error':str(exc),'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
