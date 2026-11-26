from collections import Counter,defaultdict
import json
from pathlib import Path
import time

import crop_full as c

REPAIR=c.PROJECT/'data/processed/chen_yolo_2x_clean_v1/raw_repaired_lower'


def main():
    assert not (c.OUT/'raw_completion.json').exists()
    repair=json.loads((c.ROOT/'repair_lower_v1/completion.json').read_text());assert repair['phase']=='completed'
    selected=json.loads((c.ROOT/'repair_lower_v1/selection_private.json').read_text())
    repair_cases={t['row']['case_id'] for t in selected};original_data=c.DATA
    tasks=c.tasks();base_identity=c.identity()['identity'];started=time.time()
    while True:
        status=json.loads((c.OUT/'status.json').read_text())
        if status['completed_cases']+status['failed_cases']==len(tasks) and status['phase']=='generated':break
        c.save(c.OUT/'merge_status.json',{'phase':'waiting_for_raw_generation','generated_cases':status['completed_cases']+status['failed_cases'],'target_cases':len(tasks),'updated_unix':time.time()})
        time.sleep(20)
    assert status['failed_cases']==len(repair_cases)
    c.RAW_MANIFEST.mkdir(parents=True,exist_ok=False)
    counts=Counter();flags=defaultdict(Counter);hashes=defaultdict(set);result_digests={};sources={}
    handles={s:(c.RAW_MANIFEST/(s+'_raw2x.jsonl')).open('w') for s in ['train','val']}
    try:
        for n,task in enumerate(tasks,1):
            case=task['row']['case_id'];fixed=case in repair_cases
            c.DATA=REPAIR if fixed else original_data
            r=c.verify_result(task,repair['identity'] if fixed else base_identity)
            path=c.DATA/case/'result.json';digest=c.e.p.sha(path);result_digests[case]=digest;sources[case]=str(path)
            for item in r['entries']:
                record={**item,'source_result':str(path),'source_result_sha256':digest,'crop_generator_identity':r['identity'],
                        'old_window_geometry_repair':fixed}
                handles[item['split']].write(json.dumps(record,ensure_ascii=False)+'\n');counts[item['split']]+=1
                flags[item['split']].update(item['selected_flags']);hashes[item['gray01_sha256']].add(item['split'])
            if n%200==0:c.save(c.OUT/'merge_status.json',{'phase':'verifying_raw2x','verified_cases':n,'target_cases':len(tasks),'updated_unix':time.time()})
    finally:
        for h in handles.values():h.close()
    assert counts==c.EXPECTED and not any(len(v)>1 for v in hashes.values())
    c.save(c.OUT/'raw_result_index_private.json',sources)
    report={'phase':'raw_cropping_completed_review_pending','base_identity':base_identity,'lower_repair_identity':repair['identity'],
            'cases':len(tasks),'counts':dict(counts),'output_shape':[1,448,448],'detector_expansion':2.0,
            'flags':{s:dict(v) for s,v in flags.items()},'parent448_exact_reproduction_knees':sum(counts.values()),
            'all_raw_arrays_verified':True,'patient_split_disjoint':True,'cross_split_roi_duplicates':0,
            'manifest_sha256':{s:c.e.p.sha(c.RAW_MANIFEST/(s+'_raw2x.jsonl')) for s in ['train','val']},
            'result_hashes_digest':c.e.p.digest(result_digests),'repaired_lower_window_cases':len(repair_cases),
            'full_anatomical_review_completed':False,'ruler_cleaning_completed':False,'training_started':False,'test_accessed':False,
            'finalizer_code_sha256':c.e.p.sha(__file__),'elapsed_seconds':time.time()-started}
    c.save(c.OUT/'raw_completion.json',report);c.save(c.OUT/'merge_status.json',{'phase':report['phase'],'counts':dict(counts)})
    print(json.dumps({k:report[k] for k in ['phase','cases','counts','repaired_lower_window_cases','all_raw_arrays_verified']}),flush=True)


if __name__=='__main__':main()
