import json,time,cv2,argparse
import numpy as np
import audit_rulers as a
import proposal_v1 as p
from run_audit_v2 import source_paths


def main():
    cv2.setNumThreads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='v1');args=parser.parse_args()
    output=a.ROOT/('proposal_validation' if args.version=='v1' else 'proposal_validation_'+args.version)
    assert not (output/'summary.json').exists(),'keep the verification evidence of the completed version'
    paths=source_paths();cache={};start=time.time();summary={}
    for name,filename,truthfile,field in [('pilot',a.REVIEW/'p_private.json',a.ROOT/'code/pilot_visual_labels.json','ruler_visible'),
                                         ('q01',a.ROOT/'review_queue/q01/mapping_private.json',a.ROOT/'code/q01_visual_review.json','confirmed_ruler')]:
        records=json.loads(filename.read_text());truth=set(json.loads(truthfile.read_text())[field]);out=[];miss=[];risk=[];mask_count=0
        for n,r in enumerate(records,1):
            i=r['item'];case=i['case_id']
            if case not in cache:
                source=json.loads(paths[case].read_text());initial=json.loads((a.ROOT/'audit_v2/cases'/(case+'.json')).read_text())
                cache[case]=p.evaluate(source,initial)
            e=next(x for x in cache[case] if x['key']==i['key']);copy={k:v for k,v in e.items() if k!='mask'}
            out.append({'alias':r['alias'],'item':i,'native_hw':r['native_hw'],'proposal':copy})
            if n in truth and not e['metrics']['mask_pixels']:miss.append(n)
            if e['review_flags']:risk.append(n)
            mask_count+=e['metrics']['mask_pixels']>0
        a.save(output/(name+'_private.json'),out)
        summary[name]={'images':len(out),'visual_positive':len(truth),'positive_without_proposal':miss,'review_flags':risk,'proposed_mask_images':mask_count}
    summary['seconds']=round(time.time()-start,1);summary['code_hashes']={'policy':a.sha(p.__file__),'refinement':a.sha(p.ref.__file__)}
    a.save(output/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
