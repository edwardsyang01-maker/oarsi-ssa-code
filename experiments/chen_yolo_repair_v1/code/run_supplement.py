import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing as mp
from pathlib import Path
import time
import candidates as c
from run_stream_v2 import eligible, verify_existing
p=c.p


def reasons(result):
    xs=[x for x in result.get('yolo',[]) if x['status']=='saved']
    if len(xs)!=2:return []
    a,b=xs; aw=a['bbox'][2]-a['bbox'][0];ah=a['bbox'][3]-a['bbox'][1]
    bw=b['bbox'][2]-b['bbox'][0];bh=b['bbox'][3]-b['bbox'][1]
    why=[]
    if abs(sum(a['bbox'][1::2])/2-sum(b['bbox'][1::2])/2)/((ah+bh)/2)>.30:why.append('vertical_asymmetry')
    if max(aw,bw)/min(aw,bw)>1.7 or max(ah,bh)/min(ah,bh)>1.7:why.append('size_asymmetry')
    if max(ah/aw,bh/bw)>1.65 or max(aw/ah,bw/bh)>1.8:why.append('extreme_aspect')
    return why


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    assert a.name.replace('_','').isalnum()
    identity=c.identity()['identity'];rows=json.loads((p.OUT/'selection_private.json').read_text())
    scans=[];tasks=[];seen=0
    for row in rows:
        path=p.DATA/row['case_id']/'result.json'
        if not path.exists():continue
        seen+=1;r=json.loads(path.read_text());why=reasons(r)
        if not why:continue
        scans.append({'case_id':row['case_id'],'reasons':why,'original_repair_eligible':eligible(r)})
        if eligible(r):continue
        task={'row':row,'phase':'full','identity':identity,'baseline_path':str(path)}
        if verify_existing(task,identity):
            if not a.resume:raise RuntimeError('Existing supplement requires --resume')
        else:tasks.append(task)
    header={'repair_identity':identity,'name':a.name,'code_sha256':p.sha(Path(__file__)),
            'base_seen':seen,'base_target':len(rows),'screened_cases':scans,'new_tasks':len(tasks),
            'uses_kl':False,'training':False,'test_accessed':False,'started_unix':time.time()}
    dest=c.OUT/(a.name+'_screen.json')
    if dest.exists():raise RuntimeError('Use a new screening name; preserve prior evidence')
    p.save(dest,header);pending=list(tasks);futures={};done=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=c.initialize,max_tasks_per_child=24) as pool:
        while pending or futures:
            while pending and len(futures)<2:
                task=pending.pop(0);futures[pool.submit(c.process_case,task)]=task['row']['case_id']
            ready,_=wait(futures,return_when=FIRST_COMPLETED)
            for fut in ready:
                case=futures.pop(fut);r=fut.result();done.append(r)
                print(json.dumps({'case_id':case,'status':r['status']}),flush=True)
    errors=[x for x in done if x['status']=='case_error']
    p.save(c.OUT/(a.name+'_completion.json'),{**header,'phase':'completed','completed':len(done),'case_errors':errors})


if __name__=='__main__':main()
