import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import candidates as c
from export_review import saved
from run_supplement_v2 import reasons
from verify_candidates import roi_check
p = c.p


def read_review(ledger):
    decisions = {}; seen = set(); evidence = {}
    for page in ledger['reviewed_pages']:
        directory=c.ROOT/'preview_server_only/review_batches'/page['batch']
        index=json.loads((directory/'public_index.json').read_text())
        entry=next(x for x in index['pages'] if x['page']==page['page'])
        assert entry['sha256']==page['page_sha256']==p.sha(directory/page['page'])
        items={x['key']:x for x in index['records']}
        assert set(page['accept_proposals_except'])<=set(entry['keys'])
        for key in entry['keys']:
            assert key not in seen;seen.add(key)
            evidence[key]=page['batch']+'/'+page['page']
            if key not in page['accept_proposals_except']:
                r=items[key];assert r['variant'] is not None
                decisions[key]={'decision':'accept','candidate_family':'full',**r,
                                'evidence':evidence[key]}
    reviewed_evidence = set(evidence.values())
    for group, nested in [('comparison_reviewed_pages', False), ('extra_reviewed_pages', True)]:
        for page in ledger.get(group, []):
            directory=c.ROOT/'preview_server_only'
            if nested: directory=directory/'review_batches'
            directory=directory/page['batch']
            index=json.loads((directory/'public_index.json').read_text())
            entry=next(x for x in index['pages'] if x['page']==page['page'])
            assert p.sha(directory/page['page'])==entry['sha256']==page.get('page_sha256',page.get('sha256'))
            reviewed_evidence.add(page['batch']+'/'+page['page'])
    for batch,pages in ledger.get('context_previews_viewed',{}).items():
        for page in pages: reviewed_evidence.add(batch+'/'+page)
    if ledger.get('edge_preview_viewed'): reviewed_evidence.add(ledger['edge_preview_viewed'])
    for key, item in ledger['overrides'].items():
        assert item['decision'] in ['accept','exclude']
        assert item['evidence'] in reviewed_evidence,(key,'unreviewed override evidence')
        decisions[key]={**item,'candidate_family':item.get('candidate_family','full')}
    for key, item in ledger.get('exclusions',{}).items():
        assert item['decision']=='exclude' and item['reason'] and item['evidence']
        assert key not in decisions or decisions[key]['decision']=='exclude'
        decisions[key]=item
    assert not ledger['unresolved_known'] and ledger['complete']
    for page in ledger['audit_reviewed_pages']:
        directory=c.ROOT/'preview_server_only'/page['batch']
        assert p.sha(directory/page['page'])==page['page_sha256']
        for key in page['accept_original_except']:assert key in decisions
    audit=json.loads((c.ROOT/'preview_server_only/audit01/public_index.json').read_text())
    assert {x['page'] for x in ledger['audit_reviewed_pages']}=={x['page'] for x in audit['pages']}
    return decisions,seen,audit


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ledger',required=True);ap.add_argument('--name',required=True)
    a=ap.parse_args();assert '/' not in a.name
    ledger_path=Path(a.ledger);ledger=json.loads(ledger_path.read_text())
    assert json.loads((p.OUT/'completion.json').read_text())['phase']=='completed'
    assert json.loads((p.OUT/'verification.json').read_text())['passed']
    technical_path=c.OUT/'final_candidate_technical_verification.json'
    technical=json.loads(technical_path.read_text());assert technical['phase']=='verified'
    decisions,seen,audit=read_review(ledger)
    rows=json.loads((p.OUT/'selection_private.json').read_text())
    assert len(rows)==19264 and Counter(x['split'] for x in rows)=={'train':17133,'val':2131}
    patient_sets={s:{r['patient_id'] for r in rows if r['split']==s} for s in ['train','val']}
    assert not patient_sets['train']&patient_sets['val']
    assert len({r['archive_sha256'] for r in rows})==len(rows)
    assert len({r['native_pixel_sha256'] for r in rows})==len(rows)
    output={'train':[],'val':[]};rejected=[];counts={s:Counter() for s in output}
    methods={s:Counter() for s in output};roi_hashes=defaultdict(set);all_keys=set();consumed=set()
    result_hashes={};full_keys=set();patient_used={s:set() for s in output}
    for row in rows:
        case=row['case_id'];split=row['split'];base_path=p.DATA/case/'result.json'
        base=json.loads(base_path.read_text());assert base['status']=='completed'
        for field,value in row.items():assert base[field]==value
        repair_path=c.DATA/'full'/case/'result.json';has_repair=repair_path.exists()
        if not has_repair:
            assert not reasons(base)
            assert len([x for x in base['yolo'] if x['status']=='saved'])==2
        for pos, side in p.SIDES.items():
            key=case+'_'+side;assert key not in all_keys;all_keys.add(key)
            original=next((x for x in base['yolo'] if x['image_position']==pos and x['status']=='saved'),None)
            counts[split]['expected']+=1;counts[split]['original_saved']+=original is not None
            if has_repair:
                full_keys.add(key);assert key in seen and key in decisions,(key,'unreviewed repair')
                decision=decisions[key];consumed.add(key)
                if decision['decision']=='exclude':
                    rejected.append({'key':key,**row,'image_position':pos,'clinical_side':side,
                                     'reason':decision['reason'],'evidence':decision['evidence']})
                    counts[split]['excluded']+=1;continue
                family=decision.get('candidate_family','full');variant=decision['variant']
                assert family in ['full','extra']
                source=c.DATA/family/case/'result.json';result=json.loads(source.read_text())
                roi=saved(result,variant,pos);assert roi is not None
                if decision.get('result_sha256'):assert p.sha(source)==decision['result_sha256']
                for f in ['gray01_sha256','valid_mask_sha256']:
                    if decision.get(f):assert roi[f]==decision[f]
                review={'kind':'individual_ai_engineering_review','evidence':decision['evidence'],
                        'reason':decision.get('reason','accepted reviewed proposal')}
            else:
                source=base_path;result=base;family='base';variant='original';roi=original
                assert not(set(roi['flags'])-{'detector_center_padding'})
                review={'kind':'technical_screen_with_stratified_ai_audit','evidence':'audit01',
                        'reason':'routine or padding-only; no other screening trigger'}
            assert roi_check(roi,result,row)
            changed=original is None or any(roi[f]!=original[f] for f in ['bbox','gray01_sha256','valid_mask_sha256'])
            counts[split]['usable']+=1
            counts[split]['rescued_missing']+=original is None
            counts[split]['repaired_existing']+=original is not None and changed
            counts[split]['retained_original']+=not changed
            methods[split][family+'/'+variant]+=1
            patient_used[split].add(row['patient_id']);roi_hashes[roi['gray01_sha256']].add(split)
            result_hashes[str(source)]=p.sha(source)
            output[split].append({'key':key,'film_id':row['film_id'],'patient_id':row['patient_id'],
                'split':split,'visit':row['visit'],'modality':row['modality'],'case_id':case,
                'archive_sha256':row['archive_sha256'],'native_pixel_sha256':row['native_pixel_sha256'],
                'image_position':pos,'clinical_side':side,'kl':row['labels'][side],
                'side_policy':'image_left=R,image_right=L; no spatial flip or metadata side inference',
                'source_result':str(source),'source_result_sha256':result_hashes[str(source)],
                'candidate_family':family,'variant':variant,'bbox':roi['bbox'],
                'gray01_npy':roi['gray01_npy'],'gray01_sha256':roi['gray01_sha256'],
                'valid_mask_npy':roi['valid_mask_npy'],'valid_mask_sha256':roi['valid_mask_sha256'],
                'shape':[1,224,224],'original_flags':original['flags'] if original else ['missing'],
                'selected_flags':roi['flags'],'review':review,'engineering_input_ready':True,
                'clinical_quality_approved':False,'individual_anatomical_side_verified':False})
    assert full_keys==seen and consumed==set(decisions)
    assert not any(len(splits)>1 for splits in roi_hashes.values())
    for split in output:
        assert counts[split]['usable']+counts[split]['excluded']==counts[split]['expected']
    directory=p.PROJECT/'data/manifests'/a.name
    directory.mkdir(parents=True,exist_ok=False)
    files={}
    for split,items in output.items():
        destination=directory/(split+'_usable.jsonl')
        destination.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items))
        assert sum(1 for _ in destination.open())==counts[split]['usable'];files[destination.name]=p.sha(destination)
    p.save(directory/'excluded_private.json',rejected);files['excluded_private.json']=p.sha(directory/'excluded_private.json')
    report={'phase':'completed','dataset':a.name,'count_unit':'single_knee_224_float32_ROI',
        'counts':{s:dict(v) for s,v in counts.items()},'methods':{s:dict(v) for s,v in methods.items()},
        'patients':{s:len(v) for s,v in patient_used.items()},'pending':0,
        'reviewed_abnormal_cases':len(full_keys)//2,'reviewed_abnormal_knees':len(full_keys),
        'audit_cases':len({x['case_id'] for x in audit['records']}),'audit_knees':len(audit['records']),
        'patient_split_disjoint':True,'duplicate_source_archive_or_pixels':0,'cross_split_selected_roi_duplicates':0,
        'all_selected_files_verified':True,'labels_match_position_side':True,
        'engineering_input_ready':True,'clinical_quality_approved':False,'individual_anatomical_side_verified':False,
        'detector_retraining':False,'new_training_started':False,'test_accessed':False,
        'base_run_id':base['run_id'],'repair_identity':c.identity()['identity'],
        'ledger_sha256':p.sha(ledger_path),'technical_verification_sha256':p.sha(technical_path),
        'selected_result_hashes_digest':p.digest(result_hashes),'manifest_sha256':files,
        'code_sha256':p.sha(Path(__file__)),'server_manifest_directory':str(directory),
        'limitations':['AI engineering review, not clinician validation','author pretrained detector overlap remains documented','position convention does not individually verify anatomical laterality']}
    p.save(directory/'summary.json',report);p.save(c.OUT/'final_usable_summary.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
