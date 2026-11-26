from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0,str(ROOT/'code'))
import pipeline as p


def main():
    run_id = p.verify_identity()
    full = json.loads((p.OUT/'selection_private.json').read_text())
    mapping = {r['film_id']:r for r in full}
    history = PROJECT/'outputs/downstream'
    groups_path = history/'laterality_image_review_20261106_v1/private_binding_index.json'
    single_path = history/'laterality_single55_review_20261106_v1/private_preview_index.json'
    text_path = history/'laterality_single55_review_20261106_v1/laterality_and_crop_review_of_55_cases_20261106.txt'
    sources = {str(path):p.sha(path) for path in [groups_path,single_path,text_path]}
    singles = {r['film_id']:r for r in json.loads(single_path.read_text())}
    insufficient = {'S05','S12','S14','S46','S47','S51','S52'}
    weak = {'S44'}
    known = []
    absent = Counter()
    for old in json.loads(groups_path.read_text()):
        row = mapping.get(old['film_id'])
        if row is None:
            absent[old['group']] += 1
            continue
        assert row['patient_id']==old['patient_id'] and row['split']==old['split']
        item = {'case_id':row['case_id'],'split':row['split'],'historical_group':old['group'],
                'historical_review_code':'','historical_side_evidence':'not_individually_classified_here',
                'historical_geometry_priority':'','current_side_policy':p.CONFIG['side_policy'],
                'clinical_quality_approved':False}
        if old['film_id'] in singles:
            code = singles[old['film_id']]['code']
            item['historical_review_code'] = code
            item['historical_side_evidence'] = (
                'insufficient_AI_visual_evidence' if code in insufficient else
                'weak_AI_visual_evidence' if code in weak else
                'AI_visual_marker_support_for_image_left_R_not_clinical_truth')
            if code=='S09':item['historical_geometry_priority']='old retained ROI too high; inspect joint and proximal tibia in new crop'
            if code=='S32':item['historical_geometry_priority']='old missing-side outer margin near original boundary; inspect new coverage'
        known.append(item)
    p.write_csv(p.OUT/'historical_review_links_private.csv',known,list(known[0]))
    summary = {'run_id':run_id,'source_sha256':sources,'analysis_code_sha256':p.sha(Path(__file__)),
               'matched_films':len(known),'groups':dict(Counter(r['historical_group'] for r in known)),
               'split':dict(Counter(r['split'] for r in known)),
               'prior_single55_matched':sum(bool(r['historical_review_code']) for r in known),
               'insufficient_prior_side_evidence':sum(r['historical_side_evidence']=='insufficient_AI_visual_evidence' for r in known),
               'weak_prior_side_evidence':sum(r['historical_side_evidence']=='weak_AI_visual_evidence' for r in known),
               'known_geometry_priority_films':sum(bool(r['historical_geometry_priority']) for r in known),
               'not_in_current_cohort':dict(absent),'labels_or_crop_policy_modified':False,
               'clinical_quality_approved':False,'note':'Prior AI observations are review leads, not clinical approval or automatic exclusion'}
    p.save(p.OUT/'historical_review_crosscheck_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
