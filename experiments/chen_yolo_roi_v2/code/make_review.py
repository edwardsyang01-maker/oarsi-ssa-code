import html
import json
import pipeline as p


def main():
    selection = json.loads((p.OUT / 'selection_private.json').read_text())
    records = []
    for row in selection:
        path = p.DATA / row['case_id'] / 'result.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        rois = [{k: x.get(k) for k in ['image_position', 'score', 'status', 'flags', 'gray_metrics', 'technical_qc']}
                for x in r['yolo']]
        records.append({'case_id': row['case_id'], 'sample_group': row['sample_group'],
                        'modality': row['modality'], 'visit': row['visit'], 'status': r['status'],
                        'missing': r.get('missing_positions', ['image_left', 'image_right']),
                        'film_flags': r.get('film_flags', []), 'rois': rois,
                        'flagged': r['status'] != 'completed' or len(r['yolo']) != 2 or
                        any(x['flags'] or x['status'] != 'saved' for x in r['yolo'])})
    payload = json.dumps(records, ensure_ascii=False).replace('<', '\\u003c')
    template = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO stage 2 crop review</title>
<style>
body{font:16px/1.65 system-ui,sans-serif;margin:0;background:#101820;color:#edf3fa}
main{max-width:1450px;margin:auto;padding:24px}h1{font-size:26px;margin:0}p{margin:10px 0}
.notice{background:#293541;padding:14px;border-left:4px solid #ffbf64;border-radius:5px}
button,select{font:inherit;padding:7px 14px;margin:5px 8px 5px 0;border-radius:6px;border:1px solid #7e93a5}
button{cursor:pointer}img{width:100%;height:auto;border:1px solid #455767;margin:10px 0}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #455767;vertical-align:top}
.pill{color:#ffce88}a{color:#73dbe9}code{overflow-wrap:anywhere}small{color:#b8c5d1}
</style><main>
<h1>YOLO stage 2 crop review</h1>
<p>304 training patients: the original 48 regression cases + 256 new random patients. No candidate has formal manual anatomical quality approval.</p>
<p class="notice">Orange boxes: old geometric method; cyan boxes: YOLO. The film preview masks the periphery and may hide part of a box; actual crops are not masked.
Automatic flags only schedule review; no flag does not mean acceptable. A physician or experienced reader should check the joint space, medial/lateral margins, femoral and tibial coverage, gray detail and left/right linkage.</p>
<label>Sample <select id="group"><option value="all">All</option><option value="random256">New 256</option><option value="regression48">Original 48</option></select></label>
<label>Type <select id="modality"><option value="all">All</option><option>CR</option><option>DX</option><option>RG</option></select></label>
<label>Queue <select id="quality"><option value="all">All</option><option value="flagged">ROI flag or missed</option><option value="unflagged">No ROI flag</option></select></label>
<br><button id="prev">Previous ←</button><button id="next">Next →</button><select id="case"></select><span id="count"></span>
<h2 id="title"></h2><p id="film"></p><img id="panel" alt="Film with boxes and left/right knee crops from both methods">
<table><thead><tr><th>Image position → clinical side</th><th>YOLO output</th><th>Review reason</th><th>Gray of valid crop pixels</th></tr></thead><tbody id="details"></tbody></table>
<p>Fill in the server file <code>human_review_template.csv</code> and save it as <code>human_review_completed.csv</code>: record reader, date, the four anatomy/gray conclusions, side, and the accept/re-crop/exclude decision.</p>
<p>KL labels are not shown. Public training patients of the current detector overlap with evaluation patients of this project; this page supports engineering quality review only and does not show independent validation localization performance.</p>
<small>For internal review on the server only; peripheral masking is not full de-identification, and case images must not be exported without authorization.</small>
</main><script>
const all=__DATA__;let cases=all,index=0;
const byId=x=>document.getElementById(x);
const names={high_saturation_fraction:'large bright saturation fraction',low_saturation_fraction:'large dark saturation fraction',low_valid_fraction:'low valid pixel fraction',low_gray_dynamic_range:'small gray dynamic range',expanded_box_clipped:'expanded box clipped by the image border',crosses_image_midline:'box crosses the image midline',detector_center_padding:'detector input needs padding',detector_frame_edge:'predicted box touches the detector input edge',multiple_detections_same_side:'several boxes on one side',bilateral_boxes_overlap:'left and right boxes overlap'};
const desc=flags=>(flags||[]).map(x=>names[x]||x).join('; ')||'no automatic flag';
function render(){
 byId('case').replaceChildren(...cases.map((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.case_id;return o}));
 if(!cases.length){byId('title').textContent='No matching cases';byId('panel').hidden=true;byId('details').replaceChildren();byId('count').textContent='0 / 0';byId('film').textContent='';return;}
 let r=cases[index];byId('case').value=index;byId('count').textContent=`${index+1} / ${cases.length}`;
 byId('title').textContent=`${r.case_id} · ${r.modality} · ${r.visit} · ${r.sample_group==='random256'?'new random sample':'regression sample'}`;
 byId('film').textContent='Whole-film gray hints: '+desc(r.film_flags)+'. ROI hints are in the table below.';
 byId('panel').hidden=r.status!=='completed';byId('panel').src=r.case_id+'.png';
 byId('details').replaceChildren(...['image_left','image_right'].map(pos=>{let x=r.rois.find(x=>x.image_position===pos),tr=document.createElement('tr');
 let m=x?.gray_metrics;
 let values=[pos==='image_left'?'image left → right knee R':'image right → left knee L',x?`${x.status}; score=${x.score.toFixed(3)}`:'missed',x?desc(x.flags):'no box; check the old-method fallback candidate',m?`high-end saturation ${(100*m.high_saturation_fraction).toFixed(1)}%; P95−P05=${m.p95_p05.toFixed(3)}`:'—'];
 values.forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.append(td)});return tr}));
}
function filter(){cases=all.filter(r=>(byId('group').value==='all'||r.sample_group===byId('group').value)&&(byId('modality').value==='all'||r.modality===byId('modality').value)&&(byId('quality').value==='all'||r.flagged===(byId('quality').value==='flagged')));index=0;render()}
['group','modality','quality'].forEach(k=>byId(k).onchange=filter);byId('case').onchange=e=>{index=Number(e.target.value);render()};
function step(n){if(cases.length){index=(index+n+cases.length)%cases.length;render()}}
byId('prev').onclick=()=>step(-1);byId('next').onclick=()=>step(1);
document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1)});render();
</script></html>'''
    path = p.OUT / 'review_server_only/index.html'
    path.write_text(template.replace('__DATA__', payload))
    p.save(p.OUT / 'review_gallery_receipt.json', {
        'records': len(records), 'png_files': len(list(path.parent.glob('*.png'))),
        'gallery_sha256': p.sha(path), 'includes_kl_labels': False,
        'human_review_completed': False, 'exported_medical_images_to_mac': False,
    })
    print(str(path), len(records), flush=True)


if __name__ == '__main__':
    main()
