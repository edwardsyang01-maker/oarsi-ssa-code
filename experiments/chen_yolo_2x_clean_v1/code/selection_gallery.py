import argparse,json,zipfile
from PIL import Image,ImageDraw,ImageFont
import audit_rulers as a
from review_gallery import safe_image


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--name',required=True);p.add_argument('--indices',required=True);p.add_argument('--paired',action='store_true');args=p.parse_args()
    root=a.ROOT/'review_queue'/args.batch;records=json.loads((root/'mapping_private.json').read_text())
    selected=sorted(set(map(int,args.indices.split(','))));out=root/args.name;out.mkdir(exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);pages=[]
    display=[]
    if args.paired:
        from run_audit_v2 import source_paths
        paths=source_paths()
        for n in selected:
            r=records[n-1];source=json.loads(paths[r['item']['case_id']].read_text())
            for item in sorted(source['entries'],key=lambda i:i['clinical_side']!='R'):
                display.append({'alias':r['alias']+' PAIR','item':item,'native_hw':source['native_hw']})
    else:display=[records[n-1] for n in selected]
    for start in range(0,len(display),16):
        canvas=Image.new('RGB',(1824,1920),'#171b22');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(display[start:start+16]):
            _,_,_,pix=safe_image(r['item'],r['native_hw']);x=(j%4)*456;y=(j//4)*480
            draw.text((x+4,y+2),r['alias']+' '+r['item']['clinical_side'],font=font,fill='white')
            canvas.paste(Image.fromarray(pix).convert('RGB'),(x+4,y+27))
        name=f'page_{start//16+1:02d}.png';canvas.save(out/name)
        pages.append({'file':name,'sha256':a.sha(out/name),'aliases':[r['alias'] for r in display[start:start+16]]})
    a.save(out/'public_index.json',{'pages':pages,'count':len(selected)})
    bundle=root/(args.name+'.zip')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):z.write(f,f.name)
    print(json.dumps({'count':len(selected),'pages':len(pages),'sha256':a.sha(bundle)}))


if __name__=='__main__':main()
