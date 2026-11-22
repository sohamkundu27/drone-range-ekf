"""CPU-only rendering of reviewed, fixed-seed test examples; never writes data/runs."""
from pathlib import Path
import hashlib
import json
import random
import shutil
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 20261120
NAMES = ['drone', 'bird', 'airplane', 'helicopter']
GREEN = (60, 200, 60)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
PICKS = {'dut_anti_uav': (0, 4), 'coco': (None, 4), 'openimages': (3, 5), 'detfly': (0, 0)}
REASONS = {
    'dut_anti_uav': 'Small drone above a treeline with campus context; visible in the inset, with a moderate annotated area rather than an extreme size.',
    'coco': 'One recognizable bird beside a feeder, with ordinary background detail and a moderate annotated area.',
    'openimages': 'One helicopter in side view with its rotor and skids visible; moderate annotated area within the helicopter subset.',
    'detfly': 'Drone above textured agricultural ground, illustrating the requested ground-background scenario. Selected for scene content, not a measured model failure.'}

def labels(path):
    return [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]

def boxes(rows, w, h):
    return [(int(c), ((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h)) for c,cx,cy,bw,bh in rows]

def draw_boxes(im, bb, scale=1, offset=(0,0)):
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, max(17, round(im.width/65)))
    for c,b in bb:
        x1,y1,x2,y2 = [(v-offset[i%2])*scale for i,v in enumerate(b)]
        if x2 < 0 or y2 < 0 or x1 >= im.width or y1 >= im.height:
            continue
        d.rectangle((x1,y1,x2,y2), outline=GREEN, width=3)
        text = NAMES[c]
        tb = d.textbbox((0,0),text,font=font); tw=tb[2]; th=tb[3]-tb[1]
        tx=max(0,min(x1,im.width-tw-10)); ty=max(0,min(y1-th-12,im.height-th-10))
        d.rectangle((tx,ty,tx+tw+10,ty+th+8),fill=GREEN)
        d.text((tx+5,ty+4-tb[1]),text,font=font,fill='black')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

records={}
for ds,(target,index) in PICKS.items():
    entries=[]
    for p in sorted((ROOT/'data/yolo'/ds/'images/test').glob('*.jpg')):
        lp=ROOT/'data/yolo'/ds/'labels/test'/(p.stem+'.txt')
        rows=labels(lp)
        targets=[r for r in rows if target is None or r[0]==target]
        if len(targets)==1:
            entries.append((targets[0][3]*targets[0][4],p,lp,rows))
    entries.sort(key=lambda e:(e[0],str(e[1])))
    eligible=sorted(entries[len(entries)//4:3*len(entries)//4],key=lambda e:str(e[1]))
    candidates=random.Random(SEED).sample(eligible,8)
    _,p,lp,rows=candidates[index]
    hashes={str(p):sha(p),str(lp):sha(lp)}
    original=Image.open(p).convert('RGB'); w,h=original.size
    bb=boxes(rows,w,h); scale=min(1,1600/w)
    out=original.resize((round(w*scale),round(h*scale)),Image.Resampling.LANCZOS)
    draw_boxes(out,bb,scale)
    if ds=='dut_anti_uav':
        _,(x1,y1,x2,y2)=bb[0]
        cx,cy=(x1+x2)/2,(y1+y2)/2
        half=max(3*(x2-x1),3*(y2-y1),60)
        cropbox=(int(max(0,cx-half)),int(max(0,cy-half)),int(min(w,cx+half)),int(min(h,cy+half)))
        crop=original.crop(cropbox); zoom=400/max(crop.size)
        crop=crop.resize((round(crop.width*zoom),round(crop.height*zoom)),Image.Resampling.BICUBIC)
        draw_boxes(crop,bb,zoom,cropbox[:2])
        x,y=out.width-crop.width-14,42
        d=ImageDraw.Draw(out)
        d.line((cx*scale,cy*scale,x,y+crop.height/2),fill='white',width=1)
        out.paste(crop,(x,y)); d=ImageDraw.Draw(out)
        d.rectangle((x-2,y-2,x+crop.width+2,y+crop.height+2),outline='white',width=2)
        d.rectangle((x,y,x+78,y+28),fill='black')
        d.text((x+6,y+3),'zoom',font=ImageFont.truetype(FONT,19),fill='white')
    filename=ds+'_example.jpg'
    out.save(OUT/filename,quality=95,subsampling=0)
    assert all(sha(Path(path))==digest for path,digest in hashes.items())
    records[ds]={'output':filename,'source':str(p),'labels':str(lp),'split':'test','seed':SEED,'reason':REASONS[ds], 'original_size':[w,h], 'output_size':list(out.size),'box_count':len(bb),'candidate_index_zero_based':index,'candidates':[str(e[1]) for e in candidates],'source_sha256':hashes}
source=ROOT/'runs/detect/drone4_bal/train_batch1.jpg'
shutil.copyfile(source,OUT/'mosaic_example.jpg')
assert sha(source)==sha(OUT/'mosaic_example.jpg')
records['mosaic']={'source':str(source),'output':'mosaic_example.jpg','split':'train','run_seed':0,'rebuilt':False,'sha256':sha(source),'size':list(Image.open(source).size)}
(OUT/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
