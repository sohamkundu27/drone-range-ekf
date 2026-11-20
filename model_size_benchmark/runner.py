"""Sequential, restart-safe benchmark. Only writes benchmark outputs; no existing run is modified."""
import argparse, csv, hashlib, importlib.util, json, os, platform, random, re, shutil, subprocess, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'model_size_benchmark'
PYTHON = str(ROOT / '.venv/bin/python')
MISSING = 'NOT MEASURED'
SIZES = ['n','s','m']
BACKENDS = ['pytorch_fp32','pytorch_fp16','onnx_fp32','onnx_fp16']

def write_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2,default=str)+'\n'); tmp.replace(p)
def read_json(p):
    return json.loads(p.read_text()) if p.exists() else {}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name):
    p=OUT/'sources'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('bench_'+name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def worker(size,stage,batch):
    import torch, ultralytics, yaml
    from ultralytics import YOLO
    d=OUT/size; d.mkdir(exist_ok=True); (d/'experiments').mkdir(exist_ok=True)
    if stage=='train':
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        class FixedBatchTrainer(DetectionTrainer):
            def _build_train_pipeline(self):
                if self.batch_size != batch:
                    raise torch.cuda.OutOfMemoryError('Automatic batch reduction blocked; only orchestrator may retry m at batch 4.')
                return super()._build_train_pipeline()
        args=yaml.safe_load((OUT/'sources/s_args.yaml').read_text())
        for k in ['save_dir','model','project','name','exist_ok']: args.pop(k,None)
        args.update(batch=batch,nbs=64,project=str(OUT/'train'),name=f'{size}_batch{batch}',exist_ok=False)
        run=OUT/'train'/f'{size}_batch{batch}'
        if run.exists(): raise RuntimeError(f'Refusing existing run: {run}')
        weights=OUT/'pretrained'/f'yolo26{size}.pt'
        model=YOLO(str(weights))
        ckpt=model.ckpt
        pre={'path':str(weights),'sha256':sha(weights),'version':ckpt.get('version'),
             'train_args':ckpt.get('train_args'), 'names':model.names}
        write_json(d/'pretrained_provenance.json',pre)
        if len(model.names)!=80 or 'coco' not in str(ckpt.get('train_args',{}).get('data','')).lower():
            raise RuntimeError('COCO pretraining metadata could not be verified; approval required.')
        write_json(d/f'requested_batch{batch}.json',args)
        start=time.monotonic()
        model.train(trainer=FixedBatchTrainer,**args)
        rows=list(csv.DictReader((run/'results.csv').open()))
        best=max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))
        write_json(d/'training.json',{'run':str(run),'weights':str(run/'weights/best.pt'),'batch':batch,'nbs':64,
            'epochs_completed':int(rows[-1]['epoch']),'best_epoch':int(best['epoch']),
            'best_validation_map5095':float(best['metrics/mAP50-95(B)']),
            'csv_elapsed_seconds':float(rows[-1]['time']),'worker_train_seconds':time.monotonic()-start,
            'reused':False,'deviation':('m batch 8 OOM; fresh retry at batch 4, nbs=64' if batch==4 else 'none')})
        return
    training=read_json(d/'training.json'); w=Path(training['weights'])
    if stage.startswith('eval_'):
        tag=stage[5:]; data={'dut':'dut_anti_uav','combined':'drone4_bal','detfly':'detfly'}[tag]
        m=YOLO(str(w)).val(data=str(ROOT/f'data/yolo/{data}.yaml'),split='test',imgsz=1280,batch=8,
            conf=0.001,iou=0.7,device=0,project=str(d/'eval'),name=tag,exist_ok=False,plots=True)
        result={'weights':str(w),'data':data,'split':'test','imgsz':1280,'batch':8,'conf':0.001,'iou':0.7,
            'overall':{'P':float(m.box.mp),'R':float(m.box.mr),'mAP50':float(m.box.map50),'mAP50-95':float(m.box.map)},
            'per_class':{m.names[int(c)]:{'P':float(m.box.p[i]),'R':float(m.box.r[i]),'mAP50':float(m.box.ap50[i]),
            'mAP50-95':float(m.box.ap[i]),'instances':int(m.nt_per_class[int(c)])} for i,c in enumerate(m.box.ap_class_index)}}
        write_json(d/f'{stage}.json',result)
    elif stage=='false_positives':
        mod=module('false_positives');mod.YOLO_DIR=ROOT/'data/yolo'
        sys.argv=['false_positives.py','--weights',str(w),'--out',str(d/'false_positives.json')];mod.main()
    elif stage=='complexity':
        from ultralytics.utils.torch_utils import get_flops, get_num_params
        model=YOLO(str(w)).model.float().eval()
        params=get_num_params(model)
        # Report deployed, fused NMS-head model complexity consistently across sizes.
        model.end2end=False;model.fuse(verbose=False)
        flops=get_flops(model,imgsz=1280)
        write_json(d/'complexity.json',{'checkpoint_parameters':params,'fused_parameters':get_num_params(model),
            'gflops_1280':flops if flops>0 else MISSING,'method':'Ultralytics get_flops (2*THOP MACs), fused FP32 NMS-head model, square 1x3x1280x1280; checkpoint count includes retained training branches'})
    elif stage.startswith('export_'):
        half=stage=='export_fp16'; dest=d/'artifacts'/('fp16' if half else 'fp32');dest.mkdir(parents=True,exist_ok=True)
        copy=dest/'best.pt'
        if copy.exists(): raise RuntimeError('Export checkpoint copy exists')
        shutil.copy2(w,copy)
        YOLO(str(copy)).export(format='onnx',imgsz=1280,half=half,simplify=True,opset=None,device=0 if half else 'cpu')
    elif stage.startswith('latency_'):
        label=stage[8:];half=label.endswith('fp16')
        lw=w if label.startswith('pytorch') else d/'artifacts'/('fp16' if half else 'fp32')/'best.onnx'
        if not lw.exists(): raise RuntimeError('Missing latency model')
        # Same original function and sampling; change output ROOT only. Verify fixed image manifest.
        files=sorted((ROOT/'data/yolo/dut_anti_uav/images/test').glob('*.jpg'));random.Random(0).shuffle(files)
        assert [str(p) for p in files[:220]]==read_json(OUT/'latency_images.json')['images']
        mod=module('latency');mod.ROOT=d
        sys.argv=['latency.py','--weights',str(lw),'--images',str(ROOT/'data/yolo/dut_anti_uav/images/test'),
                  '--imgsz','1280','--n','200','--warmup','20','--label',label]+(['--half'] if half else [])
        mod.main()
        if label.startswith('onnx'):
            # Source runner already imported torch; independently verify CUDA session is loadable.
            import onnxruntime as ort
            sess=ort.InferenceSession(str(lw),providers=['CUDAExecutionProvider','CPUExecutionProvider'])
            write_json(d/f'{label}_providers.json',{'active':sess.get_providers()})
            if 'CUDAExecutionProvider' not in sess.get_providers():
                p=d/'experiments/latency.json';r=read_json(p);r.pop(label,None);write_json(p,r)
                raise RuntimeError('CUDA provider not active; latency rejected')
    else: raise ValueError(stage)

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows=[]
    for size in SIZES:
        d=OUT/size;t=read_json(d/'training.json'); comp=read_json(d/'complexity.json')
        row={'size':size,'status':read_json(d/'status.json').get('status','pending'),
             'training_stage_total_seconds':sum(c['elapsed_seconds'] for c in [json.loads(l) for l in (OUT/'commands.jsonl').read_text().splitlines()] if '--size' in c['command'] and c['command'][c['command'].index('--size')+1]==size and c['command'][c['command'].index('--stage')+1]=='train') if (OUT/'commands.jsonl').exists() and size!='s' and t else MISSING,
             **{k:t.get(k,MISSING) for k in ['batch','epochs_completed','best_epoch','csv_elapsed_seconds','worker_train_seconds','early_stopping','deviation']},
             **{k:comp.get(k,MISSING) for k in ['checkpoint_parameters','fused_parameters','gflops_1280']}}
        for tag in ['dut','combined','detfly']:
            ev=read_json(d/f'eval_{tag}.json')
            for cls in ['overall','drone','bird','airplane','helicopter']:
                metrics=ev.get('overall',{}) if cls=='overall' else ev.get('per_class',{}).get(cls,{})
                for metric in ['P','R','mAP50','mAP50-95']:
                    row[f'{tag}_{cls}_{metric}']=metrics.get(metric,MISSING)
        fp=read_json(d/'false_positives.json').get('groups',{}).get('all/any',{}).get('0.25',{})
        row['false_drones_conf025']=fp.get('false_drones',MISSING);row['fp_test_images']=fp.get('images',MISSING)
        lat=read_json(d/'experiments/latency.json')
        for backend in BACKENDS:
            for metric in ['mean_ms','p95_ms','preprocess_ms','inference_ms','postprocess_ms','n_images']:
                row[f'{backend}_{metric}']=lat.get(backend,{}).get(metric,MISSING)
        rows.append(row)
    tmp=OUT/'results.csv.tmp'
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(OUT/'results.csv')
    def num(x):return isinstance(x,(int,float))
    fig,ax=plt.subplots(figsize=(8,5))
    measured=[]
    for r in rows:
        x=r['pytorch_fp16_mean_ms'];y=r['dut_drone_mAP50-95']
        if num(x) and num(y):
            measured.append(r);ax.scatter(x,y,s=130 if r['size']=='s' else 70,color='#d55e00' if r['size']=='s' else '#0072b2');ax.annotate('YOLO26'+r['size'],(x,y),xytext=(7,5),textcoords='offset points')
    ax.set(xlabel='PyTorch FP16 end-to-end mean latency (ms)',ylabel='DUT test drone mAP50–95',title='Desktop GPU RTX 3080 — YOLO26 size comparison')
    ax.grid(alpha=.2)
    if len(measured)<3: ax.text(.02,.02,'INCOMPLETE: missing measurements are not plotted',transform=ax.transAxes)
    fig.tight_layout();fig.savefig(OUT/'latency_vs_map.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));x=np.arange(3)
    for i,(key,label) in enumerate([('dut_drone_mAP50-95','DUT drone mAP50–95'),('dut_drone_R','DUT drone recall'),('detfly_drone_R','Det-Fly recall')]):
        vals=[r[key] if num(r[key]) else np.nan for r in rows];ax.bar(x+(i-1)*.25,vals,.25,label=label)
    ax.set_xticks(x,['YOLO26'+s for s in SIZES]);ax.set(ylim=(0,1),ylabel='Metric',title='YOLO26 test accuracy — missing values omitted');ax.legend();fig.tight_layout();fig.savefig(OUT/'tradeoff.png',dpi=160);plt.close(fig)
    lines=['# YOLO26 model-size benchmark','',
        '**l and x were NOT run.** Only m and n are trained; existing s is reused. Results update after every stage.',
        '', '## Protocol',
        'Equal time budget, not equal epochs: each new model requests epochs=150, time=7.0 hours, patience=30. Existing s used that same budget and stopped early. All use seed=0, deterministic=false, imgsz=1280, batch=8, nbs=64, AMP, workers=12, optimizer=auto (s selected MuSGD), and the complete saved s args in sources/s_args.yaml. Only pretrained architecture, output locations, and the explicitly authorized m batch-4 OOM fallback may differ. A fallback restarts from pretrained weights in a separate run directory; failed-attempt duration is separately logged.',
        'Data: original data/yolo/drone4_bal.yaml, unchanged. Best checkpoint selected on combined validation mAP50–95. Test evaluation: DUT, combined, partial 1,000-image Det-Fly; batch=8, imgsz=1280, conf=0.001, NMS IoU=0.7, GPU 0. No evaluation batch fallback. P/R use the library mean-F1-selected confidence, not conf 0.25.',
        'False drone count: existing false_positives.py on COCO/Open Images test, conf=0.25. Latency: frozen original latency.py, 20 warmup + 200 timed single-image predict calls, seed-0 shuffled DUT image list in latency_images.json; arrays decoded before timing. Includes preprocessing/transfers, inference, NMS/postprocessing and Python overhead; excludes model load, disk/JPEG reads and warmup. All four original backend/precision combinations; ONNX exports static 1280-square while PyTorch retains original rectangular preprocessing. Images have mixed native resolutions. Latencies are desktop GPU measurements.',
        'Complexity: checkpoint parameter count and fused deployment parameter count are separate; GFLOPs at square 1280 use fused NMS-head model and 2×THOP MACs. A profiler failure is NOT MEASURED.',
        '', '## Commands and provenance',
        'Exact executed commands, return codes, and durations: commands.jsonl. Worker implementations: runner.py. Original scripts are frozen under sources/ with hashes in provenance.json. Requested train arguments, checkpoint provenance, per-stage JSON and logs are retained. All writes are isolated here. Existing experiments/ and runs/ are not overwritten.',
        'Launch: `nohup .venv/bin/python -u model_size_benchmark/runner.py run > model_size_benchmark/logs/orchestrator.log 2>&1 &`',
        '', '## Hardware/software','```json',json.dumps(read_json(OUT/'environment.json'),indent=2),'```','',
        '## Results','Full per-class and overall metrics, all latencies and timing fields: results.csv. NOT MEASURED means unavailable; absent classes in single-class test sets are also NOT MEASURED.',
        '| Size | Status | Epochs | Best epoch | Train seconds (CSV) | Early stop | DUT AP50–95 | DUT recall | Det-Fly recall | PT FP16 mean ms | Deviation |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:lines.append('| '+' | '.join(str(r[k]) for k in ['size','status','epochs_completed','best_epoch','csv_elapsed_seconds','early_stopping','dut_drone_mAP50-95','dut_drone_R','detfly_drone_R','pytorch_fp16_mean_ms','deviation'])+' |')
    lines += ['', '### Backend latency and model complexity', '| Size | Backend | Mean ms | p95 ms | Timed images | Checkpoint parameters | Fused parameters | GFLOPs at 1280 |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        for b in BACKENDS:
            lines.append('| '+ ' | '.join(str(v) for v in [r['size'],b,r[b+'_mean_ms'],r[b+'_p95_ms'],r[b+'_n_images'],r['checkpoint_parameters'],r['fused_parameters'],r['gflops_1280']])+' |')
    lines += ['', 'Training-stage total seconds in results.csv includes downloads/setup/final validation and failed attempts; csv_elapsed_seconds is the library training-loop duration. Existing s only has its historical loop duration.', '', '### Executed worker commands', '```bash']
    import shlex
    if (OUT/'commands.jsonl').exists():
        lines += [shlex.join(json.loads(l)['command']) for l in (OUT/'commands.jsonl').read_text().splitlines()]
    lines += ['```']
    lines+=['','## Honest interpretation','Single seed (0); no repeated-seed uncertainty estimate. Small differences may be run-to-run variation and must not be called real improvements. Dominance below uses strictly slower PyTorch FP16 mean latency AND lower DUT drone mAP50–95; other deployment backends can rank differently.']
    if len(measured)<3:lines.append('Comparison pending: where s sits, dominance across all three sizes, and the value of added size are NOT MEASURED until all three comparable measurements are available.')
    else:
        for a in measured:
            dominators=[b['size'] for b in measured if b['pytorch_fp16_mean_ms']<a['pytorch_fp16_mean_ms'] and b['dut_drone_mAP50-95']>a['dut_drone_mAP50-95']]
            lines.append(f"YOLO26{a['size']}: observed dominators = {', '.join(dominators) or 'none'}.")
        s=next(r for r in rows if r['size']=='s')
        for r in rows:
            if r['size']!='s':lines.append(f"{r['size']} minus s: DUT AP50–95 delta {r['dut_drone_mAP50-95']-s['dut_drone_mAP50-95']:+.6f}; PT FP16 latency delta {r['pytorch_fp16_mean_ms']-s['pytorch_fp16_mean_ms']:+.4f} ms.")
        lines.append('These are observed differences only. Whether an accuracy gain is meaningful for deployment requires a specified minimum benefit and repeated-seed measurements; this benchmark cannot establish statistical significance. Any m batch-4 deviation also limits attribution solely to model size.')
    lines+=['','## Failures / pending actions',json.dumps(read_json(OUT/'state.json'),indent=2)]
    (OUT/'BENCHMARK.md').write_text('\n'.join(lines)+'\n')

def call(size,stage,batch=8):
    cmd=[PYTHON,'-u',str(OUT/'runner.py'),'worker','--size',size,'--stage',stage,'--batch',str(batch)]
    log=OUT/'logs'/f'{size}_{stage}_batch{batch}.log'
    if log.exists(): raise RuntimeError(f'Refusing to overwrite {log}')
    start=time.time();write_json(OUT/size/'status.json',{'status':stage,'started_unix':start});report()
    with log.open('x') as f:rc=subprocess.call(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps({'command':cmd,'cwd':str(ROOT),'log':str(log),'returncode':rc,'elapsed_seconds':time.time()-start,'started_unix':start})+'\n')
    if stage=='train' and rc==0:
        p=OUT/size/'training.json';d=read_json(p);text=log.read_text();d['early_stopping']='EarlyStopping: Training stopped early' in text;write_json(p,d)
    report();return rc,log

def run():
    import fcntl
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'running','failures':[]};write_json(OUT/'state.json',state)
    for size in ['m','n','s']:
        if size!='s':
            rc,log=call(size,'train')
            if rc and size=='m' and any(t in log.read_text().lower() for t in ['out of memory','cublas_status_alloc_failed','cuda backend memory error']):
                rc,log=call(size,'train',4)
            if rc:
                state['failures'].append({'size':size,'stage':'train','log':str(log),'action':'User input required; no additional recipe changes authorized.'})
                write_json(OUT/size/'status.json',{'status':'training failed'});write_json(OUT/'state.json',state);report();continue
        for stage in ['eval_dut','eval_combined','eval_detfly','false_positives','complexity','export_fp32','export_fp16']+['latency_'+b for b in BACKENDS]:
            rc,log=call(size,stage)
            if rc:
                state['failures'].append({'size':size,'stage':stage,'log':str(log),'action':'NOT MEASURED; settings not changed.'});write_json(OUT/'state.json',state)
        write_json(OUT/size/'status.json',{'status':'complete' if not any(f['size']==size for f in state['failures']) else 'complete with failures'});report()
    state['status']='complete' if not state['failures'] else 'finished with failures requiring review';write_json(OUT/'state.json',state);report()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','worker','report']);ap.add_argument('--size',choices=SIZES);ap.add_argument('--stage');ap.add_argument('--batch',type=int,default=8);a=ap.parse_args()
    if a.action=='worker':worker(a.size,a.stage,a.batch)
    elif a.action=='report':report()
    else:
        try:run()
        except Exception:
            write_json(OUT/'state.json',{'status':'orchestrator failed','traceback':traceback.format_exc()});report();raise
