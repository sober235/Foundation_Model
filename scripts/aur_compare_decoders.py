#!/usr/bin/env python
"""Fixed-weight query-CC counterfactual on frozen validation only; resumable private per-row records."""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import signal
import tempfile
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from anatobind.aur import eval as E, infer as I
from anatobind.aur.diagnostics import identity, select_cases, verify_provenance, verify_detections, verify_truth, verify_bindings
from anatobind.aur.query_decoder import decoder_rules
from anatobind.aur.ssl.checkpoint import code_sha, file_sha256
from anatobind.aur.train import split_rows

DATA=Path('/data2/congcong/data/FM_data/derived/aur')
SAVED=Path('/data0/congcong/code/Project_Doing/foundation_model-conformance/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_val_20261014_1344')
SCORES={'bmsr':.70,'isles':.60,'pdgm':.75}
CKPT_SHA='38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde'
SAMPLES_SHA='fb96acd741aa6c0ee689eb772a4332304501c8d63ebd7ccddcd657f41eb86d5d'


def serial(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def write(path,value):
    """Retain evaluator NaNs in private records, as existing evaluation files do."""
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=1,default=serial));os.replace(temp,path)


def freeze_contract(out,contract,resume):
    out=Path(out)
    if out.exists():
        if not resume:raise FileExistsError('output exists; explicit resume required')
        if json.loads((out/'contract.json').read_text()) != contract:
            raise ValueError('resume contract differs; code/rules/data must remain frozen')
    else:
        if resume:raise ValueError('resume contract directory does not exist')
        out.mkdir(parents=True);os.chmod(out,0o700);write(out/'contract.json',contract)


def equal_a(x,y):
    if isinstance(x,dict) and isinstance(y,dict):
        x={str(k):v for k,v in x.items()};y={str(k):v for k,v in y.items()}
        return x.keys()==y.keys() and all(equal_a(x[k],y[k]) for k in x)
    if isinstance(x,(tuple,list)) and isinstance(y,(tuple,list)):
        return len(x)==len(y) and all(equal_a(a,b) for a,b in zip(x,y))
    if isinstance(x,(int,float,np.number)) and isinstance(y,(int,float,np.number)):
        return bool(np.isclose(x,y,atol=1e-10,rtol=0,equal_nan=True))
    return x==y


def verify_reference(actual,saved):
    if identity(actual)!=identity(saved) or not equal_a(actual['a'],saved['a']):
        raise ValueError('baseline reproduction: identity or A scores changed')
    if bool(actual.get('u')) != bool(saved.get('u')):
        raise ValueError('baseline reproduction: U supervision changed')
    if actual.get('u'):
        verify_truth(actual['u']['gt'],saved['u']['gt'])
        verify_detections(actual['legacy_reference_u']['dets'],saved['u']['dets'])
    current=(actual.get('r') or {}).get('instances',[])
    previous=(saved.get('r') or {}).get('instances',[])
    bound=[dict(instance=r['instance'],host=r['r'],host_probs=r['r_probs'],query_iou=r['query_iou'],in_crop_share=r['in_crop_share']) for r in current]
    verify_bindings(bound,previous)
    old={r['instance']:r for r in previous}
    if any(r['truth']!=old[r['instance']]['truth'] or r['b0']!=old[r['instance']]['b0'] for r in current):
        raise ValueError('baseline reproduction: controlled truth/B0 changed')


def fixed_legacy_e2e(row):
    """Re-label cached legacy bindings at the fixed operating point; no model output is changed."""
    result=copy.deepcopy(row)
    if result['source'] in SCORES and result.get('u'):
        result['e2e_score_floor']=SCORES[result['source']]
    if not result.get('e2e') or not result.get('r'):return result
    truth=np.zeros(max(g['instance'] or 0 for g in result['u']['gt']),np.int64)
    for r in result['r']['instances']:truth[r['instance']-1]=r['truth']
    match=E.predicted_detection_truth(result['u']['gt'],result['u']['dets'],truth,SCORES[result['source']])
    for r in result['e2e']:r['truth']=match.get(r['det'])
    return result


def input_manifest(rows):
    """Hash ALL selected image/label bytes before checkpoint loading, caching shared files."""
    cache={};manifest={}
    for row in rows:
        token=hashlib.sha256(json.dumps(identity(row),separators=(',',':')).encode()).hexdigest()
        hashes={}
        for key in ('image','anatomy','lesion'):
            if row.get(key):
                path=str(row[key])
                if path not in cache:cache[path]=file_sha256(path)
                hashes[key]=cache[path]
        manifest[token]=hashes
    return manifest


@contextmanager
def hard_deadline(seconds):
    """OS termination from a watchdog also interrupts a native/GPU work item; safe rows are atomic."""
    if not np.isfinite(seconds) or seconds<=0:raise ValueError('deadline must be positive')
    timer=threading.Timer(seconds,os.kill,args=(os.getpid(),signal.SIGTERM));timer.daemon=True;timer.start()
    try:yield
    finally:timer.cancel()


def finalize_report(path,aggregate,results,reference):
    """Verify complete stages or preserve interrupted stages, then atomically install a whole report."""
    path=Path(path)
    required=('aggregate.json','cases.json','REPORT.md','reference.json')
    if path.exists():
        if all((path/name).is_file() for name in required):
            if not (equal_a(json.loads((path/'aggregate.json').read_text()),aggregate)
                    and equal_a(json.loads((path/'cases.json').read_text()),results)
                    and equal_a(json.loads((path/'reference.json').read_text()),reference)):
                raise ValueError('completed report differs from frozen results')
            return
        path.rename(path.with_name(path.name+'.incomplete-'+uuid.uuid4().hex))
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent,prefix='.'+path.name+'-') as tmp:
        staging=Path(tmp)/'report'
        E.write_report(staging,aggregate,results);write(staging/'reference.json',reference)
        staging.rename(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,default=DATA/'samples_1mm_v1.json')
    p.add_argument('--val-patients',type=Path,default=DATA/'ssl_manifest_v1/val_patients.json')
    p.add_argument('--checkpoint',type=Path,default=DATA/'ssl_runs/c2_stage3_80k_20261014_1344/aur_stage3_best.pt')
    p.add_argument('--saved-eval',type=Path,default=SAVED)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('six','val'),required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--shard',default='0/1')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-seconds',type=int,default=28800)
    a=p.parse_args()
    a.started=time.monotonic()
    with hard_deadline(min(a.max_seconds,7200 if a.mode=='six' else 86400)):
        return run(a)


def run(a):
    old=json.loads((a.saved_eval/'cases.json').read_text())
    old_run=json.loads((a.saved_eval/'aggregate.json').read_text())['run']
    verify_provenance(old_run,a.samples,a.checkpoint)
    if old_run['checkpoint_sha256']!=CKPT_SHA or old_run['samples_sha256']!=SAMPLES_SHA:
        raise ValueError('not the frozen evaluated model/data')
    _,val=split_rows(json.loads(a.samples.read_text()),json.loads(a.val_patients.read_text()))
    if {identity(r) for r in val}!={identity(r) for r in old}:raise ValueError('validation membership differs')
    rows=select_cases(val,old) if a.mode=='six' else sorted(val,key=lambda r:(r['source'],r['case'],r['sequence']))
    k,n=map(int,a.shard.split('/'))
    if not 0<=k<n or a.mode=='six' and (k,n)!=(0,1):raise ValueError('invalid shard')
    rows=rows[k::n]
    manifest=input_manifest(rows)
    baseline_reference=json.loads((a.saved_eval/'reference.json').read_text())
    if a.mode=='six':
        reference={s:E.reference_summary(s,E.reference_scans(s,sorted({r['case'] for r in rows if r['source']==s}))) for s in SCORES}
    else:reference=baseline_reference
    reference_sha=hashlib.sha256(json.dumps(reference,indent=1,default=serial).encode()).hexdigest()
    selection_hash=hashlib.sha256(json.dumps([identity(r) for r in rows],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    contract=dict(code_sha=code_sha(),checkpoint_sha256=CKPT_SHA,samples_sha256=SAMPLES_SHA,
                  val_patients_sha256=file_sha256(a.val_patients),saved_cases_sha256=file_sha256(a.saved_eval/'cases.json'),
                  selection_sha256=selection_hash,mode=a.mode,shard=[k,n],n_rows=len(rows),crop=[128,160,160],batch_size=2,
                  decoder_profile='query_cc_v1',decoder_rules=decoder_rules('query_cc_v1'),score_thresholds=SCORES,
                  e2e_matching='one-to-one after fixed score filter for BOTH decoders',
                  raw_inputs_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                  reference_sha256=reference_sha,saved_reference_sha256=file_sha256(a.saved_eval/'reference.json'),
                  saved_aggregate_sha256=file_sha256(a.saved_eval/'aggregate.json'))
    freeze_contract(a.out,contract,a.resume)
    if (a.out/'COMPLETE.json').exists():print('Frozen evaluation already complete',flush=True);return 0
    write(a.out/'selection_private.json',rows)
    write(a.out/'input_manifest_private.json',manifest)
    write(a.out/'reference_frozen.json',reference)
    records=a.out/'row_records';records.mkdir(exist_ok=True)
    expected={identity(r):r for r in old}
    model,_=I.model_from_export(a.checkpoint,torch.device('cuda',a.gpu))
    t0=a.started;deadline=t0+min(a.max_seconds,7200 if a.mode=='six' else 86400)
    results=[];row_times=[];peak=0.
    for i,row in enumerate(rows):
        if time.monotonic()>deadline:raise TimeoutError('frozen replay time cap exceeded')
        token=hashlib.sha256(json.dumps(identity(row),separators=(',',':')).encode()).hexdigest()
        path=records/(token+'.json')
        hashes={key:file_sha256(row[key]) for key in ('image','anatomy','lesion') if row.get(key)}
        if hashes!=manifest[token]:raise ValueError('raw input changed after initial freeze')
        if path.exists():
            record=json.loads(path.read_text())
            if record['input_sha256']!=hashes or record['code_sha']!=contract['code_sha']:
                raise ValueError('completed row provenance differs on resume')
            result=record['result'];verify_reference(result,expected[identity(row)])
        else:
            start=time.monotonic();torch.cuda.reset_peak_memory_stats(a.gpu)
            result=E.evaluate_case(model,row,(128,160,160),torch.device('cuda',a.gpu),batch_size=2,
                                   decoder_profile='query_cc_v1',decoder_workers=4,bind_score_threshold=SCORES.get(row['source']))
            verify_reference(result,expected[identity(row)])
            record=dict(code_sha=contract['code_sha'],input_sha256=hashes,result=result,
                        seconds=time.monotonic()-start,peak_cuda_gib=torch.cuda.max_memory_allocated(a.gpu)/2**30)
            write(path,record)
        results.append(result);row_times.append(record['seconds']);peak=max(peak,record['peak_cuda_gib'])
        progress=dict(completed=i+1,total=len(rows),source=row['source'],row_seconds=round(record['seconds'],2),
                      elapsed_seconds=round(time.monotonic()-t0,2),peak_cuda_gib=round(peak,3),reproduction='PASS')
        write(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    write(a.out/'cases.json',results)
    info=dict(shard=[k,n],checkpoint=str(a.checkpoint),checkpoint_sha256=CKPT_SHA,code_sha=contract['code_sha'],split='val',
              samples=str(a.samples),samples_sha256=SAMPLES_SHA,crop=[128,160,160],n_rows=len(results),sources=None,limit=None,
              instance_threshold=.3,stage='III',e2e=True,decoder_profile='query_cc_v1',decoder_rules=contract['decoder_rules'],
              binding_score_floors=SCORES,seconds=round(sum(row_times),2),seconds_basis='sum of frozen per-row timings')
    info.update(reference_json=str(a.out/'reference_frozen.json'),reference_sha256=reference_sha)
    write(a.out/'shard.json',info)
    if n==1:
        aggregate=E.aggregate(results,reference,fixed_thresholds=SCORES)
        aggregate['run']={**info,'mode':a.mode,'fixed_thresholds':SCORES}
        finalize_report(a.out/'query',aggregate,results,reference)
        legacy=[fixed_legacy_e2e(expected[identity(r)]) for r in rows]
        old_agg=E.aggregate(legacy,reference,fixed_thresholds=SCORES)
        old_agg['run']={**info,'decoder_profile':'legacy_max','decoder_rules':decoder_rules('legacy_max'),
                        'note':'A/U/R predictions reused; e2e GT matching normalized at fixed scores'}
        finalize_report(a.out/'legacy_fixed',old_agg,legacy,reference)
    done=dict(status='complete',mode=a.mode,shard=[k,n],rows=len(results),seconds=round(time.monotonic()-t0,2),
              recorded_row_seconds=round(sum(row_times),2),peak_cuda_gib=peak,all_reproduction_checks_passed=True)
    write(a.out/'COMPLETE.json',done);print(json.dumps(done),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
