#!/usr/bin/env python
"""Optimistic score-order feasibility on archived development cases; no chosen threshold."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from anatobind.aur import eval as E
import aur_diagnose_confidence as F

PUBLIC_CONFIDENCE_RUN=ROOT/'docs/verification/2026-10-17/aur_residual_confidence/RUN.json'
PUBLIC_QUERY_SUMMARY=ROOT/'docs/verification/2026-10-17/aur_query_cc/six_probe_accelerated.json'
QUERY_SIX=F.REFERENCE_SIX
ARCHIVE=Path('/home/congcongliu/outputs/aur_residual_confidence_20261017/replay_6b26137')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_order_bound(scans,fp_budget):
    if not math.isfinite(fp_budget) or fp_budget<0:
        raise ValueError('Invalid FP/scan budget')
    values=[float(d['score']) for scan in scans for d in scan['dets']]
    if any(not math.isfinite(s) or not 0<=s<=1 for s in values):
        raise ValueError('Scores must be finite probabilities')
    scores=sorted(set(values),reverse=True)
    cap=math.floor(fp_budget*len(scans))
    best=feasible=0
    for cut in [math.inf]+scores:
        point=E.sensitivity_at_threshold(scans,cut)
        if point['n_fp']<=cap:
            feasible+=1;best=max(best,point['n_hit'])
    n_gt=sum(not g.get('ignore') for s in scans for g in s['gt'])
    return dict(n_scans=len(scans),n_gt=n_gt,n_ignored_gt=sum(bool(g.get('ignore')) for s in scans for g in s['gt']),
        n_detections=len(values),n_unique_scores=len(scores),n_operating_sets=len(scores)+1,
        n_feasible_sets=feasible,fp_per_scan_budget=float(fp_budget),integer_fp_cap=cap,
        max_observed_hits_within_budget=best,optimistic_sensitivity=best/n_gt if n_gt else None)


def load_archive(run,anchor_path=PUBLIC_CONFIDENCE_RUN,query_reference=QUERY_SIX,query_summary=PUBLIC_QUERY_SUMMARY):
    """Validate historical observations with their own contract; never resume inference."""
    run,query_reference=Path(run),Path(query_reference)
    anchor=F.read(anchor_path)
    for name,key in [('contract.json','private_contract_sha256'),('COMPLETE.json','private_complete_sha256'),
                     ('input_manifest_private.json','private_input_manifest_sha256')]:
        if sha(run/name)!=anchor[key]:raise ValueError('Archived file differs from public commitment')
    contract,done,aggregate=F.read(run/'contract.json'),F.read(run/'COMPLETE.json'),F.read(run/'aggregate.json')
    if (done.get('status')!='complete' or done.get('rows')!=6 or done.get('all_reproduction_checks_passed') is not True
            or contract['code_sha']!=anchor['code_sha'] or contract['n_rows']!=6
            or sha(run/'aggregate.json')!=done['aggregate_sha256']
            or done['aggregate_sha256']!=anchor['artifact_hashes']['aggregate_anonymous.json']):
        raise ValueError('Incomplete or changed archive/aggregate')
    for name,h in {**contract['core_dependency_sha256'],**contract['diagnostic_source_sha256']}.items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen source dependency changed')
    public=F.read(query_summary);old_anchor=public['provenance'][0]
    if (sha(query_summary)!=contract['reference_summary_sha256']
            or sha(query_reference/'contract.json')!=contract['reference_contract_sha256']
            or contract['reference_contract_sha256']!=old_anchor['contract_sha256']
            or sha(query_reference/'COMPLETE.json')!=contract['reference_complete_sha256']):
        raise ValueError('Original query reference commitment changed')
    old_contract=F.read(query_reference/'contract.json')
    if (sha(query_reference/'reference_frozen.json')!=old_contract['reference_sha256']
            or old_contract['reference_sha256']!=contract['reference_native_snapshot_sha256']):
        raise ValueError('Six-patient native reference changed')
    native=F.read(query_reference/'reference_frozen.json')
    rows=F.read(run/'selection_private.json')
    selection=hashlib.sha256(json.dumps([F.identity(r) for r in rows],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    names={F.token(r)+'.json' for r in rows}
    if (len(rows)!=6 or len(names)!=6 or selection!=contract['selection_sha256']
            or names!={p.name for p in (run/'row_records').glob('*.json')}):
        raise ValueError('Archived row membership changed')
    original=[];records=[]
    for row in rows:
        key=F.token(row);path=query_reference/'row_records'/(key+'.json')
        if sha(path)!=contract['reference_row_files'][key]:raise ValueError('Original query row changed')
        original.append(F.read(path));records.append(F.read(run/'row_records'/(key+'.json')))
    if (F.records_digest(original)!=contract['reference_rows_content_sha256']
            or F.digest(records)!=done['records_sha256'] or done['records_sha256']!=anchor['private_records_content_sha256']):
        raise ValueError('Frozen record corpus differs')
    manifest=F.read(run/'input_manifest_private.json')
    if F.digest(manifest)!=contract['raw_inputs_sha256']:raise ValueError('Archived input manifest commitment differs')
    prepared=dict(contract=contract,manifest=manifest,records={F.identity(r['result']):r for r in original})
    for row,record in zip(rows,records):F.validate_cached(record,row,prepared)
    payloads=[r['payload'] for r in records]
    rebuilt=F.aggregate_records(payloads)
    if not F.exact(rebuilt,{k:v for k,v in aggregate.items() if k!='provenance'}):
        raise ValueError('Published fixed-point aggregate does not reproduce')
    children=sum(len(r['measurement']['children']) for r in payloads)
    parents=sum(len(r['measurement']['parents']) for r in payloads)
    if (children!=anchor['observed']['measured_children'] or parents!=anchor['observed']['active_parents']
            or not F.exact(aggregate['controls'],anchor['observed']['controls'])):
        raise ValueError('Published measurement/control counts differ')
    registrations={r['source']+'/'+r['sequence'] for r in payloads}
    for key in registrations:
        source=key.split('/')[0];registered=public['source_sequence'][key]
        cases=[r for r in payloads if r['source']+'/'+r['sequence']==key]
        ref=native[source]
        if (registered['S7_comparable_sequence'] is not True or not F.exact(ref,registered['S7'])
                or ref['n_scans']!=len(cases) or ref['n_gt']!=sum(not g.get('ignore') for r in cases for g in r['gt'])
                or ref['fp_per_scan']!=ref['n_fp']/ref['n_scans']):
            raise ValueError('Reference registration/patient counts/budget differ')
    return dict(payloads=payloads,reference=native,controls=aggregate['controls'],
        provenance=dict(archive_contract_sha256=sha(run/'contract.json'),archive_complete_sha256=sha(run/'COMPLETE.json'),
            archive_records_sha256=done['records_sha256'],native_reference_sha256=old_contract['reference_sha256'],
            public_anchor_sha256=sha(anchor_path),query_summary_sha256=sha(query_summary),
            measurement_code_sha=contract['code_sha'],frozen_source_files=len(contract['core_dependency_sha256'])+len(contract['diagnostic_source_sha256'])))


def analyze(prepared):
    groups=defaultdict(list)
    for r in prepared['payloads']:groups[r['source']+'/'+r['sequence']].append(r)
    out=dict(evidence='NOT_EVIDENCE',scope='Optimistic exact retained-set ceiling on previously exposed six development cases',
        n_rows=len(prepared['payloads']),source_sequence={},controls=prepared['controls'],provenance=prepared['provenance'])
    total=0
    for key,cases in sorted(groups.items()):
        ref=prepared['reference'][cases[0]['source']];variants={}
        for name in F.SCORE_NAMES:
            scans=[E.scan('anonymous',r['gt'],F.score_copies(r['detections'],r['measurement']['children'],name)) for r in cases]
            variants[name]=score_order_bound(scans,ref['fp_per_scan']);total+=variants[name]['n_operating_sets']
        out['source_sequence'][key]=dict(reference={k:ref[k] for k in ('n_scans','n_gt','n_hit','n_fp','sensitivity','fp_per_scan')},score_order_bounds=variants)
    out['n_operating_sets_total']=total
    out['limitations']=['Exposed development sample; maximization is optimistic and not a selected operating point.',
        'Only per-source final-scalar monotone maps share these retained sets; voxel transforms/new features/nonmonotone models are excluded.',
        'Official summed-IoU Hungarian then cutoff matching is preserved; no conventional ROC interpolation or monotonic-hit assumption.',
        'No score/mask/R improvement, calibration fit, independent evaluation or clinical claim; complete405valFAIL/defaultlegacy_max unchanged.']
    return out


def run(a):
    if a.out.exists():raise FileExistsError('Use a fresh output directory')
    started=time.monotonic();prepared=load_archive(a.run)
    contract=dict(code_sha=F.code_sha(),source_sha256=sha(__file__),provenance=prepared['provenance'],
                  method='All exact distinct inclusive-score retained sets plus empty set; no threshold output',max_seconds=a.max_seconds)
    F.C.freeze_contract(a.out,contract,False)
    result=analyze(prepared);result['seconds']=time.monotonic()-started
    F.C.write(a.out/'aggregate.json',result)
    F.C.write(a.out/'COMPLETE.json',dict(status='complete',all_original_controls_passed=True,
        rows=result['n_rows'],operating_sets=result['n_operating_sets_total'],aggregate_sha256=sha(a.out/'aggregate.json')))
    print(json.dumps(dict(status='complete',rows=result['n_rows'],operating_sets=result['n_operating_sets_total'],seconds=round(result['seconds'],2))),flush=True)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ARCHIVE);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=600);a=p.parse_args()
    if not 0<a.max_seconds<=600:p.error('max-seconds must be in1..600')
    with F.C.hard_deadline(a.max_seconds):return run(a)


if __name__=='__main__':raise SystemExit(main())
