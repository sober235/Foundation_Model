"""Finite development evaluation and deployment freeze; never train or read test."""
import argparse
from copy import deepcopy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

SELF_BYTES = Path(__file__).read_bytes()
ROOT = Path(__file__).resolve().parents[1]
INPUTS = ('contract', 'export', 'completion', 'reference_complete', 'reference_contract',
          'reference_summary', 'reference_private')
POLICY = dict(version=1, evidence='NOT_EVIDENCE', role='development_validation',
    total_seconds=43200, threshold_grid=[round(i*.05, 2) for i in range(1,20)],
    threshold_rule='max_sensitivity_within_reference_fp_then_highest_threshold',
    threshold_fallback='strictest_with_explicit_infeasible_budget', scout_score_floor=0.,
    relation_rule='max_min_mode_agreement_then_equal_mode_mean', relation_tie='lookup',
    empty_relation='preserve_development_without_deployment_freeze', stride_mm=8.)


def helpers():
    path = Path(__file__).with_name('aur_supervised_capability.py')
    spec = importlib.util.spec_from_file_location('aur_dev_private_helpers',path)
    result = importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def snapshot():
    from anatobind.aur.unified_evaluation import _evaluator_source
    from anatobind.aur.data_contract import file_sha256
    result = _evaluator_source()
    for name in ('aur_development_selection.py','aur_supervised_capability.py'):
        result['scripts/'+name] = file_sha256(ROOT/'scripts'/name)
    if result['scripts/aur_development_selection.py'] != hashlib.sha256(SELF_BYTES).hexdigest():
        raise ValueError('Loaded development controller source changed')
    return result


def make_registration(inputs, qualified_source, out, crops=None):
    h=helpers()
    return dict(schema_version=1,inputs=deepcopy(inputs),
        qualified_source_root=str(Path(qualified_source).expanduser().resolve()),
        output_root=str(h._private_path(out)),source_snapshot=snapshot(),policy=deepcopy(POLICY),
        crops=deepcopy(crops or {'2d':[1,160,160],'3d':[128,160,160]}))


def check_registration(reg, out):
    h=helpers()
    if (set(reg)!={'schema_version','inputs','qualified_source_root','output_root','source_snapshot','policy','crops'}
            or reg['schema_version']!=1 or reg['policy']!=POLICY
            or reg['output_root']!=str(h._private_path(out)) or set(reg['inputs'])!=set(INPUTS)
            or snapshot()!=reg['source_snapshot']):
        raise ValueError('Development registration differs')
    for value in reg['inputs'].values():
        if (set(value)!={'path','sha256'} or str(Path(value['path']).expanduser().resolve())!=value['path']
                or h._sha(value['path'])!=value['sha256']):
            raise ValueError('Development registered input changed')
    if set(reg['crops'])!={'2d','3d'}:
        raise ValueError('Both mode crops are required')
    for mode,patch in [('2d',(1,4,4)),('3d',(2,4,4))]:
        crop=reg['crops'][mode]
        if (not isinstance(crop,(list,tuple)) or len(crop)!=3
                or any(type(n) is not int or n<=0 or n%p for n,p in zip(crop,patch))
                or (mode=='2d' and crop[0]!=1)):
            raise ValueError('Invalid development crop')


def _execute(registration, out, device, send):
    import torch
    from anatobind.aur import data_contract as dc, unified_evaluation as ue
    from anatobind.aur.development_selection import (choose_relation,choose_thresholds,
        read_reference,selection_rows)
    from anatobind.aur.gate import GATE_V1,judge_r,verdict
    from anatobind.aur.unified_launcher import _json as atomic_json
    h=helpers();torch.set_num_threads(2)
    registration_bytes=Path(registration).read_bytes()
    reg=json.loads(registration_bytes);registration_sha=hashlib.sha256(registration_bytes).hexdigest()
    check_registration(reg,out)
    h._json(out/'registration.json',reg)
    paths={key:Path(value['path']) for key,value in reg['inputs'].items()}
    send({'phase':'admission'})
    atomic_json(out/'progress.json',dict(phase='admission',completed=0,evidence='NOT_EVIDENCE'))
    frozen=dc.read_contract(paths['contract'],source_mode='evaluation',
        qualified_source_root=reg['qualified_source_root'])
    if frozen['contract_sha256']!=reg['inputs']['contract']['sha256']:
        raise ValueError('Qualified development contract changed')
    ref_root=paths['reference_complete'].parent
    for key,name in [('reference_complete','complete.json'),('reference_contract','contract.json'),
                     ('reference_summary','reference_anonymous.json'),('reference_private','reference_private.json')]:
        if paths[key]!=ref_root/name:raise ValueError('Reference locations differ')
    reference=read_reference(ref_root,frozen['contract_sha256'])
    rows=selection_rows(frozen['validation'],reference)
    payload=torch.load(paths['export'],map_location='cpu',weights_only=True)
    if (payload.get('stage')!='III' or payload.get('format_version')!=2
            or payload.get('purpose')!='formal'
            or payload['data_contract']['contract_sha256']!=frozen['contract_sha256']
            or dc.read_json(paths['completion']).get('selected_export_sha256')!=reg['inputs']['export']['sha256']):
        raise ValueError('A completed formal Stage III export is required')
    identity=deepcopy(payload['data_contract']);del payload
    model,metadata=ue.model_from_unified_export(paths['export'],device='cpu',data_contract=identity)
    if metadata['pilot_only'] or not metadata['phase_completed']:raise ValueError('Incomplete formal model')
    admission=h._gpu_admission(device)
    h._json(out/'gpu_admission.json',admission)
    model=model.to(device).eval()
    scout=ue.freeze_gate_observations(rows,modes=('3d',))
    scout.update(artifact_identity=identity,export_sha256=reg['inputs']['export']['sha256'],
        registration_sha256=registration_sha,source_snapshot=reg['source_snapshot'],
        score_threshold=POLICY['scout_score_floor'],purpose='development_threshold_selection')
    dc.write_private_json(out/'selection_plan_private.json',scout)
    records=[];send({'phase':'threshold_selection'})
    for index,obs in enumerate(scout['observations'],1):
        row=rows[obs['row_index']]
        record=ue.evaluate_observation(model,row,obs,crop=tuple(reg['crops']['3d']),
            device=device,score_threshold=0.)
        dc.write_private_json(out/'selection_records'/f'{index:06d}.json',record)
        records.append(record)
        atomic_json(out/'progress.json',dict(phase='threshold_selection',completed=index,
            total=len(scout['observations']),evidence='NOT_EVIDENCE'))
    points=choose_thresholds(records,reference)
    dc.write_private_json(out/'threshold_selection.json',dict(evidence='NOT_EVIDENCE',sources=points,
        plan_sha256=dc.file_sha256(out/'selection_plan_private.json'),
        record_sha256=[dc.file_sha256(out/'selection_records'/f'{i:06d}.json') for i in range(1,len(records)+1)]))
    del records,model;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    diagnostic=dict(gate=deepcopy(GATE_V1),relation_implementation='head',
        validation_selection_sha256=dc.file_sha256(out/'threshold_selection.json'),
        score_thresholds={s:points[s]['thr'] for s in points},frozen_before_evaluation=True)
    dc.write_private_json(out/'diagnostic_registration.json',diagnostic)
    send({'phase':'full_development'})
    def persist(record,index,total):
        dc.write_private_json(out/'development_records'/f'{index:06d}.json',record)
        atomic_json(out/'progress.json',dict(phase='full_development',completed=index,total=total,
            input_mode=record['input_mode'],evidence='NOT_EVIDENCE'))
    ue.run_gate_evaluation(frozen['validation'],frozen['contract_sha256'],paths['export'],out/'development',
        device=device,crops=reg['crops'],score_thresholds=diagnostic['score_thresholds'],
        role_contract=paths['contract'],qualified_source_root=reg['qualified_source_root'],
        gate_registration=diagnostic,expected_registration_sha256=dc.json_digest(diagnostic),on_record=persist)
    records=dc.read_json(out/'development/records_private.json')
    comparison={}
    for track in ('head','lookup'):
        candidate={**diagnostic,'relation_implementation':track}
        comparison[track]={mode:judge_r(records,mode,candidate) for mode in ('2d','3d')}
    diagnostic_metrics=verdict(records,reference,registration=diagnostic)
    selection=dict(evidence='NOT_EVIDENCE',role='development_validation',
        purpose='development_selection_not_final_test',sources=points,relation_comparison=comparison,
        diagnostic_metrics=diagnostic_metrics,development_completion_sha256=dc.file_sha256(out/'development/complete.json'),
        diagnostic_registration_sha256=dc.json_digest(diagnostic),export_sha256=reg['inputs']['export']['sha256'])
    try:chosen=choose_relation(comparison)
    except ValueError:chosen=None
    selection['chosen_relation_implementation']=chosen
    # Reverify full qualified data and all bound sources/inputs before freezing.
    send({'phase':'closing'})
    closed=dc.read_contract(paths['contract'],source_mode='evaluation',qualified_source_root=reg['qualified_source_root'])
    check_registration(reg,out)
    if (closed['contract_sha256']!=frozen['contract_sha256']
            or hashlib.sha256(Path(registration).read_bytes()).hexdigest()!=registration_sha):
        raise ValueError('Development closing commitments changed')
    dc.write_private_json(out/'development_selection.json',selection)
    if chosen is not None:
        final={**diagnostic,'relation_implementation':chosen,
            'validation_selection_sha256':dc.file_sha256(out/'development_selection.json')}
        dc.write_private_json(out/'deployment_registration.json',final)
    outputs={name:dc.file_sha256(out/name) for name in ('registration.json','selection_plan_private.json',
        'threshold_selection.json','diagnostic_registration.json','development_selection.json')}
    if chosen is not None:outputs['deployment_registration.json']=dc.file_sha256(out/'deployment_registration.json')
    dc.write_private_json(out/'complete.json',dict(version='aur_development_selection_v1',
        evidence='NOT_EVIDENCE',outputs=outputs,complete_live_verification=True,
        registration_sha256=registration_sha,deployment_frozen=chosen is not None))
    result=dict(status='COMPLETE',evidence='NOT_EVIDENCE',deployment_frozen=chosen is not None,
        final_test_evaluated=False,fastmri_full_executed=False)
    atomic_json(out/'progress.json',dict(phase='complete',evidence='NOT_EVIDENCE'))
    return result


def run(registration,out,device='cpu'):
    h=helpers();owned=False
    try:
        out=h._private_path(out);out.mkdir(mode=0o700);owned=True
        result=h._supervise(lambda send:_execute(registration,out,device,send),
            total_seconds=POLICY['total_seconds'],phase_limits={})
        h._json(out/'terminal.json',result)
        return result
    except Exception:
        result=dict(h.FAILED)
        if owned:
            try:h._json(out/'terminal.json',result)
            except Exception:pass
        return result


class _Parser(argparse.ArgumentParser):
    def error(self,message):self.exit(2,'Invalid development selection arguments\n')


def main(argv=None):
    p=_Parser(description=__doc__)
    p.add_argument('--registration',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    args=p.parse_args(argv)
    result=run(args.registration,args.out,args.device)
    print(json.dumps({k:result[k] for k in ('status','evidence')}))
    return 0 if result['status']=='COMPLETE' else 1


if __name__=='__main__':sys.exit(main())
