"""Development selection must not use test data or stale threshold matches."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
from test_aur_unified_evaluation import qualified
from test_aur_unified_infer import supervised_boundaries


def api():
    return importlib.import_module('anatobind.aur.development_selection')


def source_reference(source, row_hash='a' * 64, budget=0):
    return dict(source=source, included_row_sha256=[row_hash], n_scans=1,
                n_gt=1, n_hit=1, n_fp=budget, sensitivity=1., fp_per_scan=float(budget),
                grid='aur_1mm_canonical')


def test_threshold_selection_rematches_full_candidates_within_exact_reference_rows():
    # Removing the lower-score, better-overlap detection changes the GT match.
    record = dict(source='isles', input_mode='3d', row_sha256='a'*64,
        u=dict(gt=[dict(instance=1, box=[0,0,0,2,2,2])], detections=[
            dict(instance=1, box=[0,0,0,2,2,2], score=.4),
            dict(instance=2, box=[0,0,0,2,2,3], score=.8)]))
    refs={s:source_reference(s) for s in ('pdgm','bmsr','isles')}
    records=[dict(deepcopy(record),source=s) for s in refs]
    points=api().choose_thresholds(records,refs)
    assert points['isles']['thr']==.8
    assert points['isles']['n_hit']==1 and points['isles']['n_fp']==0
    assert points['isles']['within_budget'] is True
    with pytest.raises(ValueError):api().choose_thresholds(records+[record],refs)
    damaged=deepcopy(records);damaged[0]['row_sha256']='b'*64
    with pytest.raises(ValueError):api().choose_thresholds(damaged,refs)


def test_infeasible_budget_is_not_reported_as_feasible():
    refs={s:source_reference(s) for s in ('pdgm','bmsr','isles')}
    records=[dict(source=s,input_mode='3d',row_sha256='a'*64,
        u=dict(gt=[dict(instance=1,box=[0,0,0,2,2,2])],detections=[
            dict(instance=1,box=[0,0,0,2,2,2],score=.99),
            dict(instance=2,box=[10,10,10,12,12,12],score=.99)])) for s in refs]
    p=api().choose_thresholds(records,refs)['isles']
    assert p['thr']==.95 and p['within_budget'] is False and p['n_fp']==1


def test_r_choice_protects_weaker_mode_and_lookup_wins_exact_tie():
    comparison={'head':{'2d':dict(n=10,agreement=.9),'3d':dict(n=10,agreement=.7)},
                'lookup':{'2d':dict(n=10,agreement=.8),'3d':dict(n=10,agreement=.8)}}
    assert api().choose_relation(comparison)=='lookup'
    comparison['head']=deepcopy(comparison['lookup'])
    assert api().choose_relation(comparison)=='lookup'
    comparison['head']['2d']['agreement']=.9
    assert api().choose_relation(comparison)=='head'
    comparison['head']['3d']=dict(n=0,agreement=None)
    with pytest.raises(ValueError):api().choose_relation(comparison)
    comparison['head']['3d']=dict(status='unavailable',evidence='NOT_EVIDENCE')
    with pytest.raises(ValueError):api().choose_relation(comparison)


def test_selection_rows_require_development_and_exact_source_sequence():
    refs={s:source_reference(s) for s in ('pdgm','bmsr','isles')}
    row=dict(source='isles',sequence='DWI',u_supervised=True,
             role='development_validation',split='val')
    from anatobind.aur.data_contract import json_digest
    refs['isles']['included_row_sha256']=[json_digest(row)]
    rows=[row]
    with pytest.raises(ValueError):api().selection_rows(rows,refs)
    bad=dict(row,role='historical_test',split='test')
    with pytest.raises(ValueError):api().selection_rows([bad],refs)


def test_reference_outputs_and_role_are_verified_before_selection(tmp_path):
    from anatobind.aur.data_contract import file_sha256
    p=tmp_path/'reference';p.mkdir()
    summary={'version':'aur_reference_1mm_v1','contract_sha256':'a'*64,'sources':{}}
    (p/'reference_anonymous.json').write_text(json.dumps(summary))
    (p/'reference_private.json').write_text('{}')
    outputs={n:file_sha256(p/n) for n in ('reference_anonymous.json','reference_private.json')}
    (p/'contract.json').write_text(json.dumps(dict(version='aur_reference_1mm_v1',
        outputs=outputs,data_contract_sha256='a'*64,parent_contract_sha256=None)))
    (p/'complete.json').write_text(json.dumps(dict(version='aur_reference_1mm_v1',
        outputs=outputs,contract_sha256=file_sha256(p/'contract.json'))))
    assert api().read_reference(p,'a'*64)=={}
    with pytest.raises(ValueError):api().read_reference(p,'b'*64)
    (p/'reference_private.json').write_text('changed')
    with pytest.raises(ValueError):api().read_reference(p,'a'*64)


def test_gate_progress_keeps_completed_observations_on_later_failure(qualified,monkeypatch,tmp_path,supervised_boundaries):
    import torch
    from anatobind.aur import unified_evaluation as ue, data_contract as dc
    from anatobind.aur.gate import GATE_V1
    from test_aur_unified_infer import certified_export
    home,contract,row=qualified
    digest=dc.file_sha256(contract/'contract.json')
    identity={**supervised_boundaries[0]['data_contract'],'contract_sha256':digest}
    exported=certified_export(tmp_path,monkeypatch,supervised_boundaries,
        lambda s,f:[st.__setitem__('data_contract',identity) for st in (s,f)])
    path=home/'formal.pt';torch.save(exported,path)
    monkeypatch.setattr(ue,'validate_private_output',lambda p:dc.validate_private_output(p,private_root=home))
    reg={'gate':GATE_V1,'relation_implementation':'head','validation_selection_sha256':'a'*64,
         'score_thresholds':{'isles':.4,'pdgm':.5,'bmsr':.5},'frozen_before_evaluation':True}
    saved=[]
    def persist(record,index,total):
        saved.append((record['input_mode'],index,total))
        if index==2:raise RuntimeError('synthetic_private_failure')
    with pytest.raises(ValueError,match='Gate evaluation failed'):
        ue.run_gate_evaluation([row],digest,path,home/'gate',role_contract=contract,
            crops={'2d':(1,16,16),'3d':(4,16,16)},score_thresholds={'isles':.4},
            gate_registration=reg,expected_registration_sha256=dc.json_digest(reg),on_record=persist)
    assert [v[1] for v in saved]==[1,2]
    assert all(v[2]==3 for v in saved)
    assert not (home/'gate/complete.json').exists()


def controller():
    path=Path(__file__).resolve().parents[1]/'scripts/aur_development_selection.py'
    spec=importlib.util.spec_from_file_location('development_controller_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_controller_parser_and_bootstrap_never_echo_private_values(tmp_path,monkeypatch,capsys):
    mod=controller()
    with pytest.raises(SystemExit):mod.main(['--device','synthetic_private_patient'])
    assert 'synthetic_private' not in capsys.readouterr().err
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    out=tmp_path/'output'
    result=mod.run(tmp_path/'synthetic_private_missing.json',out)
    assert result['status']=='FAILED'
    assert 'synthetic_private' not in json.dumps(result)
    assert json.loads((out/'terminal.json').read_text())==result
    before=(out/'terminal.json').read_bytes()
    assert mod.run(tmp_path/'missing',out)['status']=='FAILED'
    assert (out/'terminal.json').read_bytes()==before


def test_controller_refuses_changed_source_input_protocol_and_output(tmp_path,monkeypatch):
    mod=controller();monkeypatch.setattr(Path,'home',lambda:tmp_path)
    f=tmp_path/'input.json';f.write_text('{}')
    from anatobind.aur.data_contract import file_sha256
    inputs={k:{'path':str(f),'sha256':file_sha256(f)} for k in mod.INPUTS}
    reg=mod.make_registration(inputs,tmp_path/'qualified',tmp_path/'output')
    mod.check_registration(reg,tmp_path/'output')
    for field in ('source','input','protocol','output'):
        bad=deepcopy(reg)
        if field=='source':bad['source_snapshot'][next(iter(bad['source_snapshot']))]='0'*64
        if field=='input':bad['inputs']['export']['sha256']='0'*64
        if field=='protocol':bad['policy']['relation_tie']='head'
        if field=='output':bad['output_root']=str(tmp_path/'other')
        with pytest.raises(ValueError):mod.check_registration(bad,tmp_path/'output')


def test_controller_watchdog_terminates_blocked_work():
    import time
    mod=controller()
    result=mod.helpers()._supervise(lambda send:time.sleep(10),total_seconds=.15,phase_limits={})
    assert result['status']=='TIMEOUT'


def test_real_controller_uses_certified_development_and_commits_actual_freeze(
        qualified,monkeypatch,tmp_path,supervised_boundaries):
    import torch
    from anatobind.aur import data_contract as dc
    from test_aur_unified_infer import certified_export
    home,contract,row=qualified
    mod=controller();monkeypatch.setattr(Path,'home',lambda:home)
    from anatobind.aur import unified_evaluation as ue
    monkeypatch.setattr(ue,'validate_private_output',lambda p:dc.validate_private_output(p,private_root=home))
    rows=[dict(row,source=s,sequence=seq,source_sequence=seq)
          for s,seq in [('pdgm','FLAIR'),('bmsr','T1c'),('isles','DWI')]]
    (contract/'validation.json').write_text(json.dumps(rows))
    parent=dc.read_json(contract/'contract.json')
    parent['outputs']['validation.json']=dc.file_sha256(contract/'validation.json')
    (contract/'contract.json').write_text(json.dumps(parent))
    complete=dc.read_json(contract/'complete.json')
    complete.update(contract_sha256=dc.file_sha256(contract/'contract.json'),outputs=parent['outputs'])
    (contract/'complete.json').write_text(json.dumps(complete))
    digest=dc.file_sha256(contract/'contract.json')
    identity={**supervised_boundaries[0]['data_contract'],'contract_sha256':digest}
    payload=certified_export(tmp_path,monkeypatch,supervised_boundaries,
        lambda s,f:[st.__setitem__('data_contract',identity) for st in (s,f)])
    export=home/'formal.pt';torch.save(payload,export)
    completion=home/'phase_completion.json';completion.write_text(json.dumps(
        dict(selected_export_sha256=dc.file_sha256(export))))
    ref=home/'reference';ref.mkdir()
    refs={r['source']:source_reference(r['source'],dc.json_digest(r)) for r in rows}
    (ref/'reference_anonymous.json').write_text(json.dumps(dict(version='aur_reference_1mm_v1',
        contract_sha256=digest,sources=refs)))
    (ref/'reference_private.json').write_text('{}')
    outputs={n:dc.file_sha256(ref/n) for n in ('reference_anonymous.json','reference_private.json')}
    (ref/'contract.json').write_text(json.dumps(dict(version='aur_reference_1mm_v1',
        outputs=outputs,data_contract_sha256=digest,parent_contract_sha256=None)))
    (ref/'complete.json').write_text(json.dumps(dict(version='aur_reference_1mm_v1',
        outputs=outputs,contract_sha256=dc.file_sha256(ref/'contract.json'))))
    paths=dict(contract=contract/'contract.json',export=export,completion=completion,
        reference_complete=ref/'complete.json',reference_contract=ref/'contract.json',
        reference_summary=ref/'reference_anonymous.json',reference_private=ref/'reference_private.json')
    inputs={k:dict(path=str(v),sha256=dc.file_sha256(v)) for k,v in paths.items()}
    out=home/'evaluation';reg=mod.make_registration(inputs,home/'unused_archive',out,
        crops={'2d':[1,16,16],'3d':[4,16,16]})
    registration=home/'registration.json';registration.write_text(json.dumps(reg));out.mkdir()
    if payload['stage']=='II':
        with pytest.raises(ValueError,match='Stage III'):mod._execute(registration,out,'cpu',lambda m:None)
        assert not (out/'selection_plan_private.json').exists()
        return
    result=mod._execute(registration,out,'cpu',lambda m:None)
    assert result['status']=='COMPLETE' and result['final_test_evaluated'] is False
    if result['deployment_frozen']:
        final=dc.read_json(out/'deployment_registration.json')
        assert final['validation_selection_sha256']==dc.file_sha256(out/'development_selection.json')
        diagnostic=dc.read_json(out/'diagnostic_registration.json')
        assert final['score_thresholds']==diagnostic['score_thresholds']
        assert final['validation_selection_sha256']!=diagnostic['validation_selection_sha256']
        records=dc.read_json(out/'development/records_private.json')
        assert all(r['registration_sha256']==dc.json_digest(diagnostic) for r in records)
    else:
        assert not (out/'deployment_registration.json').exists()
    assert (out/'complete.json').is_file()
    assert len(list((out/'selection_records').glob('*.json')))==3
    assert len(list((out/'development_records').glob('*.json')))==9
    assert 'synthetic_private' not in (out/'progress.json').read_text()
