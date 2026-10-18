"""Score-set feasibility must respect ties, official rematching and frozen archives."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pytest

from anatobind.aur import eval as E


def module():
    spec=importlib.util.spec_from_file_location('aur_order_audit',Path(__file__).resolve().parents[1]/'scripts/aur_audit_score_order.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def detection(k,score,box=None):
    return dict(instance=k,score=score,box=box or [10*k,0,0,10*k+1,1,1])


def test_equal_score_tp_and_fp_are_inseparable_at_zero_fp_budget():
    m=module();dets=[detection(1,.8),detection(2,.8)]
    result=m.score_order_bound([E.scan('anonymous',[dict(box=dets[0]['box'])],dets)],0.)
    assert result['max_observed_hits_within_budget']==0
    assert result['n_unique_scores']==1 and result['n_operating_sets']==2
    assert result['n_feasible_sets']==1
    assert not any('threshold' in k or k=='thr' for k in result)


@pytest.mark.parametrize('score',[0.,1.])
def test_inclusive_score_endpoints_can_retain_a_true_detection(score):
    m=module();d=detection(1,score)
    r=m.score_order_bound([E.scan('anonymous',[dict(box=d['box'])],[d])],0.)
    assert r['max_observed_hits_within_budget']==1 and r['optimistic_sensitivity']==1.
    assert r['n_operating_sets']==2 and r['n_feasible_sets']==2


def test_empty_inputs_include_the_empty_set_and_distinguish_no_truths():
    m=module();r=m.score_order_bound([],0.)
    assert r['n_operating_sets']==r['n_feasible_sets']==1 and r['n_scans']==0
    assert r['max_observed_hits_within_budget']==0 and r['optimistic_sensitivity'] is None
    r=m.score_order_bound([E.scan('anonymous',[dict(box=[0,0,0,1,1,1])],[])],0.)
    assert r['n_gt']==1 and r['optimistic_sensitivity']==0.


def test_real_first_then_ignored_matching_excuses_only_remaining_detections():
    m=module();box=[0,0,0,1,1,1]
    scans=[E.scan('anonymous',[dict(box=box),dict(box=box,ignore=True)],
                 [detection(1,.9,box),detection(2,.8,box),detection(3,.7)])]
    r=m.score_order_bound(scans,0.)
    assert r['max_observed_hits_within_budget']==1 and r['n_gt']==r['n_ignored_gt']==1
    assert r['n_feasible_sets']==3 and r['n_operating_sets']==4


def test_all_cuts_are_rematched_even_when_hungarian_hit_count_decreases():
    m=module();box=lambda a,b:[a,0,0,b,1,1]
    gt=[dict(box=box(0,21)),dict(box=box(89,100))]
    dets=[detection(1,.9,box(0,2.31)),detection(2,.8,box(0,100)),detection(3,.7,box(0,200))]
    scan=E.scan('anonymous',gt,dets)
    assert len(E.scan_matches(scan,.8)[0])==2 and len(E.scan_matches(scan,.7)[0])==1
    before=deepcopy(scan);r=m.score_order_bound([scan],0.)
    assert r['max_observed_hits_within_budget']==2 and r['n_operating_sets']==4
    assert r['n_feasible_sets']==3 and scan==before


def test_strict_monotone_final_score_map_preserves_operating_set_ceiling():
    m=module();dets=[detection(1,.9),detection(2,.8),detection(3,.2)]
    scans=[E.scan('anonymous',[dict(box=dets[0]['box']),dict(box=dets[2]['box'])],dets)]
    transformed=deepcopy(scans)
    for d in transformed[0]['dets']:d['score']=d['score']**2
    assert m.score_order_bound(scans,1.)==m.score_order_bound(transformed,1.)


def test_nondecreasing_map_can_remove_feasible_operating_sets_by_merging_ties():
    m=module();dets=[detection(1,.9),detection(2,.8)]
    scans=[E.scan('anonymous',[dict(box=dets[0]['box'])],dets)]
    tied=deepcopy(scans)
    for d in tied[0]['dets']:d['score']=.5
    assert m.score_order_bound(scans,0.)['max_observed_hits_within_budget']==1
    assert m.score_order_bound(tied,0.)['max_observed_hits_within_budget']==0


@pytest.mark.parametrize('value',[-.1,1.1,float('nan'),float('inf')])
def test_invalid_scores_and_budgets_are_rejected(value):
    m=module()
    with pytest.raises(ValueError):m.score_order_bound([E.scan('anonymous',[],[detection(1,value)])],0.)
    if value!=1.1:
        with pytest.raises(ValueError):m.score_order_bound([],value)


def frozen_fixture(tmp_path,monkeypatch,native_metadata=False):
    from test_aur_confidence_replay import setup_source,install_fake_forward
    old,a,vol,p,b=setup_source(tmp_path,monkeypatch)
    reference={s:dict(n_scans=2,n_gt=2,n_hit=2,n_fp=n,fp_per_scan=n/2,sensitivity=1.,thr=.5)
               for s,n in [('bmsr',1),('isles',3),('pdgm',0)]}
    public_metrics=deepcopy(reference)
    if native_metadata:
        for s,r in reference.items():r.update(source=s,disease='annotation-source metadata')
    write=lambda p,v:p.write_text(json.dumps(v))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    write(a.reference_six/'reference_frozen.json',reference)
    contract=json.loads((a.reference_six/'contract.json').read_text());contract['reference_sha256']=sha(a.reference_six/'reference_frozen.json')
    write(a.reference_six/'contract.json',contract)
    summary=json.loads(old.REFERENCE_SUMMARY.read_text());summary['provenance'][0]['contract_sha256']=sha(a.reference_six/'contract.json')
    summary['source_sequence']={s+'/'+q:dict(S7=public_metrics[s],S7_comparable_sequence=True)
                               for s,q in [('bmsr','T1c'),('isles','DWI'),('pdgm','FLAIR')]}
    write(old.REFERENCE_SUMMARY,summary)
    install_fake_forward(old,monkeypatch,vol,p,b);old.run(a)
    measured=json.loads((a.out/'contract.json').read_text());complete=json.loads((a.out/'COMPLETE.json').read_text())
    aggregate=json.loads((a.out/'aggregate.json').read_text())
    anchor=tmp_path/'confidence_public.json'
    write(anchor,dict(code_sha=measured['code_sha'],private_contract_sha256=sha(a.out/'contract.json'),
        private_complete_sha256=sha(a.out/'COMPLETE.json'),private_records_content_sha256=complete['records_sha256'],
        private_input_manifest_sha256=sha(a.out/'input_manifest_private.json'),
        artifact_hashes={'aggregate_anonymous.json':sha(a.out/'aggregate.json')},
        observed=dict(controls=aggregate['controls'],measured_children=6,active_parents=6)))
    m=module()
    load=lambda:m.load_archive(a.out,anchor_path=anchor,query_reference=a.reference_six,query_summary=old.REFERENCE_SUMMARY)
    return m,a,anchor,old,load


def test_native_reference_metadata_does_not_change_its_published_metric_projection(tmp_path,monkeypatch):
    m,_,_,_,load=frozen_fixture(tmp_path,monkeypatch,native_metadata=True)
    prepared=load()
    assert prepared['reference']['bmsr']['source']=='bmsr'
    assert {s:r['fp_per_scan'] for s,r in prepared['reference'].items()}=={'bmsr':.5,'isles':1.5,'pdgm':0.}
    assert m.analyze(prepared)['n_rows']==6


def test_frozen_archive_reproduces_old_points_and_outputs_only_anonymous_bounds(tmp_path,monkeypatch):
    m,a,_,_,load=frozen_fixture(tmp_path,monkeypatch);prepared=load()
    before={p.name:p.read_bytes() for p in (a.out/'row_records').glob('*.json')}
    result=m.analyze(prepared)
    assert result['n_rows']==6 and result['n_operating_sets_total']==18
    for source in result['source_sequence'].values():
        assert all(x['max_observed_hits_within_budget']==2 for x in source['score_order_bounds'].values())
    assert 'PRIVATE_' not in json.dumps(result)
    assert before=={p.name:p.read_bytes() for p in (a.out/'row_records').glob('*.json')}
    monkeypatch.setattr(m,'load_archive',lambda run:prepared)
    args=SimpleNamespace(run=a.out,out=tmp_path/'result',max_seconds=60)
    assert m.run(args)==0
    complete=(args.out/'COMPLETE.json').read_bytes()
    with pytest.raises(FileExistsError):m.run(args)
    assert (args.out/'COMPLETE.json').read_bytes()==complete
    def keys(x):
        if isinstance(x,dict):
            for k,v in x.items():yield k;yield from keys(v)
        elif isinstance(x,list):
            for v in x:yield from keys(v)
    assert not {'thr','threshold','selected_threshold','patient','case','instance','box','window'} & set(keys(result))


@pytest.mark.parametrize('fault',['record','membership','reference','semantic_control','fixed_point'])
def test_changed_or_semantically_inconsistent_archive_stops_before_analysis(tmp_path,monkeypatch,fault):
    m,a,anchor,old,load=frozen_fixture(tmp_path,monkeypatch)
    read=lambda p:json.loads(p.read_text())
    write=lambda p,v:p.write_text(json.dumps(v))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    record=next((a.out/'row_records').glob('*.json'))
    if fault=='record':
        v=read(record);v['payload']['measurement']['children'][0]['child_raw_mean']=.1;write(record,v)
    elif fault=='membership':record.unlink()
    elif fault=='reference':
        p=a.reference_six/'reference_frozen.json';v=read(p);v['bmsr']['fp_per_scan']=.6;write(p,v)
    elif fault=='semantic_control':
        v=read(record);v['payload']['controls']['returned_maps_exact']=False;v['payload_sha256']=old.digest(v['payload']);write(record,v)
        rows=read(a.out/'selection_private.json')
        corpus=[read(a.out/'row_records'/(old.token(r)+'.json')) for r in rows]
        complete=read(a.out/'COMPLETE.json');complete['records_sha256']=old.digest(corpus);write(a.out/'COMPLETE.json',complete)
        public=read(anchor);public['private_complete_sha256']=sha(a.out/'COMPLETE.json');public['private_records_content_sha256']=complete['records_sha256'];write(anchor,public)
    else:
        p=a.out/'aggregate.json';v=read(p);v['source_sequence']['bmsr/T1c']['hypothetical_score_filters_after_frozen_ownership']['child_raw_mean']['n_hit']=0;write(p,v)
        complete=read(a.out/'COMPLETE.json');complete['aggregate_sha256']=sha(p);write(a.out/'COMPLETE.json',complete)
        public=read(anchor);public['private_complete_sha256']=sha(a.out/'COMPLETE.json');public['artifact_hashes']['aggregate_anonymous.json']=sha(p);write(anchor,public)
    with pytest.raises(ValueError):load()
