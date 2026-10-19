"""Frozen-source integrity, real measurement controls and anonymous fixed-score analysis."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anatobind.aur import eval as E, infer as I
from anatobind.aur.diagnostics import identity, select_cases
from anatobind.aur.labels import entity_to_synthseg
from anatobind.aur.query_decoder import QueryCCDecoder, decoder_rules
from anatobind.aur.targets import host_targets


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/aur_diagnose_confidence.py'
    spec = importlib.util.spec_from_file_location('aur_confidence_runner', path)
    m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def record_digest(records):
    return hashlib.sha256(''.join(sorted(hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
                                        for r in records)).encode()).hexdigest()


def setup_source(tmp_path, monkeypatch):
    m = module();root = tmp_path / 'data';root.mkdir();source = tmp_path / 'reference';source.mkdir()
    # These toy records freeze this checkout's source snapshot, not the historical
    # medical run. Keep the real dependency checks active against an isolated root.
    source_root = tmp_path / 'source'
    dependencies = json.loads(m.SOURCE_HASHES.read_text())['files']
    for name in {*dependencies, 'scripts/aur_diagnose_confidence.py',
                 'anatobind/aur/confidence_diagnostics.py'}:
        path = source_root / name;path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((m.ROOT / name).read_bytes())
    source_hashes = source_root / 'synthetic_source_hashes.json'
    source_hashes.write_text(json.dumps({'code_sha':m.REFERENCE_CODE_SHA,
        'files':{name:sha(source_root / name) for name in dependencies}}))
    monkeypatch.setattr(m, 'ROOT', source_root)
    monkeypatch.setattr(m, 'SOURCE_HASHES', source_hashes)
    (source / 'row_records').mkdir();saved = tmp_path / 'saved';saved.mkdir()
    checkpoint = root / 'model.pt';checkpoint.write_bytes(b'fixture export, no model loading')
    p = np.full((1, 1, 1, 3), .8, np.float32);inst = np.ones((1, 1, 3), np.int32)
    entity = np.zeros(inst.shape, np.uint8);seg = entity_to_synthseg(entity)
    a = E.host_dice(seg, seg);gt = E.instance_rows(inst, 10.)
    with_decoder = QueryCCDecoder(inst.shape, 10., workers=1)
    with_decoder(window=[(0, 1), (0, 1), (0, 3)], event_prob=p, event_presence=np.ones(1, np.float32),
                 valid=np.ones(inst.shape, np.float32), entity_prob=None, weight=None)
    _, dets, stats = with_decoder.finish()
    legacy = I.instances_from_probability(p[0], 10.)[1]
    probs = np.zeros(14, np.float32);probs[-1] = 1
    binding = dict(instance=1, host=13, host_probs=probs, query=0, query_iou=1., zero_overlap=False,
                   window=[(0, 1), (0, 1), (0, 3)], in_crop_share=1.)
    old_r = dict(instance=1, truth=13, r=13, b0=13, r_probs=probs.tolist(), query_iou=1.,
                 zero_overlap=False, in_crop_share=1., local_dice=float('nan'),
                 volume_mm3=gt[0]['volume_mm3'], stratum=gt[0]['stratum'], box=gt[0]['box'],
                 binding_origin={'query':0, 'window':[[0,1],[0,1],[0,3]], 'in_crop_share':1.})
    rows, results = [], []
    for src, seq in [('bmsr', 'T1c'), ('isles', 'DWI'), ('pdgm', 'FLAIR')]:
        for n in (1, 2):
            f = root / (src + str(n) + '.bin');f.write_bytes(bytes([n]))
            row = dict(source=src, sequence=seq, patient=src + 'PRIVATE_PATIENT' + str(n),
                       case=src + 'PRIVATE_CASE' + str(n), image=str(f), anatomy=str(f), lesion=str(f),
                       split='train', u_supervised=True, r_supervised=True)
            rows.append(row)
            results.append({**{k:row[k] for k in ('source','sequence','patient','case')}, 'a':a,
                'n_windows':1, 'seq_probs':[1.,0.,0.,0.,0.,0.],
                'u':{'gt':gt,'dets':dets}, 'r':{'instances':[old_r]},
                'legacy_reference_u':{'dets':legacy}, 'decoder_stats':stats})
    samples = root / 'samples.json';samples.write_text(json.dumps(rows))
    val = root / 'val.json';val.write_text(json.dumps({s:[r['patient'] for r in rows if r['source']==s]
                                                    for s in ('bmsr','isles','pdgm')}))
    (saved / 'cases.json').write_text(json.dumps(results))
    selected = select_cases(rows, results);(source / 'selection_private.json').write_text(json.dumps(selected))
    records=[]
    by_id={identity(r):r for r in results}
    manifest={}
    for row in selected:
        token=hashlib.sha256(json.dumps(identity(row),separators=(',',':')).encode()).hexdigest()
        h={k:sha(row[k]) for k in ('image','anatomy','lesion')};manifest[token]=h
        rec={'result':by_id[identity(row)],'input_sha256':h,'code_sha':m.REFERENCE_CODE_SHA,
             'seconds':1.,'peak_cuda_gib':0.}
        serialized=json.dumps(rec)
        records.append(json.loads(serialized));(source/'row_records'/(token+'.json')).write_text(serialized)
    (source/'input_manifest_private.json').write_text(json.dumps(manifest))
    (source/'reference_frozen.json').write_text('{}')
    contract={'code_sha':m.REFERENCE_CODE_SHA,'checkpoint_sha256':sha(checkpoint),'samples_sha256':sha(samples),
              'val_patients_sha256':sha(val),'saved_cases_sha256':sha(saved/'cases.json'),
              'selection_sha256':hashlib.sha256(json.dumps([identity(r) for r in selected],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
              'raw_inputs_sha256':canonical(manifest),'reference_sha256':sha(source/'reference_frozen.json'),
              'decoder_profile':'query_cc_v1','decoder_rules':decoder_rules('query_cc_v1'),
              'score_thresholds':m.SCORES,'crop':[128,160,160],'batch_size':2,'mode':'six','n_rows':6}
    (source/'contract.json').write_text(json.dumps(contract))
    (source/'COMPLETE.json').write_text(json.dumps({'status':'complete','rows':6,'all_reproduction_checks_passed':True}))
    summary=root/'public_reference.json';summary.write_text(json.dumps({'rows':6,'frozen_code_sha':m.REFERENCE_CODE_SHA,
        'invariance':{'controlled_instances':6},'inference_resources':{'windows':6},
        'provenance':[{'contract_sha256':sha(source/'contract.json'),'complete_sha256':sha(source/'COMPLETE.json'),
                       'row_records_content_sha256':record_digest(records)}]}))
    monkeypatch.setattr(m,'REFERENCE_SUMMARY',summary);monkeypatch.setattr(m,'DATA_ROOT',root)
    monkeypatch.setattr(m,'CKPT_SHA',sha(checkpoint));monkeypatch.setattr(m,'SAMPLES_SHA',sha(samples))
    args=SimpleNamespace(reference_six=source,samples=samples,val_patients=val,checkpoint=checkpoint,
                         saved_eval=saved,out=tmp_path/'out',gpu=0,max_seconds=60,resume=False)
    vol={'image':np.zeros(inst.shape,np.float32),'entity':entity,'instance':inst,'small':np.zeros(inst.shape,bool),
         'affine':np.diag([10.,1.,1.,1.]),'spacing':(1.,1.,10.),'voxel_mm3':10.,
         'hosts':host_targets(inst,seg,(1.,1.,10.)), 'r_supervised':True,'u_supervised':True,
         'a_supervised':True, 'a_ignore':np.zeros(inst.shape,bool), 'seq':0, 'case':'fixture'}
    return m,args,vol,p,binding


def install_fake_forward(m, monkeypatch, vol, p, binding):
    def predict(model,image,affine,crop,device,batch_size=1,observer=None,**kw):
        assert tuple(crop)==(128,160,160) and batch_size==2
        observer(window=[(0,1),(0,1),(0,3)],event_prob=torch.from_numpy(p),event_presence=torch.ones(1),
                 valid=torch.ones(vol['image'].shape),entity_prob=None,weight=None)
        return {'entity':vol['entity'].copy(),'entity_max_prob':np.zeros(vol['image'].shape,np.float16),
                'lesion_prob':p[0].copy(),'seq_probs':np.array([1.,0.,0.,0.,0.,0.],np.float32),
                'entity_presence':np.ones(32,np.float32),'n_windows':1}
    monkeypatch.setattr(m,'load_volume',lambda row:deepcopy(vol))
    monkeypatch.setattr(m.I,'predict_volume',predict)
    monkeypatch.setattr(m.I,'bind_instances',lambda *a,**kw:[deepcopy(binding)])
    monkeypatch.setattr(m.I,'model_from_export',lambda *a:(object(),{}))
    original=m.replay_case
    monkeypatch.setattr(m,'replay_case',lambda model,row,ref,crop,device:original(model,row,ref,crop,torch.device('cpu')))
    return predict,original


def test_rescoring_keeps_geometry_and_uses_matching_after_fixed_score_filter():
    m=module()
    gt=[{'instance':1,'box':[0,0,0,2,2,2]}, {'instance':None,'box':[10,0,0,12,2,2],'ignore':True}]
    dets=[{'instance':1,'box':gt[0]['box'],'score':.6},
          {'instance':2,'box':[0,0,0,2,2,3],'score':.8},
          {'instance':3,'box':gt[1]['box'],'score':.8},
          {'instance':4,'box':[20,0,0,22,2,2],'score':.8}]
    children=[{'instance':k,'parent_score':d['score'],'child_raw_mean':s,'child_fused_mean':.5}
              for k,d,s in zip((1,2,3,4),dets,(.9,.6,.9,.6))]
    original=deepcopy(dets)
    assert m.classify_original(gt,dets,.7)=={1:'below_original_threshold',2:'TP',3:'ignored_excused',4:'FP'}
    new=m.score_copies(dets,children,'child_raw_mean')
    hits,fp,_=E.scan_matches(E.scan('anonymous',gt,new),.7)
    assert len(hits)==1 and fp==0 and dets==original
    assert [r['box'] for r in new]==[r['box'] for r in dets]
    with pytest.raises(ValueError):m.score_copies(dets,children[:-1],'child_raw_mean')


def test_payload_integrity_digest_survives_JSON_key_normalization():
    m=module();payload={'queries':{2:'second',10:'tenth'},'A':{5:.1,44:float('nan')}}
    assert m.digest(payload)==m.digest(json.loads(json.dumps(payload)))


@pytest.mark.parametrize('target', ['raw','reference','selection','record'])
def test_changed_original_frozen_inputs_stop_before_model_loading(tmp_path,monkeypatch,target):
    m,a,_,_,_=setup_source(tmp_path,monkeypatch)
    if target=='raw':next((tmp_path/'data').glob('bmsr*.bin')).write_bytes(b'replaced unfinished input')
    elif target=='reference':(a.reference_six/'reference_frozen.json').write_text('{"changed":1}')
    elif target=='selection':(a.reference_six/'selection_private.json').write_text('[]')
    else:
        f=next((a.reference_six/'row_records').glob('*.json'));r=json.loads(f.read_text());r['seconds']=2;f.write_text(json.dumps(r))
    with pytest.raises(ValueError):m.prepare_reference(a)
    assert not a.out.exists()


def test_complete_synthetic_replay_is_anonymous_and_resumes_without_inference(tmp_path,monkeypatch):
    m,a,vol,p,b=setup_source(tmp_path,monkeypatch);install_fake_forward(m,monkeypatch,vol,p,b)
    assert m.run(a)==0
    aggregate=json.loads((a.out/'aggregate.json').read_text())
    assert aggregate['n_rows']==6 and aggregate['controls']['controlled_R_instances']==6
    assert aggregate['controls']['first_windows']==aggregate['controls']['replay_windows']==6
    assert 'PRIVATE_' not in json.dumps(aggregate)
    before=(a.out/'COMPLETE.json').read_bytes()
    def unexpected(*args):raise AssertionError('Completed resume must not load a model')
    monkeypatch.setattr(m.I,'model_from_export',unexpected);a.resume=True
    assert m.run(a)==0 and (a.out/'COMPLETE.json').read_bytes()==before


@pytest.mark.parametrize('fault', ['dependency', 'manifest_revision'])
def test_synthetic_source_snapshot_still_rejects_dependency_drift(tmp_path,monkeypatch,fault):
    m,a,_,_,_=setup_source(tmp_path,monkeypatch)
    m.prepare_reference(a)
    if fault=='dependency':
        path=m.ROOT/'anatobind/aur/model.py';path.write_bytes(path.read_bytes()+b'\n# changed\n')
        message='A frozen inference/model dependency changed'
    else:
        manifest=json.loads(m.SOURCE_HASHES.read_text());manifest['code_sha']='different'
        m.SOURCE_HASHES.write_text(json.dumps(manifest))
        message='Frozen dependency manifest refers to a different source revision'
    with pytest.raises(ValueError,match=message):m.prepare_reference(a)
    assert not a.out.exists()


def test_default_historical_source_manifest_rejects_changed_dependency(tmp_path,monkeypatch):
    default=module();manifest_before=default.SOURCE_HASHES.read_bytes()
    m,a,_,_,_=setup_source(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'SOURCE_HASHES',default.SOURCE_HASHES)
    # Mutate the temporary source copy only, retaining the real default manifest.
    name=next(iter(json.loads(manifest_before)['files']))
    (m.ROOT/name).write_bytes(b'changed historical inference/model dependency\n')
    with pytest.raises(ValueError,match='A frozen inference/model dependency changed'):
        m.prepare_reference(a)
    assert default.SOURCE_HASHES.read_bytes()==manifest_before and not a.out.exists()


def test_resume_rejects_changed_code_and_corrupt_cached_measurements(tmp_path,monkeypatch):
    m,a,vol,p,b=setup_source(tmp_path,monkeypatch);install_fake_forward(m,monkeypatch,vol,p,b);m.run(a);a.resume=True
    original=m.code_sha;monkeypatch.setattr(m,'code_sha',lambda:'different')
    with pytest.raises(ValueError):m.run(a)
    monkeypatch.setattr(m,'code_sha',original)
    f=next((a.out/'row_records').glob('*.json'));r=json.loads(f.read_text());r['payload']['measurement']['children'][0]['child_raw_mean']=.1;f.write_text(json.dumps(r))
    with pytest.raises(ValueError):m.run(a)


@pytest.mark.parametrize('fault', ['R_origin', 'replay_flag'])
def test_interrupted_resume_revalidates_control_values_not_only_cache_checksum(tmp_path,monkeypatch,fault):
    m,a,vol,p,b=setup_source(tmp_path,monkeypatch);install_fake_forward(m,monkeypatch,vol,p,b);m.run(a);a.resume=True
    (a.out/'COMPLETE.json').unlink()  # interrupt after aggregate, before completion marker
    f=next((a.out/'row_records').glob('*.json'));r=json.loads(f.read_text())
    if fault=='R_origin':r['payload']['observed_controlled_R'][0]['binding_origin']['query']=99
    else:r['payload']['measurement']['checks']['raw_probability_planes_exact']=False
    r['payload_sha256']=m.digest(r['payload']);f.write_text(json.dumps(r))
    with pytest.raises(ValueError):m.run(a)


@pytest.mark.parametrize('fault', ['missing_parent', 'missing_windows', 'window_digest', 'query_count',
    'query_membership', 'parent_rank', 'child_count', 'voxel_accounting', 'nonfinite_mean', 'retained_mean'])
def test_resume_rejects_internally_incomplete_measurements(tmp_path,monkeypatch,fault):
    m,a,vol,p,b=setup_source(tmp_path,monkeypatch);install_fake_forward(m,monkeypatch,vol,p,b);m.run(a)
    prepared=m.prepare_reference(a);row=prepared['rows'][0]
    record=json.loads((a.out/'row_records'/(m.token(row)+'.json')).read_text())
    measurement=record['payload']['measurement'];parent=measurement['parents'][0]
    if fault=='missing_parent':measurement['parents']=[]
    elif fault=='missing_windows':measurement['window_hashes']=[]
    elif fault=='window_digest':measurement['checks']['window_hashes_sha256']='0'*64
    elif fault=='query_count':measurement['checks']['present_query_planes']+=1
    elif fault=='query_membership':
        measurement['window_hashes'][0]['queries']={'99':'0'*64}
        measurement['checks']['window_hashes_sha256']=hashlib.sha256(
            json.dumps(measurement['window_hashes'],sort_keys=True).encode()).hexdigest()
    elif fault=='parent_rank':parent['rank']+=1
    elif fault=='child_count':parent['final_children']+=1
    elif fault=='voxel_accounting':parent['removed_voxels']+=1
    elif fault=='nonfinite_mean':parent['retained_raw_mean']=float('nan')
    else:parent['retained_raw_mean']-=.1
    record['payload_sha256']=m.digest(record['payload'])
    with pytest.raises(ValueError):m.validate_cached(record,row,prepared)


def test_aggregate_preserves_outcomes_with_physical_size_and_retention_score_summaries():
    m=module();dets=[];children=[]
    for k,(volume,fraction,raw,fused) in enumerate([(10.,.1,.4,.3),(100.,.5,.5,.4),(1000.,1.,.8,.7)],1):
        dets.append(dict(instance=k,box=[k*10,0,0,k*10+1,1,1],score=.9,volume_mm3=volume,
                         stratum=E.size_stratum(volume)))
        children.append(dict(instance=k,parent_score=.9,child_raw_mean=raw,child_fused_mean=fused,
                             retained_fraction=fraction))
    record=dict(source='bmsr',sequence='T1c',gt=[],detections=dets,
        measurement=dict(children=children,parents=[],checks=dict(first_windows=1,replay_windows=1,
            present_query_planes=1,raw_probability_planes_exact=True)),
        controls=dict(A_saved_scores_exact=True,controlled_R_instances=0,
            controlled_R_outputs_and_origins_exact=True,returned_maps_exact=True),seconds=0.,peak_cuda_gib=0.)
    original=deepcopy(record)
    summary=m.aggregate_records([record])['source_sequence']['bmsr/T1c']['original_point_groups']['FP']
    for key,gap in zip(E.STRATA,(.5,.4,.1)):
        group=summary['physical_size_strata'][key]
        assert group['n_children']==1 and group['parent_minus_raw']['mean']==pytest.approx(gap)
    for key,gap in zip(('<=0.1','(0.1,0.5]','>0.5'),(.6,.5,.2)):
        group=summary['retained_fraction_score_summaries'][key]
        assert group['n_children']==1 and group['parent_minus_fused']['mean']==pytest.approx(gap)
    assert record==original


def test_controlled_R_origin_change_fails_even_if_host_probabilities_match(tmp_path,monkeypatch):
    m,a,vol,p,b=setup_source(tmp_path,monkeypatch);_,replay=install_fake_forward(m,monkeypatch,vol,p,b)
    refs=m.prepare_reference(a);row=refs['rows'][0];ref=refs['records'][identity(row)]
    wrong=deepcopy(b);wrong['query']=1;monkeypatch.setattr(m.I,'bind_instances',lambda *a,**kw:[wrong])
    with pytest.raises(ValueError):replay(object(),row,ref,(128,160,160),torch.device('cpu'))


def test_returned_legacy_map_drift_is_detected_even_when_raw_query_hashes_match(tmp_path,monkeypatch):
    m,a,vol,p,b=setup_source(tmp_path,monkeypatch);predict,replay=install_fake_forward(m,monkeypatch,vol,p,b)
    refs=m.prepare_reference(a);row=refs['rows'][0];ref=refs['records'][identity(row)];n=[0]
    def drift(*args,**kw):
        out=predict(*args,**kw);n[0]+=1
        if n[0]==2:out['lesion_prob'][0,0,0]-=.001
        return out
    monkeypatch.setattr(m.I,'predict_volume',drift)
    with pytest.raises(ValueError):replay(object(),row,ref,(128,160,160),torch.device('cpu'))
