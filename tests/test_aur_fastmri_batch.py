"""Native-grid batch outputs and byte-bound immutable restart behavior."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_aur_dicom import make_mr
from test_aur_unified_infer import tiny_model


def api():return importlib.import_module('anatobind.aur.fastmri_batch')


@pytest.fixture
def batch(tmp_path,monkeypatch):
    fb=api();root=tmp_path/'synthetic_private_sources';root.mkdir()
    for name,kind in [('file_brain_AXFLAIR_200_000001','good'),('file_brain_AXT2_200_000002','empty'),
                      ('file_brain_AXT1_200_000003','bad')]:
        folder=root/name;folder.mkdir()
        if kind=='empty':continue
        for i in range(3):
            rows=8 if kind=='bad' and i==1 else 16
            make_mr(folder,i,Rows=rows,Columns=16,PixelSpacing=[1,1],SpacingBetweenSlices=1,
                pixels=np.arange(rows*16).reshape(rows,16),ImagePositionPatient=None,ImageOrientationPatient=None)
    model=tiny_model();export=tmp_path/'pilot.pt'
    torch.save({'format_version':1,'stage':'III','model_config':model.cfg,
        'model_state_dict':model.state_dict(),'data_contract':{'kind':'synthetic_test'}},export)
    monkeypatch.setattr(fb,'validate_private_output',lambda p:Path(p).resolve())
    return fb,root,export,tmp_path/'output'


def test_screen_side_and_area_vs_volume_records():
    fb=api()
    from anatobind.aur.labels import HOST_SIDE,NO_HOST
    assert fb.screen_side(HOST_SIDE.index('left'))=='screen_right'
    assert fb.screen_side(HOST_SIDE.index('right'))=='screen_left'
    assert fb.screen_side(HOST_SIDE.index(None)) is None and fb.screen_side(NO_HOST) is None
    lesion={'instance':1,'box':[0,1,2,1,3,4],'host':0,'host_probs':[1]+[0]*13,'score':.8,'area_mm2':4}
    record=fb.lesion_record(lesion,'2d',7)
    assert record['slice']==7 and 'area_mm2' in record and 'volume_mm3' not in record
    assert 'patient_side' not in record


def test_one_mm_roundtrip_and_anisotropic_sampling():
    fb=api();image=np.arange(3*16*16,dtype=np.float32).reshape(3,16,16);aff=np.diag([-1.,-1.,1.,1.])
    out,aff2=fb.resample_to_1mm(image,aff,'3d')
    np.testing.assert_array_equal(out,image);np.testing.assert_allclose(aff2,aff)
    lab=np.zeros(image.shape,np.int32);lab[1,4:8,4:8]=5
    np.testing.assert_array_equal(fb.map_back(lab,aff2,image.shape,aff),lab)
    out,aff2=fb.resample_to_1mm(image,np.diag([-2.,-3.,5.,1.]),'3d')
    assert out.shape==(15,48,32) and np.allclose(np.linalg.norm(aff2[:3,:3],axis=0),1)
    plane,a=fb.resample_to_1mm(image[:1],np.diag([-2.,-3.,1.,1.]),'2d')
    assert plane.shape==(1,48,32)


def test_actual_tiny_batch_maps_both_modes_and_skips_whole_bad_series(batch):
    fb,root,export,out=batch
    result=fb.run_batch(root,export,out,crops={'2d':(1,16,16),'3d':(2,16,16)})
    assert result['n_series']==3 and result['n_ok']==1
    assert result['skipped']=={'empty_input':1,'geometry_unsupported':1}
    assert result['by_sequence']=={'AXFLAIR':1} and result['geometry']=='assumed'
    assert result['patient_coordinates']=='not_produced' and result['pilot_only']
    assert result['anatomy_presence']['2d']['n_observations']==3
    assert result['anatomy_presence']['3d']['n_observations']==1
    assert len(result['anatomy_presence']['2d']['present_counts'])==32
    record=json.loads((out/'series_00000/record_private.json').read_text())
    assert set(record['modes'])=={'2d','3d'} and record['native_shape']==[3,16,16]
    with np.load(out/'series_00000/arrays.npz') as data:
        assert data['anatomy_3d'].shape==data['anatomy_2d'].shape==(3,16,16)
    assert 'synthetic_private' not in json.dumps(result) and 'file_brain' not in json.dumps(result)


def fake_series(model,folder,**kwargs):
    source=api().read_pixel_grid_series(sorted(folder.glob('*.dcm')))
    return {'sequence':api().sequence_of(folder),'modes':list(kwargs['modes']),'native_shape':list(source.image_dhw.shape),
        'geometry':'assumed','patient_coordinates':'not_produced','lesions':[]}, {'anatomy_3d':np.zeros(source.image_dhw.shape,np.int16)}


@pytest.mark.parametrize('damage',['none','source','inventory','export','crops','threshold','mode','array','record','unbound_output',
                                  'root_empty_commitment','root_missing_case','root_version'])
def test_resume_requires_all_original_input_and_output_commitments(batch,monkeypatch,damage):
    fb,root,export,out=batch
    monkeypatch.setattr(fb,'run_series',fake_series)
    kwargs={'crops':{'2d':(1,16,16),'3d':(2,16,16)}}
    fb.run_batch(root,export,out,**kwargs)
    monkeypatch.setattr(fb,'run_series',lambda *a,**k:pytest.fail('committed series was recomputed'))
    if damage=='source':
        p=next(root.glob('*/*.dcm'));p.write_bytes(p.read_bytes()+b'changed')
    elif damage=='inventory':(root/'file_brain_AXT2_200_000004').mkdir()
    elif damage=='export':export.write_bytes(export.read_bytes()+b'changed')
    elif damage=='crops':kwargs['crops']['3d']=(4,16,16)
    elif damage=='threshold':kwargs['score_threshold']=.6
    elif damage=='mode':kwargs['modes']=('3d',)
    elif damage=='array':(out/'series_00000/arrays.npz').write_bytes(b'changed')
    elif damage=='record':(out/'series_00000/record_private.json').write_text('{}')
    elif damage=='unbound_output':
        (out/'complete.json').unlink()
        path=out/'series_00000/complete.json';value=json.loads(path.read_text());value['outputs']={}
        path.write_text(json.dumps(value))
    elif damage.startswith('root_'):
        path=out/'complete.json';value=json.loads(path.read_text())
        if damage=='root_empty_commitment':value['outputs']={}
        elif damage=='root_missing_case':value['outputs'].pop('series_00000/complete.json')
        else:value['version']='invalid'
        path.write_text(json.dumps(value))
    if damage=='none':
        result=fb.run_batch(root,export,out,**kwargs)
        assert result['resumed']==1 and result['n_ok']==1
    else:
        with pytest.raises(ValueError,match='batch') as err:fb.run_batch(root,export,out,**kwargs)
        assert 'synthetic_private' not in str(err.value)


def test_native_boxes_come_from_remapped_instances(batch,monkeypatch):
    fb,root,export,out=batch
    def inference(model,source,crop,device,**kw):
        labels=np.zeros(source.image_dhw.shape,np.int32);labels[:,2:6,4:8]=1
        return SimpleNamespace(anatomy=labels,instances=labels,lesions=[{'instance':1,'host':0,
            'host_probs':[1]+[0]*13,'score':.9,'box':[0,999,999,1,1000,1000]}])
    monkeypatch.setattr(fb,'infer_input',inference)
    record,arrays=fb.run_series(None,root/'file_brain_AXFLAIR_200_000001',modes=('2d','3d'),
        crops={'2d':(1,16,16),'3d':(2,16,16)},score_threshold=.5,device='cpu')
    assert all(r['box'][1:3]==[2,4] and r['box'][4:]==[6,8] for r in record['lesions'])
    assert {r['slice'] for r in record['lesions'] if r['mode']=='2d'}=={0,1,2}


def test_cli_errors_do_not_echo_private_values(capsys):
    import runpy
    path=Path(__file__).resolve().parents[1]/'scripts/aur_fastmri_batch.py'
    main=runpy.run_path(str(path))['main']
    with pytest.raises(SystemExit):main(['--synthetic_private'])
    assert 'synthetic_private' not in capsys.readouterr().err


def test_partial_attempt_is_preserved_and_restarted_without_reusing_arrays(batch,monkeypatch):
    fb,root,export,out=batch
    monkeypatch.setattr(fb,'run_series',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('synthetic_private')))
    with pytest.raises(ValueError):fb.run_batch(root,export,out)
    attempts=list(out.glob('.partial_*'));assert len(attempts)==1
    monkeypatch.setattr(fb,'run_series',fake_series)
    assert fb.run_batch(root,export,out)['n_ok']==1
    assert attempts[0].exists()


def test_batch_source_closure_includes_model_dependencies_and_entrypoint():
    keys=api()._sources()
    assert {'anatobind/aur/unified.py','anatobind/aur/unified_artifacts.py','anatobind/aur/targets.py',
            'anatobind/aur/dicom_pixelgrid.py','scripts/aur_fastmri_batch.py'}<=set(keys)


def test_untrusted_skip_reason_cannot_enter_public_summary(batch,monkeypatch):
    fb,root,export,out=batch;monkeypatch.setattr(fb,'run_series',fake_series)
    fb.run_batch(root,export,out)
    (out/'complete.json').unlink()
    path=out/'series_00002/complete.json';value=json.loads(path.read_text())
    assert value['status']=='SKIPPED'
    value['reason']='synthetic_private';path.write_text(json.dumps(value))
    (out/'anonymous_summary.json').unlink()
    with pytest.raises(ValueError,match='batch') as err:fb.run_batch(root,export,out)
    assert 'synthetic_private' not in str(err.value)


def test_source_changed_during_inference_never_commits_resumable_series(batch,monkeypatch):
    fb,root,export,out=batch;original={p:p.read_bytes() for p in root.glob('*/*.dcm')}
    def changed(model,folder,**kwargs):
        record,arrays=fake_series(model,folder,**kwargs)
        path=next(folder.glob('*.dcm'));path.write_bytes(path.read_bytes()+b'changed')
        return record,arrays
    monkeypatch.setattr(fb,'run_series',changed)
    with pytest.raises(ValueError,match='batch'):fb.run_batch(root,export,out)
    assert not (out/'series_00000').exists() and list(out.glob('.partial_*'))
    for path,data in original.items():path.write_bytes(data)
    called=[]
    monkeypatch.setattr(fb,'run_series',lambda model,folder,**kw:called.append(folder) or fake_series(model,folder,**kw))
    assert fb.run_batch(root,export,out)['resumed']==0 and len(called)==1


def test_member_added_during_inference_invalidates_series_commit(batch,monkeypatch):
    fb,root,export,out=batch
    def changed(model,folder,**kwargs):
        result=fake_series(model,folder,**kwargs)
        path=next(folder.glob('*.dcm'));(folder/'extra_003.dcm').write_bytes(path.read_bytes())
        return result
    monkeypatch.setattr(fb,'run_series',changed)
    with pytest.raises(ValueError):fb.run_batch(root,export,out)
    assert not (out/'series_00000').exists()


def gate_registration(algorithm='lookup'):
    from copy import deepcopy
    from anatobind.aur.gate import GATE_V1
    return dict(gate=deepcopy(GATE_V1),relation_implementation=algorithm,
        validation_selection_sha256='a'*64,score_thresholds={'pdgm':.2,'bmsr':.6,'isles':.8},
        frozen_before_evaluation=True)


def test_formal_model_requires_registration_before_first_inference(batch,monkeypatch):
    fb,root,export,out=batch
    monkeypatch.setattr(fb,'model_from_unified_export',lambda *a:(None,{'pilot_only':False,'format_version':2}))
    monkeypatch.setattr(fb,'run_series',lambda *a,**k:pytest.fail('unregistered formal inference ran'))
    with pytest.raises(ValueError,match='batch'):fb.run_batch(root,export,out)


def test_registered_batch_uses_median_threshold_and_deployed_relation(batch,monkeypatch):
    fb,root,export,out=batch;reg=gate_registration();seen=[]
    monkeypatch.setattr(fb,'model_from_unified_export',lambda *a:(None,{'pilot_only':False,'format_version':2}))
    def series(model,folder,**kw):seen.append(kw);return fake_series(model,folder,**kw)
    monkeypatch.setattr(fb,'run_series',series)
    result=fb.run_batch(root,export,out,gate_registration=reg,expected_registration_sha256=fb.json_digest(reg))
    assert result['score_threshold']==.6 and result['relation_implementation']=='lookup'
    assert result['registration_sha256']==fb.json_digest(reg) and not result['pilot_only']
    assert seen[0]['relation_implementation']=='lookup' and seen[0]['score_threshold']==.6
    job=json.loads((out/'run_contract_private.json').read_text())
    assert job['gate_registration']==reg and job['relation_implementation']=='lookup'


@pytest.mark.parametrize('damage',['hash','gate','override','algorithm'])
def test_registered_policy_refuses_hash_protocol_or_threshold_override(batch,monkeypatch,damage):
    fb,root,export,out=batch;reg=gate_registration();expected=fb.json_digest(reg)
    kwargs={}
    if damage=='hash':expected='0'*64
    elif damage=='gate':reg['gate']['r']['agreement_min']=.01;expected=fb.json_digest(reg)
    elif damage=='algorithm':reg['relation_implementation']='per_case';expected=fb.json_digest(reg)
    else:kwargs['score_threshold']=.9
    with pytest.raises(ValueError,match='batch'):
        fb.run_batch(root,export,out,gate_registration=reg,expected_registration_sha256=expected,**kwargs)
    assert not out.exists()


def test_lookup_replaces_head_hosts_before_native_grid_mapping(batch,monkeypatch):
    fb,root,export,out=batch
    def inference(model,source,crop,device,**kwargs):
        anatomy=np.ones(source.image_dhw.shape,np.int32)
        instances=np.zeros_like(anatomy);instances[:,2:6,4:8]=1
        return SimpleNamespace(anatomy=anatomy,instances=instances,lesions=[{'instance':1,'host':13,
            'host_probs':[0]*13+[1],'score':.9,'box':[0,2,4,source.image_dhw.shape[0],6,8]}])
    monkeypatch.setattr(fb,'infer_input',inference)
    record,_=fb.run_series(None,root/'file_brain_AXFLAIR_200_000001',modes=('2d','3d'),
        crops={'2d':(1,16,16),'3d':(2,16,16)},score_threshold=.6,device='cpu',relation_implementation='lookup')
    assert record['relation_implementation']=='lookup'
    assert all(row['host']==0 and row['host_probs'][0]==1 for row in record['lesions'])
    assert all(row['relation_implementation']=='lookup' for row in record['lesions'])


def test_actual_decoded_sources_must_equal_frozen_inventory_before_inference(batch,monkeypatch):
    fb,root,export,out=batch;folder=root/'file_brain_AXFLAIR_200_000001'
    hashes=fb._series_files(folder);hashes[next(iter(hashes))]='0'*64
    monkeypatch.setattr(fb,'infer_input',lambda *a,**k:pytest.fail('unbound source inference'))
    with pytest.raises(ValueError,match='source'):
        fb.run_series(None,folder,modes=('3d',),crops={'3d':(2,16,16)},score_threshold=.5,
                      device='cpu',expected_source_hashes=hashes)


def test_registered_relation_policy_cannot_change_on_resume(batch,monkeypatch):
    fb,root,export,out=batch;reg=gate_registration();monkeypatch.setattr(fb,'run_series',fake_series)
    fb.run_batch(root,export,out,gate_registration=reg,expected_registration_sha256=fb.json_digest(reg))
    changed=gate_registration('head')
    with pytest.raises(ValueError,match='batch'):
        fb.run_batch(root,export,out,gate_registration=changed,expected_registration_sha256=fb.json_digest(changed))
