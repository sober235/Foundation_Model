"""Gate observations preserve roles, same-lesion pairing and private evidence."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from anatobind.aur import unified_evaluation as ue
from anatobind.aur.data_contract import json_digest, file_sha256
from test_aur_unified_infer import tiny_model, certified_export, supervised_boundaries
from test_aur_unified_evaluation import qualified


@pytest.fixture
def gate_row(qualified):
    home, contract, row = qualified
    return home, contract, {**row, 'role':'historical_test', 'split':'test'}


def test_freeze_pairs_centroid_and_stride_without_duplicate_planes(gate_row):
    _, _, row = gate_row
    plan = ue.freeze_gate_observations([row])
    assert plan['selection'] == 'gate_v1_stride_and_lesion_centroid'
    assert plan['evidence'] == 'NOT_EVIDENCE'
    obs = plan['observations']
    assert len({(o['row_index'],o['input_mode'],o['plane_index']) for o in obs}) == len(obs)
    assert [(o['plane_index'],o['target_instances']) for o in obs if 'lesion_centroid' in o['plane_kinds']] == [(2,[1])]
    assert any(o['plane_index']==0 and o['plane_kinds']==['stride'] for o in obs)
    assert any(o['input_mode']=='3d' and o['plane_kinds']==['volume'] for o in obs)
    assert all(o['pixel_area_mm2']==.25 for o in obs)


def test_gate_roles_must_match_and_cannot_mix_or_duplicate(gate_row):
    _, _, row = gate_row
    for rows in ([dict(row,split='val')],[row,row],[row,dict(row,case='other',role='development_validation',split='val')]):
        with pytest.raises(ValueError): ue.freeze_gate_observations(rows)


def test_gate_path_has_mask_dice_and_inplane_refs_without_mutating_training(gate_row):
    _, _, row = gate_row
    plan = ue.freeze_gate_observations([row])
    model = tiny_model().eval()
    records=[ue.evaluate_observation(model,row,o,crop=(1 if o['input_mode']=='2d' else 4,16,16)) for o in plan['observations']]
    volume=next(r for r in records if r['input_mode']=='3d')
    plane=next(r for r in records if 'lesion_centroid' in r['plane_kinds'])
    assert set(volume['u']['matched_dice'])==set(volume['u']['matches'])
    assert all(0<=v<=1 for v in volume['u']['matched_dice'].values())
    assert volume['r']['instances'][0]['top_share']==pytest.approx(1.)
    assert plane['u']['source_ids']==[1,None]
    assert plane['r_inplane']['eligible']==2
    assert all(r['truth']==0 for r in plane['r_inplane']['instances'])
    assert plane['row_sha256']==json_digest(row)
    assert row['split']=='test' and row['role']=='historical_test'
    # Historical role never enters the old central-plane development API.
    old=deepcopy(plan['observations'][0]);old.pop('gate_version')
    with pytest.raises(ValueError):ue.evaluate_observation(model,row,old,crop=(1,16,16))


def test_direct_row_runner_requires_verified_role_contract(gate_row,monkeypatch):
    home,_,row=gate_row
    def forbidden(*args,**kwargs):pytest.fail('unverified member loaded')
    monkeypatch.setattr(ue,'model_from_unified_export',forbidden)
    with pytest.raises(ValueError):
        ue.run_gate_evaluation([row],'0'*64,home/'missing.pt',home/'unsafe',score_thresholds={'isles':.5})
    assert not (home/'unsafe').exists()


def test_real_synthetic_export_and_qualified_development_end_to_end(qualified,monkeypatch,tmp_path,supervised_boundaries):
    from anatobind.aur import data_contract as dc
    from anatobind.aur.gate import GATE_V1
    home,contract,row=qualified
    digest=file_sha256(contract/'contract.json')
    identity={**supervised_boundaries[0]['data_contract'],'contract_sha256':digest}
    exported=certified_export(tmp_path,monkeypatch,supervised_boundaries,
        lambda s,f:[st.__setitem__('data_contract',identity) for st in (s,f)])
    path=home/'formal.pt';torch.save(exported,path)
    monkeypatch.setattr(ue,'validate_private_output',lambda p:dc.validate_private_output(p,private_root=home))
    registration={'gate':GATE_V1,'relation_implementation':'head','validation_selection_sha256':'a'*64,
                  'score_thresholds':{'isles':.4,'pdgm':.5,'bmsr':.5},'frozen_before_evaluation':True}
    summary=ue.run_gate_evaluation([row],digest,path,home/'gate',role_contract=contract,
        crops={'2d':(1,16,16),'3d':(4,16,16)},score_thresholds={'isles':.4},
        gate_registration=registration,expected_registration_sha256=json_digest(registration))
    assert summary['gate_version']==ue.GATE_VERSION and not summary['quality_certified']
    assert 'synthetic_private' not in json.dumps(summary)
    assert (home/'gate/complete.json').is_file()
    plan=json.loads((home/'gate/plan_private.json').read_text())
    assert plan['configuration']['score_thresholds']=={'isles':.4}
    assert plan['input_files'] and all(len(v)==64 for v in plan['input_files'].values())


def test_committed_rows_cannot_be_changed_before_gate_run(qualified,monkeypatch):
    home,contract,row=qualified
    bad={**row,'u_supervised':False}
    with pytest.raises(ValueError):
        ue.run_gate_evaluation([bad],file_sha256(contract/'contract.json'),home/'no.pt',home/'bad',
                              role_contract=contract,score_thresholds={'isles':.5})
    assert not (home/'bad').exists()


def test_anatomy_only_source_does_not_need_a_disease_threshold(qualified,monkeypatch,tmp_path,supervised_boundaries):
    from anatobind.aur import data_contract as dc
    from anatobind.aur.gate import GATE_V1
    home,contract,row=qualified
    row={**row,'source':'sibbms','u_supervised':False,'r_supervised':False}
    identity=supervised_boundaries[0]['data_contract']
    exported=certified_export(tmp_path,monkeypatch,supervised_boundaries)
    path=home/'formal.pt';torch.save(exported,path)
    reg={'gate':GATE_V1,'relation_implementation':'head','validation_selection_sha256':'a'*64,
         'score_thresholds':{'isles':.4,'pdgm':.5,'bmsr':.5},'frozen_before_evaluation':True}
    monkeypatch.setattr(ue,'validate_private_output',lambda p:dc.validate_private_output(p,private_root=home))
    monkeypatch.setattr(ue,'_verify_gate_role',lambda *a:'c'*64)
    summary=ue.run_gate_evaluation([row],identity['contract_sha256'],path,home/'anatomy',
        role_contract=contract,crops={'2d':(1,16,16),'3d':(4,16,16)},
        score_thresholds=reg['score_thresholds'],gate_registration=reg,
        expected_registration_sha256=json_digest(reg))
    assert summary['modes']['2d']['coverage']['a']>0 and summary['modes']['2d']['coverage']['u']==0


def test_large_detection_hit_retains_poor_instance_dice(gate_row,monkeypatch):
    _,_,row=gate_row
    obs=ue.freeze_gate_observations([row],modes=('3d',))['observations'][0]
    def broad(model,source,*args,**kwargs):
        shape=source.image_dhw.shape
        inst=np.ones(shape,np.int32)
        return SimpleNamespace(anatomy=np.ones(shape,np.int16),instances=inst,
            lesions=[dict(instance=1,host=0,score=.8,box=[0,0,0,*shape],
                          volume_mm3=float(inst.size)*.25)])
    monkeypatch.setattr(ue,'infer_input',broad)
    record=ue.evaluate_observation(tiny_model().eval(),row,obs,crop=(4,16,16))
    assert record['u']['n_hit']==1
    assert record['u']['matched_dice'][1]==pytest.approx(2*144/(1024+144))
    assert record['u']['matched_dice'][1]<.5


@pytest.mark.parametrize('change',[{'split':'train'}, {'role':'historical_test','split':'val'}])
def test_invalid_gate_role_refused_before_any_array_load(gate_row,monkeypatch,change):
    _,_,row=gate_row
    def forbidden(*args,**kwargs):pytest.fail('invalid role read images')
    monkeypatch.setattr(ue,'load_unified_volume',forbidden)
    with pytest.raises(ValueError):ue.freeze_gate_observations([{**row,**change}])


def test_historical_gate_refuses_stage_two_before_freezing_arrays(gate_row,monkeypatch,tmp_path,supervised_boundaries):
    from anatobind.aur import data_contract as dc
    from anatobind.aur.gate import GATE_V1
    home,_,row=gate_row
    exported=certified_export(tmp_path,monkeypatch,supervised_boundaries)
    exported['stage']='II'
    path=home/'stage_two.pt';torch.save(exported,path)
    reg={'gate':GATE_V1,'relation_implementation':'head','validation_selection_sha256':'a'*64,
         'score_thresholds':{'isles':.4,'pdgm':.5,'bmsr':.5},'frozen_before_evaluation':True}
    monkeypatch.setattr(ue,'validate_private_output',lambda p:dc.validate_private_output(p,private_root=home))
    monkeypatch.setattr(ue,'_verify_gate_role',lambda *a:'c'*64)
    def forbidden(*args,**kwargs):pytest.fail('premature historical gate read arrays')
    monkeypatch.setattr(ue,'freeze_gate_observations',forbidden)
    with pytest.raises(ValueError):
        ue.run_gate_evaluation([row],exported['data_contract']['contract_sha256'],path,home/'premature',
            role_contract=home/'roles',score_thresholds={'isles':.4},gate_registration=reg,
            expected_registration_sha256=json_digest(reg))


def test_uncertified_stage_three_cannot_open_historical_labels(gate_row,monkeypatch):
    from anatobind.aur import data_contract as dc
    from anatobind.aur.gate import GATE_V1
    home,_,row=gate_row
    identity={'contract_sha256':'b'*64}
    path=home/'uncertified.pt'
    torch.save({'format_version':2,'stage':'III','purpose':'resource_probe','data_contract':identity},path)
    reg={'gate':GATE_V1,'relation_implementation':'head','validation_selection_sha256':'a'*64,
         'score_thresholds':{'isles':.4,'pdgm':.5,'bmsr':.5},'frozen_before_evaluation':True}
    monkeypatch.setattr(ue,'validate_private_output',lambda p:dc.validate_private_output(p,private_root=home))
    monkeypatch.setattr(ue,'_verify_gate_role',lambda *a:'c'*64)
    def forbidden(*args,**kwargs):pytest.fail('unverified stageIII read historical labels')
    monkeypatch.setattr(ue,'freeze_gate_observations',forbidden)
    with pytest.raises(ValueError):
        ue.run_gate_evaluation([row],identity['contract_sha256'],path,home/'uncertified',
            role_contract=home/'roles',score_thresholds={'isles':.4},gate_registration=reg,
            expected_registration_sha256=json_digest(reg))


def test_historical_model_can_only_claim_one_final_evaluation(gate_row,monkeypatch):
    from anatobind.aur import data_contract as dc
    from anatobind.aur.gate import GATE_V1
    home,_,row=gate_row
    identity={'contract_sha256':'b'*64};path=home/'certified_metadata.pt'
    torch.save({'stage':'III','data_contract':identity},path)
    model=tiny_model().eval()
    monkeypatch.setattr(ue,'model_from_unified_export',lambda *a,**k:(model,{'pilot_only':False,'phase_completed':True}))
    monkeypatch.setattr(ue,'validate_private_output',lambda p:dc.validate_private_output(p,private_root=home))
    monkeypatch.setattr(ue,'_verify_gate_role',lambda *a:'c'*64)
    reg={'gate':GATE_V1,'relation_implementation':'head','validation_selection_sha256':'a'*64,
         'score_thresholds':{'isles':.4,'pdgm':.5,'bmsr':.5},'frozen_before_evaluation':True}
    args=dict(role_contract=home/'roles',score_thresholds={'isles':.4},gate_registration=reg,
              expected_registration_sha256=json_digest(reg),modes=('2d',),crops={'2d':(1,16,16)})
    ue.run_gate_evaluation([row],identity['contract_sha256'],path,home/'first',**args)
    def forbidden(*a,**k):pytest.fail('same final model evaluated twice')
    monkeypatch.setattr(ue,'freeze_gate_observations',forbidden)
    with pytest.raises(ValueError):
        ue.run_gate_evaluation([row],identity['contract_sha256'],path,home/'second',**args)
    assert len(list((home/'roles/final_evaluation_claims').glob('*.json')))==1
