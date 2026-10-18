"""Synthetic frozen-forward boundary checks; no private evidence is accessed."""
import importlib
import time

import numpy as np
import pytest
import torch


def modules():
    try:
        return importlib.import_module('scripts.aur_diagnose_u_background')
    except ModuleNotFoundError:
        pytest.fail('Paired background allocation runner is missing')


def test_public_runner_apis_exist():
    runner=modules()
    assert callable(runner.collect_crop) and callable(runner.run)


@pytest.fixture
def tiny(monkeypatch):
    runner=modules()
    from scripts import aur_diagnose_u_supervision as prior
    from anatobind.aur.model import AnatoBindBrain
    from test_aur_model import TINY, _batch
    recipe=dict(prior.A.RECIPE,crop=[8,32,32],n_queries=4,points=32,n_rows=1,n_crops=2,
                pairs=[['bmsr','T1c']])
    monkeypatch.setattr(prior.A,'RECIPE',recipe)
    monkeypatch.setattr(runner,'RECIPE',recipe)
    torch.manual_seed(91)
    model=AnatoBindBrain(**TINY).eval().requires_grad_(False)
    monkeypatch.setattr(prior.A,'CHECKPOINT_STATE_SHA',prior.A.model_state_digest(model))
    monkeypatch.setattr(runner.B,'CHECKPOINT_STATE_SHA',prior.A.model_state_digest(model))
    image,valid,coords,local=_batch()
    inst=torch.zeros_like(valid[0],dtype=torch.int64); inst[:2,:3,:3]=1
    entity=torch.zeros_like(inst); entity[2:]=1
    present=torch.zeros(32,dtype=torch.bool); present[0]=True
    crop=dict(image=image[0],valid=valid[0],coords=coords[0],local=local[0],instance=inst,
              point_weight=valid[0].clone(),n_instances=1,u_supervised=torch.tensor(True),
              entity=entity,entity_present=present,a_ignore=inst>0,a_supervised=torch.tensor(True))
    old_meta,old_arrays=prior.collect_crop(model,crop,1.,source='bmsr',sequence='T1c',
        selection_ordinal=0,crop_ordinal=0,device=torch.device('cpu'),deadline=time.monotonic()+60)
    return runner,model,crop,old_meta,old_arrays


def test_one_frozen_forward_preserves_original_controls_and_model(tiny,monkeypatch):
    runner,model,crop,old_meta,old_arrays=tiny
    before=runner.P.A.model_state_digest(model)
    calls=[]; forward=model.forward
    def count(*args):
        calls.append(1)
        return forward(*args)
    monkeypatch.setattr(model,'forward',count)
    meta,arrays=runner.collect_crop(model,crop,old_meta,old_arrays,device=torch.device('cpu'),
                                    deadline=time.monotonic()+90)
    assert len(calls)==1
    assert meta['model_state_before']==meta['model_state_after']==before
    assert before==runner.P.A.model_state_digest(model)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    np.testing.assert_array_equal(arrays['u_points'][:,0],old_arrays['points'])
    np.testing.assert_array_equal(arrays['u_logits'][:,0],old_arrays['sampled_mask_logits'])
    np.testing.assert_array_equal(arrays['r_logits_before'],arrays['r_logits_after'])
    assert meta['controls']['native_logits_verified'] and meta['controls']['decoder_verified']
    assert arrays['roi_features'].shape[0]==len(meta['roi_rows'])
    assert meta['report']==runner.B.reconstruct_crop(meta,arrays)


@pytest.mark.parametrize('field',['image','instance'])
def test_changed_crop_rejected_before_forward(tiny,monkeypatch,field):
    runner,model,crop,old_meta,old_arrays=tiny
    crop=dict(crop); crop[field]=crop[field].clone(); crop[field].view(-1)[0]+=1
    monkeypatch.setattr(model,'forward',lambda *args:pytest.fail('changed crop entered model'))
    with pytest.raises(ValueError):
        runner.collect_crop(model,crop,old_meta,old_arrays,device=torch.device('cpu'),deadline=time.monotonic()+10)


@pytest.mark.parametrize('field',['presence_logits','sampled_mask_logits','detection_map'])
def test_changed_native_reference_is_rejected(tiny,field):
    runner,model,crop,old_meta,old_arrays=tiny
    old_arrays={k:v.copy() for k,v in old_arrays.items()}
    old_arrays[field].flat[0]+=1
    with pytest.raises(ValueError):
        runner.collect_crop(model,crop,old_meta,old_arrays,device=torch.device('cpu'),deadline=time.monotonic()+90)


def test_expired_budget_stops_before_forward(tiny,monkeypatch):
    runner,model,crop,old_meta,old_arrays=tiny
    monkeypatch.setattr(model,'forward',lambda *args:pytest.fail('expired run entered model'))
    with pytest.raises(TimeoutError):
        runner.collect_crop(model,crop,old_meta,old_arrays,device=torch.device('cpu'),deadline=0)


def test_replay_branch_does_not_load_model(tmp_path,monkeypatch):
    runner=modules()
    args=runner.parser().parse_args(['--out',str(tmp_path/'synthetic'),'--replay-only'])
    stored={'code_sha':'synthetic'}
    monkeypatch.setattr(runner.B,'read_json',lambda *a:stored)
    monkeypatch.setattr(runner,'prepare_contract',lambda *a,**kw:(None,None,stored))
    monkeypatch.setattr(runner.B,'replay_archive',lambda out,contract:{'verified':True})
    monkeypatch.setattr(runner.P.I,'model_from_export',lambda *a:pytest.fail('replay loaded model'))
    assert runner.run(args)==0


def test_output_escape_is_rejected_before_private_preflight(tmp_path,monkeypatch):
    runner=modules()
    monkeypatch.setattr(runner,'OUTPUT_ROOT',tmp_path/'approved')
    monkeypatch.setattr(runner,'prepare_contract',lambda *a,**kw:pytest.fail('escaped run read inputs'))
    args=runner.parser().parse_args(['--out',str(tmp_path/'escape')])
    with pytest.raises(ValueError): runner.run(args)
