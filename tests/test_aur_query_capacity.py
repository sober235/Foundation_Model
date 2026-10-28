"""Explicit query capacity preserves every eligible dense target."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from anatobind.aur.unified_data import make_unified_crop
from anatobind.aur.losses import event_loss
from anatobind.aur.unified_run import validate_run_config
from test_aur_unified_run import config


def crowded(n):
    shape=(4,64,64)
    inst=np.zeros(shape,np.int32)
    for i in range(n):
        y,x=divmod(i,16)
        inst[:,y*4:y*4+2,x*4:x*4+2]=i+1
    return {'image':np.zeros(shape,np.float32),'raw_image':np.ones(shape,np.float32),
        'instance':inst,'entity':np.ones(shape,np.int16),'small':np.zeros(shape,bool),
        'a_ignore':np.zeros(shape,bool),'source_host_map':np.ones(shape,np.int16),
        'hosts':{'host':np.zeros(n,np.int64),'probs':np.eye(14,dtype=np.float32)[np.zeros(n,np.int64)],
                 'negatives':np.full((n,2),-1,np.int64)},
        'spacing':(1.,1.,1.),'voxel_mm3':1.,'affine':np.eye(4),'seq':3,
        'a_supervised':True,'u_supervised':True,'r_supervised':True,'case':'synthetic','split':'train'}


@pytest.mark.parametrize('count',[65,85,102,128])
@pytest.mark.parametrize('mode',['2d','3d'])
def test_explicit_128_preserves_all_instances_and_relation_eligibility(count,mode):
    out=make_unified_crop(crowded(count),np.random.default_rng(1),mode,
        (1,64,64) if mode=='2d' else (4,64,64),do_augment=False,n_events=128,
        plane_index=0 if mode=='2d' else None,
        planar_relation_policy='geometry_weak2d_v1' if mode=='2d' else None)
    assert out['n_instances']==count and int(out['instance'].max())==count
    assert len(out['host'])==count and out['r_eligible'].all()
    # One positive point per target proves the full Hungarian match has no truncation.
    target=torch.eye(count)
    mask=torch.full((1,128,count),-8.);mask[0,:count]=torch.where(target.bool(),8.,-8.)
    mask.requires_grad_()
    parts,matches=event_loss(torch.zeros(1,128,requires_grad=True),mask,[target],
        torch.ones(1,count),torch.tensor([True]),'unmatched_empty_v1')
    assert matches[0][0].numel()==count and sorted(matches[0][1].tolist())==list(range(count))
    sum(parts.values()).backward();assert torch.isfinite(mask.grad).all()


@pytest.mark.parametrize('mode',['2d','3d'])
def test_capacity_overflow_still_fails_without_dropping_instances(mode):
    for n,capacity in ((65,64),(129,128)):
        with pytest.raises(ValueError,match='capacity'):
            make_unified_crop(crowded(n),np.random.default_rng(1),mode,
                (1,64,64) if mode=='2d' else (4,64,64),do_augment=False,n_events=capacity,
                plane_index=0 if mode=='2d' else None)


@pytest.mark.parametrize('capacity',[True,0,-1,129,128.0])
def test_run_config_refuses_invalid_capacity(capacity):
    cfg=config();cfg['model']=deepcopy(cfg['model']);cfg['model']['n_events']=capacity
    with pytest.raises(ValueError,match='configuration'):validate_run_config(cfg)


def test_legacy_detection_summary_uses_explicit_model_capacity():
    from anatobind.aur.eval import _u_entry
    scans=[{'case':'synthetic','gt':[{'box':[0,0,0,1,1,1]} for _ in range(65)],'dets':[]}]
    old=_u_entry(scans,None,None,.05)
    expanded=_u_entry(scans,None,None,.05,query_capacity=128)
    assert old['query_capacity']==64 and old['saturated_cases']==['synthetic']
    assert expanded['query_capacity']==128 and expanded['saturated_cases']==[]


def test_evaluation_capacity_is_bound_to_model_and_shards():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    from anatobind.aur.eval import model_query_capacity
    assert model_query_capacity(SimpleNamespace(cfg={'n_events':128}))==128
    with pytest.raises(ValueError,match='capacity'):
        model_query_capacity(SimpleNamespace(cfg={}))
    path=Path(__file__).resolve().parents[1]/'scripts/aur_eval_merge.py'
    spec=importlib.util.spec_from_file_location('capacity_merge',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.shard_query_capacity([{'query_capacity':128},{'query_capacity':128}])==128
    assert module.shard_query_capacity([{},{}]) is None
    for infos in ([{'query_capacity':64},{'query_capacity':128}], [{},{'query_capacity':128}]):
        with pytest.raises(ValueError,match='capacity'):
            module.shard_query_capacity(infos)
