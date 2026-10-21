"""Opt-in planar routing, legacy preservation and explicit initialization contracts."""
from copy import deepcopy

import pytest
import torch

from anatobind.aur.model import AnatoBindBrain
from anatobind.aur.ssl.model import StageOne
from anatobind.aur.ssl.unified_init import warm_start_stage_one
from anatobind.aur.swin import SwinBackbone
from anatobind.aur.unified import inactive_adapter_parameters, migrate_legacy_state
from anatobind.aur.unified_run import UnifiedRun, validate_run_config
from anatobind.aur.unified_train import _config
from test_aur_unified_model import BACKBONE, MODEL, inputs, assert_same_tree
from test_aur_unified_run import CONTRACT, config, factory
from test_aur_ssl_unified_init import legacy_checkpoint, SMALL
from test_aur_unified_training import assert_state_equal


def planar_config(**changes):
    cfg = config(**changes)
    cfg['model']['planar_merge'] = True
    return cfg


def test_merge_uses_four_observed_neighbors_and_planar_border_pooling():
    import anatobind.aur.swin as swin
    assert hasattr(swin, 'PatchMerging2D'), 'explicit planar merging is missing'
    merge = swin.PatchMerging2D(2)
    x = torch.arange(30.).reshape(1, 1, 3, 5, 2).requires_grad_()
    coords = torch.stack(torch.meshgrid(torch.arange(1.), torch.arange(3.), torch.arange(5.), indexing='ij'), -1)[None]
    valid = torch.tensor([[[[1.,0.,0.,0.,0.],[0.,0.,1.,0.,0.],[0.,0.,0.,0.,1.]]]])
    seen = []
    hook = merge.norm.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach().clone()))
    out, c, v = merge(x, coords, valid)
    hook.remove()
    # j/k order; no all-zero synthetic depth half in the normalization input.
    assert seen[0].shape == (1,1,2,3,8)
    assert seen[0][0,0,0,0].tolist() == [0.,1.,2.,3.,10.,11.,12.,13.]
    assert seen[0][0,0,1,2].tolist() == [28.,29.,0.,0.,0.,0.,0.,0.]
    expected_coords = torch.tensor([[[[[0.,.5,.5],[0.,.5,2.5],[0.,.5,4.]],
                                       [[0.,2.,.5],[0.,2.,2.5],[0.,2.,4.]]]]])
    assert torch.equal(c, expected_coords)
    assert torch.equal(v, torch.tensor([[[[1.,1.,0.],[0.,0.,1.]]]]))
    assert out.shape == (1,1,2,3,4)
    out.square().sum().backward()
    assert x.grad.shape == x.shape and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='D=1'):
        merge(x.detach().expand(1,2,3,5,2), coords.expand(1,2,3,5,3), valid.expand(1,2,3,5))


@pytest.mark.parametrize('constructor,options', [(StageOne,BACKBONE),(AnatoBindBrain,MODEL),(SwinBackbone,BACKBONE)])
def test_default_false_and_planar_preserve_all_old_tensors_rng_and_true3d(constructor, options):
    torch.manual_seed(17)
    old = constructor(**options, dual_input=True).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(17)
    disabled = constructor(**options, dual_input=True, planar_merge=False).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert_state_equal(old.state_dict(), disabled.state_dict())
    torch.manual_seed(17)
    planar = constructor(**options, dual_input=True, planar_merge=True).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, tensor in old.state_dict().items():
        assert torch.equal(tensor, planar.state_dict()[key]), key
    assert len(set(planar.state_dict()) - set(old.state_dict())) == 9
    if constructor != SwinBackbone:
        assert 'planar_merge' not in old.cfg and disabled.cfg['planar_merge'] is False
    with torch.no_grad():
        data = inputs((8,32,32))
        if constructor == StageOne:
            hidden = torch.zeros(1,4,8,8,dtype=torch.bool)
            assert_same_tree(old.view(*data, hidden), planar.view(*data, hidden))
        else:
            assert_same_tree(old(*data), planar(*data))
    with pytest.raises(RuntimeError):
        planar.load_state_dict(old.state_dict(), strict=True)


@pytest.mark.parametrize('value', [1,0,'true',None,[],{}])
def test_planar_flag_requires_a_real_boolean(value):
    with pytest.raises(ValueError):
        StageOne(**SMALL, dual_input=True, planar_merge=value)
    cfg = config(); cfg['model']['planar_merge'] = value
    with pytest.raises(ValueError):
        validate_run_config(cfg)
    with pytest.raises(ValueError):
        _config({'model':cfg['model']})


def test_planar_requires_dual_input_and_old_config_absence_is_canonical():
    with pytest.raises(ValueError):
        SwinBackbone(**SMALL, planar_merge=True)
    assert 'planar_merge' not in validate_run_config(config())['model']
    assert 'planar_merge' not in _config({'model':config()['model']})['model']
    assert validate_run_config(planar_config())['model']['planar_merge'] is True


@pytest.mark.parametrize('constructor,options', [(StageOne,SMALL),(AnatoBindBrain,MODEL),(SwinBackbone,SMALL)])
def test_explicit_migration_recognizes_nested_merge_state_and_keeps_constructor_policy(constructor,options):
    old = constructor(**options)
    target = constructor(**options, dual_input=True, planar_merge=True)
    before = deepcopy(target.state_dict())
    audit = migrate_legacy_state(target, old.state_dict())
    expected = set(before) - set(old.state_dict())
    assert set(audit['initialized_keys']) == expected
    assert sum('merge_2d' in k for k in expected) == 9
    for key in expected:
        assert torch.equal(target.state_dict()[key], before[key])
    for key, tensor in old.state_dict().items():
        assert torch.equal(target.state_dict()[key], tensor)


@pytest.mark.parametrize('policy', ['full_stage_one','backbone_only'])
def test_warm_start_preserves_legacy_and_seeded_io_and_derives_nine_slices(tmp_path, policy):
    source, _, path = legacy_checkpoint(tmp_path, policy)
    # Distinct norm values make slicing mistakes visible, unlike default all-ones/all-zeros.
    for stage in source.backbone.stages[:-1]:
        with torch.no_grad():
            stage.merge.norm.weight.copy_(torch.arange(stage.merge.norm.weight.numel()).float()+1)
            stage.merge.norm.bias.copy_(torch.arange(stage.merge.norm.bias.numel()).float()/10)
    payload = torch.load(path, weights_only=True)
    key = 'model' if policy == 'full_stage_one' else 'backbone_state_dict'
    payload[key] = source.state_dict() if policy == 'full_stage_one' else source.backbone.state_dict()
    torch.save(payload,path)
    old = StageOne(**SMALL, dual_input=True)
    old_lineage = warm_start_stage_one(old,path,policy=policy)
    target = StageOne(**SMALL,dual_input=True,planar_merge=True)
    rng = torch.get_rng_state().clone()
    lineage = warm_start_stage_one(target,path,policy=policy)
    assert torch.equal(rng, torch.get_rng_state())
    for name, tensor in old.state_dict().items():
        assert torch.equal(tensor,target.state_dict()[name]), name
    assert lineage['copied_keys'] == old_lineage['copied_keys']
    assert set(lineage['initialized_keys']) == set(old_lineage['initialized_keys'])
    records = lineage['derived_tensors']
    assert len(records) == 9 and set(records) == set(target.state_dict())-set(old.state_dict())
    for name, record in records.items():
        origin = name.replace('.merge_2d.', '.merge.')
        tensor = source.state_dict()[origin]
        width = target.state_dict()[name].shape[-1]
        assert torch.equal(target.state_dict()[name], tensor[..., :width])
        assert record['source_key'] == origin
        assert record['policy'] == 'legacy_depth0_first4c'
        assert record['slice'] == {'axis': tensor.ndim-1, 'start':0, 'stop':width}
        assert len(record['source_sha256']) == len(record['derived_sha256']) == 64
    assert lineage['planar_merge_initialization']['exact_legacy_2d_equivalence'] is False
    assert lineage['planar_merge_initialization']['decoder_policy'] == ('legacy_copied' if policy=='full_stage_one' else 'seeded_reset')


@pytest.mark.parametrize('damage', ['source_shape','source_dtype','source_nan','derived_shape','derived_dtype','source_flag'])
def test_invalid_planar_initialization_is_atomic_for_all_weights_and_rng(tmp_path,damage):
    _, saved, path = legacy_checkpoint(tmp_path)
    target = StageOne(**SMALL,dual_input=True,planar_merge=True)
    key = 'backbone.stages.2.merge.reduction.weight'
    if damage == 'source_shape': saved['model'][key] = torch.zeros(1)
    elif damage == 'source_dtype': saved['model'][key] = saved['model'][key].double()
    elif damage == 'source_nan': saved['model'][key][0,0] = float('nan')
    elif damage == 'source_flag': saved['meta']['config']['planar_merge'] = True
    elif damage == 'derived_shape': target.backbone.stages[2].merge_2d.reduction = torch.nn.Linear(256,127,bias=False)
    else: target.backbone.stages[2].merge_2d.double()
    torch.save(saved,path)
    before,rng = deepcopy(target.state_dict()),torch.get_rng_state().clone()
    with pytest.raises((ValueError,RuntimeError)):
        warm_start_stage_one(target,path)
    assert_state_equal(before,target.state_dict())
    assert torch.equal(rng,torch.get_rng_state())


def test_planar_runner_resumes_with_old_momenta_and_leaves_inactive_merges_exact(tmp_path):
    cfg = planar_config()
    full = UnifiedRun('I',cfg,CONTRACT)
    logs = [full.step(factory) for _ in range(4)]
    expected = torch.load(full.save(tmp_path/'full.pt'),weights_only=True)
    split = UnifiedRun('I',cfg,CONTRACT)
    prefix = [split.step(factory) for _ in range(2)]
    path = split.save(tmp_path/'split.pt')
    restored = UnifiedRun.resume(path,cfg,CONTRACT)
    for mode in ('2d','3d'):
        inactive = list(inactive_adapter_parameters(restored.model,mode))
        assert len(inactive) == 13
        for parameter in inactive:
            assert restored.optimizer.state[parameter]['exp_avg'].abs().sum() > 0
        before = [p.detach().clone() for p in inactive]
        prefix.append(restored.step(factory))
        assert all(p.grad is None and torch.equal(p,v) for p,v in zip(inactive,before))
    assert prefix == logs
    assert_state_equal(expected,torch.load(restored.save(tmp_path/'resumed.pt'),weights_only=True))
    with pytest.raises(ValueError,match='configuration'):
        UnifiedRun.resume(path,config(),CONTRACT)


def test_supervised_planar_aur_gradients_and_zero_touch_optimizer_inactivity():
    model = AnatoBindBrain(**MODEL,dual_input=True,planar_merge=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
    for mode,shape in [('2d',(1,36,28)),('3d',(8,32,32)),('2d',(1,36,28))]:
        optimizer.zero_grad(set_to_none=True)
        out = model(*inputs(shape),input_mode=mode)
        anatomy,lesions,relation = model.entity_masks(out),model.event_masks(out),model.bind(out,0,torch.tensor([0]))
        assert anatomy.shape == (1,32,*shape) and lesions.shape == (1,3,*shape) and relation.shape == (1,14)
        (anatomy.square().mean()+lesions.square().mean()+relation.square().mean()).backward()
        inactive = list(inactive_adapter_parameters(model,mode))
        before = [p.detach().clone() for p in inactive]
        for stage in model.backbone.stages[:-1]:
            active = stage.merge_2d if mode == '2d' else stage.merge
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active.parameters())
            assert sum(p.grad.abs().sum() for p in active.parameters()) > 0
        # The supervised objective zero-touches unused parameters; clearing them must also defeat Adam history.
        for p in inactive: p.grad = torch.zeros_like(p)
        for p in inactive_adapter_parameters(model,mode): p.grad = None
        optimizer.step()
        assert all(torch.equal(p,v) for p,v in zip(inactive,before))


def test_planar_pilot_training_and_formal_certificate_export_load_boundary(tmp_path,monkeypatch):
    from anatobind.aur.unified_train import UnifiedPilot
    from anatobind.aur.unified_infer import model_from_unified_export
    from test_aur_unified_infer import certified_export
    from test_aur_unified_training import factory as pilot_factory
    cfg = planar_config()
    pilot = UnifiedPilot('I',{'model':cfg['model'],'steps':2,'ssl':cfg['ssl']},CONTRACT)
    for _ in range(2): pilot.step(pilot_factory)
    assert pilot.model.cfg['planar_merge'] is True
    first = UnifiedRun('I',cfg,CONTRACT)
    runner = UnifiedRun('II',cfg,CONTRACT,initializer=first.export())
    states = []
    for i in range(4):
        runner.step(factory)
        if i in (1,3):
            saved = torch.load(runner.save(tmp_path/f'boundary{i}.pt'),weights_only=True)
            saved['purpose'] = saved['cfg']['purpose'] = 'formal'
            saved['execution_id'] = '4'*64
            states.append(saved)
    exported = certified_export(tmp_path,monkeypatch,states)
    path=tmp_path/'export.pt'; torch.save(exported,path)
    loaded,meta=model_from_unified_export(path,data_contract=CONTRACT)
    assert loaded.cfg['planar_merge'] is True and meta['phase_completed']
    selected=AnatoBindBrain(**exported['model_config']).eval()
    selected.load_state_dict(exported['model_state_dict'],strict=True)
    with torch.no_grad():
        for mode,shape in [('2d',(1,16,16)),('3d',(4,16,16))]:
            args=inputs(shape)
            assert_same_tree(selected(*args,input_mode=mode),loaded(*args,input_mode=mode))
    # A certificate with matching config/hash commitments still cannot load tensors from a different architecture.
    for state in states:
        state['model_config']['planar_merge']=False
        state['cfg']['model']['planar_merge']=False
    wrong=certified_export(tmp_path,monkeypatch,states)
    torch.save(wrong,path)
    with pytest.raises(RuntimeError,match='incompatible'):
        model_from_unified_export(path)


def test_historical_176_tensor_layout_copies_every_value_and_preserves_four_seeded_io(tmp_path):
    options = {**SMALL, 'depths': (2,2,6,2)}
    source = StageOne(**options)
    assert len(source.state_dict()) == 176
    path = tmp_path/'synthetic_legacy176.pt'
    torch.save({'model':source.state_dict(),'meta':{'stage':'I','config':source.cfg}},path)
    old = StageOne(**options,dual_input=True)
    target = StageOne(**options,dual_input=True,planar_merge=True)
    old_audit = warm_start_stage_one(old,path)
    audit = warm_start_stage_one(target,path)
    assert len(audit['copied_keys']) == 176
    assert len(audit['initialized_keys']) == 4
    assert len(audit['derived_tensors']) == 9
    assert audit['initialized_state_sha256'] == old_audit['initialized_state_sha256']
    for name,tensor in old.state_dict().items():
        assert torch.equal(tensor,target.state_dict()[name])
