"""Frozen runner and complete-only replay regression checks (synthetic, CPU)."""
import copy
import importlib
import json
import time

import numpy as np
import pytest
import torch


def modules():
    return (importlib.import_module('scripts.aur_diagnose_u_supervision'),
            importlib.import_module('anatobind.aur.u_supervision_archive'))


def test_runner_and_archive_public_apis_exist():
    try:
        runner, archive = modules()
    except ModuleNotFoundError:
        pytest.fail('Frozen TRAIN runner and complete-only archive APIs are missing')
    assert callable(runner.collect_crop) and callable(runner.run) and callable(archive.replay_archive)


@pytest.fixture
def tiny(monkeypatch):
    runner, archive = modules()
    from anatobind.aur.model import AnatoBindBrain
    from test_aur_model import TINY, _batch
    recipe = dict(archive.RECIPE, crop=[8, 32, 32], n_queries=4, points=32, n_rows=1, n_crops=2,
                  pairs=[['bmsr', 'T1c']])
    monkeypatch.setattr(archive, 'RECIPE', recipe)
    torch.manual_seed(91)
    model = AnatoBindBrain(**TINY).eval().requires_grad_(False)
    monkeypatch.setattr(archive,'CHECKPOINT_STATE_SHA',archive.model_state_digest(model))
    image, valid, coords, local = _batch()
    inst = torch.zeros_like(valid[0], dtype=torch.int64)
    inst[:2, :3, :3] = 1
    crop = dict(image=image[0], valid=valid[0], coords=coords[0], local=local[0], instance=inst,
                point_weight=valid[0].clone(), n_instances=1, u_supervised=torch.tensor(True))
    return runner, archive, model, crop


def test_collection_preserves_real_tiny_outputs_bind_state_and_parameter_gradients(tiny):
    runner, archive, model, crop = tiny
    before = archive.model_state_digest(model)
    args = [crop[k][None] for k in ('image', 'valid', 'coords', 'local')]
    with torch.no_grad():
        out = model(*args)
        logits = model.bind(out, 0, torch.tensor([0, 2])).clone()
        dense = model.event_masks(out).clone()
    meta, arrays = runner.collect_crop(model, crop, 1., source='bmsr', sequence='T1c',
        selection_ordinal=0, crop_ordinal=0, device=torch.device('cpu'), deadline=time.monotonic()+60)
    with torch.no_grad():
        after = model(*args)
        assert torch.equal(dense, model.event_masks(after))
        assert torch.equal(logits, model.bind(after, 0, torch.tensor([0, 2])))
        assert all(torch.equal(out[k], after[k]) for k in ('event_presence', 'entity_embed', 'seq_logits'))
        for draw in range(2):
            actual = model.event_masks(out, torch.from_numpy(arrays['points'][draw:draw+1])).float()[0].numpy()
            np.testing.assert_array_equal(actual, arrays['sampled_mask_logits'][draw])
    assert before == archive.model_state_digest(model)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert meta['summaries'] == archive.reconstruct_crop(meta, arrays)


def make_archive(tiny, tmp_path):
    runner, archive, model, crop = tiny
    out = tmp_path / 'archive'
    selection = [{'train_row_position': 0, 'row': {'source':'bmsr', 'sequence':'T1c', 'patient':'PRIVATE',
                  'case':'PRIVATE', 'u_supervised':True}}]
    contract = {'recipe': copy.deepcopy(archive.RECIPE), 'selection': selection, 'inputs': {},
                'model_state_sha256':archive.CHECKPOINT_STATE_SHA,
                'controls': archive.fixed_controls(), 'code_sha':'synthetic', 'source_sha256':{},
                'checkpoint_sha256':'synthetic'}
    archive.create_archive(out, contract, selection, {})
    for c in range(2):
        meta, arrays = runner.collect_crop(model, crop, 1., source='bmsr', sequence='T1c',
            selection_ordinal=0, crop_ordinal=c, device=torch.device('cpu'), deadline=time.monotonic()+60)
        archive.write_crop(out, c, meta, arrays)
    archive.complete_archive(out, contract)
    return archive, out, contract


def test_complete_replay_is_cpu_numeric_and_read_only(tiny, tmp_path, monkeypatch):
    archive, out, contract = make_archive(tiny, tmp_path)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    import anatobind.aur.infer as infer
    import anatobind.aur.dataset as dataset
    monkeypatch.setattr(infer, 'model_from_export', lambda *a: pytest.fail('model loaded during replay'))
    monkeypatch.setattr(dataset, 'load_volume', lambda *a: pytest.fail('image loaded during replay'))
    assert archive.replay_archive(out, contract) == json.loads((out/'aggregate.json').read_text())
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    assert out.stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        archive.create_archive(out, contract, contract['selection'], {})


@pytest.mark.parametrize('field', ['points', 'presence_logits', 'gt_instance_map', 'outcomes', 'controls', 'membership'])
def test_semantic_corruption_is_rejected_even_after_rehashing(tiny, tmp_path, field):
    archive, out, contract = make_archive(tiny, tmp_path)
    p = out/'crop_00.json'
    meta = json.loads(p.read_text())
    with np.load(out/'crop_00.npz', allow_pickle=False) as z:
        arrays = {k:z[k] for k in z.files}
    if field == 'points': arrays['points'][0, 0] = arrays['valid'].size
    elif field == 'presence_logits': arrays[field] = arrays[field][:-1]
    elif field == 'gt_instance_map': arrays[field].flat[0] = 3
    elif field == 'outcomes': meta['summaries']['outcomes'] = [{'instance':999, 'status':'TP'}]
    elif field == 'controls': meta['controls']['parameters_frozen'] = False
    else: meta['row_ordinal'] = 1
    archive.write_crop(out, 0, meta, arrays)
    marker = json.loads((out/'COMPLETE.json').read_text())
    marker['files'] = archive.archive_hashes(out)
    archive.atomic_json(out/'COMPLETE.json', marker)
    with pytest.raises(ValueError): archive.replay_archive(out, contract)


def test_incomplete_and_changed_contract_rejected(tiny, tmp_path):
    archive, out, contract = make_archive(tiny, tmp_path)
    wrong = copy.deepcopy(contract)
    wrong['checkpoint_sha256'] = 'changed'
    with pytest.raises(ValueError): archive.replay_archive(out, wrong)
    (out/'COMPLETE.json').unlink()
    with pytest.raises(ValueError): archive.replay_archive(out, contract)


def test_dense_decoder_uses_float32_before_sigmoid(tiny, monkeypatch):
    runner, archive, model, crop = tiny
    from contextlib import nullcontext
    original = model.event_masks
    # Force bf16 mask/presence outputs even on CPU, to detect sigmoid rounding.
    def masks(*args, **kwargs): return original(*args, **kwargs).to(torch.bfloat16)
    original_forward=model.forward
    def forward(*args):
        out=original_forward(*args)
        out['event_presence']=out['event_presence'].to(torch.bfloat16)
        return out
    monkeypatch.setattr(model,'event_masks',masks)
    monkeypatch.setattr(model,'forward',forward)
    seen={}
    actual_decoder=runner.QueryCCDecoder
    class Capture(actual_decoder):
        def __call__(self,**kwargs):
            seen['event']=kwargs['event_prob'].clone()
            seen['presence']=kwargs['event_presence'].clone()
            return super().__call__(**kwargs)
    monkeypatch.setattr(runner,'QueryCCDecoder',Capture)
    runner.collect_crop(model,crop,1.,source='bmsr',sequence='T1c',selection_ordinal=0,crop_ordinal=0,
                        device=torch.device('cpu'),deadline=time.monotonic()+60)
    with torch.no_grad():
        out=model(*[crop[k][None] for k in ('image','valid','coords','local')])
        raw=model.event_masks(out)[0]
        assert torch.equal(seen['event'],raw.float().sigmoid())
        assert torch.equal(seen['presence'],out['event_presence'][0].float().sigmoid())
        assert not torch.equal(raw.sigmoid().float(),raw.float().sigmoid())


def test_metadata_selection_verified_before_any_model_or_volume_load(tmp_path,monkeypatch):
    runner,archive=modules()
    from types import SimpleNamespace
    pairs=archive.RECIPE['pairs']
    rows=[dict(source=s,sequence=q,patient=f'PRIVATE_{s}_{i}',case=f'PRIVATE_{s}_{i}',split='train',
               u_supervised=True,image='PRIVATE',anatomy='PRIVATE',lesion=None) for s,q in pairs for i in range(3)]
    # Match split_rows' requirement that nonempty validation metadata has held rows.
    val={pairs[0][0]:['HELD']}
    rows.append(dict(rows[0],patient='HELD'))
    rows.append(dict(rows[0],patient='TEST',split='test'))
    _,selection=runner.select_train_rows(rows,val)
    assert len(selection)==6
    samples=tmp_path/'samples.json'; samples.write_text(json.dumps(rows))
    vals=tmp_path/'val.json'; vals.write_text(json.dumps(val))
    chosen=tmp_path/'selection.json'; chosen.write_text(json.dumps(selection))
    checkpoint=tmp_path/'ckpt'; checkpoint.write_bytes(b'synthetic checkpoint')
    args=SimpleNamespace(samples=samples,val_patients=vals,selection_private=chosen,checkpoint=checkpoint,
        out=tmp_path/'out',replay_only=False,max_seconds=10,gpu=0)
    monkeypatch.setattr(runner,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(runner,'confined',lambda p:p)
    monkeypatch.setattr(runner,'FROZEN_HASHES',{k:archive.file_sha256(getattr(args,k)) for k in runner.FROZEN_HASHES})
    monkeypatch.setattr(runner,'POSITIONS_SHA',runner.digest_json([m['train_row_position'] for m in selection]))
    monkeypatch.setattr(runner,'input_manifest',lambda *a:pytest.fail('Raw inputs read before selection verified'))
    monkeypatch.setattr(runner.I,'model_from_export',lambda *a:pytest.fail('Model loaded before selection verified'))
    selection[0]['train_row_position']+=1
    chosen.write_text(json.dumps(selection))
    runner.FROZEN_HASHES['selection_private']=archive.file_sha256(chosen)
    with pytest.raises(ValueError,match='selection differs'): runner.run(args)
    assert not args.out.exists()


@pytest.mark.parametrize('change',['not_eval','not_frozen','wrong_queries','wrong_shape','unsupervised'])
def test_collect_rejects_invalid_model_or_crop_before_forward(tiny,monkeypatch,change):
    runner,archive,model,crop=tiny
    if change=='not_eval': model.train()
    elif change=='not_frozen': next(model.parameters()).requires_grad_(True)
    elif change=='wrong_queries': model.cfg['n_events']=3
    elif change=='wrong_shape': crop['valid']=crop['valid'][:1]
    else: crop['u_supervised']=torch.tensor(False)
    monkeypatch.setattr(model,'forward',lambda *a:pytest.fail('Invalid collection entered forward'))
    with pytest.raises(ValueError):
        runner.collect_crop(model,crop,1.,source='bmsr',sequence='T1c',selection_ordinal=0,crop_ordinal=0,
                            device=torch.device('cpu'),deadline=time.monotonic()+60)


def test_expired_collection_budget_leaves_no_result(tiny,monkeypatch):
    runner,archive,model,crop=tiny
    monkeypatch.setattr(model,'forward',lambda *a:pytest.fail('Expired collection entered forward'))
    with pytest.raises(TimeoutError):
        runner.collect_crop(model,crop,1.,source='bmsr',sequence='T1c',selection_ordinal=0,crop_ordinal=0,
                            device=torch.device('cpu'),deadline=0)


def test_replay_rejects_checksum_consistent_decoder_count_corruption(tiny,tmp_path):
    archive,out,contract=make_archive(tiny,tmp_path)
    p=out/'crop_00.json'; meta=archive.read_json(p)
    meta['decoder_stats']['final_instances']+=1
    archive.atomic_json(p,meta)
    archive.atomic_json(out/'COMPLETE.json',dict(schema=1,files=archive.archive_hashes(out)))
    with pytest.raises(ValueError): archive.replay_archive(out,contract)


def test_collection_records_observed_resource_usage(tiny):
    runner,archive,model,crop=tiny
    meta,_=runner.collect_crop(model,crop,1.,source='bmsr',sequence='T1c',selection_ordinal=0,crop_ordinal=0,
                              device=torch.device('cpu'),deadline=time.monotonic()+60)
    assert meta['resources']['elapsed_seconds']>0
    assert meta['resources']['rss_max_kib']>0
    assert meta['resources']['cuda_peak_allocated_bytes']==0
    assert meta['resources']['cuda_peak_reserved_bytes']==0


def test_deadline_before_completion_marker_keeps_archive_incomplete(tiny,tmp_path):
    runner,archive,model,crop=tiny
    out=tmp_path/'archive'
    selection=[{'train_row_position':0,'row':{'source':'bmsr','sequence':'T1c'}}]
    contract={'recipe':copy.deepcopy(archive.RECIPE),'selection':selection,'inputs':{},'controls':archive.fixed_controls(),
              'model_state_sha256':archive.CHECKPOINT_STATE_SHA}
    archive.create_archive(out,contract,selection,{})
    for i in range(2):
        meta,arrays=runner.collect_crop(model,crop,1.,source='bmsr',sequence='T1c',selection_ordinal=0,crop_ordinal=i,
                                       device=torch.device('cpu'),deadline=time.monotonic()+60)
        archive.write_crop(out,i,meta,arrays)
    with pytest.raises(TimeoutError): archive.complete_archive(out,contract,deadline=0)
    assert not (out/'COMPLETE.json').exists()


def test_replay_rejects_forged_equal_model_state_claims(tiny,tmp_path):
    archive,out,contract=make_archive(tiny,tmp_path)
    for i in range(2):
        p=out/f'crop_{i:02d}.json'; meta=archive.read_json(p)
        meta['state_before']=meta['state_after']='f'*64
        archive.atomic_json(p,meta)
    archive.atomic_json(out/'COMPLETE.json',dict(schema=1,files=archive.archive_hashes(out)))
    with pytest.raises(ValueError): archive.replay_archive(out,contract)


def test_model_state_digest_covers_parameters_buffers_dtype_shape_and_names():
    runner,archive=modules()
    m=torch.nn.Module(); m.register_parameter('p',torch.nn.Parameter(torch.tensor([1.,2.])))
    initial=archive.model_state_digest(m)
    m.register_buffer('b',torch.tensor([3.]))
    assert archive.model_state_digest(m)!=initial
    buf=archive.model_state_digest(m)
    m.b[0]=4.
    assert archive.model_state_digest(m)!=buf
    m.b=m.b.double()
    dtype=archive.model_state_digest(m)
    m.b=m.b.reshape(1,1)
    assert archive.model_state_digest(m)!=dtype


def test_replay_rejects_valid_but_nonrecipe_point_order(tiny,tmp_path):
    archive,out,contract=make_archive(tiny,tmp_path)
    meta=archive.read_json(out/'crop_00.json')
    with np.load(out/'crop_00.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    # Preserve the point/logit pairing, shape, ranges and empirical samples.
    arrays['points'][0]=arrays['points'][0,::-1].copy()
    arrays['sampled_mask_logits'][0]=arrays['sampled_mask_logits'][0,:,::-1].copy()
    meta['array_sha256']={k:archive.array_digest(v) for k,v in arrays.items()}
    with pytest.raises(ValueError): archive.reconstruct_crop(meta,arrays)


def test_replay_rejects_crop_tensor_hash_corruption(tiny,tmp_path):
    archive,out,contract=make_archive(tiny,tmp_path)
    p=out/'crop_00.json'; meta=archive.read_json(p)
    meta['crop_tensor_sha256']['valid']='0'*64
    archive.atomic_json(p,meta)
    archive.atomic_json(out/'COMPLETE.json',dict(schema=1,files=archive.archive_hashes(out)))
    with pytest.raises(ValueError): archive.replay_archive(out,contract)


@pytest.mark.parametrize('nonfinite',[float('nan'),float('inf'),float('-inf')])
def test_all_queries_must_have_finite_dense_probabilities(tiny,monkeypatch,nonfinite):
    runner,archive,model,crop=tiny
    original=model.event_masks
    def masks(out,points=None):
        value=original(out,points)
        if points is None: value[0,0,0,0,0]=nonfinite
        return value
    monkeypatch.setattr(model,'event_masks',masks)
    original_forward=model.forward
    def forward(*args):
        out=original_forward(*args)
        out['event_presence'].fill_(-10.)
        return out
    monkeypatch.setattr(model,'forward',forward)
    # Every query is absent; validation still has to inspect dense outputs.
    with pytest.raises(ValueError):
        runner.collect_crop(model,crop,1.,source='bmsr',sequence='T1c',selection_ordinal=0,crop_ordinal=0,
                            device=torch.device('cpu'),deadline=time.monotonic()+60)


def test_real_runner_refuses_output_outside_private_root_before_preflight(tmp_path):
    from types import SimpleNamespace
    runner,_=modules()
    args=SimpleNamespace(out=tmp_path/'outside',replay_only=False,max_seconds=10)
    with pytest.raises(ValueError,match='private output root'): runner.run(args)


def test_collection_uses_official_ignored_rows_without_fake_gt_ids(tiny):
    runner,archive,model,crop=tiny
    crop['point_weight'][4:6,4:7,4:7]=0
    meta,arrays=runner.collect_crop(model,crop,1.,source='bmsr',sequence='T1c',selection_ordinal=0,crop_ordinal=0,
                                  device=torch.device('cpu'),deadline=time.monotonic()+60)
    assert meta['n_instances']==1
    assert np.count_nonzero((arrays['valid']>.5)&(arrays['point_weight']<.5))==18
    assert archive.reconstruct_crop(meta,arrays)==meta['summaries']


def test_expiry_during_completion_hashing_leaves_no_marker(tiny,tmp_path,monkeypatch):
    archive,out,contract=make_archive(tiny,tmp_path)
    (out/'COMPLETE.json').unlink()
    (out/'aggregate.json').unlink()
    clock={'now':0.}
    original=archive.archive_hashes
    def hashes(path):
        result=original(path)
        clock['now']=2.
        return result
    monkeypatch.setattr(archive.time,'monotonic',lambda:clock['now'])
    monkeypatch.setattr(archive,'archive_hashes',hashes)
    with pytest.raises(TimeoutError): archive.complete_archive(out,contract,deadline=1.)
    assert not (out/'COMPLETE.json').exists()


def test_expiry_during_completion_marker_write_removes_marker(tiny,tmp_path,monkeypatch):
    archive,out,contract=make_archive(tiny,tmp_path)
    (out/'COMPLETE.json').unlink()
    (out/'aggregate.json').unlink()
    clock={'now':0.}
    original=archive.atomic_json
    def write(path,value):
        original(path,value)
        if path.name=='COMPLETE.json': clock['now']=2.
    monkeypatch.setattr(archive.time,'monotonic',lambda:clock['now'])
    monkeypatch.setattr(archive,'atomic_json',write)
    with pytest.raises(TimeoutError): archive.complete_archive(out,contract,deadline=1.)
    assert not (out/'COMPLETE.json').exists()


def test_replay_uses_official_tolerant_volume_floor(tiny,monkeypatch):
    runner,archive,model,crop=tiny
    original_forward=model.forward
    def forward(*args):
        out=original_forward(*args)
        out['event_presence'].fill_(-10.)
        out['event_presence'][0,0]=10.
        return out
    def masks(out,points=None):
        dense=torch.full((1,4,*archive.RECIPE['crop']),-10.)
        dense[0,0,0,0,:10]=10.
        if points is None: return dense
        return dense.flatten(2).gather(2,points[:,None].expand(-1,4,-1))
    monkeypatch.setattr(model,'forward',forward)
    monkeypatch.setattr(model,'event_masks',masks)
    meta,arrays=runner.collect_crop(model,crop,.99999999,source='bmsr',sequence='T1c',selection_ordinal=0,crop_ordinal=0,
                                  device=torch.device('cpu'),deadline=time.monotonic()+60)
    assert [r['n_voxels'] for r in meta['detection_rows']]==[10]
    assert archive.reconstruct_crop(meta,arrays)==meta['summaries']
