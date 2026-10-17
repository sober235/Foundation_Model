"""Behavioral checks for an observational, validation-only A/U/R diagnostic."""
import importlib

import numpy as np
import pytest
import torch

from anatobind.aur import infer as I
from anatobind.aur.model import AnatoBindBrain

TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4), d_model=16,
            n_events=4, mask_dim=4, pixel_dim=8, dec_layers=1, dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False)


def test_observer_does_not_change_natural_window_predictions_or_binding():
    torch.manual_seed(4)
    model = AnatoBindBrain(**TINY).eval()
    image = np.random.default_rng(2).normal(size=(8, 18, 18)).astype('float32')
    affine = np.eye(4)
    args = (model, image, affine, (8, 16, 16), torch.device('cpu'))
    baseline = I.predict_volume(*args, batch_size=2)
    calls = []
    observed = I.predict_volume(*args, batch_size=2, observer=lambda **kw: calls.append(kw['window']))
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], observed[key])
    assert len(calls) == 4
    inst = np.zeros(image.shape, np.int32)
    inst[2:5, 6:10, 6:10] = 1
    ref = I.bind_instances(model, image, affine, inst, (8, 16, 16), torch.device('cpu'))
    binds = []
    obs = I.bind_instances(model, image, affine, inst, (8, 16, 16), torch.device('cpu'), observer=lambda **kw: binds.append(kw))
    np.testing.assert_array_equal(ref[0]['host_probs'], obs[0]['host_probs'])
    assert len(binds) == 1 and binds[0]['query'] == ref[0]['query']


def _row(source, sequence, case, patient, split='train'):
    return dict(source=source, sequence=sequence, case=case, patient=patient, split=split, u_supervised=True)


def test_selection_is_order_independent_distinct_patient_and_validation_only():
    D = importlib.import_module('anatobind.aur.diagnostics')
    rows = [_row(s, q, str(i), str(i // 2)) for s, q in [('bmsr', 'T1c'), ('isles', 'DWI'), ('pdgm', 'FLAIR')] for i in range(6)]
    saved = [{**r, 'u': {'gt': [{'ignore': False}]}} for r in rows]
    selected = D.select_cases(rows, saved)
    assert [D.identity(r) for r in selected] == [D.identity(r) for r in D.select_cases(rows[::-1], saved[::-1])]
    assert len(selected) == 6
    for s in ('bmsr', 'isles', 'pdgm'):
        assert len({r['patient'] for r in selected if r['source'] == s}) == 2
    bad = [{**r, 'split': 'test'} for r in rows]
    with pytest.raises(ValueError, match='test'):
        D.select_cases(bad, saved)
    with pytest.raises(ValueError, match='duplicate'):
        D.select_cases(rows + [rows[0]], saved)


def test_baseline_reproduction_rejects_changed_components_and_scores():
    D = importlib.import_module('anatobind.aur.diagnostics')
    row = dict(box=[0, 0, 0, 2, 2, 3], n_voxels=12, volume_mm3=12., score=.6)
    D.verify_detections([row], [row])
    D.verify_detections([{**row, 'score': .600001}], [row])
    for changed in ([], [{**row, 'n_voxels': 13}], [{**row, 'score': .6001}], [{**row, 'score': float('nan')}]):
        with pytest.raises(ValueError, match='reproduction'):
            D.verify_detections(changed, [row])


def test_window_measurement_separates_presence_loss_and_excludes_padding(tmp_path):
    D = importlib.import_module('anatobind.aur.diagnostics')
    instance = np.zeros((2, 2, 2), np.int32)
    instance[0, 0, 0] = 1
    audit = D.WindowAudit(instance, np.zeros_like(instance, bool), tmp_path)
    prob = torch.zeros((2, 2, 2, 2))
    prob[0, 0, 0, 0] = .9
    prob[1, 0, 1, 1] = .9
    valid = torch.ones((2, 2, 2))
    valid[1] = 0
    prob[:, 1] = .99  # outside image: must contribute neither union nor predicted volume
    audit(window=[(0, 2)] * 3, event_prob=prob, event_presence=torch.tensor([.2, .9]),
          entity_prob=torch.zeros((32, 2, 2, 2)), valid=valid, weight=torch.ones_like(valid))
    g = audit.window_rows[0]['gt'][0]
    assert g['at_0.3']['best_all_iou'] == 1.
    assert g['at_0.3']['best_kept_iou'] == 0.
    assert g['at_0.3']['best_all_query'] == 0
    assert g['at_0.3']['best_all_presence'] == pytest.approx(.2)
    assert g['at_0.3']['max_iou'] == 0.
    assert g['visible_fraction'] == 1.
    assert audit.window_rows[0]['at_0.3']['query_voxels'] == [1, 1]
    assert g['at_0.3']['all_query_iou'] == [1., 0.]
    cache = np.load(tmp_path / 'window_000.npz')
    restored = np.unpackbits(cache['all_query_hard_03_packed'], axis=1, count=8).reshape(2, 2, 2, 2)
    assert int(restored.sum()) == 2 and restored[0, 0, 0, 0] == 1


def test_empty_presence_windows_still_contribute_weight_and_observer_reconstruction(tmp_path):
    D = importlib.import_module('anatobind.aur.diagnostics')
    inst = np.ones((1, 1, 1), np.int32)
    audit = D.WindowAudit(inst, np.zeros_like(inst, bool), tmp_path)
    for p, presence in ((.8, .9), (.9, .1)):
        audit(window=[(0, 1)] * 3, event_prob=torch.tensor([[[[p]]]]), event_presence=torch.tensor([presence]),
              entity_prob=torch.zeros((32, 1, 1, 1)), valid=torch.ones((1, 1, 1)), weight=torch.ones((1, 1, 1)))
    audit.verify_probability(np.array([[[.4]]], np.float32))
    assert audit.window_rows[1]['n_present'] == 0
    with pytest.raises(ValueError, match='probability'):
        audit.verify_probability(np.array([[[.8]]], np.float32))


def test_provenance_refuses_test_split_and_changed_checkpoint(tmp_path):
    D = importlib.import_module('anatobind.aur.diagnostics')
    from anatobind.aur.ssl.checkpoint import file_sha256
    samples, ckpt = tmp_path / 'samples.json', tmp_path / 'weights.pt'
    samples.write_text('[]')
    ckpt.write_bytes(b'frozen')
    run = dict(split='val', samples_sha256=file_sha256(samples), checkpoint_sha256=file_sha256(ckpt))
    D.verify_provenance(run, samples, ckpt)
    with pytest.raises(ValueError, match='validation'):
        D.verify_provenance({**run, 'split': 'test'}, samples, ckpt)
    ckpt.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        D.verify_provenance(run, samples, ckpt)


def test_component_intersection_reports_real_voxels_not_bbox_containment(tmp_path):
    D = importlib.import_module('anatobind.aur.diagnostics')
    inst = np.zeros((1, 3, 3), np.int32)
    inst[0, 1, 1] = 1
    audit = D.WindowAudit(inst, np.zeros_like(inst, bool), tmp_path)
    component = np.ones_like(inst)
    component[0, 1, 1] = 0  # surrounding ring has enclosing bbox, but no lesion voxel
    result = audit.finish(np.zeros(inst.shape), component, [dict(instance=1, score=.8, n_voxels=8)], .7)
    assert result['gt'][0]['components'] == []
    assert result['gt'][0]['threshold_fn'] == 1


def test_anatomy_diagnostic_excludes_padded_voxels():
    D = importlib.import_module('anatobind.aur.diagnostics')
    pred = np.array([2, 2, 2])
    true = np.array([2, 0, 0])
    assert D.valid_host_dice(pred, true, np.array([1, 0, 0], bool))['host_macro'] == 1.


def test_replay_rejects_missing_relation_instances_and_changed_gt():
    D = importlib.import_module('anatobind.aur.diagnostics')
    old = dict(instance=1, r=2, query_iou=.2, r_probs=[0., 0., 1.], in_crop_share=1.)
    bound = dict(instance=1, host=2, query_iou=.2, host_probs=np.array([0., 0., 1.]), in_crop_share=1.)
    D.verify_bindings([bound], [old])
    with pytest.raises(ValueError, match='reproduction'):
        D.verify_bindings([], [old])
    gt = dict(instance=1, box=[0, 0, 0, 1, 1, 10], n_voxels=10, volume_mm3=10.)
    D.verify_truth([gt], [gt])
    with pytest.raises(ValueError, match='reproduction'):
        D.verify_truth([{**gt, 'ignore': True}], [gt])
