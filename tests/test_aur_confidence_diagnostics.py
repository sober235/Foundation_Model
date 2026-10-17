"""Raw child scores must measure their own voxels, with exact frozen replay."""
from copy import deepcopy
import importlib
import hashlib
import json

import numpy as np
import pytest
import torch

from anatobind.aur.query_decoder import QueryCCDecoder


def audit(shape, voxel_mm3=10.):
    cls = importlib.import_module('anatobind.aur.confidence_diagnostics').ResidualConfidenceAudit
    return cls(shape, voxel_mm3, workers=1)


def feed(obj, p, *, replay=False, window=None, valid=None, presence=None):
    p = torch.as_tensor(p)
    fn = obj.observe_replay if replay else obj.observe_first
    fn(window=window or [(0, n) for n in p.shape[1:]], event_prob=p,
       event_presence=torch.ones(len(p)) if presence is None else torch.as_tensor(presence, dtype=torch.float32),
       valid=torch.ones(p.shape[1:]) if valid is None else torch.as_tensor(valid, dtype=torch.float32),
       entity_prob=None, weight=None)


def split_prob():
    p = np.zeros((2, 1, 1, 9), np.float32)
    p[0, 0, 0, 3:6] = .95
    p[1, 0, 0] = [.6, .7, .8, .9, .9, .9, .7, .6, .6]
    return p


def test_children_use_their_own_values_not_inherited_parent_mean():
    p = split_prob()
    fused = np.array([[[.4, .5, .6, .95, .95, .95, .3, .4, .5]]], np.float32)
    with audit((1, 1, 9)) as a:
        feed(a, p)
        inst, rows, stats = a.decode(fused)
        before_map, before_rows, before_prob = inst.copy(), deepcopy(rows), p.copy()
        assert inst.tolist() == [[[2, 2, 2, 1, 1, 1, 3, 3, 3]]]
        feed(a, p, replay=True)
        result = a.finish_replay()
        children = {r['instance']: r for r in result['children']}
        assert children[2]['parent_score'] == pytest.approx(6.7 / 9, abs=1e-7)
        assert children[2]['child_raw_mean'] == pytest.approx(.7, abs=1e-7)
        assert children[3]['child_raw_mean'] == pytest.approx(1.9 / 3, abs=1e-7)
        assert children[2]['child_fused_mean'] == pytest.approx(.5, abs=1e-7)
        assert children[3]['child_fused_mean'] == pytest.approx(.4, abs=1e-7)
        parent = next(r for r in result['parents'] if r['rank'] == 2)
        assert parent['retained_voxels'] == 6 and parent['removed_voxels'] == 3
        assert parent['retained_raw_mean'] == pytest.approx(4 / 6, abs=1e-7)
        assert parent['removed_raw_mean'] == pytest.approx(.9, abs=1e-7)
        np.testing.assert_array_equal(inst, before_map)
        np.testing.assert_array_equal(p, before_prob)
        assert rows == before_rows and stats['final_instances'] == 3
        assert result['checks']['first_windows'] == result['checks']['replay_windows'] == 1
        assert result['checks']['present_query_planes'] == 2


def test_full_retention_uses_original_float32_values_with_float64_reduction():
    p = np.array([[[[.300001, .700003, .900009]]]], np.float32)
    expected = sum(float(v) for v in p.ravel()) / 3
    with audit((1, 1, 3)) as a:
        feed(a, p)
        _, rows, _ = a.decode(p[0])
        feed(a, p, replay=True)
        r = a.finish_replay()
        assert r['children'][0]['child_raw_mean'] == expected
        assert r['children'][0]['parent_score'] == rows[0]['score'] == expected
        assert r['parents'][0]['removed_raw_mean'] is None
        assert expected != float(p.mean(dtype=np.float32))


def test_negative_window_padding_and_valid_hole_do_not_contribute_to_means():
    p = np.full((1, 2, 1, 5), .99, np.float32)
    p[0, 1, 0] = [.8, .8, .95, .7, .7]
    valid = np.ones((2, 1, 5), np.float32);valid[1, 0, 2] = 0
    window = [(-1, 1), (0, 1), (0, 5)]
    with audit((1, 1, 5)) as a:
        feed(a, p, window=window, valid=valid)
        inst, _, _ = a.decode(np.full((1, 1, 5), .4, np.float32))
        feed(a, p, replay=True, window=window, valid=valid)
        children = a.finish_replay()['children']
        assert inst.tolist() == [[[1, 1, 0, 2, 2]]]
        assert [r['n_voxels'] for r in children] == [2, 2]
        assert [r['child_raw_mean'] for r in children] == [float(np.float32(.8)), float(np.float32(.7))]


def test_same_query_in_two_windows_is_measured_from_its_actual_window():
    with audit((1, 1, 6)) as a:
        p = np.full((1, 1, 1, 3), .8, np.float32)
        q = np.full((1, 1, 1, 3), .6, np.float32)
        wa, wb = [(0, 1), (0, 1), (0, 3)], [(0, 1), (0, 1), (3, 6)]
        feed(a, p, window=wa);feed(a, q, window=wb)
        a.decode(np.zeros((1, 1, 6), np.float32))
        feed(a, p, replay=True, window=wa);feed(a, q, replay=True, window=wb)
        r = a.finish_replay()
        assert [x['child_raw_mean'] for x in r['children']] == [float(np.float32(.8)), float(np.float32(.6))]
        assert r['checks']['first_windows'] == 2


@pytest.mark.parametrize('changed', ['prob', 'presence', 'valid'])
def test_replay_rejects_any_changed_observed_values(changed):
    p = np.full((1, 1, 1, 4), .8, np.float32)
    with audit((1, 1, 4)) as a:
        feed(a, p);a.decode(p[0])
        kw = {}
        if changed == 'prob':p = p.copy();p[..., 0] = .80001
        if changed == 'presence':kw['presence'] = [.9]
        if changed == 'valid':kw['valid'] = [[[1, 1, 1, 0]]]
        with pytest.raises(ValueError):feed(a, p, replay=True, **kw)


def test_duplicate_first_window_is_rejected():
    p = np.full((1, 1, 1, 4), .8, np.float32)
    with audit((1, 1, 4)) as a:
        feed(a, p)
        with pytest.raises(ValueError):feed(a, p)


def test_missing_and_duplicate_replay_windows_are_rejected():
    p = np.full((1, 1, 1, 4), .8, np.float32)
    with audit((1, 1, 4)) as a:
        feed(a, p);a.decode(p[0])
        with pytest.raises(ValueError):a.finish_replay()
    with audit((1, 1, 4)) as a:
        feed(a, p);a.decode(p[0]);feed(a, p, replay=True)
        with pytest.raises(ValueError):feed(a, p, replay=True)


@pytest.mark.parametrize('dtype', [np.float16, np.float64])
def test_quantized_or_wrong_precision_probabilities_are_rejected(dtype):
    with audit((1, 1, 2)) as a:
        with pytest.raises((ValueError, TypeError)):feed(a, np.full((1, 1, 1, 2), .8, dtype))


def test_nonfinite_present_query_is_rejected():
    p = np.array([[[[.8, np.nan]]]], np.float32)
    with audit((1, 1, 2)) as a:
        with pytest.raises(ValueError):feed(a, p)


def test_no_present_query_completes_with_empty_measurements():
    p = np.full((1, 1, 1, 2), .8, np.float32)
    with audit((1, 1, 2)) as a:
        feed(a, p, presence=[.5]);inst, rows, _ = a.decode(p[0])
        feed(a, p, presence=[.5], replay=True)
        r = a.finish_replay()
        assert not inst.any() and rows == [] and r['children'] == [] and r['parents'] == []


def test_window_integrity_digest_survives_JSON_key_normalization():
    p = np.full((11, 1, 1, 3), .6, np.float32);p[10] = .8
    presence = np.zeros(11, np.float32);presence[[2, 10]] = 1
    with audit((1, 1, 3)) as a:
        feed(a, p, presence=presence);a.decode(p[10]);feed(a, p, presence=presence, replay=True)
        result = a.finish_replay()
        wire = json.loads(json.dumps(result['window_hashes']))
        expected = hashlib.sha256(json.dumps(wire, sort_keys=True).encode()).hexdigest()
        assert result['checks']['window_hashes_sha256'] == expected


def test_real_model_observer_matches_production_decoder_and_preserves_A_R():
    from anatobind.aur import infer as I
    from anatobind.aur.model import AnatoBindBrain
    torch.manual_seed(19)
    model = AnatoBindBrain(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4),
        patch=(2, 4, 4), d_model=16, n_events=4, mask_dim=4, pixel_dim=8, dec_layers=1,
        dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False).eval()
    with torch.no_grad():
        model.events.presence.weight.zero_();model.events.presence.bias.fill_(2)
        model.masks.embed_full[-1].weight.zero_();model.masks.embed_full[-1].bias.zero_()
    image = np.random.default_rng(8).normal(size=(8, 18, 18)).astype(np.float32)
    args = model, image, np.eye(4), (8, 16, 16), torch.device('cpu')
    gt = np.zeros(image.shape, np.int32);gt[2:5, 6:9, 6:9] = 1
    before_r = I.bind_instances(model, image, np.eye(4), gt, (8, 16, 16), torch.device('cpu'))
    expected = I.predict_volume(*args, batch_size=2, decoder_profile='query_cc_v1', voxel_mm3=1.)
    with audit(image.shape, voxel_mm3=1.) as a:
        first = I.predict_volume(*args, batch_size=2, observer=a.observe_first)
        inst, rows, stats = a.decode(first['lesion_prob'])
        second = I.predict_volume(*args, batch_size=2, observer=a.observe_replay)
        result = a.finish_replay()
    np.testing.assert_array_equal(inst, expected['query_instances'])
    assert rows == expected['query_detections'] and stats == expected['decoder_stats']
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first[key], expected[key])
    assert result['checks']['first_windows'] == expected['n_windows'] == 4
    after_r = I.bind_instances(model, image, np.eye(4), gt, (8, 16, 16), torch.device('cpu'))
    assert before_r[0]['query'] == after_r[0]['query'] and before_r[0]['window'] == after_r[0]['window']
    np.testing.assert_array_equal(before_r[0]['host_probs'], after_r[0]['host_probs'])


@pytest.mark.parametrize('fault', ['count', 'containment'])
def test_malformed_upstream_child_geometry_stops_the_audit(monkeypatch, fault):
    original = QueryCCDecoder.finish
    def faulty(self):
        inst, rows, stats = original(self)
        if fault == 'count':rows[0]['n_voxels'] += 1
        else:inst[0, 0, 0] = 0;inst[0, 1, 1] = 1
        return inst, rows, stats
    monkeypatch.setattr(QueryCCDecoder, 'finish', faulty)
    p = np.full((1, 1, 3, 3), .8, np.float32);p[0, 0, 1, 1] = 0
    with audit((1, 3, 3)) as a:
        feed(a, p)
        with pytest.raises(ValueError):a.decode(np.zeros((1, 3, 3), np.float32))
