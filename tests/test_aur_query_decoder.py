"""GT-free query decoder regressions using hand-checked voxel fixtures."""
import json
import time

import numpy as np
import pytest
import torch

from anatobind.aur.query_decoder import DECODER_PROFILES, QueryCCDecoder, decoder_rules, filter_instances


def collect(prob, *, presence=None, valid=None, shape=None, window=None, voxel_mm3=10., workers=1):
    prob = torch.as_tensor(prob, dtype=torch.float32)
    shape = tuple(prob.shape[1:]) if shape is None else shape
    window = [(0, n) for n in prob.shape[1:]] if window is None else window
    decoder = QueryCCDecoder(shape, voxel_mm3, workers=workers)
    feed(decoder, prob, presence=presence, valid=valid, window=window)
    return decoder


def feed(decoder, prob, *, presence=None, valid=None, window=None):
    prob = torch.as_tensor(prob, dtype=torch.float32)
    decoder(window=window or [(0, n) for n in prob.shape[1:]], event_prob=prob,
            event_presence=torch.ones(len(prob)) if presence is None else torch.tensor(presence),
            entity_prob=None, valid=torch.ones(prob.shape[1:]) if valid is None else torch.as_tensor(valid), weight=None)


def test_touching_queries_stay_separate_even_when_maximum_has_one_component():
    p = np.zeros((2, 1, 1, 6), np.float32)
    p[0, 0, 0, :3], p[1, 0, 0, 3:] = .8, .7
    inst, rows, stats = collect(p).finish()
    assert inst.tolist() == [[[1, 1, 1, 2, 2, 2]]]
    assert [r['n_voxels'] for r in rows] == [3, 3]
    assert [r['detector_origin']['query'] for r in rows] == [0, 1]
    assert stats['final_instances'] == 2


def test_nms_includes_exact_half_iou_and_records_earliest_suppressor():
    p = np.zeros((3, 1, 1, 10), np.float32)
    p[0, 0, 0, :6], p[1, 0, 0, 4:] = .9, .8
    p[2, 0, 0, 2:8] = .7
    decoder = collect(p)
    inst, rows, stats = decoder.finish()
    assert len(rows) == 2 and stats['suppressed_candidates'] == 1
    assert decoder.nms_trace == [{'rank': 3, 'suppressor_rank': 1, 'iou': .5}]
    assert inst.tolist() == [[[1, 1, 1, 1, 1, 1, 2, 2, 2, 2]]]


def test_nms_uses_masks_instead_of_equal_bounding_boxes():
    p = np.zeros((2, 1, 3, 3), np.float32)
    p[0, 0, np.arange(3), np.arange(3)] = .9
    p[1, 0, np.arange(3), np.arange(2, -1, -1)] = .8
    inst, rows, stats = collect(p).finish()
    assert stats['suppressed_candidates'] == 0
    assert len(rows) == 3 and np.count_nonzero(inst) == 5


def test_nms_compares_original_masks_before_ownership():
    p = np.zeros((3, 1, 1, 12), np.float32)
    p[0, 0, 0, :3], p[1, 0, 0, :], p[2, 0, 0, :8] = .9, .8, .7
    decoder = collect(p)
    inst, rows, stats = decoder.finish()
    assert stats['suppressed_candidates'] == 1
    assert decoder.nms_trace == [{'rank': 3, 'suppressor_rank': 2, 'iou': 2 / 3}]
    assert [r['n_voxels'] for r in rows] == [3, 9]


def test_valid_holes_split_components_before_floor_and_score():
    p = np.full((1, 1, 1, 5), .8, np.float32)
    p[..., 2] = .99
    inst, rows, stats = collect(p, valid=[[[1, 1, 0, 1, 1]]], voxel_mm3=5.).finish()
    assert inst.tolist() == [[[1, 1, 0, 2, 2]]]
    assert [r['score'] for r in rows] == [float(np.float32(.8))] * 2
    assert stats['raw_components'] == 2


def test_outside_volume_padding_cannot_bridge_or_inflate_component():
    p = np.full((1, 2, 1, 5), .99, np.float32)
    p[0, 1, 0] = [.8, .8, 0., .7, .7]
    inst, rows, stats = collect(p, shape=(1, 1, 5), window=[(-1, 1), (0, 1), (0, 5)]).finish()
    assert inst.tolist() == [[[1, 1, 0, 2, 2]]]
    assert [r['n_voxels'] for r in rows] == [2, 2]
    assert rows[0]['box'] == [0, 0, 0, 1, 1, 2]
    assert rows[0]['score'] == float(np.float32(.8))


def test_presence_is_strict_and_mask_threshold_is_inclusive():
    p = np.full((3, 1, 1, 1), .3, np.float32)
    inst, rows, stats = collect(p, presence=[.5, .50001, .1]).finish()
    assert len(rows) == 1 and rows[0]['detector_origin']['query'] == 1
    assert rows[0]['score'] == float(np.float32(.3))


def test_shared_physical_floor_applies_before_nms():
    p = np.zeros((1, 1, 1, 7), np.float32)
    p[0, 0, 0, :2], p[0, 0, 0, 4:] = .9, .8
    inst, rows, stats = collect(p, voxel_mm3=4.).finish()
    assert inst.tolist() == [[[0, 0, 0, 0, 1, 1, 1]]]
    assert stats['raw_components'] == 2 and stats['eligible_candidates'] == 1
    assert rows[0]['volume_mm3'] == 12.
    assert rows[0]['detector_origin']['component'] == 2


def test_score_uses_float64_mean_of_unquantized_float32_values():
    p = np.array([[[[.300001, .700003, .900009]]]], np.float32)
    _, rows, _ = collect(p).finish()
    expected = (float(p[0, 0, 0, 0]) + float(p[0, 0, 0, 1]) + float(p[0, 0, 0, 2])) / 3
    assert rows[0]['score'] == expected
    assert expected != float(p.mean(dtype=np.float32))


def test_same_query_number_in_different_windows_is_not_identity():
    decoder = QueryCCDecoder((1, 1, 6), 10.)
    feed(decoder, np.full((1, 1, 1, 3), .8), window=[(0, 1), (0, 1), (3, 6)])
    feed(decoder, np.full((1, 1, 1, 3), .8), window=[(0, 1), (0, 1), (0, 3)])
    inst, rows, _ = decoder.finish()
    assert inst.tolist() == [[[1, 1, 1, 2, 2, 2]]]
    assert [r['detector_origin']['window'][2] for r in rows] == [[0, 3], [3, 6]]


def test_cross_window_duplicate_is_suppressed():
    decoder = QueryCCDecoder((1, 1, 5), 10.)
    feed(decoder, [[[[.8, .8, .8]]]], window=[(0, 1), (0, 1), (0, 3)])
    feed(decoder, [[[[0., .7, .7, .7, 0.]]]])
    inst, rows, stats = decoder.finish()
    assert len(rows) == 1 and stats['suppressed_candidates'] == 1
    assert inst.tolist() == [[[1, 1, 1, 0, 0]]]


def test_ownership_splits_children_and_inherits_fixed_parent_score():
    p = np.zeros((2, 1, 1, 9), np.float32)
    p[0, 0, 0, 3:6] = .9
    p[1, 0, 0] = [.6, .7, .8, .8, .8, .8, .7, .6, .6]
    inst, rows, _ = collect(p, voxel_mm3=4.).finish()
    assert inst.tolist() == [[[2, 2, 2, 1, 1, 1, 3, 3, 3]]]
    assert rows[1]['score'] == rows[2]['score'] == float(p[1].mean(dtype=np.float64))
    assert [r['detector_origin']['residual_child'] for r in rows] == [1, 1, 2]
    assert rows[1]['detector_origin']['parent_voxels'] == 9
    assert rows[1]['detector_origin']['retained_fraction'] == 1 / 3


def test_tiny_fragments_are_not_backfilled_by_later_fully_occluded_candidate():
    p = np.zeros((3, 1, 1, 7), np.float32)
    p[0, 0, 0, 2:5], p[1, 0, 0, :], p[2, 0, 0, 0:3] = .9, .8, .7
    inst, rows, stats = collect(p, voxel_mm3=4.).finish()
    assert inst.tolist() == [[[0, 0, 1, 1, 1, 0, 0]]]
    assert len(rows) == 1 and stats['tiny_fragments'] == 2
    assert stats['fully_occluded_candidates'] == 1


def test_filter_compacts_ids_and_preserves_origins_without_mutation():
    inst = np.array([[[3, 0, 8, 8]]], np.int32)
    rows = [dict(instance=3, score=.2, detector_origin={'query': 4}),
            dict(instance=8, score=.9, detector_origin={'query': 7})]
    filtered, kept = filter_instances(inst, rows, .9)
    assert filtered.tolist() == [[[0, 0, 1, 1]]]
    assert kept == [dict(instance=1, score=.9, detector_origin={'query': 7})]
    assert rows[1]['instance'] == 8 and inst[0, 0, 2] == 8
    kept[0]['detector_origin']['query'] = 99
    assert rows[1]['detector_origin']['query'] == 7


def test_empty_candidates_and_empty_filter_return_int32_background():
    decoder = QueryCCDecoder((2, 3, 4), 1.)
    inst, rows, stats = decoder.finish()
    assert inst.dtype == np.int32 and not inst.any() and rows == []
    assert stats['final_instances'] == stats['raw_components'] == 0
    filtered, kept = filter_instances(inst, rows, .8)
    assert filtered.dtype == np.int32 and not filtered.any() and kept == []


def test_workers_completion_and_submission_order_do_not_change_priority(monkeypatch):
    import anatobind.aur.query_decoder as module
    original = module._query_candidates
    def delayed(*args, **kwargs):
        if args[1] == 0:
            time.sleep(.02)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, '_query_candidates', delayed)
    p = np.zeros((3, 1, 1, 9), np.float32)
    p[0, 0, 0, :3], p[1, 0, 0, 3:6], p[2, 0, 0, 6:] = .8, .8, .8
    serial = collect(p, workers=1).finish()
    parallel = collect(p, workers=4).finish()
    np.testing.assert_array_equal(serial[0], parallel[0])
    assert serial[1:] == parallel[1:]
    assert [r['detector_origin']['query'] for r in parallel[1]] == [0, 1, 2]


def test_rules_are_json_serializable_independent_and_unknown_profiles_fail():
    assert DECODER_PROFILES == ('legacy_max', 'query_cc_v1')
    rules = decoder_rules('query_cc_v1')
    assert json.loads(json.dumps(rules)) == rules
    rules['mask_threshold'] = 7
    assert decoder_rules('query_cc_v1')['mask_threshold'] == .3
    assert decoder_rules('legacy_max') != decoder_rules('query_cc_v1')
    with pytest.raises(ValueError, match='profile'):
        decoder_rules('unknown')


@pytest.mark.parametrize('shape,voxel_mm3,workers', [((1, 0, 2), 1., 1), ((1, 1, 2), 0., 1), ((1, 1, 2), 1., 0)])
def test_invalid_geometry_or_worker_limit_fails_early(shape, voxel_mm3, workers):
    with pytest.raises(ValueError):
        QueryCCDecoder(shape, voxel_mm3, workers=workers)


def test_negative_valid_values_are_excluded_like_zero():
    p = np.full((1, 1, 1, 3), .8, np.float32)
    inst, rows, _ = collect(p, valid=np.array([[[1., -1., 1.]]])).finish()
    assert inst.tolist() == [[[1, 0, 2]]]
    assert [r['n_voxels'] for r in rows] == [1, 1]


def test_all_absent_queries_do_not_create_candidates():
    inst, rows, stats = collect(np.ones((2, 1, 1, 3)), presence=[.2, .5]).finish()
    assert not inst.any() and rows == []
    assert stats['eligible_candidates'] == stats['raw_components'] == 0


def test_none_score_threshold_retains_all_instances_and_compacts_ids():
    inst = np.array([[[3, 0, 8]]], np.int32)
    rows = [dict(instance=3, score=.2, detector_origin={'query': 4}),
            dict(instance=8, score=.9, detector_origin={'query': 7})]
    filtered, kept = filter_instances(inst, rows, None)
    assert filtered.tolist() == [[[1, 0, 2]]]
    assert [r['instance'] for r in kept] == [1, 2]
    assert [r['detector_origin']['query'] for r in kept] == [4, 7]
