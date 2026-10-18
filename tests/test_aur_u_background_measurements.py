"""Synthetic checks for paired allocation and query-weighted tangent reductions."""
import json

import numpy as np
import pytest
import torch

from scripts.aur_u_background_measurements import (
    aggregate_reports, build_rois, paired_points, response_report,
)


def _dense_case():
    shape = (1, 3, 8)
    instance = np.zeros(24, np.int64)
    instance[:2], instance[2:4] = 1, 2
    valid, weight = np.ones(24, np.float32), np.ones(24, np.float32)
    valid[22], weight[4] = 0, 0
    detection = np.zeros(24, np.int32)
    components = [(11, 0, 'FP', [2, 4, 5]), (19, 0, 'FP', [6, 7, 8]),
                  (31, 1, 'TP', [9]), (41, 2, 'FP', [10])]
    rows, outcomes = [], []
    for identity, query, status, loc in components:
        detection[loc] = identity
        origin = dict(query=query, parent_score=.9)
        rows.append(dict(instance=identity, detector_origin=origin))
        outcomes.append(dict(instance=identity, status=status, detector_origin=origin))
    pixels = torch.stack((torch.arange(24).float(), torch.ones(24)))
    points = np.array([[[5, 11], [6, 12]], [[7, 11], [8, 12]]], np.int64)
    matches = [(torch.tensor([0, 1]), torch.tensor([0, 1])) for _ in range(2)]
    args = (detection.reshape(shape), rows, outcomes, instance.reshape(shape),
            valid.reshape(shape), weight.reshape(shape), pixels, points, matches)
    return args


def _report_case(scale=1., source='bmsr', sequence='T1c'):
    args = _dense_case()
    rows, features, indices = build_rois(*args)
    jvp = np.zeros((2, 2, 2, 2, 3, 2), np.float32)
    for draw in range(2):
        for arm in range(2):
            jvp[draw, :, arm, :, :, 0] = -(arm + 1) * (draw + 1) * scale
    defined = np.ones((2, 2, 2, 2), bool)
    reference = np.zeros((len(rows), 2, 2, 2), np.float64)
    for i, (row, loc) in enumerate(zip(rows, indices)):
        if len(loc):
            # Independent dense projection, rather than reusing ROI means.
            dense = args[6][:, loc].double().numpy()
            reference[i] = np.einsum('oanc,cv->oanv',
                jvp[row['draw'], :, :, :, row['query']].astype(np.float64), dense).mean(-1)
    meta = dict(source=source, sequence=sequence, n_instances=2,
                a_supervised=True, u_supervised=True, roi_rows=rows,
                patient='PRIVATE_SENTINEL', arbitrary='PRIVATE_SENTINEL')
    arrays = dict(roi_features=features, roi_reference_responses=reference)
    analysis = dict(summary=dict(draws=[dict(draw=i) for i in range(2)]),
                    numerics=dict(event_jvp=jvp, direction_defined=defined,
                                  a_loss_response=np.zeros((2, 2, 2, 2), np.float64),
                                  a_query_loss_response=np.zeros((2, 2, 2, 2, 3), np.float64)))
    return meta, arrays, analysis


def test_sampler_preserves_original_focus_nonring_and_restores_uniform():
    valid = torch.ones((5, 5, 5))
    instance = torch.zeros_like(valid, dtype=torch.long)
    instance[1:4, 1:4, 1:4] = 1
    points, summary = paired_points(valid, instance, valid, n=128, seed=7)
    uniform = torch.multinomial(valid.flatten() + 1e-12, 128, replacement=True,
                                generator=torch.Generator().manual_seed(7)).numpy()
    assert points.dtype == np.int64 and points.shape == (2, 128)
    np.testing.assert_array_equal(points[0, :64], points[1, :64])
    np.testing.assert_array_equal(points[0, 89:], points[1, 89:])
    np.testing.assert_array_equal(points[1, 64:89], uniform[64:89])
    assert summary['focus_slots'] == 64 and summary['ring_slots'] == 25
    assert summary['arms'][1]['positive_points'] > 64
    assert summary['focus_slots_equal'] and summary['nonring_slots_equal']
    assert summary['restored_uniform_equal']


@pytest.mark.parametrize('kind', ['no_instance', 'no_ring', 'no_valid'])
def test_sampler_keeps_empty_and_no_ring_arms_identical(kind):
    valid = torch.ones((2, 2, 2))
    instance = torch.zeros_like(valid, dtype=torch.long)
    if kind == 'no_ring':
        instance[:] = 1
    elif kind == 'no_valid':
        valid.zero_()
    points, summary = paired_points(valid, instance, torch.ones_like(valid), n=20, seed=2)
    np.testing.assert_array_equal(points[0], points[1])
    assert summary['ring_slots'] == 0
    if kind == 'no_valid':
        assert summary['arms'][0]['effective_points'] == 0


def test_sampler_quota_zero_uses_original_slot_rule_and_counts_ignored_duplicates():
    instance = torch.arange(27).reshape(3, 3, 3)
    valid = torch.ones_like(instance).float()
    weight = torch.zeros_like(valid)
    points, summary = paired_points(valid, instance, weight, n=10, seed=4)
    assert summary['focus_slots'] == 0 and summary['ring_slots'] == 2
    assert summary['arms'][0]['ignored_points'] == 10
    assert summary['arms'][0]['effective_points'] == 0
    assert summary['arms'][0]['duplicate_points'] == 10 - np.unique(points[0]).size
    assert points[0, 0] == points[0, 1] == 0


def test_rois_union_matched_fp_background_exclude_both_arms_and_retain_empty():
    rows, features, indices = build_rois(*_dense_case())
    selected = [(r, f, loc) for r, f, loc in zip(rows, features, indices)
                if r['draw'] == 0]
    by = {(r['panel'], r.get('component'), r['query']): (r, f, loc)
          for r, f, loc in selected}
    r, f, loc = by['fp_all_query', None, 0]
    np.testing.assert_array_equal(loc, [5, 6, 7, 8])
    np.testing.assert_allclose(f, [6.5, 1])
    assert r['n_other_gt_voxels'] == 1 and r['n_ignored_voxels'] == 1
    assert r['n_assigned_target_voxels'] == 0
    np.testing.assert_array_equal(by['fp_unsampled_query', None, 0][2], [7, 8])
    empty, zero, loc = by['fp_unsampled_component', 11, 0]
    assert empty['n_voxels'] == 0 and loc.size == 0
    np.testing.assert_array_equal(zero, [0, 0])
    assert not any(r['query'] == 2 for r in rows)
    assert not any(r['query'] == 1 and r['panel'].startswith('fp_') for r in rows)
    np.testing.assert_array_equal(by['target_all', None, 0][2], [0, 1])
    np.testing.assert_array_equal(by['target_fp_query', None, 0][2], [0, 1])
    np.testing.assert_array_equal(by['target_tp_query', None, 1][2], [2, 3])


def test_roi_features_use_float64_reduction_before_float32_export():
    args = list(_dense_case())
    args[6][0, [6, 7, 8]] = torch.tensor([1e8, 1., -1e8])
    rows, features, _ = build_rois(*args)
    index = next(i for i, r in enumerate(rows) if r['draw'] == 0
                 and r['panel'] == 'fp_all_component' and r['component'] == 19)
    assert features.dtype == np.float32
    assert features[index, 0] == np.float32(1 / 3)


def test_response_uses_query_union_and_keeps_component_and_empty_denominators():
    report = response_report(*_report_case())
    panel = report['draws'][0]['panels']
    assert panel['fp_all_query']['responses'][0][0][0]['mean'] == -6.5
    assert panel['fp_all_component']['responses'][0][0][0]['mean'] == -6.
    assert panel['fp_all_query']['paired_differences'][0][0]['mean'] == -6.5
    assert panel['fp_unsampled_component']['empty_rows'] == 1
    stat = panel['fp_unsampled_component']['responses'][0][0][0]
    assert stat['n'] == 1 and stat['undefined'] == 1 and stat['negative'] == 1
    assert 'PRIVATE_SENTINEL' not in json.dumps(report)


def test_response_rejects_streamed_reference_disagreement_and_nonfinite():
    meta, arrays, analysis = _report_case()
    arrays['roi_reference_responses'][0, 0, 0, 0] += .1
    with pytest.raises(ValueError):
        response_report(meta, arrays, analysis)
    meta, arrays, analysis = _report_case()
    arrays['roi_features'][0, 0] = np.nan
    with pytest.raises(ValueError):
        response_report(meta, arrays, analysis)


def test_zero_directions_and_no_support_have_explicit_undefined_denominators():
    meta, arrays, analysis = _report_case()
    analysis['numerics']['direction_defined'][:, :, 1, 1] = False
    analysis['numerics']['event_jvp'][:, :, 1, 1] = 0
    arrays['roi_reference_responses'][:, :, 1, 1] = 0
    report = response_report(meta, arrays, analysis)
    panel = report['draws'][0]['panels']['fp_all_query']
    stat = panel['responses'][0][1][1]
    assert stat['n'] == 0 and stat['undefined'] == 1 and stat['mean'] is None
    diff = panel['paired_differences'][0][1]
    assert diff['n'] == 0 and diff['fraction_negative'] is None
    empty_meta, empty_arrays, analysis = _report_case()
    empty_meta['roi_rows'] = []
    empty_arrays['roi_features'] = empty_arrays['roi_features'][:0]
    empty_arrays['roi_reference_responses'] = empty_arrays['roi_reference_responses'][:0]
    empty = response_report(empty_meta, empty_arrays, analysis)
    assert empty['draws'][0]['panels']['fp_all_query']['rows'] == 0
    assert empty['draws'][0]['panels']['target_all']['responses'][0][0][0]['mean'] is None


def test_aggregate_retains_source_crop_draw_denominators_and_drops_metadata():
    first = response_report(*_report_case())
    second = response_report(*_report_case(scale=2))
    second['private_crop'] = 'PRIVATE_SENTINEL'
    second['draws'][0]['private_query'] = 'PRIVATE_SENTINEL'
    other = response_report(*_report_case(source='isles', sequence='DWI'))
    result = aggregate_reports([first, second, other])
    assert result['draws_are_repeated_measurements'] is True
    source = result['sources'][0]
    assert source['source'] == 'bmsr' and source['crops'] == 2 and source['crop_draws'] == 4
    panel = source['panels']['fp_all_query']
    assert panel['rows'] == 4 and panel['supported_crop_draws'] == 4
    pooled = panel['crop_draw_responses'][0][0][0]
    assert pooled['n'] == 4 and pooled['mean'] == pytest.approx(-14.625)
    crop = panel['crop_mean_responses'][0][0][0]
    assert crop['n'] == 2 and crop['mean'] == pytest.approx(-14.625)
    by_draw = panel['by_draw'][0]['responses'][0][0][0]
    assert by_draw['n'] == 2 and by_draw['mean'] == pytest.approx(-9.75)
    assert 'PRIVATE_SENTINEL' not in json.dumps(result)
    assert all(s['source'] in ('bmsr', 'isles', 'pdgm') for s in result['sources'])


@pytest.mark.parametrize('field,value', [('source', 'BMSR'), ('sequence', 'PRIVATE_SENTINEL')])
def test_report_and_aggregate_reject_unknown_source_enum(field, value):
    meta, arrays, analysis = _report_case()
    meta[field] = value
    with pytest.raises(ValueError):
        response_report(meta, arrays, analysis)
    report = response_report(*_report_case())
    report[field] = value
    with pytest.raises(ValueError):
        aggregate_reports([report])


def test_roi_and_response_reject_unknown_outcome_and_panel_enums():
    args = list(_dense_case())
    args[2][0]['status'] = 'PRIVATE_SENTINEL'
    with pytest.raises(ValueError):
        build_rois(*args)
    meta, arrays, analysis = _report_case()
    meta['roi_rows'][0]['panel'] = 'PRIVATE_SENTINEL'
    with pytest.raises(ValueError):
        response_report(meta, arrays, analysis)


def test_component_composition_counts_survive_private_and_public_reductions():
    report = response_report(*_report_case())
    composition = report['draws'][0]['panels']['fp_all_query']['component_composition']
    assert composition['n_component_voxels'] == 6
    assert composition['n_background_voxels'] == 4
    assert composition['n_other_gt_voxels'] == 1 and composition['n_ignored_voxels'] == 1
    public = aggregate_reports([report])['sources'][0]['panels']['fp_all_query']
    assert public['component_composition']['n_other_gt_voxels'] == 2
    assert public['component_composition']['n_ignored_voxels'] == 2


def test_anatomy_paired_response_and_positive_query_support_exclude_disabled_crops():
    meta, arrays, analysis = _report_case()
    response = analysis['numerics']['a_query_loss_response']
    response[:, :, 0] = np.array([1., -2., 0.])
    response[:, :, 1] = np.array([2., -1., 0.])
    analysis['numerics']['a_loss_response'][:] = response.sum(-1)
    analysis['summary']['draws'] = [dict(draw=d, a_query_support=[
        dict(query=q, enabled=True, present=q == 0, effective_positive_points=2 if q == 0 else 0)
        for q in range(3)]) for d in range(2)]
    enabled = response_report(meta, arrays, analysis)
    meta['a_supervised'] = False
    disabled = response_report(meta, arrays, analysis)
    assert disabled['draws'][0]['a_loss_response'][0][0][0] is None
    a_loss = aggregate_reports([enabled, disabled])['sources'][0]['a_loss']
    paired = a_loss['crop_draw_paired_differences'][0][0]
    assert paired['mean'] == 2 and paired['n'] == 2 and paired['undefined'] == 2
    positive = a_loss['positive_query_sign_counts'][0][0][0]
    assert positive['n'] == 2 and positive['positive'] == 2
    assert positive['undefined'] == 10


def test_rois_keep_target_rows_when_target_is_entirely_ignored():
    args = list(_dense_case())
    args[5].ravel()[:2] = 0
    rows, features, indices = build_rois(*args)
    i = next(i for i, r in enumerate(rows) if r['draw'] == 0
             and r['query'] == 0 and r['panel'] == 'target_all')
    assert rows[i]['n_voxels'] == 0 and indices[i].size == 0
    np.testing.assert_array_equal(features[i], [0, 0])


def test_aggregate_unsupported_panel_does_not_invent_zero_responses():
    meta, arrays, analysis = _report_case()
    meta['roi_rows'] = []
    arrays['roi_features'] = arrays['roi_features'][:0]
    arrays['roi_reference_responses'] = arrays['roi_reference_responses'][:0]
    report = response_report(meta, arrays, analysis)
    panel = aggregate_reports([report])['sources'][0]['panels']['fp_all_query']
    assert panel['supported_crop_draws'] == 0 and panel['empty_crop_draws'] == 2
    assert panel['crop_draw_responses'][0][0][0]['n'] == 0
    assert panel['crop_draw_responses'][0][0][0]['undefined'] == 2
    assert panel['crop_mean_responses'][0][0][0]['undefined'] == 1
