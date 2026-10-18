"""Original IDs, independent detection/training labels, and anonymous summaries."""
import copy
import json

import numpy as np
import pytest

from anatobind.aur.u_supervision_components import (
    aggregate_records, component_measurements, detection_outcomes,
)

A = [0, 0, 0, 4, 4, 4]
B = [10, 10, 10, 14, 14, 14]
C = [20, 20, 20, 24, 24, 24]


def det(i, box=A, score=.8, query=0):
    return dict(instance=i, box=box, score=score,
                detector_origin=dict(query=query, parent_score=score, retained_fraction=.5,
                                     private='PRIVATE'))


def test_score_first_matching_preserves_nonconsecutive_ids_and_inclusive_cutoff():
    rows = [det(2, score=.49), det(19, score=.5), det(70, B)]
    before = copy.deepcopy(rows)
    out = detection_outcomes([dict(instance=31, box=A)], rows, .5)
    assert [(r['instance'], r['status']) for r in out] == [(2, 'below_threshold'), (19, 'TP'), (70, 'FP')]
    assert out[1]['official_gt_index'] == 0
    assert out[1]['official_gt_instance'] == 31
    out[1]['detector_origin']['query'] = 99
    assert rows == before


def test_real_first_and_one_ignored_identity_excuses_exactly_one_detection():
    gt = [dict(instance=99, box=A, ignore=True), dict(instance=7, box=A),
          dict(instance=120, box=B, ignore=True)]
    out = detection_outcomes(gt, [det(4), det(11, B), det(23, [10, 10, 10, 15, 15, 15]), det(40, C)], .5)
    assert [r['status'] for r in out] == ['TP', 'ignored_excused', 'FP', 'FP']
    assert [(r['official_gt_index'], r['official_gt_instance']) for r in out] == [
        (1, 7), (2, None), (None, None), (None, None)]


def fixture():
    # Component 8 spans foreground, ignored and padding; 44 is unsampled background.
    dm = np.array([8, 8, 8, 8, 44, 0, 0], np.int32)
    gt = np.array([1, 2, 0, 0, 0, 1, 2], np.int32)
    valid = np.array([1, 1, 1, 0, 1, 1, 1], bool)
    weight = np.array([1, 1, 0, 0, 1, 1, 1], np.float32)
    rows = [det(8, query=0), det(44, B, query=1)]
    outcomes = [dict(instance=8, status='FP', official_gt_index=None, official_gt_instance=None,
                     detector_origin=rows[0]['detector_origin']),
                dict(instance=44, status='TP', official_gt_index=1, official_gt_instance=2,
                     detector_origin=rows[1]['detector_origin'])]
    draws = [dict(draw=0, points=np.array([0, 0, 1, 2, 3], np.int64),
                  query_target_instance=[1, None], u_supervised=True,
                  mask_logit_gradient=np.array([[2, -2, 3, 0, 0], [1, 2, 3, 0, 0]], np.float32)),
             dict(draw=1, points=np.array([1, 1], np.int64),
                  query_target_instance=[2, None], u_supervised=True,
                  mask_logit_gradient=np.array([[1, 2], [3, 4]], np.float32))]
    return dm, rows, gt, valid, weight, outcomes, draws


def measure():
    dm, rows, gt, valid, weight, outcomes, draws = fixture()
    return component_measurements(dm, rows, gt, valid, weight, outcomes=outcomes, draws=draws)


def test_dense_valid_denominators_best_iou_tie_and_padding_separation():
    first, second = measure()
    assert [first[k] for k in ('n_voxels', 'n_valid_voxels', 'n_foreground_voxels',
                              'n_background_voxels', 'n_ignored_voxels', 'n_padded_voxels')] == [4, 3, 2, 0, 1, 1]
    assert first['foreground_fraction'] == pytest.approx(2 / 3)
    assert first['ignored_fraction'] == pytest.approx(1 / 3)
    assert first['best_gt_instance'] == 1
    assert first['best_gt_intersection'] == 1
    assert first['best_gt_iou'] == pytest.approx(1 / 4)
    assert first['best_gt_dice'] == pytest.approx(2 / 5)
    assert second['best_gt_instance'] is None
    assert second['best_gt_iou'] == 0


def test_draws_keep_training_labels_independent_and_sum_duplicate_gradients():
    first, second = measure()
    d0, d1 = first['draws']
    assert first['status'] == 'FP' and d0['query_status'] == 'matched'
    assert second['status'] == 'TP' and second['draws'][0]['query_status'] == 'unmatched'
    assert second['draws'][0]['official_target_relation'] == 'unmatched_query'
    assert [d0[k] for k in ('sampled_occurrences', 'effective_sampled_occurrences',
                            'unique_sampled_voxels', 'coverage_fraction')] == [5, 3, 2, 1.]
    assert d0['gradient_unique']['n'] == 2
    assert d0['gradient_unique']['zero'] == 1
    assert d0['gradient_unique']['positive'] == 1
    assert d0['gradient_unique']['sum'] == 3
    assert d0['gradient_unique']['absolute_sum'] == 3
    assert d0['negative_target_gradient_unique']['sum'] == 3
    assert d1['gradient_unique']['sum'] == 3
    assert d1['target_intersection'] == 1
    assert second['draws'][0]['coverage_fraction'] == 0
    assert second['draws'][0]['gradient_unique']['n'] == 0
    assert second['draws'][0]['gradient_unique']['mean'] is None


def test_tp_with_different_target_and_unsupervised_query_are_explicit():
    dm, rows, gt, valid, weight, outcomes, draws = fixture()
    outcomes[0].update(status='TP', official_gt_index=1, official_gt_instance=2)
    draws[1]['u_supervised'] = False
    draws[1]['query_target_instance'] = [None, None]
    draws[1]['mask_logit_gradient'][:] = 0
    out = component_measurements(dm, rows, gt, valid, weight, outcomes=outcomes, draws=draws)
    assert out[0]['draws'][0]['official_target_relation'] == 'different_target'
    assert out[0]['draws'][1]['query_status'] == 'unsupervised'


def test_no_gt_no_detections_and_malformed_joins_or_gradients():
    assert detection_outcomes([], [], .5) == []
    assert component_measurements(np.zeros(4, np.int32), [], np.zeros(4, np.int32),
                                  np.ones(4), np.ones(4), outcomes=[], draws=[]) == []
    dm, rows, gt, valid, weight, outcomes, draws = fixture()
    with pytest.raises(ValueError):
        component_measurements(dm, rows, gt, valid, weight, outcomes=outcomes[:1], draws=draws)
    draws[0]['mask_logit_gradient'][0, 0] = np.nan
    with pytest.raises(ValueError):
        component_measurements(dm, rows, gt, valid, weight, outcomes=outcomes, draws=draws)


def record():
    components = measure()
    return dict(source='bmsr', sequence='T1c', row_ordinal=0, n_instances=2, n_queries=2,
                patient='PRIVATE', crop='PRIVATE', components=components,
                outcomes=[{k: c[k] for k in ('instance', 'status', 'official_gt_instance')} for c in components],
                draws=[dict(draw=i, query_target_instance=target, matches=[[0, target[0] - 1]],
                            losses=dict(total=1.), queries=[dict(query=0, presence_probability=.9),
                                                          dict(query=1, presence_probability=.8)],
                            stability={}) for i, target in enumerate(([1, None], [2, None]))])


def test_aggregate_anonymous_counts_weighting_and_matching_stability():
    r = record()
    # Query 0 produces two components, but should have weight one in per-query summaries.
    extra = copy.deepcopy(r['components'][0])
    extra['instance'] = 81
    extra['foreground_fraction'] = 0.
    r['components'].append(extra)
    r['outcomes'].append(dict(instance=81, status='FP', official_gt_instance=None))
    out = aggregate_records([r])
    assert 'PRIVATE' not in json.dumps(out, allow_nan=False)
    assert (out['n_rows'], out['n_crops'], out['n_queries'], out['n_draws']) == (1, 1, 2, 2)
    assert out['detection_counts'] == dict(TP=1, FP=2, ignored_excused=0, below_threshold=0)
    assert out['query_detection_cross_tab']['matched']['FP'] == 4
    assert out['query_detection_cross_tab']['unmatched']['TP'] == 2
    assert out['matching_stability']['query_assignment_comparisons'] == 2
    assert out['matching_stability']['query_assignment_changes'] == 1
    stats = out['dense_mask']['foreground_fraction']
    assert stats['component_weighted']['n'] == 3
    assert stats['query_weighted']['n'] == 2
    assert stats['component_weighted']['mean'] == pytest.approx(2 / 9)
    assert stats['query_weighted']['mean'] == pytest.approx(1 / 6)
    assert out['source_sequence']['bmsr/T1c']['n_crops'] == 1
    assert out['gt_present']['present']['n_crops'] == 1
    assert out['gt_present']['empty']['n_crops'] == 0


def test_aggregate_rejects_identity_keys_as_group_values_nonfinite_and_bad_joins():
    r = record()
    r['source'] = 'PRIVATE'
    with pytest.raises(ValueError):
        aggregate_records([r])
    r = record()
    r['components'][0]['foreground_fraction'] = np.inf
    with pytest.raises(ValueError):
        aggregate_records([r])
    r = record()
    r['components'][0]['instance'] = 99
    with pytest.raises(ValueError):
        aggregate_records([r])
    empty = aggregate_records([])
    assert empty['n_crops'] == 0 and empty['fp_per_crop'] is None
    assert empty['dense_mask']['foreground_fraction']['component_weighted']['mean'] is None


def test_zero_effective_coverage_and_unsupervised_stability_stay_undefined():
    dm, rows, gt, valid, weight, outcomes, draws = fixture()
    weight[dm == 44] = 0
    out = component_measurements(dm, rows, gt, valid, weight, outcomes=outcomes, draws=draws)
    assert out[1]['draws'][0]['coverage_fraction'] is None
    assert out[1]['draws'][0]['coverage_denominator'] == 0
    r = record()
    for draw in r['draws']:
        draw['u_supervised'] = False
        draw['query_target_instance'] = [None, None]
    for c in r['components']:
        for d in c['draws']:
            d['query_status'] = 'unsupervised'
            d['target_instance'] = None
            d['official_target_relation'] = 'unsupervised'
    summary = aggregate_records([r])
    assert summary['matching_stability']['target_assignment_comparisons'] == 0
    assert summary['matching_stability']['target_assignment_change_fraction'] is None
    assert summary['matching_stability']['unassigned_gt_draws'] == 0
    assert summary['matching_stability']['unsupervised_draws'] == 2


def test_stability_checks_supplied_counts_and_reports_zero_weight_targets():
    r = record()
    r['matching_stability'] = dict(n_instances=2, n_queries=2, u_supervised=True,
        capacity_overflow=0, assignment_denominator=2, first_matched=1, second_matched=1,
        both_matched=0, same_query=0, different_query=0, first_only=1, second_only=1, neither=0,
        first_zero_positive_targets=1, second_zero_positive_targets=1,
        first_zero_effective_positive_targets=2, second_zero_effective_positive_targets=1)
    summary = aggregate_records([r])['matching_stability']
    assert summary['assignment_denominator'] == 2
    assert summary['first_only'] == 1 and summary['second_only'] == 1
    assert summary['zero_effective_positive_target_draws'] == 3
    assert summary['target_point_count_denominator'] == 4
    r['matching_stability']['same_query'] = 2
    with pytest.raises(ValueError):
        aggregate_records([r])


def test_components_reject_duplicate_training_targets_and_changed_origin():
    dm, rows, gt, valid, weight, outcomes, draws = fixture()
    draws[0]['query_target_instance'] = [1, 1]
    with pytest.raises(ValueError):
        component_measurements(dm, rows, gt, valid, weight, outcomes=outcomes, draws=draws)
    dm, rows, gt, valid, weight, outcomes, draws = fixture()
    outcomes[0]['detector_origin'] = dict(query=7)
    with pytest.raises(ValueError):
        component_measurements(dm, rows, gt, valid, weight, outcomes=outcomes, draws=draws)


def test_assigned_overlap_is_undefined_without_assigned_target():
    _, second = measure()
    assert second['draws'][0]['target_dice'] is None
    assert second['draws'][0]['target_iou'] is None
    dm, rows, gt, valid, weight, outcomes, draws = fixture()
    draws[0]['u_supervised'] = False
    draws[0]['query_target_instance'] = [None, None]
    draws[0]['mask_logit_gradient'][:] = 0
    out = component_measurements(dm, rows, gt, valid, weight, outcomes=outcomes, draws=draws)
    assert out[0]['draws'][0]['negative_target_gradient_unique']['n'] == 0


def test_empty_gt_detections_and_capacity_overflow_are_counted():
    assert detection_outcomes([], [det(21)], .8)[0]['status'] == 'FP'
    r = dict(source='isles', sequence='DWI', n_instances=3, n_queries=1, components=[], outcomes=[],
             draws=[dict(draw=0, query_target_instance=[1]), dict(draw=1, query_target_instance=[2])])
    out = aggregate_records([r])
    assert out['n_rows'] is None and out['row_count_defined'] is False
    assert out['matching_stability']['capacity_overflow_gt_draws'] == 4
    assert out['matching_stability']['unassigned_gt_draws'] == 4
    assert out['matching_stability']['target_assignment_comparisons'] == 3
    assert out['matching_stability']['target_assignment_changes'] == 2


def test_actual_loss_probe_schema_has_losses_and_normalization_in_aggregate():
    import torch
    from anatobind.aur.u_supervision_diagnostics import probe_event_loss, compare_draw_assignments
    probes = []
    for i in range(2):
        p = probe_event_loss(torch.ones(2), torch.ones(2, 3), torch.zeros(3, dtype=torch.long),
                             torch.ones(3), torch.tensor([0, 0, 1]), 0)
        p['draw'] = i
        probes.append(p)
    r = dict(source='bmsr', sequence='T1c', row_ordinal=0, n_instances=0, n_queries=2,
             draws=probes, matching_stability=compare_draw_assignments(*probes, 0), components=[], outcomes=[])
    out = aggregate_records([r])
    assert out['losses']['u_mask']['n'] == 2
    assert out['losses']['u_presence']['mean'] > 0
    assert out['normalization']['sampled_points']['mean'] == 3
    assert out['normalization']['unique_points']['mean'] == 2


def test_all_unique_sampled_voxels_remain_distinct_from_effective_coverage():
    first, _ = measure()
    assert first['draws'][0]['unique_sampled_voxels_all'] == 4
    assert first['draws'][0]['unique_sampled_voxels'] == 2


def test_aggregate_validates_query_targets_and_target_count_context():
    r = record()
    r['draws'][0]['queries'][0]['target_instance'] = 2
    with pytest.raises(ValueError):
        aggregate_records([r])
    r = record()
    r['draws'][0]['target_point_counts'] = [0, 0]
    r['draws'][0]['target_effective_point_counts'] = [0, 1]
    with pytest.raises(ValueError):
        aggregate_records([r])


def test_actual_anonymous_ignored_rows_allow_no_detections_without_mutation():
    from anatobind.aur.eval import ignored_rows
    gt = ignored_rows(np.ones((2, 2, 2), dtype=bool), 1.)
    before = copy.deepcopy(gt)
    assert len(gt) == 1 and gt[0]['instance'] is None
    assert detection_outcomes(gt, [], .5) == []
    assert gt == before


def test_actual_disconnected_ignored_rows_keep_indices_and_real_first_priority():
    from anatobind.aur.eval import ignored_rows
    small = np.zeros((8, 8, 8), dtype=bool)
    small[:2, :2, :2] = True
    small[5:7, 5:7, 5:7] = True
    ignored = ignored_rows(small, 1.)
    assert len(ignored) == 2 and [g['instance'] for g in ignored] == [None, None]
    gt = [ignored[0], dict(instance=73, box=ignored[0]['box']), ignored[1]]
    before = copy.deepcopy(gt)
    rows = [det(4, ignored[0]['box']), det(19, ignored[1]['box']),
            det(51, [5, 5, 5, 8, 8, 8]), det(99, C, score=.49)]
    out = detection_outcomes(gt, rows, .5)
    assert [(o['instance'], o['status']) for o in out] == [
        (4, 'TP'), (19, 'ignored_excused'), (51, 'FP'), (99, 'below_threshold')]
    assert [(o['official_gt_index'], o['official_gt_instance']) for o in out] == [
        (1, 73), (2, None), (None, None), (None, None)]
    assert gt == before
    # Both anonymous ignored components can independently excuse one detection.
    excused = detection_outcomes(ignored, [det(7, ignored[0]['box']), det(32, ignored[1]['box'])], .5)
    assert [(o['official_gt_index'], o['official_gt_instance'], o['status']) for o in excused] == [
        (0, None, 'ignored_excused'), (1, None, 'ignored_excused')]


def test_real_gt_still_requires_unique_positive_original_identity():
    with pytest.raises(ValueError):
        detection_outcomes([dict(instance=None, box=A)], [], .5)
    with pytest.raises(ValueError):
        detection_outcomes([dict(instance=7, box=A), dict(instance=7, box=B)], [], .5)
