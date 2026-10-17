"""Exact scalar-mask oracle and bounded-work regressions for NMS pruning."""
import numpy as np
import pytest

from anatobind.aur import query_decoder as D


def _record(mask, *, score=.8, query=0, component=1, window=None):
    """Keep a full test mask as an independent reference for the packed candidate."""
    return dict(mask=mask, score=score, query=query, component=component,
                window=window or tuple((0, n) for n in mask.shape))


def _candidate(record):
    mask = record['mask']
    coords = np.nonzero(mask)
    box = tuple((int(axis.min()), int(axis.max()) + 1) for axis in coords)
    local = mask[tuple(slice(a, b) for a, b in box)]
    return D._Candidate(record['window'], record['query'], record['component'], box,
                        np.packbits(local.ravel()), int(mask.sum()), record['score'])


def _decode(records, shape):
    decoder = D.QueryCCDecoder(shape, 10., workers=1)
    decoder._candidates = [_candidate(record) for record in records]
    decoder._raw_components = len(records)
    return decoder, decoder.finish()


def _scalar_nms(records):
    """Independent NMS only: set intersections of full original global masks."""
    ordered = sorted(records, key=lambda r: (-r['score'], tuple(a for a, _ in r['window']),
                                             r['query'], r['component']))
    survivors, trace = [], []
    for rank, record in enumerate(ordered, 1):
        voxels = frozenset(zip(*np.nonzero(record['mask'])))
        for earlier_rank, earlier_voxels in survivors:
            iou = len(voxels & earlier_voxels) / len(voxels | earlier_voxels)
            if iou >= .5:
                trace.append(dict(rank=rank, suppressor_rank=earlier_rank, iou=iou))
                break
        else:
            survivors.append((rank, voxels))
    return {rank for rank, _ in survivors}, trace


def _line(shape, start, stop):
    mask = np.zeros(shape, bool)
    mask[0, 0, start:stop] = True
    return mask


@pytest.mark.parametrize('layout', ['disjoint', 'volume_bound', 'bbox_bound'])
def test_impossible_pairs_skip_scalar_mask_iou(monkeypatch, layout):
    shape = (1, 1, 240)
    if layout == 'disjoint':
        records = [_record(_line(shape, 3 * i, 3 * i + 2), query=i) for i in range(80)]
        expected_instances, expected_occupied, expected_occluded = 80, 160, 0
    elif layout == 'volume_bound':
        records = [_record(_line(shape, 0, 6), score=.9),
                   _record(_line(shape, 0, 2), query=1)]
        expected_instances, expected_occupied, expected_occluded = 1, 6, 1
    else:
        records = [_record(_line(shape, 0, 6), score=.9),
                   _record(_line(shape, 4, 10), query=1)]
        expected_instances, expected_occupied, expected_occluded = 2, 10, 0

    def unexpected_exact_pair(*args):
        raise AssertionError('Impossible suppression pair must be pruned before scalar mask IoU')

    monkeypatch.setattr(D, '_mask_iou', unexpected_exact_pair)
    decoder, (instance, rows, stats) = _decode(records, shape)
    assert len(rows) == expected_instances
    assert np.count_nonzero(instance) == expected_occupied
    assert stats['eligible_candidates'] == len(records)
    assert stats['suppressed_candidates'] == 0
    assert stats['fully_occluded_candidates'] == expected_occluded
    assert decoder.nms_trace == []


@pytest.mark.parametrize('seed', [4, 17, 101])
def test_pruned_nms_matches_scalar_original_mask_oracle_with_ties(seed):
    rng = np.random.default_rng(seed)
    shape = (32, 32, 32)
    records = []
    for i in range(180):
        start = rng.integers(0, 18, 3)
        size = rng.integers(2, 10, 3)
        local = np.zeros(tuple(size), bool)
        local[:, :, 0] = True
        for z in range(int(size[0])):
            local[z, rng.integers(int(size[1])), :] = True
        # Every row meets the common plane, so each original mask is connected.
        mask = np.zeros(shape, bool)
        mask[tuple(slice(int(a), int(a + b)) for a, b in zip(start, size))] = local
        window = tuple((int(a // 4 * 4), int(a // 4 * 4 + 32)) for a in start)
        if i % 6 == 5:
            # Include many genuine suppressions among the randomized connected masks.
            mask, window = records[-1]['mask'].copy(), records[-1]['window']
        records.append(_record(mask, score=.9 - (i % 9) * .01,
                               query=i % 7, component=i + 1, window=window))
    # Spatially separate, exact .5 overlap with two possible suppressors.
    for i, (a, b) in enumerate([(0, 6), (4, 10), (2, 8)]):
        mask = np.zeros(shape, bool)
        mask[29, 0, a:b] = True
        records.append(_record(mask, score=1. - i * .001, query=i, component=999 + i))
    rng.shuffle(records)
    expected_survivors, expected_trace = _scalar_nms(records)
    decoder, (_, rows, stats) = _decode(records, shape)
    assert decoder.nms_trace == expected_trace
    assert expected_trace[0] == dict(rank=3, suppressor_rank=1, iou=.5)
    assert len(expected_trace) >= 31
    assert stats['raw_components'] == stats['eligible_candidates'] == 183
    assert stats['suppressed_candidates'] == len(expected_trace)
    assert stats['eligible_candidates'] - stats['suppressed_candidates'] == len(expected_survivors)
    assert {row['detector_origin']['rank'] for row in rows} <= expected_survivors


def test_equal_scores_use_window_query_component_priority_before_suppression():
    shape = (3, 1, 3)
    mask = np.zeros(shape, bool)
    mask[2, 0, :] = True
    origin = ((0, 3), (0, 1), (0, 3))
    records = [_record(mask, window=((1, 4), (0, 1), (0, 3))),
               _record(mask, window=origin, query=4),
               _record(mask, window=origin, component=2),
               _record(mask, window=origin)]
    decoder, (_, rows, stats) = _decode(records, shape)
    assert len(rows) == 1 and stats['suppressed_candidates'] == 3
    assert rows[0]['detector_origin']['window'] == [[0, 3], [0, 1], [0, 3]]
    assert rows[0]['detector_origin']['query'] == 0
    assert rows[0]['detector_origin']['component'] == 1
    assert decoder.nms_trace == [dict(rank=rank, suppressor_rank=1, iou=1.) for rank in [2, 3, 4]]


@pytest.mark.parametrize('count', [0, 1])
def test_empty_prior_survivors_preserve_first_candidate_and_empty_output(count):
    shape = (1, 1, 3)
    records = [_record(_line(shape, 0, 3))] if count else []
    decoder, (instance, rows, stats) = _decode(records, shape)
    assert instance.dtype == np.int32
    assert len(rows) == stats['eligible_candidates'] == stats['final_instances'] == count
    assert np.count_nonzero(instance) == 3 * count
    assert stats['suppressed_candidates'] == 0 and decoder.nms_trace == []


def test_vector_bounds_use_int64_for_coordinates_products_and_threshold_arithmetic():
    # Cross int32 coordinate and product limits without constructing huge voxel masks.
    origin = 2 ** 31 - 35000
    box = np.array([[origin, origin + 70000], [0, 70000], [0, 1]], dtype=np.int64)
    survivors = np.array([box, box, box, box], dtype=np.int64)
    survivors[1, 0] += 35000
    survivors[2, 0, 0] += 35000
    survivors[3, 0] += 70000
    volumes = np.array([4_900_000_000, 4_900_000_000, 2_450_000_000, 4_900_000_000], dtype=np.int64)
    possible = D._possible_nms_pairs(box, 4_900_000_000, survivors, volumes)
    # Index 2 reaches exactly .5: 3*2.45e9 = 4.9e9+2.45e9.
    np.testing.assert_array_equal(possible, [0, 2])
    empty = D._possible_nms_pairs(box, 4_900_000_000, survivors[:0], volumes[:0])
    assert empty.shape == (0,)
