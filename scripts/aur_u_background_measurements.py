"""Paired TRAIN U allocation and private/source-only tangent measurements.

The seven ROI panels use fixed baseline assignments. Public reductions consume
only explicitly named numeric fields: private query/component identities and
arbitrary crop metadata never become output keys or values.
"""
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from anatobind.aur.losses import sample_points


PANELS = ('fp_unsampled_query', 'fp_all_query', 'fp_unsampled_component',
          'fp_all_component', 'target_all', 'target_fp_query', 'target_tp_query')
SOURCE_PAIRS = (('bmsr', 'T1c'), ('isles', 'DWI'), ('pdgm', 'FLAIR'))
STATUSES = ('TP', 'FP', 'ignored_excused', 'below_threshold')
RESPONSE_RTOL = 1e-5
RESPONSE_ATOL = 1e-5
_CHUNK = 16384
_COMPOSITION = ('n_component_voxels', 'n_background_voxels', 'n_assigned_target_voxels',
                'n_other_gt_voxels', 'n_ignored_voxels', 'n_invalid_voxels')


def _integer(value, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < int(positive)):
        raise ValueError('Invalid integer measurement')
    return int(value)


def _boolean(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('Invalid supervision flag')
    return bool(value)


def _number(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)):
        raise ValueError('Invalid scalar measurement')
    return float(value)


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _source(record):
    pair = (record['source'], record['sequence'])
    if pair not in SOURCE_PAIRS:
        raise ValueError('Invalid source and sequence')
    return pair


def _maps(instance, valid, weight):
    gt, vv, ww = [_array(x) for x in (instance, valid, weight)]
    if (gt.ndim != 3 or gt.size == 0 or gt.shape != vv.shape or gt.shape != ww.shape
            or gt.dtype.kind not in 'iu' or (gt < 0).any()
            or vv.dtype.kind not in 'biuf' or ww.dtype.kind not in 'biuf'
            or not np.isfinite(vv).all() or not np.isfinite(ww).all()
            or (vv < 0).any() or (vv > 1).any() or (ww < 0).any() or (ww > 1).any()):
        raise ValueError('Invalid dense maps')
    return gt, vv, ww


def paired_points(valid, instance, weight, *, n, seed):
    """Call the original sampler twice and independently verify overwritten slots.

    Counts describe sampled occurrences, including duplicates. Positive/ring/
    distant-background describe spatial location; ignored/effective describe
    loss support and may overlap those spatial categories. All-invalid maps
    retain the original sampler's epsilon-uniform fallback and zero support.
    """
    n, seed = _integer(n, positive=True), _integer(seed)
    gt, vv, ww = _maps(instance, valid, weight)
    v = torch.as_tensor(vv.copy())
    i = torch.as_tensor(gt.astype(np.int64, copy=True))

    def draw(boundary, with_instance=True):
        return sample_points(v[None], n, torch.Generator().manual_seed(seed),
                             instance=i[None] if with_instance else None,
                             focus_share=.5, boundary_share=boundary)[0].numpy()

    points = np.stack((draw(.2), draw(0.)))
    uniform = draw(0., with_instance=False)
    mask = vv > .5
    masked_gt = gt * mask
    ids = np.unique(masked_gt[masked_gt > 0])
    focus = (int(n * .5) // len(ids)) * len(ids) if len(ids) else 0
    lesion = torch.as_tensor((masked_gt > 0).astype(np.float32))[None, None]
    near = F.max_pool3d(lesion, 3, stride=1, padding=1)[0, 0].numpy() > 0
    ring = near & (masked_gt == 0) & mask
    ring_slots = min(int(n * .2), n - focus) if len(ids) and ring.any() else 0
    end = focus + ring_slots
    focus_equal = np.array_equal(points[0, :focus], points[1, :focus])
    other_equal = np.array_equal(points[0, end:], points[1, end:])
    restored = np.array_equal(points[1, focus:end], uniform[focus:end])
    if not (focus_equal and other_equal and restored):
        raise ValueError('Paired sampler slot control failed')
    if not ring_slots and not np.array_equal(points[0], points[1]):
        raise ValueError('No-ring sampler control failed')
    flat_gt, flat_v, flat_w, flat_ring = gt.ravel(), mask.ravel(), ww.ravel(), ring.ravel()
    arms = []
    for arm, loc in zip(('control', 'global'), points):
        effective = flat_v[loc] & (flat_w[loc] > 0)
        arms.append(dict(arm=arm, sampled_points=n,
            positive_points=int(((flat_gt[loc] > 0) & flat_v[loc]).sum()),
            ring_points=int(flat_ring[loc].sum()),
            distant_background_points=int((flat_v[loc] & (flat_gt[loc] == 0) & ~flat_ring[loc]).sum()),
            ignored_points=int((flat_v[loc] & (flat_w[loc] == 0)).sum()),
            invalid_points=int((~flat_v[loc]).sum()), effective_points=int(effective.sum()),
            unique_points=int(np.unique(loc).size), duplicate_points=n - int(np.unique(loc).size),
            unique_effective_points=int(np.unique(loc[effective]).size),
            point_weight_sum=float(flat_w[loc].astype(np.float64).sum())))
    return points, dict(focus_slots=focus, ring_slots=ring_slots,
                        focus_slots_equal=bool(focus_equal), nonring_slots_equal=bool(other_equal),
                        restored_uniform_equal=bool(restored), arms=arms)


def _matches(pair):
    if pair is None:
        return {}
    qi, ti = [_array(x) for x in pair]
    if (qi.ndim != 1 or ti.shape != qi.shape or qi.dtype.kind not in 'iu' or ti.dtype.kind not in 'iu'
            or (qi < 0).any() or (ti < 0).any() or len(np.unique(qi)) != len(qi)
            or len(np.unique(ti)) != len(ti)):
        raise ValueError('Invalid baseline assignment')
    return {int(q): int(t) + 1 for q, t in zip(qi, ti)}


def _feature_mean(pixels, loc):
    total = torch.zeros(pixels.shape[0], dtype=torch.float64, device=pixels.device)
    for start in range(0, len(loc), _CHUNK):
        idx = torch.as_tensor(loc[start:start + _CHUNK], dtype=torch.long, device=pixels.device)
        block = pixels.index_select(1, idx)
        if not bool(torch.isfinite(block).all()):
            raise ValueError('Nonfinite ROI pixels')
        total += block.sum(1, dtype=torch.float64)
    if len(loc):
        total /= len(loc)
    result = total.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite ROI mean')
    return result


def build_rois(detection_map, detection_rows, outcomes, instance, valid, weight,
               pixels, points, matches):
    """Return private row identities, float32 means, and aligned transient indices.

    Only effective instance==0 voxels in baseline-matched official FP components
    enter background panels. Assigned-target, other-GT, ignored and invalid
    component voxels have distinct count fields. Target panels use the complete
    assigned target rather than any component bounding box.
    """
    gt, vv, ww = _maps(instance, valid, weight)
    dm, pp = _array(detection_map), _array(points)
    if (dm.shape != gt.shape or dm.dtype.kind not in 'iu' or (dm < 0).any()
            or pp.dtype != np.int64 or pp.ndim != 3 or pp.shape[:2] != (2, 2)
            or (pp < 0).any() or (pp >= gt.size).any() or len(matches) != 2
            or not isinstance(pixels, torch.Tensor) or pixels.requires_grad
            or pixels.ndim != 2 or pixels.shape[1] != gt.size or not pixels.is_floating_point()):
        raise ValueError('Invalid ROI inputs')
    ids = [_integer(r['instance'], positive=True) for r in detection_rows]
    outcome_ids = [_integer(r['instance'], positive=True) for r in outcomes]
    if (len(ids) != len(set(ids)) or len(outcome_ids) != len(set(outcome_ids))
            or set(ids) != set(outcome_ids) or set(np.unique(dm[dm > 0])) != set(ids)):
        raise ValueError('Component identity join failed')
    by_id = {r['instance']: r for r in outcomes}
    components = []
    dm, gt, vv, ww = dm.ravel(), gt.ravel(), vv.ravel() > .5, ww.ravel()
    effective = vv & (ww > 0)
    # One indexed partition avoids scanning the volume once per component.
    loc = np.flatnonzero(dm)
    loc = loc[np.argsort(dm[loc], kind='stable')]
    labels, starts, counts = np.unique(dm[loc], return_index=True, return_counts=True)
    positions = {int(k): loc[s:s + count] for k, s, count in zip(labels, starts, counts)}
    for identity, original in zip(ids, detection_rows):
        outcome = by_id[identity]
        query = _integer(original['detector_origin']['query'])
        if (outcome['status'] not in STATUSES
                or outcome.get('detector_origin', original['detector_origin']) != original['detector_origin']):
            raise ValueError('Invalid component outcome or provenance')
        components.append((identity, query, outcome['status'], positions[identity]))
    rows, features, indices = [], [], []

    def append(draw, query, panel, selected, composition=None, component=None):
        selected = np.asarray(selected, dtype=np.int64)
        row = dict(draw=draw, query=query, panel=panel, n_voxels=int(len(selected)))
        if component is not None:
            row['component'] = component
        if composition is not None:
            row.update(composition)
        rows.append(row)
        indices.append(selected)
        features.append(_feature_mean(pixels, selected))

    for draw, pair in enumerate(matches):
        assignment = _matches(pair)
        sampled = np.unique(pp[draw])
        fp_groups = defaultdict(list)
        tp_queries = {q for _, q, status, _ in components if status == 'TP' and q in assignment}

        def composition(loc, target):
            return dict(n_component_voxels=int(len(loc)),
                n_background_voxels=int((effective[loc] & (gt[loc] == 0)).sum()),
                n_assigned_target_voxels=int((effective[loc] & (gt[loc] == target)).sum()),
                n_other_gt_voxels=int((effective[loc] & (gt[loc] > 0) & (gt[loc] != target)).sum()),
                n_ignored_voxels=int((vv[loc] & (ww[loc] == 0)).sum()),
                n_invalid_voxels=int((~vv[loc]).sum()))

        for identity, query, status, loc in components:
            if status != 'FP' or query not in assignment:
                continue
            fp_groups[query].append(loc)
            bg = loc[effective[loc] & (gt[loc] == 0)]
            unsampled = np.setdiff1d(bg, sampled, assume_unique=True)
            counts = composition(loc, assignment[query])
            append(draw, query, 'fp_all_component', bg, counts, identity)
            append(draw, query, 'fp_unsampled_component', unsampled, counts, identity)
        for query in sorted(fp_groups):
            loc = np.unique(np.concatenate(fp_groups[query]))
            bg = loc[effective[loc] & (gt[loc] == 0)]
            counts = composition(loc, assignment[query])
            append(draw, query, 'fp_all_query', bg, counts)
            append(draw, query, 'fp_unsampled_query', np.setdiff1d(bg, sampled, assume_unique=True), counts)
        for query, target in sorted(assignment.items()):
            loc = np.flatnonzero(effective & (gt == target))
            append(draw, query, 'target_all', loc)
            if query in fp_groups:
                append(draw, query, 'target_fp_query', loc)
            if query in tp_queries:
                append(draw, query, 'target_tp_query', loc)
    means = np.stack(features) if features else np.zeros((0, pixels.shape[0]), np.float32)
    return rows, means, indices


def _stats(values):
    """Explicit support/sign denominators; None is never a zero measurement."""
    vals = [_number(v) for v in values if v is not None]
    a = np.asarray(vals, np.float64)
    n = len(vals)
    signs = dict(positive=int((a > 0).sum()), negative=int((a < 0).sum()), zero=int((a == 0).sum()))
    return dict(n=n, undefined=len(values) - n, mean=float(a.mean()) if n else None,
                **signs, **{'fraction_' + k: v / n if n else None for k, v in signs.items()})


def _describe(values):
    result = _stats(values)
    a = np.asarray([v for v in values if v is not None], np.float64)
    result.update(median=float(np.median(a)) if a.size else None,
                  min=float(a.min()) if a.size else None, max=float(a.max()) if a.size else None,
                  q05=float(np.quantile(a, .05)) if a.size else None,
                  q95=float(np.quantile(a, .95)) if a.size else None)
    return result


def _grid(shape, function, prefix=()):
    return ([_grid(shape[1:], function, prefix + (i,)) for i in range(shape[0])]
            if shape else function(prefix))


def _at(value, index):
    for i in index:
        value = value[i]
    return value


def response_report(meta, arrays, analysis):
    """Validate independent dense projections and return private crop reductions.

    Row responses average equally within crop/draw, including query-unioned and
    separate component-weighted panels. Empty rows and undefined directions are
    retained in every denominator. Only ``tangent_summary`` carries private
    mathematical details; it is never traversed by the public aggregator.
    """
    source, sequence = _source(meta)
    a_enabled, u_enabled = _boolean(meta['a_supervised']), _boolean(meta['u_supervised'])
    n_instances = _integer(meta['n_instances'])
    rows = meta['roi_rows']
    features, reference = np.asarray(arrays['roi_features']), np.asarray(arrays['roi_reference_responses'])
    numerics = analysis['numerics']
    jvp, defined = np.asarray(numerics['event_jvp']), np.asarray(numerics['direction_defined'])
    if (features.dtype != np.float32 or features.ndim != 2 or features.shape[0] != len(rows)
            or reference.dtype != np.float64 or reference.shape != (len(rows), 2, 2, 2)
            or jvp.ndim != 6 or jvp.shape[:4] != (2, 2, 2, 2) or jvp.shape[-1] != features.shape[-1]
            or defined.dtype != bool or defined.shape != (2, 2, 2, 2)
            or not all(np.isfinite(a).all() for a in (features, reference, jvp))
            or np.any(jvp[~defined])):
        raise ValueError('Invalid tangent response arrays')
    values = np.zeros_like(reference)
    masks = np.zeros(reference.shape, dtype=bool)
    groups = defaultdict(list)
    seen = set()
    for i, row in enumerate(rows):
        draw, query, count = [_integer(row[k]) for k in ('draw', 'query', 'n_voxels')]
        panel = row['panel']
        if draw not in (0, 1) or query >= jvp.shape[-2] or panel not in PANELS:
            raise ValueError('Invalid ROI row category')
        component = _integer(row['component'], positive=True) if panel.endswith('_component') else None
        identity = (draw, panel, query, component)
        if identity in seen:
            raise ValueError('Duplicate ROI row')
        seen.add(identity)
        groups[draw, panel].append(i)
        if not count and np.any(features[i]):
            raise ValueError('Empty ROI must have a zero feature vector')
        values[i] = np.einsum('oanc,c->oan', jvp[draw, :, :, :, query].astype(np.float64),
                              features[i].astype(np.float64))
        masks[i] = (count > 0) & defined[draw] & u_enabled
        check = (count > 0) & defined[draw]
        if not np.allclose(values[i][check], reference[i][check], rtol=RESPONSE_RTOL, atol=RESPONSE_ATOL):
            raise ValueError('Independent dense response control failed')
    a_response = numerics.get('a_loss_response')
    a_query = numerics.get('a_query_loss_response')
    if (a_response is None) != (a_query is None):
        raise ValueError('Incomplete anatomy response arrays')
    if a_response is not None:
        a_response, a_query = np.asarray(a_response), np.asarray(a_query)
        if (a_response.shape != (2, 2, 2, 2) or a_query.ndim != 5
                or a_query.shape[:4] != a_response.shape
                or not np.isfinite(a_response).all() or not np.isfinite(a_query).all()
                or not np.allclose(a_query.sum(-1), a_response, rtol=RESPONSE_RTOL, atol=RESPONSE_ATOL)):
            raise ValueError('Invalid anatomy response arrays')
    draws = []
    for draw in range(2):
        panels = {}
        for panel in PANELS:
            loc = groups[draw, panel]
            support = sum(rows[i]['n_voxels'] > 0 for i in loc)

            def response(index):
                return _stats([float(values[(i,) + index]) if masks[(i,) + index] else None for i in loc])

            def difference(index):
                objective, normalization = index
                control, global_ = (objective, 0, normalization), (objective, 1, normalization)
                return _stats([float(values[(i,) + global_] - values[(i,) + control])
                               if masks[(i,) + control] and masks[(i,) + global_] else None for i in loc])

            panels[panel] = dict(rows=len(loc), supported_rows=int(support), empty_rows=len(loc) - int(support),
                                voxels=sum(_integer(rows[i]['n_voxels']) for i in loc),
                                component_composition={key: sum(_integer(rows[i].get(key, 0)) for i in loc)
                                                       for key in _COMPOSITION},
                                responses=_grid((2, 2, 2), response),
                                paired_differences=_grid((2, 2), difference))

        positive_support = None
        support_rows = analysis['summary']['draws'][draw].get('a_query_support')
        if a_query is not None and support_rows is not None:
            if len(support_rows) != a_query.shape[-1]:
                raise ValueError('Invalid anatomy query support')
            positive_support = []
            for query, row in enumerate(support_rows):
                if _integer(row['query']) != query:
                    raise ValueError('Invalid anatomy query identity')
                positive_support.append(_boolean(row['enabled'])
                                        and _integer(row['effective_positive_points']) > 0)

        def anatomy(index):
            return (float(a_response[(draw,) + index])
                    if a_response is not None and a_enabled and defined[(draw,) + index] else None)

        def anatomy_queries(index):
            return (a_query[(draw,) + index].astype(float).tolist()
                    if a_query is not None and a_enabled and defined[(draw,) + index]
                    else [None] * (a_query.shape[-1] if a_query is not None else 0))

        def positive_queries(index):
            all_queries = anatomy_queries(index)
            return [value if positive_support is not None and positive_support[q] else None
                    for q, value in enumerate(all_queries)]

        draws.append(dict(draw=draw, panels=panels, a_loss_response=_grid((2, 2, 2), anatomy),
                          a_query_loss_response=_grid((2, 2, 2), anatomy_queries),
                          a_positive_query_loss_response=_grid((2, 2, 2), positive_queries)))
    return dict(source=source, sequence=sequence, n_instances=n_instances,
                a_supervised=a_enabled, u_supervised=u_enabled, draws=draws,
                tangent_summary=deepcopy(analysis['summary']))


def _sign_totals(stats):
    keys = ('n', 'undefined', 'positive', 'negative', 'zero')
    result = {k: sum(_integer(s[k]) for s in stats) for k in keys}
    if result['n'] != result['positive'] + result['negative'] + result['zero']:
        raise ValueError('Inconsistent response signs')
    result.update({'fraction_' + k: result[k] / result['n'] if result['n'] else None
                   for k in ('positive', 'negative', 'zero')})
    return result


def _paired_crop_values(crops, getter):
    values = []
    for crop in crops:
        draw_values = [getter(draw) for draw in crop['draws']]
        values.append(sum(draw_values) / 2 if all(v is not None for v in draw_values) else None)
    return values


def _a_difference(draw, index):
    objective, normalization = index
    control = _at(draw['a_loss_response'], (objective, 0, normalization))
    global_ = _at(draw['a_loss_response'], (objective, 1, normalization))
    return _number(global_) - _number(control) if control is not None and global_ is not None else None


def aggregate_reports(reports):
    """Fixed source whitelist; source distributions use crop/draw means.

    Both point draws are repeated measurements. ``crop_mean`` summaries give
    each crop one vote and require both draws to be defined. Anonymous row sign
    counts are descriptive support counts, never independent-patient tests.
    """
    grouped = defaultdict(list)
    for report in reports:
        pair = _source(report)
        if (len(report['draws']) != 2 or [d['draw'] for d in report['draws']] != [0, 1]):
            raise ValueError('Invalid repeated draw schema')
        _boolean(report['a_supervised'])
        _boolean(report['u_supervised'])
        for draw in report['draws']:
            if set(draw['panels']) != set(PANELS):
                raise ValueError('Invalid panel categories')
        grouped[pair].append(report)
    sources = []
    for source, sequence in SOURCE_PAIRS:
        crops = grouped[source, sequence]
        if not crops:
            continue
        draws = [draw for crop in crops for draw in crop['draws']]
        panels = {}
        for panel in PANELS:
            items = [draw['panels'][panel] for draw in draws]
            for item in items:
                if (_integer(item['rows']) != _integer(item['supported_rows']) + _integer(item['empty_rows'])):
                    raise ValueError('Inconsistent ROI support counts')

            def getter(index, paired=False):
                field = 'paired_differences' if paired else 'responses'
                return lambda draw: _at(draw['panels'][panel][field], index)['mean']

            summaries = {}
            for field, shape, paired in [('responses', (2, 2, 2), False),
                                          ('paired_differences', (2, 2), True)]:
                summaries['crop_draw_' + field] = _grid(shape, lambda ix: _describe([getter(ix, paired)(d) for d in draws]))
                summaries['crop_mean_' + field] = _grid(shape, lambda ix: _describe(_paired_crop_values(crops, getter(ix, paired))))
                summaries['row_' + field + '_sign_counts'] = _grid(shape, lambda ix: _sign_totals([_at(i[field], ix) for i in items]))
            by_draw = []
            for draw_id in range(2):
                group = [crop['draws'][draw_id] for crop in crops]
                by_draw.append(dict(draw=draw_id,
                    responses=_grid((2, 2, 2), lambda ix: _describe([getter(ix)(d) for d in group])),
                    paired_differences=_grid((2, 2), lambda ix: _describe([getter(ix, True)(d) for d in group]))))
            panels[panel] = dict(
                **{k: sum(_integer(i[k]) for i in items) for k in ('rows', 'supported_rows', 'empty_rows', 'voxels')},
                component_composition={key: sum(_integer(i['component_composition'][key]) for i in items)
                                       for key in _COMPOSITION},
                crop_draws=len(draws), supported_crop_draws=sum(i['supported_rows'] > 0 for i in items),
                empty_crop_draws=sum(i['supported_rows'] == 0 for i in items), by_draw=by_draw, **summaries)

        def a_getter(index):
            return lambda draw: _at(draw['a_loss_response'], index)

        a_loss = dict(
            crop_draw_responses=_grid((2, 2, 2), lambda ix: _describe([a_getter(ix)(d) for d in draws])),
            crop_mean_responses=_grid((2, 2, 2), lambda ix: _describe(_paired_crop_values(crops, a_getter(ix)))),
            crop_draw_paired_differences=_grid((2, 2), lambda ix: _describe([_a_difference(d, ix) for d in draws])),
            crop_mean_paired_differences=_grid((2, 2), lambda ix: _describe(
                _paired_crop_values(crops, lambda d: _a_difference(d, ix)))),
            query_sign_counts=_grid((2, 2, 2), lambda ix: _stats(
                [v for d in draws for v in _at(d['a_query_loss_response'], ix)])),
            positive_query_sign_counts=_grid((2, 2, 2), lambda ix: _stats(
                [v for d in draws for v in _at(d['a_positive_query_loss_response'], ix)])),
            by_draw=[dict(draw=d, responses=_grid((2, 2, 2), lambda ix: _describe(
                [a_getter(ix)(c['draws'][d]) for c in crops])),
                paired_differences=_grid((2, 2), lambda ix: _describe(
                    [_a_difference(c['draws'][d], ix) for c in crops]))) for d in range(2)])
        sources.append(dict(source=source, sequence=sequence, crops=len(crops), crop_draws=len(draws),
            a_supervised_crops=sum(c['a_supervised'] for c in crops),
            u_supervised_crops=sum(c['u_supervised'] for c in crops), panels=panels, a_loss=a_loss))
    return dict(schema='aur_u_background_allocation_v1',
                measurement='float32_frozen_feature_head_tangent_at_native_outputs',
                draws_are_repeated_measurements=True, crop_mean_requires_both_draws=True,
                objectives=['combined', 'u_only'], arms=['control', 'global'], normalizations=['raw', 'unit'],
                sources=sources)
