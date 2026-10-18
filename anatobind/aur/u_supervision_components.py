"""Private original-ID component measurements and explicitly anonymous TRAIN aggregates.

These are observational diagnostics. Detection matching, sampled training assignment,
and descriptive dense-mask overlap have separate fields and never alter one another.
"""
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np

from anatobind.eval.detection_metrics import match_scan, scan_matches

_STATUSES = ('TP', 'FP', 'ignored_excused', 'below_threshold')
_QUERY_STATUSES = ('matched', 'unmatched', 'unsupervised')
_SOURCES = (('bmsr', 'T1c'), ('isles', 'DWI'), ('pdgm', 'FLAIR'))
_DENSE = ('n_voxels', 'volume_mm3', 'parent_score', 'retained_fraction',
          'foreground_fraction', 'background_fraction', 'ignored_fraction',
          'best_gt_dice', 'best_gt_iou')
_LOSSES = ('u_presence', 'u_mask', 'total', 'presence', 'mask', 'bce', 'dice')
_NORMALIZATION = ('sampled_points', 'unique_points', 'effective_points', 'unique_effective_points',
                  'point_denominator', 'matched_queries', 'unmatched_queries',
                  'matched_query_denominator', 'unmatched_query_denominator', 'presence_weight_sum',
                  'supervised_samples', 'unmatched_mask_weight')
_DRAW = ('coverage_fraction', 'sampled_occurrences', 'effective_sampled_occurrences',
         'unique_sampled_voxels', 'unique_sampled_voxels_all', 'target_dice', 'target_iou')
_QUERY = ('presence_probability', 'presence_target', 'presence_weight', 'presence_gradient',
          'presence_bce', 'presence_contribution', 'mask_bce', 'mask_dice', 'mask_contribution',
          'target_positive_points', 'target_positive_unique', 'target_positive_effective',
          'target_positive_effective_unique', 'mask_gradient_sum', 'mask_gradient_abs_sum',
          'mask_gradient_l2', 'mask_gradient_positive_points', 'mask_gradient_negative_points',
          'mask_gradient_zero_points')


def _int(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('Expected nonnegative integer')
    return int(value)


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        raise ValueError('Expected finite numeric measurement')
    return float(value)


def _ids(rows):
    ids = [_int(r['instance']) for r in rows]
    if 0 in ids or len(ids) != len(set(ids)):
        raise ValueError('Component identity must be positive and unique')
    return ids


def detection_outcomes(gt_rows, detection_rows, score_threshold):
    """Classify every original detection, in row order, at one inclusive cutoff.

    ``official_gt_index`` indexes the supplied GT rows, including ignored rows;
    ``official_gt_instance`` is their unmodified identity. Missing GT instance
    fields are rejected to prevent an implicit identity renumbering.
    """
    threshold = _number(score_threshold)
    _ids(gt_rows)
    ids = _ids(detection_rows)
    for row in detection_rows:
        _number(row['score'])
    hits, fp, kept = scan_matches(dict(gt=gt_rows, dets=detection_rows), threshold)
    assigned = {p: g for g, p in hits.items()}
    rest = [p for p in range(len(kept)) if p not in assigned]
    ignored = [g for g, row in enumerate(gt_rows) if row.get('ignore')]
    ignored_pairs = match_scan([gt_rows[g] for g in ignored], [kept[p] for p in rest]) if ignored and rest else {}
    excused = {rest[p]: ignored[g] for g, p in ignored_pairs.items()}
    by_id = {}
    for p, row in enumerate(kept):
        g = assigned.get(p, excused.get(p))
        by_id[row['instance']] = ('TP' if p in assigned else 'ignored_excused' if p in excused else 'FP', g)
    result = []
    for identity, row in zip(ids, detection_rows):
        status, g = by_id.get(identity, ('below_threshold', None))
        result.append(dict(instance=identity, status=status, official_gt_index=g,
                           official_gt_instance=None if g is None else int(gt_rows[g]['instance']),
                           detector_origin=deepcopy(row['detector_origin'])))
    counts = Counter(row['status'] for row in result)
    if (counts['TP'] != len(hits) or counts['FP'] != fp or counts['ignored_excused'] != len(excused)
            or len(kept) != counts['TP'] + counts['FP'] + counts['ignored_excused']
            or len(result) != len(kept) + counts['below_threshold']):
        raise ValueError('Original detection labels disagree with official matching')
    return result


def _gradient(values):
    v = np.asarray(values, dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError('Nonfinite gradient measurement')
    return dict(n=int(v.size), sum=float(v.sum()), absolute_sum=float(np.abs(v).sum()),
                l2=float(np.sqrt(np.square(v).sum())), mean=float(v.mean()) if v.size else None,
                positive=int((v > 0).sum()), negative=int((v < 0).sum()), zero=int((v == 0).sum()))


def _overlap(intersection, component_size, target_size):
    return (2 * intersection / (component_size + target_size) if component_size + target_size else 0.,
            intersection / (component_size + target_size - intersection)
            if component_size + target_size - intersection else 0.)


def component_measurements(detection_map, detection_rows, gt_instance_map, valid, point_weight, *, outcomes, draws):
    """Measure actual original-ID masks, effective point coverage, and detached gradients.

    Dense fractions use valid component voxels, including ignored voxels. Mask
    overlap uses valid component/GT masks. Coverage uses valid component voxels
    with positive point weight. Occurrence gradients and duplicate-summed unique
    voxel gradients are reported separately; only the latter describe the net
    sampled gradient at a voxel. No dense query-by-volume array is constructed.
    """
    arrays = [np.asarray(a) for a in (detection_map, gt_instance_map, valid, point_weight)]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError('Dense measurement shapes disagree')
    dm, gt, validity, weight = [a.reshape(-1) for a in arrays]
    if (dm.dtype.kind not in 'iu' or gt.dtype.kind not in 'iu' or (dm < 0).any() or (gt < 0).any()
            or not np.isfinite(validity).all() or not np.isfinite(weight).all()
            or (weight < 0).any() or (weight > 1).any()):
        raise ValueError('Invalid dense labels or weights')
    valid_mask = validity > .5
    effective = valid_mask & (weight > 0)
    ignored = valid_mask & (weight < .5)
    ids = _ids(detection_rows)
    if set(np.unique(dm[dm > 0])) != set(ids) or set(_ids(outcomes)) != set(ids):
        raise ValueError('Detection map, rows, and outcomes have different identities')
    outcome_by_id = {r['instance']: r for r in outcomes}
    gt_ids, gt_counts = np.unique(gt[valid_mask & (gt > 0)], return_counts=True)
    gt_sizes = dict(zip(gt_ids.tolist(), gt_counts.tolist()))
    # Group nonzero voxels once, preserving actual (possibly sparse) IDs.
    positions = np.flatnonzero(dm)
    order = np.argsort(dm[positions], kind='stable')
    positions = positions[order]
    labels, starts, counts = np.unique(dm[positions], return_index=True, return_counts=True)
    component_positions = {int(i): positions[s:s + n] for i, s, n in zip(labels, starts, counts)}
    prepared_draws = []
    draw_ids = set()
    for draw in draws:
        draw_id = _int(draw['draw'])
        points = np.asarray(draw['points'])
        gradients = np.asarray(draw['mask_logit_gradient'])
        targets = draw['query_target_instance']
        if (draw_id in draw_ids or points.dtype != np.int64 or points.ndim != 1
                or (points < 0).any() or (points >= dm.size).any()
                or gradients.dtype != np.float32 or gradients.shape != (len(targets), points.size)
                or not np.isfinite(gradients).all() or not isinstance(draw['u_supervised'], (bool, np.bool_))):
            raise ValueError('Invalid supplied draw schema')
        positive_targets = [t for t in targets if t is not None]
        if len(positive_targets) != len(set(positive_targets)):
            raise ValueError('Repeated training assignment')
        for target in targets:
            if target is not None and (_int(target) == 0 or target not in gt_sizes):
                raise ValueError('Training target does not occur in GT map')
        if not draw['u_supervised'] and (any(t is not None for t in targets) or np.any(gradients)):
            raise ValueError('Unsupervised draw has targets or nonzero loss gradients')
        draw_ids.add(draw_id)
        unique, inverse = np.unique(points, return_inverse=True)
        prepared_draws.append((draw, points, gradients, unique, inverse, {}))
    result = []
    for identity, row in zip(ids, detection_rows):
        outcome = outcome_by_id[identity]
        origin = row['detector_origin']
        if outcome['status'] not in _STATUSES or outcome.get('detector_origin', origin) != origin:
            raise ValueError('Outcome provenance or category differs from detection')
        query = _int(origin['query'])
        loc = component_positions[identity]
        vv = loc[valid_mask[loc]]
        n_valid = int(vv.size)
        fg = int(((gt[vv] > 0) & ~ignored[vv]).sum())
        ig = int(ignored[vv].sum())
        bg = n_valid - fg - ig
        overlap_ids, intersections = np.unique(gt[vv][gt[vv] > 0], return_counts=True)
        overlap = dict(zip(overlap_ids.tolist(), intersections.tolist()))
        best = min(overlap, key=lambda i: (-_overlap(overlap[i], n_valid, gt_sizes[i])[1], i)) if overlap else None
        best_intersection = overlap.get(best, 0)
        dice, iou = _overlap(best_intersection, n_valid, gt_sizes.get(best, 0))
        item = dict(instance=identity, query=query, status=outcome['status'],
                    official_gt_index=outcome.get('official_gt_index'),
                    official_gt_instance=outcome.get('official_gt_instance'), detector_origin=deepcopy(origin),
                    n_voxels=int(loc.size), n_valid_voxels=n_valid, n_effective_voxels=int(effective[loc].sum()),
                    n_foreground_voxels=fg, n_background_voxels=bg, n_ignored_voxels=ig,
                    n_padded_voxels=int(loc.size) - n_valid, fraction_denominator=n_valid,
                    foreground_fraction=fg / n_valid if n_valid else None,
                    background_fraction=bg / n_valid if n_valid else None,
                    ignored_fraction=ig / n_valid if n_valid else None,
                    best_gt_instance=best, best_gt_intersection=best_intersection,
                    best_gt_dice=dice, best_gt_iou=iou, parent_score=_number(origin['parent_score']),
                    retained_fraction=_number(origin['retained_fraction']), draws=[])
        if 'volume_mm3' in row:
            item['volume_mm3'] = _number(row['volume_mm3'])
        for draw, points, gradients, unique, inverse, gradient_cache in prepared_draws:
            if query >= gradients.shape[0]:
                raise ValueError('Detector query is outside supplied query set')
            target = draw['query_target_instance'][query]
            target = None if target is None else int(target)
            query_status = 'unsupervised' if not draw['u_supervised'] else 'unmatched' if target is None else 'matched'
            at_component = dm[points] == identity
            effective_points = at_component & effective[points]
            unique_in = (dm[unique] == identity) & effective[unique]
            if query not in gradient_cache:
                gradient_cache[query] = np.bincount(inverse, weights=gradients[query], minlength=unique.size)
            net = gradient_cache[query]
            unique_gradients = net[unique_in]
            negative = (unique_in & ((gt[unique] != target) if target is not None else np.ones(unique.size, bool))
                        if query_status != 'unsupervised' else np.zeros(unique.size, bool))
            intersection = overlap.get(target, 0)
            target_dice, target_iou = (_overlap(intersection, n_valid, gt_sizes[target])
                                       if target is not None else (None, None))
            official_target = outcome.get('official_gt_instance')
            relation = ('unsupervised' if query_status == 'unsupervised' else
                        'unmatched_query' if target is None else
                        'no_official_target' if official_target is None else
                        'same_target' if target == official_target else 'different_target')
            denominator = item['n_effective_voxels']
            item['draws'].append(dict(draw=int(draw['draw']), query_status=query_status,
                target_instance=target, official_target_relation=relation, target_intersection=intersection,
                target_dice=target_dice, target_iou=target_iou,
                sampled_occurrences=int(at_component.sum()), effective_sampled_occurrences=int(effective_points.sum()),
                unique_sampled_voxels=int(unique_in.sum()),
                unique_sampled_voxels_all=int((dm[unique] == identity).sum()), coverage_denominator=denominator,
                coverage_fraction=float(unique_in.sum() / denominator) if denominator else None,
                gradient_occurrences=_gradient(gradients[query, effective_points]),
                gradient_unique=_gradient(unique_gradients), negative_target_gradient_unique=_gradient(net[negative])))
        result.append(item)
    return result


def _describe(values, undefined=0):
    values = [_number(v) for v in values]
    if not values:
        return dict(n=0, undefined=undefined, mean=None, median=None, min=None, max=None, q05=None, q95=None)
    a = np.asarray(values, np.float64)
    return dict(n=len(values), undefined=undefined, mean=float(a.mean()), median=float(np.median(a)),
                min=float(a.min()), max=float(a.max()), q05=float(np.quantile(a, .05)), q95=float(np.quantile(a, .95)))


def _weighted(items, fields):
    """Each item is (private query group, scalar mapping); never serialize group."""
    out = {}
    for field in fields:
        vals, groups, undefined = [], defaultdict(list), 0
        undefined_groups = set()
        for group, item in items:
            v = item.get(field)
            if v is None:
                undefined += 1
                undefined_groups.add(group)
            else:
                v = _number(v)
                vals.append(v)
                groups[group].append(v)
        out[field] = dict(component_weighted=_describe(vals, undefined),
                          query_weighted=_describe([np.mean(v) for v in groups.values()],
                                                   len(undefined_groups - groups.keys())))
    return out


def _pair_counts(left, right, n_gt):
    a = {t: q for q, t in enumerate(left['query_target_instance']) if t is not None}
    b = {t: q for q, t in enumerate(right['query_target_instance']) if t is not None}
    both = a.keys() & b.keys()
    same = sum(a[t] == b[t] for t in both)
    return dict(first_matched=len(a), second_matched=len(b), both_matched=len(both),
                same_query=same, different_query=len(both) - same,
                first_only=len(a.keys() - b.keys()), second_only=len(b.keys() - a.keys()),
                neither=n_gt - len(a.keys() | b.keys()))


def _assignment_counts(records):
    changes = comparisons = target_changes = target_comparisons = unassigned = overflow = zero_gt = 0
    unsupervised = supervised = pairs = undefined_pairs = 0
    zero_positive = zero_effective = point_denominator = undefined_point_targets = 0
    totals = Counter({key: 0 for key in ('first_matched', 'second_matched', 'both_matched',
        'same_query', 'different_query', 'first_only', 'second_only', 'neither')})
    for record in records:
        n_gt, n_queries = record['n_instances'], record['n_queries']
        draws = sorted(record['draws'], key=lambda d: d['draw'])
        supplied = record.get('matching_stability')
        for draw in draws:
            if not draw.get('u_supervised', True):
                unsupervised += 1
                continue
            supervised += 1
            targets = draw['query_target_instance']
            unassigned += n_gt - len({t for t in targets if t is not None})
            overflow += max(0, n_gt - n_queries)
            zero_gt += int(n_gt == 0)
        if supplied is not None:
            point_denominator += 2 * n_gt
            zero_positive += supplied['first_zero_positive_targets'] + supplied['second_zero_positive_targets']
            zero_effective += (supplied['first_zero_effective_positive_targets']
                               + supplied['second_zero_effective_positive_targets'])
        else:
            for draw in draws:
                positive = draw.get('target_point_counts')
                effective = draw.get('target_effective_point_counts')
                if positive is None or effective is None:
                    undefined_point_targets += n_gt
                else:
                    point_denominator += n_gt
                    zero_positive += sum(v == 0 for v in positive)
                    zero_effective += sum(v == 0 for v in effective)
        for left, right in zip(draws, draws[1:]):
            if not left.get('u_supervised', True) or not right.get('u_supervised', True):
                undefined_pairs += 1
                continue
            pairs += 1
            a, b = left['query_target_instance'], right['query_target_instance']
            comparisons += n_queries
            changes += sum(x != y for x, y in zip(a, b))
            counts = _pair_counts(left, right, n_gt)
            target_comparisons += n_gt
            target_changes += counts['different_query'] + counts['first_only'] + counts['second_only']
            totals.update(counts)
    return dict(query_assignment_changes=changes, query_assignment_comparisons=comparisons,
                query_assignment_change_fraction=changes / comparisons if comparisons else None,
                target_assignment_changes=target_changes, target_assignment_comparisons=target_comparisons,
                target_assignment_change_fraction=target_changes / target_comparisons if target_comparisons else None,
                assignment_denominator=target_comparisons, **totals,
                supervised_draws=supervised, unsupervised_draws=unsupervised,
                supervised_draw_pairs=pairs, undefined_draw_pairs=undefined_pairs,
                unassigned_gt_draws=unassigned, capacity_overflow_gt_draws=overflow, empty_gt_draws=zero_gt,
                zero_positive_target_draws=zero_positive,
                zero_effective_positive_target_draws=zero_effective,
                target_point_count_denominator=point_denominator,
                undefined_target_point_counts=undefined_point_targets)


def _summarize(records):
    detection_counts = {s: 0 for s in _STATUSES}
    query_counts = {s: 0 for s in _QUERY_STATUSES}
    cross = {q: {s: 0 for s in _STATUSES} for q in _QUERY_STATUSES}
    relations = {r: 0 for r in ('same_target', 'different_target', 'no_official_target', 'unmatched_query', 'unsupervised')}
    dense, coverage, gradients = [], [], []
    dense_by_status = {s: [] for s in _STATUSES}
    coverage_by_status = {s: [] for s in _STATUSES}
    gradients_by_status = {s: [] for s in _STATUSES}
    query_values = {s: defaultdict(list) for s in _QUERY_STATUSES}
    query_undefined = {s: Counter() for s in _QUERY_STATUSES}
    loss_values = defaultdict(list)
    normalization_values = defaultdict(list)
    normalization_undefined = Counter()
    query_component_counts = []
    for ri, record in enumerate(records):
        for draw in record['draws']:
            targets = draw['query_target_instance']
            for q, target in enumerate(targets):
                status = ('unsupervised' if not draw.get('u_supervised', True) else
                          'matched' if target is not None else 'unmatched')
                query_counts[status] += 1
                qs = draw.get('queries', [])
                scalar = qs[q] if qs else {}
                for field in _QUERY:
                    if scalar.get(field) is None:
                        query_undefined[status][field] += 1
                    else:
                        query_values[status][field].append(_number(scalar[field]))
            for field in _NORMALIZATION:
                value = draw.get('normalization', {}).get(field)
                if value is None:
                    normalization_undefined[field] += 1
                else:
                    normalization_values[field].append(_number(value))
            for field in _LOSSES:
                if field in draw.get('losses', {}):
                    loss_values[field].append(_number(draw['losses'][field]))
        per_query = Counter(c['query'] for c in record['components'])
        query_component_counts.extend(per_query.get(q, 0) for q in range(record['n_queries']))
        for c in record['components']:
            status = c['status']
            detection_counts[status] += 1
            group = (ri, c['query'])
            dense.append((group, c))
            dense_by_status[status].append((group, c))
            for d in c['draws']:
                cross[d['query_status']][status] += 1
                relations[d['official_target_relation']] += 1
                group = (ri, c['query'], d['draw'])
                coverage.append((group, d))
                coverage_by_status[status].append((group, d))
                grad = {}
                for kind in ('gradient_occurrences', 'gradient_unique', 'negative_target_gradient_unique'):
                    for field in ('n', 'sum', 'absolute_sum', 'l2', 'mean', 'positive', 'negative', 'zero'):
                        grad[kind + '_' + field] = d[kind][field]
                gradients.append((group, grad))
                gradients_by_status[status].append((group, grad))
    gradient_fields = tuple(kind + '_' + field for kind in ('gradient_occurrences', 'gradient_unique', 'negative_target_gradient_unique')
                            for field in ('n', 'sum', 'absolute_sum', 'l2', 'mean', 'positive', 'negative', 'zero'))
    n_crops = len(records)
    return dict(n_crops=n_crops, n_queries=sum(r['n_queries'] for r in records),
                n_draws=sum(len(r['draws']) for r in records),
                n_query_draws=sum(r['n_queries'] * len(r['draws']) for r in records),
                n_gt=sum(r['n_instances'] for r in records), n_components=sum(detection_counts.values()),
                detection_counts=detection_counts, query_counts=query_counts,
                fp_per_crop=detection_counts['FP'] / n_crops if n_crops else None,
                fp_per_crop_denominator=n_crops, query_detection_cross_tab=cross,
                query_detection_cross_tab_denominator=sum(sum(v.values()) for v in cross.values()),
                official_training_target_relation=relations, components_per_query=_describe(query_component_counts),
                dense_mask=_weighted(dense, _DENSE), sampled_coverage=_weighted(coverage, _DRAW),
                gradients=_weighted(gradients, gradient_fields),
                by_detection_status={s: dict(n_components=detection_counts[s],
                    dense_mask=_weighted(dense_by_status[s], _DENSE),
                    sampled_coverage=_weighted(coverage_by_status[s], _DRAW),
                    gradients=_weighted(gradients_by_status[s], gradient_fields)) for s in _STATUSES},
                query_supervision={s: {k: _describe(query_values[s][k], query_undefined[s][k]) for k in _QUERY}
                                   for s in _QUERY_STATUSES},
                losses={k: _describe(loss_values[k], sum(len(r['draws']) for r in records) - len(loss_values[k])) for k in _LOSSES},
                normalization={k: _describe(normalization_values[k], normalization_undefined[k]) for k in _NORMALIZATION},
                matching_stability=_assignment_counts(records))


def _validate_record(record):
    if (record['source'], record['sequence']) not in _SOURCES:
        raise ValueError('Unregistered anonymous source/sequence')
    n_queries, n_gt = _int(record['n_queries']), _int(record['n_instances'])
    if 'row_ordinal' in record:
        _int(record['row_ordinal'])
    ids = _ids(record['components'])
    if set(ids) != set(_ids(record['outcomes'])):
        raise ValueError('Aggregate component and outcome identities disagree')
    outcomes = {o['instance']: o for o in record['outcomes']}
    draw_by_id = {}
    for draw in record['draws']:
        identity = _int(draw['draw'])
        targets = draw['query_target_instance']
        if identity in draw_by_id or len(targets) != n_queries:
            raise ValueError('Aggregate draw identity or query count differs')
        positive = [t for t in targets if t is not None]
        if any(_int(t) == 0 or t > n_gt for t in positive) or len(positive) != len(set(positive)):
            raise ValueError('Invalid or repeated training assignment')
        queries = draw.get('queries', [])
        if queries and (len(queries) != n_queries or [q['query'] for q in queries] != list(range(n_queries))):
            raise ValueError('Probe query identities disagree')
        for q, scalar in enumerate(queries):
            if ('target_instance' in scalar and scalar['target_instance'] != targets[q]):
                raise ValueError('Probe query training target differs')
            if ('matched' in scalar and scalar['matched'] !=
                    ((targets[q] is not None) if draw.get('u_supervised', True) else None)):
                raise ValueError('Probe query match status differs')
        point_counts = draw.get('target_point_counts')
        effective_counts = draw.get('target_effective_point_counts')
        if point_counts is not None or effective_counts is not None:
            if point_counts is None or effective_counts is None or len(point_counts) != n_gt or len(effective_counts) != n_gt:
                raise ValueError('Target point count cardinality differs')
            if any(_int(e) > _int(p) for p, e in zip(point_counts, effective_counts)):
                raise ValueError('Effective target count exceeds raw target count')
        draw_by_id[identity] = draw
    supplied = record.get('matching_stability')
    if supplied is not None:
        ordered = sorted(draw_by_id.values(), key=lambda d: d['draw'])
        enabled = len(ordered) == 2 and all(d.get('u_supervised', True) for d in ordered)
        if (len(ordered) != 2 or supplied['n_instances'] != n_gt or supplied['n_queries'] != n_queries
                or supplied['u_supervised'] != enabled or supplied['capacity_overflow'] != max(n_gt - n_queries, 0)
                or supplied['assignment_denominator'] != (n_gt if enabled else None)):
            raise ValueError('Supplied matching stability context differs')
        counts = _pair_counts(*ordered, n_gt)
        for key, value in counts.items():
            if supplied[key] != (value if enabled else None):
                raise ValueError('Supplied matching stability assignments differ')
        for side in ('first', 'second'):
            for key in ('zero_positive_targets', 'zero_effective_positive_targets'):
                if _int(supplied[side + '_' + key]) > n_gt:
                    raise ValueError('Invalid zero-positive target count')
                draw = ordered[0 if side == 'first' else 1]
                values = draw.get('target_point_counts' if key == 'zero_positive_targets' else 'target_effective_point_counts')
                if values is not None and supplied[side + '_' + key] != sum(v == 0 for v in values):
                    raise ValueError('Supplied zero-target counts differ from probe')
    for component in record['components']:
        status = component['status']
        q = _int(component['query'])
        outcome = outcomes[component['instance']]
        if (status not in _STATUSES or q >= n_queries or status != outcome['status']
                or component.get('official_gt_instance') != outcome.get('official_gt_instance')):
            raise ValueError('Aggregate detection join differs')
        cd = component['draws']
        if len(cd) != len(draw_by_id) or {d['draw'] for d in cd} != draw_by_id.keys():
            raise ValueError('Aggregate component draw join differs')
        for d in cd:
            draw = draw_by_id[d['draw']]
            target = draw['query_target_instance'][q]
            expected = 'unsupervised' if not draw.get('u_supervised', True) else 'unmatched' if target is None else 'matched'
            if (d['target_instance'] != target or d['query_status'] != expected
                    or d['official_target_relation'] not in ('same_target', 'different_target', 'no_official_target',
                                                           'unmatched_query', 'unsupervised')):
                raise ValueError('Aggregate training join differs')
        # Validate every consumed scalar before any summary; unknown private keys are discarded.
        for _, item in [(None, component)] + [(None, d) for d in cd]:
            for key in _DENSE + _DRAW:
                if item.get(key) is not None:
                    _number(item[key])


def aggregate_records(records):
    """Return a fixed scalar/category whitelist, without private rows or identities.

    A record is one crop. ``row_ordinal`` is optional private bookkeeping; if it
    is absent, row count is explicitly undefined. Repeated components within a
    crop/query first average into one observation in query-weighted summaries.
    Coverage/gradient query weights additionally distinguish point draws.
    """
    records = list(records)
    for record in records:
        _validate_record(record)
    rows_defined = all('row_ordinal' in r for r in records)
    result = dict(evidence='NOT_EVIDENCE', scope='Frozen TRAIN crops; descriptive diagnostic only',
                  n_rows=len({r['row_ordinal'] for r in records}) if rows_defined else None,
                  row_count_defined=rows_defined, **_summarize(records))
    result['source_sequence'] = {s + '/' + q: _summarize([r for r in records if (r['source'], r['sequence']) == (s, q)])
                                 for s, q in _SOURCES}
    result['gt_present'] = {label: _summarize([r for r in records if bool(r['n_instances']) == present])
                            for label, present in (('present', True), ('empty', False))}
    result['weighting'] = dict(component_weighted='One observation per component; draw summaries per component/draw',
        query_weighted='Mean of components per crop/query; draw summaries per crop/query/draw',
        query_supervision='One observation per crop/query/draw, including queries without components',
        fractions='Dense: valid component voxels; coverage: valid positive-weight component voxels',
        dependence='Components, queries, draws and crops are not independent patients')
    result['limitations'] = ['No historical training causality or generalization claim.',
        'Sampled logit gradients are local detached-loss gradients, not network-parameter gradients.',
        'A zero sampled coverage does not establish absence of shared-parameter learning.',
        'Fixed-cutoff FP/crop is not full-volume FP/scan or an S7 gate verdict.',
        'A/R remain pseudo-reference tasks without Level R human evidence.']
    return result
