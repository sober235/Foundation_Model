"""Registered AUR engineering acceptance, using pseudo-references: NOT_EVIDENCE.

Functions consume private records and return anonymous aggregates. No inference,
threshold search, patient relabeling, or clinical claim is performed here.
"""
import math
import re

import numpy as np

from anatobind.anatomy.labels import HOST_IDS, NAMES
from anatobind.aur.data_contract import json_digest
from anatobind.aur.gate_metrics import (class_presence, host_dice_by_host,
                                       support_arrays, visible_classes)
from anatobind.aur.labels import ENTITY_LABELS

GATE_V1 = {'version': 'aur_gate_v1', 'evidence': 'NOT_EVIDENCE',
           'a': {'presence_rate_min': .95, 'presence_fraction': .05,
                 'host_macro_min': .80, 'host_min': .60, 'visible_area_mm2': 20.},
           'u': {'margin': .05, 'matched_dice_median_min': .5, 'bbox_iou': .1,
                 'sources': ['pdgm', 'bmsr', 'isles']},
           'r': {'agreement_min': .90, 'ambiguous_share': .6},
           'two_d': {'stride_mm': 8., 'u_margin_vs_3d': .05}}
_PROTOCOL_SHA256 = json_digest(GATE_V1)
SOURCES = ('pdgm', 'bmsr', 'isles')
_HOST_NAMES = tuple(NAMES[k] for k in HOST_IDS)
_ERRORS = (KeyError, TypeError, ValueError, IndexError, OverflowError)


def _finite(value, low=0., high=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < low or (high is not None and value > high)):
        raise ValueError('Invalid gate measurement')
    return float(value)


def _integer(value, low=0):
    if type(value) is not int or value < low:
        raise ValueError('Invalid gate count')
    return value


def validate_registration(registration):
    """Require the fixed protocol and a validation-selected deployed R track.

    The caller binds this document's digest before evaluation. The boolean is a
    declaration, not a replacement for the upstream frozen digest commitment.
    """
    try:
        if (json_digest(registration['gate']) != _PROTOCOL_SHA256
                or registration['relation_implementation'] not in ('head', 'lookup')
                or registration['frozen_before_evaluation'] is not True
                or not re.fullmatch('[0-9a-f]{64}', registration['validation_selection_sha256'])
                or set(registration['score_thresholds']) != set(SOURCES)):
            raise ValueError()
        for value in registration['score_thresholds'].values(): _finite(value, 0., 1.)
    except _ERRORS:
        raise ValueError('Invalid frozen gate registration') from None
    return registration


def _protocol(registration):
    protocol = validate_registration(registration)['gate'] if 'gate' in registration else registration
    if json_digest(protocol) != _PROTOCOL_SHA256:
        raise ValueError('Gate protocol differs from the registered version')
    return protocol


def _status(values):
    states = list(values)
    return 'unavailable' if not states or 'unavailable' in states else 'pass' if all(s == 'pass' for s in states) else 'fail'


def _unavailable():
    return {'status': 'unavailable', 'evidence': 'NOT_EVIDENCE'}


def _observation(record):
    return record.get('observation', record)


def _row(record):
    row = _observation(record)['row_sha256']
    if not isinstance(row, str) or not row:
        raise ValueError('Missing observation identity')
    return row


def _mode_records(records, mode):
    if mode not in ('2d', '3d'): raise ValueError('Invalid input mode')
    result = [r for r in records if r['input_mode'] == mode]
    seen = set()
    for r in result:
        obs = _observation(r)
        if obs.get('input_mode', mode) != mode:
            raise ValueError('Observation mode differs')
        kinds = obs['plane_kinds']
        if mode == '3d':
            if kinds != ['volume'] or obs['plane_index'] is not None:
                raise ValueError('Invalid volume observation')
        elif (not kinds or not set(kinds) <= {'stride', 'lesion_centroid'}
              or len(set(kinds)) != len(kinds)):
            raise ValueError('Invalid planar observation')
        else:
            _integer(obs['plane_index'])
        if r['source'] not in (*SOURCES, 'sibbms'):
            raise ValueError('Unexpected evaluation source')
        key = (_row(r), obs.get('plane_index'))
        if key in seen: raise ValueError('Duplicate gate observation')
        seen.add(key)
    return result


def judge_a(records, mode, registration=GATE_V1):
    """Mean of observation macros; each host and each A32 class needs support."""
    try:
        protocol = _protocol(registration); cfg = protocol['a']
        selected = [r for r in _mode_records(records, mode) if r.get('a') is not None]
        presence = [[] for _ in ENTITY_LABELS]; hosts = [[] for _ in HOST_IDS]
        entity_dice = [[] for _ in ENTITY_LABELS]; macros = []
        pooled = {name: np.zeros((3, n), np.float64) for name, n in [('entities', 32), ('hosts', 13)]}
        for record in selected:
            a = record['a']; entities, h = a['entities'], a['hosts']
            if tuple(entities['labels']) != ENTITY_LABELS or tuple(h['labels']) != _HOST_NAMES:
                raise ValueError('Incomplete anatomy classes')
            for name in pooled: pooled[name] += np.stack(support_arrays(a[name]))
            ep = class_presence(entities, fraction=cfg['presence_fraction'])
            hd, ed = host_dice_by_host(h), host_dice_by_host(entities)
            if mode == '2d':
                area = _observation(record)['pixel_area_mm2']
                ev = visible_classes(entities, area, min_area_mm2=cfg['visible_area_mm2'])
                hv = visible_classes(h, area, min_area_mm2=cfg['visible_area_mm2'])
                ep = [v if visible else None for v, visible in zip(ep, ev)]
                hd = [v if visible else None for v, visible in zip(hd, hv)]
                ed = [v if visible else None for v, visible in zip(ed, ev)]
            for values, value in zip(presence, ep):
                if value is not None: values.append(value)
            for values, value in zip(hosts, hd):
                if value is not None: values.append(value)
            for values, value in zip(entity_dice, ed):
                if value is not None: values.append(value)
            supported = [v for v in hd if v is not None]
            if supported: macros.append(float(np.mean(supported)))
        rates = [float(np.mean(v)) if v else None for v in presence]
        host_means = [float(np.mean(v)) if v else None for v in hosts]
        complete = all(v is not None for v in rates + host_means) and bool(macros)
        macro = float(np.mean(macros)) if macros else None
        minimum = min(host_means) if all(v is not None for v in host_means) else None
        rate_min = min(rates) if all(v is not None for v in rates) else None
        passed = complete and rate_min >= cfg['presence_rate_min'] and macro >= cfg['host_macro_min'] and minimum >= cfg['host_min']
        pooled_dice = {}
        for name, (truth, predicted, intersection) in pooled.items():
            pooled_dice[name] = [float(2 * i / (t + p)) if t + p else None for t, p, i in zip(truth, predicted, intersection)]
        return dict(status=('pass' if passed else 'fail') if complete else 'unavailable',
                    n_observations=len(selected), host_macro_observations=len(macros),
                    presence_rate=rates, presence_rate_min=rate_min,
                    presence_denominators=[len(v) for v in presence], host_macro=macro, host_min=minimum,
                    host_dice=host_means, host_denominators=[len(v) for v in hosts],
                    entity_dice=[float(np.mean(v)) if v else None for v in entity_dice],
                    pooled_dice=pooled_dice, thresholds=cfg, evidence='NOT_EVIDENCE')
    except _ERRORS:
        return _unavailable()


def _id_map(values):
    if not isinstance(values, dict): raise ValueError('Invalid instance measurements')
    result = {}
    for key, value in values.items():
        if isinstance(key, bool) or str(int(key)) != str(key) or int(key) <= 0 or int(key) in result:
            raise ValueError('Invalid instance identity')
        result[int(key)] = value
    return result


def _u(record):
    value = record['u']
    n_gt, n_hit, n_fp = (_integer(value[k]) for k in ('n_gt', 'n_hit', 'n_fp'))
    matches, dice = _id_map(value['matches']), _id_map(value['matched_dice'])
    if (len(matches) != n_hit or set(matches) != set(dice) or n_hit > n_gt
            or any(k > n_gt for k in matches) or len(set(matches.values())) != len(matches)):
        raise ValueError('Inconsistent detection measurements')
    for k in matches:
        _integer(matches[k], 1); _finite(dice[k], 0., 1.)
    return value, matches, dice, n_gt, n_hit, n_fp


def _reference(ref):
    ids = ref['included_row_sha256']
    n, gt, hit, fp = (_integer(ref[k]) for k in ('n_scans', 'n_gt', 'n_hit', 'n_fp'))
    sensitivity, budget = _finite(ref['sensitivity'], 0., 1.), _finite(ref['fp_per_scan'])
    if (ref['grid'] != 'aur_1mm_canonical' or not n or not gt or hit > gt
            or not isinstance(ids, list) or len(ids) != n or len(set(ids)) != n
            or any(not isinstance(v, str) or not v for v in ids)
            or not math.isclose(sensitivity, hit / gt, abs_tol=1e-9, rel_tol=0)
            or not math.isclose(budget, fp / n, abs_tol=1e-9, rel_tol=0)):
        raise ValueError('Invalid reference cohort')
    return set(ids), gt, sensitivity, budget


def judge_u_3d(records, reference, registration=GATE_V1):
    sources = {}
    try:
        cfg = _protocol(registration)['u']
        if not set(reference) <= set(SOURCES): raise ValueError('Unexpected reference source')
        selected = _mode_records(records, '3d')
    except _ERRORS:
        return _unavailable()
    for source in SOURCES:
        try:
            if reference[source]['source'] != source: raise ValueError('Reference source differs')
            ids, reference_gt, reference_sensitivity, budget = _reference(reference[source])
            rows = [r for r in selected if r['source'] == source and r.get('u') is not None]
            kept = [r for r in rows if _row(r) in ids]
            if {_row(r) for r in kept} != ids: raise ValueError()
            values = [_u(r) for r in kept]
            gt, hit, fp = (sum(v[i] for v in values) for i in (3, 4, 5))
            dice = [d for v in values for d in v[2].values()]
            if gt != reference_gt or not gt or not dice: raise ValueError()
            sensitivity, fps, median = hit / gt, fp / len(kept), float(np.median(dice))
            passed = (fps <= budget + 1e-9 and sensitivity >= reference_sensitivity - cfg['margin']
                      and median >= cfg['matched_dice_median_min'])
            sources[source] = dict(status='pass' if passed else 'fail', sensitivity=sensitivity,
                fp_per_scan=fps, matched_dice_median=median, n_scans=len(kept), n_gt=gt, n_hit=hit, n_fp=fp,
                n_matched_masks=len(dice), reference_sensitivity=reference_sensitivity,
                reference_fp_per_scan=budget, excluded_records=len(rows)-len(kept))
        except _ERRORS:
            sources[source] = _unavailable()
    return dict(status=_status(v['status'] for v in sources.values()), sources=sources, evidence='NOT_EVIDENCE')


def _centroid_records(records):
    return [r for r in _mode_records(records, '2d') if r.get('u') is not None
            and 'lesion_centroid' in _observation(r)['plane_kinds']]


def judge_u_2d(records_2d, records_3d, registration=GATE_V1):
    sources = {}; total_fp = total_images = 0
    try:
        cfg = _protocol(registration)
        planes = _centroid_records(records_2d)
        volumes = [r for r in _mode_records(records_3d, '3d') if r.get('u') is not None]
    except _ERRORS:
        return _unavailable()
    for source in SOURCES:
        try:
            by_row = {_row(r): _u(r) for r in volumes if r['source'] == source}
            expected = {(key, k) for key, v in by_row.items() for k in range(1, v[3] + 1)}
            if not expected: raise ValueError()
            observed = set(); hits2 = hits3 = fp = images = 0; dice = []
            for record in (r for r in planes if r['source'] == source):
                key = _row(record); value, matches, md, _, _, nfp = _u(record)
                targets = _observation(record)['target_instances']; source_ids = value['source_ids']
                if key not in by_row or len(source_ids) != value['n_gt'] or not targets: raise ValueError()
                fp += nfp; images += 1
                for target in targets:
                    _integer(target, 1); pair = (key, target)
                    if pair not in expected or pair in observed: raise ValueError()
                    observed.add(pair)
                    eligible = [i + 1 for i, sid in enumerate(source_ids) if sid == target]
                    hit = [i for i in eligible if i in matches]
                    hits2 += bool(hit); hits3 += target in by_row[key][1]
                    dice.extend(md[i] for i in hit)
            if observed != expected or not dice: raise ValueError()
            n = len(expected); sens2, sens3 = hits2 / n, hits3 / n; median = float(np.median(dice))
            passed = sens2 >= sens3 - cfg['two_d']['u_margin_vs_3d'] and median >= cfg['u']['matched_dice_median_min']
            sources[source] = dict(status='pass' if passed else 'fail', n_gt=n, n_hit_2d=hits2,
                n_hit_3d=hits3, sensitivity_2d=sens2, sensitivity_3d=sens3, n_images=images, n_fp=fp,
                fp_per_image=fp/images if images else None, matched_dice_median=median, n_matched_masks=len(dice))
            total_fp += fp; total_images += images
        except _ERRORS:
            sources[source] = _unavailable()
    return dict(status=_status(v['status'] for v in sources.values()), sources=sources,
                fp_per_image=total_fp/total_images if total_images else None, evidence='NOT_EVIDENCE')


def judge_r(records, mode, registration=GATE_V1):
    try:
        reg = validate_registration(registration); cfg = reg['gate']['r']
        chosen = 'end_to_end' if reg['relation_implementation'] == 'head' else 'end_to_end_b0'
        selected = _centroid_records(records) if mode == '2d' else _mode_records(records, mode)
        n = hits = head_hits = lookup_hits = ambiguous = unmatched = unknown = 0
        seen_targets = set()
        for record in selected:
            task = record.get('r_inplane' if mode == '2d' else 'r')
            if task is None: continue
            value, matches, _, _, _, _ = _u(record)
            if len(task['instances']) + _integer(task.get('unavailable_instances', 0)) != value['n_gt']:
                raise ValueError('Incomplete relation references')
            source_ids = value['source_ids'] if mode == '2d' else None
            targets = _observation(record)['target_instances'] if mode == '2d' else None
            if mode == '2d':
                for target in targets:
                    key = (_row(record), target)
                    if key in seen_targets: raise ValueError()
                    seen_targets.add(key)
            ids = set()
            for row in task['instances']:
                instance = _integer(row['instance'], 1)
                if instance in ids: raise ValueError()
                ids.add(instance)
                if mode == '2d' and (instance > len(source_ids) or source_ids[instance-1] not in targets): continue
                truth = row['truth']
                if truth is None or truth == -1: unknown += 1; continue
                _integer(truth)
                if truth > 13: raise ValueError()
                share = _finite(row['top_share'], 0., 1.)
                if share < cfg['ambiguous_share']: ambiguous += 1; continue
                detected = row['detected_instance']
                if detected is None:
                    if instance in matches: raise ValueError('Hidden relation match')
                    unmatched += 1
                    continue
                if matches.get(instance) != detected: raise ValueError()
                for field in ('end_to_end', 'end_to_end_b0'):
                    _integer(row[field])
                    if row[field] > 13: raise ValueError()
                n += 1; hits += row[chosen] == truth
                head_hits += row['end_to_end'] == truth; lookup_hits += row['end_to_end_b0'] == truth
        agreement = hits/n if n else None
        return dict(status=('pass' if agreement >= cfg['agreement_min'] else 'fail') if n else 'unavailable',
                    n=n, hits=hits, agreement=agreement, head_agreement=head_hits/n if n else None,
                    b0_agreement=lookup_hits/n if n else None, ambiguous=ambiguous, unmatched=unmatched,
                    unknown=unknown, implementation=reg['relation_implementation'],
                    denominator='matched_planar_instances_on_centroid_planes' if mode == '2d' else 'matched_volume_instances',
                    evidence='NOT_EVIDENCE')
    except _ERRORS:
        return _unavailable()


def verdict(records, reference, *, registration):
    """Compute all six gates from records using an explicit frozen registration."""
    validate_registration(registration)
    # U modes use the same S7-covered row intersection. Other tasks retain their
    # own denominators; absence of S7 is not absence of A or R supervision.
    try:
        covered = {s: _reference(reference[s])[0] for s in SOURCES}
        paired = [r for r in records if r['source'] in covered and _row(r) in covered[r['source']]]
    except _ERRORS:
        paired = []
    modes = {}
    for mode in ('3d', '2d'):
        modes[mode] = dict(a=judge_a(records, mode, registration),
            u=judge_u_3d(records, reference, registration) if mode == '3d' else judge_u_2d(paired, paired, registration),
            r=judge_r(records, mode, registration))
    return dict(version=GATE_V1['version'], evidence='NOT_EVIDENCE',
                registration_sha256=json_digest(registration), thresholds=registration['gate'], modes=modes,
                overall=_status(v['status'] for m in modes.values() for v in m.values()))
