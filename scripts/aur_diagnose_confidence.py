#!/usr/bin/env python
"""Frozen six-case residual confidence measurement, not a new decoder or calibration run."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import aur_compare_decoders as C
from anatobind.aur import eval as E, infer as I
from anatobind.aur.confidence_diagnostics import ResidualConfidenceAudit, array_digest
from anatobind.aur.dataset import load_volume
from anatobind.aur.diagnostics import identity, select_cases
from anatobind.aur.labels import entity_to_synthseg
from anatobind.aur.query_decoder import decoder_rules
from anatobind.aur.ssl.checkpoint import code_sha, file_sha256
from anatobind.aur.targets import host_targets
from anatobind.aur.train import split_rows
from anatobind.eval.detection_metrics import match_scan

DATA_ROOT = Path('/data2/congcong/data/FM_data')
REFERENCE_SIX = Path('/home/congcongliu/outputs/aur_query_cc_20261017/six_1874af2')
REFERENCE_SUMMARY = ROOT / 'docs/verification/2026-10-17/aur_query_cc/six_probe_accelerated.json'
REFERENCE_CODE_SHA = '1874af20a460cf86c46029cc83d2df0801a10910'
CKPT_SHA, SAMPLES_SHA, SCORES = C.CKPT_SHA, C.SAMPLES_SHA, C.SCORES
SCORE_NAMES = ('parent_score', 'child_raw_mean', 'child_fused_mean')
SOURCE_HASHES = ROOT / 'docs/superpowers/specs/2026-10-17-aur-residual-confidence-sources.json'


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    wire = json.loads(json.dumps(value, default=C.serial))
    return hashlib.sha256(json.dumps(wire, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def records_digest(records):
    parts = sorted(hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in records)
    return hashlib.sha256(''.join(parts).encode()).hexdigest()


def token(row):
    return hashlib.sha256(json.dumps(identity(row), separators=(',', ':')).encode()).hexdigest()


def exact(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        a, b = {str(k): v for k, v in a.items()}, {str(k): v for k, v in b.items()}
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list, np.ndarray)) and isinstance(b, (tuple, list, np.ndarray)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)) and np.isnan(a) and np.isnan(b):
        return True
    return bool(a == b)


def score_copies(dets, children, name):
    if name not in SCORE_NAMES:
        raise ValueError('Unknown diagnostic score')
    by_id = {r['instance']: r for r in children}
    ids = [d['instance'] for d in dets]
    if len(by_id) != len(children) or len(set(ids)) != len(ids) or set(ids) != by_id.keys():
        raise ValueError('Measurement and original child membership differ')
    result = deepcopy(dets)
    for d in result:
        r = by_id[d['instance']]
        if r['parent_score'] != d['score'] or not np.isfinite(r[name]) or not 0 <= r[name] <= 1:
            raise ValueError('Invalid score measurement or changed original score')
        d['score'] = float(r[name])
    return result


def classify_original(gt, dets, threshold):
    hits, fp, kept = E.scan_matches(E.scan('anonymous', gt, dets), threshold)
    labels = {d['instance']: 'below_original_threshold' for d in dets}
    used = set(hits.values())
    rest = [i for i in range(len(kept)) if i not in used]
    ignored = [g for g in gt if g.get('ignore')]
    pairs = match_scan(ignored, [kept[i] for i in rest]) if ignored and rest else {}
    excused = {rest[i] for i in pairs.values()}
    for i, d in enumerate(kept):
        labels[d['instance']] = 'TP' if i in used else 'ignored_excused' if i in excused else 'FP'
    if sum(v == 'FP' for v in labels.values()) != fp:
        raise ValueError('Diagnostic labels differ from official matching')
    return labels


def describe(values):
    if not values:
        return {'n': 0}
    v = np.asarray(values, np.float64)
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                q05=float(np.quantile(v, .05)), q95=float(np.quantile(v, .95)))


def child_summary(children, floor):
    return dict(n_children=len(children),
        scores={s: describe([c[s] for c in children]) for s in SCORE_NAMES},
        parent_minus_raw=describe([c['parent_score'] - c['child_raw_mean'] for c in children]),
        parent_minus_fused=describe([c['parent_score'] - c['child_fused_mean'] for c in children]),
        retained_fraction=describe([c['retained_fraction'] for c in children]),
        raw_below_fixed_score=sum(c['child_raw_mean'] < floor for c in children),
        fused_below_fixed_score=sum(c['child_fused_mean'] < floor for c in children))


def aggregate_records(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['source'] + '/' + r['sequence']].append(r)
    out = dict(evidence='NOT_EVIDENCE', scope='Previously exposed six development-val cases; mechanism diagnostic only',
               n_rows=len(records), score_points=SCORES, source_sequence={},
               physical_size_definition='Original final child volume; equivalent spherical diameter in mm: <5, 5-10, >=10')
    for key, cases in sorted(groups.items()):
        floor = SCORES[cases[0]['source']]
        variants = {}
        for name in SCORE_NAMES:
            scans = [E.scan('anonymous', r['gt'], score_copies(r['detections'], r['measurement']['children'], name)) for r in cases]
            variants[name] = E.sensitivity_at_threshold(scans, floor)
        child_groups = defaultdict(list)
        parents = []
        for case in cases:
            labels = classify_original(case['gt'], case['detections'], floor)
            strata = {d['instance']: E.size_stratum(d['volume_mm3']) for d in case['detections']}
            for c in case['measurement']['children']:
                child_groups[labels[c['instance']]].append({**c, 'stratum': strata[c['instance']]})
            parents.extend(case['measurement']['parents'])
        summaries = {}
        for label in ('TP', 'FP', 'ignored_excused', 'below_original_threshold'):
            children = child_groups[label]
            bins = {'<=0.1': [c for c in children if c['retained_fraction'] <= .1],
                    '(0.1,0.5]': [c for c in children if .1 < c['retained_fraction'] <= .5],
                    '>0.5': [c for c in children if c['retained_fraction'] > .5]}
            summaries[label] = dict(**child_summary(children, floor),
                retained_fraction_bins={b: len(cs) for b, cs in bins.items()},
                retained_fraction_score_summaries={b: child_summary(cs, floor) for b, cs in bins.items()},
                physical_size_strata={s: child_summary([c for c in children if c['stratum'] == s], floor)
                                      for s in E.STRATA})
        removed = [p for p in parents if p['removed_voxels']]
        out['source_sequence'][key] = dict(n_scans=len(cases), fixed_score_threshold=floor,
            hypothetical_score_filters_after_frozen_ownership=variants, original_point_groups=summaries,
            parent_decomposition=dict(n_active_parents=len(parents), n_with_removed_voxels=len(removed),
                parent_minus_retained_mean=describe([p['parent_score'] - p['retained_raw_mean'] for p in parents]),
                removed_minus_retained_mean=describe([p['removed_raw_mean'] - p['retained_raw_mean'] for p in removed]),
                removed_raw_mean=describe([p['removed_raw_mean'] for p in removed])))
    out['controls'] = dict(A_saved_scores_exact=all(r['controls']['A_saved_scores_exact'] for r in records),
        controlled_R_instances=sum(r['controls']['controlled_R_instances'] for r in records),
        controlled_R_outputs_and_origins_exact=all(r['controls']['controlled_R_outputs_and_origins_exact'] for r in records),
        first_windows=sum(r['measurement']['checks']['first_windows'] for r in records),
        replay_windows=sum(r['measurement']['checks']['replay_windows'] for r in records),
        present_query_planes_per_pass=sum(r['measurement']['checks']['present_query_planes'] for r in records),
        returned_maps_exact=all(r['controls']['returned_maps_exact'] for r in records),
        raw_probability_planes_exact=all(r['measurement']['checks']['raw_probability_planes_exact'] for r in records))
    out['resources'] = dict(recorded_row_seconds=sum(r['seconds'] for r in records),
                            peak_cuda_allocated_gib=max((r['peak_cuda_gib'] for r in records), default=0.))
    out['limitations'] = ['Six cases are not representative and are previously exposed development validation.',
        'No calibration, threshold selection or production decoder change; no full-val/test efficacy claim.',
        'The fused control mixes natural windows/queries and differs from the originating-query raw probability.',
        'Removed parent voxels combine ownership and post-ownership floor losses, not separately identified.',
        'Raw means are not calibrated correctness probabilities. A/R use pseudo references.']
    return out


def prepare_reference(a):
    source, public = a.reference_six, read(REFERENCE_SUMMARY)
    anchor = public['provenance'][0]
    if (file_sha256(source / 'contract.json') != anchor['contract_sha256']
            or file_sha256(source / 'COMPLETE.json') != anchor['complete_sha256']):
        raise ValueError('Original reference contract/completion changed')
    old, done = read(source / 'contract.json'), read(source / 'COMPLETE.json')
    if (old['mode'] != 'six' or old['n_rows'] != 6 or done['status'] != 'complete'
            or not done['all_reproduction_checks_passed'] or done['rows'] != 6
            or old['code_sha'] != REFERENCE_CODE_SHA or public['frozen_code_sha'] != REFERENCE_CODE_SHA
            or old['decoder_rules'] != decoder_rules('query_cc_v1') or old['score_thresholds'] != SCORES
            or old['crop'] != [128, 160, 160] or old['batch_size'] != 2):
        raise ValueError('Reference is not the frozen completed six-query replay')
    if (file_sha256(a.checkpoint) != old['checkpoint_sha256'] or old['checkpoint_sha256'] != CKPT_SHA
            or file_sha256(a.samples) != old['samples_sha256'] or old['samples_sha256'] != SAMPLES_SHA
            or file_sha256(a.val_patients) != old['val_patients_sha256']
            or file_sha256(a.saved_eval / 'cases.json') != old['saved_cases_sha256']
            or file_sha256(source / 'reference_frozen.json') != old['reference_sha256']):
        raise ValueError('Checkpoint/data/reference hash differs from original freeze')
    _, val = split_rows(read(a.samples), read(a.val_patients))
    saved = read(a.saved_eval / 'cases.json')
    if {identity(r) for r in val} != {identity(r) for r in saved}:
        raise ValueError('Frozen validation membership changed')
    rows = read(source / 'selection_private.json')
    selection_sha = hashlib.sha256(json.dumps([identity(r) for r in rows], ensure_ascii=False,
                                             separators=(',', ':')).encode()).hexdigest()
    if selection_sha != old['selection_sha256'] or not exact(rows, select_cases(val, saved)):
        raise ValueError('Original six selection changed')
    for row in rows:
        for key in ('image', 'anatomy', 'lesion'):
            if row.get(key) and not Path(row[key]).resolve().is_relative_to(DATA_ROOT.resolve()):
                raise ValueError('Raw input is outside the authorized data root')
    original_manifest = read(source / 'input_manifest_private.json')
    current_manifest = C.input_manifest(rows)
    if digest(original_manifest) != old['raw_inputs_sha256'] or current_manifest != original_manifest:
        raise ValueError('An originally frozen raw input changed, including unfinished rows')
    paths = sorted((source / 'row_records').glob('*.json'))
    records = [read(p) for p in paths]
    if records_digest(records) != anchor['row_records_content_sha256']:
        raise ValueError('Original reference row content changed')
    by_id = {identity(r['result']): r for r in records}
    if len(records) != 6 or len(by_id) != 6 or set(by_id) != {identity(r) for r in rows}:
        raise ValueError('Original reference row membership changed')
    reference_files = {}
    for path, record in zip(paths, records):
        key = token(record['result'])
        if (path.name != key + '.json' or record['code_sha'] != old['code_sha']
                or record['input_sha256'] != original_manifest[key]):
            raise ValueError('Original reference row provenance changed')
        reference_files[key] = file_sha256(path)
    if (sum(r['result']['n_windows'] for r in records) != public['inference_resources']['windows']
            or sum(len((r['result'].get('r') or {}).get('instances', [])) for r in records)
            != public['invariance']['controlled_instances']):
        raise ValueError('Original aggregate and private reference counts disagree')
    pinned = read(SOURCE_HASHES)
    if pinned['code_sha'] != REFERENCE_CODE_SHA:
        raise ValueError('Frozen dependency manifest refers to a different source revision')
    core_hashes = pinned['files']
    for path, expected_hash in core_hashes.items():
        if expected_hash != file_sha256(ROOT / path):
            raise ValueError('A frozen inference/model dependency changed')
    contract = dict(code_sha=code_sha(), checkpoint_sha256=CKPT_SHA, samples_sha256=SAMPLES_SHA,
        val_patients_sha256=file_sha256(a.val_patients), raw_inputs_sha256=digest(current_manifest),
        selection_sha256=selection_sha, reference_summary_sha256=file_sha256(REFERENCE_SUMMARY),
        reference_contract_sha256=anchor['contract_sha256'], reference_complete_sha256=anchor['complete_sha256'],
        reference_rows_content_sha256=anchor['row_records_content_sha256'], reference_row_files=reference_files,
        reference_native_snapshot_sha256=old['reference_sha256'], decoder_rules=old['decoder_rules'],
        reference_source_manifest_sha256=file_sha256(SOURCE_HASHES),
        diagnostic_source_sha256={p: file_sha256(ROOT / p) for p in
            ('scripts/aur_diagnose_confidence.py', 'anatobind/aur/confidence_diagnostics.py')},
        core_dependency_sha256=core_hashes, crop=old['crop'], batch_size=2, n_rows=6, score_points=SCORES,
        score_variants=list(SCORE_NAMES), max_seconds=min(a.max_seconds, 7200), mode='six_confidence_diagnostic',
        expected_windows_per_pass=public['inference_resources']['windows'],
        expected_controlled_R_instances=public['invariance']['controlled_instances'],
        measurement='exact float32 replay; float64 child bincount sums; original ownership/masks unchanged')
    return dict(rows=rows, records=by_id, manifest=current_manifest, contract=contract)


def replay_case(model, row, reference, crop, device):
    start = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    expected = reference['result']
    vol = load_volume(row)
    with ResidualConfidenceAudit(vol['image'].shape, vol['voxel_mm3']) as audit:
        first = I.predict_volume(model, vol['image'], vol['affine'], crop, device, batch_size=2,
                                 observer=audit.observe_first)
        inst, dets, stats = audit.decode(first['lesion_prob'])
        pred_seg, true_seg = entity_to_synthseg(first['entity']), entity_to_synthseg(vol['entity'])
        a_scores = E.host_dice(pred_seg, true_seg)
        gt = E.instance_rows(vol['instance'], vol['voxel_mm3']) + E.ignored_rows(vol['small'], vol['voxel_mm3'])
        legacy = I.instances_from_probability(first['lesion_prob'], vol['voxel_mm3'])[1]
        if not (exact(a_scores, expected['a']) and exact(gt, expected['u']['gt'])
                and exact(dets, expected['u']['dets']) and exact(stats, expected['decoder_stats'])
                and exact(legacy, expected['legacy_reference_u']['dets'])
                and first['n_windows'] == expected['n_windows'] and exact(first['seq_probs'], expected['seq_probs'])):
            raise ValueError('Original A/U/query/GT reference did not reproduce exactly')
        signature = {k: array_digest(v) for k, v in first.items() if isinstance(v, np.ndarray)}
        second = I.predict_volume(model, vol['image'], vol['affine'], crop, device, batch_size=2,
                                  observer=audit.observe_replay)
        measured = audit.finish_replay()
        if signature != {k: array_digest(v) for k, v in second.items() if isinstance(v, np.ndarray)} or first['n_windows'] != second['n_windows']:
            raise ValueError('Returned A/legacy-U maps or context changed between passes')
        bindings = I.bind_instances(model, vol['image'], vol['affine'], vol['instance'], crop, device) if vol['r_supervised'] else []
        old_r = (expected.get('r') or {}).get('instances', [])
        by_gt = {r['instance']: r for r in gt if not r.get('ignore')}
        b0 = host_targets(vol['instance'], pred_seg, vol['spacing'])['host']
        observed_r = []
        for b in bindings:
            k = b['instance'];g = by_gt[k]
            observed_r.append(dict(instance=k, truth=int(vol['hosts']['host'][k - 1]), r=b['host'],
                b0=int(b0[k - 1]), r_probs=b['host_probs'].tolist(), query_iou=b['query_iou'],
                zero_overlap=b['zero_overlap'], in_crop_share=b['in_crop_share'],
                volume_mm3=g['volume_mm3'], stratum=g['stratum'], box=g['box'],
                local_dice=E.local_host_dice(pred_seg, true_seg, b['window']),
                binding_origin=dict(query=b['query'], window=[list(p) for p in b['window']], in_crop_share=b['in_crop_share'])))
        shared = [{k: v for k, v in r.items() if k not in ('matched_det', 'matched_score')} for r in old_r]
        if not exact(observed_r, shared):
            raise ValueError('Controlled R outputs or actual query/window origins changed')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return dict(source=row['source'], sequence=row['sequence'], gt=gt, detections=dets, measurement=measured,
        observed_controlled_R=observed_r, A_scores=a_scores, returned_map_hashes=signature,
        controls=dict(A_saved_scores_exact=True, original_U_rows_exact=True, returned_maps_exact=True,
                      controlled_R_instances=len(observed_r), controlled_R_outputs_and_origins_exact=True),
        seconds=time.monotonic() - start,
        peak_cuda_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.)


def validate_measurement(measurement, detections, n_windows):
    """Recheck saved decomposition and window evidence instead of trusting flags."""
    windows, checks = measurement['window_hashes'], measurement['checks']
    window_digest = hashlib.sha256(json.dumps(windows, sort_keys=True).encode()).hexdigest()
    if len(windows) != n_windows or checks['window_hashes_sha256'] != window_digest:
        raise ValueError('Cached window count/digest differs')
    by_window = {}
    def valid_hash(value):
        return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')
    for event in windows:
        window = event['window']
        if (len(window) != 3 or any(len(p) != 2 or p[0] >= p[1] or
                any(not isinstance(v, int) for v in p) for p in window)):
            raise ValueError('Cached window coordinates are invalid')
        key = tuple(tuple(p) for p in window)
        queries = event['queries']
        if (key in by_window or not valid_hash(event['presence_sha256']) or not valid_hash(event['valid_sha256'])
                or any(not str(q).isdigit() or str(int(q)) != str(q) or not valid_hash(h)
                       for q, h in queries.items())):
            raise ValueError('Cached window/query membership or hashes are invalid')
        by_window[key] = {int(q) for q in queries}
    if checks['present_query_planes'] != sum(len(qs) for qs in by_window.values()):
        raise ValueError('Cached present-query count differs')
    children = {c['instance']: c for c in measurement['children']}
    by_parent = defaultdict(list)
    for det in detections:
        origin = det['detector_origin'];window = tuple(tuple(p) for p in origin['window'])
        if window not in by_window or origin['query'] not in by_window[window]:
            raise ValueError('Cached child origin is absent from measured windows/queries')
        by_parent[origin['rank']].append(det)
    parents = {p['rank']: p for p in measurement['parents']}
    if len(parents) != len(measurement['parents']) or parents.keys() != by_parent.keys():
        raise ValueError('Cached active-parent membership differs')
    def probability(value):
        return value is not None and np.isfinite(value) and 0 <= value <= 1
    for rank, dets in by_parent.items():
        parent = parents[rank];origin = dets[0]['detector_origin']
        retained = sum(d['n_voxels'] for d in dets)
        removed = origin['parent_voxels'] - retained
        if (parent['parent_score'] != origin['parent_score'] or parent['parent_voxels'] != origin['parent_voxels']
                or parent['final_children'] != len(dets) or parent['retained_voxels'] != retained
                or retained <= 0 or removed < 0 or parent['removed_voxels'] != removed
                or any(d['score'] != parent['parent_score'] or d['detector_origin']['parent_voxels'] !=
                       parent['parent_voxels'] for d in dets)
                or not probability(parent['retained_raw_mean'])
                or (not probability(parent['removed_raw_mean']) if removed else parent['removed_raw_mean'] is not None)):
            raise ValueError('Cached parent counts/probabilities differ')
        child_mean = sum(children[d['instance']]['child_raw_mean'] * d['n_voxels'] for d in dets) / retained
        reconstructed = (retained * parent['retained_raw_mean'] +
                         (removed * parent['removed_raw_mean'] if removed else 0)) / parent['parent_voxels']
        if not (np.isclose(child_mean, parent['retained_raw_mean'], rtol=0, atol=1e-12)
                and np.isclose(reconstructed, parent['parent_score'], rtol=0, atol=1e-12)):
            raise ValueError('Cached parent/child probability decomposition differs')


def validate_cached(record, row, prepared):
    payload, meta = record['payload'], record['meta']
    key = token(row);contract = prepared['contract']
    if (digest(payload) != record['payload_sha256'] or meta['identity'] != list(identity(row))
            or meta['code_sha'] != contract['code_sha'] or meta['input_sha256'] != prepared['manifest'][key]
            or meta['reference_record_sha256'] != contract['reference_row_files'][key]):
        raise ValueError('Cached observation hash/provenance differs')
    for flag in ('A_saved_scores_exact', 'original_U_rows_exact', 'returned_maps_exact', 'controlled_R_outputs_and_origins_exact'):
        if payload['controls'][flag] is not True:
            raise ValueError('Cached controls did not pass')
    expected = prepared['records'][identity(row)]['result']
    shared_r = [{k: v for k, v in r.items() if k not in ('matched_det', 'matched_score')}
                for r in (expected.get('r') or {}).get('instances', [])]
    checks = payload['measurement']['checks']
    if (not exact(payload['observed_controlled_R'], shared_r)
            or payload['controls']['controlled_R_instances'] != len(shared_r)
            or checks['first_windows'] != expected['n_windows'] or checks['replay_windows'] != expected['n_windows']
            or any(checks.get(k) is not True for k in ('raw_probability_planes_exact',
                'original_parent_means_exact', 'decoded_rows_and_map_unchanged'))):
        raise ValueError('Cached R or measurement controls differ from the original reference')
    if not exact(payload['detections'], expected['u']['dets']) or not exact(payload['gt'], expected['u']['gt']):
        raise ValueError('Cached original U/GT changed')
    if not exact(payload['A_scores'], expected['a']):
        raise ValueError('Cached original A changed')
    by_id = {d['instance']: d for d in payload['detections']}
    for child in payload['measurement']['children']:
        original = by_id.get(child['instance'])
        if (original is None or child['n_voxels'] != original['n_voxels']
                or child['parent_voxels'] != original['detector_origin']['parent_voxels']
                or child['retained_fraction'] != original['detector_origin']['retained_fraction']):
            raise ValueError('Cached child geometry metadata changed')
    score_copies(payload['detections'], payload['measurement']['children'], 'child_raw_mean')
    score_copies(payload['detections'], payload['measurement']['children'], 'child_fused_mean')
    validate_measurement(payload['measurement'], payload['detections'], expected['n_windows'])


def run(a):
    started = time.monotonic()
    prepared = prepare_reference(a)
    C.freeze_contract(a.out, prepared['contract'], a.resume)
    records_dir = a.out / 'row_records';records_dir.mkdir(exist_ok=True)
    C.write(a.out / 'selection_private.json', prepared['rows'])
    C.write(a.out / 'input_manifest_private.json', prepared['manifest'])
    valid_names = {token(r) + '.json' for r in prepared['rows']}
    if any(p.name not in valid_names for p in records_dir.glob('*.json')):
        raise ValueError('Unexpected cached row membership')
    cached = {}
    for row in prepared['rows']:
        path = records_dir / (token(row) + '.json')
        if path.exists():
            record = read(path);validate_cached(record, row, prepared);cached[token(row)] = record
    completed = a.out / 'COMPLETE.json'
    if completed.exists():
        done = read(completed)
        if (len(cached) != 6 or done.get('status') != 'complete'
                or done['records_sha256'] != digest([cached[token(r)] for r in prepared['rows']])
                or done['aggregate_sha256'] != file_sha256(a.out / 'aggregate.json')):
            raise ValueError('Completed observation/report was changed')
        print('Frozen confidence replay already complete', flush=True)
        return 0
    model = None
    if len(cached) < 6:
        model, _ = I.model_from_export(a.checkpoint, torch.device('cuda', a.gpu))
    records = []
    for i, row in enumerate(prepared['rows']):
        key = token(row)
        if C.input_manifest([row])[key] != prepared['manifest'][key]:
            raise ValueError('Raw input changed after freeze')
        record = cached.get(key)
        if record is None:
            payload = replay_case(model, row, prepared['records'][identity(row)], tuple(prepared['contract']['crop']),
                                  torch.device('cuda', a.gpu))
            if C.input_manifest([row])[key] != prepared['manifest'][key]:
                raise ValueError('Raw input changed during measurement')
            record = dict(payload=payload, payload_sha256=digest(payload), meta=dict(identity=list(identity(row)),
                code_sha=prepared['contract']['code_sha'], input_sha256=prepared['manifest'][key],
                reference_record_sha256=prepared['contract']['reference_row_files'][key]))
            validate_cached(record, row, prepared)
            C.write(records_dir / (key + '.json'), record)
        records.append(record)
        progress=dict(completed=i + 1, total=6, source=row['source'], reproduction='PASS',
                      row_seconds=round(record['payload']['seconds'], 2), elapsed_seconds=round(time.monotonic() - started, 2))
        C.write(a.out / 'progress.json', progress);print(json.dumps(progress), flush=True)
    aggregate = aggregate_records([r['payload'] for r in records])
    checks = aggregate['controls'];contract = prepared['contract']
    if (checks['first_windows'] != contract['expected_windows_per_pass']
            or checks['replay_windows'] != contract['expected_windows_per_pass']
            or checks['controlled_R_instances'] != contract['expected_controlled_R_instances']):
        raise ValueError('Complete counts differ from the original six replay')
    aggregate['provenance'] = dict(contract_sha256=file_sha256(a.out / 'contract.json'),
        records_sha256=digest(records), code_sha=contract['code_sha'], checkpoint_sha256=CKPT_SHA,
        reference_summary_sha256=contract['reference_summary_sha256'])
    if (a.out / 'aggregate.json').exists() and not exact(read(a.out / 'aggregate.json'), aggregate):
        raise ValueError('Existing complete aggregate differs from frozen observations')
    C.write(a.out / 'aggregate.json', aggregate)
    C.write(completed, dict(status='complete', rows=6, all_reproduction_checks_passed=True,
        aggregate_sha256=file_sha256(a.out / 'aggregate.json'), records_sha256=digest(records),
        seconds=round(time.monotonic() - started, 2)))
    print('Complete six-case confidence diagnostic; all declared controls passed', flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--reference-six', type=Path, default=REFERENCE_SIX)
    p.add_argument('--samples', type=Path, default=C.DATA / 'samples_1mm_v1.json')
    p.add_argument('--val-patients', type=Path, default=C.DATA / 'ssl_manifest_v1/val_patients.json')
    p.add_argument('--checkpoint', type=Path, default=C.DATA / 'ssl_runs/c2_stage3_80k_20261014_1344/aur_stage3_best.pt')
    p.add_argument('--saved-eval', type=Path, default=C.SAVED)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-seconds', type=int, default=1800)
    a = p.parse_args()
    with C.hard_deadline(min(a.max_seconds, 7200)):
        return run(a)


if __name__ == '__main__':
    raise SystemExit(main())
