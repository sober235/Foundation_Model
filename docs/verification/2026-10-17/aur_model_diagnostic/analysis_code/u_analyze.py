#!/usr/bin/env python3
"""Anonymous, CPU-only analysis of frozen six-case U caches (no model imports)."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
try:
    os.nice(19)
except OSError:
    pass
import numpy as np
from scipy import ndimage


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def distribution(values):
    a = np.asarray(values, float)
    return dict(n=len(a), min=float(a.min()), q25=float(np.quantile(a, .25)), median=float(np.median(a)),
                q75=float(np.quantile(a, .75)), max=float(a.max()), mean=float(a.mean())) if len(a) else dict(n=0)


def cc_statistics(mask, local, gt_ids):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    ids = np.asarray(gt_ids, int)
    intersections = {int(k): np.bincount(lab[local == k], minlength=n + 1) for k in ids}
    touched = np.zeros(n + 1, int)
    for inter in intersections.values():
        touched += inter > 0
    rows = {}
    for k, inter in intersections.items():
        count = int((local == k).sum())
        ious = inter / np.maximum(sizes + count - inter, 1)
        ious[0] = 0
        c = int(ious.argmax())
        rows[k] = dict(max_cc_iou=float(ious[c]), max_cc_id=c, max_cc_voxels=int(sizes[c]) if c else 0,
                       max_cc_gt_tp=int(inter[c]) if c else 0, max_cc_gt_ids=int(touched[c]) if c else 0,
                       max_cc_touching_components=int(np.count_nonzero(inter[1:])),
                       max_cc_healthy_voxels=int(sizes[c] - sum(x[c] for x in intersections.values())) if c else 0)
    return rows, lab


def summarize_rows(rows):
    return dict(n=len(rows),
                best_all_iou=distribution([r['best_all_iou'] for r in rows]),
                best_kept_iou=distribution([r['best_kept_iou'] for r in rows]),
                local_max_union_iou=distribution([r['max_iou'] for r in rows]),
                best_max_cc_iou=distribution([r['max_cc_iou'] for r in rows]),
                best_all_ge_05=sum(r['best_all_iou'] >= .5 for r in rows),
                best_kept_ge_05=sum(r['best_kept_iou'] >= .5 for r in rows),
                best_max_cc_ge_05=sum(r['max_cc_iou'] >= .5 for r in rows),
                presence_reduces_iou=sum(r['best_all_iou'] > r['best_kept_iou'] + 1e-7 for r in rows),
                presence_loses_ge_05=sum(r['best_all_iou'] >= .5 > r['best_kept_iou'] for r in rows),
                max_cc_loses_kept_ge_05=sum(r['best_kept_iou'] >= .5 > r['max_cc_iou'] for r in rows),
                max_union_loses_kept_ge_05=sum(r['best_kept_iou'] >= .5 > r['max_iou'] for r in rows),
                max_cc_touches_multiple_gt=sum(r['max_cc_gt_ids'] >= 2 for r in rows),
                max_gt_voxel_recall=distribution([r['max_gt_tp'] / r['visible_voxels'] for r in rows]),
                mean_prob_max=distribution([r['max_gt_mean_prob'] for r in rows]),
                kept_minus_all_iou=distribution([r['best_kept_iou'] - r['best_all_iou'] for r in rows]),
                max_cc_minus_kept_iou=distribution([r['max_cc_iou'] - r['best_kept_iou'] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/home/congcongliu/outputs/aur_model_diagnostic/20261017_replay_b240962'))
    ap.add_argument('--output', type=Path, default=Path('/tmp/aur-six-u-analysis-20261017'))
    args = ap.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    meta = json.loads((args.input / 'summary.json').read_text())
    assert meta['status'] == 'complete' and len(meta['cases']) == 6 and all(c['reproduced'] for c in meta['cases'])
    all_rows, case_summaries, gt_summaries, stage_rows, window_summaries = [], [], [], [], []
    for case in meta['cases']:
        case_id, source = case['anonymous_case'], case['source']
        case_dir = args.input / f'case_{case_id:02d}'
        data = json.loads((case_dir / 'diagnostic.json').read_text())
        u = data['u']
        vol = np.load(case_dir / 'volume.npz')
        truth, final_prob, comps = vol['instance'], vol['lesion_prob'], vol['predicted_components']
        assert len(np.unique(truth[truth > 0])) == case['n_gt'] == len(u['gt'])
        shape = truth.shape
        acc, wsum = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        any_support = {t: np.zeros(shape, bool) for t in (.3, .5)}
        n_support = {t: np.zeros(shape, np.uint8) for t in (.3, .5)}
        case_rows = []
        whole_gt = {g['instance']: g for g in u['gt']}
        for wi, window in enumerate(u['windows']):
            z = np.load(case_dir / 'windows' / f'window_{wi:03d}.npz')
            maximum, valid, weight, local = z['maximum'], z['valid'].astype(bool), z['weight'], z['instance']
            window_arr = z['window']
            dst = tuple(slice(max(int(a), 0), min(int(b), int(n))) for (a, b), n in zip(window_arr, shape))
            src = tuple(slice(max(int(a), 0) - int(a), min(int(b), int(n)) - int(a)) for (a, b), n in zip(window_arr, shape))
            wb = weight * valid
            acc[dst] += maximum[src] * wb[src]
            wsum[dst] += wb[src]
            presence = z['presence']
            gt_ids = [g['instance'] for g in window['gt']]
            local_final = np.zeros(local.shape, np.float32)
            local_final[src] = final_prob[dst]
            for t, packed_name in ((.3, 'all_query_hard_03_packed'), (.5, 'all_query_hard_05_packed')):
                hard_max = ((maximum >= t) if t == .3 else (maximum > t)) & valid
                any_support[t][dst] |= hard_max[src]
                n_support[t][dst] += hard_max[src]
                cc, labels = cc_statistics(hard_max, local, gt_ids)
                packed = z[packed_name]
                packed_kept = packed[presence > .5]
                packed_union = np.bitwise_or.reduce(packed_kept, axis=0) if len(packed_kept) else np.zeros(packed.shape[1],np.uint8)
                assert np.array_equal(np.unpackbits(packed_union, count=local.size).reshape(local.shape).astype(bool), hard_max)
                cache, label_cache = {}, {}
                for g in window['gt']:
                    a, k = g[f'at_{t}'], g['instance']
                    sel = (local == k) & valid
                    q, qk = a['best_all_query'], a['best_kept_query']
                    for qid in {q, qk} - {-1}:
                        if qid not in cache:
                            mask = np.unpackbits(packed[qid], count=local.size).reshape(local.shape).astype(bool)
                            nv = int(mask.sum())
                            assert nv == window[f'at_{t}']['query_voxels'][qid]
                            cache[qid] = mask
                    qa = cache[q]
                    inter = int((qa & sel).sum())
                    assert inter == a['all_query_gt_intersection'][q]
                    assert np.isclose(inter / max(int(qa.sum()) + int(sel.sum()) - inter, 1), a['best_all_iou'], atol=2e-7)
                    # Strong same-window bridge diagnostic: one max CC intersects multiple distinct
                    # CCs of the best kept query, including one touching this GT.
                    bridge = dict(best_kept_query_components_joined_by_max=0, best_kept_gt_cc_iou=0.,
                                  gt_cc_before_voxels=0, gt_cc_before_gt_tp=0, gt_cc_after_voxels=0,
                                  gt_cc_after_gt_tp=0, gt_cc_added_non_gt=0, gt_cc_after_gt_ids=0)
                    if qk >= 0:
                        qmask = cache[qk]
                        if qk not in label_cache:
                            qlab, qn = ndimage.label(qmask, structure=np.ones((3, 3, 3), bool))
                            qsize = np.bincount(qlab.ravel(), minlength=qn + 1)
                            label_cache[qk] = (qlab,qsize,qn)
                        qlab,qsize,qn=label_cache[qk]
                        qinter = np.bincount(qlab[sel], minlength=qn + 1)
                        qiou = qinter / np.maximum(qsize + int(sel.sum()) - qinter, 1)
                        qiou[0] = 0
                        qc = int(qiou.argmax())
                        if qc:
                            # A query CC is a subset of one max CC because this query is kept.
                            after_ids = np.unique(labels[qlab == qc])
                            after_ids = after_ids[after_ids > 0]
                            assert len(after_ids) == 1
                            mc = int(after_ids[0])
                            joined = np.unique(qlab[labels == mc]); joined = joined[joined > 0]
                            before_tp = int(qinter[qc]); after_tp = int((sel & (labels == mc)).sum())
                            bridge = dict(best_kept_query_components_joined_by_max=len(joined), best_kept_gt_cc_iou=float(qiou[qc]),
                                          gt_cc_before_voxels=int(qsize[qc]), gt_cc_before_gt_tp=before_tp,
                                          gt_cc_after_voxels=int((labels == mc).sum()), gt_cc_after_gt_tp=after_tp,
                                          gt_cc_added_non_gt=int((labels == mc).sum() - qsize[qc] - (after_tp - before_tp)),
                                          gt_cc_after_gt_ids=len(np.unique(local[(labels == mc) & (local > 0)])))
                    final_hard = ((local_final >= t) if t == .3 else (local_final > t)) & valid
                    gained = int((sel & ~hard_max & final_hard).sum())
                    lost = int((sel & hard_max & ~final_hard).sum())
                    row = dict(case=case_id, source=source, window=wi, instance=k, threshold=t,
                               visible_voxels=g['visible_voxels'], visible_fraction=g['visible_fraction'],
                               best_all_iou=a['best_all_iou'], best_kept_iou=a['best_kept_iou'], max_iou=a['max_iou'],
                               best_all_query=q, best_kept_query=qk, best_all_presence=a['best_all_presence'],
                               best_kept_presence=float(presence[qk]) if qk >= 0 else None,
                               best_all_mask_voxels=a['best_all_mask_voxels'], best_all_healthy_fp=a['best_all_healthy_fp'],
                               best_all_other_lesion_voxels=a['best_all_other_lesion_voxels'],
                               best_all_gt_tp=inter, max_gt_tp=a['max_gt_tp'], max_gt_fn=a['max_gt_fn'],
                               max_gt_mean_prob=a['max_gt_mean_prob'], gaussian_gt_mean=a['gaussian_gt_mean'],
                               final_gt_tp_inside_window=int((sel & final_hard).sum()),
                               final_gt_mean_prob_inside_window=float(local_final[sel].mean()),
                               average_lost_local_positive_gt_voxels=lost, average_gained_local_negative_gt_voxels=gained,
                               **cc[k], **bridge)
                    case_rows.append(row)
            print(f'case_{case_id:02d} window_{wi:03d} complete', flush=True)
        assert np.all(wsum > 0)
        max_abs = float(np.max(np.abs(acc / wsum.clip(min=1e-12) - final_prob)))
        assert np.allclose(acc / wsum.clip(min=1e-12), final_prob, atol=1e-6, rtol=1e-6)
        # Final component intersections validate cache semantics, including volume floor omissions.
        for g in u['gt']:
            k = g['instance']; sel = truth == k
            assert int(sel.sum()) == g['n_voxels']
            assert int((final_prob[sel] >= .3).sum()) == g['threshold_tp']
            for c in g['components']:
                assert int((comps[sel] == c['component']).sum()) == c['intersection']
            gr = dict(case=case_id, source=source, instance=k, n_voxels=g['n_voxels'],
                      final_gt_mean_prob=g['mean_prob'], final_tp_at03=g['threshold_tp'], final_fn_at03=g['threshold_fn'],
                      final_gt_recall_at03=g['threshold_tp'] / g['n_voxels'],
                      best_final_component_iou_at03=max((c['mask_iou'] for c in g['components']), default=0),
                      final_component_ids_at03=[c['component'] for c in g['components']],
                      passes_component_score_at03=any(c['passes_score'] for c in g['components']))
            for t in (.3, .5):
                candidates = [r for r in case_rows if r['instance'] == k and r['threshold'] == t]
                best = max(candidates, key=lambda r: (r['best_all_iou'], r['visible_fraction'], -r['window']))
                full = [r for r in candidates if r['visible_fraction'] >= 1]
                full_best = max(full,key=lambda r:(r['best_all_iou'],-r['window'])) if full else None
                gt_summaries.append({**gr, 'threshold': t, 'best_window': best['window'],
                                     'best_visible_fraction': best['visible_fraction'],
                                     'best_all_iou': best['best_all_iou'], 'best_kept_iou_same_window': best['best_kept_iou'],
                                     'max_union_iou_same_window': best['max_iou'], 'max_cc_iou_same_window': best['max_cc_iou'],
                                     'presence_same_window': best['best_all_presence'],
                                     'any_all_query_ge_05': any(r['best_all_iou'] >= .5 for r in candidates),
                                     'any_kept_query_ge_05': any(r['best_kept_iou'] >= .5 for r in candidates),
                                     'any_max_cc_ge_05': any(r['max_cc_iou'] >= .5 for r in candidates),
                                     'any_fully_visible_window': any(r['visible_fraction'] >= 1 for r in candidates),
                                     'max_visible_fraction': max(r['visible_fraction'] for r in candidates),
                                     'full_best_window': full_best['window'] if full_best else None,
                                     'full_best_all_iou': full_best['best_all_iou'] if full_best else None,
                                     'full_best_kept_iou_same_window': full_best['best_kept_iou'] if full_best else None,
                                     'full_best_max_union_iou_same_window': full_best['max_iou'] if full_best else None,
                                     'full_best_max_cc_iou_same_window': full_best['max_cc_iou'] if full_best else None,
                                     'full_best_presence_same_window': full_best['best_all_presence'] if full_best else None,
                                     'full_best_max_gt_recall_same_window': full_best['max_gt_tp']/full_best['visible_voxels'] if full_best else None,
                                     'full_best_final_gt_recall_same_window': full_best['final_gt_tp_inside_window']/full_best['visible_voxels'] if full_best else None})
                hard = (final_prob >= t) if t == .3 else (final_prob > t)
                pos, final = any_support[t][sel], hard[sel]
                stage_rows.append(dict(case=case_id, source=source, instance=k, threshold=t, n_voxels=int(sel.sum()),
                                       any_window_positive_gt=int(pos.sum()), final_positive_gt=int(final.sum()),
                                       averaging_suppressed_any_window_positive_gt=int((pos & ~final).sum()),
                                       final_positive_without_any_window_positive_gt=int((~pos & final).sum()),
                                       support_windows_mean=float(n_support[t][sel].mean()),
                                       final_gt_mean_prob=float(final_prob[sel].mean())))
        giant = max(data['detections'], key=lambda d: d['n_voxels'])
        giant_sel = comps == giant['instance']
        giant_truth = np.unique(truth[giant_sel]); giant_truth = giant_truth[giant_truth > 0]
        case_summaries.append(dict(case=case_id, source=source, n_gt=case['n_gt'], n_windows=case['n_windows'],
                                   visible_gt_window_rows=len(case_rows)//2, n_detections=case['n_detections'],
                                   probability_reconstruction_max_abs_error=max_abs,
                                   giant_component=giant['instance'], giant_component_voxels=giant['n_voxels'],
                                   giant_component_score=giant['score'], giant_component_gt_count=len(giant_truth),
                                   giant_component_gt_ids=giant_truth.tolist(),
                                   giant_component_non_gt_voxels=int((giant_sel & (truth == 0)).sum()),
                                   total_gt_voxels=int((truth > 0).sum()), final_threshold_gt_tp=sum(g['threshold_tp'] for g in u['gt']),
                                   total_pred_component_voxels=int((comps > 0).sum()),
                                   no_present_windows=u['no_present_windows'],
                                   rows={str(t): summarize_rows([r for r in case_rows if r['threshold'] == t]) for t in (.3,.5)}))
        all_rows.extend(case_rows)
        del truth, final_prob, comps, acc, wsum, any_support, n_support
    for name, rows in [('window_gt_rows',all_rows),('gt_summary',gt_summaries),('averaging_gt_rows',stage_rows)]:
        dump(args.output / f'{name}.json', rows)
        if rows:
            with (args.output / f'{name}.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    groups = {}
    for source in ['all','bmsr','isles','pdgm']:
        groups[source] = {}
        for t in (.3,.5):
            rows=[r for r in all_rows if r['threshold']==t and (source=='all' or r['source']==source)]
            gs=[g for g in gt_summaries if g['threshold']==t and (source=='all' or g['source']==source)]
            sr=[g for g in stage_rows if g['threshold']==t and (source=='all' or g['source']==source)]
            groups[source][str(t)]=dict(rows=summarize_rows(rows), gt_count=len(gs),
                gt_best_all_iou=distribution([g['best_all_iou'] for g in gs]),
                gt_any_all_query_ge_05=sum(g['any_all_query_ge_05'] for g in gs),
                gt_any_kept_query_ge_05=sum(g['any_kept_query_ge_05'] for g in gs),
                gt_any_max_cc_ge_05=sum(g['any_max_cc_ge_05'] for g in gs),
                gt_any_fully_visible_window=sum(g['any_fully_visible_window'] for g in gs),
                gt_all_window_positive_voxels=sum(g['any_window_positive_gt'] for g in sr),
                gt_averaging_suppressed_voxels=sum(g['averaging_suppressed_any_window_positive_gt'] for g in sr),
                gt_final_positive_voxels=sum(g['final_positive_gt'] for g in sr),
                validation_final_positive_without_any_window_positive_gt=sum(g['final_positive_without_any_window_positive_gt'] for g in sr))
    summary=dict(evidence='NOT_EVIDENCE; six-case observational pseudo-label diagnostic only',
                 input_directory_name=args.input.name, code_sha=meta['contract']['code_sha'],
                 n_cases=6,n_gt=sum(c['n_gt'] for c in case_summaries),n_windows=sum(c['n_windows'] for c in case_summaries),
                 unique_gt_window_rows=len(all_rows)//2,case_summaries=case_summaries,groups=groups,
                 validation=dict(all_six_reproduced=True,probability_reconstruction_allclose=True,
                                 hard_mask_counts_and_best_query_intersections=True,whole_component_gt_intersections=True),
                 semantics=dict(local_queries='natural-window local GT mask IoU; 64 all queries; >=0.3 or >0.5',
                                presence='strict >0.5',maximum='voxelwise max over retained queries in the SAME window',
                                max_cc='best 26-connected component of retained max in SAME window; no size floor applied for local diagnostic',
                                averaging='Gaussian weighted mean of natural-window maxima, reconstructed exactly; any-window upper bound is optimistic',
                                matching='No full-volume bbox matches or gate claims are computed by this script'))
    dump(args.output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['n_cases','n_gt','n_windows','unique_gt_window_rows','validation']},indent=2))


if __name__=='__main__':
    main()
