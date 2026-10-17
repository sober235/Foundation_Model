#!/usr/bin/env python3
"""Exact per-query CC oracles on fully visible natural-window GT; no inference.

Prune using total query GT intersection / GT count as an upper bound on any
one CC's IoU. Both all-query and kept-query maxima are tracked independently.
"""
import argparse,csv,json,os
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'
try: os.nice(19)
except OSError: pass
import numpy as np
from scipy import ndimage


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path('/home/congcongliu/outputs/aur_model_diagnostic/20261017_replay_b240962'))
    ap.add_argument('--output',type=Path,default=Path('/tmp/aur-six-u-analysis-20261017'))
    args=ap.parse_args(); out=args.output
    meta=json.loads((args.input/'summary.json').read_text())
    window_rows=json.loads((out/'window_gt_rows.json').read_text())
    old={(r['case'],r['window'],r['instance'],r['threshold']):r for r in window_rows}
    result=[]; labels_processed=0;label_queries_pruned=0
    for case in meta['cases']:
        cid=case['anonymous_case']; cd=args.input/f'case_{cid:02d}'
        d=json.loads((cd/'diagnostic.json').read_text())
        for wi,w in enumerate(d['u']['windows']):
            full=[g for g in w['gt'] if g['visible_fraction']>=1]
            if not full:continue
            z=np.load(cd/'windows'/f'window_{wi:03d}.npz')
            shape=z['instance'].shape; local=z['instance']; presence=z['presence']
            for t,key in ((.3,'all_query_hard_03_packed'),(.5,'all_query_hard_05_packed')):
                packed=z[key]
                counts=np.asarray([g['visible_voxels'] for g in full])
                total_tp=np.asarray([g[f'at_{t}']['all_query_gt_intersection'] for g in full],float)
                upper=total_tp/counts[:,None]
                states=[dict(all_iou=0.,all_query=-1,all_component=-1,all_voxels=0,all_tp=0,
                             kept_iou=0.,kept_query=-1,kept_component=-1,kept_voxels=0,kept_tp=0) for g in full]
                # Process likely useful queries first. The upper bound is exact for pruning;
                # no query with potential to improve either oracle is omitted.
                for q in np.argsort(-upper.max(axis=0),kind='stable'):
                    active=[j for j,s in enumerate(states) if upper[j,q]>s['all_iou']+1e-12 or
                            (presence[q]>.5 and upper[j,q]>s['kept_iou']+1e-12)]
                    if not active:
                        label_queries_pruned+=1;continue
                    mask=np.unpackbits(packed[q],count=local.size).reshape(shape).astype(bool)
                    lab,n=ndimage.label(mask,structure=np.ones((3,3,3),bool))
                    sizes=np.bincount(lab.ravel(),minlength=n+1)
                    labels_processed+=1
                    for j in active:
                        g,s=full[j],states[j]
                        inter=np.bincount(lab[local==g['instance']],minlength=n+1)
                        assert int(inter[1:].sum())==int(total_tp[j,q])
                        ious=inter/np.maximum(sizes+counts[j]-inter,1);ious[0]=0
                        c=int(ious.argmax());v=float(ious[c])
                        if v>s['all_iou']+1e-12:
                            s.update(all_iou=v,all_query=int(q),all_component=c,all_voxels=int(sizes[c]),all_tp=int(inter[c]))
                        if presence[q]>.5 and v>s['kept_iou']+1e-12:
                            s.update(kept_iou=v,kept_query=int(q),kept_component=c,kept_voxels=int(sizes[c]),kept_tp=int(inter[c]))
                for g,s in zip(full,states):
                    prev=old[cid,wi,g['instance'],t]
                    aq,kq=s['all_query'],s['kept_query']
                    result.append(dict(case=cid,source=case['source'],window=wi,instance=g['instance'],threshold=t,
                                       visible_fraction=1.,visible_voxels=g['visible_voxels'],
                                       best_all_query_cc_iou=s['all_iou'],best_all_query_cc_query=aq,
                                       best_all_query_cc_presence=float(presence[aq]) if aq>=0 else None,
                                       best_all_query_cc_voxels=s['all_voxels'],best_all_query_cc_gt_tp=s['all_tp'],
                                       best_all_query_cc_parent_whole_query_iou=g[f'at_{t}']['all_query_iou'][aq] if aq>=0 else 0.,
                                       best_all_query_cc_parent_whole_query_voxels=w[f'at_{t}']['query_voxels'][aq] if aq>=0 else 0,
                                       best_kept_query_cc_iou=s['kept_iou'],best_kept_query_cc_query=kq,
                                       best_kept_query_cc_voxels=s['kept_voxels'],best_kept_query_cc_gt_tp=s['kept_tp'],
                                       max_map_cc_iou_same_window=prev['max_cc_iou'],max_union_iou_same_window=prev['max_iou'],
                                       best_whole_query_iou_same_window=prev['best_all_iou'],
                                       best_kept_whole_query_iou_same_window=prev['best_kept_iou'],
                                       final_gt_mean_prob=prev['final_gt_mean_prob_inside_window'],
                                       final_gt_recall_same_threshold=prev['final_gt_tp_inside_window']/g['visible_voxels']))
            print(f'case_{cid:02d} window_{wi:03d} complete; {labels_processed} CC labels',flush=True)
    gt=[]
    for case in meta['cases']:
        cid=case['anonymous_case']
        ks=sorted({r['instance'] for r in result if r['case']==cid})
        for k in ks:
            for t in (.3,.5):
                rs=[r for r in result if r['case']==cid and r['instance']==k and r['threshold']==t]
                b=max(rs,key=lambda r:r['best_all_query_cc_iou']);bk=max(rs,key=lambda r:r['best_kept_query_cc_iou'])
                gt.append(dict(case=cid,source=case['source'],instance=k,threshold=t,n_full_windows=len(rs),
                               best_all_cc_iou=b['best_all_query_cc_iou'],best_all_cc_window=b['window'],
                               best_all_cc_query=b['best_all_query_cc_query'],best_all_cc_presence=b['best_all_query_cc_presence'],
                               kept_cc_iou_same_window=b['best_kept_query_cc_iou'],max_map_cc_iou_same_window=b['max_map_cc_iou_same_window'],
                               best_kept_cc_iou=bk['best_kept_query_cc_iou'],best_kept_cc_window=bk['window'],
                               max_map_cc_iou_at_best_kept_window=bk['max_map_cc_iou_same_window'],
                               any_all_cc_ge05=any(r['best_all_query_cc_iou']>=.5 for r in rs),
                               any_kept_cc_ge05=any(r['best_kept_query_cc_iou']>=.5 for r in rs),
                               any_max_map_cc_ge05=any(r['max_map_cc_iou_same_window']>=.5 for r in rs),
                               any_same_window_kept_cc_ge05_but_max_cc_lt05=any(r['best_kept_query_cc_iou']>=.5>r['max_map_cc_iou_same_window'] for r in rs),
                               all_query_cc_good_but_no_kept_query_cc_good=any(r['best_all_query_cc_iou']>=.5 for r in rs) and not any(r['best_kept_query_cc_iou']>=.5 for r in rs)))
    for name,rows in [('full_window_query_cc_rows',result),('full_window_query_cc_gt',gt)]:
        (out/f'{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
        with (out/f'{name}.csv').open('w') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    summary=dict(exact_oracles=True,pruning_bound='CC IoU <= total query GT intersection / GT count',
                 labels_processed=labels_processed,query_window_labels_pruned=label_queries_pruned,
                 n_fully_visible_gt_window_rows=len(result)//2,groups={})
    for src in ['all','bmsr','isles','pdgm']:
        summary['groups'][src]={}
        for t in [.3,.5]:
            gs=[r for r in gt if r['threshold']==t and (src=='all' or r['source']==src)]
            rs=[r for r in result if r['threshold']==t and (src=='all' or r['source']==src)]
            keys=['any_all_cc_ge05','any_kept_cc_ge05','any_max_map_cc_ge05','any_same_window_kept_cc_ge05_but_max_cc_lt05','all_query_cc_good_but_no_kept_query_cc_good']
            summary['groups'][src][str(t)]=dict(n_gt=len(gs),n_rows=len(rs),**{key:sum(g[key] for g in gs) for key in keys},
                  same_window_rows_kept_cc_ge05_max_cc_lt05=sum(r['best_kept_query_cc_iou']>=.5>r['max_map_cc_iou_same_window'] for r in rs),
                  same_window_rows_all_cc_ge05_kept_cc_lt05=sum(r['best_all_query_cc_iou']>=.5>r['best_kept_query_cc_iou'] for r in rs),
                  best_all_cc_iou_quantiles=np.quantile([g['best_all_cc_iou'] for g in gs],[0,.25,.5,.75,1]).tolist(),
                  best_kept_cc_iou_quantiles=np.quantile([g['best_kept_cc_iou'] for g in gs],[0,.25,.5,.75,1]).tolist())
    (out/'query_cc_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
