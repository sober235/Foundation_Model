#!/usr/bin/env python3
"""Anonymous read-only CPU analysis of completed six-case replay; never imports a model."""
import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


BASE = Path('/home/congcongliu/outputs/aur_model_diagnostic/20261017_replay_b240962')
REPO = Path('/home/congcongliu/src/Foundation_Model-aur-diagnostic')


def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write_json(p, x): p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')


def write_csv(p, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(clean(rows))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def literal(path,key):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('mapping missing')


def stats(xs):
    a=np.asarray([x for x in xs if x is not None and np.isfinite(x)],float)
    return {'n':int(a.size),'mean':float(a.mean()) if a.size else None,'median':float(np.median(a)) if a.size else None,
            'min':float(a.min()) if a.size else None,'max':float(a.max()) if a.size else None}


def nearest(a,shape):
    # Exact floor mapping used by torch.interpolate(mode='nearest') for these divisible dimensions.
    ix=[np.floor(np.arange(n)*a.shape[i]/n).astype(int) for i,n in enumerate(shape)]
    return a[np.ix_(*ix)]


def centre_sample(a,shape,upper=False):
    ix=[]
    for i,n in enumerate(shape):
        x=(np.arange(n)+.5)*a.shape[i]/n-.5
        ix.append((np.ceil(x) if upper else np.floor(x)).astype(int))
    return a[np.ix_(*ix)]


def dice_binary(a,b):
    den=int(a.sum())+int(b.sum())
    return 2*int(np.count_nonzero(a&b))/den if den else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=BASE)
    ap.add_argument('--repo',type=Path,default=REPO);ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.input/'summary.json').read_text())
    assert summary['status']=='complete' and len(summary['cases'])==6 and all(c['reproduced'] for c in summary['cases'])
    student=literal(args.repo/'anatobind/anatomy/labels.py','STUDENT')
    entities=literal(args.repo/'anatobind/eval/lookup.py','BRAIN_ALL')
    names=literal(args.repo/'docs/verification/2026-09-16-brain-probe/probe_common.py','LABEL_NAMES')
    hosts=[name for cid,name,ls,rep in student if 1<=cid<=13]+['no_host']
    l2h={l:cid for cid,name,ls,rep in student if 1<=cid<=13 for l in ls}
    emap2h=np.asarray([0]+[l2h.get(l,0) for l in entities],np.uint8)
    host_members=[[i for i,l in enumerate(entities) if l2h.get(l)==h+1] for h in range(13)]
    def hostdice(pred,truth,valid):
        p,t=emap2h[pred][valid],emap2h[truth][valid]
        ds=[dice_binary(p==i,t==i) for i in range(1,14)]
        return stats(ds)['mean']
    def hard(a):
        return np.where(a.max(0)>=.5,a.argmax(0)+1,0).astype(np.uint8)

    cases=[];entity_rows=[];binds=[];hostrows=[];hashes={};max_overlap_error=0.;max_dice_error=0.
    for item in summary['cases']:
        anon=int(item['anonymous_case']);cid=f'case_{anon:02d}';cd=args.input/cid
        p=cd/'diagnostic.json';hashes[f'{cid}/diagnostic.json']=sha(p)
        d=json.loads(p.read_text());ident={'anonymous_case':anon,'source':item['source'],'sequence':item['sequence']}
        cases.append({**ident,'n_bindings':len(d['r']),
                      **{f'{region}_{metric}':d['a'][region][metric] for region in ['all','lesion','outside_lesion'] for metric in ['host_macro','entity_macro']}})
        for l,v in d['a']['entity_counts'].items():
            label=int(l)
            entity_rows.append({**ident,'label':label,'entity_name_repo':names[label],**v,'dice':d['a']['all']['entities'][l],
                                'state':'absent_prediction' if v['pred']==0 and v['truth']>0 else 'both_empty' if v['pred']==v['truth']==0 else 'zero_overlap' if d['a']['all']['entities'][l]==0 else 'positive_overlap'})
        for row in d['r']:
            ip=cd/f"bind_{row['instance']:03d}.npz";hashes[f'{cid}/{ip.name}']=sha(ip)
            with np.load(ip,allow_pickle=False) as z:
                raw=z['coarse_a_ungated'];gated=z['coarse_a_gated'];u=z['coarse_u'][0];v=z['valid'];valid=v.astype(bool)
                truth_full=z['true_a'];pred_full=z['full_a'];target_full=z['target']
                truth=nearest(truth_full,u.shape);full_nn=nearest(pred_full,u.shape);target=nearest(target_full,u.shape)&valid
                geom=z['geometry'][0];gt_h=emap2h[truth]
                hosts_raw=np.stack([raw[ix].max(0) for ix in host_members]);hosts_gated=np.stack([gated[ix].max(0) for ix in host_members])
                ev=u*v;hp=hosts_gated*v
                overlap=np.sum(hp*ev[None],axis=(1,2,3),dtype=np.float64)/max(float(ev.sum(dtype=np.float64)),1e-6)
                overlap_error=float(np.max(np.abs(overlap-geom[:,0])));max_overlap_error=max(max_overlap_error,overlap_error)
                # geometry's matmul ran under bf16 autocast; this CPU float64 sum is a numeric cross-check,
                # not a bitwise replay of that matmul. Use saved geometry as the model input throughout.
                assert np.max(np.abs(geom[:,0]-np.asarray(row['coarse_overlap'])))==0,'Stored geometry/JSON mismatch'
                metrics={name:hostdice(hard(a),truth,valid) for name,a in [('ungated',raw),('gated',gated)]}
                for key,val in metrics.items():
                    err=abs(val-row['coarse_a'][f'coarse_{key}']['host_macro']);max_dice_error=max(max_dice_error,err)
                    assert err<1e-12,'Stored coarse A Dice could not be reproduced'
                uvals=u[valid];uthr=(u>.5)&valid
                uid=row['truth'];tmask=(gt_h==uid+1)&valid if uid<13 else np.zeros_like(valid)
                htrue=hosts_gated[uid] if uid<13 else np.zeros_like(u)
                htrue_hard=(htrue>.5)&valid
                nhard=((hosts_gated>.5)&valid).sum(0)[valid]
                outcome=('both_right' if row['b0']==row['truth'] else 'rescue') if row['r']==row['truth'] else ('harm' if row['b0']==row['truth'] else 'both_wrong')
                record={**ident,'instance':row['instance'],'selected_query':row['query'],
                    'truth':hosts[row['truth']],'local_truth':hosts[row['local_truth']],'r':hosts[row['r']],'b0':hosts[row['b0']],
                    'coarse_overlap_choice':hosts[row['coarse_overlap_choice']],'outcome':outcome,
                    'query_iou_dense':row['query_iou'],'in_crop_share':row['in_crop_share'],
                    'saved_whole_volume_local_a':row['saved_whole_volume_local_a'],
                    'single_crop_dense_a':row['single_crop_dense_a']['host_macro'],
                    'same_grid_nearest_dense_a':hostdice(full_nn,truth,valid),
                    'same_grid_centre_floor_dense_a':hostdice(centre_sample(pred_full,u.shape),centre_sample(truth_full,u.shape),valid),
                    'same_grid_centre_ceil_dense_a':hostdice(centre_sample(pred_full,u.shape,True),centre_sample(truth_full,u.shape,True),valid),
                    'coarse_a_ungated':metrics['ungated'],'coarse_a_gated':metrics['gated'],
                    'coarse_a_gated_centre_floor_gt':hostdice(hard(gated),centre_sample(truth_full,u.shape),valid),
                    'coarse_a_gated_centre_ceil_gt':hostdice(hard(gated),centre_sample(truth_full,u.shape,True),valid),
                    'gating_changes_hard_a_voxel_fraction':float((hard(raw)[valid]!=hard(gated)[valid]).mean()),
                    'valid_coarse_voxels':int(valid.sum()),'gt_target_full_voxels':int(z['target'].sum()),
                    'gt_target_nearest_voxels':int(target.sum()),
                    'gt_target_centre_floor_voxels':int((centre_sample(target_full,u.shape)&valid).sum()),
                    'gt_target_centre_ceil_voxels':int((centre_sample(target_full,u.shape,True)&valid).sum()),
                    'coarse_u_mean_valid':float(uvals.mean(dtype=np.float64)), 'coarse_u_std_valid':float(uvals.std(dtype=np.float64)),
                    'coarse_u_min_valid':float(uvals.min()),'coarse_u_max_valid':float(uvals.max()),
                    'coarse_u_fraction_gt_0_5':float((uvals>.5).mean()),
                    'coarse_u_target_iou_nn':dice_binary(uthr,target)/(2-dice_binary(uthr,target)) if dice_binary(uthr,target) is not None else None,
                    'coarse_u_prob_inside_target_nn':float(u[target].mean(dtype=np.float64)) if target.any() else None,
                    'coarse_u_prob_outside_target_nn':float(u[valid&~target].mean(dtype=np.float64)) if (valid&~target).any() else None,
                    'coarse_u_soft_mass_inside_target_share_nn':float((u[target]*v[target]).sum(dtype=np.float64)/max(ev.sum(dtype=np.float64),1e-6)),
                    'coarse_host_hard_channels_per_valid_voxel':float(nhard.mean()),
                    'coarse_host_multichannel_fraction':float((nhard>=2).mean()),
                    'n_host_geometry_present':int((geom[:,5]>.5).sum()),
                    'n_geometry_overlap_gt_0_05':int((geom[:,0]>.05).sum()),
                    'n_geometry_zero_distance':int((geom[:,4]==0).sum()),
                    'geometry_overlap_max':float(geom[:,0].max()),'geometry_overlap_true_host':float(geom[uid,0]) if uid<13 else None,
                    'coarse_true_host_hard_dice_nn':dice_binary(htrue_hard,tmask),
                    'coarse_true_host_hard_precision_nn':int(np.count_nonzero(htrue_hard&tmask))/int(htrue_hard.sum()) if htrue_hard.any() else None,
                    'coarse_true_host_hard_recall_nn':int(np.count_nonzero(htrue_hard&tmask))/int(tmask.sum()) if tmask.any() else None,
                    'coarse_true_host_prob_in_host_nn':float(htrue[tmask].mean(dtype=np.float64)) if tmask.any() else None,
                    'coarse_true_host_prob_out_host_nn':float(htrue[valid&~tmask].mean(dtype=np.float64)) if (valid&~tmask).any() else None,
                    'coarse_overlap_vs_geometry_max_abs_error':overlap_error,
                    'r_max_probability':max(row['r_probs'])}
                binds.append(record)
                for mode,hm in [('ungated',hosts_raw),('gated',hosts_gated)]:
                    for hi,name in enumerate(hosts[:13]):
                        vals=hm[hi][valid];gt=(gt_h==hi+1)&valid;ph=(hm[hi]>.5)&valid
                        inter=int(np.count_nonzero(gt&ph))
                        hostrows.append({**ident,'instance':row['instance'],'mode':mode,'host':name,'truth_host':hosts[uid],
                            'prob_mean':float(vals.mean(dtype=np.float64)),'prob_std':float(vals.std(dtype=np.float64)),
                            'hard_valid_fraction':float((vals>.5).mean()),'gt_valid_fraction':float(gt[valid].mean()),
                            'mean_inside_gt':float(hm[hi][gt].mean(dtype=np.float64)) if gt.any() else None,
                            'mean_outside_gt':float(hm[hi][valid&~gt].mean(dtype=np.float64)) if (valid&~gt).any() else None,
                            'hard_dice':dice_binary(gt,ph),'hard_precision':inter/int(ph.sum()) if ph.any() else None,
                            'hard_recall':inter/int(gt.sum()) if gt.any() else None})
    dims={
        'ALL':lambda r:'ALL','anonymous_case':lambda r:f"case_{r['anonymous_case']:02d}",
        'source':lambda r:r['source'],'outcome':lambda r:r['outcome'],
        'dense_query_iou_ge_0_5':lambda r:str(r['query_iou_dense']>=.5),
        'truth_host':lambda r:r['truth']}
    fieldnames=['saved_whole_volume_local_a','single_crop_dense_a','same_grid_nearest_dense_a','same_grid_centre_floor_dense_a','same_grid_centre_ceil_dense_a',
        'coarse_a_ungated','coarse_a_gated','coarse_a_gated_centre_floor_gt','coarse_a_gated_centre_ceil_gt',
        'gating_changes_hard_a_voxel_fraction','coarse_u_mean_valid','coarse_u_std_valid','coarse_u_fraction_gt_0_5','coarse_u_target_iou_nn',
        'coarse_u_soft_mass_inside_target_share_nn','coarse_host_hard_channels_per_valid_voxel','coarse_host_multichannel_fraction',
        'n_host_geometry_present','n_geometry_overlap_gt_0_05','n_geometry_zero_distance','geometry_overlap_max','geometry_overlap_true_host',
        'coarse_true_host_hard_dice_nn','coarse_true_host_hard_precision_nn','coarse_true_host_hard_recall_nn','coarse_true_host_prob_in_host_nn','coarse_true_host_prob_out_host_nn']
    strata=[]
    for dim,fn in dims.items():
        for key in sorted({fn(r) for r in binds}):
            rs=[r for r in binds if fn(r)==key];n=len(rs)
            z={'dimension':dim,'group':key,'n':n,**{out:sum(r['outcome']==out for r in rs) for out in ['both_right','both_wrong','rescue','harm']},
               'r_correct':sum(r['r']==r['truth'] for r in rs),'b0_correct':sum(r['b0']==r['truth'] for r in rs),
               'local_truth_matches':sum(r['local_truth']==r['truth'] for r in rs),
               'overlap_choice_correct':sum(r['coarse_overlap_choice']==r['truth'] for r in rs),
               'r_matches_overlap_choice':sum(r['r']==r['coarse_overlap_choice'] for r in rs),
               'gt_nearest_target_empty':sum(r['gt_target_nearest_voxels']==0 for r in rs),
               'coarse_u_hard_empty':sum(r['coarse_u_fraction_gt_0_5']==0 for r in rs),
               'coarse_u_hard_full':sum(r['coarse_u_fraction_gt_0_5']==1 for r in rs),
               'fields':{f:stats(r[f] for r in rs) for f in fieldnames}}
            strata.append(z)
    zero=[r for r in entity_rows if r['label'] in [5,44,18,26,58]]
    verification={'n_cases':len(cases),'n_controlled_bindings':len(binds),'stored_coarse_dice_max_abs_error':max_dice_error,
                  'cpu_unquantized_vs_saved_autocast_geometry_overlap_max_abs_difference':max_overlap_error,
                  'stored_geometry_equals_diagnostic_json_overlap':True,
                  'all55_local_truth_matches_whole':all(r['truth']==r['local_truth'] for r in binds),
                  'zero_entities_all30_reference_present_prediction_absent':all(r['truth']>0 and r['pred']==0 for r in zero),
                  'nearest_grid_target_empty':sum(r['gt_target_nearest_voxels']==0 for r in binds),
                  'in_crop_shares_all_one':all(r['in_crop_share']==1 for r in binds)}
    provenance={'evidence':'NOT_EVIDENCE','claim_type':'computed_from_completed_replay','input_hashes':hashes,'script_sha256':sha(Path(__file__)),
      'implementation_hashes':{f:sha(args.repo/f) for f in ['scripts/aur_diagnose_model.py','anatobind/aur/model.py','anatobind/aur/relation.py','anatobind/aur/heads.py']},
      'no_inference_training_or_repo_writes':True,
      'limitations':['Six deterministic validation cases only; 33/55 bindings come from one case. No causal or population claim.',
        'Nearest downsampled hard pseudoGT and hard dense maps are sampling controls, not calibrated dense/coarse equivalence; tiny targets may vanish.',
        'Coarse hard A Dice uses argmax plus >=0.5; actual R consumes independent soft host masks, so both are reported.',
        'Single-crop dense A and whole-volume A are ungated by presence; R coarse A is multiplied by entity presence.',
        'Whole-volume inside/outside Dice averages different present-class sets and is not an additive decomposition.',
        'No relation logits under any substituted geometry have been computed.']}
    write_csv(args.output/'a_cases.csv',cases);write_csv(args.output/'a_entities.csv',entity_rows)
    write_csv(args.output/'r_bindings.csv',binds);write_csv(args.output/'r_host_soft_masks.csv',hostrows)
    flat=[]
    for s in strata:
        for f,st in s['fields'].items():flat.append({**{k:v for k,v in s.items() if k!='fields'},'field':f,**{f'field_{k}':v for k,v in st.items()}})
    write_csv(args.output/'r_strata.csv',flat)
    write_json(args.output/'analysis.json',{'provenance':provenance,'verification':verification,'cases':cases,'entities':entity_rows,
                                         'bindings':binds,'strata':strata})
    print(json.dumps(clean({'verification':verification,'overall':strata[0],
            'by_case':[x for x in strata if x['dimension']=='anonymous_case'],
            'five_zero_entities':zero}),indent=2))


if __name__=='__main__':main()
