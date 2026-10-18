"""Bounded frozen TRAIN ring-to-global head-tangent experiment; no training."""
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import aur_diagnose_u_supervision as P
from scripts import aur_u_background_archive as B
from scripts import aur_u_background_measurements as M
from scripts import aur_u_background_tangent as T
from anatobind.aur.confidence_diagnostics import array_digest
from anatobind.aur.query_decoder import QueryCCDecoder

RECIPE=dict(P.A.RECIPE)
OUTPUT_ROOT=Path('/home/congcongliu/outputs/aur_u_background_allocation_20261018')
REFERENCE=Path('/home/congcongliu/outputs/aur_u_train_supervision_20261018/run_1a7de61')
NEW_SOURCES=('scripts/aur_diagnose_u_background.py','scripts/aur_u_background_archive.py',
             'scripts/aur_u_background_measurements.py','scripts/aur_u_background_tangent.py',
             'docs/superpowers/specs/2026-10-18-aur-u-background-allocation-design.md')


def inventory(path):
    members=list(Path(path).iterdir())
    if any(p.is_symlink() or not p.is_file() for p in members):
        raise ValueError('Reference contains nonregular members')
    return {p.name:P.file_sha256(p) for p in sorted(members)}


def prepare_contract(args, *, execution_sha=None):
    """Private metadata/hash checks only; never emit selections or path keys."""
    stored=P.A.read_json(Path(args.reference)/'contract.json')
    train,selection,prior=P.prepare_contract(args,execution_sha=stored['code_sha'])
    if stored!=prior: raise ValueError('Reference external contract differs')
    before=inventory(args.reference)
    marker=P.A.read_json(Path(args.reference)/'COMPLETE.json')
    if marker!={'schema':1,'files':{k:v for k,v in before.items() if k!='COMPLETE.json'}}:
        raise ValueError('Reference completion hashes differ')
    if set(before)!=P.A.expected_names()|{'COMPLETE.json'}:
        raise ValueError('Reference membership differs')
    sources=dict(prior['source_sha256'])
    sources.update({p:P.file_sha256(ROOT/p) for p in NEW_SOURCES})
    contract=dict(recipe=RECIPE,selection=selection,inputs=prior['inputs'],
        model_state_sha256=P.A.CHECKPOINT_STATE_SHA,checkpoint_sha256=prior['checkpoint_sha256'],
        reference_files=before,reference_path=str(Path(args.reference).resolve()),
        input_sha256=prior['input_sha256'],source_sha256=sources,
        code_sha=execution_sha or P.code_sha(),environment=prior['environment'],
        protocol=dict(boundary_shares=[.2,0.],focus_share=.5,head_parameters=74016,
                      primary='combined_unit_direction_unsampled_FP_background',
                      tangent='native_anchored_float32',native_rtol=1e-5,native_atol=1e-5))
    return train,selection,contract


def _numpy(value,dtype=np.float32):
    result=value.detach().cpu().float().numpy() if value.dtype==torch.bfloat16 else value.detach().cpu().numpy()
    return result.astype(dtype,copy=True)


def _native_equal(actual,expected):
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid native control values')
    if not np.allclose(actual,expected,rtol=1e-5,atol=1e-5):
        raise ValueError('Native control logits differ')
    return float(np.max(np.abs(actual-expected),initial=0.))


def _stream_responses(pixels,rows,indices,analysis,deadline):
    """Independent reduction of dense projection, not dotting feature means."""
    result=np.zeros((len(rows),2,2,2),dtype=np.float64)
    jvps=analysis['numerics']['event_jvp']
    defined=analysis['numerics']['direction_defined']
    for i,(row,loc) in enumerate(zip(rows,indices)):
        P.check_deadline(deadline)
        if not len(loc): continue
        d,q=row['draw'],row['query']
        j=torch.as_tensor(jvps[d,:,:,:,q,:],device=pixels.device,dtype=torch.float64)
        total=torch.zeros((2,2,2),device=pixels.device,dtype=torch.float64)
        for start in range(0,len(loc),65536):
            ix=torch.as_tensor(loc[start:start+65536],device=pixels.device,dtype=torch.long)
            feat=pixels[:,ix].to(torch.float64)
            total+=torch.einsum('oanc,cv->oanv',j,feat).sum(-1)
        value=(total/len(loc)).cpu().numpy()
        result[i]=np.where(defined[d],value,0.)
    return result


def collect_crop(model,crop,reference_meta,reference_arrays,*,device,deadline):
    started=time.monotonic(); P.check_deadline(deadline)
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Model must be frozen eval with no parameter gradients')
    if model.cfg['n_events']!=RECIPE['n_queries'] or tuple(crop['valid'].shape)!=tuple(RECIPE['crop']):
        raise ValueError('Frozen query count or crop dimensions differ')
    if not bool(crop['u_supervised']): raise ValueError('U supervision must be enabled')
    crop_hashes={k:array_digest(_numpy(crop[k],np.int64 if k=='instance' else np.float32))
                 for k in ('image','valid','coords','local','instance','point_weight')}
    if crop_hashes!=reference_meta['crop_tensor_sha256']:
        raise ValueError('Regenerated crop differs from frozen reference')
    before=P.A.model_state_digest(model)
    if before!=reference_meta['state_before'] or before!=reference_meta['state_after']:
        raise ValueError('Reference state anchor differs')
    row=reference_meta['row_ordinal']; crop_index=reference_meta['crop_ordinal']
    pairs=[]; point_summaries=[]
    for draw in range(2):
        pts,summary=M.paired_points(crop['valid'].cpu(),crop['instance'].cpu(),crop['point_weight'].cpu(),
            n=RECIPE['points'],seed=RECIPE['seed']+1000*row+10*crop_index+draw)
        if not np.array_equal(pts[0],reference_arrays['points'][draw]):
            raise ValueError('Original point draw differs')
        pairs.append(pts); point_summaries.append(summary)
    points=np.stack(pairs)
    arrays={}
    for layer in (0,2):
        for kind in ('weight','bias'):
            arrays[f'head_{layer}_{kind}']=_numpy(getattr(model.masks.embed_full[layer],kind))
    P.check_deadline(deadline)
    with torch.no_grad():
        with P.I._autocast(device):
            out=model(*[crop[k][None].to(device) for k in ('image','valid','coords','local')])
            presence=out['event_presence'].float()[0]
            raw=model.event_masks(out).float()[0]
            if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(presence).all()):
                raise ValueError('Nonfinite native logits')
            event=raw.sigmoid(); del raw
            r_before=model.bind(out,0,torch.arange(model.cfg['n_events'],device=device)).float()
            pixels=out['pix']['full'][0].flatten(1).detach()
            u_logits=[]; u_features=[]; a_logits=[]
            for draw in range(2):
                ls=[]; fs=[]
                for arm in range(2):
                    pts=torch.as_tensor(points[draw,arm],device=device,dtype=torch.long)[None]
                    ls.append(_numpy(model.event_masks(out,pts).float()[0]))
                    fs.append(_numpy(pixels[:,pts[0]]))
                    if arm==0: a_logits.append(_numpy(model.entity_masks(out,pts).float()[0]))
                u_logits.append(ls); u_features.append(fs)
    arrays.update(event_tokens=_numpy(out['event_embed'][0]),entity_tokens=_numpy(out['entity_embed'][0]),
        event_presence=_numpy(presence),entity_presence=_numpy(out['entity_presence'][0]),
        u_logits=np.asarray(u_logits,dtype=np.float32),u_features=np.asarray(u_features,dtype=np.float32),
        u_points=points,a_logits=np.stack(a_logits),a_present=_numpy(crop['entity_present'],np.bool_),
        r_logits_before=_numpy(r_before))
    native_errors=dict(presence=_native_equal(arrays['event_presence'],reference_arrays['presence_logits']),
        mask=_native_equal(arrays['u_logits'][:,0],reference_arrays['sampled_mask_logits']))
    weight=_numpy(crop['point_weight']).ravel(); inst=_numpy(crop['instance'],np.int64).ravel()
    entity=_numpy(crop['entity'],np.int64).ravel(); ignore=_numpy(crop['a_ignore'],np.bool_).ravel()
    arrays.update(sampled_instances=inst[points],sampled_weights=weight[points],
                  a_entity_points=entity[points[:,0]],a_ignore=ignore[points[:,0]])
    decoder=QueryCCDecoder(tuple(RECIPE['crop']),reference_meta['voxel_mm3'],workers=2)
    try:
        decoder(window=tuple((0,n) for n in RECIPE['crop']),event_prob=event,event_presence=presence.sigmoid(),
                entity_prob=None,valid=crop['valid'],weight=None)
        detection_map,detection_rows,stats=decoder.finish()
    finally:
        decoder._executor.shutdown(wait=True,cancel_futures=True)
    del event
    if not np.array_equal(detection_map,reference_arrays['detection_map']):
        raise ValueError('Native decoded component map differs')
    P.A.compare_values(detection_rows,reference_meta['detection_rows'])
    P.A.compare_values(stats,reference_meta['decoder_stats'])
    meta=dict(source=reference_meta['source'],sequence=reference_meta['sequence'],row_ordinal=row,
        crop_ordinal=crop_index,n_instances=int(crop['n_instances']),u_supervised=True,
        a_supervised=bool(crop['a_supervised']),model_state_before=before,model_state_after=before,
        crop_tensor_sha256=crop_hashes,paired_sampling=point_summaries,native_control_errors=native_errors,
        controls=dict(native_logits_verified=True,decoder_verified=True,crop_verified=True,
                      model_frozen=True,fixed_matching=True,optimizer=False))
    P.check_deadline(deadline)
    analysis=T.analyze_crop(meta,arrays)
    matches=[]
    for draw in range(2):
        found=analysis['summary']['draws'][draw]['control_matches']
        if found!=reference_meta['summaries']['draws'][draw]['matches']:
            raise ValueError('Original training assignment differs')
        matches.append((np.asarray(found['query_indices'],np.int64),np.asarray(found['target_indices'],np.int64)))
    rows,means,indices=M.build_rois(reference_arrays['detection_map'],reference_meta['detection_rows'],
        reference_meta['summaries']['outcomes'],reference_arrays['gt_instance_map'],reference_arrays['valid'],
        reference_arrays['point_weight'],pixels,points,matches)
    meta['roi_rows']=rows; arrays['roi_features']=means
    arrays['roi_reference_responses']=_stream_responses(pixels,rows,indices,analysis,deadline)
    with torch.no_grad(),P.I._autocast(device):
        arrays['r_logits_after']=_numpy(model.bind(out,0,torch.arange(model.cfg['n_events'],device=device)).float())
    if not np.array_equal(arrays['r_logits_before'],arrays['r_logits_after']):
        raise ValueError('Fixed-query R logits changed')
    after=P.A.model_state_digest(model)
    if before!=after or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Actual model state or gradients changed')
    meta['model_state_after']=after
    meta['array_sha256']={k:array_digest(v) for k,v in arrays.items()}
    meta['report']=M.response_report(meta,arrays,analysis)
    meta['resources']=dict(elapsed_seconds=time.monotonic()-started,
        rss_max_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,
        cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.type=='cuda' else 0)
    P.check_deadline(deadline)
    return meta,arrays


def run(args):
    started=time.monotonic()
    if not 0<float(args.max_seconds)<=1800: raise ValueError('Invalid experiment time budget')
    torch.set_num_threads(2)
    if args.replay_only:
        stored=B.read_json(Path(args.out)/'contract.json')
        _,_,contract=prepare_contract(args,execution_sha=stored['code_sha'])
        B.replay_archive(args.out,contract)
        print('Complete CPU tangent replay verified',flush=True)
        return 0
    if Path(args.out).exists(): raise FileExistsError('Experiment output already exists')
    if not Path(args.out).resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Experiment output escapes approved private root')
    deadline=started+float(args.max_seconds)
    train,selection,contract=prepare_contract(args)
    # Full old numeric validation, not only a marker/hash check, before new images.
    prior_contract=P.A.read_json(Path(args.reference)/'contract.json')
    P.A.replay_archive(args.reference,prior_contract)
    P.check_deadline(deadline)
    OUTPUT_ROOT.mkdir(mode=0o700,exist_ok=True)
    if OUTPUT_ROOT.stat().st_mode&0o777!=0o700: raise ValueError('Private output root permissions differ')
    B.create_archive(args.out,contract)
    device=torch.device('cuda',args.gpu)
    model,checkpoint=P.I.model_from_export(args.checkpoint,device)
    model.eval().requires_grad_(False); P.validate_checkpoint(checkpoint,model)
    if P.A.model_state_digest(model)!=P.A.CHECKPOINT_STATE_SHA:
        raise ValueError('Loaded checkpoint state differs')
    if sum(p.numel() for p in model.masks.embed_full.parameters())!=74016:
        raise ValueError('Shared head parameter count differs')
    dataset=P.AURDataset(train,crop=tuple(RECIPE['crop']),crops_per_volume=2,do_augment=True,
                         lesion_share=.5,seed=RECIPE['seed']); dataset.set_epoch(0)
    for ordinal,member in enumerate(selection):
        P.check_deadline(deadline)
        current=P.input_manifest([member])
        if current!={k:v for k,v in contract['inputs'].items() if k in current}:
            raise ValueError('Selected inputs changed before volume load')
        crops=dataset[member['train_row_position']]
        for crop_index,crop in enumerate(crops):
            index=2*ordinal+crop_index
            reference_meta=P.A.read_json(Path(args.reference)/f'crop_{index:02d}.json')
            with np.load(Path(args.reference)/f'crop_{index:02d}.npz',allow_pickle=False) as z:
                reference_arrays={k:z[k] for k in z.files}
            meta,arrays=collect_crop(model,crop,reference_meta,reference_arrays,device=device,deadline=deadline)
            B.write_crop(args.out,index,meta,arrays)
            print(f'Completed crops {index+1}/{RECIPE["n_crops"]}',flush=True)
        del crops
    _,_,after=prepare_contract(args,execution_sha=contract['code_sha'])
    if after!=contract: raise ValueError('Final input/source/reference contract differs')
    if P.A.model_state_digest(model)!=P.A.CHECKPOINT_STATE_SHA:
        raise ValueError('Final model state differs')
    P.check_deadline(deadline)
    B.complete_archive(args.out,contract,deadline=deadline)
    print('Frozen TRAIN tangent experiment complete',flush=True)
    return 0


def parser():
    p=P.parser(); p.description=__doc__
    p.add_argument('--reference',type=Path,default=REFERENCE)
    return p


if __name__=='__main__':
    try:
        raise SystemExit(run(parser().parse_args()))
    except Exception as exc:
        print(f'Experiment incomplete ({type(exc).__name__}); no completed result accepted',file=sys.stderr,flush=True)
        raise SystemExit(1) from None
