#!/usr/bin/env python
"""Bounded frozen TRAIN U-supervision diagnostic; no training or score search."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from anatobind.aur import infer as I
from anatobind.aur import u_supervision_archive as A
from anatobind.aur.confidence_diagnostics import array_digest
from anatobind.aur.dataset import AURDataset
from anatobind.aur.losses import sample_points
from anatobind.aur.query_decoder import QueryCCDecoder, decoder_rules
from anatobind.aur.ssl.checkpoint import code_sha, file_sha256
from anatobind.aur.train import split_rows

DATA_ROOT=Path('/data2/congcong/data/FM_data')
DATA=DATA_ROOT/'derived/aur'
OUTPUT_ROOT=Path('/home/congcongliu/outputs/aur_u_train_supervision_20261018')
SELECTION=Path('/home/congcongliu/outputs/aur_u_train_supervision_20261018/design_preflight/selection_private.json')
FROZEN_HASHES=dict(samples='fb96acd741aa6c0ee689eb772a4332304501c8d63ebd7ccddcd657f41eb86d5d',
 val_patients='475f91d97d8b03f312df560f98f39ab81c068c39d0a24c632d8efa5fd52d9d34',
 checkpoint='38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde',
 selection_private='7f542122cc1dd1623f616681d47b96c7003e1488718ab33f964533bfb5a42bfb')
POSITIONS_SHA='719779c337e9240e828abb1db94cb48c42e8c1c1f305624620974252de084858'


def check_deadline(deadline):
    if time.monotonic()>=deadline: raise TimeoutError('Diagnostic time budget exhausted')


def digest_json(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def source_hashes():
    tracked=subprocess.run(['git','ls-files','-z','anatobind/**/*.py'],cwd=ROOT,
                           capture_output=True,check=True).stdout.decode().split('\0')
    paths=set(p for p in tracked if p)
    paths.update(['scripts/aur_diagnose_u_supervision.py','anatobind/aur/u_supervision_archive.py',
                  'anatobind/aur/u_supervision_diagnostics.py','anatobind/aur/u_supervision_components.py'])
    # The approved design/rules are part of the executable contract too.
    paths.add('docs/superpowers/specs/2026-10-18-aur-u-train-supervision-design.md')
    return {p:file_sha256(ROOT/p) for p in sorted(paths)}


def confined(path):
    path=Path(path).resolve(strict=True)
    if not path.is_relative_to(DATA_ROOT.resolve()): raise ValueError('Input path escapes approved data root')
    return path


def select_train_rows(rows, val):
    train,held=split_rows(rows,val)
    excluded={(r['source'],str(r['patient'])) for r in held}
    excluded|={(r['source'],str(r['patient'])) for r in rows if r.get('split')=='test'}
    selected=[]
    for source,sequence in A.RECIPE['pairs']:
        candidates=[(i,r) for i,r in enumerate(train) if r['source']==source and r['sequence']==sequence
                    and r.get('u_supervised') is True and (source,str(r['patient'])) not in excluded]
        candidates.sort(key=lambda pair:(digest_json([A.RECIPE['seed'],source,str(pair[1]['patient']),sequence]),pair[0]))
        patients=set()
        for i,row in candidates:
            if str(row['patient']) in patients: continue
            patients.add(str(row['patient'])); selected.append(dict(train_row_position=i,row=row))
            if len(patients)==2: break
        if len(patients)!=2: raise ValueError('Insufficient fixed TRAIN membership')
    if len(selected)!=A.RECIPE['n_rows']: raise ValueError('Fixed TRAIN row count differs')
    return train,selected


def input_manifest(selection):
    result={}
    for member in selection:
        for key in ('image','anatomy','lesion'):
            value=member['row'].get(key)
            if value:
                path=confined(value)
                result[str(path)]=file_sha256(path)
    return result


def prepare_contract(args, *, execution_sha=None):
    """Metadata/hash preflight only; membership is fixed before volume/model load."""
    for name in ('samples','val_patients','checkpoint','selection_private'):
        path=getattr(args,name)
        if name!='selection_private': confined(path)
        if file_sha256(path)!=FROZEN_HASHES[name]: raise ValueError('Frozen input commitment differs')
    rows=A.read_json(args.samples)
    if not isinstance(rows,list): raise ValueError('Frozen samples must be a row list')
    train,selection=select_train_rows(rows,A.read_json(args.val_patients))
    if selection!=A.read_json(args.selection_private): raise ValueError('Frozen TRAIN selection differs')
    if digest_json([m['train_row_position'] for m in selection])!=POSITIONS_SHA:
        raise ValueError('Frozen TRAIN row positions differ')
    inputs=input_manifest(selection)
    contract=dict(recipe=A.RECIPE,controls=A.fixed_controls(),selection=selection,inputs=inputs,
                  input_sha256=FROZEN_HASHES,checkpoint_sha256=FROZEN_HASHES['checkpoint'],
                  code_sha=execution_sha or code_sha(),source_sha256=source_hashes(),
                  model_state_sha256=A.CHECKPOINT_STATE_SHA,
                  decoder_rules=decoder_rules('query_cc_v1'),scores=A.SCORES,
                  environment=dict(python=sys.version.split()[0],torch=str(torch.__version__),numpy=np.__version__))
    return train,selection,contract


def validate_checkpoint(meta,model):
    cfg=meta.get('config',{})
    if (meta.get('stage',cfg.get('stage'))!='III' or meta.get('step')!=3000 or meta.get('seen_crops')!=72000
        or meta.get('seed')!=0 or cfg.get('u_mask_profile')!=A.RECIPE['profile']
        or cfg.get('points')!=A.RECIPE['points'] or list(cfg.get('crop',[]))!=A.RECIPE['crop']
        or cfg.get('crops_per_volume')!=2 or cfg.get('lesion_share')!=.5
        or model.cfg['n_events']!=A.RECIPE['n_queries']):
        raise ValueError('Frozen checkpoint profile or model configuration differs')


def collect_crop(model,crop,voxel_mm3,*,source,sequence,selection_ordinal,crop_ordinal,device,deadline):
    """One frozen forward, unchanged dense decoder, and two detached CPU probes."""
    started=time.monotonic()
    check_deadline(deadline)
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Diagnostic model must be eval, frozen and gradient-free')
    if model.cfg['n_events']!=A.RECIPE['n_queries'] or tuple(crop['valid'].shape)!=tuple(A.RECIPE['crop']):
        raise ValueError('Frozen model query count or crop shape differs')
    if not bool(crop['u_supervised']): raise ValueError('Selected crop must be U supervised')
    before=A.model_state_digest(model)
    arrays={k:crop[v].detach().cpu().numpy().astype(A.ARRAY_DTYPES[k],copy=True)
            for k,v in [('gt_instance_map','instance'),('valid','valid'),('point_weight','point_weight')]}
    points=[]; sampled=[]
    args=[crop[k][None].to(device) for k in ('image','valid','coords','local')]
    with torch.no_grad():
        with I._autocast(device):
            out=model(*args)
            presence=out['event_presence'].float()
            ev_pres=presence.sigmoid()[0]
            event=model.event_masks(out).float().sigmoid()[0]
            for draw in range(2):
                generator=torch.Generator().manual_seed(A.RECIPE['seed']+1000*selection_ordinal+10*crop_ordinal+draw)
                pts=sample_points(crop['valid'][None].cpu(),A.RECIPE['points'],generator,instance=crop['instance'][None].cpu())
                points.append(pts[0].numpy().copy())
                sampled.append(model.event_masks(out,pts.to(device)).float()[0].cpu().numpy().copy())
    arrays.update(presence_logits=presence[0].cpu().numpy().copy(),points=np.stack(points),sampled_mask_logits=np.stack(sampled))
    if not bool(torch.isfinite(event).all()) or not bool(torch.isfinite(ev_pres).all()):
        raise ValueError('Nonfinite dense query or presence probabilities')
    check_deadline(deadline)
    decoder=QueryCCDecoder(tuple(A.RECIPE['crop']),voxel_mm3,workers=2)
    try:
        decoder(window=tuple((0,n) for n in A.RECIPE['crop']),event_prob=event,event_presence=ev_pres,
                entity_prob=None,valid=crop['valid'],weight=None)
        detection_map,rows,stats=decoder.finish()
    finally:
        decoder._executor.shutdown(wait=True,cancel_futures=True)
    arrays['detection_map']=detection_map.astype(np.int32,copy=False)
    after=A.model_state_digest(model)
    if before!=after or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Model state or parameter gradients changed')
    meta=dict(row_ordinal=int(selection_ordinal),crop_ordinal=int(crop_ordinal),source=source,sequence=sequence,
              n_instances=int(crop['n_instances']),n_queries=A.RECIPE['n_queries'],u_supervised=True,
              voxel_mm3=float(voxel_mm3),detection_rows=rows,decoder_stats=stats,controls=A.fixed_controls(),
              state_before=before,state_after=after,array_sha256={k:array_digest(v) for k,v in arrays.items()},
              crop_tensor_sha256={k:array_digest(crop[k].detach().cpu().numpy()) for k in ('image','valid','coords','local','instance','point_weight')})
    meta['summaries']=A.reconstruct_crop(meta,arrays)
    meta['resources']=dict(elapsed_seconds=time.monotonic()-started,
        rss_max_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,
        cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.type=='cuda' else 0)
    check_deadline(deadline)
    return meta,arrays


def run(args):
    started=time.monotonic()
    if not 0<float(args.max_seconds)<=1800: raise ValueError('Time budget must be in (0,1800]')
    torch.set_num_threads(2)
    if args.replay_only:
        stored=A.read_json(Path(args.out)/'contract.json')
        _,_,contract=prepare_contract(args,execution_sha=stored['code_sha'])
        A.replay_archive(args.out,contract)
        print('Complete CPU replay verified',flush=True)
        return 0
    if Path(args.out).exists(): raise FileExistsError('Diagnostic output already exists')
    if not Path(args.out).resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Diagnostic must use the approved private output root')
    deadline=started+float(args.max_seconds)
    train,selection,contract=prepare_contract(args)
    check_deadline(deadline)
    A.create_archive(args.out,contract,selection,contract['inputs'])
    device=torch.device('cuda',args.gpu)
    model,meta=I.model_from_export(args.checkpoint,device)
    model.eval().requires_grad_(False)
    validate_checkpoint(meta,model)
    original_state=A.model_state_digest(model)
    if original_state!=A.CHECKPOINT_STATE_SHA: raise ValueError('Loaded model differs from checkpoint state anchor')
    dataset=AURDataset(train,crop=tuple(A.RECIPE['crop']),crops_per_volume=2,do_augment=True,lesion_share=.5,seed=A.RECIPE['seed'])
    dataset.set_epoch(0)
    import nibabel as nib
    for ordinal,member in enumerate(selection):
        check_deadline(deadline)
        current_inputs=input_manifest([member])
        if current_inputs!={k:v for k,v in contract['inputs'].items() if k in current_inputs}:
            raise ValueError('Selected input changed before volume loading')
        # Header-only read yields the exact affine volume used by load_volume.
        voxel_mm3=abs(float(np.linalg.det(np.asarray(nib.load(member['row']['image']).affine)[:3,:3])))
        crops=dataset[member['train_row_position']]
        for crop_ordinal,crop in enumerate(crops):
            metadata,arrays=collect_crop(model,crop,voxel_mm3,source=member['row']['source'],sequence=member['row']['sequence'],
                selection_ordinal=ordinal,crop_ordinal=crop_ordinal,device=device,deadline=deadline)
            A.write_crop(args.out,2*ordinal+crop_ordinal,metadata,arrays)
            print(f'Completed crops {2*ordinal+crop_ordinal+1}/{A.RECIPE["n_crops"]}',flush=True)
        del crops
    if A.model_state_digest(model)!=original_state: raise ValueError('Final model state differs')
    _,_,after=prepare_contract(args,execution_sha=contract['code_sha'])
    if after!=contract: raise ValueError('Final input/code/checkpoint contract differs')
    check_deadline(deadline)
    A.complete_archive(args.out,contract,deadline=deadline)
    print('Frozen TRAIN diagnostic complete',flush=True)
    return 0


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,default=DATA/'samples_1mm_v1.json')
    p.add_argument('--val-patients',type=Path,default=DATA/'ssl_manifest_v1/val_patients.json')
    p.add_argument('--checkpoint',type=Path,default=DATA/'ssl_runs/c2_stage3_80k_20261014_1344/aur_stage3_best.pt')
    p.add_argument('--selection-private',type=Path,default=SELECTION)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--max-seconds',type=float,default=1800)
    p.add_argument('--replay-only',action='store_true')
    return p


if __name__=='__main__':
    try:
        raise SystemExit(run(parser().parse_args()))
    except Exception as exc:
        print(f'Diagnostic incomplete ({type(exc).__name__}); no completed result accepted',file=sys.stderr,flush=True)
        raise SystemExit(1) from None
