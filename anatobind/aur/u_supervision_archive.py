"""Private, complete-only numeric archive for the frozen TRAIN diagnostic.

Replay is read-only and never constructs a model or opens a medical image. A caller
must supply the freshly verified external input/code/checkpoint contract.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from anatobind.aur.confidence_diagnostics import array_digest
from anatobind.aur.ssl.checkpoint import file_sha256

RECIPE = dict(seed=20261018, crop=[128,160,160], points=16384, n_queries=64,
              n_rows=6, n_crops=12, crops_per_volume=2, epoch=0, lesion_share=.5,
              do_augment=True, profile='unmatched_empty_v1',
              pairs=[['bmsr','T1c'],['isles','DWI'],['pdgm','FLAIR']])
SCORES = {'bmsr':.70, 'isles':.60, 'pdgm':.75}
CHECKPOINT_STATE_SHA = '5fbb0d6f9513fdebf66831033c6a8d32710a05ce2ec900656c062a61546e482f'
ARRAY_DTYPES = {'gt_instance_map':np.dtype('int64'), 'valid':np.dtype('float32'),
                'point_weight':np.dtype('float32'), 'detection_map':np.dtype('int32'),
                'presence_logits':np.dtype('float32'), 'points':np.dtype('int64'),
                'sampled_mask_logits':np.dtype('float32')}


def fixed_controls():
    return dict(parameters_frozen=True, eval_mode=True, parameter_gradients_none=True,
                optimizer=False, batch_size=1, decoder_workers=2, numerical_threads=2,
                activation='float32_before_sigmoid', profile=RECIPE['profile'],
                decoder='query_cc_v1', parameter_backward=False)


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n'
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def tensor_state_digest(state):
    """Canonical tensor-state commitment, also computable without a model."""
    h = hashlib.sha256()
    for name,tensor in sorted(state.items()):
        t=tensor.detach().cpu().contiguous()
        h.update(json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def model_state_digest(model):
    """Commit all parameters and buffers, including nonpersistent buffers."""
    return tensor_state_digest(dict(list(model.named_parameters())+list(model.named_buffers())))


def scalar_probe(probe):
    return {k:v for k,v in probe.items() if k not in ('mask_logit_gradient','presence_logit_gradient','points')}


def compare_values(actual, expected):
    """Exact schemas/IDs; tolerance only for floating scalar recomputation."""
    if isinstance(actual,dict) and isinstance(expected,dict):
        if actual.keys()!=expected.keys(): raise ValueError('Archive summary schema differs')
        for key in actual: compare_values(actual[key],expected[key])
    elif isinstance(actual,list) and isinstance(expected,list):
        if len(actual)!=len(expected): raise ValueError('Archive summary length differs')
        for a,b in zip(actual,expected): compare_values(a,b)
    elif isinstance(actual,float) and isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(actual,expected,atol=1e-6,rtol=1e-5):
            raise ValueError('Archive floating summary differs')
    elif type(actual) is not type(expected) or actual!=expected:
        raise ValueError('Archive exact summary differs')


def validate_arrays(meta, arrays):
    if arrays.keys()!=ARRAY_DTYPES.keys(): raise ValueError('Archive numeric array membership differs')
    shape = tuple(RECIPE['crop']); q,p=RECIPE['n_queries'],RECIPE['points']
    expected = {k:shape for k in ('gt_instance_map','valid','point_weight','detection_map')}
    expected.update(presence_logits=(q,),points=(2,p),sampled_mask_logits=(2,q,p))
    for key,value in arrays.items():
        if value.dtype!=ARRAY_DTYPES[key] or value.shape!=expected[key] or not np.isfinite(value).all():
            raise ValueError('Invalid archived numeric dtype, shape or finite values')
    gt,valid,weight,det=(arrays[k] for k in ('gt_instance_map','valid','point_weight','detection_map'))
    if any(not np.isin(v,[0,1]).all() for v in (valid,weight)) or (weight>valid).any():
        raise ValueError('Invalid validity or point weights')
    n=meta['n_instances']
    if type(n) is not int or n<0 or (gt<0).any() or set(np.unique(gt))-{0}!=set(range(1,n+1)):
        raise ValueError('Invalid retained GT cardinality')
    if ((gt>0)&((valid==0)|(weight==0))).any() or (det<0).any() or ((det>0)&(valid==0)).any():
        raise ValueError('Invalid retained GT or detection support')
    if (arrays['points']<0).any() or (arrays['points']>=gt.size).any():
        raise ValueError('Out of range sampled points')
    if meta['n_queries']!=q or meta['u_supervised'] is not True or meta['controls']!=fixed_controls():
        raise ValueError('Frozen query count, supervision or controls differ')
    if [meta['source'],meta['sequence']] not in RECIPE['pairs']:
        raise ValueError('Invalid source/sequence membership')
    if not np.isfinite(meta['voxel_mm3']) or meta['voxel_mm3']<=0:
        raise ValueError('Invalid voxel volume')
    hashes={k:array_digest(v) for k,v in arrays.items()}
    if meta.get('array_sha256')!=hashes: raise ValueError('Archived numeric hashes differ')
    crop_hashes=meta['crop_tensor_sha256']
    if (set(crop_hashes)!={'image','valid','coords','local','instance','point_weight'}
            or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
                   for v in crop_hashes.values())
            or any(crop_hashes[k]!=hashes[a] for k,a in
                   [('valid','valid'),('instance','gt_instance_map'),('point_weight','point_weight')])):
        raise ValueError('Crop tensor hashes differ from archived tensors')
    from anatobind.aur.losses import sample_points
    for draw in range(2):
        seed=RECIPE['seed']+1000*meta['row_ordinal']+10*meta['crop_ordinal']+draw
        sampled=sample_points(torch.from_numpy(valid)[None],p,torch.Generator().manual_seed(seed),
                              instance=torch.from_numpy(gt)[None])[0].numpy()
        if not np.array_equal(sampled,arrays['points'][draw]):
            raise ValueError('Archived point draw differs from frozen sampler recipe')
    rows=meta['detection_rows']
    ids=[r['instance'] for r in rows]
    if len(set(ids))!=len(ids) or set(ids)!=set(np.unique(det))-{0}:
        raise ValueError('Detection row/map identities differ')
    stats=meta['decoder_stats']
    if (stats['final_instances']!=len(rows) or any(type(stats[k]) is not int or stats[k]<0
            for k in ('raw_components','eligible_candidates','suppressed_candidates',
                      'fully_occluded_candidates','tiny_fragments','final_instances'))
            or stats['suppressed_candidates']>stats['eligible_candidates']
            or stats['eligible_candidates']>stats['raw_components']):
        raise ValueError('Decoder conservation counts differ')
    from anatobind.aur.eval import instance_rows
    dense={r['instance']:r for r in instance_rows(det,meta['voxel_mm3'])}
    floor=int(np.ceil(10./meta['voxel_mm3']-1e-9))
    for row in rows:
        actual=dense[row['instance']]
        if any(row[k]!=actual[k] for k in ('box','n_voxels','volume_mm3')) or row['n_voxels']<floor:
            raise ValueError('Detection geometry differs')
        origin=row['detector_origin']; query=origin['query']
        if type(query) is not int or not 0<=query<q or arrays['presence_logits'][query]<=0:
            raise ValueError('Invalid detector query')
        if (not .3<=row['score']<=1. or origin['parent_score']!=row['score']
            or origin['parent_voxels']<row['n_voxels']
            or origin['retained_fraction']!=row['n_voxels']/origin['parent_voxels']
            or origin['window']!=[[0,n] for n in shape]
            or any(type(origin[k]) is not int or origin[k]<=0 for k in ('component','rank','residual_child'))):
            raise ValueError('Invalid detector origin')


def reconstruct_crop(meta, arrays):
    """Rebuild original official targets/loss/matches and all reported measurements."""
    validate_arrays(meta,arrays)
    from anatobind.aur.u_supervision_diagnostics import probe_event_loss, compare_draw_assignments
    from anatobind.aur.u_supervision_components import detection_outcomes, component_measurements
    from anatobind.aur.eval import instance_rows, ignored_rows
    gt,valid,weight=(arrays[k] for k in ('gt_instance_map','valid','point_weight'))
    probes=[]
    for draw,points in enumerate(arrays['points']):
        probe=probe_event_loss(arrays['presence_logits'],arrays['sampled_mask_logits'][draw],
            gt.ravel()[points],weight.ravel()[points],points,meta['n_instances'],
            u_supervised=meta['u_supervised'],profile=RECIPE['profile'])
        probe.update(draw=draw,points=points)
        probes.append(probe)
    gt_rows=instance_rows(gt,meta['voxel_mm3'])+ignored_rows((valid>.5)&(weight<.5),meta['voxel_mm3'])
    outcomes=detection_outcomes(gt_rows,meta['detection_rows'],SCORES[meta['source']])
    components=component_measurements(arrays['detection_map'],meta['detection_rows'],gt,valid,weight,
                                    outcomes=outcomes,draws=probes)
    return dict(source=meta['source'],sequence=meta['sequence'],row_ordinal=meta['row_ordinal'],
                crop_ordinal=meta['crop_ordinal'],n_instances=meta['n_instances'],n_queries=meta['n_queries'],
                draws=[scalar_probe(p) for p in probes],components=components,outcomes=outcomes,
                matching_stability=compare_draw_assignments(probes[0],probes[1],meta['n_instances']))


def create_archive(out, contract, selection, inputs):
    out=Path(out)
    out.mkdir(mode=0o700,parents=False,exist_ok=False)
    os.chmod(out,0o700)
    atomic_json(out/'contract.json',contract)
    atomic_json(out/'selection_private.json',selection)
    atomic_json(out/'input_manifest_private.json',inputs)


def write_crop(out, ordinal, metadata, arrays):
    """Atomic pair writes; writing a crop never creates a completion marker."""
    out=Path(out); path=out/f'crop_{ordinal:02d}.npz'
    fd,temp=tempfile.mkstemp(prefix='.crop_',dir=out)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,**arrays)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    atomic_json(out/f'crop_{ordinal:02d}.json',metadata)


def expected_names():
    return {'contract.json','selection_private.json','input_manifest_private.json','aggregate.json'}|{
        f'crop_{i:02d}.{ext}' for i in range(RECIPE['n_crops']) for ext in ('json','npz')}


def archive_hashes(out):
    return {p.name:file_sha256(p) for p in sorted(Path(out).iterdir()) if p.name!='COMPLETE.json'}


def _rebuild(out, expected_contract, *, complete):
    out=Path(out)
    if out.stat().st_mode&0o777!=0o700: raise ValueError('Archive directory must be private')
    if expected_contract.get('recipe')!=RECIPE or expected_contract.get('controls')!=fixed_controls():
        raise ValueError('External frozen recipe or controls differ')
    if expected_contract.get('model_state_sha256')!=CHECKPOINT_STATE_SHA:
        raise ValueError('Frozen checkpoint state anchor differs')
    if read_json(out/'contract.json')!=expected_contract: raise ValueError('Archive external contract differs')
    selection=read_json(out/'selection_private.json'); inputs=read_json(out/'input_manifest_private.json')
    if selection!=expected_contract['selection'] or inputs!=expected_contract['inputs']:
        raise ValueError('Archive input or selection commitment differs')
    if len(selection)!=RECIPE['n_rows']: raise ValueError('Archive row count differs')
    names={p.name for p in out.iterdir()}
    if any(p.is_symlink() or not p.is_file() for p in out.iterdir()):
        raise ValueError('Archive contains a nonregular file')
    required=expected_names()|{'COMPLETE.json'} if complete else expected_names()-{'aggregate.json'}
    if names!=required: raise ValueError('Incomplete or unexpected archive file membership')
    if complete:
        marker=read_json(out/'COMPLETE.json')
        if marker!={'schema':1,'files':archive_hashes(out)}: raise ValueError('Archive completion checksums differ')
    records=[]; states=set()
    for i in range(RECIPE['n_crops']):
        meta=read_json(out/f'crop_{i:02d}.json')
        row,crop=divmod(i,RECIPE['crops_per_volume'])
        member=selection[row]['row']
        if (meta['row_ordinal'],meta['crop_ordinal'],meta['source'],meta['sequence'])!=(row,crop,member['source'],member['sequence']):
            raise ValueError('Archive crop selection membership differs')
        if meta['state_before']!=meta['state_after'] or meta['state_before']!=CHECKPOINT_STATE_SHA:
            raise ValueError('Model state invariance control differs')
        states.add(meta['state_before'])
        with np.load(out/f'crop_{i:02d}.npz',allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        actual=reconstruct_crop(meta,arrays)
        compare_values(actual,meta['summaries'])
        records.append(actual)
    if len(states)!=1: raise ValueError('Model state changed across crops')
    from anatobind.aur.u_supervision_components import aggregate_records
    result=aggregate_records(records)
    if complete: compare_values(result,read_json(out/'aggregate.json'))
    return result


def complete_archive(out, expected_contract, *, deadline=None):
    result=_rebuild(out,expected_contract,complete=False)
    atomic_json(Path(out)/'aggregate.json',result)
    if deadline is not None and time.monotonic()>=deadline:
        raise TimeoutError('Diagnostic time budget exhausted before completion')
    atomic_json(Path(out)/'COMPLETE.json',dict(schema=1,files=archive_hashes(out)))
    return result


def replay_archive(out, expected_contract):
    try:
        return _rebuild(out,expected_contract,complete=True)
    except (KeyError,TypeError,OSError,IndexError) as exc:
        raise ValueError('Invalid or incomplete diagnostic archive') from exc
