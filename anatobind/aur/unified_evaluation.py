"""Frozen development A/U/R evaluation for standalone planes and volumes.

All references are pseudo-labels (NOT_EVIDENCE). A complete central canonical
plane is chosen from headers, independently of labels and predictions. No
threshold search or new quality gate is implemented here.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from anatobind.aur.crops import normalise
from anatobind.aur.data_contract import (decoder_environment, file_sha256, json_digest,
    read_contract, validate_private_output, write_private_json)
from anatobind.aur.eval import host_dice, ignored_rows, instance_rows
from anatobind.aur.infer import bind_instances
from anatobind.aur.labels import ENTITY_LABELS, HOST_NAMES, entity_to_synthseg, host_map
from anatobind.aur.planar_relations import planar_evaluation_reference
from anatobind.aur.targets import host_targets
from anatobind.aur.unified_data import load_unified_volume
from anatobind.aur.unified_infer import _ModeView, infer_input, model_from_unified_export
from anatobind.eval.detection_metrics import scan_matches

VERSION = 'aur_development_evaluation_v1'
DEFAULT_CROPS = {'2d': (1, 160, 160), '3d': (128, 160, 160)}


def _modes(modes):
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(m not in ('2d','3d') for m in modes):
        raise ValueError('Invalid evaluation modes')
    return modes


def _header_grid(path):
    """Compute canonical shape/affine without materializing a reoriented proxy."""
    image = nib.load(path)
    if len(image.shape) != 3 or any(n <= 0 for n in image.shape):
        raise ValueError('Invalid evaluation image header')
    affine = np.asarray(image.affine, np.float64)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3])) <= 0:
        raise ValueError('Invalid evaluation image header')
    orientation = nib.orientations.ornt_transform(nib.io_orientation(affine),
                                                 nib.orientations.axcodes2ornt(('R','A','S')))
    shape = np.asarray(image.shape)[np.argsort(orientation[:,0])]
    canonical_affine = affine @ nib.orientations.inv_ornt_aff(orientation, image.shape)
    return tuple(int(n) for n in shape[::-1]), canonical_affine


def _freeze_verified(contract, modes):
    modes = _modes(modes)
    rows = deepcopy(contract['validation'])
    if not rows or any(row.get('role') != 'development_validation' or row.get('split') != 'val' for row in rows):
        raise ValueError('Evaluation requires development validation members')
    identities = [(r['source'], r['patient'], r['case'], r['sequence']) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate development evaluation member')
    observations = []
    for index, row in enumerate(rows):
        shape, affine = _header_grid(row['image'])
        for mode in modes:
            plane = shape[0]//2 if mode == '2d' else None
            local_affine = affine.copy()
            if plane is not None:
                local_affine[:3,3] += local_affine[:3,2]*plane
            observations.append(dict(row_index=index, row_sha256=json_digest(row), input_mode=mode,
                plane_index=plane, shape=[1,*shape[1:]] if mode=='2d' else list(shape),
                source_shape=list(shape), affine_ras=local_affine.tolist()))
    return dict(version=VERSION, evidence='NOT_EVIDENCE', contract_sha256=contract['contract_sha256'],
                selection='complete_canonical_central_plane', modes=list(modes), rows=rows,
                observations=observations)


def freeze_observations(contract, modes=('2d','3d'), *, qualified_source_root=None):
    """Verify a file contract, then select complete unaugmented observations by header."""
    frozen = read_contract(contract, source_mode='evaluation', qualified_source_root=qualified_source_root)
    try:
        return _freeze_verified(frozen, modes)
    except Exception:
        raise ValueError('Invalid development evaluation membership or image headers') from None


def _supports(pred, truth, labels):
    return dict(labels=list(labels), truth=[int(np.count_nonzero(truth==v)) for v in labels],
                predicted=[int(np.count_nonzero(pred==v)) for v in labels],
                intersection=[int(np.count_nonzero((pred==v)&(truth==v))) for v in labels])


def _with_dice(supports):
    out = dict(supports)
    out['dice'] = [2*i/(p+t) if p+t else None for i,p,t in
                   zip(out['intersection'], out['predicted'], out['truth'])]
    available = [v for v in out['dice'] if v is not None]
    out['macro'] = float(np.mean(available)) if available else None
    return out


def anatomy_metrics(prediction, truth, a_ignore):
    """A32/host13 Dice on the valid anatomy grid, with both support counts."""
    pred = entity_to_synthseg(np.asarray(prediction))
    true = entity_to_synthseg(np.asarray(truth))
    ignore = np.asarray(a_ignore, bool)
    if pred.shape != true.shape or ignore.shape != true.shape:
        raise ValueError('Invalid anatomy evaluation grids')
    pred, true = np.where(ignore,0,pred), np.where(ignore,0,true)
    scores = host_dice(pred, true)
    entities = _with_dice(_supports(pred, true, [int(v) for v in ENTITY_LABELS]))
    hosts = _with_dice(_supports(host_map(pred), host_map(true), list(range(1,14))))
    hosts['labels'] = list(HOST_NAMES)
    # Keep the existing implementation's definition: classes present in either
    # prediction or truth. Neither-empty classes are explicitly unavailable.
    entities['macro'] = float(scores['entity_macro']) if np.isfinite(scores['entity_macro']) else None
    hosts['macro'] = float(scores['host_macro']) if np.isfinite(scores['host_macro']) else None
    return dict(entities=entities, hosts=hosts, n_valid=int((~ignore).sum()))


def detection_metrics(gt, detections, *, score_threshold):
    """Fixed score filter before existing one-to-one IoU>=.1 ignored-GT matching."""
    hits, fp, kept = scan_matches(dict(gt=gt,dets=detections), score_threshold, iou=.1)
    return dict(n_gt=sum(not g.get('ignore',False) for g in gt), n_hit=len(hits), n_fp=fp,
                n_ignored=sum(bool(g.get('ignore',False)) for g in gt),
                matches={int(gt[g]['instance']): int(kept[d]['instance']) for g,d in hits.items()},
                gt=gt, detections=kept)


def _plane_truth(volume, plane, area):
    """Planar U uses decoded raw lesion, independently of the 3D volume floor."""
    sl = slice(plane,plane+1)
    mask = ((volume['instance']>0)|volume['small'])[sl]
    components, _ = ndimage.label(mask[0], structure=np.ones((3,3), bool))
    instances = np.zeros(mask.shape, np.int32)
    rows, source_ids = [], []
    for component, box in enumerate(ndimage.find_objects(components),1):
        if box is None: continue
        selected = components==component
        count = int(selected.sum())
        ignored = count*area < 1.*(1-1e-6)
        instance = None if ignored else len(source_ids)+1
        rows.append(dict(instance=instance, box=[0,box[0].start,box[1].start,1,box[0].stop,box[1].stop],
                         n_voxels=count, area_mm2=count*area, ignore=bool(ignored)))
        if not ignored:
            instances[0,selected] = instance
            ids = np.unique(volume['instance'][sl][0,selected])
            source_ids.append(int(ids[0]) if len(ids)==1 and ids[0]>0 else None)
    return instances, rows, source_ids


def _planar_targets(volume, row, instances, source_ids, plane, area):
    sl = slice(plane,plane+1)
    n = len(source_ids)
    truth = np.full(n,-1,np.int64)
    reasons = ['source_identity_unavailable']*n
    good = [i for i,k in enumerate(source_ids) if k is not None]
    compact = np.zeros_like(instances)
    for j,i in enumerate(good,1): compact[instances==i+1] = j
    if good:
        reference = planar_evaluation_reference(volume['instance'], volume['source_host_map'],
            dict(instance=compact,host_map=volume['source_host_map'][sl]), np.ones_like(instances,bool),
            volume['a_ignore'][sl], [source_ids[i] for i in good], area,
            voxel_volume_mm3=volume['voxel_mm3'], plane_index=plane,split=row['split'],role=row['role'],
            a_supervised=bool(row.get('a_supervised',False)),u_supervised=bool(row.get('u_supervised',False)),
            r_supervised=bool(row.get('r_supervised',False)))
        for j,i in enumerate(good):
            truth[i] = reference['host'][j]
            reasons[i] = reference['reasons'][j]
    return truth, reasons


def _binding_metrics(truth, controlled, b0, matches, predicted, predicted_b0, reasons):
    eligible = [i+1 for i,t in enumerate(truth) if t>=0]
    by_id = {int(r['instance']):r for r in controlled}
    detections = {int(r['instance']):r for r in predicted}
    rows = []
    for k in eligible:
        target, controlled_host = int(truth[k-1]), int(by_id[k]['host'])
        matched = matches.get(k)
        detected_host = int(detections[matched]['host']) if matched is not None else None
        detected_b0 = int(predicted_b0[matched-1]) if matched is not None else None
        rows.append(dict(instance=k,truth=target,controlled=controlled_host,b0=int(b0[k-1]),
                         detected_instance=matched,end_to_end=detected_host,end_to_end_b0=detected_b0))
    return dict(eligible=len(eligible), unavailable_instances=len(truth)-len(eligible),
        controlled_hits=sum(r['controlled']==r['truth'] for r in rows),
        controlled_b0_hits=sum(r['b0']==r['truth'] for r in rows),
        end_to_end_hits=sum(r['end_to_end']==r['truth'] for r in rows),
        end_to_end_b0_hits=sum(r['end_to_end_b0']==r['truth'] for r in rows),
        matched_eligible=sum(r['detected_instance'] is not None for r in rows),
        instances=rows, unavailable_reasons=[r for r in reasons if r], evidence='NOT_EVIDENCE')


def evaluate_observation(model, row, observation, *, crop, device='cpu', score_threshold=.5):
    """Run real inference and controlled binding on one frozen observation."""
    try:
        return _evaluate_observation(model,row,observation,crop=crop,device=device,score_threshold=score_threshold)
    except Exception:
        raise ValueError('Unable to evaluate development observation') from None


def _evaluate_observation(model,row,observation,*,crop,device,score_threshold):
    if row.get('role')!='development_validation' or row.get('split')!='val':
        raise ValueError()
    if not np.isfinite(score_threshold) or not 0<=score_threshold<=1: raise ValueError()
    if json_digest(row) != observation['row_sha256']: raise ValueError()
    mode, plane = observation['input_mode'], observation['plane_index']
    if mode not in ('2d','3d') or (mode=='3d' and plane is not None): raise ValueError()
    volume = load_unified_volume(row)
    raw, affine = volume['raw_image'], np.asarray(volume['affine']).copy()
    if list(raw.shape)!=observation['source_shape']: raise ValueError()
    sl = slice(None) if mode=='3d' else slice(plane,plane+1)
    raw = raw[sl]
    if mode=='2d': affine[:3,3] += affine[:3,2]*plane
    if list(raw.shape)!=observation['shape'] or not np.allclose(affine,observation['affine_ras']): raise ValueError()
    source = SimpleNamespace(input_mode=mode,image_dhw=raw,affine_ras=affine,source_refs=(),frame_map=())
    prediction = infer_input(model,source,crop,device=device,score_threshold=score_threshold)
    record = dict(source=row['source'],patient=row['patient'],case=row['case'],sequence=row['sequence'],
                  input_mode=mode,measurement_unit='mm2' if mode=='2d' else 'mm3',
                  observation=observation,a=None,u=None,r=None,evidence='NOT_EVIDENCE')
    if row.get('a_supervised',False):
        record['a'] = anatomy_metrics(prediction.anatomy,volume['entity'][sl],volume['a_ignore'][sl])
    if not row.get('u_supervised',False): return record
    area = float(np.linalg.norm(np.cross(affine[:3,0],affine[:3,1])))
    if mode=='2d':
        instances, gt, source_ids = _plane_truth(volume,plane,area)
        truth, reasons = _planar_targets(volume,row,instances,source_ids,plane,area)
    else:
        instances = volume['instance']
        gt = instance_rows(instances,volume['voxel_mm3'])+ignored_rows(volume['small'],volume['voxel_mm3'])
        truth, reasons = volume['hosts']['host'].copy(), ['']*int(instances.max(initial=0))
    record['u'] = detection_metrics(gt,prediction.lesions,score_threshold=score_threshold)
    if not row.get('r_supervised',False) or not row.get('a_supervised',False): return record
    eligible = np.flatnonzero(truth>=0)+1
    bind_map = np.where(np.isin(instances,eligible),instances,0)
    controlled = bind_instances(_ModeView(model,mode),normalise(raw),affine,bind_map,crop,torch.device(device))
    pred_seg = entity_to_synthseg(prediction.anatomy)
    # B0 always uses this observation's predicted anatomy and the same instances
    # as its corresponding learned binding track.
    b0 = host_targets(instances,pred_seg,volume['spacing'])['host']
    predicted_b0 = host_targets(prediction.instances,pred_seg,volume['spacing'])['host']
    record['r'] = _binding_metrics(truth,controlled,b0,record['u']['matches'],prediction.lesions,predicted_b0,reasons)
    return record


def _merge_anatomy(records):
    out = {}
    for name in ('entities','hosts'):
        supports = dict(labels=records[0]['a'][name]['labels'])
        for field in ('truth','predicted','intersection'):
            supports[field] = np.sum([r['a'][name][field] for r in records],axis=0).astype(int).tolist()
        out[name] = _with_dice(supports)
    out['n_valid'] = sum(r['a']['n_valid'] for r in records)
    return out


def _counts(records):
    return dict(n_observations=len(records),n_patients=len({(r['source'],r['patient']) for r in records}),
                n_sources=len({r['source'] for r in records}))


def _aggregate_mode(records, mode):
    out = {**_counts(records),'coverage':{k:sum(r[k] is not None for r in records) for k in ('a','u','r')},
           'a':None,'u':None,'r':None}
    a = [r for r in records if r['a'] is not None]
    if a: out['a'] = {**_merge_anatomy(a),**_counts(a)}
    u = [r for r in records if r['u'] is not None]
    if u:
        counts = {key:sum(r['u'][key] for r in u) for key in ('n_gt','n_hit','n_fp','n_ignored')}
        measure = 'area_mm2' if mode=='2d' else 'volume_mm3'
        physical = dict(truth=sum(g[measure] for r in u for g in r['u']['gt'] if not g.get('ignore')),
                        ignored=sum(g[measure] for r in u for g in r['u']['gt'] if g.get('ignore')),
                        predicted=sum(d[measure] for r in u for d in r['u']['detections']))
        out['u'] = {**counts,**_counts(u),'sensitivity':counts['n_hit']/counts['n_gt'] if counts['n_gt'] else None,
                    'fp_per_image' if mode=='2d' else 'fp_per_scan':counts['n_fp']/len(u),measure:physical}
    r = [r for r in records if r['r'] is not None]
    if r:
        counts = {key:sum(item['r'][key] for item in r) for key in ('eligible','unavailable_instances',
            'controlled_hits','controlled_b0_hits','end_to_end_hits','end_to_end_b0_hits','matched_eligible')}
        out['r'] = {**counts,**_counts(r),'reference':'geometry_only_pseudo_reference','evidence':'NOT_EVIDENCE'}
        for track in ('controlled','controlled_b0','end_to_end','end_to_end_b0'):
            out['r'][track+'_accuracy'] = counts[track+'_hits']/counts['eligible'] if counts['eligible'] else None
    return out


def aggregate_observations(records):
    """Anonymous aggregate; missing supervision and zero denominators remain null."""
    records = list(records)
    return dict(version=VERSION,evidence='NOT_EVIDENCE',quality_certified=False,**_counts(records),
        modes={mode:_aggregate_mode([r for r in records if r['input_mode']==mode],mode) for mode in ('2d','3d')},
        sources={source:{mode:_aggregate_mode([r for r in records if r['source']==source and r['input_mode']==mode],mode)
                         for mode in ('2d','3d')} for source in sorted({r['source'] for r in records})})


def _evaluator_source():
    from anatobind.aur.unified_launcher import REPO, source_snapshot
    snapshot = source_snapshot()
    snapshot['scripts/aur_unified_eval.py'] = file_sha256(REPO/'scripts/aur_unified_eval.py')
    return snapshot


def run_evaluation(contract_path, export_path, out_dir, *, device='cpu', modes=('2d','3d'),
                   crops=None, score_threshold=.5, qualified_source_root=None):
    """Freeze configuration, infer all development observations, and publish privately.

    An optional source archive supplies exact qualification bytes missing from
    historical contracts. It is checked against their immutable hashes; it does
    not imply that an archive reconstructed later existed at qualification time.
    """
    try:
        out = validate_private_output(out_dir)
        if out.exists(): raise ValueError()
        modes = _modes(modes)
        crops = {mode:tuple((crops or DEFAULT_CROPS)[mode]) for mode in modes}
        if (not np.isfinite(score_threshold) or not 0<=score_threshold<=1 or any(
                len(crop)!=3 or any(type(n) is not int or n<=0 for n in crop) or
                any(n%p for n,p in zip(crop,(1,4,4) if mode=='2d' else (2,4,4))) or
                (mode=='2d' and crop[0]!=1) for mode,crop in crops.items())): raise ValueError()
        frozen = read_contract(contract_path,source_mode='evaluation',qualified_source_root=qualified_source_root)
        plan = _freeze_verified(frozen,modes)
        snapshot = _evaluator_source()
        exported_sha = file_sha256(export_path)
        payload = torch.load(export_path,map_location='cpu',weights_only=True)
        identity = payload['data_contract']
        if not isinstance(identity,dict) or identity.get('contract_sha256') != frozen['contract_sha256']: raise ValueError()
        del payload
        # The certificate-bound training runtime identity is preserved. Current
        # evaluation source is committed separately, never substituted for it.
        plan.update(configuration=dict(crops=crops,score_threshold=float(score_threshold),device=str(device),
                    lesion_threshold=.3,entity_threshold=.5,presence_threshold=.5,overlap=.5,
                    planar_min_area_mm2=1.,volume_min_mm3=10.,matching_bbox_iou=.1),
                    artifact_identity=identity,export_sha256=exported_sha,evaluator_source=snapshot,
                    evaluator_source_sha256=json_digest(snapshot),environment=decoder_environment(),
                    qualified_source_root=str(Path(qualified_source_root).expanduser().resolve()) if qualified_source_root else None,
                    source_verification='historical_qualified_bytes_with_current_roles_inputs_environment')
        write_private_json(out/'plan_private.json',plan)
        model, metadata = model_from_unified_export(export_path,device=device,data_contract=identity)
        records = [evaluate_observation(model,plan['rows'][o['row_index']],o,crop=crops[o['input_mode']],
                    device=device,score_threshold=score_threshold) for o in plan['observations']]
        # Recheck all committed inputs, evaluator code and artifact bytes before
        # publishing a completion marker. A failed run leaves its private plan.
        closed = read_contract(contract_path,source_mode='evaluation',qualified_source_root=qualified_source_root)
        if (closed['contract_sha256']!=plan['contract_sha256'] or file_sha256(export_path)!=exported_sha or
                _evaluator_source()!=snapshot): raise ValueError()
        summary = aggregate_observations(records)
        summary['artifact'] = {key:metadata[key] for key in ('format_version','stage','pilot_only','phase_completed','quality_certified')}
        summary['contract_sha256'] = frozen['contract_sha256']
        summary['export_sha256'] = exported_sha
        summary['evaluator_source_sha256'] = json_digest(snapshot)
        write_private_json(out/'records_private.json',records)
        write_private_json(out/'anonymous_summary.json',summary)
        members = {name:file_sha256(out/name) for name in ('plan_private.json','records_private.json','anonymous_summary.json')}
        write_private_json(out/'complete.json',dict(version=VERSION,evidence='NOT_EVIDENCE',outputs=members))
        return summary
    except Exception:
        raise ValueError('Development evaluation failed; private evidence preserved') from None
