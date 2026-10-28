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
GATE_VERSION = 'aur_gate_evaluation_v1'
ROLES = {'development_validation':'val', 'historical_test':'test'}
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


def _binding_metrics(truth, controlled, b0, matches, predicted, predicted_b0, reasons, shares=None):
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
        if shares is not None:
            rows[-1]['top_share'] = float(np.max(shares[k-1]))
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
    gate = observation.get('gate_version') == GATE_VERSION
    if (ROLES.get(row.get('role')) != row.get('split') or
            (not gate and row.get('role')!='development_validation')):
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
    if gate:
        record.update({key:deepcopy(observation[key]) for key in
                       ('row_sha256','plane_index','plane_kinds','target_instances','pixel_area_mm2')})
        record['r_inplane'] = None
    if row.get('a_supervised',False):
        record['a'] = anatomy_metrics(prediction.anatomy,volume['entity'][sl],volume['a_ignore'][sl])
    if not row.get('u_supervised',False): return record
    area = float(np.linalg.norm(np.cross(affine[:3,0],affine[:3,1])))
    if mode=='2d':
        instances, gt, source_ids = _plane_truth(volume,plane,area)
        if gate and row['role']=='historical_test':
            truth = np.full(len(source_ids),-1,np.int64)
            reasons = ['development_only_strict_reference']*len(source_ids)
        else:
            truth, reasons = _planar_targets(volume,row,instances,source_ids,plane,area)
    else:
        instances = volume['instance']
        gt = instance_rows(instances,volume['voxel_mm3'])+ignored_rows(volume['small'],volume['voxel_mm3'])
        truth, reasons = volume['hosts']['host'].copy(), ['']*int(instances.max(initial=0))
    record['u'] = detection_metrics(gt,prediction.lesions,score_threshold=score_threshold)
    if gate:
        from anatobind.aur.gate_metrics import matched_instance_dice
        record['u']['matched_dice'] = matched_instance_dice(prediction.instances,instances,record['u']['matches'])
        record['u']['source_ids'] = source_ids if mode=='2d' else list(range(1,int(instances.max(initial=0))+1))
        if mode=='2d' and row.get('a_supervised',False):
            record['r_inplane'] = _inplane_relation(instances,entity_to_synthseg(volume['entity'][sl]),
                volume['spacing'],record['u']['matches'],prediction.lesions,
                host_targets(prediction.instances,entity_to_synthseg(prediction.anatomy),volume['spacing'])['host'])
    if not row.get('r_supervised',False) or not row.get('a_supervised',False): return record
    eligible = np.flatnonzero(truth>=0)+1
    bind_map = np.where(np.isin(instances,eligible),instances,0)
    controlled = bind_instances(_ModeView(model,mode),normalise(raw),affine,bind_map,crop,torch.device(device))
    pred_seg = entity_to_synthseg(prediction.anatomy)
    # B0 always uses this observation's predicted anatomy and the same instances
    # as its corresponding learned binding track.
    b0 = host_targets(instances,pred_seg,volume['spacing'])['host']
    predicted_b0 = host_targets(prediction.instances,pred_seg,volume['spacing'])['host']
    shares = volume['hosts']['probs'] if gate and mode=='3d' else None
    record['r'] = _binding_metrics(truth,controlled,b0,record['u']['matches'],prediction.lesions,predicted_b0,reasons,shares)
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
        query_capacity=payload['model_config']['n_events']
        if not isinstance(identity,dict) or identity.get('contract_sha256') != frozen['contract_sha256']: raise ValueError()
        del payload
        # The certificate-bound training runtime identity is preserved. Current
        # evaluation source is committed separately, never substituted for it.
        plan.update(configuration=dict(crops=crops,score_threshold=float(score_threshold),device=str(device),
                    lesion_threshold=.3,entity_threshold=.5,presence_threshold=.5,overlap=.5,
                    planar_min_area_mm2=1.,volume_min_mm3=10.,matching_bbox_iou=.1,query_capacity=query_capacity),
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
        summary['query_capacity'] = query_capacity
        summary['evaluator_source_sha256'] = json_digest(snapshot)
        write_private_json(out/'records_private.json',records)
        write_private_json(out/'anonymous_summary.json',summary)
        members = {name:file_sha256(out/name) for name in ('plan_private.json','records_private.json','anonymous_summary.json')}
        write_private_json(out/'complete.json',dict(version=VERSION,evidence='NOT_EVIDENCE',outputs=members))
        return summary
    except Exception:
        raise ValueError('Development evaluation failed; private evidence preserved') from None


def _gate_rows(rows):
    rows = deepcopy(list(rows))
    if (not rows or len({r.get('role') for r in rows})!=1
            or any(ROLES.get(r.get('role'))!=r.get('split') for r in rows)):
        raise ValueError('Invalid gate evaluation membership')
    identities=[(r['source'],str(r['patient']),r['case'],r['sequence']) for r in rows]
    if len(identities)!=len(set(identities)):
        raise ValueError('Duplicate gate evaluation member')
    return rows


def freeze_gate_observations(rows,modes=('2d','3d'),*,stride_mm=8.):
    """Freeze a role-consistent library selection; the runner separately verifies its contract.

    Header-stride observations are label independent. The extra lesion-centroid
    observations are explicitly label-defined and never called independent data.
    """
    from anatobind.aur.gate_metrics import stride_planes,lesion_centroid_planes
    try:
        rows,modes=_gate_rows(rows),_modes(modes)
        if not np.isfinite(stride_mm) or stride_mm<=0:raise ValueError()
        observations=[]
        for index,row in enumerate(rows):
            shape,affine=_header_grid(row['image'])
            spacing=float(np.linalg.norm(affine[:3,2]))
            area=float(np.linalg.norm(np.cross(affine[:3,0],affine[:3,1])))
            planes={p:{'stride'} for p in stride_planes(shape[0],spacing,stride_mm=stride_mm)}
            targets={}
            if '2d' in modes and row.get('u_supervised',False):
                volume=load_unified_volume(row)
                for instance,plane in lesion_centroid_planes(volume['instance']).items():
                    planes.setdefault(plane,set()).add('lesion_centroid')
                    targets.setdefault(plane,[]).append(int(instance))
            for mode in modes:
                for plane in (sorted(planes) if mode=='2d' else [None]):
                    local=affine.copy()
                    if plane is not None:local[:3,3]+=local[:3,2]*plane
                    kinds=([k for k in ('stride','lesion_centroid') if k in planes[plane]]
                           if mode=='2d' else ['volume'])
                    observations.append(dict(row_index=index,row_sha256=json_digest(row),input_mode=mode,
                        plane_index=plane,plane_kinds=kinds,target_instances=sorted(targets.get(plane,[])),
                        shape=[1,*shape[1:]] if mode=='2d' else list(shape),source_shape=list(shape),
                        affine_ras=local.tolist(),pixel_area_mm2=area,
                        voxel_volume_mm3=None if mode=='2d' else abs(float(np.linalg.det(affine[:3,:3]))),
                        gate_version=GATE_VERSION))
        return dict(version=GATE_VERSION,evidence='NOT_EVIDENCE',selection='gate_v1_stride_and_lesion_centroid',
                    modes=list(modes),rows=rows,observations=observations)
    except Exception:
        raise ValueError('Invalid gate observation selection') from None


def _inplane_relation(instances,seg,spacing,matches,predicted,predicted_b0):
    reference=host_targets(instances,seg,spacing)
    by_id={int(p['instance']):p for p in predicted}
    rows=[]
    for i,truth in enumerate(reference['host']):
        k=i+1
        if not np.any(instances==k):continue
        detected=matches.get(k)
        rows.append(dict(instance=k,truth=int(truth),top_share=float(reference['probs'][i].max()),
            detected_instance=detected,end_to_end=int(by_id[detected]['host']) if detected is not None else None,
            end_to_end_b0=int(predicted_b0[detected-1]) if detected is not None else None))
    return dict(eligible=len(rows),matched=sum(r['detected_instance'] is not None for r in rows),
                hits=sum(r['end_to_end']==r['truth'] for r in rows),instances=rows,evidence='NOT_EVIDENCE')


def _verify_gate_role(rows,parent_sha,role_contract,qualified_source_root):
    if role_contract is None:raise ValueError('A verified role contract is required')
    rows=_gate_rows(rows)
    if rows[0]['role']=='development_validation':
        contract=read_contract(role_contract,source_mode='evaluation',qualified_source_root=qualified_source_root)
        expected,actual_parent=contract['validation'],contract['contract_sha256']
    else:
        from anatobind.aur.test_role import read_test_role
        contract=read_test_role(role_contract,qualified_source_root=qualified_source_root)
        expected,actual_parent=contract['rows'],contract['parent_contract_sha256']
    if actual_parent!=parent_sha or rows!=expected:
        raise ValueError('Gate rows or parent differ from qualified role contract')
    return contract['contract_sha256']


def run_gate_evaluation(rows,parent_contract_sha256,export_path,out_dir,*,device='cpu',
                        modes=('2d','3d'),crops=None,score_thresholds,stride_mm=8.,
                        role_contract=None,qualified_source_root=None,gate_registration=None,
                        expected_registration_sha256=None):
    """Evaluate exactly the qualified role under a previously frozen gate registration.

    Supplying rows and an arbitrary parent digest alone is insufficient. A role
    contract and frozen registration are mandatory; pure observation/metric
    functions remain available for synthetic controls. This call never judges
    quality or changes a validation-selected threshold.
    """
    try:
        from anatobind.aur.gate import GATE_V1,validate_registration
        out=validate_private_output(out_dir)
        if out.exists():raise ValueError()
        rows=_gate_rows(rows);modes=_modes(modes)
        thresholds=dict(score_thresholds)
        if (not {r['source'] for r in rows if r.get('u_supervised',False)}<=set(thresholds) or any(
                type(v) not in (int,float) or not np.isfinite(v) or not 0<=v<=1 for v in thresholds.values())):
            raise ValueError()
        reg=deepcopy(gate_registration)
        validate_registration(reg)
        if (not isinstance(reg,dict) or json_digest(reg)!=expected_registration_sha256
                or json_digest(reg.get('gate'))!=json_digest(GATE_V1)
                or reg.get('relation_implementation') not in ('head','lookup')
                or reg.get('frozen_before_evaluation') is not True
                or not isinstance(reg.get('validation_selection_sha256'),str)
                or len(reg['validation_selection_sha256'])!=64
                or any(reg.get('score_thresholds',{}).get(k)!=v for k,v in thresholds.items())
                or stride_mm!=GATE_V1['two_d']['stride_mm']):raise ValueError()
        role_sha=_verify_gate_role(rows,parent_contract_sha256,role_contract,qualified_source_root)
        inputs={str(Path(row[key]).resolve()):file_sha256(row[key]) for row in rows
                for key in ('image','anatomy','lesion') if row.get(key)}
        crops={mode:tuple((crops or DEFAULT_CROPS)[mode]) for mode in modes}
        if any(len(c)!=3 or any(type(n) is not int or n<=0 for n in c)
               or any(n%p for n,p in zip(c,(1,4,4) if mode=='2d' else (2,4,4)))
               or (mode=='2d' and c[0]!=1) for mode,c in crops.items()):raise ValueError()
        snapshot=_evaluator_source();export_sha=file_sha256(export_path)
        payload=torch.load(export_path,map_location='cpu',weights_only=True)
        if rows[0]['role']=='historical_test' and payload.get('stage')!='III':raise ValueError()
        identity=deepcopy(payload.get('data_contract'));del payload
        if not isinstance(identity,dict) or identity.get('contract_sha256')!=parent_contract_sha256:raise ValueError()
        # Validate complete formal weights on CPU before opening any centroid
        # labels. Literal stage/purpose fields alone cannot release test data.
        model,metadata=model_from_unified_export(export_path,device='cpu',data_contract=identity)
        if metadata['pilot_only'] or not metadata['phase_completed']:raise ValueError()
        query_capacity=int(model.cfg['n_events'])
        if rows[0]['role']=='historical_test':
            from anatobind.aur.unified_train import _state_hash
            role_root=validate_private_output(role_contract)
            if role_root.is_file():role_root=role_root.parent
            model_sha=_state_hash(model.state_dict())
            # Claim before the first label-based selection, even if execution
            # later fails. Recovery needs explicit audit, never a second score.
            write_private_json(role_root/'final_evaluation_claims'/f'{model_sha}.json',
                dict(model_sha256=model_sha,export_sha256=export_sha,role_contract_sha256=role_sha,
                     registration_sha256=expected_registration_sha256,output=str(out)))
        plan=freeze_gate_observations(rows,modes,stride_mm=stride_mm)
        plan.update(parent_contract_sha256=parent_contract_sha256,role_contract_sha256=role_sha,
            input_files=inputs,artifact_identity=identity,export_sha256=export_sha,
            evaluator_source=snapshot,environment=decoder_environment(),registration=reg,
            registration_sha256=expected_registration_sha256,query_capacity=query_capacity,
            configuration=dict(crops=crops,score_thresholds=thresholds,stride_mm=stride_mm,device=str(device),
                registration_sha256=expected_registration_sha256,relation_implementation=reg['relation_implementation'],
                anatomy_only_detection_score_threshold=.5,
                entity_threshold=.5,presence_threshold=.5,lesion_threshold=.3,overlap=.5,matching_bbox_iou=.1))
        write_private_json(out/'plan_private.json',plan)
        model=model.to(device).eval()
        records=[]
        for obs in plan['observations']:
            row=rows[obs['row_index']]
            record=evaluate_observation(model,row,obs,crop=crops[obs['input_mode']],device=device,
                                        score_threshold=thresholds[row['source']] if row.get('u_supervised',False) else .5)
            record['registration_sha256']=expected_registration_sha256
            records.append(record)
        if (_verify_gate_role(rows,parent_contract_sha256,role_contract,qualified_source_root)!=role_sha
                or any(file_sha256(p)!=h for p,h in inputs.items()) or file_sha256(export_path)!=export_sha
                or _evaluator_source()!=snapshot):raise ValueError()
        summary=aggregate_observations(records)
        summary.update(gate_version=GATE_VERSION,role=rows[0]['role'],parent_contract_sha256=parent_contract_sha256,
            role_contract_sha256=role_sha,registration_sha256=expected_registration_sha256,
            export_sha256=export_sha,evaluator_source_sha256=json_digest(snapshot),query_capacity=query_capacity)
        write_private_json(out/'records_private.json',records)
        write_private_json(out/'anonymous_summary.json',summary)
        outputs={n:file_sha256(out/n) for n in ('plan_private.json','records_private.json','anonymous_summary.json')}
        write_private_json(out/'complete.json',dict(version=GATE_VERSION,evidence='NOT_EVIDENCE',outputs=outputs,
            registration_sha256=expected_registration_sha256,role_contract_sha256=role_sha,export_sha256=export_sha))
        return summary
    except Exception:
        raise ValueError('Gate evaluation failed; private evidence preserved') from None
