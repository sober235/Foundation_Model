"""Out-of-fold S7 references on the qualified AUR grid, NOT_EVIDENCE.

Native S7 components retain their original size floor and mean-probability
threshold. Nearest-neighbour transport moves the selected binary mask onto the
model grid; transported probabilities are diagnostic and never select again.
"""
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from anatobind.aur import data_contract as dc
from anatobind.aur.dataset import canonical
from anatobind.aur.eval import REFERENCE, instance_rows, ignored_rows, scan, sensitivity_at_threshold
from anatobind.aur.labels import discrete_labels
from anatobind.aur.unified_data import load_unified_volume
from anatobind.eval.lesion_boxes import load_nnunet_probabilities
from anatobind.infer.brain_disease import detections
from anatobind.nnunet.brain_disease import DISEASES, fold_dir

VERSION = 'aur_reference_1mm_v1'
NNUNET_ROOT = dc.DATA_ROOT/'derived/nnunet'
CLASSIFICATION = 'native_component_mean_threshold_then_nearest_binary'


def _native(prediction_path, probability_path):
    image=nib.load(str(prediction_path));dc.inspect_geometry(image)
    labels=discrete_labels(np.asarray(image.dataobj))
    if not set(map(int,np.unique(labels)))<={0,1}:raise ValueError()
    probabilities=load_nnunet_probabilities(probability_path,labels)
    if (probabilities.shape!=(2,*image.shape) or not np.isfinite(probabilities).all()
            or (probabilities<0).any() or (probabilities>1).any()
            or not np.allclose(probabilities.sum(0),1.,atol=1e-3,rtol=0)):
        raise ValueError()
    return image,probabilities,labels


def _target(image_path):
    image=canonical(nib.load(str(image_path)));geometry=dc.inspect_geometry(image)
    if not np.allclose(geometry['spacing_mm'],1.,atol=1e-3,rtol=0):raise ValueError()
    return image


def _transport(values, affine, target):
    image=nib.Nifti1Image(values,affine)
    result=resample_from_to(image,(target.shape,target.affine),order=0,mode='constant',cval=0.)
    return np.ascontiguousarray(np.asarray(result.dataobj).transpose(2,1,0))


def reference_probabilities_on_grid(prediction_path, probability_path, image_path):
    """Return nearest-neighbour native probabilities as (C,z,y,x), never argmax."""
    try:
        native,probabilities,_=_native(prediction_path,probability_path)
        target=_target(image_path)
        return np.stack([_transport(channel,native.affine,target) for channel in probabilities]).astype(np.float32)
    except Exception:
        raise ValueError('Invalid same-grid reference probabilities or geometry') from None


def reference_scan_1mm(row, volume, prediction_path, probability_path, disease):
    try:
        source=row['source'];rule=REFERENCE[source]
        if rule['disease']!=disease:raise ValueError()
        target=_target(row['image']);native,probabilities,labels=_native(prediction_path,probability_path)
        native_rows,native_components=detections(labels,probabilities,abs(float(np.linalg.det(native.affine[:3,:3]))),'lesion')
        selected=[r for r in native_rows if r['score']>=rule['thr']]
        selected_ids=[r['component'] for r in selected]
        native_mask=np.isin(native_components,selected_ids).astype(np.uint8)
        predicted=_transport(native_mask,native.affine,target).astype(bool)
        on_grid=np.stack([_transport(channel,native.affine,target) for channel in probabilities]).astype(np.float32)
        instance=np.asarray(volume['instance']);small=np.asarray(volume['small'])
        voxel=float(volume['voxel_mm3'])
        if (instance.shape!=predicted.shape or small.shape!=predicted.shape or small.dtype.kind!='b'
                or not np.isfinite(instance).all() or (instance<0).any()
                or not np.array_equal(instance,discrete_labels(instance,np.int32))
                or not np.isfinite(voxel) or voxel<=0
                or not np.isclose(voxel,abs(np.linalg.det(target.affine[:3,:3])),atol=1e-6,rtol=1e-5)):
            raise ValueError()
        found,_=detections(predicted.astype(np.uint8),on_grid,voxel,'lesion')
        found=[{**r,'diagnostic_probability_mean':r['score'],'score':1.0} for r in found]
        result=scan(dc.json_digest(row)[:12],instance_rows(instance,voxel)+ignored_rows(small,voxel),found)
        result.update(row_sha256=dc.json_digest(row),classification=CLASSIFICATION,
                      native_selected_components=selected)
        return result
    except Exception:
        raise ValueError('Invalid same-grid reference scan') from None


def reference_rows(source, rows):
    if source not in REFERENCE:raise ValueError('Unsupported reference source')
    return [row for row in rows if row.get('source')==source
            and row.get('sequence')==REFERENCE[source]['sequence'] and row.get('u_supervised') is True]


def _source_plan(source,rows,nnunet_root):
    rule=REFERENCE[source];root=Path(nnunet_root)
    split_path=root/'preprocessed'/DISEASES[rule['disease']]['name']/'splits_final.json'
    split_sha=dc.file_sha256(split_path);splits=dc.read_json(split_path);fold_of={}
    if not isinstance(splits,list) or not splits:raise ValueError()
    for fold,partition in enumerate(splits):
        for case in partition['val']:
            if not isinstance(case,str) or not case or case in fold_of:raise ValueError()
            fold_of[case]=fold
    selected=reference_rows(source,rows);cases=[row['case'] for row in selected]
    if len(set(cases))!=len(cases):raise ValueError()
    included=[];excluded=[];inputs={str(split_path.resolve())}
    for row in sorted(selected,key=dc.json_digest):
        case=row['case']
        if not isinstance(case,str) or not case or Path(case).name!=case:raise ValueError()
        if case not in fold_of:
            excluded.append(dc.json_digest(row));continue
        folder=Path(fold_dir(root/'results',rule['disease'],fold_of[case]))/'validation'
        prediction=folder/(case+'.nii.gz');probabilities=folder/(case+'.npz')
        inputs.update((str(prediction.resolve()),str(probabilities.resolve())))
        inputs.update(str(Path(row[k]).resolve()) for k in ('image','anatomy','lesion') if row.get(k))
        included.append((row,prediction,probabilities))
    return {'included':included,'excluded':excluded,'inputs':inputs,'split_path':str(split_path.resolve()),'split_sha256':split_sha}


def _source_result(source,plan,load_volume):
    rule=REFERENCE[source];records=[];included=[]
    for row,prediction,probabilities in plan['included']:
        records.append(reference_scan_1mm(row,load_volume(row),prediction,probabilities,rule['disease']))
        included.append(dc.json_digest(row))
    at=sensitivity_at_threshold(records,rule['thr'])
    if not records:at.update(sensitivity=None,fp_per_scan=None)
    elif not at['n_gt']:at['sensitivity']=None
    summary={**at,'source':source,'disease':rule['disease'],'grid':'aur_1mm_canonical',
             'excluded_cases':len(plan['excluded']),'included_row_sha256':included,'excluded_row_sha256':plan['excluded'],
             'classification':CLASSIFICATION,'evidence':'NOT_EVIDENCE'}
    return summary,records


def reference_summary_1mm(source, rows, *, nnunet_root=NNUNET_ROOT, load_volume=load_unified_volume):
    try:
        plan=_source_plan(source,rows,nnunet_root);before=dc.hash_files(plan['inputs'],workers=2)
        if before[plan['split_path']]!=plan['split_sha256']:raise ValueError()
        summary,_=_source_result(source,plan,load_volume)
        if dc.hash_files(plan['inputs'],workers=2)!=before:raise ValueError('Reference inputs changed')
        return summary
    except Exception:raise ValueError('Unable to qualify same-grid reference source') from None


def source_snapshot():
    paths=sorted((dc.REPO/'anatobind').rglob('*.py'))
    paths.append(dc.REPO/'scripts/aur_reference_1mm.py')
    return {str(path.relative_to(dc.REPO)):dc.file_sha256(path) for path in paths}


def run_reference(contract, out, *, qualified_source_root=None, test_role=None,
                  sources=('pdgm','bmsr','isles'), nnunet_root=NNUNET_ROOT):
    """Write a private reference and a complete source/input/output commitment."""
    out=dc.private_output(out)
    if out.exists():raise ValueError('Reference output already exists')
    if not sources or len(set(sources))!=len(sources) or not set(sources)<=set(REFERENCE):
        raise ValueError('Invalid reference sources')
    root=dc.source_path(nnunet_root)
    if test_role is None:
        frozen=dc.read_contract(contract,source_mode='evaluation',qualified_source_root=qualified_source_root)
        rows=frozen['validation'];contract_sha=frozen['contract_sha256'];parent_sha=None
    else:
        from anatobind.aur.test_role import read_test_role
        frozen=read_test_role(test_role,qualified_source_root=qualified_source_root)
        rows=frozen['rows'];contract_sha=frozen['contract_sha256'];parent_sha=frozen['parent_contract_sha256']
        parent=Path(contract).expanduser().resolve();parent=parent.parent if parent.is_file() else parent
        if dc.file_sha256(parent/'contract.json')!=parent_sha:raise ValueError('Reference parent contract differs')
    role_path=Path(test_role if test_role is not None else contract).expanduser().resolve()
    role_path=role_path.parent if role_path.is_file() else role_path
    inputs={str(role_path/name):dc.file_sha256(role_path/name) for name in ('contract.json','complete.json')}
    if inputs[str(role_path/'contract.json')]!=contract_sha:raise ValueError('Verified reference contract changed')
    role_hashes={}
    for row in rows:
        for field in ('image','anatomy','lesion'):
            if not row.get(field):continue
            path=str(Path(row[field]).expanduser().resolve())
            expected=row['source_hashes'][field]
            if path!=row[field] or (path in role_hashes and role_hashes[path]!=expected):
                raise ValueError('Invalid verified reference role link')
            role_hashes[path]=expected
    snapshot=source_snapshot();records=[];summaries={};source_files=set();plans={}
    for source in sources:
        plans[source]=_source_plan(source,rows,root);source_files.update(plans[source]['inputs'])
    for path in source_files:
        if path not in role_hashes:dc.source_path(path)
    hashes=dc.hash_files(source_files,workers=2)
    if (any(hashes[path]!=digest for path,digest in role_hashes.items() if path in hashes)
            or any(hashes[plan['split_path']]!=plan['split_sha256'] for plan in plans.values())):
        raise ValueError('Verified reference role source changed')
    for source in sources:
        summary,scans=_source_result(source,plans[source],load_unified_volume)
        summaries[source]=summary;records.append({'source':source,'summary':summary,'scans':scans})
    if dc.hash_files(source_files,workers=2)!=hashes:raise ValueError('Reference inputs changed during scoring')
    if source_snapshot()!=snapshot or any(dc.file_sha256(path)!=digest for path,digest in inputs.items()):
        raise ValueError('Reference execution source or input changed')
    private={'version':VERSION,'contract_sha256':contract_sha,'parent_contract_sha256':parent_sha,'records':records}
    anonymous={'version':VERSION,'evidence':'NOT_EVIDENCE','contract_sha256':contract_sha,'sources':summaries}
    dc.write_private_json(out/'reference_private.json',private)
    dc.write_private_json(out/'reference_anonymous.json',anonymous)
    outputs={name:dc.file_sha256(out/name) for name in ('reference_private.json','reference_anonymous.json')}
    commitment={'version':VERSION,'data_contract_sha256':contract_sha,'parent_contract_sha256':parent_sha,
                'input_files':inputs,'source_files':hashes,'source_snapshot':snapshot,
                'environment':dc.decoder_environment(),'outputs':outputs}
    dc.write_private_json(out/'contract.json',commitment)
    dc.write_private_json(out/'complete.json',{'version':VERSION,'contract_sha256':dc.file_sha256(out/'contract.json'),'outputs':outputs})
    return anonymous
