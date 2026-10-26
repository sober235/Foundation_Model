"""Private, byte-bound batch inference on explicitly assumed fastMRI pixel grids."""
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import tempfile
from time import monotonic

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from anatobind.aur.data_contract import validate_private_output,write_private_json,file_sha256,json_digest
from anatobind.aur.dicom import DicomInputError
from anatobind.aur.dicom_pixelgrid import read_pixel_grid_series
from anatobind.aur.labels import HOST_SIDE,HOST_NAMES,NO_HOST,ENTITY_LABELS,discrete_labels,entity_to_synthseg
from anatobind.aur.targets import host_targets
from anatobind.aur.unified_infer import infer_input,model_from_unified_export

SEQUENCE_PATTERN=re.compile(r'^file_brain_(AX[A-Z0-9]+)_')
SEQUENCES={'AXT2','AXT1POST','AXFLAIR','AXT1PRE','AXT1'}
VERSION='fastmri_pixelgrid_batch_v1'
SKIP_REASONS={'empty_input','filename_order_unsupported','input_mode_mismatch','input_files_unsupported',
    'source_integrity_failed','pixel_data_unsupported','read_failed','dependency_unavailable',
    'geometry_unsupported','mixed_series','mixed_study','mixed_patient','multidimensional_unsupported',
    'duplicate_source','input_unsupported','sop_class_unsupported','multiframe_unsupported',
    'pixel_format_unsupported','identifiers_unsupported','intensity_transform_unsupported'}


def series_dirs(root):
    return sorted((p for p in Path(root).iterdir() if p.is_dir() and SEQUENCE_PATTERN.match(p.name)),key=lambda p:p.name)


def sequence_of(series_dir):
    match=SEQUENCE_PATTERN.match(Path(series_dir).name)
    return match.group(1) if match and match.group(1) in SEQUENCES else None


def _grid(image,affine):
    image,affine=np.asarray(image),np.asarray(affine,dtype=np.float64)
    if (image.ndim!=3 or min(image.shape)<=0 or not np.isfinite(image).all()
            or affine.shape!=(4,4) or not np.isfinite(affine).all()
            or abs(np.linalg.det(affine[:3,:3]))<=0):raise ValueError('invalid pixel grid')
    return image,affine


def resample_to_1mm(image_dhw,affine_ras,mode):
    image,affine=_grid(image_dhw,affine_ras)
    if mode not in ('2d','3d') or (mode=='2d' and image.shape[0]!=1):raise ValueError('invalid input mode')
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    target=affine.copy();target[:3,:3]/=spacing
    shape=np.maximum(1,np.floor(np.asarray(image.shape[::-1])*spacing+.5).astype(int))
    if mode=='2d':shape[2]=1
    source=nib.Nifti1Image(image.astype(np.float32).transpose(2,1,0),affine)
    mapped=resample_from_to(source,(tuple(shape),target),order=1,mode='nearest')
    return np.asarray(mapped.dataobj,dtype=np.float32).transpose(2,1,0),target


def map_back(labels_1mm_dhw,affine_1mm,native_shape_dhw,native_affine):
    labels,affine=_grid(labels_1mm_dhw,affine_1mm)
    values=discrete_labels(labels,dtype=np.int32)
    result=resample_from_to(nib.Nifti1Image(values.transpose(2,1,0),affine),
        (tuple(native_shape_dhw[::-1]),np.asarray(native_affine)),order=0,mode='nearest')
    return discrete_labels(np.asarray(result.dataobj),dtype=np.int32).transpose(2,1,0)


def screen_side(host_index):
    if type(host_index) not in (int,np.int32,np.int64) or not 0<=host_index<=NO_HOST:
        raise ValueError('invalid host index')
    return None if host_index==NO_HOST else {'left':'screen_right','right':'screen_left',None:None}[HOST_SIDE[host_index]]


def lesion_record(lesion,mode,slice_index=None):
    host=int(lesion['host']);side=screen_side(host)
    name=HOST_NAMES[host]
    if side is not None:name=name.rsplit('_',1)[0]+'_'+side
    record={key:lesion[key] for key in ('instance','box','host_probs','score')}
    record.update(mode=mode,slice=slice_index,host=host,host_name=name,screen_side=side,
                  geometry='assumed',patient_coordinates='not_produced',evidence='NOT_EVIDENCE')
    key='area_mm2' if mode=='2d' else 'volume_mm3';record[key]=float(lesion[key])
    return record


def _mapped_prediction(prediction,target_affine,native,mode,slice_index=None):
    anatomy=map_back(prediction.anatomy,target_affine,native.image_dhw.shape,native.affine_ras)
    instances=map_back(prediction.instances,target_affine,native.image_dhw.shape,native.affine_ras)
    lesions=[];lost=0
    for lesion in prediction.lesions:
        points=np.argwhere(instances==lesion['instance'])
        if not len(points):lost+=1;continue
        box=np.r_[points.min(0),points.max(0)+1].astype(int).tolist()
        if slice_index is not None:box[0]+=slice_index;box[3]+=slice_index
        size={'area_mm2':len(points)*native.pixel_area_mm2} if mode=='2d' else {'volume_mm3':len(points)*native.voxel_volume_mm3}
        lesions.append(lesion_record({**lesion,**size,'box':box},mode,slice_index))
    return anatomy,instances,lesions,lost


def run_series(model,series_dir,*,modes,crops,score_threshold,device,relation_implementation='head',
               expected_source_hashes=None):
    if relation_implementation not in ('head','lookup'):raise ValueError('invalid deployed relation')
    started=monotonic()
    source=read_pixel_grid_series(sorted(Path(series_dir).glob('*.dcm')))
    if expected_source_hashes is not None:
        actual={str(ref.path):ref.sha256 for ref in source.source_refs}
        expected={str(Path(p).resolve()):digest for p,digest in expected_source_hashes.items()}
        if len(expected_source_hashes)!=len(source.source_refs) or actual!=expected:
            raise ValueError('decoded source differs from frozen input')
    if '3d' in modes and source.input_mode!='3d':raise DicomInputError('input_mode_mismatch')
    arrays={};lesions=[];lost=0
    for mode in modes:
        frames=[source] if mode=='3d' else [replace(source,image_dhw=source.image_dhw[z:z+1],
            input_mode='2d',affine_ras=np.diag([source.affine_ras[0,0],source.affine_ras[1,1],1.,1.]),
            voxel_volume_mm3=None,measurement_unit='mm2',frame_map=(source.frame_map[z],))
            for z in range(source.image_dhw.shape[0])]
        anatomy=[];instances=[]
        for z,frame in enumerate(frames):
            image,affine=resample_to_1mm(frame.image_dhw,frame.affine_ras,mode)
            resampled=replace(frame,image_dhw=image,affine_ras=affine,pixel_area_mm2=1.,
                              voxel_volume_mm3=1. if mode=='3d' else None)
            predicted=infer_input(model,resampled,crops[mode],device,score_threshold=score_threshold)
            if relation_implementation=='lookup':
                looked_up=host_targets(predicted.instances,entity_to_synthseg(predicted.anatomy),(1.,1.,1.))
                predicted.lesions=[{**row,'host':int(looked_up['host'][int(row['instance'])-1]),
                    'host_probs':looked_up['probs'][int(row['instance'])-1].tolist()} for row in predicted.lesions]
            a,u,rows,dropped=_mapped_prediction(predicted,affine,frame,mode,z if mode=='2d' else None)
            for row in rows:row['relation_implementation']=relation_implementation
            anatomy.append(a);instances.append(u);lesions.extend(rows);lost+=dropped
        arrays['anatomy_'+mode]=np.concatenate(anatomy,axis=0)
        arrays['instances_'+mode]=np.concatenate(instances,axis=0)
    source.verify_sources();presence={}
    for mode in modes:
        anatomy=arrays['anatomy_'+mode]
        observations=anatomy[:,None] if mode=='2d' else anatomy[None]
        presence[mode]={'n_observations':len(observations),'labels':list(ENTITY_LABELS),
            'present_counts':[sum(bool(np.any(a==i+1)) for a in observations) for i in range(len(ENTITY_LABELS))]}
    return {'version':VERSION,'sequence':sequence_of(series_dir),'modes':list(modes),
        'relation_implementation':relation_implementation,'score_threshold':score_threshold,
        'native_shape':list(source.image_dhw.shape),'geometry':'assumed','patient_coordinates':'not_produced',
        'native_spacing_dhw':np.linalg.norm(source.affine_ras[:3,:3],axis=0)[::-1].tolist(),
        'anatomy_presence':presence,'wall_seconds':monotonic()-started,
        'instance_identity_2d':'slice_and_instance_pair',
        'screen_convention':'native_array_row0_top_column0_left','slice_order':'filename_index',
        'frame_map':[{'slice':m.output_depth_index,'input_index':m.source_ref_index} for m in source.frame_map],
        'lesions':lesions,'instances_lost_on_native_mapping':lost,'evidence':'NOT_EVIDENCE'},arrays


def _series_files(directory):
    # Preserve directory-entry names as well as bytes; aliases must not collapse
    # into one resolved-path key and hide a changed slice inventory.
    return {str(f.absolute()):file_sha256(f) for f in sorted(Path(directory).glob('*.dcm'))}


def _verify_input(item):
    if not Path(item['series_dir']).is_dir() or _series_files(item['series_dir'])!=item['files']:
        raise ValueError('frozen series input changed')


def _inventory(root,limit):
    directories=series_dirs(root)
    if limit is not None:directories=directories[:limit]
    return [{'series_dir':str(p.resolve()),'files':_series_files(p)} for p in directories]


def _sources():
    from anatobind.aur.unified_launcher import source_snapshot,REPO
    return {**source_snapshot(),'scripts/aur_fastmri_batch.py':file_sha256(REPO/'scripts/aur_fastmri_batch.py')}


def _read(path):return json.loads(Path(path).read_text())


def _verify_series(out,commitment):
    complete=_read(out/'complete.json')
    if complete['input_sha256']!=json_digest(commitment):raise ValueError('series input changed')
    status=complete['status']
    required={'record_private.json','arrays.npz'} if status=='COMPLETE' else set()
    fields=({'status','sequence','input_sha256','pilot_only','anatomy_presence','outputs'} if status=='COMPLETE'
            else {'status','reason','input_sha256','outputs'})
    if status not in ('COMPLETE','SKIPPED') or set(complete)!=fields or set(complete['outputs'])!=required:
        raise ValueError('series completion is incomplete')
    if status=='COMPLETE' and (type(complete['pilot_only']) is not bool or complete['sequence'] not in SEQUENCES|{None}):
        raise ValueError('series completion is invalid')
    if status=='SKIPPED' and complete['reason'] not in SKIP_REASONS:
        raise ValueError('series skip reason is invalid')
    for name,digest in complete['outputs'].items():
        if Path(name).name!=name or file_sha256(out/name)!=digest:raise ValueError('series output changed')
    return complete


def verify_batch_completion(out):
    """Verify the required complete inventory and every committed series output."""
    out=Path(out).resolve();contract=_read(out/'run_contract_private.json');complete=_read(out/'complete.json')
    if contract['version']!=VERSION or len({r['series_dir'] for r in contract['inventory']})!=len(contract['inventory']):
        raise ValueError('batch inventory is invalid')
    threshold,relation,registration,reg_sha=_selected_policy(contract['score_threshold'],
        contract['gate_registration'],contract['registration_sha256'])
    if relation!=contract['relation_implementation']:raise ValueError('batch relation policy differs')
    required={'run_contract_private.json','anonymous_summary.json'}|{
        f'series_{i:05d}/complete.json' for i in range(len(contract['inventory']))}
    if (set(complete)!={'version','contract_sha256','outputs'} or complete['version']!=VERSION
            or complete['contract_sha256']!=file_sha256(out/'run_contract_private.json')
            or set(complete['outputs'])!=required):
        raise ValueError('batch completion is incomplete')
    for name,digest in complete['outputs'].items():
        member=(out/name).resolve()
        if not member.is_relative_to(out) or file_sha256(member)!=digest:raise ValueError('batch output changed')
    for index,item in enumerate(contract['inventory']):
        completed=_verify_series(out/f'series_{index:05d}',item)
        if completed['status']=='COMPLETE' and not completed['pilot_only'] and registration is None:
            raise ValueError('formal batch is missing frozen gate registration')
    return contract


def _selected_policy(score_threshold,registration,expected_sha):
    if registration is None:
        if expected_sha is not None:raise ValueError('missing frozen gate registration')
        return .5 if score_threshold is None else score_threshold,'head',None,None
    from anatobind.aur.gate import validate_registration
    registration=json.loads(json.dumps(registration,allow_nan=False))
    validate_registration(registration)
    if json_digest(registration)!=expected_sha:raise ValueError('frozen gate registration hash differs')
    median=float(np.median(list(registration['score_thresholds'].values())))
    if score_threshold is not None and score_threshold!=median:raise ValueError('registered median threshold differs')
    return median,registration['relation_implementation'],registration,expected_sha


def run_batch(root,export_path,out_dir,*,modes=('2d','3d'),crops=None,score_threshold=None,device='cpu',limit=None,
              gate_registration=None,expected_registration_sha256=None):
    """Resume only the same frozen job; incomplete attempts stay in private staging dirs."""
    try:return _run_batch(root,export_path,out_dir,modes=modes,crops=crops,score_threshold=score_threshold,device=device,limit=limit,
                          gate_registration=gate_registration,expected_registration_sha256=expected_registration_sha256)
    except Exception:raise ValueError('fastMRI batch input, output or integrity check failed') from None


def _run_batch(root,export_path,out_dir,*,modes,crops,score_threshold,device,limit,gate_registration,expected_registration_sha256):
    out=validate_private_output(out_dir);root=Path(root).expanduser().resolve();export_path=Path(export_path).expanduser().resolve()
    score_threshold,relation,registration,registration_sha=_selected_policy(score_threshold,gate_registration,expected_registration_sha256)
    modes=list(modes);crops=crops or {'2d':(1,160,160),'3d':(128,160,160)}
    if (not modes or len(set(modes))!=len(modes) or not set(modes)<={'2d','3d'}
            or not np.isfinite(score_threshold) or not 0<=score_threshold<=1
            or (limit is not None and (type(limit) is not int or limit<1))):raise ValueError()
    crops={m:list(crops[m]) for m in modes}
    if any(len(c)!=3 or any(type(n) is not int or n<=0 for n in c) or (m=='2d' and c[0]!=1) for m,c in crops.items()):raise ValueError()
    inventory=_inventory(root,limit);snapshot=_sources()
    job={'version':VERSION,'root':str(root),'inventory':inventory,'export_path':str(export_path),
         'export_sha256':file_sha256(export_path),'modes':modes,'crops':crops,'score_threshold':score_threshold,
         'gate_registration':registration,'registration_sha256':registration_sha,'relation_implementation':relation,
         'device':str(device),'limit':limit,'source_snapshot':snapshot}
    if out.exists():
        if _read(out/'run_contract_private.json')!=job:raise ValueError('batch registration changed')
        if (out/'complete.json').exists():
            verify_batch_completion(out)
    else:
        write_private_json(out/'run_contract_private.json',job)
    model=None;metadata=None;ok=0;resumed=0;skipped=Counter();by_sequence=Counter();presence={}
    for index,item in enumerate(inventory):
        dest=out/f'series_{index:05d}'
        if dest.exists():
            complete=_verify_series(dest,item)
            if complete['status']=='COMPLETE':resumed+=1
        else:
            temp=Path(tempfile.mkdtemp(prefix=f'.partial_{index:05d}_',dir=out))
            _verify_input(item)
            try:
                # Validate every source member before constructing a model or returning any partial stack.
                read_pixel_grid_series(item['files'])
                if model is None:model,metadata=model_from_unified_export(export_path,device)
                if metadata.get('format_version')==2 and registration is None:
                    raise ValueError('formal inference requires frozen gate registration')
                record,arrays=run_series(model,Path(item['series_dir']),modes=modes,crops=crops,
                    score_threshold=score_threshold,device=device,relation_implementation=relation,
                    expected_source_hashes=item['files'])
                record['series_stem']=Path(item['series_dir']).name
                record.update(relation_implementation=relation,score_threshold=score_threshold,registration_sha256=registration_sha)
                write_private_json(temp/'record_private.json',record)
                with os.fdopen(os.open(temp/'arrays.npz',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),'wb') as stream:
                    np.savez_compressed(stream,**arrays)
                complete={'status':'COMPLETE','sequence':record['sequence'],'input_sha256':json_digest(item),
                          'pilot_only':bool(metadata['pilot_only']),
                          'anatomy_presence':record.get('anatomy_presence',{}),
                          'outputs':{n:file_sha256(temp/n) for n in ('record_private.json','arrays.npz')}}
            except DicomInputError as error:
                complete={'status':'SKIPPED','reason':error.code,'input_sha256':json_digest(item),'outputs':{}}
            # No output may become resumable under an earlier inventory. This is
            # outside the DICOM skip handler: source mutation aborts the attempt.
            _verify_input(item)
            write_private_json(temp/'complete.json',complete)
            _verify_input(item)
            os.rename(temp,dest)
        if complete['status']=='COMPLETE':
            ok+=1;by_sequence[complete['sequence'] or 'unknown']+=1
            for mode,value in complete.get('anatomy_presence',{}).items():
                total=presence.setdefault(mode,{'labels':list(ENTITY_LABELS),'n_observations':0,
                                               'present_counts':[0]*len(ENTITY_LABELS)})
                total['n_observations']+=value['n_observations']
                total['present_counts']=[a+b for a,b in zip(total['present_counts'],value['present_counts'])]
        else:skipped[complete['reason']]+=1
    if (_inventory(root,limit)!=inventory or file_sha256(export_path)!=job['export_sha256'] or _sources()!=snapshot):
        raise ValueError('batch source changed during inference')
    # Resuming needs no model deserialization when all series already committed.
    pilots=[_read(out/f'series_{i:05d}/complete.json')['pilot_only'] for i in range(len(inventory))
            if _read(out/f'series_{i:05d}/complete.json')['status']=='COMPLETE']
    summary={'version':VERSION,'n_series':len(inventory),'n_ok':ok,'skipped':dict(skipped),
        'by_sequence':dict(by_sequence),'export_sha256':job['export_sha256'],'score_threshold':score_threshold,
        'relation_implementation':relation,'registration_sha256':registration_sha,
        'threshold_selection':'median_registered_disease_thresholds' if registration else 'pilot_threshold_not_validation_selected',
        'modes':modes,'geometry':'assumed','patient_coordinates':'not_produced',
        'pilot_only':any(pilots) if pilots else None,'anatomy_presence':presence,'evidence':'NOT_EVIDENCE'}
    if not (out/'anonymous_summary.json').exists():write_private_json(out/'anonymous_summary.json',summary)
    elif _read(out/'anonymous_summary.json')!=summary:raise ValueError('batch summary changed')
    if not (out/'complete.json').exists():
        members=['run_contract_private.json','anonymous_summary.json']+[f'series_{i:05d}/complete.json' for i in range(len(inventory))]
        write_private_json(out/'complete.json',{'version':VERSION,
            'contract_sha256':file_sha256(out/'run_contract_private.json'),
            'outputs':{n:file_sha256(out/n) for n in members}})
    return {**summary,'resumed':resumed}
