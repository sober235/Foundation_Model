"""Diagnostic fastMRI+ box coverage with the official CSV-to-RSS convention.

Intensity contrast is reported for both conventions but cannot choose policy.
Unannotated slices are not asserted to be negative medical ground truth.
"""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from anatobind.data_engine.fastmri import read_fastmri_plus_rows,convert_box_csv_to_rss,box_iou_2d,TRANSFORM_VERSION
from anatobind.aur.data_contract import file_sha256,validate_private_output,write_private_json
from anatobind.aur.dicom import DicomInputError
from anatobind.aur.dicom_pixelgrid import read_pixel_grid_series
from anatobind.aur.fastmri_batch import series_dirs,sequence_of,_verify_series,verify_batch_completion

CONVENTION='bottom'


def series_stem(series_dir):return Path(series_dir).name


def boxes_by_file(csv_path):
    try:
        result={}
        for row in read_fastmri_plus_rows(csv_path):
            if row['width']<=0 or row['height']<=0 or min(row['x'],row['y'])<0:raise ValueError
            name=row['file'][:-3] if row['file'].endswith('.h5') else row['file']
            result.setdefault(name,{}).setdefault(row['slice'],[]).append(
                {key:row[key] for key in ('x','y','width','height','label')})
        return result
    except Exception:raise ValueError('invalid annotation table') from None


def box_rows(box,n_rows,convention):
    if convention not in ('top','bottom') or n_rows<=0:raise ValueError('invalid box convention')
    if convention=='bottom':return tuple(convert_box_csv_to_rss(box['x'],box['y'],box['width'],box['height'],n_rows)[:2])
    return box['y'],box['y']+box['height']


def box_contrast(pixels_hw,boxes,convention):
    pixels=np.asarray(pixels_hw,dtype=np.float64)
    if pixels.ndim!=2 or not np.isfinite(pixels).all():raise ValueError('invalid diagnostic pixels')
    mask=np.zeros(pixels.shape,bool)
    for box in boxes:
        y0,y1=box_rows(box,pixels.shape[0],convention)
        mask[max(0,y0):min(pixels.shape[0],y1),max(0,box['x']):min(pixels.shape[1],box['x']+box['width'])]=True
    if not mask.any() or mask.all():return None
    inside,outside=float(pixels[mask].mean()),float(pixels[~mask].mean())
    if inside==0 and outside==0:return None
    return inside/max(abs(outside),1e-12)


def convention_audit(images,boxes_by_file,*,limit=20):
    if type(limit) is not int or limit<1:raise ValueError('invalid audit limit')
    values={'top':[],'bottom':[]};count=0
    for name,index,pixels in images:
        boxes=boxes_by_file.get(name,{}).get(index,[])
        if not boxes:continue
        for convention in values:
            contrast=box_contrast(pixels,boxes,convention)
            if contrast is not None:values[convention].append(contrast)
        count+=1
        if count>=limit:break
    return {**{k:float(np.mean(v)) if v else None for k,v in values.items()},'n_images':count,
        'chosen':CONVENTION,'selection_basis':'official_fastmri_plus_csv_to_rss',
        'transform_version':TRANSFORM_VERSION,'contrast_selects_policy':False,
        'pixel_frame_alignment':'assumed_rss_display_not_patient_frame','evidence':'NOT_EVIDENCE'}


def _detection_box(box):
    values=np.asarray(box,dtype=float)
    if values.shape!=(6,) or not np.isfinite(values).all() or np.any(values[3:]<=values[:3]):
        raise ValueError('invalid detection box')
    return dict(x=float(values[2]),y=float(values[1]),width=float(values[5]-values[2]),height=float(values[4]-values[1]))


def slice_hit_rate(detections_by_slice,boxes_by_slice,n_rows,*,convention,iou=.1):
    if not np.isfinite(iou) or not 0<iou<=1:raise ValueError('invalid box overlap threshold')
    result=dict(n_boxes=0,n_hit=0,n_fp=0,n_images=0)
    for index in sorted(set(detections_by_slice)|set(boxes_by_slice)):
        detections=[_detection_box(b) for b in detections_by_slice.get(index,[])]
        truth=[]
        for box in boxes_by_slice.get(index,[]):
            y0,y1=box_rows(box,n_rows,convention)
            truth.append(dict(x=box['x'],y=y0,width=box['width'],height=y1-y0))
        pairs=sorted(((box_iou_2d(a,b),i,j) for i,a in enumerate(truth) for j,b in enumerate(detections)),reverse=True)
        used_truth=set();used_pred=set()
        for overlap,i,j in pairs:
            if overlap>=iou and i not in used_truth and j not in used_pred:
                used_truth.add(i);used_pred.add(j)
        result['n_boxes']+=len(truth);result['n_hit']+=len(used_truth)
        result['n_fp']+=len(detections)-len(used_pred);result['n_images']+=1
    return result


def _batch_records(batch_dir):
    batch=Path(batch_dir)
    if not batch.exists():return []
    contract=verify_batch_completion(batch)
    records=[]
    for index,item in enumerate(contract['inventory']):
        out=batch/f'series_{index:05d}'
        committed=_verify_series(out,item)
        if committed['status']=='COMPLETE':
            record=json.loads((out/'record_private.json').read_text())
            if (record['series_stem']!=series_stem(item['series_dir']) or record['modes']!=contract['modes']
                    or record['sequence']!=committed['sequence'] or record['geometry']!='assumed'
                    or record['relation_implementation']!=contract['relation_implementation']
                    or record['registration_sha256']!=contract['registration_sha256']
                    or record['score_threshold']!=contract['score_threshold']
                    or record['patient_coordinates']!='not_produced' or len(record['native_shape'])!=3
                    or any(type(n) is not int or n<=0 for n in record['native_shape'])):
                raise ValueError('batch series mapping or geometry differs')
            records.append(record)
    return records


def _sources():
    from anatobind.aur.fastmri_batch import _sources as batch_sources
    root=Path(__file__).resolve().parents[2]
    return {**batch_sources(),'scripts/aur_fastmri_plus_report.py':file_sha256(root/'scripts/aur_fastmri_plus_report.py')}


def _add(target,value):
    for key in ('n_boxes','n_hit','n_fp','n_images'):target[key]=target.get(key,0)+value[key]


def report_batch(batch_dir,csv_path,*,series_depths=None):
    try:return _report_batch(batch_dir,csv_path,series_depths=series_depths)
    except Exception:raise ValueError('fastMRI annotation report integrity or input invalid') from None


def _report_batch(batch_dir,csv_path,*,series_depths):
    annotations=boxes_by_file(csv_path);records=_batch_records(batch_dir)
    depths=dict(series_depths or {})
    for row in records:
        name=row['series_stem'];depth=row['native_shape'][0]
        if name in depths and depths[name]!=depth:raise ValueError('series depth differs')
        depths[name]=depth
    unmatched_files=sum(name not in depths for name in annotations)
    unmatched_slices=sum(index<0 or index>=depths[name] for name,rows in annotations.items() if name in depths for index in rows)
    by_mode={};by_sequence={};unannotated=0;unique_boxes=0;scored_files=set()
    for row in records:
        name=row['series_stem'];depth,height,width=row['native_shape']
        truth={i:b for i,b in annotations.get(name,{}).items() if 0<=i<depth}
        unique_boxes+=sum(map(len,truth.values()))
        if truth:scored_files.add(name)
        sequence=row['sequence'] if row['sequence'] in ('AXT2','AXT1POST','AXFLAIR','AXT1PRE','AXT1') else 'unknown'
        for mode in row['modes']:
            detections=defaultdict(list)
            for lesion in row['lesions']:
                if lesion['mode']!=mode:continue
                box=lesion['box'];_detection_box(box)
                indices=[lesion['slice']] if mode=='2d' else range(max(0,int(box[0])),min(depth,int(box[3])))
                for index in indices:
                    if index in truth:detections[index].append(box)
                    else:unannotated+=1
            counts=slice_hit_rate(detections,truth,height,convention=CONVENTION)
            _add(by_mode.setdefault(mode,{}),counts)
            _add(by_sequence.setdefault(sequence,{}).setdefault(mode,{}),counts)
    aggregate=dict(n_boxes=0,n_hit=0,n_fp=0,n_images=0)
    for counts in by_mode.values():_add(aggregate,counts)
    return {**aggregate,'by_mode':by_mode,'by_sequence':by_sequence,'unique_annotation_boxes':unique_boxes,
        'aggregation_unit':'mode_slice','coverage':'annotation_bearing_slices_only',
        'unmatched_files':unmatched_files,'unmatched_slices':unmatched_slices,
        'unscored_annotation_files':len(set(annotations)&set(depths)-scored_files),
        'unannotated_detections_not_scored':unannotated,'convention':CONVENTION,
        'selection_basis':'official_fastmri_plus_csv_to_rss','transform_version':TRANSFORM_VERSION,
        'pixel_frame_alignment':'assumed_rss_display_not_patient_frame',
        'false_positive_meaning':'unmatched_detection_on_annotation_bearing_slice',
        'geometry':'assumed','patient_coordinates':'not_produced','evidence':'NOT_EVIDENCE'}


def write_report(batch_dir,csv_path,dicom_root,out_dir,*,audit_limit=20):
    """Freeze the exact annotation, batch and sampled-pixel source commitments."""
    try:
        out=validate_private_output(out_dir)
        if out.exists():raise ValueError()
        batch=Path(batch_dir).expanduser().resolve();csv=Path(csv_path).expanduser().resolve()
        snapshot=_sources()
        before={'csv_sha256':file_sha256(csv),'batch_complete_sha256':file_sha256(batch/'complete.json')}
        annotations=boxes_by_file(csv);refs=[];rejected=0
        def images():
            nonlocal rejected
            for folder in series_dirs(dicom_root):
                name=series_stem(folder)
                if name not in annotations:continue
                try:source=read_pixel_grid_series(sorted(folder.glob('*.dcm')))
                except DicomInputError:rejected+=1;continue
                refs.extend(source.source_refs)
                for index in sorted(annotations[name]):
                    if 0<=index<len(source.image_dhw):yield name,index,source.image_dhw[index]
        audit=convention_audit(images(),annotations,limit=audit_limit);audit['rejected_series']=rejected
        summary=report_batch(batch,csv)
        from anatobind.aur.dicom import verify_sources
        verify_sources(refs)
        if before!={'csv_sha256':file_sha256(csv),'batch_complete_sha256':file_sha256(batch/'complete.json')}:raise ValueError()
        _batch_records(batch)
        if _sources()!=snapshot:raise ValueError('report source changed during audit')
        write_private_json(out/'convention_audit_anonymous.json',audit)
        write_private_json(out/'hit_rate_anonymous.json',summary)
        write_private_json(out/'source_commitments_private.json',{**before,
            'sampled_dicoms':{str(ref.path):ref.sha256 for ref in refs},'audit_limit':audit_limit,
            'source_snapshot':snapshot})
        names=('convention_audit_anonymous.json','hit_rate_anonymous.json','source_commitments_private.json')
        write_private_json(out/'complete.json',{'outputs':{n:file_sha256(out/n) for n in names}})
        return {'audit':audit,'hit_rate':summary}
    except Exception:raise ValueError('fastMRI annotation report failed') from None
