"""Official annotation orientation is provenance, never selected from brightness."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from test_aur_fastmri_batch import batch, fake_series


def api():return importlib.import_module('anatobind.aur.fastmri_plus_report')


def test_contrast_audit_cannot_select_a_different_official_convention():
    img=np.zeros((20,20),np.float32);img[2:6,3:9]=100.
    boxes={'synthetic_private':{0:[dict(x=3,y=2,width=6,height=4,label='lesion')]}}
    audit=api().convention_audit([('synthetic_private',0,img)],boxes)
    assert audit['top']>audit['bottom'] and audit['chosen']=='bottom' and audit['n_images']==1
    assert audit['selection_basis']=='official_fastmri_plus_csv_to_rss'
    assert 'synthetic_private' not in json.dumps(audit)


def test_one_to_one_hit_rate_counts_unmatched_detections():
    dets={0:[[0,2,3,1,6,9],[0,15,15,1,18,18],[0,2,3,1,6,9]]}
    boxes={0:[dict(x=3,y=2,width=6,height=4,label='x')]}
    assert api().slice_hit_rate(dets,boxes,20,convention='top')==dict(n_boxes=1,n_hit=1,n_fp=2,n_images=1)
    assert api().box_rows(boxes[0][0],20,'bottom')==(14,18)


def test_unmatched_files_and_slices_are_counted(tmp_path):
    path=tmp_path/'annotations.csv'
    path.write_text('file,slice,study_level,x,y,width,height,label\n'
        'synthetic_private_missing,99,No,1,1,2,2,Edema\n'
        'synthetic_private_present,99,No,1,1,2,2,Edema\n')
    summary=api().report_batch(tmp_path/'absent',path,series_depths={'synthetic_private_present':16})
    assert summary['unmatched_files']==1 and summary['unmatched_slices']==1 and summary['n_boxes']==0
    assert 'synthetic_private' not in json.dumps(summary)


def test_report_uses_committed_series_mapping_and_projects_3d_boxes(tmp_path,monkeypatch):
    mod=api()
    record={'series_stem':'synthetic_private','sequence':'AXFLAIR','modes':['2d','3d'],
        'native_shape':[3,20,20],'geometry':'assumed','patient_coordinates':'not_produced',
        'lesions':[{'mode':'2d','slice':1,'box':[1,14,3,2,18,9]},
                   {'mode':'3d','slice':None,'box':[0,14,3,3,18,9]},
                   {'mode':'2d','slice':0,'box':[0,0,0,1,2,2]}]}
    monkeypatch.setattr(mod,'_batch_records',lambda p:[record])
    csv=tmp_path/'annotations.csv'
    csv.write_text('file,slice,study_level,x,y,width,height,label\nsynthetic_private,1,No,3,2,6,4,Edema\n')
    summary=mod.report_batch(tmp_path/'batch',csv)
    for mode in ('2d','3d'):
        assert summary['by_mode'][mode]['n_boxes']==1 and summary['by_mode'][mode]['n_hit']==1
        assert summary['by_mode'][mode]['n_images']==1 and summary['by_mode'][mode]['n_fp']==0
    assert summary['unannotated_detections_not_scored']==3
    assert summary['by_sequence']['AXFLAIR']['2d']['n_hit']==1
    assert 'synthetic_private' not in json.dumps(summary)


def test_invalid_detection_geometry_is_refused_without_private_echo():
    with pytest.raises(ValueError,match='box') as err:
        api().slice_hit_rate({0:[[0,2,3,1,1,2]]},{},20,convention='bottom')
    assert 'synthetic_private' not in str(err.value)


def test_report_cli_rejects_private_arguments_without_echo(capsys):
    import runpy
    path=Path(__file__).resolve().parents[1]/'scripts/aur_fastmri_plus_report.py'
    main=runpy.run_path(str(path))['main']
    with pytest.raises(SystemExit):main(['--synthetic_private'])
    assert 'synthetic_private' not in capsys.readouterr().err


def test_private_report_binds_csv_sampled_frames_and_committed_batch(batch,monkeypatch,tmp_path):
    fb,root,export,batch_out=batch
    monkeypatch.setattr(fb,'run_series',fake_series)
    fb.run_batch(root,export,batch_out)
    mod=api();monkeypatch.setattr(mod,'validate_private_output',lambda p:Path(p).resolve())
    csv=tmp_path/'annotations.csv'
    csv.write_text('file,slice,study_level,x,y,width,height,label\n'
        'file_brain_AXFLAIR_200_000001,1,No,3,2,6,4,Edema\n')
    out=tmp_path/'private_report'
    report=mod.write_report(batch_out,csv,root,out,audit_limit=1)
    assert report['audit']['chosen']=='bottom' and report['audit']['n_images']==1
    assert report['hit_rate']['by_mode']['2d']['n_boxes']==1
    assert report['hit_rate']['by_mode']['3d']['n_boxes']==1
    assert report['hit_rate']['n_hit']==0
    complete=json.loads((out/'complete.json').read_text())
    assert set(complete['outputs'])=={'convention_audit_anonymous.json','hit_rate_anonymous.json','source_commitments_private.json'}
    assert 'synthetic_private' not in json.dumps(report) and 'file_brain' not in json.dumps(report)
    (batch_out/'series_00000/record_private.json').write_text('{}')
    with pytest.raises(ValueError,match='report'):mod.report_batch(batch_out,csv)


def test_report_refuses_incomplete_batch_commitments(batch,monkeypatch,tmp_path):
    fb,root,export,out=batch;monkeypatch.setattr(fb,'run_series',fake_series)
    fb.run_batch(root,export,out)
    complete=json.loads((out/'complete.json').read_text());complete['outputs']={}
    (out/'complete.json').write_text(json.dumps(complete))
    csv=tmp_path/'empty.csv';csv.write_text('file,slice,study_level,x,y,width,height,label\n')
    with pytest.raises(ValueError,match='report'):api().report_batch(out,csv)


def test_report_rejects_source_change_during_pixel_audit(batch,monkeypatch,tmp_path):
    fb,root,export,out=batch;monkeypatch.setattr(fb,'run_series',fake_series)
    fb.run_batch(root,export,out)
    mod=api();monkeypatch.setattr(mod,'validate_private_output',lambda p:Path(p).resolve())
    assert hasattr(mod,'_sources')
    calls=[]
    monkeypatch.setattr(mod,'_sources',lambda:calls.append(1) or {'code':'first' if len(calls)==1 else 'changed'})
    csv=tmp_path/'empty.csv';csv.write_text('file,slice,study_level,x,y,width,height,label\n')
    with pytest.raises(ValueError,match='report'):mod.write_report(out,csv,root,tmp_path/'report')


def test_zero_area_annotation_rows_are_excluded_and_counted_without_changing_valid_boxes(tmp_path,monkeypatch):
    mod=api();path=tmp_path/'mixed.csv'
    path.write_text('file,slice,study_level,x,y,width,height,label\n'
        'synthetic_private,1,No,3,2,6,4,Edema\n'
        'synthetic_private,2,No,1,1,2,0,Edema\n'
        'synthetic_private,3,No,1,1,0,0,Edema\n')
    before=path.read_bytes()
    rows=mod.boxes_by_file(path)
    assert rows=={'synthetic_private':{1:[{'x':3,'y':2,'width':6,'height':4,'label':'Edema'}]}}
    record={'series_stem':'synthetic_private','sequence':'AXFLAIR','modes':['2d'],
        'native_shape':[4,20,20],'geometry':'assumed','patient_coordinates':'not_produced',
        'lesions':[{'mode':'2d','slice':1,'box':[1,14,3,2,18,9]},
                   {'mode':'2d','slice':2,'box':[2,0,0,3,2,2]}]}
    monkeypatch.setattr(mod,'_batch_records',lambda p:[record])
    summary=mod.report_batch(tmp_path/'batch',path)
    assert summary['n_boxes']==1 and summary['n_hit']==1 and summary['n_fp']==0
    assert summary['n_images']==1 and summary['unannotated_detections_not_scored']==1
    assert summary['rejected_annotation_boxes']==2
    assert summary['rejected_annotation_reasons']=={'zero_width':1,'zero_height':2}
    assert path.read_bytes()==before and 'synthetic_private' not in json.dumps(summary)


@pytest.mark.parametrize('content',[
    'file,slice,x,y,width,height,label\nsynthetic_private,1,1,1,2,2,Edema\n',
    'file,slice,study_level,x,y,width,height,label\nsynthetic_private,1,No,1,1,-2,2,Edema\n',
    'file,slice,study_level,x,y,width,height,label\nsynthetic_private,1,No,-1,1,2,2,Edema\n'])
def test_other_malformed_annotation_inputs_still_fail_privately(tmp_path,content):
    path=tmp_path/'malformed.csv';path.write_text(content)
    with pytest.raises(ValueError,match='annotation') as err:api().boxes_by_file(path)
    assert 'synthetic_private' not in str(err.value)
