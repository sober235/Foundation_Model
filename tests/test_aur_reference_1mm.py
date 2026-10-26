"""Same-grid S7 references preserve the registered native classifier and cohort."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pytest


def api():
    try:return importlib.import_module('anatobind.aur.reference_1mm')
    except ModuleNotFoundError:pytest.fail('The same-grid S7 reference module is missing')


def save(path, values, affine):
    obj=nib.Nifti1Image(values,affine);obj.header.set_xyzt_units('mm');nib.save(obj,path)


@pytest.fixture
def reference(tmp_path,monkeypatch):
    from anatobind.nnunet.brain_disease import DISEASES
    root=tmp_path/'nnunet';folder=root/'validation';folder.mkdir(parents=True)
    split=root/'preprocessed'/DISEASES['infarct']['name'];split.mkdir(parents=True)
    (split/'splits_final.json').write_text(json.dumps([{'train':[],'val':['synthetic_case']}]))
    native=np.zeros((5,5,5),np.float32);native[1:4,1:4,1:4]=.9
    nii=folder/'synthetic_case.nii.gz';npz=folder/'synthetic_case.npz'
    save(nii,(native>=.5).astype(np.int16),np.diag([2.,2.,2.,1.]))
    np.savez(npz,probabilities=np.stack([1-native,native]).transpose(0,3,2,1))
    image=tmp_path/'image.nii';save(image,np.zeros((9,9,9),np.float32),np.eye(4))
    mask=np.asarray(resample_from_to(nib.Nifti1Image((native>=.5).astype(np.int16),np.diag([2.,2.,2.,1.])),
                                    ((9,9,9),np.eye(4)),order=0).dataobj).transpose(2,1,0)
    volume={'instance':mask.astype(np.int32),'small':np.zeros(mask.shape,bool),'voxel_mm3':1.}
    row={'source':'isles','patient':'synthetic_patient','case':'synthetic_case','sequence':'DWI',
         'source_sequence':'DWI','u_supervised':True,'image':str(image)}
    return SimpleNamespace(root=root,folder=folder,nii=nii,npz=npz,image=image,row=row,volume=volume,
                           load_volume=lambda row:volume)


def test_probabilities_use_nearest_native_values_and_image_grid(reference):
    m=api();p=m.reference_probabilities_on_grid(reference.nii,reference.npz,reference.image)
    assert p.shape==(2,9,9,9) and p.dtype==np.float32
    assert set(np.unique(p[1]))=={np.float32(0),np.float32(.9)}


def test_summary_has_exact_overlap_cohort_and_missing_fold_counts(reference,monkeypatch):
    m=api();monkeypatch.setattr(m,'fold_dir',lambda *a:reference.root)
    rows=[reference.row,dict(reference.row,case='synthetic_missing')]
    summary=m.reference_summary_1mm('isles',rows,nnunet_root=reference.root,load_volume=reference.load_volume)
    assert summary['n_scans']==1 and summary['excluded_cases']==1
    assert summary['n_gt']==summary['n_hit']==1 and summary['n_fp']==0
    assert summary['sensitivity']==1 and summary['fp_per_scan']==0
    assert summary['grid']=='aur_1mm_canonical' and summary['thr']==.5
    assert summary['evidence']=='NOT_EVIDENCE'
    from anatobind.aur.data_contract import json_digest
    assert summary['included_row_sha256']==[json_digest(reference.row)]
    assert 'synthetic_' not in json.dumps(summary)


def test_registered_native_component_threshold_is_applied_before_resampling(reference):
    m=api()
    # The .55 slab is argmax-positive but below PDGM's registered .60 cutoff.
    p=np.zeros((5,5,5),np.float32);p[1:4,1:4,1:4]=.55
    save(reference.nii,(p>=.5).astype(np.int16),np.diag([2.,2.,2.,1.]))
    np.savez(reference.npz,probabilities=np.stack([1-p,p]).transpose(0,3,2,1))
    record=m.reference_scan_1mm(dict(reference.row,source='pdgm'),reference.volume,reference.nii,reference.npz,'glioma')
    assert record['dets']==[]
    assert record['classification']=='native_component_mean_threshold_then_nearest_binary'


def test_reference_rows_keep_only_the_gated_source_sequence():
    m=api();rows=[dict(source='pdgm',sequence='FLAIR',u_supervised=True),
        dict(source='pdgm',sequence='T1',u_supervised=True),dict(source='pdgm',sequence='FLAIR',u_supervised=False)]
    assert m.reference_rows('pdgm',rows)==rows[:1]


def test_missing_prediction_for_known_fold_is_error(reference,monkeypatch):
    m=api();monkeypatch.setattr(m,'fold_dir',lambda *a:reference.root)
    reference.npz.unlink()
    with pytest.raises(ValueError,match='reference'):
        m.reference_summary_1mm('isles',[reference.row],nnunet_root=reference.root,load_volume=reference.load_volume)


def test_probability_shape_nonfinite_and_non_mm_geometry_are_rejected(reference):
    m=api();np.savez(reference.npz,probabilities=np.full((2,5,5,5),np.nan,np.float32))
    with pytest.raises(ValueError,match='reference'):
        m.reference_probabilities_on_grid(reference.nii,reference.npz,reference.image)


def test_reference_run_accepts_only_hash_committed_local_role_repairs(reference,monkeypatch,tmp_path):
    m=api();from anatobind.aur import data_contract as dc
    monkeypatch.setattr(m,'fold_dir',lambda *a:reference.root)
    monkeypatch.setattr(m,'load_unified_volume',reference.load_volume)
    role=tmp_path/'role';role.mkdir();(role/'contract.json').write_text('{}');(role/'complete.json').write_text('{}')
    row={**reference.row,'source_hashes':{'image':dc.file_sha256(reference.image)}}
    monkeypatch.setattr(dc,'read_contract',lambda *a,**k:{'validation':[row],'contract_sha256':dc.file_sha256(role/'contract.json')})
    monkeypatch.setattr(dc,'private_output',lambda p:Path(p).resolve())
    def confined(path):
        path=Path(path).resolve()
        if not path.is_relative_to(reference.root):raise ValueError('unapproved source')
        return path
    monkeypatch.setattr(dc,'source_path',confined)
    result=m.run_reference(role,tmp_path/'result',sources=('isles',),nnunet_root=reference.root)
    assert result['sources']['isles']['n_hit']==1
    commitment=json.loads((tmp_path/'result/contract.json').read_text())
    complete=json.loads((tmp_path/'result/complete.json').read_text())
    assert complete['contract_sha256']==dc.file_sha256(tmp_path/'result/contract.json')
    assert commitment['source_files'][str(reference.image)]==row['source_hashes']['image']
    assert complete['outputs']['reference_anonymous.json']==dc.file_sha256(tmp_path/'result/reference_anonymous.json')


def test_native_prediction_requires_binary_classes_even_if_global_agreement_passes(reference):
    m=api();image=nib.load(reference.nii);values=np.asarray(image.dataobj).copy();values[0,0,0]=2
    save(reference.nii,values,image.affine)
    with pytest.raises(ValueError,match='reference'):
        m.reference_probabilities_on_grid(reference.nii,reference.npz,reference.image)


def test_cli_redacts_argument_values_and_does_not_publish_cohort_hashes(monkeypatch,capsys):
    import runpy
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/aur_reference_1mm.py'))
    main=namespace['main']
    with pytest.raises(SystemExit):main(['--private-patient-marker'])
    assert 'private-patient-marker' not in capsys.readouterr().err
    monkeypatch.setitem(main.__globals__,'run_reference',lambda *a,**k:{'sources':{'isles':{
        'source':'isles','n_scans':1,'excluded_cases':0,'sensitivity':1.,'fp_per_scan':0.,
        'included_row_sha256':['private-cohort-digest'],'case':'private-patient-marker'}}})
    assert main(['--contract','private-manifest','--out','private-output'])==0
    printed=capsys.readouterr().out
    assert set(json.loads(printed))=={'source','n_scans','excluded_cases','sensitivity','fp_per_scan'}
    assert 'private-cohort' not in printed and 'private-patient' not in printed


def test_native_component_mean_keeps_low_probability_member(reference):
    m=api();p=np.zeros((5,5,5),np.float32);p[1:4,1:4,1:4]=.9;p[2,2,2]=.51
    save(reference.nii,(p>=.5).astype(np.int16),np.diag([2.,2.,2.,1.]))
    np.savez(reference.npz,probabilities=np.stack([1-p,p]).transpose(0,3,2,1))
    result=m.reference_scan_1mm(dict(reference.row,source='pdgm'),reference.volume,reference.nii,reference.npz,'glioma')
    assert sum(d['n_voxels'] for d in result['dets'])==int(reference.volume['instance'].sum())
    assert result['classification']=='native_component_mean_threshold_then_nearest_binary'
    assert all(d['score']==1. for d in result['dets'])


def test_reference_changes_during_scoring_cannot_be_certified(reference,monkeypatch,tmp_path):
    m=api();from anatobind.aur import data_contract as dc
    monkeypatch.setattr(m,'fold_dir',lambda *a:reference.root)
    monkeypatch.setattr(m,'load_unified_volume',reference.load_volume)
    role=tmp_path/'role';role.mkdir();(role/'contract.json').write_text('{}');(role/'complete.json').write_text('{}')
    row={**reference.row,'source_hashes':{'image':dc.file_sha256(reference.image)}}
    monkeypatch.setattr(dc,'read_contract',lambda *a,**k:{'validation':[row],'contract_sha256':dc.file_sha256(role/'contract.json')})
    monkeypatch.setattr(dc,'private_output',lambda p:Path(p).resolve())
    monkeypatch.setattr(dc,'source_path',lambda p:Path(p).resolve())
    original=m.reference_scan_1mm
    def changed(*args,**kwargs):
        result=original(*args,**kwargs)
        p=np.zeros((5,5,5),np.float32);p[1:4,1:4,1:4]=.95
        np.savez(reference.npz,probabilities=np.stack([1-p,p]).transpose(0,3,2,1))
        return result
    monkeypatch.setattr(m,'reference_scan_1mm',changed)
    with pytest.raises(ValueError,match='changed'):
        m.run_reference(role,tmp_path/'rejected',sources=('isles',),nnunet_root=reference.root)
    assert not (tmp_path/'rejected/complete.json').exists()


def test_native_subfloor_component_is_not_promoted_by_grid_aliasing(reference):
    m=api();p=np.zeros((22,22,22),np.float32);p[10,10,10]=.95
    save(reference.nii,(p>=.5).astype(np.int16),np.diag([2.1,2.1,2.1,1.]))
    np.savez(reference.npz,probabilities=np.stack([1-p,p]).transpose(0,3,2,1))
    save(reference.image,np.zeros((46,46,46),np.float32),np.eye(4))
    volume={'instance':np.zeros((46,46,46),np.int32),'small':np.zeros((46,46,46),bool),'voxel_mm3':1.}
    record=m.reference_scan_1mm(reference.row,volume,reference.nii,reference.npz,'infarct')
    assert record['dets']==[]
