"""A separate, immutable historical-test role never changes training roles."""
import importlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from test_aur_unified_evaluation import qualified


def api():
    try:return importlib.import_module('anatobind.aur.test_role')
    except ModuleNotFoundError:pytest.fail('The historical-test successor module is missing')


@pytest.fixture
def parent(qualified,monkeypatch):
    from anatobind.aur import data_contract as dc
    home,parent_dir,validation=qualified
    source=home/'synthetic_source';source.mkdir()
    names=('anatobind/aur/test_role.py','anatobind/aur/data_contract.py','anatobind/aur/labels.py',
           'anatobind/aur/resample.py','anatobind/aur/samples.py')
    for name in names:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic execution source')
    monkeypatch.setattr(dc,'REPO',source)
    paths={}
    for name,values,affine in [('image',np.zeros((9,9,9),np.float32),np.eye(4)),
          ('anatomy',np.full((9,9,9),2,np.int16),np.eye(4)),
          ('lesion',np.zeros((9,9,9),np.int16),np.eye(4)),
          ('native_lesion',np.pad(np.ones((3,3,3),np.float32)*.99999994,1),np.diag([2.,2.,2.,1.]))]:
        path=home/(name+'_test.nii');obj=nib.Nifti1Image(values,affine);obj.header.set_xyzt_units('mm');nib.save(obj,path);paths[name]=str(path)
    test=dict(source='isles',patient='synthetic_test_patient',case='synthetic_test_case',sequence='DWI',source_sequence='DWI',
              split='test',a_supervised=True,u_supervised=True,r_supervised=True,u_values=[1],a_ignore_values=[1],
              **{k:paths[k] for k in ('image','anatomy','lesion')})
    train={**validation,'split':'train'}
    current=home/'current.json';native=home/'native.json'
    current.write_text(json.dumps([train,test]));native.write_text(json.dumps([train,{**test,'lesion':paths['native_lesion']}]))
    return home,parent_dir,current,native


def test_successor_repairs_test_labels_without_modifying_parent(parent):
    m=api();from anatobind.aur import data_contract as dc
    home,p,current,native=parent
    before={str(f):dc.file_sha256(f) for f in p.rglob('*') if f.is_file()}
    summary=m.build_historical_test_role(p,home/'role',current=current,native=native)
    rows=json.loads((home/'role/historical_test.json').read_text())
    assert len(rows)==1 and rows[0]['role']=='historical_test' and rows[0]['split']=='test'
    assert rows[0]['lesion'].endswith('test_repair_v1/label_0000.nii.gz')
    assert summary['repair']['isles_repaired_cases']==1 and 'synthetic_' not in json.dumps(summary)
    assert json.loads((home/'role/policy.json').read_text())['model_evaluations_allowed']==1
    assert before=={str(f):dc.file_sha256(f) for f in p.rglob('*') if f.is_file()}


def test_reader_checks_parent_outputs_sources_and_own_environment(parent):
    m=api();home,p,current,native=parent
    m.build_historical_test_role(p,home/'role',current=current,native=native)
    result=m.read_test_role(home/'role')
    from anatobind.aur import data_contract as dc
    assert result['parent_contract_sha256']==dc.file_sha256(p/'contract.json')
    assert result['contract_sha256']==dc.file_sha256(home/'role/contract.json')
    member=home/'role/historical_test.json';member.write_text(member.read_text().replace('"split": "test"','"split": "val"'))
    with pytest.raises(ValueError,match='Historical test role verification failed'):m.read_test_role(home/'role')


def test_parent_patient_and_native_split_conflicts_refuse_before_output(parent):
    m=api();home,p,current,native=parent
    rows=json.loads(current.read_text());rows[1]['patient']=rows[0]['patient'];current.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='intersect'):m.build_historical_test_role(p,home/'refused',current=current,native=native)
    assert not (home/'refused').exists()


def test_source_label_change_is_detected_on_read(parent):
    m=api();home,p,current,native=parent
    m.build_historical_test_role(p,home/'role',current=current,native=native)
    path=Path(json.loads(native.read_text())[1]['lesion']);path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Historical test role verification failed'):m.read_test_role(home/'role')


def test_empty_native_unknown_policy_is_consistent_across_cached_sequences(parent):
    m=api();home,p,current,native=parent
    current_rows=json.loads(current.read_text());native_rows=json.loads(native.read_text())
    path=Path(native_rows[1]['lesion']);obj=nib.load(path)
    zero=nib.Nifti1Image(np.zeros(obj.shape,np.float32),obj.affine);zero.header.set_xyzt_units('mm');nib.save(zero,path)
    for rows in (current_rows,native_rows):
        rows.append({**rows[1],'sequence':'ADC','source_sequence':'ADC','u_supervised':False,'u_values':[]})
    current.write_text(json.dumps(current_rows));native.write_text(json.dumps(native_rows))
    m.build_historical_test_role(p,home/'role',current=current,native=native)
    rows=json.loads((home/'role/historical_test.json').read_text())
    assert all(r['u_supervised'] is False and r['r_supervised'] is False for r in rows)
    assert all(r['qualification_reason']['u']=='source_empty_unknown' for r in rows)
    assert all(r['label_provenance']['negative_truth'] is False for r in rows)


def test_non_isles_native_empty_is_unknown_not_a_supervised_negative(parent):
    m=api();home,p,current,native=parent
    current_rows=json.loads(current.read_text());native_rows=json.loads(native.read_text())
    for rows in (current_rows,native_rows):rows[1].update(source='pdgm',sequence='T1c',source_sequence='T1c',u_values=[1,4],a_ignore_values=[1,2,4])
    native_label=Path(native_rows[1]['lesion']);obj=nib.load(native_label)
    empty=nib.Nifti1Image(np.zeros(obj.shape,np.int16),obj.affine);empty.header.set_xyzt_units('mm');nib.save(empty,native_label)
    current.write_text(json.dumps(current_rows));native.write_text(json.dumps(native_rows))
    m.build_historical_test_role(p,home/'role',current=current,native=native)
    row=json.loads((home/'role/historical_test.json').read_text())[0]
    assert row['u_supervised'] is False and row['r_supervised'] is False
    assert row['qualification_reason']['u']=='source_empty_unknown'


def test_changes_during_test_array_qualification_are_rejected(parent,monkeypatch):
    m=api();from anatobind.aur import data_contract as dc
    home,p,current,native=parent;target=Path(json.loads(current.read_text())[1]['anatomy'])
    original=dc.inspect_label;changed=[]
    def inspect(path,image=None,*,decode=True):
        values=original(path,image,decode=decode)
        if decode and Path(path)==target and not changed:
            obj=nib.load(target);new=nib.Nifti1Image(np.full(obj.shape,3,np.int16),obj.affine)
            new.header.set_xyzt_units('mm');nib.save(new,target);changed.append(True)
        return values
    monkeypatch.setattr(dc,'inspect_label',inspect)
    with pytest.raises(ValueError,match='changed'):
        m.build_historical_test_role(p,home/'rejected',current=current,native=native)
    assert not (home/'rejected/complete.json').exists()


def test_pdgm_sequence_reference_empty_stays_supervised_without_healthy_claim(parent):
    m=api();home,p,current,native=parent
    current_rows=json.loads(current.read_text());native_rows=json.loads(native.read_text())
    for rows in (current_rows,native_rows):rows[1].update(source='pdgm',sequence='T1c',source_sequence='T1c',u_values=[1,4],a_ignore_values=[1,2,4])
    for rows in (current_rows,native_rows):
        path=Path(rows[1]['lesion']);obj=nib.load(path)
        value=nib.Nifti1Image(np.full(obj.shape,2,np.int16),obj.affine);value.header.set_xyzt_units('mm');nib.save(value,path)
    current.write_text(json.dumps(current_rows));native.write_text(json.dumps(native_rows))
    m.build_historical_test_role(p,home/'role',current=current,native=native)
    row=json.loads((home/'role/historical_test.json').read_text())[0]
    assert row['u_supervised'] is True and row['label_provenance']['lesion_state']=='sequence_empty'
    assert row['label_provenance']['negative_truth'] is False


def test_non_isles_conversion_loss_is_refused_without_rewriting_source(parent):
    m=api();home,p,current,native=parent
    current_rows=json.loads(current.read_text());native_rows=json.loads(native.read_text())
    for rows in (current_rows,native_rows):rows[1].update(source='bmsr',sequence='T1c',source_sequence='T1post')
    current.write_text(json.dumps(current_rows));native.write_text(json.dumps(native_rows))
    from anatobind.aur.data_contract import file_sha256
    before=file_sha256(current_rows[1]['lesion'])
    with pytest.raises(ValueError,match='conversion loss'):
        m.build_historical_test_role(p,home/'rejected',current=current,native=native)
    assert file_sha256(current_rows[1]['lesion'])==before
    assert not (home/'rejected/complete.json').exists()


def test_replay_cli_checks_the_requested_parent_and_redacts_values(tmp_path,monkeypatch,capsys):
    import runpy
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/aur_test_role.py'))
    main=namespace['main']
    parent=tmp_path/'parent';parent.mkdir();(parent/'contract.json').write_text('{}')
    monkeypatch.setitem(main.__globals__,'read_test_role',lambda *a,**k:{
        'rows':[],'contract_sha256':'b'*64,'parent_contract_sha256':'0'*64})
    assert main(['--parent',str(parent),'--out','private-role','--replay-only'])==1
    assert 'private-role' not in capsys.readouterr().err


def test_unknown_source_cannot_be_published_as_a_private_summary_key(parent):
    m=api();home,p,current,native=parent
    for path in (current,native):
        rows=json.loads(path.read_text());rows[1].update(source='synthetic_private_source',lesion=None,u_supervised=False,r_supervised=False)
        path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='source'):
        m.build_historical_test_role(p,home/'rejected',current=current,native=native)
    assert not (home/'rejected').exists()


def test_native_positive_isles_outside_target_grid_cannot_become_negative(parent):
    m=api();home,p,current,native=parent
    from anatobind.aur.data_contract import file_sha256
    path=Path(json.loads(native.read_text())[1]['lesion']);obj=nib.load(path)
    affine=obj.affine.copy();affine[:3,3]+=100.
    shifted=nib.Nifti1Image(np.asarray(obj.dataobj).copy(),affine);shifted.header.set_xyzt_units('mm');nib.save(shifted,path)
    assert np.any(np.asarray(nib.load(path).dataobj)>0)
    label=Path(json.loads(current.read_text())[1]['lesion']);before=file_sha256(label)
    parent_before=file_sha256(p/'contract.json')
    with pytest.raises(ValueError,match='foreground|grid|empty'):
        m.build_historical_test_role(p,home/'rejected',current=current,native=native)
    assert file_sha256(label)==before and file_sha256(p/'contract.json')==parent_before
    assert not (home/'rejected/complete.json').exists()
