import importlib

import nibabel as nib
import numpy as np
import pytest
import torch

from test_aur_unified_run import config


def rows(tmp_path):
    out=[]
    for i,source in enumerate(('pdgm','bmsr')):
        image=tmp_path/f'image_{i}.nii.gz'
        array=np.random.default_rng(i).uniform(1,10,(20,20,10)).astype(np.float32)
        obj=nib.Nifti1Image(array,np.eye(4));obj.header.set_xyzt_units('mm');nib.save(obj,image)
        out.append(dict(source=source,patient=f'p{i}',case=f'c{i}',sequence='T1',image=str(image),
                        split='train',role='train_ssl',patient_group=f'{source}:p{i}'))
    return out


def test_ssl_factories_are_image_only_real_depth1_and_stateless(tmp_path,monkeypatch):
    import anatobind.aur.unified_data as supervised
    monkeypatch.setattr(supervised,'load_unified_volume',lambda *a:pytest.fail('SSL read supervision'))
    cls=importlib.import_module('anatobind.aur.unified_views').QualifiedViews
    f=cls(rows(tmp_path),'I',config(batch_size=2,accumulation_steps=1),augment=False)
    for mode in ('2d','3d'):
        a,b=f(mode,4),f(mode,4)
        assert a['view1'].shape==(2,1,*config()['crops'][mode])
        assert set(a['source_labels'])=={'pdgm','bmsr'}
        assert len(set(a['patient'].tolist()))==2
        assert torch.equal(a['view1'],b['view1'])
        assert not any(k in a for k in ('entity','instance','host','anatomy','lesion'))


def test_validation_rows_never_enter_training_factory(tmp_path):
    cls=importlib.import_module('anatobind.aur.unified_views').QualifiedViews
    r=rows(tmp_path)
    r[0].update(role='ssl_validation',split='val')
    with pytest.raises(ValueError,match='role|partition'):
        cls(r,'I',config())
    valid=[{**x,'role':'ssl_validation','split':'val'} for x in r]
    f=cls(valid,'I',config(),augment=False,validation=True)
    assert f('2d',0)['input_mode']=='2d'
