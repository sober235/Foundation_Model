"""Explicit assumed geometry never weakens the strict DICOM entry."""
import importlib
import numpy as np
import pytest
from test_aur_dicom import make_mr
from anatobind.aur.dicom import DicomInputError, read_dicom_input


def read(files,mode=None):
    return importlib.import_module('anatobind.aur.dicom_pixelgrid').read_pixel_grid_series(files,mode)


def test_single_plane_assumes_only_spacing_and_retains_private_provenance(tmp_path):
    p=make_mr(tmp_path,ImagePositionPatient=None,ImageOrientationPatient=None)
    x=read([p])
    assert x.input_mode=='2d' and x.geometry=='assumed' and x.patient_coordinates=='not_produced'
    assert x.pixel_area_mm2==6 and x.voxel_volume_mm3 is None
    np.testing.assert_allclose(x.affine_ras,np.diag([-3,-2,1,1]))
    np.testing.assert_array_equal(x.image_dhw,[[[0,1,2],[3,4,5]]])
    assert x.verify_sources()
    assert 'synthetic_private' not in repr(x)+repr(x.source_refs)
    assert 'SYNTHETIC_PRIVATE' not in repr(x)+repr(x.source_refs)
    with pytest.raises(DicomInputError):read_dicom_input([p])


def test_stack_filename_order_and_rescale(tmp_path):
    p2=make_mr(tmp_path,2,ImagePositionPatient=None,ImageOrientationPatient=None,
               pixels=np.arange(10,16).reshape(2,3),RescaleSlope=2,RescaleIntercept=-1)
    p1=make_mr(tmp_path,1,ImagePositionPatient=None,ImageOrientationPatient=None)
    x=read([p2,p1])
    assert x.input_mode=='3d' and x.affine_ras[2,2]==9 and x.voxel_volume_mm3==54
    np.testing.assert_array_equal(x.image_dhw[:,0,0],[0,19])
    assert [(m.output_depth_index,m.source_ref_index) for m in x.frame_map]==[(0,1),(1,0)]


def test_present_patient_positions_never_change_assumed_pixel_grid(tmp_path):
    p=make_mr(tmp_path,ImageOrientationPatient=[0,1,0,.6,0,.8],ImagePositionPatient=[10,20,30])
    np.testing.assert_allclose(read([p]).affine_ras,np.diag([-3,-2,1,1]))


@pytest.mark.parametrize('change,code',[
    ({'Rows':4,'pixels':np.zeros((4,3),np.uint16)},'geometry_unsupported'),
    ({'SpacingBetweenSlices':None},'geometry_unsupported'),
    ({'SpacingBetweenSlices':-1},'geometry_unsupported'),
    ({'PixelSpacing':[2,4]},'geometry_unsupported'),
    ({'SeriesInstanceUID':'1.2.826.0.1.3680043.10.987.9'},'mixed_series'),
    ({'PhotometricInterpretation':'MONOCHROME1'},'pixel_format_unsupported'),
    ({'NumberOfFrames':2},'multiframe_unsupported')])
def test_bad_member_rejects_entire_stack(tmp_path,change,code):
    p1=make_mr(tmp_path,1);p2=make_mr(tmp_path,2,**change)
    with pytest.raises(DicomInputError) as err:read([p1,p2])
    assert err.value.code==code and 'synthetic_private' not in str(err.value)


def test_missing_duplicate_and_gapped_filename_indices_are_refused(tmp_path):
    p=make_mr(tmp_path)
    for name in ('noindex.dcm','duplicate1.dcm','gap3.dcm'):
        other=p.with_name(name);other.write_bytes(p.read_bytes())
        with pytest.raises(DicomInputError) as err:read([p,other] if name!='noindex.dcm' else [other])
        assert err.value.code=='filename_order_unsupported'
    with pytest.raises(DicomInputError) as err:read([])
    assert err.value.code=='empty_input'
