"""Explicit assumed pixel geometry for the approved fastMRI RSS-derived series.

This entry does not recover patient position/orientation and never changes the
strict DICOM reader. Coordinates describe an assumed display grid only.
"""
from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np

from anatobind.aur.dicom import (CLASSIC_MR, DicomInputError, SourceRef, FrameMapping,
    _Header, _paths, _read_header, _read_pixels, _hash_source, _private_intake,
    _vector, _dimension_value, verify_sources)

ASSUMED_IOP = np.array([1.,0.,0.,0.,1.,0.])


@dataclass(frozen=True)
class PixelGridInput:
    image_dhw: np.ndarray = field(repr=False)
    affine_ras: np.ndarray = field(repr=False)
    input_mode: str
    source_refs: tuple = field(repr=False)
    frame_map: tuple
    measurement_unit: str
    voxel_volume_mm3: float | None
    pixel_area_mm2: float
    geometry: str = 'assumed'
    slice_order: str = 'filename_index'
    patient_coordinates: str = 'not_produced'

    def verify_sources(self):
        return verify_sources(self.source_refs)


def _header(path):
    ds=_read_header(path)
    if str(getattr(ds,'SOPClassUID',''))!=CLASSIC_MR or getattr(ds,'Modality',None)!='MR':
        raise DicomInputError('sop_class_unsupported')
    if int(getattr(ds,'NumberOfFrames',1))!=1:
        raise DicomInputError('multiframe_unsupported')
    if str(getattr(ds,'PhotometricInterpretation',''))!='MONOCHROME2' or int(getattr(ds,'SamplesPerPixel',0))!=1:
        raise DicomInputError('pixel_format_unsupported')
    if not str(getattr(ds,'SeriesInstanceUID','')) or not str(getattr(ds,'SOPInstanceUID','')):
        raise DicomInputError('identifiers_unsupported')
    if any(key in ds for key in ('SharedFunctionalGroupsSequence','PerFrameFunctionalGroupsSequence',
                                'DimensionIndexSequence','MRDiffusionSequence')):
        raise DicomInputError('multidimensional_unsupported')
    try:
        rows,columns=int(ds.Rows),int(ds.Columns)
        if min(rows,columns)<=0:raise ValueError
    except Exception:raise DicomInputError('geometry_unsupported') from None
    spacing=_vector(ds,'PixelSpacing',2,positive=True)
    if ('ModalityLUTSequence' in ds or 'RealWorldValueMappingSequence' in ds
            or ('RescaleSlope' in ds)!=('RescaleIntercept' in ds)):
        raise DicomInputError('intensity_transform_unsupported')
    try:
        slope,intercept=float(getattr(ds,'RescaleSlope',1)),float(getattr(ds,'RescaleIntercept',0))
        if slope==0 or not np.isfinite([slope,intercept]).all():raise ValueError
    except Exception:raise DicomInputError('intensity_transform_unsupported') from None
    return _Header(ds,path,np.zeros(3),ASSUMED_IOP.reshape(2,3),spacing,rows,columns,slope,intercept)


@_private_intake
def read_pixel_grid_series(files,input_mode=None):
    paths=_paths(files)
    if not paths:raise DicomInputError('empty_input')
    indices=[]
    for path in paths:
        match=re.search(r'(\d+)\s*$',path.stem)
        if match is None:raise DicomInputError('filename_order_unsupported')
        indices.append(int(match.group(1)))
    if len(set(indices))!=len(indices) or any(b-a!=1 for a,b in zip(sorted(indices),sorted(indices)[1:])):
        raise DicomInputError('filename_order_unsupported')
    mode='2d' if len(paths)==1 else '3d'
    if input_mode is not None and input_mode!=mode:raise DicomInputError('input_mode_mismatch')
    hashes=[_hash_source(p) for p in paths]
    headers=[_header(p) for p in paths];first=headers[0]
    for key,code in (('SeriesInstanceUID','mixed_series'),('StudyInstanceUID','mixed_study'),('PatientID','mixed_patient')):
        if len({_dimension_value(h.dataset,key) for h in headers})!=1:raise DicomInputError(code)
    for key in ('EchoNumbers','EchoTime','TemporalPositionIdentifier','NumberOfTemporalPositions',
                'AcquisitionNumber','DiffusionBValue','ComplexImageComponent'):
        if len({_dimension_value(h.dataset,key) for h in headers})!=1:
            raise DicomInputError('multidimensional_unsupported')
    if len({str(h.dataset.SOPInstanceUID) for h in headers})!=len(headers):
        raise DicomInputError('duplicate_source')
    depth_spacing=1.
    for h in headers:
        if ((h.rows,h.columns)!=(first.rows,first.columns)
                or not np.allclose(h.spacing,first.spacing,atol=1e-4,rtol=0)):
            raise DicomInputError('geometry_unsupported')
        if mode=='3d':
            try:value=float(h.dataset.SpacingBetweenSlices)
            except Exception:raise DicomInputError('geometry_unsupported') from None
            if not np.isfinite(value) or value<=0:raise DicomInputError('geometry_unsupported')
            if h is first:depth_spacing=value
            elif not np.isclose(value,depth_spacing,atol=1e-4,rtol=0):
                raise DicomInputError('geometry_unsupported')
    order=np.argsort(indices,kind='stable')
    pixels=np.stack([_read_pixels(headers[int(i)]) for i in order])
    refs=tuple(SourceRef(p,d,str(h.dataset.SOPInstanceUID)) for p,d,h in zip(paths,hashes,headers))
    verify_sources(refs)
    affine=np.diag([-first.spacing[1],-first.spacing[0],depth_spacing,1.])
    area=float(np.prod(first.spacing))
    return PixelGridInput(pixels,affine,mode,refs,
        tuple(FrameMapping(z,int(i)) for z,i in enumerate(order)),
        'mm2' if mode=='2d' else 'mm3',None if mode=='2d' else area*depth_spacing,area)
