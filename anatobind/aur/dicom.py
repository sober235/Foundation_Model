"""Strict, read-only classic MR intake with private provenance.

Arrays use (depth, row, column). ``affine_ras`` maps (column, row, depth)
pixel-center indices to millimetres in RAS. A single plane has a unit normal
as its third affine axis solely to describe its orientation; it has no voxel
volume. Original files remain the authority for metadata and pixel contents.

Only scalar MONOCHROME2 classic MR with an unambiguous regular spatial stack
is supported. Enhanced MR, modality LUTs, and multidimensional acquisitions
are deliberately rejected. No missing coordinates are inferred from thickness.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import warnings
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Iterable

import numpy as np


CLASSIC_MR = "1.2.840.10008.5.1.4.1.1.4"
ENHANCED_MR = frozenset({
    "1.2.840.10008.5.1.4.1.1.4.1", "1.2.840.10008.5.1.4.1.1.4.3",
    "1.2.840.10008.5.1.4.1.1.4.4",
})
_GEOMETRY_ATOL = 1e-4
_POSITION_ATOL_MM = 1e-3
_PRIVATE_PARSE = ContextVar("aur_private_dicom_parse", default=False)
_SCHEMA_FIELDS = {
    "rows": "Rows", "columns": "Columns", "pixel_spacing": "PixelSpacing",
    "image_position": "ImagePositionPatient", "image_orientation": "ImageOrientationPatient",
    "slice_thickness": "SliceThickness", "spacing_between_slices": "SpacingBetweenSlices",
    "frame_of_reference": "FrameOfReferenceUID",
}


class DicomInputError(ValueError):
    """Public failure carrying a fixed code, never a source value or path."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _PrivateDicomLogFilter(logging.Filter):
    def filter(self, record):
        # pydicom can log invalid raw values in addition to issuing warnings.
        # Context-local filtering leaves concurrent unrelated reads unaffected.
        return not _PRIVATE_PARSE.get()


logging.getLogger("pydicom").addFilter(_PrivateDicomLogFilter())


def _private_intake(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _PRIVATE_PARSE.set(True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return function(*args, **kwargs)
        except DicomInputError:
            raise
        except Exception:
            raise DicomInputError("input_unsupported") from None
        finally:
            _PRIVATE_PARSE.reset(token)
    return wrapped


@dataclass(frozen=True)
class SourceRef:
    """Private reference to an unchanged original object; never serialize publicly."""

    path: Path = field(repr=False)
    sha256: str = field(repr=False)
    sop_instance_uid: str = field(repr=False)
    frame_index: int = 0


@dataclass(frozen=True)
class FrameMapping:
    output_depth_index: int
    source_ref_index: int
    source_frame_index: int = 0


@dataclass(frozen=True)
class DicomInput:
    image_dhw: np.ndarray = field(repr=False)
    affine_ras: np.ndarray = field(repr=False)
    input_mode: str
    source_refs: tuple[SourceRef, ...] = field(repr=False)
    frame_map: tuple[FrameMapping, ...]
    measurement_unit: str
    voxel_volume_mm3: float | None
    pixel_area_mm2: float

    def verify_sources(self) -> bool:
        return verify_sources(self.source_refs)


@dataclass(frozen=True)
class _Header:
    dataset: object = field(repr=False)
    path: Path = field(repr=False)
    position: np.ndarray = field(repr=False)
    orientation: np.ndarray = field(repr=False)
    spacing: np.ndarray = field(repr=False)
    rows: int
    columns: int
    slope: float
    intercept: float


def _paths(files: Iterable[os.PathLike | str]) -> tuple[Path, ...]:
    try:
        if isinstance(files, (str, bytes, os.PathLike)):
            raise ValueError
        return tuple(Path(item).resolve() for item in files)
    except Exception:
        raise DicomInputError("input_files_unsupported") from None


def _read_header(path: Path):
    try:
        import pydicom
    except ImportError:
        raise DicomInputError("dependency_unavailable") from None
    try:
        # pydicom warnings may contain private values on malformed input.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        raise DicomInputError("read_failed") from None


def _vector(dataset, key: str, size: int, *, positive=False) -> np.ndarray:
    try:
        values = np.asarray(getattr(dataset, key), dtype=np.float64)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError
        if positive and not (values > 0).all():
            raise ValueError
        return values
    except Exception:
        raise DicomInputError("geometry_unsupported") from None


def _required_identifier(dataset, key: str) -> str:
    value = str(getattr(dataset, key, "")).strip()
    if (not value or len(value) > 64 or re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value) is None
            or any(len(part) > 1 and part.startswith("0") for part in value.split("."))):
        raise DicomInputError("identifiers_unsupported")
    return value


def _validate_header(dataset, path: Path) -> _Header:
    sop_class = str(getattr(dataset, "SOPClassUID", ""))
    if sop_class in ENHANCED_MR:
        raise DicomInputError("enhanced_unsupported")
    if sop_class != CLASSIC_MR or getattr(dataset, "Modality", None) != "MR":
        raise DicomInputError("sop_class_unsupported")
    for identifier in ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID", "FrameOfReferenceUID"):
        _required_identifier(dataset, identifier)
    try:
        frames = int(getattr(dataset, "NumberOfFrames", 1))
    except Exception:
        raise DicomInputError("multiframe_unsupported") from None
    if frames != 1:
        raise DicomInputError("multiframe_unsupported")
    try:
        if (str(getattr(dataset, "PhotometricInterpretation", "")) != "MONOCHROME2"
                or int(getattr(dataset, "SamplesPerPixel", 0)) != 1):
            raise ValueError
    except Exception:
        raise DicomInputError("pixel_format_unsupported") from None
    try:
        rows, columns = int(dataset.Rows), int(dataset.Columns)
        if rows <= 0 or columns <= 0:
            raise ValueError
    except Exception:
        raise DicomInputError("geometry_unsupported") from None
    position = _vector(dataset, "ImagePositionPatient", 3)
    orientation = _vector(dataset, "ImageOrientationPatient", 6).reshape(2, 3)
    spacing = _vector(dataset, "PixelSpacing", 2, positive=True)
    if (not np.allclose(np.linalg.norm(orientation, axis=1), 1, rtol=0, atol=_GEOMETRY_ATOL)
            or abs(float(np.dot(*orientation))) > _GEOMETRY_ATOL):
        raise DicomInputError("geometry_unsupported")
    # Unparsed nested acquisition dimensions cannot be treated as a classic
    # spatial stack, even if every top-level dimension signature agrees.
    if any(key in dataset for key in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence",
                                       "DimensionIndexSequence", "DimensionOrganizationSequence",
                                       "MRDiffusionSequence")):
        raise DicomInputError("multidimensional_unsupported")
    # A declared multi-time acquisition is only unambiguous after selecting an
    # explicit, valid time position. Stack members must still share that value.
    temporal_count = getattr(dataset, "NumberOfTemporalPositions", None)
    temporal_position = getattr(dataset, "TemporalPositionIdentifier", None)
    try:
        if temporal_count is not None:
            count = int(temporal_count)
            if count < 1 or float(temporal_count) != count:
                raise ValueError
            if count > 1 and temporal_position is None:
                raise ValueError
        if temporal_position is not None:
            position_index = int(temporal_position)
            if (position_index < 1 or float(temporal_position) != position_index
                    or (temporal_count is not None and position_index > count)):
                raise ValueError
    except Exception:
        raise DicomInputError("multidimensional_unsupported") from None
    if "ModalityLUTSequence" in dataset or "RealWorldValueMappingSequence" in dataset:
        raise DicomInputError("intensity_transform_unsupported")
    if ("RescaleSlope" in dataset) != ("RescaleIntercept" in dataset):
        raise DicomInputError("intensity_transform_unsupported")
    try:
        slope = float(getattr(dataset, "RescaleSlope", 1))
        intercept = float(getattr(dataset, "RescaleIntercept", 0))
        if not np.isfinite([slope, intercept]).all() or slope == 0:
            raise ValueError
    except Exception:
        raise DicomInputError("intensity_transform_unsupported") from None
    return _Header(dataset, path, position, orientation, spacing, rows, columns, slope, intercept)


def _dimension_value(dataset, key):
    """A private comparison signature, never returned in summaries or exceptions."""
    if key not in dataset:
        return None
    value = getattr(dataset, key)
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return tuple(str(item) for item in value)
    return str(value)


def _stack_geometry(headers: tuple[_Header, ...], input_mode: str | None):
    if not headers:
        raise DicomInputError("empty_input")
    mode = "2d" if len(headers) == 1 else "3d"
    if input_mode is not None and mode != input_mode:
        raise DicomInputError("input_mode_mismatch")
    first = headers[0]
    identifiers = [str(item.dataset.SOPInstanceUID) for item in headers]
    if len(set(identifiers)) != len(identifiers):
        raise DicomInputError("duplicate_source")
    for key, code in (("SeriesInstanceUID", "mixed_series"),
                      ("FrameOfReferenceUID", "mixed_frame_of_reference"),
                      ("StudyInstanceUID", "mixed_study"), ("PatientID", "mixed_patient")):
        if len({_dimension_value(item.dataset, key) for item in headers}) != 1:
            raise DicomInputError(code)
    # These fields separate time, echo, contrast, diffusion and complex components.
    dimension_keys = (
        "TemporalPositionIdentifier", "NumberOfTemporalPositions", "EchoNumbers", "EchoTime",
        "EffectiveEchoTime", "AcquisitionNumber", "TriggerTime", "DiffusionBValue",
        "DiffusionGradientOrientation", "DiffusionDirectionality", "ImageType",
        "ComplexImageComponent", "ScanningSequence", "SequenceVariant", "ScanOptions",
        "InversionTime", "RepetitionTime", "FlipAngle",
    )
    for key in dimension_keys:
        if len({_dimension_value(item.dataset, key) for item in headers}) != 1:
            raise DicomInputError("multidimensional_unsupported")
    for item in headers:
        if ((item.rows, item.columns) != (first.rows, first.columns)
                or not np.allclose(item.orientation, first.orientation, atol=_GEOMETRY_ATOL, rtol=0)
                or not np.allclose(item.spacing, first.spacing, atol=_GEOMETRY_ATOL, rtol=0)):
            raise DicomInputError("geometry_unsupported")
    normal = np.cross(*first.orientation)
    normal /= np.linalg.norm(normal)
    projections = np.array([np.dot(item.position, normal) for item in headers])
    if not np.isfinite(projections).all():
        raise DicomInputError("geometry_unsupported")
    order = np.argsort(projections, kind="stable")
    depth_spacing = 1.0  # Orientation placeholder for an independently measured 2D plane.
    if len(headers) > 1:
        increments = np.diff(projections[order])
        if not np.isfinite(increments).all():
            raise DicomInputError("geometry_unsupported")
        if (increments <= _POSITION_ATOL_MM).any():
            raise DicomInputError("duplicate_position")
        depth_spacing = float(increments[0])
        if not np.allclose(increments, depth_spacing, rtol=1e-4, atol=_POSITION_ATOL_MM):
            raise DicomInputError("geometry_unsupported")
        origin = headers[int(order[0])].position
        for index in order:
            displacement = headers[int(index)].position - origin
            drift = displacement - normal * np.dot(displacement, normal)
            if np.linalg.norm(drift) > _POSITION_ATOL_MM:
                raise DicomInputError("geometry_unsupported")
    affine_lps = np.eye(4, dtype=np.float64)
    # DICOM IOP first vector advances COLUMN; PixelSpacing is (ROW, COLUMN).
    affine_lps[:3, 0] = first.orientation[0] * first.spacing[1]
    affine_lps[:3, 1] = first.orientation[1] * first.spacing[0]
    affine_lps[:3, 2] = normal * depth_spacing
    affine_lps[:3, 3] = headers[int(order[0])].position
    affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps
    area = float(np.linalg.norm(np.cross(affine_lps[:3, 0], affine_lps[:3, 1])))
    volume = area * depth_spacing if mode == "3d" else None
    if (not np.isfinite(affine_ras).all() or not np.isfinite(area) or area <= 0
            or (volume is not None and (not np.isfinite(volume) or volume <= 0))):
        raise DicomInputError("geometry_unsupported")
    return mode, order, affine_ras, area, volume


def _hash_source(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception:
        raise DicomInputError("source_integrity_failed") from None


def verify_sources(source_refs: Iterable[SourceRef]) -> bool:
    """Raise a safe error if an original is missing or has changed byte for byte."""
    for ref in source_refs:
        if _hash_source(ref.path) != ref.sha256:
            raise DicomInputError("source_integrity_failed")
    return True


def _read_pixels(header: _Header) -> np.ndarray:
    try:
        import pydicom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = pydicom.dcmread(header.path)
            pixels = dataset.pixel_array
        if pixels.shape != (header.rows, header.columns):
            raise ValueError
        # Decode to float64 first so integer rescale cannot wrap before float32.
        values = np.asarray(pixels, dtype=np.float64) * header.slope + header.intercept
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError
        return values
    except Exception:
        raise DicomInputError("pixel_data_unsupported") from None


@_private_intake
def read_dicom_input(files: Iterable[os.PathLike | str], input_mode: str | None = None) -> DicomInput:
    """Read a single plane or one regular classic MR series without modifying it.

    ``source_refs`` retain input order. Each ``frame_map`` entry maps an output
    depth index to its original object and zero-based source frame index.
    All provenance fields are private even though their repr is deliberately safe.
    """
    if input_mode not in (None, "2d", "3d"):
        raise DicomInputError("input_mode_unsupported")
    paths = _paths(files)
    hashes = tuple(_hash_source(path) for path in paths)
    headers = tuple(_validate_header(_read_header(path), path) for path in paths)
    mode, order, affine, area, volume = _stack_geometry(headers, input_mode)
    refs = tuple(SourceRef(path, digest, str(header.dataset.SOPInstanceUID))
                 for path, digest, header in zip(paths, hashes, headers))
    values = np.stack([_read_pixels(headers[int(index)]) for index in order])
    # Detect an original changing during metadata/pixel loading; never repair it.
    verify_sources(refs)
    mapping = tuple(FrameMapping(depth, int(index)) for depth, index in enumerate(order))
    return DicomInput(values, affine, mode, refs, mapping, "mm2" if mode == "2d" else "mm3", volume, area)


@_private_intake
def inspect_dicom_files(files: Iterable[os.PathLike | str]) -> dict:
    """Return only anonymous aggregate metadata counts, without decoding pixels."""
    paths = _paths(files)
    rejection_counts = Counter()
    schema_counts = Counter({key: 0 for key in _SCHEMA_FIELDS})
    headers = []
    classic_count = enhanced_count = 0
    for path in paths:
        try:
            dataset = _read_header(path)
            sop_class = str(getattr(dataset, "SOPClassUID", ""))
            classic_count += sop_class == CLASSIC_MR
            enhanced_count += sop_class in ENHANCED_MR
            for name, keyword in _SCHEMA_FIELDS.items():
                schema_counts[name] += keyword in dataset
            headers.append(_validate_header(dataset, path))
        except DicomInputError as error:
            rejection_counts[error.code] += 1
    support = "unsupported"
    selection_rejection = None
    if len(headers) == len(paths):
        try:
            mode, *_ = _stack_geometry(tuple(headers), None)
            support = f"supported_{mode}"
        except DicomInputError as error:
            selection_rejection = error.code
    return {
        "schema_version": 1, "metadata_only": True, "object_count": len(paths),
        "classic_mr_count": int(classic_count), "enhanced_mr_count": int(enhanced_count),
        "supported_object_count": len(headers), "schema_counts": dict(schema_counts),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "input_support": support, "selection_rejection_code": selection_rejection,
    }
