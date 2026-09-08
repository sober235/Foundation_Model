"""Label-map helpers shared by the pseudo-label pipeline."""
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def _as_image(obj):
    return obj if isinstance(obj, nib.spatialimages.SpatialImage) else nib.load(str(obj))


def resample_labels_to_reference(label_img, ref_img):
    """Nearest-neighbour resample of an integer label image onto ``ref_img``'s grid.

    Geometry comes from both affines, so the label image may live at a different
    resolution and axis ordering (e.g. SynthSeg output at 1 mm vs. a fastMRI
    stack at 0.6875 x 0.6875 x 5 mm). Returns a NIfTI image with the reference
    shape and affine and an int16 data array.
    """
    label_img = _as_image(label_img)
    ref_img = _as_image(ref_img)
    out = resample_from_to(label_img, (ref_img.shape[:3], ref_img.affine), order=0, mode="constant", cval=0)
    data = np.rint(np.asarray(out.dataobj)).astype(np.int16)
    return nib.Nifti1Image(data, ref_img.affine)
