"""Throwaway probe helpers (2026-09-15): fastMRI+ brain small lesions x SynthSeg x raw k-space degradation.
Not part of the repo. Reuses the repo's geometry / degradation / SynthSeg plumbing unchanged."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, h5py, nibabel as nib
from scipy import ndimage

REPO = '/data0/congcong/code/Project_Doing/foundation_model'
sys.path.insert(0, REPO)
from anatobind.data_engine.fastmri import parse_recon_geometry, rss_affine, _pad_slices, PAD_TO_SLICES  # noqa
from anatobind.data_engine.fastmri_knee import reconstruct_rss, degrade, VIEWS, _view_seed, _iou  # noqa
from anatobind.data_engine.synthseg_pipeline import synthseg_command, native_labels, native_reference  # noqa

FM = Path('/data2/congcong/data/FM_data')
CSV = FM / 'fastMRI_lh_brain_knee/Annotations/brain.csv'
KROOT = FM / 'fastMRI_lh_brain_knee/kspace/brain'
SEG_EXISTING = FM / 'derived/synthseg/fastmri_brain/seg_native'
OUT = Path.home() / 'outputs/anatobind_brain_probe_2026-09-15'
WORK = OUT / 'work'
SMALL = {'Nonspecific white matter lesion', 'Lacunar infarct'}
LABEL_NAMES = {0: 'background', 2: 'L cerebral WM', 41: 'R cerebral WM', 3: 'L cortex', 42: 'R cortex',
               4: 'L lat ventricle', 43: 'R lat ventricle', 5: 'L inf lat vent', 44: 'R inf lat vent',
               7: 'L cerebellum WM', 46: 'R cerebellum WM', 8: 'L cerebellum cortex', 47: 'R cerebellum cortex',
               10: 'L thalamus', 49: 'R thalamus', 11: 'L caudate', 50: 'R caudate', 12: 'L putamen', 51: 'R putamen',
               13: 'L pallidum', 52: 'R pallidum', 17: 'L hippocampus', 53: 'R hippocampus', 18: 'L amygdala',
               54: 'R amygdala', 26: 'L accumbens', 58: 'R accumbens', 28: 'L ventral DC', 60: 'R ventral DC',
               14: '3rd ventricle', 15: '4th ventricle', 16: 'brainstem', 24: 'CSF'}
ALL_CAND = sorted(k for k in LABEL_NAMES if k != 0)
NON_PARENCHYMA = {4, 43, 5, 44, 14, 15, 24}
PARENCHYMA_CAND = sorted(k for k in ALL_CAND if k not in NON_PARENCHYMA)
RULES = {'all_structures': ALL_CAND, 'parenchyma_only': PARENCHYMA_CAND}


def h5_path(stem):
    for split in ('multicoil_train', 'multicoil_val'):
        p = KROOT / split / f'{stem}.h5'
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def read_boxes():
    rows = []
    for r in csv.DictReader(open(CSV)):
        if r['study_level'].strip() == 'Yes':
            continue
        try:
            rows.append(dict(file=r['file'], slice=int(r['slice']), x=int(r['x']), y=int(r['y']),
                             width=int(r['width']), height=int(r['height']), label=r['label'].strip()))
        except ValueError:
            continue
    return [r for r in rows if r['width'] >= 3 and r['height'] >= 3]


def merge_lesions(rows, iou_min=0.3):
    """Adjacent-slice boxes of the same (file, label) with in-plane IoU >= 0.3 form one lesion (leg 2 rule)."""
    by = defaultdict(list)
    for r in rows:
        by[(r['file'], r['label'])].append(r)
    lesions = []
    for (file, label), group in sorted(by.items()):
        parent = list(range(len(group)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, a in enumerate(group):
            for j, b in enumerate(group):
                if j <= i or abs(a['slice'] - b['slice']) != 1:
                    continue
                if _iou(a, b) >= iou_min:
                    parent[find(i)] = find(j)
        comps = defaultdict(list)
        for i in range(len(group)):
            comps[find(i)].append(group[i])
        for k, members in enumerate(sorted(comps.values(), key=lambda m: (m[0]['slice'], m[0]['x'], m[0]['y']))):
            lesions.append(dict(file=file, label=label, members=members, small=label in SMALL,
                                lesion_id=f'{file}:{label}:{k}'))
    return lesions


def region_mask(lesion, shape, flip_y=True, flip_x=False, dx=0, dy=0):
    """Boolean mask on the (col, row, slice) grid. fastMRI+ README: images were flipped up/down for labelling,
    so y counts rows from the bottom of the RSS array (flip_y=True; verified in probe0b/0c on 24 volumes)."""
    m = np.zeros(shape, dtype=bool)
    nc, nr, ns = shape
    for b in lesion['members']:
        x0, x1 = b['x'] + dx, b['x'] + b['width'] + dx
        y0, y1 = b['y'] + dy, b['y'] + b['height'] + dy
        if flip_x:
            x0, x1 = nc - x1, nc - x0
        if flip_y:
            y0, y1 = nr - y1, nr - y0
        x0, x1 = max(0, x0), min(nc, x1)
        y0, y1 = max(0, y0), min(nr, y1)
        if x1 > x0 and y1 > y0 and 0 <= b['slice'] < ns:
            m[x0:x1, y0:y1, b['slice']] = True
    return m


class Lookup:
    """Category-free geometric host lookup: argmax overlap over candidate labels, zero overlap -> nearest candidate."""

    def __init__(self, seg, spacing, candidates):
        self.seg = seg
        self.cand = np.array(candidates)
        self.is_cand = np.isin(seg, self.cand)
        _, idx = ndimage.distance_transform_edt(~self.is_cand, sampling=spacing, return_indices=True)
        self.nearest = seg[tuple(idx)]

    def host(self, mask):
        vals = self.seg[mask & self.is_cand]
        if vals.size:
            c = np.bincount(vals, minlength=int(self.cand.max()) + 1)
            return int(np.argmax(c)), float(c.max() / mask.sum())
        near = self.nearest[mask]
        c = np.bincount(near[near > 0], minlength=int(self.cand.max()) + 1)
        return int(np.argmax(c)), 0.0


def load_seg(path):
    img = nib.load(str(path))
    return np.asarray(img.dataobj).astype(np.int16), tuple(float(z) for z in img.header.get_zooms()[:3])


def geometry(h5):
    with h5py.File(h5, 'r') as f:
        hdr = f['ismrmrd_header'][()]
        rss_shape = f['reconstruction_rss'].shape
    if isinstance(hdr, bytes):
        hdr = hdr.decode()
    g = parse_recon_geometry(hdr)
    return dict(row_sp=g['enc_fov_x_mm'] / g['enc_nx'], col_sp=g['enc_fov_y_mm'] / g['enc_ny'],
                slice_sp=g['fov_z_mm'], rss_shape=rss_shape)


def write_view_nifti(vol_src, geo, out_path):
    """vol_src: (slice, row, col) float32 -> NIfTI on the same grid/affine the pipeline uses."""
    vol = np.ascontiguousarray(vol_src.transpose(2, 1, 0)).astype(np.float32)
    vol = _pad_slices(vol, PAD_TO_SLICES)
    aff = rss_affine(geo['row_sp'], geo['col_sp'], geo['slice_sp'], shape=vol.shape)
    img = nib.Nifti1Image(vol, aff)
    img.header.set_xyzt_units('mm')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))
    return out_path


def lesion_rects(lesion, shape, flip_y=True, dx=0, dy=0, ds=0):
    """Axis-aligned (x0, x1, y0, y1, s) rectangles of the lesion on the (col, row, slice) grid, shifted by voxels."""
    nc, nr, ns = shape
    out = []
    for b in lesion['members']:
        x0, x1 = b['x'] + dx, b['x'] + b['width'] + dx
        y0, y1 = b['y'] + dy, b['y'] + b['height'] + dy
        if flip_y:
            y0, y1 = nr - y1, nr - y0
        x0, x1, y0, y1, s = max(0, x0), min(nc, x1), max(0, y0), min(nr, y1), b['slice'] + ds
        if x1 > x0 and y1 > y0 and 0 <= s < ns:
            out.append((x0, x1, y0, y1, s))
    return out


def host_rects(lookup, rects):
    """Same rule as Lookup.host but only touching the lesion's own voxels (fast)."""
    vals = np.concatenate([lookup.seg[x0:x1, y0:y1, s].ravel() for x0, x1, y0, y1, s in rects]) if rects else np.zeros(0, np.int16)
    if vals.size == 0:
        return -1, 0.0
    cv = vals[np.isin(vals, lookup.cand)]
    if cv.size:
        c = np.bincount(cv, minlength=int(lookup.cand.max()) + 1)
        return int(np.argmax(c)), float(c.max() / vals.size)
    near = np.concatenate([lookup.nearest[x0:x1, y0:y1, s].ravel() for x0, x1, y0, y1, s in rects])
    c = np.bincount(near[near > 0], minlength=int(lookup.cand.max()) + 1)
    return int(np.argmax(c)), 0.0
