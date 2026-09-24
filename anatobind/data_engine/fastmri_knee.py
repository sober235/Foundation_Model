"""fastMRI+ 膝关节的标注读取、清洗、3D 合并、患者划分与受控退化（leg 2 spec 第 1 节）。

标注是逐层 2D 框；CSV 里的 y 从 RSS 数组底部数起，读入后先经 rows_to_rss_frame 转到行从顶部数的 RSS 帧（320x320）再合并。3D 病灶由"相邻层、面内 IoU >= 0.3 相连"
的连通分量得到；同一类别族但空间分离的两串因此分成两个病灶。
"""
import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from anatobind.data_engine.fastmri import BOX_CONVENTION_RSS, TRANSFORM_VERSION, box_iou_2d, merge_boxes_3d, read_fastmri_plus_rows, rss_spacing_mm
from anatobind.train.dataset import normalise_volume

FAMILIES = ("meniscus", "cartilage", "bone", "ligament", "effusion")
FAMILY_OF_LABEL = {
    "Meniscus Tear": "meniscus",
    "Displaced Meniscal Tissue": "meniscus",
    "Cartilage - Partial Thickness loss/defect": "cartilage",
    "Cartilage - Full Thickness loss/defect": "cartilage",
    "Bone-Fracture/Contusion/dislocation": "bone",
    "Bone- Subchondral edema": "bone",
    "Bone - Lesion": "bone",
    "Ligament - ACL Low Grade sprain": "ligament",
    "Ligament - ACL High Grade Sprain": "ligament",
    "Ligament - MCL Low-Mod Grade Sprain": "ligament",
    "Ligament - MCL High Grade sprain": "ligament",
    "Ligament - PCL Low-Mod grade sprain": "ligament",
    "Ligament - PCL High Grade": "ligament",
    "LCL Complex - Low-Mod Grade Sprain": "ligament",
    "LCL Complex- High Grade Sprain": "ligament",
    "Patellar Retinaculum - High grade sprain": "ligament",
    "Joint Effusion": "effusion",
}
MIN_SIDE = 3


def read_annotations(csv_path):
    """CSV rows -> rows with a family (labels outside FAMILY_OF_LABEL are dropped). Still in the CSV frame:
    the row flip happens once, in rows_to_rss_frame, before merging."""
    out = []
    for r in read_fastmri_plus_rows(csv_path):
        fam = FAMILY_OF_LABEL.get(r["label"])
        if fam is not None:
            out.append({**r, "family": fam})
    return out


def clean_boxes(rows):
    kept, dropped = [], Counter()
    for r in rows:
        if r["width"] < MIN_SIDE or r["height"] < MIN_SIDE:
            dropped["too_small"] += 1
            continue
        kept.append(r)
    return kept, dropped


_iou = box_iou_2d       # old name, still imported by the probe scripts under docs/verification


def merge_to_3d(rows, iou_min=0.3):
    """Adjacent-slice boxes of one family with in-plane IoU >= iou_min are one lesion (fastmri.merge_boxes_3d)."""
    return merge_boxes_3d(rows, "family", iou_min)


VIEWS = ("clean", "noise_q1", "noise_q2", "noise_q3", "us4", "us8", "us16")
NOISE_C = {"noise_q1": 1.0, "noise_q2": 2.0, "noise_q3": 4.0}
ACCEL = {"us4": (4, 0.08), "us8": (8, 0.04), "us16": (16, 0.04)}


def reconstruct_rss(kspace, size=320):
    """干净与退化共用的唯一重建算子：逐线圈中心化 2D IFFT，线圈平方和开方，中心裁剪。

    与 fastMRI 自带的 reconstruction_rss 一致（2026-09-14 在 file1000001 第 10 层实测 NRMSE 7.3e-8）。
    """
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
    rss = np.sqrt((np.abs(img) ** 2).sum(axis=-3))
    out = rss
    for axis in (-2, -1):
        n = out.shape[axis]
        if n > size:
            lo = (n - size) // 2
            out = out.take(range(lo, lo + size), axis=axis)
    return out.astype(np.float32)


def noise_sigma(kspace, c):
    return float(c * np.median(np.abs(kspace)))


def equispaced_mask(n_pe, accel, centre_fraction, seed, nest_centre_fraction=0.08):
    """1D 相位编码掩膜。中心全采；外围顺序由最宽的中心块决定，因此不同加速倍数的掩膜互相嵌套。"""
    mask = np.zeros(n_pe, dtype=bool)
    n_centre = int(round(centre_fraction * n_pe))
    lo = (n_pe - n_centre) // 2
    mask[lo:lo + n_centre] = True
    n_wide = int(round(nest_centre_fraction * n_pe))
    wlo = (n_pe - n_wide) // 2
    rest = np.array([i for i in range(n_pe) if not (wlo <= i < wlo + n_wide)])
    order = np.random.default_rng(seed).permutation(len(rest))
    n_extra = max(0, int(round(n_pe / accel)) - n_centre)
    mask[rest[order[:n_extra]]] = True
    return mask


def degrade(kspace, view, seed):
    if view == "clean":
        return kspace
    if view in NOISE_C:
        sigma = noise_sigma(kspace, NOISE_C[view])
        rng = np.random.default_rng(seed)
        n = rng.normal(scale=sigma / np.sqrt(2), size=kspace.shape) \
            + 1j * rng.normal(scale=sigma / np.sqrt(2), size=kspace.shape)
        return kspace + n.astype(kspace.dtype)
    accel, centre = ACCEL[view]
    mask = equispaced_mask(kspace.shape[-1], accel, centre, seed)
    return kspace * mask


KSPACE_ROOT = Path("/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/knee")
ANNOTATIONS = Path("/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/knee.csv")
EXPORT_ROOT = Path("/data2/congcong/data/FM_data/derived/fastmri_knee/leg2_gate0")     # boxes in the RSS frame (Gate 0)
LEGACY_EXPORT_ROOT = Path("/data2/congcong/data/FM_data/derived/fastmri_knee/leg2")    # 2026-09-14, boxes as-is: read-only history
IMAGE_ORIENTATION = "reconstruction_rss order (slice, row, col); rows counted from the top"
MANIFEST_FIELDS = ["file", "out_dir", "slices", "n_lesions", "status", "patient_id", "n_rows", "n_cols",
                   "spacing_slice_mm", "spacing_row_mm", "spacing_col_mm", "image_orientation",
                   "box_coordinate_convention", "transform_version"]
LESION_FIELDS = ["lesion_id", "file", "family", "z0", "z1", "x0", "y0", "x1", "y1", "n_boxes"]


class LegacyBoxConvention(ValueError):
    """The export was written before Gate 0 (CSV rows used as-is, i.e. mirrored boxes); refuse to use it."""


def volume_paths(root=KSPACE_ROOT):
    return {p.stem: p for split in ("multicoil_train", "multicoil_val") for p in sorted((Path(root) / split).glob("*.h5"))}


def _patient_attr(h, path):
    if "patient_id" not in h.attrs:
        raise KeyError(f"{path}: no patient_id attribute; folds are split by patient (v2.6 §4.4)")
    pid = h.attrs["patient_id"]
    return pid.decode() if isinstance(pid, bytes) else str(pid)


def patient_of(paths):
    out = {}
    for name, p in paths.items():
        with h5py.File(p) as h:
            out[name] = _patient_attr(h, p)
    return out


def volume_geometry(h5_path):
    """patient_id, RSS grid and (slice, row, col) spacing of one fastMRI h5 file."""
    with h5py.File(h5_path) as h:
        pid = _patient_attr(h, h5_path)
        hdr = h["ismrmrd_header"][()]
        n_slices, n_rows, n_cols = h["reconstruction_rss"].shape
    sp = rss_spacing_mm(hdr.decode() if isinstance(hdr, bytes) else hdr)
    return {"patient_id": pid, "n_rows": int(n_rows), "n_cols": int(n_cols), "slices": int(n_slices),
            "spacing_slice_mm": sp[0], "spacing_row_mm": sp[1], "spacing_col_mm": sp[2]}


def manifest_row(name, h5_path, out_dir, n_lesions, status):
    g = volume_geometry(h5_path)
    return {"file": name, "out_dir": str(out_dir), "slices": g["slices"], "n_lesions": n_lesions, "status": status,
            "patient_id": g["patient_id"], "n_rows": g["n_rows"], "n_cols": g["n_cols"],
            "spacing_slice_mm": g["spacing_slice_mm"], "spacing_row_mm": g["spacing_row_mm"],
            "spacing_col_mm": g["spacing_col_mm"], "image_orientation": IMAGE_ORIENTATION,
            "box_coordinate_convention": BOX_CONVENTION_RSS, "transform_version": TRANSFORM_VERSION}


def write_manifest(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_lesions(path, lesions):
    """lesions.csv: one row per 3D lesion, numbered in (file, family, z0, x0) order; members are not written."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LESION_FIELDS, extrasaction="ignore")
        w.writeheader()
        for i, L in enumerate(sorted(lesions, key=lambda L: (L["file"], L["family"], L["z0"], L["x0"]))):
            w.writerow({"lesion_id": i, **L})


def load_manifest(export_root):
    """manifest.csv -> {file: row}; refuses exports whose boxes are not transform_version 2."""
    path = Path(export_root) / "manifest.csv"
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r["file"] for r in rows if str(r.get("transform_version", "")) != str(TRANSFORM_VERSION)]
    if bad:
        raise LegacyBoxConvention(
            f"{path}: {len(bad)} of {len(rows)} volumes lack transform_version {TRANSFORM_VERSION} (first {bad[:3]}); "
            f"their boxes are the mirrored CSV boxes of leg 2 before Gate 0. Build a converted root with "
            f"scripts/relink_fastmri_knee_gate0.py.")
    return {r["file"]: r for r in rows}


def load_lesions(export_root):
    """lesions.csv of a Gate-0 export (RSS frame) with integers parsed; refuses legacy exports."""
    load_manifest(export_root)
    with open(Path(export_root) / "lesions.csv", newline="") as fh:
        return [{**r, **{k: int(r[k]) for k in ("z0", "z1", "x0", "y0", "x1", "y1", "n_boxes")}}
                for r in csv.DictReader(fh)]


def assert_folds_by_patient(folds, patients):
    """Raise if any patient has volumes in two folds. folds: {file: fold}; patients: {file: patient_id}."""
    fold_of_patient = {}
    for vol, fold in folds.items():
        p = patients[vol]
        if fold_of_patient.setdefault(p, fold) != fold:
            raise ValueError(f"patient {p} is in fold {fold_of_patient[p]} and fold {fold} ({vol}); "
                             f"folds must be patient-disjoint (v2.6 §4.4)")


def load_folds(export_root):
    """folds.json -> {file: fold}, asserted patient-disjoint against the manifest's patient_id column."""
    folds = json.loads((Path(export_root) / "folds.json").read_text())["folds"]
    manifest = load_manifest(export_root)
    assert_folds_by_patient(folds, {f: manifest[f]["patient_id"] for f in folds})
    return folds


def make_folds(patients, k=5, seed=0):
    """按患者切 k 折；同一患者的全部卷同折。"""
    uniq = sorted(set(patients.values()))
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of_patient = {uniq[int(j)]: i % k for i, j in enumerate(order)}
    return {vol: fold_of_patient[p] for vol, p in patients.items()}


def _view_seed(view, volume_seed, slice_index):
    """One mask per volume for undersampling; independent noise per slice."""
    return volume_seed if view in ACCEL else volume_seed + 1 + slice_index


def export_volume(h5_path, lesions, out_dir, seed, size=320):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(h5_path) as h:
            kspace = h["kspace"][()]
            meta = {"patient_id": str(h.attrs.get("patient_id", "")),
                    "acquisition": str(h.attrs.get("acquisition", ""))}
        for view in VIEWS:
            vol = np.stack([reconstruct_rss(degrade(kspace[s], view, _view_seed(view, seed, s)), size)
                            for s in range(kspace.shape[0])])
            np.save(out_dir / f"{view}.npy", normalise_volume(vol).astype(np.float16))
        meta.update({"slices": int(kspace.shape[0]), "size": size, "n_lesions": len(lesions), "seed": seed})
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        return {"file": Path(h5_path).stem, "out_dir": str(out_dir), "slices": meta["slices"],
                "n_lesions": len(lesions), "status": "ok"}
    except Exception as exc:
        return {"file": Path(h5_path).stem, "out_dir": str(out_dir), "slices": 0, "n_lesions": 0,
                "status": f"error: {type(exc).__name__}: {exc}"[:200]}
