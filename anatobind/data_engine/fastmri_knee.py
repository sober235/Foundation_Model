"""fastMRI+ 膝关节的标注读取、清洗、3D 合并、患者划分与受控退化（leg 2 spec 第 1 节）。

标注是逐层 2D 框，坐标在 320x320 的 RSS 图像空间。3D 病灶由"相邻层、面内 IoU >= 0.3 相连"
的连通分量得到；同一类别族但空间分离的两串因此分成两个病灶。
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

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
    """CSV 行 -> 带 family 的整数化行；标签两端的空白必须 strip（原文件里 "Joint Effusion " 带尾空格）。"""
    out = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["study_level"].strip() == "Yes":
                continue
            fam = FAMILY_OF_LABEL.get(r["label"].strip())
            if fam is None:
                continue
            try:
                row = {"file": r["file"], "slice": int(r["slice"]), "x": int(r["x"]), "y": int(r["y"]),
                       "width": int(r["width"]), "height": int(r["height"]), "family": fam}
            except ValueError:
                continue
            out.append(row)
    return out


def clean_boxes(rows):
    kept, dropped = [], Counter()
    for r in rows:
        if r["width"] < MIN_SIDE or r["height"] < MIN_SIDE:
            dropped["too_small"] += 1
            continue
        kept.append(r)
    return kept, dropped


def _iou(a, b):
    ax1, ay1 = a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1 = b["x"] + b["width"], b["y"] + b["height"]
    iw = max(0, min(ax1, bx1) - max(a["x"], b["x"]))
    ih = max(0, min(ay1, by1) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union else 0.0


def merge_to_3d(rows, iou_min=0.3):
    """相邻层、面内 IoU >= iou_min 的框属于同一个 3D 病灶。"""
    lesions = []
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["file"], r["family"])].append(r)
    for (file, family), group in sorted(by_group.items()):
        parent = list(range(len(group)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, a in enumerate(group):
            for j, b in enumerate(group):
                if j <= i or abs(a["slice"] - b["slice"]) != 1:
                    continue
                if _iou(a, b) >= iou_min:
                    parent[find(i)] = find(j)
        comps = defaultdict(list)
        for i in range(len(group)):
            comps[find(i)].append(group[i])
        for members in comps.values():
            lesions.append({
                "file": file, "family": family,
                "z0": min(m["slice"] for m in members), "z1": max(m["slice"] for m in members),
                "x0": min(m["x"] for m in members), "y0": min(m["y"] for m in members),
                "x1": max(m["x"] + m["width"] for m in members),
                "y1": max(m["y"] + m["height"] for m in members),
                "n_boxes": len(members),
            })
    return lesions


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
EXPORT_ROOT = Path("/data2/congcong/data/FM_data/derived/fastmri_knee/leg2")


def volume_paths(root=KSPACE_ROOT):
    return {p.stem: p for split in ("multicoil_train", "multicoil_val") for p in sorted((Path(root) / split).glob("*.h5"))}


def patient_of(paths):
    out = {}
    for name, p in paths.items():
        with h5py.File(p) as h:
            out[name] = str(h.attrs.get("patient_id", name))
    return out


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
