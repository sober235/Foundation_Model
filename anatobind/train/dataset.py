"""Read the M1 SKM-TEA export into arm-B training samples.

The export stores (X, Y, Z) arrays with in-plane X, Y at 0.625 mm and slice Z at
0.8 mm.  The model works in (Z, Y, X) so that the stem (2, 4, 4) reduces the thin
axis by 2 and the in-plane axes by 4, which reproduces the token budget in
RESEARCH_PLAN.md 4.2 (160x256x256 -> 80x64x64).  Every array, box and spacing
crossing into the model therefore goes through the permutation helpers below.

Relation truth is the ``host_label`` column, i.e. the annotator's ``tissue_id``
side-resolved by the data engine.  It is never derived from overlap (spec 5.1).
"""

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

K_STRUCTURES = 6
# query k-1 is permanently segmentation label k (identity-anchored, spec 4.3)
SEG_NAMES = {
    1: "patellar cartilage", 2: "femoral cartilage",
    3: "tibial cartilage medial", 4: "tibial cartilage lateral",
    5: "meniscus medial", 6: "meniscus lateral",
}
SUPER_TO_CLASS = {"Meniscal Tear": 0, "Cartilage Lesion": 1}
# host_side "unresolved" leaves host_label empty: the medial/lateral side could
# not be decided, so the relation truth is UNKNOWN.  Spec 4.3 and decision 2A
# keep present, confirmed-absent and unknown apart, so no host is invented; the
# box still supervises U_B but is dropped from every relation term.
UNKNOWN_HOST = 0
CLIP_PERCENTILES = (0.5, 99.5)


def to_model_frame(arr):
    """(X, Y, Z) -> (Z, Y, X)."""
    return np.ascontiguousarray(arr.transpose(2, 1, 0))


def box_to_model_frame(box):
    """(x0, y0, z0, x1, y1, z1) -> (z0, y0, x0, z1, y1, x1)."""
    x0, y0, z0, x1, y1, z1 = box
    return (z0, y0, x0, z1, y1, x1)


def normalise_volume(vol):
    """Percentile-clip over the non-zero voxels, then z-score (spec 4.1)."""
    v = np.asarray(vol, dtype=np.float32)
    nz = v[v > 0]
    if nz.size == 0:
        nz = v.reshape(-1)
    lo, hi = np.percentile(nz, CLIP_PERCENTILES)
    v = np.clip(v, lo, hi)
    std = float(v.std())
    return ((v - float(v.mean())) / (std if std > 0 else 1.0)).astype(np.float32)


def list_ready_scans(root):
    """Scans the exporter finished: an ``ok`` manifest row and a boxes.csv."""
    root = Path(root)
    ready = []
    with open(root / "manifest.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["status"] == "ok" and (root / row["scan_id"] / "boxes.csv").exists():
                ready.append(row["scan_id"])
    return sorted(ready)


def load_fold(root, fold):
    """(train_ids, val_ids) for one fold, restricted to finished scans."""
    root = Path(root)
    folds = json.loads((root / "splits.json").read_text())["folds"]
    ready = set(list_ready_scans(root))
    val = sorted(s for s, f in folds.items() if f == fold and s in ready)
    train = sorted(s for s, f in folds.items() if f != fold and s in ready)
    return train, val


def read_in_seg_boxes(path):
    """In-segmentation rows only, already permuted into the model frame."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["layer"] != "in_seg":
                continue
            box = box_to_model_frame(tuple(int(r[k]) for k in
                                           ("x0", "y0", "z0", "x1", "y1", "z1")))
            host = r["host_label"].strip()
            rows.append({
                "box": box,
                "host_label": int(host) if host else UNKNOWN_HOST,
                "cls": SUPER_TO_CLASS[r["supercategory"]],
            })
    return rows


class SkmteaArmBDataset:
    """One lesion-centred crop per scan, in the model frame."""

    def __init__(self, scan_ids, root, patch=(64, 128, 128), train=True, seed=0):
        self.scan_ids = list(scan_ids)
        self.root = Path(root)
        self.patch = tuple(patch)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.scan_ids)

    def _load(self, scan_id):
        d = self.root / scan_id
        img = nib.load(d / "image_clean_e1.nii.gz")
        seg = nib.load(d / "seg.nii.gz")
        image = to_model_frame(normalise_volume(img.get_fdata(dtype=np.float32)))
        labels = to_model_frame(np.asanyarray(seg.dataobj).astype(np.int64))
        spacing = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)[::-1]
        return image, labels, np.ascontiguousarray(spacing), read_in_seg_boxes(d / "boxes.csv")

    def _origin(self, shape, boxes):
        patch = np.asarray(self.patch)
        limit = np.asarray(shape) - patch
        if not boxes:
            return np.maximum(limit // 2, 0)
        seed = boxes[self.rng.integers(len(boxes))] if self.train else boxes[0]
        box = np.asarray(seed["box"], dtype=np.float64)
        centre = (box[:3] + box[3:]) / 2.0
        origin = centre - patch / 2.0
        if self.train:
            origin = origin + self.rng.integers(-(patch // 4), patch // 4 + 1)
        return np.clip(np.round(origin).astype(int), 0, np.maximum(limit, 0))

    def __getitem__(self, idx):
        scan_id = self.scan_ids[idx]
        image, labels, spacing, boxes = self._load(scan_id)
        origin = self._origin(image.shape, boxes)
        d, h, w = self.patch
        sl = (slice(origin[0], origin[0] + d),
              slice(origin[1], origin[1] + h),
              slice(origin[2], origin[2] + w))

        kept, classes, hosts = [], [], []
        upper = origin + np.asarray(self.patch)
        for b in boxes:
            box = np.asarray(b["box"], dtype=np.int64)
            lo = np.maximum(box[:3], origin)
            hi = np.minimum(box[3:], upper)
            if (hi <= lo).any():
                continue
            kept.append(np.concatenate([lo - origin, hi - origin]))
            classes.append(b["cls"])
            hosts.append(b["host_label"])

        seg_crop = labels[sl]
        present = np.array([(seg_crop == k + 1).any() for k in range(K_STRUCTURES)])
        return {
            "scan_id": scan_id,
            "crop_origin": origin.astype(np.int64),
            "image": image[sl][None].astype(np.float32),
            "seg": seg_crop,
            "boxes": (np.stack(kept).astype(np.float32) if kept
                      else np.zeros((0, 6), dtype=np.float32)),
            "box_classes": np.asarray(classes, dtype=np.int64),
            "host_label": np.asarray(hosts, dtype=np.int64),
            "spacing_mm": spacing,
            "present": present,
        }


def collate(samples):
    """Stack the dense fields; box counts differ per sample so those stay lists."""
    stacked = {k: torch.from_numpy(np.stack([s[k] for s in samples]))
               for k in ("image", "seg", "present", "spacing_mm")}
    ragged = {k: [torch.from_numpy(s[k]) for s in samples]
              for k in ("boxes", "box_classes", "host_label")}
    return {**stacked, **ragged, "scan_id": [s["scan_id"] for s in samples]}
