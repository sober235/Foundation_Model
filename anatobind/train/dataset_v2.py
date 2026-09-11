"""Whole-volume samples for the stage-II upstream (RESEARCH_PLAN v2.2 13.6).

One sample is one scan in one view. Training draws the clean view with probability 0.5 and
otherwise one of the six degraded views uniformly; evaluation enumerates (scan, view) pairs.
Every kept annotation is a U_B target: meniscal tears and cartilage lesions carry their host
label, effusions carry NONE_HOST (an event with no host among the K structures), and ligament
tears carry UNKNOWN_HOST (their ACL/PCL host is outside the ontology, so they are detected but
excluded from relation terms). Family truth for scoring is tissue_id, carried alongside.

The medial-lateral axis is model-frame Z. Mirroring it keeps every anatomical label: a mirrored
left knee is a right knee, with its medial meniscus still medial.
"""
import csv
from pathlib import Path

import numpy as np
import torch

from anatobind.train.cache import VIEWS, load_array, load_meta
from anatobind.train.dataset import K_STRUCTURES, UNKNOWN_HOST, box_to_model_frame

CLASSES = {"Meniscal Tear": 0, "Cartilage Lesion": 1, "Effusion": 2, "Ligament Tear": 3}
NUM_CLASSES = len(CLASSES)
NONE_HOST = K_STRUCTURES + 1
DEGRADED = VIEWS[1:]
DEPTH_MULTIPLE = 32


def read_all_boxes(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["layer"] == "effusion":
                host = NONE_HOST
            elif r["layer"] == "ligament":
                host = UNKNOWN_HOST
            else:
                host = int(r["host_label"]) if r["host_label"].strip() else UNKNOWN_HOST
            rows.append({
                "box": box_to_model_frame(tuple(int(r[k]) for k in ("x0", "y0", "z0", "x1", "y1", "z1"))),
                "cls": CLASSES[r["supercategory"]], "host_label": host, "tissue_id": int(r["tissue_id"]),
                "layer": r["layer"], "ann_id": int(r["ann_id"]),
            })
    return rows


def pad_depth(vol, multiple=DEPTH_MULTIPLE, value=0):
    depth = vol.shape[0]
    target = -(-depth // multiple) * multiple
    if target == depth:
        return vol, depth
    pad = [(0, target - depth)] + [(0, 0)] * (vol.ndim - 1)
    return np.pad(vol, pad, constant_values=value), depth


def flip_depth(image, seg, boxes, depth):
    out = boxes.copy()
    if len(out):
        out[:, 0], out[:, 3] = depth - boxes[:, 3], depth - boxes[:, 0]
    return image[::-1].copy(), seg[::-1].copy(), out


def seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)


class WholeVolumeDataset:
    def __init__(self, scan_ids, cache_root, export_root, train=True, views=VIEWS, p_clean=0.5, flip=True, seed=0):
        self.scan_ids = list(scan_ids)
        self.cache_root, self.export_root = Path(cache_root), Path(export_root)
        self.train, self.p_clean, self.flip = train, p_clean, flip
        self.items = [(s, None) for s in self.scan_ids] if train else [(s, v) for s in self.scan_ids for v in views]
        self.rng = np.random.default_rng(seed)
        self.boxes = {s: read_all_boxes(self.export_root / s / "boxes.csv") for s in self.scan_ids}

    def __len__(self):
        return len(self.items)

    def _view(self):
        if self.rng.random() < self.p_clean:
            return "clean"
        return DEGRADED[int(self.rng.integers(len(DEGRADED)))]

    def __getitem__(self, idx):
        scan, view = self.items[idx]
        view = view or self._view()
        d = self.cache_root / scan
        image = np.asarray(load_array(d, view), dtype=np.float32)
        seg = np.asarray(load_array(d, "seg"), dtype=np.int64)
        rows = self.boxes[scan]
        boxes = np.array([r["box"] for r in rows], dtype=np.float32).reshape(-1, 6)
        depth = image.shape[0]
        if self.train and self.flip and self.rng.random() < 0.5:
            image, seg, boxes = flip_depth(image, seg, boxes, depth)
        image, _ = pad_depth(image, value=float(image.min()))
        seg, _ = pad_depth(seg)
        return {
            "scan_id": scan, "view": view, "image": image[None], "seg": seg, "valid_depth": depth,
            "boxes": boxes,
            "box_classes": np.array([r["cls"] for r in rows], dtype=np.int64),
            "host_label": np.array([r["host_label"] for r in rows], dtype=np.int64),
            "tissue_id": np.array([r["tissue_id"] for r in rows], dtype=np.int64),
            "ann_id": np.array([r["ann_id"] for r in rows], dtype=np.int64),
            "spacing_mm": np.asarray(load_meta(d)["spacing_mm"], dtype=np.float32),
            "present": np.array([(seg == k + 1).any() for k in range(K_STRUCTURES)]),
        }


def collate_one(samples):
    assert len(samples) == 1, "whole-volume training runs at batch size 1"
    s = samples[0]
    return {
        "image": torch.from_numpy(s["image"])[None], "seg": torch.from_numpy(s["seg"])[None],
        "valid_depth": torch.tensor([s["valid_depth"]]), "present": torch.from_numpy(s["present"])[None],
        "spacing_mm": torch.from_numpy(s["spacing_mm"])[None],
        "boxes": [torch.from_numpy(s["boxes"])], "box_classes": [torch.from_numpy(s["box_classes"])],
        "host_label": [torch.from_numpy(s["host_label"])], "tissue_id": [torch.from_numpy(s["tissue_id"])],
        "ann_id": [torch.from_numpy(s["ann_id"])], "scan_id": [s["scan_id"]], "view": [s["view"]],
    }
