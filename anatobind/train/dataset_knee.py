"""leg 2 的薄块数据集：一个样本 = 某视图下以某层为中心的 5 层薄块，只在中心层监督。

增广是这条腿的重点。第一条腿用 124 个整卷、只有一次翻转，检测头因此背下了训练集
（docs/verification/2026-09-12/fold0_pilot.md 第 6 节）。这里每个标注层都是一个样本，
并加入左右翻转、平移与强度扰动。

样本索引（Task 1 结论，docs/verification/2026-09-14/fastmri_plus_negatives.md）：
`UNANNOTATED_VOLUMES_ARE_NEGATIVE = True`——198 个完全无标注的卷是放射科医生读过的
正常片，因此它们的每一层都是真负样本；`UNANNOTATED_SLICES_IN_ANNOTATED_VOLUMES_ARE_NEGATIVE
= False`——论文没有确立"有标注的卷里，没框的层也被判读过"，所以这类层既非正也非负，
必须被排除在索引之外。
"""
import json
from pathlib import Path

import numpy as np
import torch

from anatobind.data_engine.fastmri_knee import FAMILIES, VIEWS

DEGRADED = tuple(v for v in VIEWS if v != "clean")
CLASS_OF_FAMILY = {f: i for i, f in enumerate(FAMILIES)}
MAX_SHIFT = 16


def seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2 ** 32)


class SlabDataset(torch.utils.data.Dataset):
    def __init__(self, files, export_root, lesions, train=True, views=VIEWS, p_clean=0.5, slab=5, seed=0):
        self.root = Path(export_root)
        self.train, self.views, self.p_clean, self.slab = train, tuple(views), p_clean, slab
        self.rng = np.random.default_rng(seed)
        self.files = list(files)
        self.meta = {f: json.loads((self.root / f / "meta.json").read_text()) for f in self.files}
        self.by_slice = {}
        for L in lesions:
            if L["file"] not in self.meta:
                continue
            for z in range(int(L["z0"]), int(L["z1"]) + 1):
                self.by_slice.setdefault((L["file"], z), []).append(L)
        annotated = {f for (f, z) in self.by_slice}
        self.index = [(f, z) for f in self.files for z in range(self.meta[f]["slices"])
                      if f not in annotated or (f, z) in self.by_slice]

    def __len__(self):
        return len(self.index) * (1 if self.train else len(self.views))

    def _view(self, i):
        if self.train:
            return "clean" if self.rng.random() < self.p_clean else DEGRADED[int(self.rng.integers(len(DEGRADED)))]
        return self.views[i // len(self.index)]

    def __getitem__(self, i):
        file, z = self.index[i % len(self.index)]
        view = self._view(i)
        vol = np.load(self.root / file / f"{view}.npy", mmap_mode="r")
        half = self.slab // 2
        idx = np.clip(np.arange(z - half, z + half + 1), 0, vol.shape[0] - 1)   # edge replicate
        img = np.ascontiguousarray(vol[idx]).astype(np.float32)
        boxes = [[L["y0"], L["x0"], L["y1"], L["x1"]] for L in self.by_slice.get((file, z), [])]
        classes = [CLASS_OF_FAMILY[L["family"]] for L in self.by_slice.get((file, z), [])]
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        if self.train:
            img, boxes = self._augment(img, boxes)
        return {"image": torch.from_numpy(img)[None], "boxes": torch.from_numpy(boxes),
                "box_classes": torch.tensor(classes, dtype=torch.long),
                "file": file, "slice": int(z), "view": view}

    def _augment(self, img, boxes):
        w = img.shape[-1]
        if self.rng.random() < 0.5:                                  # left-right flip
            img = img[..., ::-1]
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [1, 3]] = w - boxes[:, [3, 1]]
        dy, dx = self.rng.integers(-MAX_SHIFT, MAX_SHIFT + 1, size=2)
        img = np.roll(img, (int(dy), int(dx)), axis=(-2, -1))
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] += dy
            boxes[:, [1, 3]] += dx
        img = img * float(self.rng.uniform(0.9, 1.1)) + float(self.rng.normal(0, 0.02))
        return np.ascontiguousarray(img), boxes


def collate_slabs(samples):
    return {
        "image": torch.stack([s["image"] for s in samples]),
        "boxes": [s["boxes"] for s in samples],
        "box_classes": [s["box_classes"] for s in samples],
        "file": [s["file"] for s in samples],
        "slice": [s["slice"] for s in samples],
        "view": [s["view"] for s in samples],
    }
