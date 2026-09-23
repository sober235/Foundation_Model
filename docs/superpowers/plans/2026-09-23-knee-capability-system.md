# 膝侧 A/U/R 能力系统 · 实施计划（计划 1）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 SKM-TEA 膝关节上交付三条能力：解剖分割（已有 Dataset901）、病灶 3D 框加大类（新训 Dataset902）、病灶所在结构（查表加集合值），外加一条从 NIfTI 体积到三个输出的单命令入口，并按患者五折给出验收数字。

**Architecture:** 病灶检测走"框填成掩膜 → nnU-Net 3D 分割 → 连通域取框"，与已跑通的解剖 nnU-Net 同数据、同折、同训练器；评估直接读 nnU-Net 折末验证输出（标签图 + `--npz` 概率）；绑定复用 `anatobind/eval/lookup.py` 的类别限定查表并加集合值输出；入口用 `nnUNetv2_predict` 串起两个数据集的模型。

**Tech Stack:** nnU-Net v2（nvgen 环境，`nnunetv2` 已装）、nibabel、numpy、scipy.ndimage、matplotlib、pytest。

**Spec:** `docs/superpowers/specs/2026-09-23-aur-capability-system-design.md`

## Global Constraints

- 数据只从 `/data2/congcong/data/FM_data` 读；新产物只写 `/data2/congcong/data/FM_data/derived/`（nnU-Net 三个根见 `scripts/nnunet_env.sh`；入口输出 `derived/knee_infer/`）。`/data0` 已用 90%，不写数据。
- 解释器一律 `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python`；nnU-Net 命令前先 `source scripts/nnunet_env.sh`（它导出 `nnUNet_raw/preprocessed/results`、`nnUNet_n_proc_DA=8`、`PYTHONNOUSERSITE=1` 并把 nvgen 放进 PATH）。
- 测试一律 `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`（基线 311 passed，约 35 s；每个任务结束时全套必须全过）。
- 分支 `build/aur-system`（已从 main 开出）。提交作者用仓库本地身份（Congcong Liu），消息英文、句首大写，**不写任何 AI trailer**。
- GPU 只用 0、1、4、6 四张 A800（用户可改；zsh 数组从 1 起数，命令里写显式卡号，不用数组），每卡一个 nnU-Net 训练，`CUDA_VISIBLE_DEVICES` 钉死，后台用 `setsid nohup … &`，日志进 `logs/`（未跟踪）。CPU 任务 `nice -n 19`，总线程 ≤ 48。
- 标签号与坐标帧：nnU-Net 病灶标签 `1 cartilage_lesion / 2 meniscal_tear / 3 ligament_tear / 4 effusion`；查表类号 `dataset_v2.CLASSES`（`Meniscal Tear 0 / Cartilage Lesion 1 / Effusion 2 / Ligament Tear 3`）；两者的转换只在 `anatobind/nnunet/lesion_labels.py` 做。**本计划的新代码全部在导出帧 (X, Y, Z) = (256, 256, 160) 工作**（与 `seg.nii.gz`、`boxes.csv` 的 `x0..z1` 同帧），不用 `to_model_frame`。间距从 `seg.nii.gz` 头读，(X, Y, Z) 顺序约 (0.625, 0.625, 0.8) mm。
- 匹配：3D IoU ≥ 0.1，一对一（`anatobind.eval.matching.match`）。工作点：每卷假阳 ≤ 2 下灵敏度最高的分数阈值。**达标线（止损门）：干净视图、大类正确的灵敏度 ≥ 0.5 且每卷假阳 ≤ 2。Task 7 不过就停下来交用户，不做 Task 8–10。**
- 任何数字都要附可粘贴命令与原始输出（样板 `docs/verification/2026-09-08/REPORT.md`）；以伪标签或重叠率为参照的数字标 `NOT_EVIDENCE`。

---

### Task 1: 框填成掩膜的病灶标签图

**Files:**
- Create: `anatobind/nnunet/lesion_labels.py`
- Test: `tests/test_lesion_labels.py`

**Interfaces:**
- Consumes: `tests/synth.py` 的 `write_synthetic_export(root, scans, folds)`（DEFAULT_BOXES 四个框：cartilage x10:16 y10:20 z3:8、meniscal x32:40 y24:34 z5:9、effusion x46:56 y40:54 z12:20、ligament x20:26 y44:52 z13:17；seg 形状 (64, 64, 30)）。
- Produces:
  - `FAMILY_LABELS = {"Cartilage Lesion": 1, "Meniscal Tear": 2, "Ligament Tear": 3, "Effusion": 4}`
  - `LABELS = {"background": 0, "cartilage_lesion": 1, "meniscal_tear": 2, "ligament_tear": 3, "effusion": 4}`（nnU-Net dataset.json 用）
  - `FAMILY_OF_LABEL: dict[int, str]`（1 → "Cartilage Lesion" …）、`NAME_OF_LABEL: dict[int, str]`（1 → "cartilage_lesion" …）
  - `LOOKUP_CLASS_OF_FAMILY = dataset_v2.CLASSES`（同一对象，别复制）
  - `read_boxes_xyz(path) -> list[dict]`，每行 `{"ann_id": int, "supercategory": str, "layer": str, "tissue_id": int, "host_label": int | None, "host_side": str, "box": (x0, y0, z0, x1, y1, z1) 全 int}`
  - `box_volume(box) -> int`、`paint_order(rows) -> list[int]`、`boxes_to_label_map(rows, shape) -> np.uint8 (X, Y, Z)`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_lesion_labels.py
import numpy as np
from synth import DEFAULT_BOXES, write_synthetic_export

from anatobind.nnunet.lesion_labels import (
    FAMILY_LABELS, FAMILY_OF_LABEL, LABELS, NAME_OF_LABEL, box_volume, boxes_to_label_map, paint_order,
    read_boxes_xyz,
)


def _row(family, box, ann_id=1):
    return {"ann_id": ann_id, "supercategory": family, "layer": "in_seg", "tissue_id": 4, "host_label": 2,
            "host_side": "single", "box": box}


def test_label_tables_agree_with_each_other():
    assert LABELS == {"background": 0, "cartilage_lesion": 1, "meniscal_tear": 2, "ligament_tear": 3, "effusion": 4}
    assert FAMILY_OF_LABEL[FAMILY_LABELS["Effusion"]] == "Effusion"
    assert NAME_OF_LABEL == {1: "cartilage_lesion", 2: "meniscal_tear", 3: "ligament_tear", 4: "effusion"}


def test_disjoint_boxes_paint_their_own_voxels_and_nothing_else():
    rows = [_row("Cartilage Lesion", (0, 0, 0, 2, 3, 4)), _row("Meniscal Tear", (5, 5, 5, 7, 7, 7), 2)]
    lab = boxes_to_label_map(rows, (10, 10, 10))
    assert lab.dtype == np.uint8 and lab.shape == (10, 10, 10)
    assert (lab == 1).sum() == box_volume((0, 0, 0, 2, 3, 4)) == 24
    assert (lab == 2).sum() == 8 and (lab == 0).sum() == 1000 - 24 - 8


def test_a_small_box_inside_an_effusion_box_wins_the_overlap():
    big = _row("Effusion", (0, 0, 0, 10, 10, 10))
    small = _row("Cartilage Lesion", (2, 2, 2, 4, 4, 4), 2)
    lab = boxes_to_label_map([big, small], (10, 10, 10))
    assert (lab == 1).sum() == 8 and (lab == 4).sum() == 1000 - 8
    # painting order is independent of the input order
    assert np.array_equal(lab, boxes_to_label_map([small, big], (10, 10, 10)))


def test_between_two_non_effusion_boxes_the_smaller_one_wins():
    large = _row("Meniscal Tear", (0, 0, 0, 6, 6, 6))
    small = _row("Ligament Tear", (4, 4, 4, 8, 8, 8), 2)
    lab = boxes_to_label_map([large, small], (10, 10, 10))
    assert lab[5, 5, 5] == 3 and lab[1, 1, 1] == 2
    assert paint_order([small, large]) == [1, 0]


def test_effusion_is_painted_first_even_when_it_is_smaller():
    eff = _row("Effusion", (0, 0, 0, 2, 2, 2))
    tear = _row("Meniscal Tear", (0, 0, 0, 5, 5, 5), 2)
    assert paint_order([tear, eff]) == [1, 0]
    assert boxes_to_label_map([tear, eff], (6, 6, 6))[0, 0, 0] == 2


def test_boxes_are_clipped_to_the_volume_and_degenerate_boxes_paint_nothing():
    lab = boxes_to_label_map([_row("Cartilage Lesion", (8, 8, 8, 20, 20, 20)), _row("Effusion", (1, 1, 1, 1, 5, 5), 2)],
                             (10, 10, 10))
    assert (lab == 1).sum() == 8 and (lab == 4).sum() == 0


def test_read_boxes_xyz_returns_every_row_in_the_export_frame(tmp_path):
    write_synthetic_export(tmp_path, ["MTR_001"], {"MTR_001": 0})
    rows = read_boxes_xyz(tmp_path / "MTR_001" / "boxes.csv")
    assert [r["supercategory"] for r in rows] == [b["supercategory"] for b in DEFAULT_BOXES]
    assert rows[0]["box"] == (10, 10, 3, 16, 20, 8) and all(isinstance(v, int) for v in rows[0]["box"])
    assert rows[0]["host_label"] == 2 and rows[2]["host_label"] is None and rows[3]["layer"] == "ligament"
    lab = boxes_to_label_map(rows, (64, 64, 30))
    assert lab[12, 15, 5] == 1 and lab[35, 30, 7] == 2 and lab[22, 48, 15] == 3 and lab[50, 45, 15] == 4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_lesion_labels.py -q -p no:cacheprovider`
Expected: FAIL，`ModuleNotFoundError: No module named 'anatobind.nnunet.lesion_labels'`

- [ ] **Step 3: 写实现**

```python
# anatobind/nnunet/lesion_labels.py
"""Box-filled lesion label maps for the knee lesion detector (spec 2026-09-23 §3.1, decision N1/Q9).

Every annotated 3D box is painted into a (X, Y, Z) uint8 map with its family label, in the export
frame of derived/skmtea/m1r (the frame of seg.nii.gz and of the x0..z1 columns of boxes.csv).
Painting order decides the 3.3% of box voxels that lie in boxes of two families (measured
2026-09-23 on all 465 boxes): effusion boxes go first because they are huge (median 99 mL) and
box-shaped, then the rest from largest to smallest, so a smaller box overwrites a larger one.
"""
import csv

import numpy as np

from anatobind.train.dataset_v2 import CLASSES as LOOKUP_CLASS_OF_FAMILY  # noqa: F401  (single source of the lookup ids)

FAMILY_LABELS = {"Cartilage Lesion": 1, "Meniscal Tear": 2, "Ligament Tear": 3, "Effusion": 4}
LABELS = {"background": 0, "cartilage_lesion": 1, "meniscal_tear": 2, "ligament_tear": 3, "effusion": 4}
FAMILY_OF_LABEL = {v: k for k, v in FAMILY_LABELS.items()}
NAME_OF_LABEL = {v: k for k, v in LABELS.items() if v}
EFFUSION_FAMILY = "Effusion"


def read_boxes_xyz(path):
    """Every row of a boxes.csv with its box as ints in the export (X, Y, Z) frame."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            host = r["host_label"].strip()
            rows.append({
                "ann_id": int(r["ann_id"]), "supercategory": r["supercategory"], "layer": r["layer"],
                "tissue_id": int(r["tissue_id"]), "host_label": int(host) if host else None,
                "host_side": r["host_side"],
                "box": tuple(int(r[k]) for k in ("x0", "y0", "z0", "x1", "y1", "z1")),
            })
    return rows


def box_volume(box):
    x0, y0, z0, x1, y1, z1 = box
    return max(x1 - x0, 0) * max(y1 - y0, 0) * max(z1 - z0, 0)


def paint_order(rows):
    """Row indices in painting order: effusion first, then the others from largest to smallest."""
    return sorted(range(len(rows)),
                  key=lambda i: (rows[i]["supercategory"] != EFFUSION_FAMILY, -box_volume(rows[i]["box"])))


def _clip(box, shape):
    x0, y0, z0, x1, y1, z1 = box
    return (max(x0, 0), max(y0, 0), max(z0, 0), min(x1, shape[0]), min(y1, shape[1]), min(z1, shape[2]))


def boxes_to_label_map(rows, shape):
    lab = np.zeros(tuple(int(s) for s in shape), np.uint8)
    for i in paint_order(rows):
        x0, y0, z0, x1, y1, z1 = _clip(rows[i]["box"], lab.shape)
        if x1 > x0 and y1 > y0 and z1 > z0:
            lab[x0:x1, y0:y1, z0:z1] = FAMILY_LABELS[rows[i]["supercategory"]]
    return lab
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_lesion_labels.py -q -p no:cacheprovider`
Expected: `7 passed`

- [ ] **Step 5: 提交**

```bash
git add anatobind/nnunet/lesion_labels.py tests/test_lesion_labels.py
git commit -m "Data engine: box-filled lesion label maps with effusion-first, smallest-wins painting"
```

---

### Task 2: Dataset902 数据集构建、预处理与五折训练

**Files:**
- Create: `anatobind/nnunet/prepare_lesion.py`
- Create: `scripts/nnunet_prepare_lesion.py`
- Test: `tests/test_nnunet_prepare_lesion.py`

**Interfaces:**
- Consumes: `anatobind.nnunet.prepare` 的 `train_cases(scan)`、`val_cases(scan)`、`view_of_case(case)`、`make_splits(folds, n_folds=5)`、`_link(src, dst)`、`TRAINER`、`TRAINER_DIR`；`anatobind.train.cache.VIEW_FILES`；`anatobind.train.dataset.list_ready_scans(root)`；Task 1 的 `read_boxes_xyz`、`boxes_to_label_map`、`LABELS`。
- Produces:
  - `DATASET_ID = 902`、`DATASET_NAME = "Dataset902_SKMTEAlesion"`
  - `build_raw(export_root, raw_root, scans) -> int`（训练例数）
  - `write_splits(preprocessed_root, folds, n_folds=5)`
  - `validation_path(results_root, fold, scan, view) -> Path`（`.nii.gz`）与 `validation_npz_path(...) -> Path`（同名 `.npz`）
  - 磁盘上：`$nnUNet_raw/Dataset902_SKMTEAlesion/{imagesTr,labelsTr,dataset.json}`，`$nnUNet_preprocessed/Dataset902_SKMTEAlesion/splits_final.json`，`$nnUNet_results/Dataset902_SKMTEAlesion/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_{0..4}/{checkpoint_final.pth,validation/<case>.nii.gz,validation/<case>.npz}`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_nnunet_prepare_lesion.py
import json
import os

import nibabel as nib
import numpy as np
from synth import write_synthetic_export

from anatobind.nnunet.lesion_labels import LABELS
from anatobind.nnunet.prepare import make_splits
from anatobind.nnunet.prepare_lesion import (
    DATASET_NAME, build_raw, validation_npz_path, validation_path, write_splits,
)

FOLDS = {"MTR_001": 0, "MTR_002": 1}


def test_the_raw_dataset_links_images_and_shares_one_label_map_per_scan(tmp_path):
    exp = tmp_path / "m1r"
    write_synthetic_export(exp, list(FOLDS), FOLDS)
    assert build_raw(exp, tmp_path / "raw", list(FOLDS)) == 24
    base = tmp_path / "raw" / DATASET_NAME
    assert len(list((base / "imagesTr").iterdir())) == 24 and len(list((base / "labelsTr").iterdir())) == 24
    ino = lambda p: os.stat(p).st_ino  # noqa: E731
    assert ino(base / "imagesTr/MTR_001_us16_0000.nii.gz") == ino(exp / "MTR_001/image_us16_e1.nii.gz")
    assert ino(base / "labelsTr/MTR_001_clean0.nii.gz") == ino(base / "labelsTr/MTR_001_noise_q3.nii.gz")
    assert ino(base / "labelsTr/MTR_001_clean0.nii.gz") != ino(base / "labelsTr/MTR_002_clean0.nii.gz")
    lab = nib.load(str(base / "labelsTr/MTR_002_us4.nii.gz"))
    seg = nib.load(str(exp / "MTR_002/seg.nii.gz"))
    assert lab.shape == seg.shape and np.allclose(lab.affine, seg.affine) and lab.get_data_dtype() == np.uint8
    arr = np.asanyarray(lab.dataobj)
    assert arr[12, 15, 5] == 1 and arr[35, 30, 7] == 2 and arr[22, 48, 15] == 3 and arr[50, 45, 15] == 4
    assert arr[0, 0, 0] == 0
    ds = json.loads((base / "dataset.json").read_text())
    assert ds == {"channel_names": {"0": "qDESS_E1"}, "labels": LABELS, "numTraining": 24, "file_ending": ".nii.gz"}


def test_splits_and_validation_paths_mirror_dataset901(tmp_path):
    write_splits(tmp_path / "pre", FOLDS, n_folds=2)
    assert json.loads((tmp_path / "pre" / DATASET_NAME / "splits_final.json").read_text()) == make_splits(FOLDS, 2)
    p = validation_path(tmp_path, 3, "MTR_001", "clean")
    assert p == tmp_path / DATASET_NAME / "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres" / "fold_3" / "validation" / "MTR_001_clean0.nii.gz"
    assert validation_npz_path(tmp_path, 0, "MTR_001", "us8").name == "MTR_001_us8.npz"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_nnunet_prepare_lesion.py -q -p no:cacheprovider`
Expected: FAIL，`ModuleNotFoundError: No module named 'anatobind.nnunet.prepare_lesion'`

- [ ] **Step 3: 写实现**

```python
# anatobind/nnunet/prepare_lesion.py
"""nnU-Net v2 dataset for the knee lesion detector (spec 2026-09-23 §3.1, decision N1).

Same images, views, folds and case names as Dataset901 (anatobind/nnunet/prepare.py): each scan
contributes its six degraded views once and its clean view six times, all hard-linked. The label of
every case of a scan is the same box-filled map (Task 1), written once and hard-linked eleven times.
"""
import json
from pathlib import Path

import nibabel as nib

from anatobind.nnunet.lesion_labels import LABELS, boxes_to_label_map, read_boxes_xyz
from anatobind.nnunet.prepare import TRAINER_DIR, _link, make_splits, train_cases, view_of_case
from anatobind.train.cache import VIEW_FILES

DATASET_ID = 902
DATASET_NAME = f"Dataset{DATASET_ID}_SKMTEAlesion"


def write_label_map(export_dir, out_path):
    seg = nib.load(str(Path(export_dir) / "seg.nii.gz"))
    lab = boxes_to_label_map(read_boxes_xyz(Path(export_dir) / "boxes.csv"), seg.shape)
    img = nib.Nifti1Image(lab, seg.affine)
    img.set_data_dtype("uint8")
    img.header.set_xyzt_units("mm")
    nib.save(img, str(out_path))


def build_raw(export_root, raw_root, scans):
    base = Path(raw_root) / DATASET_NAME
    images, labels = base / "imagesTr", base / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    n = 0
    for scan in scans:
        src = Path(export_root) / scan
        cases = train_cases(scan)
        first = labels / f"{cases[0]}.nii.gz"
        if not first.exists():
            write_label_map(src, first)
        for case in cases:
            _link(src / VIEW_FILES[view_of_case(case)], images / f"{case}_0000.nii.gz")
            _link(first, labels / f"{case}.nii.gz")
            n += 1
    meta = {"channel_names": {"0": "qDESS_E1"}, "labels": LABELS, "numTraining": n, "file_ending": ".nii.gz"}
    (base / "dataset.json").write_text(json.dumps(meta, indent=1))
    return n


def write_splits(preprocessed_root, folds, n_folds=5):
    d = Path(preprocessed_root) / DATASET_NAME
    d.mkdir(parents=True, exist_ok=True)
    (d / "splits_final.json").write_text(json.dumps(make_splits(folds, n_folds), indent=1))


def _case(scan, view):
    return f"{scan}_clean0" if view == "clean" else f"{scan}_{view}"


def validation_path(results_root, fold, scan, view):
    return Path(results_root) / DATASET_NAME / TRAINER_DIR / f"fold_{fold}" / "validation" / f"{_case(scan, view)}.nii.gz"


def validation_npz_path(results_root, fold, scan, view):
    return validation_path(results_root, fold, scan, view).with_suffix("").with_suffix(".npz")
```

```python
#!/usr/bin/env python
# scripts/nnunet_prepare_lesion.py
"""Build the lesion nnU-Net raw dataset (--stage raw) or write its fold splits (--stage splits).

  source scripts/nnunet_env.sh
  PYTHONPATH=. python scripts/nnunet_prepare_lesion.py --stage raw
  nice -n 19 nnUNetv2_plan_and_preprocess -d 902 -c 3d_fullres --verify_dataset_integrity -np 4
  PYTHONPATH=. python scripts/nnunet_prepare_lesion.py --stage splits
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.nnunet.prepare_lesion import build_raw, write_splits  # noqa: E402
from anatobind.train.dataset import list_ready_scans  # noqa: E402

M1R = Path("/data2/congcong/data/FM_data/derived/skmtea/m1r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("raw", "splits"), required=True)
    a = ap.parse_args()
    if a.stage == "raw":
        print(f"{build_raw(M1R, os.environ['nnUNet_raw'], list_ready_scans(M1R))} training cases")
    else:
        write_splits(os.environ["nnUNet_preprocessed"], json.loads((M1R / "splits.json").read_text())["folds"])
        print("splits_final.json written")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过，再跑全套**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_nnunet_prepare_lesion.py -q -p no:cacheprovider`
Expected: `2 passed`
Run: 全套测试。Expected: `320 passed`（311 + Task 1 的 7 + 本任务 2）

- [ ] **Step 5: 提交代码**

```bash
git add anatobind/nnunet/prepare_lesion.py scripts/nnunet_prepare_lesion.py tests/test_nnunet_prepare_lesion.py
git commit -m "nnU-Net: Dataset902 lesion dataset builder sharing Dataset901 views, folds and case names"
```

- [ ] **Step 6: 建原始数据集并预处理（CPU，约 1–2 h）**

```bash
cd /data0/congcong/code/Project_Doing/foundation_model
source scripts/nnunet_env.sh
PYTHONPATH=. python scripts/nnunet_prepare_lesion.py --stage raw          # 期望打印 "1860 training cases"
ls $nnUNet_raw/Dataset902_SKMTEAlesion/labelsTr | wc -l                   # 1860
setsid nohup nice -n 19 nnUNetv2_plan_and_preprocess -d 902 -c 3d_fullres --verify_dataset_integrity -np 4 \
    > logs/nnunet902_preprocess.log 2>&1 &
```

结束判据：`tail -2 logs/nnunet902_preprocess.log` 出现 `Preprocessing cases: 100%`，且 `ls $nnUNet_preprocessed/Dataset902_SKMTEAlesion/nnUNetPlans_3d_fullres | wc -l` = 5580（与 Dataset901 相同）。然后：

```bash
PYTHONPATH=. python scripts/nnunet_prepare_lesion.py --stage splits      # 期望 "splits_final.json written"
python - <<'EOF'
import json; s=json.load(open('/data2/congcong/data/FM_data/derived/nnunet/preprocessed/Dataset902_SKMTEAlesion/splits_final.json'))
print(len(s), [ (len(f['train']), len(f['val'])) for f in s ])   # 5 [(1488, 217)] x5
EOF
grep -o '"patch_size": \[[^]]*\]' $nnUNet_preprocessed/Dataset902_SKMTEAlesion/nnUNetPlans.json | head -2   # 期望含 [96, 160, 160]
```

- [ ] **Step 7: 启动 fold 0–3 训练（四卡并行，每折 4–7 h）**

```bash
cd /data0/congcong/code/Project_Doing/foundation_model
source scripts/nnunet_env.sh
# zsh arrays are 1-indexed, so the fold -> GPU map is written out explicitly (fold 0 lost its GPU on 2026-09-23 with ${GPUS[$f]})
for pair in 0:6 1:0 2:1 3:4; do f=${pair%%:*}; g=${pair##*:}
  CUDA_VISIBLE_DEVICES=$g setsid nohup nnUNetv2_train 902 3d_fullres $f -tr nnUNetTrainer_250epochs --npz \
      > logs/nnunet902_fold$f.log 2>&1 < /dev/null &
done
sleep 120; nvidia-smi --query-compute-apps=pid,used_memory --format=csv    # 期望 4 个进程
```

- [ ] **Step 8: 烟雾门（决定 N6 的第一道汇报）**

20 轮后（约 25 分钟）检查 fold 0 的伪 Dice 是否在动：

```bash
grep -E 'Pseudo dice' logs/nnunet902_fold0.log | sed -n '1p;5p;10p;20p'
```

判据：第 20 轮的四个伪 Dice 至少有两个 > 0.05，且高于第 1 轮。全为 0 或 NaN → 停下来，把日志前 60 行贴给用户。通过 → 向用户报一段话（命令 + 上面四行原始输出），继续。

- [ ] **Step 9: fold 4 与收尾**

任一折日志末尾出现 `Training done.`（`grep -c 'Training done' logs/nnunet902_fold*.log`）后，在空出的卡上启动 fold 4：

```bash
source scripts/nnunet_env.sh
CUDA_VISIBLE_DEVICES=0 setsid nohup nnUNetv2_train 902 3d_fullres 4 -tr nnUNetTrainer_250epochs --npz \
    > logs/nnunet902_fold4.log 2>&1 &
```

全部结束判据（Task 7 的前提）：

```bash
for f in 0 1 2 3 4; do d=$nnUNet_results/Dataset902_SKMTEAlesion/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_$f/validation; echo "fold $f: $(ls $d/*.nii.gz | wc -l) nii, $(ls $d/*.npz | wc -l) npz"; done
# 期望每折 217 nii、217 npz（31 扫描 × 7 视图）
```

训练进行期间做 Task 3–6（不占 GPU）。

---

### Task 3: 从病灶标签图取 3D 框，并读 nnU-Net 的概率

**Files:**
- Create: `anatobind/eval/lesion_boxes.py`
- Test: `tests/test_lesion_boxes.py`

**Interfaces:**
- Consumes: Task 1 的 `FAMILY_OF_LABEL`。
- Produces:
  - `MIN_VOXELS = 27`、`STRUCTURE`（26 邻域）
  - `decode_boxes(label_map, probs=None, min_voxels=MIN_VOXELS) -> list[dict]`，每项 `{"family": str, "label": int, "box": (x0, y0, z0, x1, y1, z1) int, "score": float, "n_voxels": int}`，按 score 降序。`probs` 为 (C, X, Y, Z)，score = 该族概率在连通域内的均值；无 `probs` 时 score = `n / (n + min_voxels)`（只给单元测试与无概率的退路用，评估脚本必须传概率）。
  - `load_nnunet_probabilities(npz_path, label_map_xyz) -> np.float32 (C, X, Y, Z)`：把 nnU-Net `--npz` 的 `probabilities`（SimpleITK 轴序 (C, Z, Y, X)）转到导出帧，并断言 argmax 与标签图在 ≥ 99% 体素上一致，否则抛 `ValueError`。
  - `load_label_map(nii_path) -> np.uint8 (X, Y, Z)`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_lesion_boxes.py
import nibabel as nib
import numpy as np
import pytest

from anatobind.eval.lesion_boxes import MIN_VOXELS, decode_boxes, load_label_map, load_nnunet_probabilities


def _map():
    m = np.zeros((20, 20, 20), np.uint8)
    m[2:6, 2:6, 2:6] = 1        # cartilage blob A, 64 voxels
    m[12:15, 12:15, 12:15] = 1  # cartilage blob B, 27 voxels
    m[0:10, 14:20, 0:5] = 4     # effusion, 300 voxels
    m[18, 18, 18] = 2           # a 1-voxel meniscal speck: below MIN_VOXELS
    return m


def _onehot(m, n_classes=5):
    p = np.zeros((n_classes,) + m.shape, np.float32)
    for c in range(n_classes):
        p[c] = m == c
    return p


def test_one_box_per_component_with_family_label_and_corners():
    dets = decode_boxes(_map(), _onehot(_map()))
    assert len(dets) == 3 and sorted(d["n_voxels"] for d in dets) == [27, 64, 300]
    fams = sorted((d["family"], d["box"]) for d in dets)
    assert fams == [("Cartilage Lesion", (2, 2, 2, 6, 6, 6)), ("Cartilage Lesion", (12, 12, 12, 15, 15, 15)),
                    ("Effusion", (0, 14, 0, 10, 20, 5))]
    assert all(d["score"] == 1.0 for d in dets) and all(d["label"] in (1, 4) for d in dets)


def test_the_score_is_the_mean_family_probability_inside_the_component():
    m = np.zeros((8, 8, 8), np.uint8)
    m[0:4, 0:4, 0:4] = 2
    p = np.zeros((5, 8, 8, 8), np.float32)
    p[2, 0:4, 0:4, 0:4] = 0.6
    p[2, 0:2, 0:4, 0:4] = 0.8   # half the voxels at 0.8, half at 0.6
    assert decode_boxes(m, p)[0]["score"] == pytest.approx(0.7)


def test_components_below_min_voxels_are_dropped_and_the_fallback_score_grows_with_size():
    dets = decode_boxes(_map())
    assert all(d["n_voxels"] >= MIN_VOXELS for d in dets) and len(dets) == 3
    small = next(d for d in dets if d["n_voxels"] == 27)
    big = next(d for d in dets if d["n_voxels"] == 300)
    assert small["score"] == pytest.approx(0.5) and big["score"] > small["score"]
    assert dets[0]["score"] >= dets[-1]["score"]


def test_decoding_an_empty_map_gives_no_boxes():
    assert decode_boxes(np.zeros((5, 5, 5), np.uint8)) == []


def test_nnunet_probabilities_come_back_in_the_export_frame(tmp_path):
    m = _map()
    p_zyx = np.ascontiguousarray(_onehot(m).transpose(0, 3, 2, 1))          # what nnU-Net writes
    np.savez_compressed(tmp_path / "case.npz", probabilities=p_zyx.astype(np.float16))
    p = load_nnunet_probabilities(tmp_path / "case.npz", m)
    assert p.shape == (5, 20, 20, 20) and p.dtype == np.float32 and np.array_equal(p.argmax(0), m)
    with pytest.raises(ValueError):
        load_nnunet_probabilities(tmp_path / "case.npz", np.roll(m, 7, axis=0))


def test_load_label_map_keeps_the_nifti_array_order(tmp_path):
    m = _map()
    nib.save(nib.Nifti1Image(m, np.diag([0.625, 0.625, 0.8, 1.0])), str(tmp_path / "seg.nii.gz"))
    assert np.array_equal(load_label_map(tmp_path / "seg.nii.gz"), m)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_lesion_boxes.py -q -p no:cacheprovider`
Expected: FAIL，`ModuleNotFoundError: No module named 'anatobind.eval.lesion_boxes'`

- [ ] **Step 3: 写实现**

```python
# anatobind/eval/lesion_boxes.py
"""Predicted lesion boxes from an nnU-Net lesion label map (spec 2026-09-23 §3.3, §4.1).

One box per 26-connected component of each family, in the export (X, Y, Z) frame. The score is the
mean predicted probability of that family inside the component (nnU-Net --npz / --save_probabilities);
without probabilities the score is n / (n + MIN_VOXELS), which only orders components by size.
"""
import nibabel as nib
import numpy as np
from scipy import ndimage

from anatobind.nnunet.lesion_labels import FAMILY_OF_LABEL

STRUCTURE = np.ones((3, 3, 3), bool)     # 26-connectivity
MIN_VOXELS = 27                          # a 3x3x3 block; smaller blobs are decoder noise
AGREEMENT_MIN = 0.99


def decode_boxes(label_map, probs=None, min_voxels=MIN_VOXELS):
    out = []
    for label, family in FAMILY_OF_LABEL.items():
        comp, n = ndimage.label(label_map == label, structure=STRUCTURE)
        if n == 0:
            continue
        for k, sl in enumerate(ndimage.find_objects(comp), start=1):
            if sl is None:
                continue
            mask = comp[sl] == k
            nv = int(mask.sum())
            if nv < min_voxels:
                continue
            score = float(probs[label][sl][mask].mean()) if probs is not None else nv / (nv + min_voxels)
            out.append({"family": family, "label": int(label),
                        "box": (sl[0].start, sl[1].start, sl[2].start, sl[0].stop, sl[1].stop, sl[2].stop),
                        "score": float(score), "n_voxels": nv})
    return sorted(out, key=lambda d: -d["score"])


def load_label_map(nii_path):
    return np.ascontiguousarray(np.asanyarray(nib.load(str(nii_path)).dataobj)).astype(np.uint8)


def load_nnunet_probabilities(npz_path, label_map_xyz):
    """nnU-Net stores 'probabilities' as (C, Z, Y, X) (SimpleITK array order); return (C, X, Y, Z)."""
    with np.load(str(npz_path)) as z:
        p = z["probabilities"]
    p = np.ascontiguousarray(p.transpose(0, 3, 2, 1)).astype(np.float32)
    agree = float((p.argmax(0) == label_map_xyz).mean())
    if agree < AGREEMENT_MIN:
        raise ValueError(f"{npz_path}: probabilities disagree with the label map on {1 - agree:.1%} of voxels; axis order?")
    return p
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_lesion_boxes.py -q -p no:cacheprovider`
Expected: `6 passed`

- [ ] **Step 5: 提交**

```bash
git add anatobind/eval/lesion_boxes.py tests/test_lesion_boxes.py
git commit -m "Eval: lesion boxes from connected components with probability scores and nnU-Net npz loader"
```

---

### Task 4: 检测指标——灵敏度、每卷假阳、工作点

**Files:**
- Create: `anatobind/eval/detection_metrics.py`
- Test: `tests/test_detection_metrics.py`

**Interfaces:**
- Consumes: `anatobind.eval.matching.match(gt_boxes, pred_boxes, thr)`（(N,6)/(M,6) 角点框，返回 `{gt: pred}`；对帧无要求，两边同帧即可）。
- Produces:
  - `IOU = 0.1`、`FP_MAX = 2.0`、`GATE_SENSITIVITY = 0.5`、`THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)`
  - 扫描记录格式 `{"scan": str, "gt": [{"box": tuple, "family": str, ...}], "dets": [Task 3 的 dict]}`
  - `match_scan(gt, dets, iou=IOU) -> dict[int, int]`（gt 序号 → det 序号）
  - `sweep(scans, thresholds=THRESHOLDS) -> list[dict]`，每项 `{"thr", "n_gt", "n_hit", "n_hit_family", "n_fp", "n_scans", "sensitivity", "sensitivity_family", "fp_per_scan"}`
  - `operating_point(rows, fp_max=FP_MAX) -> dict | None`（fp_per_scan ≤ fp_max 中 `sensitivity_family` 最大者，并列取阈值最高者）
  - `per_family(scans, thr, iou=IOU) -> dict[str, dict]`，每族 `{"n_gt", "n_hit", "n_hit_family", "sensitivity", "sensitivity_family"}`
  - `gate(rows) -> dict`：`{"pass": bool, "thr", "sensitivity_family", "fp_per_scan"}`（工作点不存在则 `pass=False, thr=None`）

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_detection_metrics.py
import pytest

from anatobind.eval.detection_metrics import (
    FP_MAX, GATE_SENSITIVITY, gate, match_scan, operating_point, per_family, sweep,
)


def _det(box, family, score):
    return {"family": family, "box": box, "score": score, "label": 1, "n_voxels": 100}


def _gt(box, family):
    return {"box": box, "family": family}


A, B = (0, 0, 0, 10, 10, 10), (20, 20, 20, 30, 30, 30)
SCANS = [
    {"scan": "s1", "gt": [_gt(A, "Cartilage Lesion"), _gt(B, "Effusion")],
     "dets": [_det(A, "Cartilage Lesion", 0.9), _det(B, "Meniscal Tear", 0.6), _det((40, 40, 40, 45, 45, 45), "Effusion", 0.3)]},
    {"scan": "s2", "gt": [_gt(A, "Meniscal Tear")], "dets": [_det((0, 0, 0, 10, 10, 3), "Meniscal Tear", 0.5)]},
    {"scan": "s3", "gt": [], "dets": [_det(B, "Effusion", 0.2)]},
]


def test_matching_is_one_to_one_at_iou_0_1_regardless_of_family():
    assert match_scan(SCANS[0]["gt"], SCANS[0]["dets"]) == {0: 0, 1: 1}
    assert match_scan(SCANS[1]["gt"], SCANS[1]["dets"]) == {0: 0}      # IoU 0.3 >= 0.1
    assert match_scan([], SCANS[2]["dets"]) == {}


def test_sweep_counts_hits_family_hits_and_false_positives_per_threshold():
    rows = sweep(SCANS, thresholds=[0.1, 0.55, 0.95])
    r = rows[0]
    assert (r["n_gt"], r["n_hit"], r["n_hit_family"], r["n_fp"], r["n_scans"]) == (3, 3, 2, 2, 3)
    assert r["sensitivity"] == 1.0 and r["sensitivity_family"] == pytest.approx(2 / 3) and r["fp_per_scan"] == pytest.approx(2 / 3)
    r = rows[1]   # only scores >= 0.55 survive: A-hit (0.9, right family), B-hit (0.6, wrong family)
    assert (r["n_hit"], r["n_hit_family"], r["n_fp"]) == (2, 1, 0)
    assert rows[2]["n_hit"] == 0 and rows[2]["sensitivity"] == 0.0


def test_operating_point_is_the_best_family_sensitivity_under_the_fp_budget():
    rows = [{"thr": 0.1, "sensitivity_family": 0.9, "fp_per_scan": 3.0},
            {"thr": 0.3, "sensitivity_family": 0.7, "fp_per_scan": 1.5},
            {"thr": 0.5, "sensitivity_family": 0.7, "fp_per_scan": 0.5},
            {"thr": 0.9, "sensitivity_family": 0.2, "fp_per_scan": 0.0}]
    assert operating_point(rows)["thr"] == 0.5
    assert operating_point(rows, fp_max=0.1)["thr"] == 0.9
    assert operating_point([{"thr": 0.1, "sensitivity_family": 1.0, "fp_per_scan": 9.0}]) is None


def test_per_family_reports_each_family_separately():
    f = per_family(SCANS, 0.1)
    assert f["Cartilage Lesion"] == {"n_gt": 1, "n_hit": 1, "n_hit_family": 1, "sensitivity": 1.0, "sensitivity_family": 1.0}
    assert f["Effusion"]["n_hit"] == 1 and f["Effusion"]["n_hit_family"] == 0
    assert f["Meniscal Tear"]["sensitivity_family"] == 1.0 and "Ligament Tear" not in f


def test_gate_reads_the_operating_point():
    assert FP_MAX == 2.0 and GATE_SENSITIVITY == 0.5
    g = gate(sweep(SCANS, thresholds=[0.1, 0.55]))
    assert g == {"pass": True, "thr": 0.1, "sensitivity_family": pytest.approx(2 / 3), "fp_per_scan": pytest.approx(2 / 3)}
    assert gate([{"thr": 0.1, "sensitivity_family": 0.4, "fp_per_scan": 0.0}])["pass"] is False
    assert gate([])["pass"] is False and gate([])["thr"] is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_detection_metrics.py -q -p no:cacheprovider`
Expected: FAIL，`ModuleNotFoundError: No module named 'anatobind.eval.detection_metrics'`

- [ ] **Step 3: 写实现**

```python
# anatobind/eval/detection_metrics.py
"""Sensitivity and false positives per scan for 3D lesion boxes (spec 2026-09-23 §4.1).

Matching is one-to-one on 3D IoU >= IOU (anatobind.eval.matching.match), family-agnostic; a hit
whose predicted family differs from the truth still counts as localised, and 'sensitivity_family'
counts only hits with the right family. The operating point is the score threshold with the best
family sensitivity among those with <= FP_MAX false positives per scan; the gate reads it.
"""
import numpy as np

from anatobind.eval.matching import match

IOU = 0.1
FP_MAX = 2.0
GATE_SENSITIVITY = 0.5
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.05, 1.0, 0.05), 2))


def match_scan(gt, dets, iou=IOU):
    g = np.array([r["box"] for r in gt], float).reshape(-1, 6)
    p = np.array([d["box"] for d in dets], float).reshape(-1, 6)
    return match(g, p, iou)


def _count(scans, thr, iou):
    n_gt = n_hit = n_fam = n_fp = 0
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        pairs = match_scan(s["gt"], dets, iou)
        n_gt += len(s["gt"])
        n_hit += len(pairs)
        n_fam += sum(1 for g, p in pairs.items() if dets[p]["family"] == s["gt"][g]["family"])
        n_fp += len(dets) - len(pairs)
    return n_gt, n_hit, n_fam, n_fp


def _rate(a, b):
    return a / b if b else 0.0


def sweep(scans, thresholds=THRESHOLDS, iou=IOU):
    rows = []
    for thr in thresholds:
        n_gt, n_hit, n_fam, n_fp = _count(scans, thr, iou)
        rows.append({"thr": float(thr), "n_gt": n_gt, "n_hit": n_hit, "n_hit_family": n_fam, "n_fp": n_fp,
                     "n_scans": len(scans), "sensitivity": _rate(n_hit, n_gt),
                     "sensitivity_family": _rate(n_fam, n_gt), "fp_per_scan": _rate(n_fp, len(scans))})
    return rows


def operating_point(rows, fp_max=FP_MAX):
    ok = [r for r in rows if r["fp_per_scan"] <= fp_max]
    return max(ok, key=lambda r: (r["sensitivity_family"], r["thr"])) if ok else None


def per_family(scans, thr, iou=IOU):
    out = {}
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        pairs = match_scan(s["gt"], dets, iou)
        for g, r in enumerate(s["gt"]):
            f = out.setdefault(r["family"], {"n_gt": 0, "n_hit": 0, "n_hit_family": 0})
            f["n_gt"] += 1
            if g in pairs:
                f["n_hit"] += 1
                f["n_hit_family"] += int(dets[pairs[g]]["family"] == r["family"])
    for f in out.values():
        f["sensitivity"] = _rate(f["n_hit"], f["n_gt"])
        f["sensitivity_family"] = _rate(f["n_hit_family"], f["n_gt"])
    return out


def gate(rows, fp_max=FP_MAX, sensitivity_min=GATE_SENSITIVITY):
    op = operating_point(rows, fp_max)
    if op is None:
        return {"pass": False, "thr": None, "sensitivity_family": 0.0, "fp_per_scan": None}
    return {"pass": bool(op["sensitivity_family"] >= sensitivity_min), "thr": op["thr"],
            "sensitivity_family": op["sensitivity_family"], "fp_per_scan": op["fp_per_scan"]}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_detection_metrics.py -q -p no:cacheprovider`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
git add anatobind/eval/detection_metrics.py tests/test_detection_metrics.py
git commit -m "Eval: 3D detection sensitivity, false positives per scan, operating point and gate"
```

---

### Task 5: 查表加集合值输出与侧别

**Files:**
- Modify: `anatobind/eval/lookup.py`（在文件末尾追加，不改已有函数）
- Test: `tests/test_eval_lookup_matching.py`（追加测试）

**Interfaces:**
- Consumes: 已有 `LabelIndex`、`clip_box`、`b0_host`、`CANDIDATES`、`EFFUSION`、`LIGAMENT`。
- Produces:
  - `SIDE_OF_LABEL = {1: "single", 2: "single", 3: "medial", 4: "lateral", 5: "medial", 6: "lateral"}`
  - `HOST_NAMES = {1: "patellar_cartilage", 2: "femoral_cartilage", 3: "tibial_cartilage_medial", 4: "tibial_cartilage_lateral", 5: "meniscus_medial", 6: "meniscus_lateral"}`
  - `host_fractions(index, box, cls) -> dict[int, float]`：类别限定候选中每个在图里存在的结构在框内的体素占比；积液、韧带 → `{}`。
  - `describe_host(index, box, cls) -> dict`：`{"host_label": int | None, "host_name": str, "side": str, "host_fractions": dict}`，`host_name` 为 `none` / `unknown` / 结构名 / `""`（候选缺席时）。

- [ ] **Step 1: 追加失败的测试**

在 `tests/test_eval_lookup_matching.py` 末尾追加：

```python
from anatobind.eval.lookup import HOST_NAMES, SIDE_OF_LABEL, describe_host, host_fractions  # noqa: E402


def test_host_fractions_cover_only_the_class_candidates_present_in_the_map():
    idx = LabelIndex(_map(), SP)
    f = host_fractions(idx, (3, 2, 2, 17, 8, 8), 1)      # cartilage: femoral z 3-4 (2 of 14 slices) and patellar z 16
    assert set(f) == {1, 2} and f[2] == pytest.approx(2 * 36 / (14 * 36)) and f[1] == pytest.approx(36 / (14 * 36))
    assert host_fractions(idx, (10, 10, 10, 14, 11, 11), 0) == {5: 0.0, 6: 0.0}
    assert host_fractions(idx, (0, 0, 0, 5, 5, 5), 2) == {} and host_fractions(idx, (0, 0, 0, 5, 5, 5), 3) == {}


def test_describe_host_agrees_with_b0_host_and_names_the_side():
    idx = LabelIndex(_map(), SP)
    d = describe_host(idx, (10, 2, 2, 14, 8, 8), 0)
    assert d["host_label"] == b0_host(idx, (10, 2, 2, 14, 8, 8), 0) == 5
    assert d["host_name"] == "meniscus_medial" and d["side"] == "medial" and d["host_fractions"][5] == 1.0
    assert describe_host(idx, (0, 0, 0, 5, 5, 5), 2) == {"host_label": None, "host_name": "none", "side": "-", "host_fractions": {}}
    assert describe_host(idx, (0, 0, 0, 5, 5, 5), 3)["host_name"] == "unknown"
    empty = np.zeros((10, 10, 10), np.uint8)
    assert describe_host(LabelIndex(empty, SP), (0, 0, 0, 3, 3, 3), 0) == {"host_label": None, "host_name": "", "side": "-", "host_fractions": {}}
    assert SIDE_OF_LABEL[3] == "medial" and SIDE_OF_LABEL[6] == "lateral" and HOST_NAMES[2] == "femoral_cartilage"
```

同时把文件顶部的 `import numpy as np` 下面加一行 `import pytest`。

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_eval_lookup_matching.py -q -p no:cacheprovider`
Expected: FAIL，`ImportError: cannot import name 'HOST_NAMES'`

- [ ] **Step 3: 在 `anatobind/eval/lookup.py` 末尾追加**

```python

# --- set-valued host (spec 2026-09-23 §3.3, decision N4) ------------------------------------------
SIDE_OF_LABEL = {1: "single", 2: "single", 3: "medial", 4: "lateral", 5: "medial", 6: "lateral"}
HOST_NAMES = {1: "patellar_cartilage", 2: "femoral_cartilage", 3: "tibial_cartilage_medial",
              4: "tibial_cartilage_lateral", 5: "meniscus_medial", 6: "meniscus_lateral"}


def host_fractions(index, box, cls):
    """Fraction of the box's voxels inside each class candidate present in the map."""
    if cls in (EFFUSION, LIGAMENT):
        return {}
    b = clip_box(box, index.label_map.shape)
    sub = index.label_map[b[0]:b[3], b[1]:b[4], b[2]:b[5]]
    vol = max(sub.size, 1)
    return {int(label): float((sub == label).sum()) / vol for label in CANDIDATES[cls] if index.present(label)}


def describe_host(index, box, cls):
    host = b0_host(index, box, cls)
    fractions = host_fractions(index, box, cls)
    if host in (NONE, UNKNOWN):
        return {"host_label": None, "host_name": host, "side": "-", "host_fractions": fractions}
    if host is None:
        return {"host_label": None, "host_name": "", "side": "-", "host_fractions": fractions}
    return {"host_label": int(host), "host_name": HOST_NAMES[int(host)], "side": SIDE_OF_LABEL[int(host)],
            "host_fractions": fractions}
```

- [ ] **Step 4: 跑测试确认通过，再跑全套**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_eval_lookup_matching.py -q -p no:cacheprovider`
Expected: 原有 8 个 + 新 2 个全过。全套 Expected: `333 passed`（311 + 7 + 2 + 6 + 5 + 2）

- [ ] **Step 5: 提交**

```bash
git add anatobind/eval/lookup.py tests/test_eval_lookup_matching.py
git commit -m "Eval: set-valued host fractions and side names on top of the class-aware lookup"
```

---

### Task 6: 折评估脚本——检测、绑定、解剖三张表

**Files:**
- Create: `scripts/eval_knee_folds.py`
- Test: `tests/test_eval_knee_folds_script.py`

**Interfaces:**
- Consumes: Task 2 的 `prepare_lesion.validation_path / validation_npz_path`，`anatobind.nnunet.prepare.validation_path`（Dataset901），Task 3 的 `decode_boxes / load_label_map / load_nnunet_probabilities`，Task 4 的 `sweep / operating_point / per_family / gate / match_scan / THRESHOLDS`，Task 5 的 `describe_host`，`anatobind.eval.lookup.LabelIndex`，`anatobind.eval.matching.bucket`，Task 1 的 `read_boxes_xyz / LOOKUP_CLASS_OF_FAMILY`，`anatobind.train.cache.VIEWS`。
- Produces（`--out D`）：`D/records.csv`（每个真值框一行）、`D/froc_<view>.csv`（阈值扫描）、`D/summary.md`、`D/gate.json`（`{"view": "clean", "pass", "thr", "sensitivity_family", "fp_per_scan", "n_scans", "n_gt"}`）。`main(argv) -> int`（0 正常；2 有缺失文件且未加 `--allow-missing`）。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_eval_knee_folds_script.py
import csv
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from anatobind.nnunet import prepare as anat
from anatobind.nnunet import prepare_lesion as les
from anatobind.nnunet.lesion_labels import boxes_to_label_map, read_boxes_xyz
from anatobind.train.cache import VIEWS


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_knee_folds.py"
    spec = importlib.util.spec_from_file_location("eval_knee_folds", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _perfect_results(export, results, scans):
    """Dataset901 validation = the export seg; Dataset902 validation = the box-filled map, one-hot npz."""
    folds = json.loads((export / "splits.json").read_text())["folds"]
    for scan in scans:
        seg_img = nib.load(str(export / scan / "seg.nii.gz"))
        seg = np.asanyarray(seg_img.dataobj).astype(np.uint8)
        lab = boxes_to_label_map(read_boxes_xyz(export / scan / "boxes.csv"), seg.shape)
        probs = np.zeros((5,) + seg.shape, np.float32)
        for c in range(5):
            probs[c] = lab == c
        for view in VIEWS:
            a, l = anat.validation_path(results, folds[scan], scan, view), les.validation_path(results, folds[scan], scan, view)
            a.parent.mkdir(parents=True, exist_ok=True)
            l.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(seg, seg_img.affine), str(a))
            nib.save(nib.Nifti1Image(lab, seg_img.affine), str(l))
            np.savez_compressed(les.validation_npz_path(results, folds[scan], scan, view),
                                probabilities=np.ascontiguousarray(probs.transpose(0, 3, 2, 1)).astype(np.float16))


def test_perfect_outputs_pass_the_gate_and_bind_every_in_seg_lesion(synthetic_m1r, tmp_path):
    export, _, scans = synthetic_m1r
    _perfect_results(export, tmp_path / "results", scans)
    out = tmp_path / "eval"
    rc = _load_script().main(["--out", str(out), "--export-root", str(export), "--results-root", str(tmp_path / "results"),
                              "--views", "clean,noise_q1"])
    assert rc == 0
    g = json.loads((out / "gate.json").read_text())
    assert g["pass"] is True and g["view"] == "clean" and g["sensitivity_family"] == 1.0 and g["fp_per_scan"] == 0.0
    assert g["n_scans"] == 5 and g["n_gt"] == 20
    with open(out / "records.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    clean = [r for r in rows if r["view"] == "clean"]
    assert len(clean) == 20 and all(r["matched"] == "1" for r in clean)
    cart = next(r for r in clean if r["gt_family"] == "Cartilage Lesion")
    assert cart["bucket"] == "correct" and cart["pred_host_label"] == "2" and cart["given_bucket"] == "correct"
    men = next(r for r in clean if r["gt_family"] == "Meniscal Tear")
    assert men["pred_side"] == "medial" and men["side_correct"] == "1" and json.loads(men["pred_host_fractions"])["5"] == 1.0
    eff = next(r for r in clean if r["gt_family"] == "Effusion")
    assert eff["bucket"] == "" and eff["pred_host_label"] == "" and eff["pred_side"] == "-"
    froc = list(csv.DictReader(open(out / "froc_clean.csv", newline="")))
    assert len(froc) == 19 and froc[0]["thr"] == "0.05" and float(froc[-1]["sensitivity"]) == 1.0
    text = (out / "summary.md").read_text()
    assert "| clean |" in text and "GATE: PASS" in text and "Dice" in text


def test_missing_validation_files_fail_loudly_unless_allowed(synthetic_m1r, tmp_path):
    export, _, scans = synthetic_m1r
    _perfect_results(export, tmp_path / "results", scans)
    folds = json.loads((export / "splits.json").read_text())["folds"]
    les.validation_npz_path(tmp_path / "results", folds[scans[0]], scans[0], "clean").unlink()
    mod = _load_script()
    args = ["--out", str(tmp_path / "e1"), "--export-root", str(export), "--results-root", str(tmp_path / "results"), "--views", "clean"]
    assert mod.main(args) == 2
    assert mod.main(args + ["--allow-missing"]) == 0
    assert json.loads((tmp_path / "e1" / "gate.json").read_text())["n_scans"] == 4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_eval_knee_folds_script.py -q -p no:cacheprovider`
Expected: FAIL，`FileNotFoundError` 或 `AttributeError`（脚本不存在）

- [ ] **Step 3: 写脚本**

```python
#!/usr/bin/env python
# scripts/eval_knee_folds.py
"""Fold evaluation of the knee capability system on nnU-Net's held-out validation outputs (spec 2026-09-23 §4.1).

Per held-out scan and view: Dataset902's validation label map + npz probabilities give the detections,
Dataset901's validation label map gives the anatomy for the lookup, boxes.csv gives the truth.
Detection: sensitivity and false positives per scan over a score sweep, operating point at <= 2 FP/scan.
Binding: on the hits at the operating point (system path) and on the annotated boxes (given-box path,
a control, not a capability). Anatomy: Dice of Dataset901 against the export segmentation.

  D=docs/verification/$(date +%F)/knee_eval; mkdir -p $D
  source scripts/nnunet_env.sh
  PYTHONPATH=. python scripts/eval_knee_folds.py --out $D | tee $D/output.txt
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.detection_metrics import FP_MAX, GATE_SENSITIVITY, THRESHOLDS, gate, match_scan, operating_point, per_family, sweep  # noqa: E402
from anatobind.eval.lesion_boxes import decode_boxes, load_label_map, load_nnunet_probabilities  # noqa: E402
from anatobind.eval.lookup import FAMILY_OF_TISSUE, LabelIndex, describe_host  # noqa: E402
from anatobind.eval.matching import CORRECT, bucket  # noqa: E402
from anatobind.nnunet import prepare as anat  # noqa: E402
from anatobind.nnunet import prepare_lesion as les  # noqa: E402
from anatobind.nnunet.lesion_labels import LOOKUP_CLASS_OF_FAMILY, read_boxes_xyz  # noqa: E402
from anatobind.train.cache import VIEWS  # noqa: E402

FM = Path("/data2/congcong/data/FM_data/derived")
GATE_VIEW = "clean"
FIELDS = ["fold", "scan", "view", "ann_id", "layer", "gt_family", "tissue_id", "host_label", "host_side",
          "matched", "det_score", "det_family", "iou", "pred_host_label", "pred_side", "pred_host_fractions",
          "bucket", "side_correct", "given_host_label", "given_side", "given_bucket", "given_side_correct"]


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--export-root", type=Path, default=FM / "skmtea" / "m1r")
    ap.add_argument("--results-root", type=Path, default=Path(os.environ.get("nnUNet_results", FM / "nnunet" / "results")))
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args(argv)
    a.views = a.views.split(",")
    a.folds = [int(f) for f in a.folds.split(",")]
    return a


def iou_of(a, b):
    lo, hi = np.maximum(a[:3], b[:3]), np.minimum(a[3:], b[3:])
    inter = float(np.clip(hi - lo, 0, None).prod())
    va, vb = float(np.prod(np.subtract(a[3:], a[:3]))), float(np.prod(np.subtract(b[3:], b[:3])))
    return inter / (va + vb - inter) if va + vb - inter > 0 else 0.0


def load_scan(a, scan, fold, view, missing):
    paths = {"anat": anat.validation_path(a.results_root, fold, scan, view),
             "les": les.validation_path(a.results_root, fold, scan, view),
             "npz": les.validation_npz_path(a.results_root, fold, scan, view)}
    absent = [str(p) for p in paths.values() if not p.exists()]
    if absent:
        missing.extend(absent)
        return None
    seg_img = nib.load(str(a.export_root / scan / "seg.nii.gz"))
    spacing = tuple(float(v) for v in seg_img.header.get_zooms()[:3])
    lesion = load_label_map(paths["les"])
    dets = decode_boxes(lesion, load_nnunet_probabilities(paths["npz"], lesion))
    anatomy = load_label_map(paths["anat"])
    seg = np.asanyarray(seg_img.dataobj).astype(np.uint8)
    dice = {k: 2.0 * np.logical_and(anatomy == k, seg == k).sum() / ((anatomy == k).sum() + (seg == k).sum())
            for k in range(1, 7) if (seg == k).any()}
    gt = [{**r, "family": r["supercategory"]} for r in read_boxes_xyz(a.export_root / scan / "boxes.csv")]  # metrics key on "family"
    return {"scan": scan, "fold": fold, "view": view, "gt": gt,
            "dets": dets, "index": LabelIndex(anatomy, spacing), "dice": dice}


def side_ok(gt_side, pred_side):
    return "" if gt_side not in ("medial", "lateral") else str(int(gt_side == pred_side))


def binding_records(scans, thr):
    records = []
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        pairs = match_scan(s["gt"], dets)
        for g, r in enumerate(s["gt"]):
            gt_cls = LOOKUP_CLASS_OF_FAMILY[r["supercategory"]]
            in_seg = r["layer"] == "in_seg" and r["tissue_id"] in FAMILY_OF_TISSUE
            rec = {"fold": s["fold"], "scan": s["scan"], "view": s["view"], "ann_id": r["ann_id"], "layer": r["layer"],
                   "gt_family": r["supercategory"], "tissue_id": r["tissue_id"],
                   "host_label": "" if r["host_label"] is None else r["host_label"], "host_side": r["host_side"],
                   "matched": "0", "det_score": "", "det_family": "", "iou": "", "pred_host_label": "", "pred_side": "-",
                   "pred_host_fractions": "{}", "bucket": "", "side_correct": ""}
            given = describe_host(s["index"], r["box"], gt_cls)
            rec.update({"given_host_label": "" if given["host_label"] is None else given["host_label"],
                        "given_side": given["side"],
                        "given_bucket": bucket(gt_cls, r["tissue_id"], gt_cls, given["host_label"] if given["host_label"] is not None else given["host_name"]) if in_seg else "",
                        "given_side_correct": side_ok(r["host_side"], given["side"])})
            if g in pairs:
                d = dets[pairs[g]]
                pred_cls = LOOKUP_CLASS_OF_FAMILY[d["family"]]
                h = describe_host(s["index"], d["box"], pred_cls)
                rec.update({"matched": "1", "det_score": round(d["score"], 4), "det_family": d["family"],
                            "iou": round(iou_of(np.array(r["box"]), np.array(d["box"])), 4),
                            "pred_host_label": "" if h["host_label"] is None else h["host_label"], "pred_side": h["side"],
                            "pred_host_fractions": json.dumps({str(k): round(v, 4) for k, v in h["host_fractions"].items()}),
                            "bucket": bucket(gt_cls, r["tissue_id"], pred_cls, h["host_label"] if h["host_label"] is not None else h["host_name"]) if in_seg else "",
                            "side_correct": side_ok(r["host_side"], h["side"])})
            elif in_seg:
                rec["bucket"] = "miss"
            records.append(rec)
    return records


def binding_summary(records, key_bucket, key_side):
    matched = [r for r in records if r[key_bucket] not in ("", "miss")]
    sides = [r for r in records if r[key_side] != "" and r[key_bucket] not in ("", "miss")]
    return {"n": len(matched), "family_correct": sum(r[key_bucket] == CORRECT for r in matched) / len(matched) if matched else float("nan"),
            "n_side": len(sides), "side_correct": sum(r[key_side] == "1" for r in sides) / len(sides) if sides else float("nan")}


def main(argv=None):
    a = parse(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    folds = json.loads((a.export_root / "splits.json").read_text())["folds"]
    missing, per_view = [], {}
    for view in a.views:
        scans = [load_scan(a, scan, fold, view, missing) for scan, fold in sorted(folds.items()) if fold in a.folds]
        per_view[view] = [s for s in scans if s is not None]
    if missing and not a.allow_missing:
        print(f"{len(missing)} validation files missing, e.g. {missing[:3]}; rerun with --allow-missing to evaluate the rest")
        return 2
    records, gate_out = [], {}
    lines = ["# Knee capability evaluation", "",
             f"IoU >= 0.1, FP budget {FP_MAX}/scan, gate on `{GATE_VIEW}` family sensitivity >= {GATE_SENSITIVITY}", "",
             "| view | n_scans | n_gt | thr | sensitivity | sensitivity_family | fp_per_scan |", "|---|---|---|---|---|---|---|"]
    detail = []
    for view, scans in per_view.items():
        rows = sweep(scans, THRESHOLDS)
        with open(a.out / f"froc_{view}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        op = operating_point(rows)
        g = gate(rows)
        if view == GATE_VIEW:
            gate_out = {"view": view, **g, "n_scans": len(scans), "n_gt": rows[0]["n_gt"] if rows else 0}
        thr = op["thr"] if op else 1.0
        lines.append(f"| {view} | {len(scans)} | {rows[0]['n_gt'] if rows else 0} | {thr:.2f} | "
                     f"{(op or {}).get('sensitivity', 0.0):.3f} | {(op or {}).get('sensitivity_family', 0.0):.3f} | {(op or {}).get('fp_per_scan', float('nan')):.2f} |")
        recs = binding_records(scans, thr)
        records += recs
        fam = per_family(scans, thr)
        sysb, given = binding_summary(recs, "bucket", "side_correct"), binding_summary(recs, "given_bucket", "given_side_correct")
        dice = defaultdict(list)
        for s in scans:
            for k, v in s["dice"].items():
                dice[k].append(v)
        detail += [f"## {view}", "", "per family at the operating point: " + ", ".join(
            f"{f}: {v['n_hit_family']}/{v['n_gt']} ({v['sensitivity_family']:.2f})" for f, v in sorted(fam.items())), "",
            f"binding on hits (system): n={sysb['n']}, family correct {sysb['family_correct']:.3f}, side correct {sysb['side_correct']:.3f} (n_side={sysb['n_side']})",
            f"binding on annotated boxes (given-box control): n={given['n']}, family correct {given['family_correct']:.3f}, side correct {given['side_correct']:.3f} (n_side={given['n_side']})",
            "anatomy Dice (Dataset901 vs export seg): " + ", ".join(f"{k}: {np.mean(v):.3f}" for k, v in sorted(dice.items())), ""]
    lines += [""] + detail
    lines.append(f"GATE: {'PASS' if gate_out.get('pass') else 'FAIL'} ({json.dumps(gate_out)})")
    if missing:
        lines += ["", f"{len(missing)} validation files were missing and skipped."]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    (a.out / "gate.json").write_text(json.dumps(gate_out, indent=1))
    with open(a.out / "records.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

注意 `bucket()` 的第四个参数：命中但候选缺席时 `host_label` 为 `None`、`host_name` 为 `""`，传 `""` 会落入 `WRONG_HOST`，这正是想要的（找不到结构就是绑错）。

- [ ] **Step 4: 跑测试确认通过，再跑全套**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_eval_knee_folds_script.py -q -p no:cacheprovider`
Expected: `2 passed`。全套 Expected: `335 passed`

- [ ] **Step 5: 提交**

```bash
git add scripts/eval_knee_folds.py tests/test_eval_knee_folds_script.py
git commit -m "Eval: fold evaluation script for knee detection, binding and anatomy on nnU-Net validation outputs"
```

---

### Task 7: 在真实输出上跑评估，过止损门

**Files:**
- Create: `docs/verification/<YYYY-MM-DD>/knee_eval/{summary.md,gate.json,records.csv,froc_*.csv,output.txt}`（`<YYYY-MM-DD>` 为执行当天）
- Create（仅不过线时）: `docs/verification/<YYYY-MM-DD>/knee_eval/VERDICT.md`

**Interfaces:**
- Consumes: Task 2 的五折验证输出；Task 6 的脚本。
- Produces: `gate.json` 的 `pass` 决定是否进入 Task 8–10。

- [ ] **Step 1: fold 0 先看（决定 N6 的第二道汇报）**

fold 0 的 `validation/` 出现 217 个 `.npz` 后：

```bash
cd /data0/congcong/code/Project_Doing/foundation_model
source scripts/nnunet_env.sh
D=docs/verification/$(date +%F)/knee_eval_fold0; mkdir -p $D
PYTHONPATH=. python scripts/eval_knee_folds.py --out $D --folds 0 | tee $D/output.txt; echo "rc=$?"
```

期望：`rc=0`；`summary.md` 的 clean 行给出阈值、灵敏度、每卷假阳；`GATE:` 行。把 `output.txt` 前 30 行原样报给用户，不下结论（31 卷不是五折）。

- [ ] **Step 2: 五折全量**

五折 `validation/` 各 217 个 `.nii.gz` 与 `.npz` 齐全后：

```bash
D=docs/verification/$(date +%F)/knee_eval; mkdir -p $D
PYTHONPATH=. python scripts/eval_knee_folds.py --out $D | tee $D/output.txt; echo "rc=$?"
python - <<EOF
import json; print(json.load(open("$D/gate.json")))
EOF
```

期望：`rc=0`，`gate.json` 的 `n_scans` = 155、`n_gt` = 465。

- [ ] **Step 3: 判门**

- `pass: true` → 提交产物，进入 Task 8：

```bash
git add docs/verification/$(date +%F)/knee_eval
git commit -m "Verification: knee five-fold detection, binding and anatomy numbers, gate passed"
```

- `pass: false` → 写 `VERDICT.md`（结论一行、`gate.json` 原样、clean 视图的 froc 表前 10 行、每族灵敏度、09-12 检测头的对照数字），提交，**停止**，把 VERDICT.md 全文报给用户。第二臂（fastMRI+ 预训练 2.5D 检测器迁移到 SKM-TEA）由用户决定后另写计划；本计划到此为止。

```bash
git add docs/verification/$(date +%F)/knee_eval
git commit -m "Verification: knee five-fold numbers, detection gate failed, stop for user decision"
```

---

### Task 8: 单命令入口 `scripts/infer_knee.py`

**Files:**
- Create: `anatobind/infer/__init__.py`（空文件）
- Create: `anatobind/infer/canonical.py`
- Create: `anatobind/infer/knee.py`
- Create: `scripts/infer_knee.py`
- Test: `tests/test_infer_canonical.py`、`tests/test_infer_knee.py`

**Interfaces:**
- Consumes: Task 3 的 `decode_boxes / load_label_map / load_nnunet_probabilities`，Task 5 的 `describe_host`，`anatobind.eval.lookup.LabelIndex`，Task 1 的 `LOOKUP_CLASS_OF_FAMILY`，`anatobind.nnunet.prepare.{DATASET_ID, TRAINER}`，`anatobind.nnunet.prepare_lesion.DATASET_ID`。
- Produces:
  - `canonical.TARGET_AXCODES = ("I", "P", "R")`；`canonical.to_export_frame(img) -> nib.Nifti1Image`
  - `knee.CSV_FIELDS`；`knee.run_nnunet(dataset_id, in_dir, out_dir, folds, gpu, save_probabilities)`；`knee.lesion_table(dets, anatomy, spacing) -> list[dict]`；`knee.write_table(rows, path)`；`knee.render_overlay(image, anatomy, box, path)`；`knee.run(image_path, out_dir, frame, folds, gpu, predict=run_nnunet) -> list[dict]`
  - 输出目录布局见 spec §3.4。

- [ ] **Step 1: 写失败的测试（规范化）**

```python
# tests/test_infer_canonical.py
import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine

from anatobind.infer.canonical import TARGET_AXCODES, to_export_frame


def _image(axcodes, shape=(6, 8, 10)):
    ornt = nib.orientations.axcodes2ornt(axcodes)
    affine = nib.orientations.inv_ornt_aff(ornt, shape) @ np.diag([0.5, 0.7, 0.9, 1.0])
    arr = np.zeros(shape, np.float32)
    arr[1, 2, 3] = 7.0
    return nib.Nifti1Image(arr, affine)


def test_axes_come_out_as_superior_to_inferior_anterior_to_posterior_left_to_right():
    out = to_export_frame(_image(("R", "A", "S")))
    assert nib.aff2axcodes(out.affine) == TARGET_AXCODES == ("I", "P", "R")
    assert out.shape == (10, 8, 6)


def test_the_marker_voxel_keeps_its_world_coordinate():
    src = _image(("L", "P", "S"))
    out = to_export_frame(src)
    idx = np.argwhere(np.asanyarray(out.dataobj) == 7.0)[0]
    assert np.allclose(apply_affine(out.affine, idx), apply_affine(src.affine, (1, 2, 3)))


def test_an_image_already_in_the_export_frame_is_returned_unchanged():
    src = _image(("I", "P", "R"))
    out = to_export_frame(src)
    assert np.array_equal(np.asanyarray(out.dataobj), np.asanyarray(src.dataobj)) and np.allclose(out.affine, src.affine)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_infer_canonical.py -q -p no:cacheprovider`
Expected: FAIL，`ModuleNotFoundError: No module named 'anatobind.infer'`

- [ ] **Step 3: 写规范化模块**

```python
# anatobind/infer/canonical.py
"""Bring an external NIfTI onto the SKM-TEA export frame the knee models were trained on (spec §3.4).

The export grid runs superior->inferior, anterior->posterior, left->right along (X, Y, Z) at
0.625 x 0.625 x 0.8 mm (docs/data_engine_skmtea.md), and its NIfTIs carry a diagonal affine with no
real orientation. `--frame h5` therefore takes such a file as is; `--frame world` trusts the affine
and reorders the axes to axis codes ('I', 'P', 'R'). nnU-Net resamples spacing itself from the
header, so only the axis order and the pixdims matter here.
"""
import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform

TARGET_AXCODES = ("I", "P", "R")


def to_export_frame(img):
    transform = ornt_transform(io_orientation(img.affine), axcodes2ornt(TARGET_AXCODES))
    arr = apply_orientation(np.asanyarray(img.dataobj), transform)
    affine = img.affine @ inv_ornt_aff(transform, img.shape)
    out = nib.Nifti1Image(np.ascontiguousarray(arr), affine)
    out.header.set_xyzt_units("mm")
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_infer_canonical.py -q -p no:cacheprovider`
Expected: `3 passed`

- [ ] **Step 5: 写失败的测试（管线）**

```python
# tests/test_infer_knee.py
import csv
import json

import nibabel as nib
import numpy as np

from anatobind.infer.knee import CSV_FIELDS, lesion_table, render_overlay, run, write_table

SP = (0.625, 0.625, 0.8)


def _anatomy():
    a = np.zeros((40, 40, 20), np.uint8)
    a[5:20, 5:30, 2:12] = 2     # femoral cartilage
    a[22:36, 5:30, 4:10] = 5    # medial meniscus
    return a


def test_lesion_table_names_the_host_and_keeps_the_box():
    dets = [{"family": "Cartilage Lesion", "label": 1, "box": (8, 8, 3, 12, 14, 8), "score": 0.9, "n_voxels": 120},
            {"family": "Effusion", "label": 4, "box": (0, 0, 0, 3, 3, 3), "score": 0.4, "n_voxels": 27}]
    rows = lesion_table(dets, _anatomy(), SP)
    assert [r["lesion_id"] for r in rows] == [1, 2] and list(rows[0]) == CSV_FIELDS
    assert rows[0]["host_label"] == 2 and rows[0]["host_name"] == "femoral_cartilage" and rows[0]["side"] == "single"
    assert json.loads(rows[0]["host_fractions"])["2"] == 1.0 and rows[0]["x0"] == 8 and rows[0]["z1"] == 8
    assert rows[1]["host_name"] == "none" and rows[1]["host_label"] == "" and rows[1]["host_fractions"] == "{}"


def test_write_table_and_overlay_produce_files(tmp_path):
    rows = lesion_table([{"family": "Meniscal Tear", "label": 2, "box": (25, 10, 5, 30, 20, 9), "score": 0.7, "n_voxels": 200}],
                        _anatomy(), SP)
    write_table(rows, tmp_path / "lesions.csv")
    with open(tmp_path / "lesions.csv", newline="") as fh:
        back = list(csv.DictReader(fh))
    assert back[0]["host_name"] == "meniscus_medial" and back[0]["side"] == "medial"
    render_overlay(np.random.default_rng(0).random((40, 40, 20), np.float32), _anatomy(), (25, 10, 5, 30, 20, 9), tmp_path / "o.png")
    assert (tmp_path / "o.png").stat().st_size > 1000


def _fake_predict(anatomy, lesion):
    def predict(dataset_id, in_dir, out_dir, folds, gpu, save_probabilities):
        out_dir.mkdir(parents=True, exist_ok=True)
        affine = nib.load(str(in_dir / "case_0000.nii.gz")).affine
        lab = anatomy if dataset_id == 901 else lesion
        nib.save(nib.Nifti1Image(lab, affine), str(out_dir / "case.nii.gz"))
        if save_probabilities:
            p = np.zeros((5,) + lab.shape, np.float32)
            for c in range(5):
                p[c] = lab == c
            np.savez_compressed(out_dir / "case.npz", probabilities=np.ascontiguousarray(p.transpose(0, 3, 2, 1)))
    return predict


def test_run_writes_the_four_outputs_from_an_h5_frame_volume(tmp_path):
    anatomy = _anatomy()
    lesion = np.zeros_like(anatomy)
    lesion[8:12, 8:14, 3:8] = 1
    image = np.random.default_rng(0).random(anatomy.shape, np.float32)
    nib.save(nib.Nifti1Image(image, np.diag([0.625, 0.625, 0.8, 1.0])), str(tmp_path / "vol.nii.gz"))
    rows = run(tmp_path / "vol.nii.gz", tmp_path / "out", "h5", [0], 0, predict=_fake_predict(anatomy, lesion))
    for name in ("anatomy.nii.gz", "lesions.nii.gz", "lesions.csv", "overlay.png"):
        assert (tmp_path / "out" / name).exists()
    assert len(rows) == 1 and rows[0]["family"] == "Cartilage Lesion" and rows[0]["host_name"] == "femoral_cartilage"
    assert np.array_equal(np.asanyarray(nib.load(str(tmp_path / "out" / "anatomy.nii.gz")).dataobj), anatomy)


def test_run_reorients_a_world_frame_volume_before_predicting(tmp_path):
    anatomy = _anatomy()
    seen = {}

    def predict(dataset_id, in_dir, out_dir, folds, gpu, save_probabilities):
        img = nib.load(str(in_dir / "case_0000.nii.gz"))
        seen["axcodes"], seen["shape"] = nib.aff2axcodes(img.affine), img.shape
        _fake_predict(anatomy, np.zeros_like(anatomy))(dataset_id, in_dir, out_dir, folds, gpu, save_probabilities)

    ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    affine = nib.orientations.inv_ornt_aff(ornt, (20, 40, 40)) @ np.diag([0.8, 0.625, 0.625, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((20, 40, 40), np.float32), affine), str(tmp_path / "ras.nii.gz"))
    rows = run(tmp_path / "ras.nii.gz", tmp_path / "out", "world", [0], 0, predict=predict)
    assert seen["axcodes"] == ("I", "P", "R") and seen["shape"] == (40, 40, 20) and rows == []
```

- [ ] **Step 6: 跑测试确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_infer_knee.py -q -p no:cacheprovider`
Expected: FAIL，`ModuleNotFoundError: No module named 'anatobind.infer.knee'`

- [ ] **Step 7: 写管线模块与脚本**

```python
# anatobind/infer/knee.py
"""One volume in, three outputs out: anatomy label map, lesion boxes with family, host structure (spec §3.4).

Both nnU-Net models are called through nnUNetv2_predict as subprocesses (the predictor is injectable
so the pipeline is testable without a GPU); the rest is Task 3 decoding and Task 5 lookup.
"""
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from anatobind.eval.lesion_boxes import decode_boxes, load_label_map, load_nnunet_probabilities  # noqa: E402
from anatobind.eval.lookup import LabelIndex, describe_host  # noqa: E402
from anatobind.infer.canonical import to_export_frame  # noqa: E402
from anatobind.nnunet.lesion_labels import LOOKUP_CLASS_OF_FAMILY  # noqa: E402
from anatobind.nnunet.prepare import DATASET_ID as ANATOMY_ID  # noqa: E402
from anatobind.nnunet.prepare import TRAINER  # noqa: E402
from anatobind.nnunet.prepare_lesion import DATASET_ID as LESION_ID  # noqa: E402

CSV_FIELDS = ["lesion_id", "family", "score", "x0", "y0", "z0", "x1", "y1", "z1", "n_voxels",
              "host_label", "host_name", "side", "host_fractions"]
NNUNET_ROOT = Path("/data2/congcong/data/FM_data/derived/nnunet")
CASE = "case"


def nnunet_env(gpu):
    env = dict(os.environ)
    env.setdefault("nnUNet_raw", str(NNUNET_ROOT / "raw"))
    env.setdefault("nnUNet_preprocessed", str(NNUNET_ROOT / "preprocessed"))
    env.setdefault("nnUNet_results", str(NNUNET_ROOT / "results"))
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PATH"] = str(Path.home() / "anaconda3/envs/nvgen/bin") + os.pathsep + env.get("PATH", "")
    return env


def run_nnunet(dataset_id, in_dir, out_dir, folds, gpu, save_probabilities):
    cmd = ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir), "-d", str(dataset_id), "-c", "3d_fullres",
           "-tr", TRAINER, "-f", *[str(f) for f in folds], "-npp", "2", "-nps", "2", "--disable_progress_bar"]
    if save_probabilities:
        cmd.append("--save_probabilities")
    subprocess.run(cmd, check=True, env=nnunet_env(gpu))


def lesion_table(dets, anatomy, spacing):
    index = LabelIndex(anatomy, spacing)
    rows = []
    for i, d in enumerate(dets, start=1):
        h = describe_host(index, d["box"], LOOKUP_CLASS_OF_FAMILY[d["family"]])
        x0, y0, z0, x1, y1, z1 = (int(v) for v in d["box"])
        rows.append({"lesion_id": i, "family": d["family"], "score": round(float(d["score"]), 4),
                     "x0": x0, "y0": y0, "z0": z0, "x1": x1, "y1": y1, "z1": z1, "n_voxels": int(d["n_voxels"]),
                     "host_label": "" if h["host_label"] is None else h["host_label"], "host_name": h["host_name"],
                     "side": h["side"], "host_fractions": json.dumps({str(k): round(v, 4) for k, v in h["host_fractions"].items()})})
    return rows


def write_table(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def render_overlay(image, anatomy, box, path):
    """Three orthogonal slices through the box centre, anatomy in colour, the box as a rectangle."""
    c = [(box[i] + box[i + 3]) // 2 for i in range(3)]
    planes = [(image[c[0]], anatomy[c[0]], (box[1], box[2], box[4], box[5]), "x"),
              (image[:, c[1]], anatomy[:, c[1]], (box[0], box[2], box[3], box[5]), "y"),
              (image[:, :, c[2]], anatomy[:, :, c[2]], (box[0], box[1], box[3], box[4]), "z")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (im, lab, (a0, b0, a1, b1), name) in zip(axes, planes):
        ax.imshow(im.T, cmap="gray", origin="lower", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99))
        ax.imshow(np.ma.masked_where(lab.T == 0, lab.T), cmap="tab10", origin="lower", vmin=0, vmax=9, alpha=0.4, interpolation="nearest")
        ax.add_patch(plt.Rectangle((a0, b0), a1 - a0, b1 - b0, fill=False, edgecolor="yellow", linewidth=1.5))
        ax.set_title(f"slice through box centre, axis {name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(image_path, out_dir, frame, folds, gpu, predict=run_nnunet):
    out_dir = Path(out_dir)
    img = nib.load(str(image_path))
    if frame == "world":
        img = to_export_frame(img)
    in_dir = out_dir / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(in_dir / f"{CASE}_0000.nii.gz"))
    predict(ANATOMY_ID, in_dir, out_dir / "anatomy_raw", folds, gpu, False)
    predict(LESION_ID, in_dir, out_dir / "lesion_raw", folds, gpu, True)
    anatomy = load_label_map(out_dir / "anatomy_raw" / f"{CASE}.nii.gz")
    lesion = load_label_map(out_dir / "lesion_raw" / f"{CASE}.nii.gz")
    dets = decode_boxes(lesion, load_nnunet_probabilities(out_dir / "lesion_raw" / f"{CASE}.npz", lesion))
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    rows = lesion_table(dets, anatomy, spacing)
    shutil.copyfile(out_dir / "anatomy_raw" / f"{CASE}.nii.gz", out_dir / "anatomy.nii.gz")
    shutil.copyfile(out_dir / "lesion_raw" / f"{CASE}.nii.gz", out_dir / "lesions.nii.gz")
    write_table(rows, out_dir / "lesions.csv")
    image = np.asanyarray(img.dataobj).astype(np.float32)
    box = dets[0]["box"] if dets else tuple(s // 2 for s in image.shape) + tuple(s // 2 + 1 for s in image.shape)
    render_overlay(image, anatomy, box, out_dir / "overlay.png")
    return rows
```

```python
#!/usr/bin/env python
# scripts/infer_knee.py
"""Knee: one NIfTI volume -> anatomy.nii.gz, lesions.nii.gz, lesions.csv, overlay.png (spec 2026-09-23 §3.4).

  source scripts/nnunet_env.sh
  PYTHONPATH=. python scripts/infer_knee.py --image <vol.nii.gz> --out <dir> [--frame h5|world] [--folds 0 1 2 3 4] [--gpu 0]

--frame h5: the volume is already on the SKM-TEA export grid (export files, caches).
--frame world: trust the NIfTI affine and reorder the axes to (I, P, R) first (DICOM-derived volumes).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.infer.knee import CSV_FIELDS, run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frame", choices=("h5", "world"), default="h5")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(argv)
    rows = run(a.image, a.out, a.frame, a.folds, a.gpu)
    print("\t".join(CSV_FIELDS))
    for r in rows:
        print("\t".join(str(r[k]) for k in CSV_FIELDS))
    print(f"{len(rows)} lesions -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: 跑测试确认通过，再跑全套**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_infer_knee.py tests/test_infer_canonical.py -q -p no:cacheprovider`
Expected: `7 passed`。全套 Expected: `342 passed`

- [ ] **Step 9: 提交**

```bash
git add anatobind/infer scripts/infer_knee.py tests/test_infer_canonical.py tests/test_infer_knee.py
git commit -m "Inference: single-command knee entry point with frame canonicalisation, lesion table and overlay"
```

- [ ] **Step 10: 回归核对——入口与折验证输出一致（GPU，一次）**

MTR_010 在 fold 0（`splits.json`），用 fold 0 的模型跑入口，与 nnU-Net 折末验证输出比：

```bash
cd /data0/congcong/code/Project_Doing/foundation_model
source scripts/nnunet_env.sh
O=/data2/congcong/data/FM_data/derived/knee_infer/MTR_010_clean_fold0
PYTHONPATH=. python scripts/infer_knee.py --image /data2/congcong/data/FM_data/derived/skmtea/m1r/MTR_010/image_clean_e1.nii.gz \
    --out $O --frame h5 --folds 0 --gpu 0 | tee logs/infer_knee_MTR_010.txt
R=$nnUNet_results
python - <<EOF
import nibabel as nib, numpy as np
def load(p): return np.asanyarray(nib.load(p).dataobj)
a = load("$O/anatomy.nii.gz"); b = load("$R/Dataset901_SKMTEAm1r/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation/MTR_010_clean0.nii.gz")
print("anatomy shape", a.shape, b.shape, "dice per label", {k: round(2*((a==k)&(b==k)).sum()/max((a==k).sum()+(b==k).sum(),1),4) for k in range(1,7)})
l = load("$O/lesions.nii.gz"); m = load("$R/Dataset902_SKMTEAlesion/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation/MTR_010_clean0.nii.gz")
print("lesion voxels", int((l>0).sum()), int((m>0).sum()), "agreement", round(float((l==m).mean()),5))
EOF
```

期望：六类 Dice 均 ≥ 0.99，病灶图逐体素一致率 ≥ 0.999。低于此值先查 `nnUNetv2_predict` 与折末验证的推理设置（镜像 TTA、checkpoint）再继续。把输出存到 `docs/verification/<YYYY-MM-DD>/knee_eval/infer_regression.txt`（Task 10 一起提交）。

---

### Task 9: DICOM 世界帧核查（手工，一卷）

**Files:**
- Create: `docs/verification/<YYYY-MM-DD>/knee_eval/dicom_world_frame.md`

**Interfaces:**
- Consumes: Task 8 的入口；`anatobind.data_engine.seg_frames.cartilage_score(mag, seg)`（回波 1 上软骨亮，帧对了分数高；仓库门线 1.5）。

- [ ] **Step 1: 取一卷 DICOM 转 NIfTI**

```bash
S=/data2/congcong/data/FM_data/derived/knee_infer/dicom_MTR_010_src; mkdir -p $S
ls /data2/congcong/data/FM_data/SKM-TEA_ltr/dicoms | grep MTR_010
tar -xzf /data2/congcong/data/FM_data/SKM-TEA_ltr/dicoms/$(ls /data2/congcong/data/FM_data/SKM-TEA_ltr/dicoms | grep MTR_010) -C $S
find $S -name '*.dcm' | head -3; find $S -name '*.dcm' | wc -l
PYTHONNOUSERSITE=1 ~/anaconda3/envs/nvgen/bin/python - <<EOF
import dicom2nifti, glob, os
src = os.path.dirname(sorted(glob.glob("$S/**/*.dcm", recursive=True))[0])
dicom2nifti.convert_directory(src, "$S", compression=True, reorient=False)
print(os.listdir("$S"))
EOF
```

若一卷含两个回波序列，取回波 1（文件名或 DICOM `EchoNumbers` = 1）。记下 NIfTI 的 `aff2axcodes` 与形状。

- [ ] **Step 2: 走世界帧入口并算帧分数**

```bash
source scripts/nnunet_env.sh
O=/data2/congcong/data/FM_data/derived/knee_infer/MTR_010_dicom_world
PYTHONPATH=. python scripts/infer_knee.py --image $S/<echo1>.nii.gz --out $O --frame world --gpu 0
PYTHONPATH=. python - <<EOF
import nibabel as nib, numpy as np
from anatobind.data_engine.seg_frames import cartilage_score
mag = np.asanyarray(nib.load("$O/input/case_0000.nii.gz").dataobj).astype(np.float32)
seg = np.asanyarray(nib.load("$O/anatomy.nii.gz").dataobj)
print("cartilage score", round(cartilage_score(mag, seg), 3), "labels present", sorted(set(np.unique(seg)) - {0}))
EOF
```

期望：帧分数 ≥ 1.5、六类标签都出现、`overlay.png` 里软骨叠色落在关节面上。把 overlay 复制到 `~/figs/anatobind_knee/MTR_010_dicom_world_overlay.png`，报给用户两行（先 `http://localhost:8765/anatobind_knee/…png` 再绝对路径）。分数 < 1.5 或标签缺失 → 在 `dicom_world_frame.md` 记为"世界帧入口未验证"，不阻塞 Task 10，但 STATUS 里写明。

- [ ] **Step 3: 写记录并提交**

`dicom_world_frame.md`：来源 tar 名、转换命令、轴码与形状、帧分数、标签、overlay 路径、结论一行。

```bash
git add docs/verification/$(date +%F)/knee_eval/dicom_world_frame.md
git commit -m "Verification: world-frame entry point checked on one DICOM-derived SKM-TEA volume"
```

---

### Task 10: 报告、入口文件、合回 main 打 tag

**Files:**
- Create: `docs/verification/<YYYY-MM-DD>/knee_eval/REPORT.md`
- Modify: `STATUS.md`（整体重写五段）、`CLAUDE.md`（状态行、测试数、阅读顺序、代码地图）

- [ ] **Step 1: 写 REPORT.md**

按 `docs/verification/2026-09-08/REPORT.md` 的体例（每节：结论 → 命令 → 原始输出 → 支撑的决策），章节固定：

1. 结论一段：三条目标各一句话数字（解剖六类 Dice；病灶工作点的阈值、灵敏度（两种）、每卷假阳、每族；所在结构在命中上的组织族与侧别正确率，及给定标注框的对照数字），门的结论。
2. 训练事实：`grep -h 'Epoch time' logs/nnunet902_fold*.log | awk '{s+=$NF; n++} END {print n, s/n}'` 的轮数与均轮时；每折起止时间（日志首尾行）；`du -sh $nnUNet_results/Dataset902_SKMTEAlesion`。
3. 五折评估：`summary.md` 原样粘贴；`froc_clean.csv` 全表。
4. 退化视图附注：六个退化视图的工作点行。
5. 入口回归（Task 8 Step 10 的输出）与 DICOM 核查（Task 9）。
6. 未做与限制：韧带、积液无所在结构；第二臂未跑；脑侧未开始；本报告中"给定标注框"一列是对照不是能力。

- [ ] **Step 2: 更新 STATUS.md 与 CLAUDE.md**

STATUS.md 五段：已完成且已验证（带命令与输出摘录）、待用户拍板（第二臂是否跑；脑侧计划 2 开工；四张卡编号）、下一步（计划 2）、坑（npz 轴序、Dataset902 与 901 共折、`--frame` 开关、DA 进程数）、关键决定的为什么（指向 spec §1）。CLAUDE.md：状态行改为"膝侧能力系统已交付（tag …），下一步计划 2"；测试数改为实测；阅读顺序加 spec、计划 1、REPORT；代码地图加 `anatobind/nnunet/{lesion_labels,prepare_lesion}.py`、`anatobind/eval/{lesion_boxes,detection_metrics}.py`、`anatobind/infer/`、`scripts/{nnunet_prepare_lesion,eval_knee_folds,infer_knee}.py`。

- [ ] **Step 3: 全套测试、提交、合回 main、打 tag、推送**

```bash
cd /data0/congcong/code/Project_Doing/foundation_model
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider   # 期望 342 passed
git add docs/verification/$(date +%F)/knee_eval STATUS.md CLAUDE.md
git commit -m "Status $(date +%F): knee capability system delivered, plan 2 (brain masks) is next"
git checkout main && git merge --no-ff build/aur-system -m "Merge branch 'build/aur-system': knee capability system"
git tag -a handoff/$(date +%F)-knee-capability -m "Knee capability system: Dataset902 detector, fold evaluation, entry point"
git push origin main && git push origin handoff/$(date +%F)-knee-capability   # TLS 失败时前缀 https_proxy=http://127.0.0.1:7897
git ls-remote origin | grep -E 'main|knee-capability'
```

分支 `build/aur-system` 不自行删除；把 `git branch -d build/aur-system && git push origin --delete build/aur-system` 写进 STATUS §2 交用户。

- [ ] **Step 4: 汇报（决定 N6 的"膝五折"一报）**

一段话 + REPORT.md 路径 + `gate.json` 原文 + 图的两行地址。

---

## 时间线（四张 A800）

| 日 | 内容 |
|---|---|
| 1 | Task 1–2（代码 + 预处理 1–2 h + 启动四折）；烟雾门 |
| 1–2 | Task 3–6（CPU，训练期间并行） |
| 2 | fold 0 完成 → Task 7 Step 1 汇报；启动 fold 4 |
| 3 | 五折完成 → Task 7 Step 2–3 判门 |
| 3–4 | Task 8（含 GPU 回归一次）、Task 9 |
| 5 | Task 10 报告、交接 |

两周预算里余下的时间留给第二臂（若门不过）或提前进入计划 2。
