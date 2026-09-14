# Leg 2: fastMRI+ 膝关节病灶检测与重采闸门 · 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 fastMRI+ 膝关节上建立病灶检测与无参照的重采闸门，按事先写死的 H1/H2/H3 判定。

**Architecture:** 数据引擎把 974 卷多线圈 k 空间与 16154 个逐层 2D 框变成七视图导出树与 3D 病灶清单；2.5D 检测器（5 层薄块的 3D stem + MONAI Swin 2D + 2D 中心热图头）按患者五折训练并冻结；折外预测产生任务失败标签；两个可靠性头（逐病灶、扫描级）在折外标签上训练；闸门按风险–覆盖曲线定阈值。

**Tech Stack:** PyTorch 2.x、MONAI 1.5.2 SwinTransformer(spatial_dims=2)、h5py、numpy、scipy、matplotlib、pytest。

**Spec:** `docs/superpowers/specs/2026-09-14-leg2-fastmri-knee-detection-gate-design.md`

## Global Constraints

- 数据只从 `/data2/congcong/data/FM_data` 读；`/data0/congcong/data/FM_Data` 是冷备份，不读不写。第一条腿的 `derived/skmtea/` 只读，不改。
- 本条腿的导出根：`/data2/congcong/data/FM_data/derived/fastmri_knee/leg2/`。
- 解释器一律 `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python`。
- 测试一律 `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`。
- 提交在 main 上，用仓库本地身份，**任何 AI trailer 都不写**（不写 Co-Authored-By、不写 Generated with）。
- GPU 只用 0 和 1，每卡一个任务，`CUDA_VISIBLE_DEVICES` 钉死，后台用 `setsid nohup`；CPU 任务 `nice -n 19`，线程不超过 48，DataLoader worker 不超过 4。
- 任何预计单折超过 48 小时的任务，先问用户。
- 五个类别族：`("meniscus", "cartilage", "bone", "ligament", "effusion")`。
- 3D 合并的面内 IoU 阈值 0.3；检测与真值的 3D 匹配主阈值 IoU 0.1，另报 0.05 与 0.2。
- 噪声 `σ = c · median(|K|)`，`c = 1, 2, 4` 对应 q1/q2/q3；欠采 4×/8×/16×，中心全采比例 0.08/0.04/0.04，同卷三档掩膜嵌套。
- 七视图：`("clean", "noise_q1", "noise_q2", "noise_q3", "us4", "us8", "us16")`。
- 闸门风险目标 0.05；bootstrap 10000 次、种子 0、**按患者重采样**。
- **H1 是止损门。Task 10 不过就停下来交给用户，不要继续建闸门。**

---

### Task 1: 解决 spec §1.7 的两处未决事实

**Files:**
- Create: `docs/verification/2026-09-14/fastmri_plus_negatives.md`

**Interfaces:**
- Produces: 一个结论——未标注层能否作负样本（`UNANNOTATED_ARE_NEGATIVE = True/False`），以及 198 卷无标注卷的处置（纳入为正常膝 / 排除）。Task 7 的数据集按此结论取负样本。

- [ ] **Step 1: 读 fastMRI+ 的方法学**

用 WebSearch 与 WebFetch 找 fastMRI+ 的数据论文（Zhao et al., "fastMRI+: Clinical pathology annotations for knee and brain fully sampled magnetic resonance imaging data", Scientific Data 2022），读它的 Methods，回答两个问题：

1. 标注者是否逐层看完整卷？未出现标注的层是"确认无异常"还是"未审阅"？
2. 完全没有标注的卷是"读片为正常"还是"未标注"？

- [ ] **Step 2: 用数据交叉验证**

```bash
PYTHONNOUSERSITE=1 ~/anaconda3/envs/nvgen/bin/python - <<'PY'
import csv, glob, h5py, os
from collections import Counter
ann = list(csv.DictReader(open("/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/knee.csv")))
vols = {r["file"] for r in ann}
fs = {os.path.basename(f)[:-3] for f in glob.glob(
    "/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/knee/multicoil_*/*.h5")}
missing = sorted(fs - vols)
print(f"volumes with no annotation: {len(missing)}")
# 若无标注卷在两种采集上分布均衡，更像“读片正常”；若集中在一种，更像漏标
with h5py.File(glob.glob(f"/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/knee/multicoil_*/{missing[0]}.h5")[0]) as h:
    pass
acq = Counter()
for v in missing:
    p = glob.glob(f"/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/knee/multicoil_*/{v}.h5")[0]
    with h5py.File(p) as h:
        acq[h.attrs.get("acquisition", "?")] += 1
print("acquisition of unannotated volumes:", dict(acq))
PY
```

- [ ] **Step 3: 写结论**

把两个问题的答案、引用的原文句子、以及上一步的输出写进 `docs/verification/2026-09-14/fastmri_plus_negatives.md`，并给出明确结论：

- 若文献确认逐层审阅：`UNANNOTATED_ARE_NEGATIVE = True`，未标注层可作负样本，198 卷无标注卷作为全负样本纳入。
- 若无法确认：`UNANNOTATED_ARE_NEGATIVE = False`，退守——只用带标注的层作样本，某层对某类别族的负类取"该层标注了别的族但没标这一族"，198 卷排除。并在文件中写明这一限制会让假阳性率估计偏乐观。

- [ ] **Step 4: 提交**

```bash
git add docs/verification/2026-09-14/fastmri_plus_negatives.md
git commit -m "Verification: whether unannotated fastMRI+ slices count as negatives"
```

---

### Task 2: 标注读取、清洗与 3D 合并

**Files:**
- Create: `anatobind/data_engine/fastmri_knee.py`
- Test: `tests/test_fastmri_knee_boxes.py`

**Interfaces:**
- Produces: `FAMILIES`、`FAMILY_OF_LABEL`、`read_annotations(csv_path) -> list[dict]`、`clean_boxes(rows) -> (list[dict], Counter)`、`merge_to_3d(rows, iou_min=0.3) -> list[dict]`。3D 病灶字典的键：`file, family, z0, z1, x0, y0, x1, y1, n_boxes`（`z0`/`z1` 为层号闭区间，`x/y` 为 320×320 图像空间的像素，`x1`/`y1` 为开区间）。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_fastmri_knee_boxes.py
from collections import Counter

from anatobind.data_engine.fastmri_knee import FAMILIES, clean_boxes, merge_to_3d


def _box(file, slice_, x, y, w, h, family="meniscus"):
    return {"file": file, "slice": slice_, "x": x, "y": y, "width": w, "height": h, "family": family}


def test_degenerate_boxes_are_dropped_and_counted():
    rows = [_box("f1", 3, 10, 10, 20, 20), _box("f1", 4, 10, 10, 2, 20), _box("f1", 5, 10, 10, 20, 0)]
    kept, dropped = clean_boxes(rows)
    assert len(kept) == 1 and kept[0]["slice"] == 3
    assert dropped == Counter({"too_small": 2})


def test_overlapping_boxes_on_adjacent_slices_become_one_lesion():
    rows = [_box("f1", 3, 10, 10, 20, 20), _box("f1", 4, 11, 10, 20, 20), _box("f1", 5, 12, 11, 20, 20)]
    lesions = merge_to_3d(rows)
    assert len(lesions) == 1
    L = lesions[0]
    assert (L["z0"], L["z1"], L["n_boxes"]) == (3, 5, 3)
    assert (L["x0"], L["y0"], L["x1"], L["y1"]) == (10, 10, 32, 31)


def test_spatially_separate_runs_of_one_family_become_two_lesions():
    rows = [_box("f1", 3, 10, 10, 20, 20), _box("f1", 4, 10, 10, 20, 20),
            _box("f1", 3, 200, 200, 20, 20), _box("f1", 4, 200, 200, 20, 20)]
    lesions = merge_to_3d(rows)
    assert len(lesions) == 2
    assert sorted(L["x0"] for L in lesions) == [10, 200]


def test_a_slice_gap_breaks_a_lesion_in_two():
    rows = [_box("f1", 3, 10, 10, 20, 20), _box("f1", 5, 10, 10, 20, 20)]
    assert len(merge_to_3d(rows)) == 2


def test_a_single_slice_box_is_still_a_lesion():
    assert len(merge_to_3d([_box("f1", 7, 10, 10, 20, 20)])) == 1


def test_families_do_not_merge_into_each_other():
    rows = [_box("f1", 3, 10, 10, 20, 20, "meniscus"), _box("f1", 4, 10, 10, 20, 20, "cartilage")]
    lesions = merge_to_3d(rows)
    assert sorted(L["family"] for L in lesions) == ["cartilage", "meniscus"]
    assert set(FAMILIES) >= {"meniscus", "cartilage"}
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_knee_boxes.py -q -p no:cacheprovider
```
预期：`ModuleNotFoundError: No module named 'anatobind.data_engine.fastmri_knee'`

- [ ] **Step 3: 写实现**

```python
# anatobind/data_engine/fastmri_knee.py
"""fastMRI+ 膝关节的标注读取、清洗、3D 合并、患者划分与受控退化（leg 2 spec 第 1 节）。

标注是逐层 2D 框，坐标在 320x320 的 RSS 图像空间。3D 病灶由"相邻层、面内 IoU >= 0.3 相连"
的连通分量得到；同一类别族但空间分离的两串因此分成两个病灶。
"""
import csv
from collections import Counter, defaultdict

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
```

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_knee_boxes.py -q -p no:cacheprovider
```
预期：6 passed

- [ ] **Step 5: 在真实标注上跑一遍并记录计数**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python - <<'PY'
from collections import Counter
from anatobind.data_engine.fastmri_knee import clean_boxes, merge_to_3d, read_annotations
rows = read_annotations("/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/knee.csv")
kept, dropped = clean_boxes(rows)
lesions = merge_to_3d(kept)
print(f"rows kept {len(kept)}, dropped {dict(dropped)}")
print(f"3D lesions {len(lesions)}, by family {dict(Counter(L['family'] for L in lesions))}")
print(f"slices per lesion: median {sorted(L['z1']-L['z0']+1 for L in lesions)[len(lesions)//2]}")
PY
```
把输出贴进提交信息。

- [ ] **Step 6: 提交**

```bash
git add anatobind/data_engine/fastmri_knee.py tests/test_fastmri_knee_boxes.py
git commit -m "Data engine: fastMRI+ knee annotations, cleaning and 3D lesion merge"
```

---

### Task 3: 重建与退化算子

**Files:**
- Modify: `anatobind/data_engine/fastmri_knee.py`（追加）
- Test: `tests/test_fastmri_knee_degrade.py`

**Interfaces:**
- Consumes: 无
- Produces: `VIEWS`、`reconstruct_rss(kspace, size=320) -> np.ndarray`、`noise_sigma(kspace, c) -> float`、`equispaced_mask(n_pe, accel, centre_fraction, seed) -> np.ndarray[bool]`、`degrade(kspace, view, seed) -> np.ndarray`。`kspace` 形状 `(coils, ny, nx)`（单层）或 `(slices, coils, ny, nx)`；相位编码方向为**最后一维** `nx`（fastMRI 膝的 k 空间是 `(slices, coils, 640, 372)`，372 是相位编码）。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_fastmri_knee_degrade.py
import numpy as np
import pytest

from anatobind.data_engine.fastmri_knee import (
    VIEWS, degrade, equispaced_mask, noise_sigma, reconstruct_rss,
)


def _kspace(seed=0, coils=4, ny=64, nx=40):
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(coils, ny, nx)) + 1j * rng.normal(size=(coils, ny, nx))
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def test_the_seven_views_are_the_same_as_leg_one():
    assert VIEWS == ("clean", "noise_q1", "noise_q2", "noise_q3", "us4", "us8", "us16")


def test_reconstruction_centre_crops_to_the_requested_size():
    out = reconstruct_rss(_kspace(), size=32)
    assert out.shape == (32, 32) and out.dtype == np.float32 and (out >= 0).all()


def test_the_clean_view_is_the_reconstruction_itself():
    k = _kspace()
    np.testing.assert_allclose(reconstruct_rss(degrade(k, "clean", seed=0), size=32),
                               reconstruct_rss(k, size=32), rtol=0, atol=0)


def test_noise_grows_with_the_quality_level():
    k = _kspace()
    ref = reconstruct_rss(k, size=32)
    errs = [np.linalg.norm(reconstruct_rss(degrade(k, v, seed=0), size=32) - ref) for v in
            ("noise_q1", "noise_q2", "noise_q3")]
    assert errs[0] < errs[1] < errs[2]
    assert noise_sigma(k, 2.0) == pytest.approx(2 * noise_sigma(k, 1.0))


def test_the_undersampling_masks_are_nested_and_keep_the_centre():
    m4 = equispaced_mask(100, 4, 0.08, seed=0)
    m8 = equispaced_mask(100, 8, 0.04, seed=0)
    m16 = equispaced_mask(100, 16, 0.04, seed=0)
    assert m4.sum() > m8.sum() > m16.sum()
    assert (m16 <= m8).all() and (m8 <= m4).all()          # nested
    assert m16[48:52].all()                                 # the centre lines survive every level
    assert equispaced_mask(100, 4, 0.08, seed=0).tolist() == equispaced_mask(100, 4, 0.08, seed=0).tolist()


def test_undersampling_zeroes_the_unsampled_phase_encode_lines():
    k = _kspace()
    out = degrade(k, "us8", seed=0)
    kept = np.abs(out).sum(axis=(0, 1)) > 0
    assert kept.sum() < k.shape[-1]
    np.testing.assert_allclose(out[..., kept], k[..., kept])
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_knee_degrade.py -q -p no:cacheprovider
```
预期：`ImportError: cannot import name 'VIEWS'`

- [ ] **Step 3: 写实现（追加到 `anatobind/data_engine/fastmri_knee.py`）**

```python
import numpy as np

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


def equispaced_mask(n_pe, accel, centre_fraction, seed):
    """1D 相位编码掩膜。中心全采，外围按种子固定的排列取前缀，因此不同加速倍数的掩膜互相嵌套。"""
    mask = np.zeros(n_pe, dtype=bool)
    n_centre = int(round(centre_fraction * n_pe))
    lo = (n_pe - n_centre) // 2
    mask[lo:lo + n_centre] = True
    n_total = int(round(n_pe / accel))
    n_extra = max(0, n_total - n_centre)
    rest = np.array([i for i in range(n_pe) if not mask[i]])
    order = np.random.default_rng(seed).permutation(len(rest))
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
```

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_knee_degrade.py -q -p no:cacheprovider
```
预期：6 passed

- [ ] **Step 5: 在真实卷上核对重建算子**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python - <<'PY'
import glob, h5py, numpy as np
from anatobind.data_engine.fastmri_knee import reconstruct_rss
f = sorted(glob.glob("/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/knee/multicoil_train/*.h5"))[0]
with h5py.File(f) as h:
    k, ref = h["kspace"][10], h["reconstruction_rss"][10]
out = reconstruct_rss(k)
print(f"NRMSE vs reconstruction_rss: {np.linalg.norm(out - ref) / np.linalg.norm(ref):.2e}")
PY
```
预期：小于 1e-6。不满足就先查 fftshift 约定，不要继续。

- [ ] **Step 6: 提交**

```bash
git add anatobind/data_engine/fastmri_knee.py tests/test_fastmri_knee_degrade.py
git commit -m "Data engine: one RSS reconstruction operator shared by the clean and degraded views"
```

---

### Task 4: 患者划分与导出

**Files:**
- Modify: `anatobind/data_engine/fastmri_knee.py`（追加）
- Create: `scripts/build_fastmri_knee.py`
- Test: `tests/test_fastmri_knee_export.py`

**Interfaces:**
- Consumes: Task 2 的 `read_annotations/clean_boxes/merge_to_3d`，Task 3 的 `VIEWS/reconstruct_rss/degrade`
- Produces: `volume_paths(root) -> dict[str, Path]`、`patient_of(paths) -> dict[str, str]`、`make_folds(patients, k=5, seed=0) -> dict[str, int]`、`export_volume(h5_path, lesions, out_dir, seed) -> dict`。导出树：`leg2/<file>/{clean,noise_q1,...,us16}.npy`（float16，形状 `(slices, 320, 320)`）、`leg2/<file>/meta.json`、根目录 `leg2/lesions.csv`、`leg2/folds.json`、`leg2/manifest.csv`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_fastmri_knee_export.py
import json

import h5py
import numpy as np

from anatobind.data_engine.fastmri_knee import VIEWS, export_volume, make_folds


def _write_h5(path, slices=3, coils=2, ny=64, nx=40, patient="p1", seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(slices, coils, ny, nx)) + 1j * rng.normal(size=(slices, coils, ny, nx))
    k = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
    with h5py.File(path, "w") as h:
        h.create_dataset("kspace", data=k.astype(np.complex64))
        h.attrs["patient_id"] = patient
        h.attrs["acquisition"] = "CORPD_FBK"
    return path


def test_every_volume_of_a_patient_lands_in_one_fold():
    patients = {f"file{i}": f"p{i // 2}" for i in range(20)}   # two volumes per patient
    folds = make_folds(patients, k=5, seed=0)
    by_patient = {}
    for vol, f in folds.items():
        by_patient.setdefault(patients[vol], set()).add(f)
    assert all(len(v) == 1 for v in by_patient.values())
    assert set(folds.values()) == {0, 1, 2, 3, 4}


def test_export_writes_every_view_and_a_readable_meta(tmp_path):
    h5 = _write_h5(tmp_path / "file1.h5")
    lesions = [{"file": "file1", "family": "meniscus", "z0": 1, "z1": 2,
                "x0": 4, "y0": 5, "x1": 12, "y1": 15, "n_boxes": 2}]
    row = export_volume(h5, lesions, tmp_path / "out" / "file1", seed=0, size=32)
    for v in VIEWS:
        a = np.load(tmp_path / "out" / "file1" / f"{v}.npy")
        assert a.shape == (3, 32, 32) and a.dtype == np.float16
    meta = json.loads((tmp_path / "out" / "file1" / "meta.json").read_text())
    assert meta["slices"] == 3 and meta["patient_id"] == "p1" and meta["n_lesions"] == 1
    assert row["status"] == "ok"


def test_the_degraded_views_differ_from_clean_but_share_its_shape(tmp_path):
    h5 = _write_h5(tmp_path / "file2.h5", patient="p2")
    export_volume(h5, [], tmp_path / "out" / "file2", seed=0, size=32)
    clean = np.load(tmp_path / "out" / "file2" / "clean.npy").astype(np.float32)
    for v in ("noise_q3", "us16"):
        other = np.load(tmp_path / "out" / "file2" / f"{v}.npy").astype(np.float32)
        assert other.shape == clean.shape and not np.allclose(other, clean)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_knee_export.py -q -p no:cacheprovider
```
预期：`ImportError: cannot import name 'export_volume'`

- [ ] **Step 3: 写实现（追加到 `anatobind/data_engine/fastmri_knee.py`）**

```python
import json
from pathlib import Path

import h5py

from anatobind.train.dataset import normalise_volume

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
    import numpy as np
    uniq = sorted(set(patients.values()))
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of_patient = {uniq[int(j)]: i % k for i, j in enumerate(order)}
    return {vol: fold_of_patient[p] for vol, p in patients.items()}


def export_volume(h5_path, lesions, out_dir, seed, size=320):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(h5_path) as h:
            kspace = h["kspace"][()]
            meta = {"patient_id": str(h.attrs.get("patient_id", "")),
                    "acquisition": str(h.attrs.get("acquisition", ""))}
        for view in VIEWS:
            vol = np.stack([reconstruct_rss(degrade(kspace[s], view, seed + i), size)
                            for i, s in enumerate(range(kspace.shape[0]))])
            np.save(out_dir / f"{view}.npy", normalise_volume(vol).astype(np.float16))
        meta.update({"slices": int(kspace.shape[0]), "size": size, "n_lesions": len(lesions), "seed": seed})
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        return {"file": Path(h5_path).stem, "out_dir": str(out_dir), "slices": meta["slices"],
                "n_lesions": len(lesions), "status": "ok"}
    except Exception as exc:
        return {"file": Path(h5_path).stem, "out_dir": str(out_dir), "slices": 0, "n_lesions": 0,
                "status": f"error: {type(exc).__name__}: {exc}"[:200]}
```

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_knee_export.py -q -p no:cacheprovider
```
预期：3 passed

- [ ] **Step 5: 写导出脚本**

```python
#!/usr/bin/env python
# scripts/build_fastmri_knee.py
"""把 fastMRI+ 膝关节导出成 leg 2 的七视图训练树。

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/build_fastmri_knee.py --workers 4
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import (  # noqa: E402
    ANNOTATIONS, EXPORT_ROOT, clean_boxes, export_volume, make_folds, merge_to_3d,
    patient_of, read_annotations, volume_paths,
)

LESION_FIELDS = ["lesion_id", "file", "family", "z0", "z1", "x0", "y0", "x1", "y1", "n_boxes"]
MANIFEST_FIELDS = ["file", "out_dir", "slices", "n_lesions", "status"]


def job(args):
    name, path, lesions, out_root, seed = args
    os.nice(19)
    return export_volume(path, lesions, Path(out_root) / name, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="export only the first N volumes (smoke test)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = read_annotations(ANNOTATIONS)
    kept, dropped = clean_boxes(rows)
    lesions = merge_to_3d(kept)
    with open(a.out / "lesions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LESION_FIELDS)
        w.writeheader()
        for i, L in enumerate(sorted(lesions, key=lambda L: (L["file"], L["family"], L["z0"], L["x0"]))):
            w.writerow({"lesion_id": i, **{k: L[k] for k in LESION_FIELDS[1:]}})
    print(f"boxes kept {len(kept)}, dropped {dict(dropped)}; 3D lesions {len(lesions)}; "
          f"by family {dict(Counter(L['family'] for L in lesions))}")

    paths = volume_paths()
    names = sorted(paths)[:a.limit] if a.limit else sorted(paths)
    folds = make_folds(patient_of({n: paths[n] for n in names}))
    (a.out / "folds.json").write_text(json.dumps({"folds": folds}, indent=1))

    by_file = {}
    for L in lesions:
        by_file.setdefault(L["file"], []).append(L)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(job, [(n, paths[n], by_file.get(n, []), a.out, 1000 + i)
                                    for i, n in enumerate(names)]))
    with open(a.out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    bad = [r["file"] for r in results if r["status"] != "ok"]
    print(f"{len(results)} volumes, {len(bad)} failed: {bad[:10]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 先跑 8 卷的冒烟测试**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/build_fastmri_knee.py --workers 4 --limit 8 --out /tmp/leg2_smoke
du -sh /tmp/leg2_smoke
```
预期：8 卷全 ok。用单卷体积乘 1172 估算全量磁盘占用；若超过 200 GB，先问用户再跑全量。

- [ ] **Step 7: 跑全量导出**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. setsid nohup nice -n 19 ~/anaconda3/envs/nvgen/bin/python \
  scripts/build_fastmri_knee.py --workers 4 > logs/build_fastmri_knee.log 2>&1 &
```

- [ ] **Step 8: 提交**

```bash
git add anatobind/data_engine/fastmri_knee.py scripts/build_fastmri_knee.py tests/test_fastmri_knee_export.py
git commit -m "Data engine: patient-level folds and the seven-view fastMRI knee export"
```

---

### Task 5: 2D 中心热图头

**Files:**
- Create: `anatobind/model/dense_head_2d.py`
- Test: `tests/test_dense_head_2d.py`

**Interfaces:**
- Consumes: 无
- Produces: `STRIDE = 2`、`CentreHead2D(c1, num_classes=5, hidden=64, embed_dim=256)`、`centre_targets_2d(boxes, classes, grid, num_classes, device)`、`centre_loss_2d(out, batch)`、`decode_centres_2d(out, M)`。`boxes` 为 `(N, 4)` 的 `(y0, x0, y1, x1)` 像素坐标；`out` 的键为 `heat (B,C,H,W)`、`offset (B,2,H,W)`、`size (B,2,H,W)`、`feat (B,E,H,W)`。`decode_centres_2d` 返回 `(boxes (M,4), cls_prob (M,C+1), embed (M,E))`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_dense_head_2d.py
import pytest

torch = pytest.importorskip("torch")

from anatobind.model.dense_head_2d import CentreHead2D, centre_loss_2d, centre_targets_2d, decode_centres_2d

GRID = (16, 16)                                  # 32x32 输入在 stride 2 下的 F1 网格
BOX = torch.tensor([[8.0, 12.0, 20.0, 24.0]])    # y0, x0, y1, x1 -> 中心格 (6, 8)，边长 6 格


def test_the_head_predicts_on_the_f1_grid_with_a_low_prior():
    head = CentreHead2D(c1=16, num_classes=5, hidden=16, embed_dim=32)
    out = head(torch.randn(2, 16, *GRID))
    assert out["heat"].shape == (2, 5, *GRID) and out["offset"].shape == (2, 2, *GRID)
    assert out["size"].shape == (2, 2, *GRID) and out["feat"].shape == (2, 32, *GRID)
    assert abs(float(out["heat"].sigmoid().mean()) - 0.1) < 0.05


def test_targets_put_a_unit_peak_on_the_centre_cell_of_the_right_class():
    heat, cells, offsets, sizes = centre_targets_2d(BOX, [3], GRID, 5, "cpu")
    assert heat.shape == (5, *GRID) and float(heat[3, 6, 8]) == 1.0 and float(heat[0].max()) == 0.0
    assert cells[0].tolist() == [6, 8]
    torch.testing.assert_close(offsets[0], torch.zeros(2))
    torch.testing.assert_close(sizes[0], torch.log(torch.full((2,), 6.0)))


def test_decoding_recovers_the_box_class_and_embedding():
    out = {"heat": torch.full((1, 5, *GRID), -10.0), "offset": torch.zeros(1, 2, *GRID),
           "size": torch.zeros(1, 2, *GRID), "feat": torch.arange(4 * 256, dtype=torch.float32).reshape(1, 4, *GRID)}
    out["heat"][0, 2, 6, 8] = 10.0
    out["size"][0, :, 6, 8] = torch.log(torch.tensor(6.0))
    boxes, cls_prob, embed = decode_centres_2d(out, M=2)
    torch.testing.assert_close(boxes[0], torch.tensor([7.0, 11.0, 19.0, 23.0]))
    assert int(cls_prob[0].argmax()) == 2 and int(cls_prob[1].argmax()) == 5
    torch.testing.assert_close(cls_prob.sum(-1), torch.ones(2))
    torch.testing.assert_close(embed[0], out["feat"][0, :, 6, 8])


def test_the_loss_prefers_the_right_heatmap():
    heat_t, _, _, _ = centre_targets_2d(BOX, [3], GRID, 5, "cpu")
    good = {"heat": torch.where(heat_t == 1, 8.0, -8.0)[None], "offset": torch.zeros(1, 2, *GRID),
            "size": torch.log(torch.full((1, 2, *GRID), 6.0))}
    bad = {**good, "heat": torch.zeros(1, 5, *GRID)}
    batch = {"boxes": [BOX], "box_classes": [torch.tensor([3])]}
    lg, lb = centre_loss_2d(good, batch), centre_loss_2d(bad, batch)
    assert float(lg["heat"]) < 0.1 * float(lb["heat"])
    assert float(lg["offset"]) == 0.0 and float(lg["size"]) == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_dense_head_2d.py -q -p no:cacheprovider
```
预期：`ModuleNotFoundError: No module named 'anatobind.model.dense_head_2d'`

- [ ] **Step 3: 写实现**

```python
# anatobind/model/dense_head_2d.py
"""2D 中心热图检测头（leg 2 spec 2.2），是 anatobind/model/dense_head.py 的二维对应。

单独写一个模块而不是把 3D 版本泛化：第一条腿已冻结并产出结论，为省一百行去改它风险与收益不对等。
格 i 覆盖像素 [i*s, (i+1)*s)；框中心 c 落在格坐标 c/s - 0.5，峰放最近格，偏移是余数，尺寸取以格为单位的对数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 2
PRIOR_BIAS = -2.19  # sigmoid(-2.19) = 0.1


class CentreHead2D(nn.Module):
    def __init__(self, c1, num_classes=5, hidden=64, embed_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(c1, hidden, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.heat = nn.Conv2d(hidden, num_classes, 1)
        self.offset = nn.Conv2d(hidden, 2, 1)
        self.size = nn.Conv2d(hidden, 2, 1)
        self.embed = nn.Conv2d(hidden, embed_dim, 1)
        nn.init.constant_(self.heat.bias, PRIOR_BIAS)

    def forward(self, f1):
        h = self.trunk(f1)
        return {"heat": self.heat(h), "offset": self.offset(h), "size": self.size(h), "feat": self.embed(h)}


def centre_targets_2d(boxes, classes, grid, num_classes, device):
    heat = torch.zeros(num_classes, *grid, device=device)
    limit = torch.tensor(grid, dtype=torch.float32, device=device) - 1
    yy, xx = torch.meshgrid(*(torch.arange(n, dtype=torch.float32, device=device) for n in grid), indexing="ij")
    cells, offsets, sizes = [], [], []
    for box, cls in zip(boxes.float().to(device), classes):
        centre = (box[:2] + box[2:]) / 2 / STRIDE - 0.5
        extent = (box[2:] - box[:2]).clamp(min=1.0) / STRIDE
        cell = torch.minimum(centre.round().clamp(min=0), limit).long()
        sigma = (extent / 6).clamp(min=0.5)
        g = torch.exp(-0.5 * (((yy - centre[0]) / sigma[0]) ** 2 + ((xx - centre[1]) / sigma[1]) ** 2))
        heat[cls] = torch.maximum(heat[cls], g)
        heat[cls, cell[0], cell[1]] = 1.0
        cells.append(cell)
        offsets.append(centre - cell)
        sizes.append(extent.log())
    return heat, cells, offsets, sizes


def centre_loss_2d(out, batch):
    heat = out["heat"].float()
    B, C, H, W = heat.shape
    p = heat.sigmoid().clamp(1e-4, 1 - 1e-4)
    total = {k: heat.new_zeros(()) for k in ("heat", "offset", "size")}
    n_obj = 0
    for b in range(B):
        tgt, cells, offsets, sizes = centre_targets_2d(batch["boxes"][b], batch["box_classes"][b].tolist(),
                                                       (H, W), C, heat.device)
        pos = tgt.eq(1.0)
        total["heat"] = total["heat"] - (torch.log(p[b]) * (1 - p[b]) ** 2)[pos].sum() \
            - (torch.log(1 - p[b]) * p[b] ** 2 * (1 - tgt) ** 4)[~pos].sum()
        for cell, off, size in zip(cells, offsets, sizes):
            y, x = cell.tolist()
            total["offset"] = total["offset"] + (out["offset"][b, :, y, x].float() - off).abs().sum()
            total["size"] = total["size"] + (out["size"][b, :, y, x].float() - size).abs().sum()
        n_obj += len(cells)
    n = max(n_obj, 1)
    return {k: v / n for k, v in total.items()}


def decode_centres_2d(out, M):
    """第一个样本的前 M 个峰 -> 框 (M,4) 像素、类别概率 (M,C+1)、embedding (M,E)。

    槽位的 no-object 概率写成 1 - 峰值得分，因此"最可能的类别不是 no-object"等价于"峰值得分 >= 0.5"。
    """
    heat = out["heat"][0].float().sigmoid()
    C, H, W = heat.shape
    keep = heat == F.max_pool2d(heat[None], 3, stride=1, padding=1)[0]
    scores, idx = (heat * keep).flatten().topk(M)
    cls = idx // (H * W)
    rest = idx % (H * W)
    y, x = rest // W, rest % W
    offset = out["offset"][0][:, y, x].float().T
    size = out["size"][0][:, y, x].float().T.exp() * STRIDE
    centre = (torch.stack([y, x], 1).float() + offset + 0.5) * STRIDE
    boxes = torch.cat([centre - size / 2, centre + size / 2], 1)
    cls_prob = torch.zeros(M, C + 1, device=heat.device)
    cls_prob[torch.arange(M, device=heat.device), cls] = scores
    cls_prob[:, C] = 1 - scores
    return boxes, cls_prob, out["feat"][0][:, y, x].T
```

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_dense_head_2d.py -q -p no:cacheprovider
```
预期：4 passed

- [ ] **Step 5: 提交**

```bash
git add anatobind/model/dense_head_2d.py tests/test_dense_head_2d.py
git commit -m "Model: a 2D centre-heatmap head for the fastMRI knee detector"
```

---

### Task 6: 2.5D 检测器

**Files:**
- Create: `anatobind/model/detector2d.py`
- Test: `tests/test_detector2d.py`

**Interfaces:**
- Consumes: Task 5 的 `CentreHead2D`
- Produces: `Detector2D(num_classes=5, embed_dim=48, stem_ch=32, d_model=256, slab=5, use_checkpoint=False)`，`forward(slab) -> dict`，`slab` 形状 `(B, 1, 5, H, W)`，返回键 `heat/offset/size/feat`（F1 网格，stride 2）加 `global_feat (B, Cg)`（最深一级的全局平均池化，供扫描级可靠性头用）。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_detector2d.py
import inspect

import pytest

torch = pytest.importorskip("torch")

from anatobind.model.detector2d import Detector2D

SLAB = (1, 1, 5, 64, 64)


def test_forward_takes_the_slab_and_nothing_else():
    """没有真值能进入预测。"""
    assert list(inspect.signature(Detector2D.forward).parameters) == ["self", "slab"]


def test_output_is_on_the_stride_two_grid_plus_a_global_vector():
    torch.manual_seed(0)
    m = Detector2D(num_classes=5, embed_dim=12, stem_ch=8, d_model=16).eval()
    out = m(torch.randn(*SLAB))
    assert out["heat"].shape == (1, 5, 32, 32) and out["offset"].shape == (1, 2, 32, 32)
    assert out["size"].shape == (1, 2, 32, 32) and out["feat"].shape == (1, 16, 32, 32)
    assert out["global_feat"].ndim == 2 and out["global_feat"].shape[0] == 1


def test_the_stem_actually_reads_every_slice_of_the_slab():
    torch.manual_seed(0)
    m = Detector2D(num_classes=5, embed_dim=12, stem_ch=8, d_model=16).eval()
    x = torch.randn(*SLAB)
    with torch.no_grad():
        a = m(x)["heat"]
        x2 = x.clone()
        x2[:, :, 0] += 5.0                      # change only the first slice
        b = m(x2)["heat"]
    assert not torch.allclose(a, b)


def test_gradients_reach_the_stem_and_the_head():
    m = Detector2D(num_classes=5, embed_dim=12, stem_ch=8, d_model=16)
    m(torch.randn(*SLAB))["heat"].square().mean().backward()
    for p in (m.stem[0].weight, m.head.heat.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_detector2d.py -q -p no:cacheprovider
```
预期：`ModuleNotFoundError: No module named 'anatobind.model.detector2d'`

- [ ] **Step 3: 写实现**

```python
# anatobind/model/detector2d.py
"""2.5D 病灶检测器（leg 2 spec 2.1）：5 层薄块经 3D stem 压成 2D，再走 MONAI Swin 2D。

不用全 3D 主干：只有 5 层，stem 步长 2 加三次 patch merging 会把深度降到 0。第一条腿已经证明
瓶颈在样本效率而非网络容量，2.5D 让每个标注层都是一个样本（约 8800 个），而非合并后的约 3000 个 3D 病灶。
"""
import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer

from anatobind.model.dense_head_2d import CentreHead2D


class Detector2D(nn.Module):
    def __init__(self, num_classes=5, embed_dim=48, stem_ch=32, d_model=256, slab=5, use_checkpoint=False):
        super().__init__()
        self.slab = slab
        self.stem = nn.Sequential(
            nn.Conv3d(1, stem_ch, (slab, 3, 3), padding=(0, 1, 1)), nn.GELU(),
            nn.Conv3d(stem_ch, stem_ch, (1, 3, 3), padding=(0, 1, 1)), nn.GELU(),
        )
        self.swin = SwinTransformer(
            in_chans=stem_ch, embed_dim=embed_dim, window_size=(8, 8), patch_size=(2, 2),
            depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), spatial_dims=2, use_checkpoint=use_checkpoint,
        )
        del self.swin.layers4          # F1..F4 止于 stride 16；最后一级是死重量
        self.channels = tuple(embed_dim * 2 ** i for i in range(4))
        self.head = CentreHead2D(c1=self.channels[0], num_classes=num_classes, embed_dim=d_model)

    def forward(self, slab):
        x = self.stem(slab).squeeze(2)                      # (B, stem_ch, H, W)
        s = self.swin
        x0 = s.pos_drop(s.patch_embed(x))
        f1 = s.proj_out(x0, True)
        x1 = s.layers1[0](x0.contiguous())
        x2 = s.layers2[0](x1.contiguous())
        x3 = s.layers3[0](x2.contiguous())
        f4 = s.proj_out(x3, True)
        return {**self.head(f1), "global_feat": f4.mean(dim=(-2, -1))}
```

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_detector2d.py -q -p no:cacheprovider
```
预期：4 passed

- [ ] **Step 5: 量一次 320×320 下的显存与速度**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python - <<'PY'
import time, torch
from anatobind.model.detector2d import Detector2D
m = Detector2D().cuda(); opt = torch.optim.AdamW(m.parameters(), 2e-4)
x = torch.randn(16, 1, 5, 320, 320, device="cuda")
for i in range(8):
    if i == 3: torch.cuda.synchronize(); t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(x)
    out["heat"].square().mean().backward(); opt.step(); opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()
print(f"batch 16 slabs: {(time.time()-t0)/5:.3f} s/step, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB, "
      f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M params")
PY
```
把每步耗时写进提交信息。若按计划的步数换算单折超过 48 小时，先问用户。

- [ ] **Step 6: 提交**

```bash
git add anatobind/model/detector2d.py tests/test_detector2d.py
git commit -m "Model: the 2.5D fastMRI knee detector"
```

---

### Task 7: 薄块数据集

**Files:**
- Create: `anatobind/train/dataset_knee.py`
- Test: `tests/test_dataset_knee.py`

**Interfaces:**
- Consumes: Task 4 的导出树，Task 1 的 `UNANNOTATED_ARE_NEGATIVE` 结论
- Produces: `SlabDataset(files, export_root, lesions, train=True, views=VIEWS, p_clean=0.5, slab=5, seed=0)`、`collate_slabs(samples)`、`seed_worker(worker_id)`。批字典的键：`image (B,1,5,H,W)`、`boxes list[(N,4)]`、`box_classes list[(N,)]`、`file list[str]`、`slice list[int]`、`view list[str]`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_dataset_knee.py
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.data_engine.fastmri_knee import VIEWS
from anatobind.train.dataset_knee import SlabDataset, collate_slabs


def _export(tmp_path, name="file1", slices=9, size=32):
    d = tmp_path / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for v in VIEWS:
        np.save(d / f"{v}.npy", rng.normal(size=(slices, size, size)).astype(np.float16))
    (d / "meta.json").write_text(json.dumps({"slices": slices, "size": size, "patient_id": "p1"}))
    lesions = [{"lesion_id": 0, "file": name, "family": "meniscus", "z0": 3, "z1": 5,
                "x0": 4, "y0": 6, "x1": 14, "y1": 18, "n_boxes": 3}]
    return d.parent, lesions


def test_a_sample_is_a_slab_centred_on_its_slice(tmp_path):
    root, lesions = _export(tmp_path)
    ds = SlabDataset(["file1"], root, lesions=lesions, train=False, views=("clean",), slab=5)
    s = ds[0]
    assert s["image"].shape == (1, 5, 32, 32)
    assert s["view"] == "clean" and 0 <= s["slice"] < 9


def test_slabs_at_the_volume_edge_replicate_rather_than_wrap(tmp_path):
    root, lesions = _export(tmp_path)
    ds = SlabDataset(["file1"], root, lesions=lesions, train=False, views=("clean",), slab=5)
    first = [s for s in (ds[i] for i in range(len(ds))) if s["slice"] == 0][0]["image"][0]
    assert first.shape == (5, 32, 32)
    # slice 0 draws indices clip([-2,-1,0,1,2]) = [0,0,0,1,2]: the first three planes are the same
    torch.testing.assert_close(first[0], first[1])
    torch.testing.assert_close(first[1], first[2])
    assert not torch.allclose(first[2], first[3])
    last = [s for s in (ds[i] for i in range(len(ds))) if s["slice"] == 8][0]["image"][0]
    torch.testing.assert_close(last[3], last[4])           # the far edge replicates too, it does not wrap


def test_a_slice_inside_a_lesion_carries_its_box(tmp_path):
    root, lesions = _export(tmp_path)
    ds = SlabDataset(["file1"], root, lesions=lesions, train=False, views=("clean",), slab=5)
    boxed = [s for s in (ds[i] for i in range(len(ds))) if len(s["boxes"])]
    assert boxed, "slices 3-5 must carry the lesion box"
    assert {int(s["slice"]) for s in boxed} == {3, 4, 5}
    b = boxed[0]["boxes"][0]
    assert b.tolist() == [6.0, 4.0, 18.0, 14.0]                 # y0, x0, y1, x1
    assert int(boxed[0]["box_classes"][0]) == 0                 # meniscus is family index 0


def test_collate_stacks_slabs_and_keeps_boxes_per_sample(tmp_path):
    root, lesions = _export(tmp_path)
    ds = SlabDataset(["file1"], root, lesions=lesions, train=False, views=("clean",), slab=5)
    b = collate_slabs([ds[0], ds[1]])
    assert b["image"].shape == (2, 1, 5, 32, 32)
    assert len(b["boxes"]) == 2 and len(b["file"]) == 2


def test_training_mode_samples_views_and_can_flip(tmp_path):
    root, lesions = _export(tmp_path)
    ds = SlabDataset(["file1"], root, lesions=lesions, train=True, p_clean=0.0, seed=0)
    views = {ds[i]["view"] for i in range(30)}
    assert views and "clean" not in views                       # p_clean=0 never draws the clean view
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_dataset_knee.py -q -p no:cacheprovider
```
预期：`ModuleNotFoundError: No module named 'anatobind.train.dataset_knee'`

- [ ] **Step 3: 写实现**

```python
# anatobind/train/dataset_knee.py
"""leg 2 的薄块数据集：一个样本 = 某视图下以某层为中心的 5 层薄块，只在中心层监督。

增广是这条腿的重点。第一条腿用 124 个整卷、只有一次翻转，检测头因此背下了训练集
（docs/verification/2026-09-12/fold0_pilot.md 第 6 节）。这里每个标注层都是一个样本，
并加入左右翻转、平移与强度扰动。
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
        self.index = [(f, z) for f in self.files for z in range(self.meta[f]["slices"])]
        if not train:
            self.index = [(f, z) for (f, z) in self.index]

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
```

**负样本的取法**：若 Task 1 的结论是 `UNANNOTATED_ARE_NEGATIVE = True`，上面的 `self.index` 覆盖全部层即可（无框的层自然是负样本）。若结论为 `False`，把 `self.index` 改成只保留 `self.by_slice` 里出现过的层：

```python
        annotated = {(f, z) for (f, z) in self.by_slice}
        self.index = sorted(annotated) if not UNANNOTATED_ARE_NEGATIVE else \
            [(f, z) for f in self.files for z in range(self.meta[f]["slices"])]
```

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_dataset_knee.py -q -p no:cacheprovider
```
预期：5 passed

- [ ] **Step 5: 提交**

```bash
git add anatobind/train/dataset_knee.py tests/test_dataset_knee.py
git commit -m "Training: the slab dataset with the augmentation leg 1 lacked"
```

---

### Task 8: 检测器训练

**Files:**
- Create: `anatobind/train/train_detector.py`, `scripts/train_detector.py`
- Test: `tests/test_train_detector.py`

**Interfaces:**
- Consumes: Task 6 的 `Detector2D`，Task 5 的 `centre_loss_2d`，Task 7 的 `SlabDataset/collate_slabs/seed_worker`
- Produces: `FULL`、`TINY`、`lr_lambda(warmup, total)`、`main(argv)`。产物 `runs/detector_fold{f}/{config.json, metrics.jsonl, ckpt.pt, last.pt}`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_train_detector.py
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.data_engine.fastmri_knee import VIEWS


def _export(root, files, slices=9, size=32):
    import csv
    rng = np.random.default_rng(0)
    rows = []
    for i, name in enumerate(files):
        d = root / name
        d.mkdir(parents=True)
        for v in VIEWS:
            np.save(d / f"{v}.npy", rng.normal(size=(slices, size, size)).astype(np.float16))
        (d / "meta.json").write_text(json.dumps({"slices": slices, "size": size, "patient_id": f"p{i}"}))
        rows.append({"lesion_id": i, "file": name, "family": "meniscus", "z0": 3, "z1": 5,
                     "x0": 4, "y0": 6, "x1": 14, "y1": 18, "n_boxes": 3})
    with open(root / "lesions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (root / "folds.json").write_text(json.dumps({"folds": {f: i % 5 for i, f in enumerate(files)}}))


def test_training_runs_logs_and_writes_a_resumable_checkpoint(tmp_path):
    from anatobind.train.train_detector import main
    root = tmp_path / "leg2"
    root.mkdir()
    _export(root, [f"file{i}" for i in range(5)])
    out = tmp_path / "run"
    main(["--fold", "0", "--steps", "3", "--batch", "2", "--out", str(out), "--export-root", str(root),
          "--tiny", "--cpu", "--workers", "0", "--ckpt-every", "2", "--log-every", "1"])
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    assert len(rows) == 3 and all(np.isfinite(r["loss"]) for r in rows)
    assert (out / "last.pt").exists() and (out / "ckpt.pt").exists()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["fold"] == 0 and cfg["n_train"] == 4
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert last["step"] == 3 and last["config"]["model"]["num_classes"] == 5


def test_resume_continues_from_the_checkpoint(tmp_path):
    from anatobind.train.train_detector import main
    root = tmp_path / "leg2"
    root.mkdir()
    _export(root, [f"file{i}" for i in range(5)])
    out = tmp_path / "run"
    args = ["--fold", "0", "--batch", "2", "--out", str(out), "--export-root", str(root),
            "--tiny", "--cpu", "--workers", "0", "--ckpt-every", "1", "--log-every", "1"]
    main(args + ["--steps", "2"])
    main(args + ["--steps", "4", "--resume"])
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_train_detector.py -q -p no:cacheprovider
```
预期：`ModuleNotFoundError: No module named 'anatobind.train.train_detector'`

- [ ] **Step 3: 写实现**

```python
# anatobind/train/train_detector.py
"""leg 2 检测器的单折训练。固定步数、最后一个检查点、不做验证：留出患者上不选任何东西。

  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python \
      scripts/train_detector.py --fold 0 --out runs/detector_fold0 --resume
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from anatobind.data_engine.fastmri_knee import EXPORT_ROOT
from anatobind.model.dense_head_2d import centre_loss_2d
from anatobind.model.detector2d import Detector2D
from anatobind.train.dataset_knee import SlabDataset, collate_slabs, seed_worker

FULL = dict(num_classes=5, embed_dim=48, stem_ch=32, d_model=256, slab=5)
TINY = dict(num_classes=5, embed_dim=12, stem_ch=8, d_model=16, slab=5)


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


def lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return f


def load_fold(export_root, fold):
    folds = json.loads((Path(export_root) / "folds.json").read_text())["folds"]
    train = sorted(f for f, k in folds.items() if k != fold)
    held = sorted(f for f, k in folds.items() if k == fold)
    with open(Path(export_root) / "lesions.csv", newline="") as fh:
        lesions = [{**r, **{k: int(r[k]) for k in ("z0", "z1", "x0", "y0", "x1", "y1", "n_boxes")}}
                   for r in csv.DictReader(fh)]
    return train, held, lesions


def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main(argv=None):
    a = parse(argv)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    train_files, _, lesions = load_fold(a.export_root, a.fold)
    ds = SlabDataset(train_files, a.export_root, lesions, train=True, seed=a.seed)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                        collate_fn=collate_slabs, worker_init_fn=seed_worker,
                        persistent_workers=a.workers > 0)
    cfg = TINY if a.tiny else FULL
    model = Detector2D(**cfg, use_checkpoint=a.grad_ckpt).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.warmup, a.steps))
    step, ckpt = 0, out / "ckpt.pt"
    if a.resume and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "model": cfg, "n_train": len(train_files), "device": str(device),
              "parameters": sum(p.numel() for p in model.parameters())}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    amp = device.type == "cuda"
    log = open(out / "metrics.jsonl", "a")
    t0 = time.time()
    model.train()
    while step < a.steps:
        for batch in loader:
            if step >= a.steps:
                break
            image = batch["image"].to(device, non_blocking=True)
            batch = {**batch, "boxes": [b.to(device) for b in batch["boxes"]],
                     "box_classes": [c.to(device) for c in batch["box_classes"]]}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(image)
            parts = centre_loss_2d(pred, batch)
            loss = parts["heat"] + parts["offset"] + parts["size"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            row = {"step": step, "loss": float(loss), **{k: float(v) for k, v in parts.items()},
                   "grad_norm": float(gnorm), "lr": sched.get_last_lr()[0],
                   "seconds": round(time.time() - t0, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step == 1 or step % a.log_every == 0:
                print(json.dumps(row), flush=True)
            if step % a.ckpt_every == 0:
                save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                             "step": step, "config": config}, ckpt)
    save_atomic({"model": model.state_dict(), "config": config, "step": step}, out / "last.pt")
    log.close()
    print(f"done: {out} at step {step}", flush=True)
```

```python
#!/usr/bin/env python
# scripts/train_detector.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.train.train_detector import main  # noqa: E402

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_train_detector.py -q -p no:cacheprovider
```
预期：2 passed

- [ ] **Step 5: 跑 fold 0 的试跑并核时间预算**

```bash
mkdir -p runs logs
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. setsid nohup nice -n 19 \
  ~/anaconda3/envs/nvgen/bin/python scripts/train_detector.py --fold 0 --out runs/detector_fold0 --resume \
  > logs/detector_fold0.log 2>&1 &
```

到第 500 步时换算：`seconds/step × 20000 / 3600` 小时。**超过 48 小时就停下来问用户**，不要自行减步数。

- [ ] **Step 6: 训完五折**

fold 0 通过预算核对后，在 GPU 0 上依次训 fold 1–4（每卡一个任务）。

- [ ] **Step 7: 提交**

```bash
git add anatobind/train/train_detector.py scripts/train_detector.py tests/test_train_detector.py
git commit -m "Training: the fastMRI knee detector, fixed steps and resumable"
```

---

### Task 9: 折外预测、3D 聚合与失败标签

**Files:**
- Create: `anatobind/eval/detect3d.py`, `scripts/cache_detections.py`
- Test: `tests/test_detect3d.py`

**Interfaces:**
- Consumes: Task 5 的 `decode_centres_2d`，Task 6 的 `Detector2D`，Task 7 的 `SlabDataset`，`anatobind/eval/matching.py` 的 `iou3d/match`
- Produces: `detect_volume(model, vol, device, M=8, score_min=0.05, slab=5) -> (list[dict], np.ndarray)`（逐层检出 `slice, family, score, y0, x0, y1, x1, embed`，以及全卷平均的 `global_feat`，扫描级可靠性头要用）、`aggregate_to_3d(dets, iou_min=0.3) -> list[dict]`、`failure_labels(gt, pred, iou=0.1) -> (per_lesion, scan_label)`。`per_lesion` 每项含 `score, correct(0/1), family, box, embed`；`scan_label` 为 0/1，**漏检计入 1**。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_detect3d.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.eval.detect3d import aggregate_to_3d, failure_labels


def _det(slice_, y0, x0, y1, x1, family="meniscus", score=0.9):
    return {"slice": slice_, "family": family, "score": score,
            "y0": y0, "x0": x0, "y1": y1, "x1": x1, "embed": np.zeros(4, np.float32)}


def _les(z0, z1, y0, x0, y1, x1, family="meniscus"):
    return {"family": family, "z0": z0, "z1": z1, "y0": y0, "x0": x0, "y1": y1, "x1": x1}


def test_adjacent_overlapping_detections_become_one_lesion():
    dets = [_det(3, 10, 10, 30, 30), _det(4, 11, 10, 31, 30), _det(5, 12, 11, 32, 31)]
    out = aggregate_to_3d(dets)
    assert len(out) == 1 and (out[0]["z0"], out[0]["z1"]) == (3, 5)
    assert out[0]["score"] == pytest.approx(0.9)               # the aggregate keeps the peak score


def test_separate_detections_stay_separate():
    dets = [_det(3, 10, 10, 30, 30), _det(3, 200, 200, 220, 220)]
    assert len(aggregate_to_3d(dets)) == 2


def test_a_perfect_prediction_labels_everything_correct_and_the_scan_clean():
    gt = [_les(3, 5, 10, 10, 30, 30)]
    pred = [{"family": "meniscus", "score": 0.9, "z0": 3, "z1": 5,
             "y0": 10, "x0": 10, "y1": 30, "x1": 30, "embed": np.zeros(4, np.float32)}]
    per, scan = failure_labels(gt, pred)
    assert [p["correct"] for p in per] == [1] and scan == 0


def test_a_wrong_family_is_an_error_even_when_the_box_matches():
    gt = [_les(3, 5, 10, 10, 30, 30)]
    pred = [{"family": "cartilage", "score": 0.9, "z0": 3, "z1": 5,
             "y0": 10, "x0": 10, "y1": 30, "x1": 30, "embed": np.zeros(4, np.float32)}]
    per, scan = failure_labels(gt, pred)
    assert [p["correct"] for p in per] == [0] and scan == 1


def test_a_miss_makes_the_scan_label_one_with_no_per_lesion_row():
    gt = [_les(3, 5, 10, 10, 30, 30)]
    per, scan = failure_labels(gt, [])
    assert per == [] and scan == 1
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_detect3d.py -q -p no:cacheprovider
```
预期：`ModuleNotFoundError: No module named 'anatobind.eval.detect3d'`

- [ ] **Step 3: 写实现**

```python
# anatobind/eval/detect3d.py
"""逐层检出 -> 3D 病灶 -> 与真值匹配 -> 任务失败标签（leg 2 spec 3.1）。

3D 聚合用与真值同一条规则（相邻层、面内 IoU >= 0.3 相连），因此预测与真值在同一口径上比较。
扫描级标签把漏检计入：逐病灶可靠性头看不见漏检，扫描级头必须看得见。
"""
import numpy as np
import torch

from anatobind.data_engine.fastmri_knee import FAMILIES
from anatobind.eval.matching import iou3d, match
from anatobind.model.dense_head_2d import decode_centres_2d


def detect_volume(model, vol, device, M=8, score_min=0.05, slab=5):
    """vol: (S, H, W) float32。返回逐层检出。"""
    half = slab // 2
    out, globals_ = [], []
    model.eval()
    with torch.no_grad():
        for z in range(vol.shape[0]):
            idx = np.clip(np.arange(z - half, z + half + 1), 0, vol.shape[0] - 1)
            x = torch.from_numpy(np.ascontiguousarray(vol[idx]))[None, None].to(device)
            pred = model(x)
            globals_.append(pred["global_feat"][0].float().cpu().numpy())
            boxes, cls_prob, embed = decode_centres_2d(pred, M)
            for m in range(M):
                c = int(cls_prob[m, :len(FAMILIES)].argmax())
                s = float(cls_prob[m, c])
                if s < score_min:
                    continue
                y0, x0, y1, x1 = (float(v) for v in boxes[m])
                out.append({"slice": z, "family": FAMILIES[c], "score": s,
                            "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                            "embed": embed[m].float().cpu().numpy()})
    return out, np.mean(globals_, axis=0).astype(np.float32)


def _iou2d(a, b):
    iy = max(0.0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    ix = max(0.0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    inter = iy * ix
    union = (a["y1"] - a["y0"]) * (a["x1"] - a["x0"]) + (b["y1"] - b["y0"]) * (b["x1"] - b["x0"]) - inter
    return inter / union if union > 0 else 0.0


def aggregate_to_3d(dets, iou_min=0.3):
    groups = {}
    for d in dets:
        groups.setdefault(d["family"], []).append(d)
    out = []
    for family, group in sorted(groups.items()):
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
                if _iou2d(a, b) >= iou_min:
                    parent[find(i)] = find(j)
        comps = {}
        for i in range(len(group)):
            comps.setdefault(find(i), []).append(group[i])
        for members in comps.values():
            peak = max(members, key=lambda m: m["score"])
            out.append({"family": family, "score": peak["score"],
                        "z0": min(m["slice"] for m in members), "z1": max(m["slice"] for m in members),
                        "y0": min(m["y0"] for m in members), "x0": min(m["x0"] for m in members),
                        "y1": max(m["y1"] for m in members), "x1": max(m["x1"] for m in members),
                        "embed": peak["embed"], "n_slices": len(members)})
    return out


def _corners(d):
    return [d["z0"], d["y0"], d["x0"], d["z1"] + 1, d["y1"], d["x1"]]


def failure_labels(gt, pred, iou=0.1):
    """per_lesion: 每个检出一行，correct = 匹配上且类别族正确。scan_label: 有任何错误（含漏检）则 1。"""
    g = np.array([_corners(L) for L in gt], dtype=float).reshape(-1, 6)
    p = np.array([_corners(d) for d in pred], dtype=float).reshape(-1, 6)
    pairs = match(g, p, iou) if len(g) and len(p) else {}
    gt_of_pred = {q: j for j, q in pairs.items()}
    per = []
    for q, d in enumerate(pred):
        j = gt_of_pred.get(q)
        correct = int(j is not None and gt[j]["family"] == d["family"])
        per.append({"score": d["score"], "correct": correct, "family": d["family"],
                    "box": _corners(d), "embed": d["embed"]})
    n_matched_right = sum(r["correct"] for r in per)
    scan = int(n_matched_right < len(gt) or any(r["correct"] == 0 for r in per))
    return per, scan
```

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_detect3d.py -q -p no:cacheprovider
```
预期：5 passed

- [ ] **Step 5: 写缓存脚本并跑五折**

```python
#!/usr/bin/env python
# scripts/cache_detections.py
"""把某折检测器在其留出患者上的折外检出缓存成 npz。

  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python \
      scripts/cache_detections.py --fold 0
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import EXPORT_ROOT, VIEWS  # noqa: E402
from anatobind.eval.detect3d import aggregate_to_3d, detect_volume  # noqa: E402
from anatobind.model.detector2d import Detector2D  # noqa: E402
from anatobind.train.train_detector import load_fold  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--out", type=Path, default=EXPORT_ROOT / "detections")
    a = ap.parse_args()
    run = a.run or Path(f"runs/detector_fold{a.fold}")
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Detector2D(**state["config"]["model"]).to(device)
    model.load_state_dict(state["model"])
    _, held, _ = load_fold(a.export_root, a.fold)
    out_dir = a.out / f"fold{a.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in held:
        for view in VIEWS:
            vol = np.load(a.export_root / f / f"{view}.npy").astype(np.float32)
            dets, gfeat = detect_volume(model, vol, device)
            with open(out_dir / f"{f}__{view}.pkl", "wb") as fh:
                pickle.dump({"dets": aggregate_to_3d(dets), "global_feat": gfeat}, fh)
            n += 1
    print(f"fold {a.fold}: {n} (volume, view) detection files in {out_dir}")


if __name__ == "__main__":
    main()
```

五折都跑完：

```bash
for f in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 \
    ~/anaconda3/envs/nvgen/bin/python scripts/cache_detections.py --fold $f
done
```

- [ ] **Step 6: 提交**

```bash
git add anatobind/eval/detect3d.py scripts/cache_detections.py tests/test_detect3d.py
git commit -m "Evaluation: out-of-fold detections, 3D aggregation and task-failure labels"
```

---

### Task 10: H1 止损门

**Files:**
- Create: `scripts/check_h1.py`, `docs/verification/<日期>/h1.md`

**Interfaces:**
- Consumes: Task 9 的检出缓存与 `failure_labels`
- Produces: 一个 PASS/FAIL 判定与一个写死的检出阈值 `SCORE_MIN`

- [ ] **Step 1: 写 H1 检查脚本**

```python
#!/usr/bin/env python
# scripts/check_h1.py
"""H1 止损门（leg 2 spec 3.4）：折外检出是否多到、且对错是否均衡到，值得去训可靠性头。

阈值只在训练折上选：先在 folds 1-4 的折外检出上找到使扫描级正样本率落入 [0.2, 0.5] 的最小分数阈值，
再把该阈值固定，用它在全部五折上计算 H1 的两个条件。
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import EXPORT_ROOT, VIEWS  # noqa: E402
from anatobind.eval.detect3d import failure_labels  # noqa: E402
from anatobind.train.train_detector import load_fold  # noqa: E402


def gather(export_root, det_root, folds, score_min):
    per_all, scan_all = [], []
    with open(Path(export_root) / "lesions.csv", newline="") as fh:
        by_file = {}
        for r in csv.DictReader(fh):
            by_file.setdefault(r["file"], []).append(
                {**r, **{k: int(r[k]) for k in ("z0", "z1", "x0", "y0", "x1", "y1")}})
    for fold in folds:
        _, held, _ = load_fold(export_root, fold)
        for f in held:
            for view in VIEWS:
                p = Path(det_root) / f"fold{fold}" / f"{f}__{view}.pkl"
                if not p.exists():
                    continue
                dets = [d for d in pickle.load(open(p, "rb"))["dets"] if d["score"] >= score_min]
                per, scan = failure_labels(by_file.get(f, []), dets)
                for r in per:
                    per_all.append({**r, "file": f, "view": view, "fold": fold})
                scan_all.append({"file": f, "view": view, "fold": fold, "label": scan})
    return per_all, scan_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--det-root", type=Path, default=EXPORT_ROOT / "detections")
    a = ap.parse_args()

    chosen = None
    for thr in np.arange(0.05, 0.96, 0.05):
        _, scan = gather(a.export_root, a.det_root, [1, 2, 3, 4], thr)
        rate = np.mean([s["label"] for s in scan]) if scan else 1.0
        print(f"threshold {thr:.2f}: scan-level positive rate on training folds {rate:.3f}")
        if 0.2 <= rate <= 0.5:
            chosen = float(thr)
            break
    if chosen is None:
        print("H1 FAIL: no threshold puts the scan-level positive rate in [0.2, 0.5]")
        sys.exit(1)

    per, scan = gather(a.export_root, a.det_root, [0, 1, 2, 3, 4], chosen)
    n_det = len(per)
    pos = float(np.mean([r["correct"] for r in per])) if per else 0.0
    ok = n_det >= 300 and 0.2 <= pos <= 0.8
    print(f"\nSCORE_MIN = {chosen:.2f}")
    print(f"detections (all folds, all views): {n_det}; per-lesion correct rate: {pos:.3f}")
    print(f"scan-level positive rate: {np.mean([s['label'] for s in scan]):.3f}")
    print(f"H1 {'PASS' if ok else 'FAIL'} (needs >= 300 detections and correct rate in [0.2, 0.8])")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 H1**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/check_h1.py \
  | tee docs/verification/$(date +%Y-%m-%d)/h1.txt
```

- [ ] **Step 3: 把结果写成验证笔记并提交**

把命令、原始输出、选定的 `SCORE_MIN`、以及 PASS/FAIL 写进 `docs/verification/<日期>/h1.md`。

```bash
git add docs/verification/ scripts/check_h1.py
git commit -m "Verification: the H1 stop gate on the fastMRI knee detector"
```

- [ ] **Step 4: 若 H1 不过，停下来交给用户**

**不要自行放宽阈值、不要继续建闸门。** 把 H1 的输出、检出数、正样本率、以及可能的原因（检出太少 / 全对 / 全错）报告给用户，等用户决定。

---

### Task 11: 两个可靠性头

**Files:**
- Create: `anatobind/model/reliability.py`, `scripts/train_reliability.py`
- Test: `tests/test_reliability.py`

**Interfaces:**
- Consumes: Task 9 的 `per_lesion`/`scan_label`，Task 6 的 `global_feat`
- Produces: `LesionReliability(embed_dim=256, n_scalar=3, hidden=128)`、`ScanReliability(global_dim, n_scalar=5, hidden=128)`，两者 `forward(feat, scalars) -> (N,)` logits；`lesion_scalars(row)`、`scan_scalars(rows)`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_reliability.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.model.reliability import (
    LesionReliability, ScanReliability, lesion_scalars, scan_scalars,
)


def _row(score, correct=1, box=(0, 0, 0, 4, 20, 20)):
    return {"score": score, "correct": correct, "box": list(box), "embed": np.zeros(8, np.float32)}


def test_lesion_scalars_are_score_margin_and_size():
    s = lesion_scalars(_row(0.8))
    assert s.shape == (3,) and s[0] == pytest.approx(0.8)
    assert s[2] > 0                                  # box volume in log units


def test_scan_scalars_survive_a_scan_with_no_detections():
    s = scan_scalars([])
    assert s.shape == (5,) and np.isfinite(s).all()


def test_both_heads_return_one_logit_per_item():
    lr = LesionReliability(embed_dim=8, n_scalar=3, hidden=16)
    sr = ScanReliability(global_dim=12, n_scalar=5, hidden=16)
    assert lr(torch.randn(4, 8), torch.randn(4, 3)).shape == (4,)
    assert sr(torch.randn(2, 12), torch.randn(2, 5)).shape == (2,)


def test_a_lesion_head_can_learn_a_separable_toy_problem():
    torch.manual_seed(0)
    n = 256
    feat = torch.randn(n, 8)
    scal = torch.randn(n, 3)
    y = (feat[:, 0] + scal[:, 0] > 0).float()
    head = LesionReliability(embed_dim=8, n_scalar=3, hidden=32)
    opt = torch.optim.Adam(head.parameters(), 1e-2)
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(feat, scal), y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(((head(feat, scal) > 0).float() == y).float().mean()) > 0.9
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_reliability.py -q -p no:cacheprovider
```
预期：`ModuleNotFoundError: No module named 'anatobind.model.reliability'`

- [ ] **Step 3: 写实现**

```python
# anatobind/model/reliability.py
"""两个无参照的可靠性头（leg 2 spec 2.3）。

逐病灶头只能评价检测器找到的东西，漏检对它不可见；扫描级头专门覆盖漏检，因此必须是两个。
两者都只看单张图的特征与该图上检出的统计，不看任何干净图参照。
"""
import numpy as np
import torch
import torch.nn as nn


def _mlp(d_in, hidden):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, 1))


class LesionReliability(nn.Module):
    def __init__(self, embed_dim=256, n_scalar=3, hidden=128):
        super().__init__()
        self.net = _mlp(embed_dim + n_scalar, hidden)

    def forward(self, feat, scalars):
        return self.net(torch.cat([feat, scalars], -1)).squeeze(-1)


class ScanReliability(nn.Module):
    def __init__(self, global_dim, n_scalar=5, hidden=128):
        super().__init__()
        self.net = _mlp(global_dim + n_scalar, hidden)

    def forward(self, global_feat, scalars):
        return self.net(torch.cat([global_feat, scalars], -1)).squeeze(-1)


def lesion_scalars(row):
    """峰值得分、top-1 与 no-object 之差、框体积的对数。"""
    z0, y0, x0, z1, y1, x1 = row["box"]
    vol = max((z1 - z0) * (y1 - y0) * (x1 - x0), 1.0)
    return np.array([row["score"], 2 * row["score"] - 1.0, np.log(vol)], dtype=np.float32)


def scan_scalars(rows):
    """检出数，以及峰值得分的最大 / 中位 / 最小与均值；没有检出时用 0 填。"""
    if not rows:
        return np.zeros(5, dtype=np.float32)
    s = np.array([r["score"] for r in rows], dtype=np.float32)
    return np.array([len(rows), s.max(), float(np.median(s)), s.min(), s.mean()], dtype=np.float32)
```

```python
#!/usr/bin/env python
# scripts/train_reliability.py
"""按 spec 2.4 的折套法训练两个可靠性头。

第 k 个头在 folds != k 的折外检出上训练，在 fold k 的折外检出上评估；报告的分数全部来自各自 fold k，
不做二次选择。检出阈值用 Task 10 定下的 SCORE_MIN，不再调。

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/train_reliability.py --score-min 0.30
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import EXPORT_ROOT, VIEWS  # noqa: E402
from anatobind.eval.detect3d import failure_labels  # noqa: E402
from anatobind.model.reliability import (  # noqa: E402
    LesionReliability, ScanReliability, lesion_scalars, scan_scalars,
)
from anatobind.train.train_detector import load_fold  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)


def lesions_by_file(export_root):
    out = {}
    with open(Path(export_root) / "lesions.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["file"], []).append(
                {**r, **{k: int(r[k]) for k in ("z0", "z1", "x0", "y0", "x1", "y1")}})
    return out


def gather(export_root, det_root, fold, score_min, by_file):
    """fold 的折外检出 -> (逐病灶样本, 扫描级样本)。"""
    _, held, _ = load_fold(export_root, fold)
    lesion_rows, scan_rows = [], []
    for f in held:
        for view in VIEWS:
            path = Path(det_root) / f"fold{fold}" / f"{f}__{view}.pkl"
            if not path.exists():
                continue
            blob = pickle.load(open(path, "rb"))
            dets = [d for d in blob["dets"] if d["score"] >= score_min]
            per, scan = failure_labels(by_file.get(f, []), dets)
            for r in per:
                lesion_rows.append({"file": f, "view": view, "fold": fold, "patient": f,
                                    "embed": r["embed"], "scalars": lesion_scalars(r),
                                    "label": float(r["correct"]), "peak_score": r["score"]})
            scan_rows.append({"file": f, "view": view, "fold": fold, "patient": f,
                              "global_feat": blob["global_feat"], "scalars": scan_scalars(per),
                              "label": float(1 - scan),
                              "peak_score": max([r["score"] for r in per], default=0.0),
                              "min_lesion": min([r["score"] for r in per], default=0.0)})
    return lesion_rows, scan_rows


def fit(head, feats, scalars, labels, steps, batch, lr, seed):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(labels)
    head.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        loss = F.binary_cross_entropy_with_logits(head(feats[idx], scalars[idx]), labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    return head


def stack(rows, key):
    return torch.from_numpy(np.stack([r[key] for r in rows]).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-min", type=float, required=True, help="Task 10 定下的 SCORE_MIN")
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--det-root", type=Path, default=EXPORT_ROOT / "detections")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    by_file = lesions_by_file(a.export_root)
    per_fold = {k: gather(a.export_root, a.det_root, k, a.score_min, by_file) for k in FOLDS}

    out_rows = []
    for k in FOLDS:
        tr_les = [r for j in FOLDS if j != k for r in per_fold[j][0]]
        tr_scan = [r for j in FOLDS if j != k for r in per_fold[j][1]]
        te_les, te_scan = per_fold[k]
        run = Path(f"runs/reliability_fold{k}")
        run.mkdir(parents=True, exist_ok=True)

        if tr_les and te_les:
            head = LesionReliability(embed_dim=len(tr_les[0]["embed"]), n_scalar=3)
            fit(head, stack(tr_les, "embed"), stack(tr_les, "scalars"),
                torch.tensor([r["label"] for r in tr_les]), a.steps, 256, a.lr, a.seed)
            with torch.no_grad():
                score = torch.sigmoid(head(stack(te_les, "embed"), stack(te_les, "scalars"))).numpy()
            torch.save(head.state_dict(), run / "lesion.pt")
            for r, sc in zip(te_les, score):
                out_rows.append({"kind": "lesion", "file": r["file"], "view": r["view"], "fold": k,
                                 "patient": r["patient"], "head_score": float(sc),
                                 "peak_score": r["peak_score"], "correct": r["label"], "min_lesion": ""})

        head = ScanReliability(global_dim=len(tr_scan[0]["global_feat"]), n_scalar=5)
        fit(head, stack(tr_scan, "global_feat"), stack(tr_scan, "scalars"),
            torch.tensor([r["label"] for r in tr_scan]), a.steps, 64, a.lr, a.seed)
        with torch.no_grad():
            score = torch.sigmoid(head(stack(te_scan, "global_feat"), stack(te_scan, "scalars"))).numpy()
        torch.save(head.state_dict(), run / "scan.pt")
        for r, sc in zip(te_scan, score):
            out_rows.append({"kind": "scan", "file": r["file"], "view": r["view"], "fold": k,
                             "patient": r["patient"], "head_score": float(sc),
                             "peak_score": r["peak_score"], "correct": r["label"],
                             "min_lesion": r["min_lesion"]})

    fields = ["kind", "file", "view", "fold", "patient", "head_score", "peak_score", "correct", "min_lesion"]
    out = Path("runs/reliability_scores.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    n_les = sum(1 for r in out_rows if r["kind"] == "lesion")
    print(f"{out}: {n_les} lesion rows, {len(out_rows) - n_les} scan rows")


if __name__ == "__main__":
    main()
```

`patient` 一列此处用卷名填充；若要严格按患者做 bootstrap，在 Task 12 读入时用 `meta.json` 的 `patient_id` 替换（同一患者的两卷必须归为一组）。

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_reliability.py -q -p no:cacheprovider
```
预期：4 passed

- [ ] **Step 5: 提交**

```bash
git add anatobind/model/reliability.py scripts/train_reliability.py tests/test_reliability.py
git commit -m "Model: per-lesion and per-scan reliability heads, trained out of fold"
```

---

### Task 12: 闸门评估与 H2、H3

**Files:**
- Create: `anatobind/eval/gate.py`, `scripts/eval_gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: Task 11 的 `scores.csv`
- Produces: `risk_coverage(scores, labels) -> (coverage, risk)`、`aurc(scores, labels) -> float`、`threshold_at_risk(scores, labels, r_max=0.05) -> float`、`bootstrap_delta(per_patient, fn, reps=10000, seed=0) -> (delta, lo, hi)`、`h2(rows) -> dict`、`h3(rows) -> dict`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_gate.py
import numpy as np
import pytest

from anatobind.eval.gate import aurc, risk_coverage, threshold_at_risk


def test_risk_coverage_is_monotone_in_coverage_for_a_perfect_ranker():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    labels = np.array([1, 1, 1, 0, 0])          # 1 = correct
    cov, risk = risk_coverage(scores, labels)
    assert cov[0] < cov[-1] and risk[0] == 0.0
    assert risk[-1] == pytest.approx(0.4)


def test_a_perfect_ranker_beats_a_random_one_on_aurc():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    good = labels + rng.normal(0, 0.01, 200)
    bad = rng.normal(0, 1, 200)
    assert aurc(good, labels) < aurc(bad, labels)


def test_the_threshold_meets_the_requested_risk():
    scores = np.concatenate([np.linspace(0.6, 1.0, 90), np.linspace(0.0, 0.5, 10)])
    labels = np.concatenate([np.ones(90), np.zeros(10)])
    thr = threshold_at_risk(scores, labels, r_max=0.05)
    kept = scores >= thr
    assert kept.sum() > 0 and (1 - labels[kept].mean()) <= 0.05


def test_an_impossible_risk_target_returns_infinity():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 0, 1, 1])
    assert threshold_at_risk(scores, labels, r_max=0.01) == float("inf")
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_gate.py -q -p no:cacheprovider
```
预期：`ModuleNotFoundError: No module named 'anatobind.eval.gate'`

- [ ] **Step 3: 写实现**

```python
# anatobind/eval/gate.py
"""风险–覆盖、AURC、阈值选择与 H2/H3（leg 2 spec 3.3、3.4）。

labels 用 1 = 正确、0 = 错误；risk 是被接受样本中错误的比例。
"""
import numpy as np

RISK_TARGET = 0.05
REPS = 10_000


def risk_coverage(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    order = np.argsort(-scores)
    lab = labels[order]
    k = np.arange(1, len(lab) + 1)
    coverage = k / len(lab)
    risk = 1.0 - np.cumsum(lab) / k
    return coverage, risk


def aurc(scores, labels):
    coverage, risk = risk_coverage(scores, labels)
    return float(np.trapezoid(risk, coverage) / (coverage[-1] - coverage[0])) if len(coverage) > 1 else float("nan")


def threshold_at_risk(scores, labels, r_max=RISK_TARGET):
    """满足风险上限的最小分数阈值；无解返回 inf。"""
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    order = np.argsort(-scores)
    lab, sc = labels[order], scores[order]
    k = np.arange(1, len(lab) + 1)
    risk = 1.0 - np.cumsum(lab) / k
    ok = np.nonzero(risk <= r_max)[0]
    return float(sc[ok[-1]]) if len(ok) else float("inf")


def bootstrap_delta(per_patient, fn, reps=REPS, seed=0, alpha=0.05):
    """per_patient: list[list[row]]，按患者重采样；fn(rows) -> 标量。返回 (delta, lo, hi)。"""
    rng = np.random.default_rng(seed)
    base = fn([r for rows in per_patient for r in rows])
    out = np.empty(reps)
    n = len(per_patient)
    for b in range(reps):
        pick = rng.integers(0, n, n)
        out[b] = fn([r for i in pick for r in per_patient[i]])
    return float(base), float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))
```

追加到同一文件：

```python
VIEW_ORDER = ("clean", "noise_q1", "noise_q2", "noise_q3", "us4", "us8", "us16")
GATE_BINS = VIEW_ORDER[1:]          # H2 只在退化档位内部判定；clean 一并报告但不参与判定


def _by_patient(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["patient"], []).append(r)
    return list(groups.values())


def _delta_aurc(rows):
    if not rows or len(set(r["correct"] for r in rows)) < 2:
        return float("nan")
    y = [r["correct"] for r in rows]
    return aurc([r["head_score"] for r in rows], y) - aurc([r["peak_score"] for r in rows], y)


def h2(rows, bins=GATE_BINS, reps=REPS, seed=0):
    """每个退化档位内部：可靠性头的 AURC 减去峰值得分基线的 AURC。越负越好。"""
    out = {}
    for v in VIEW_ORDER:
        sel = [r for r in rows if r["view"] == v]
        d, lo, hi = bootstrap_delta(_by_patient(sel), _delta_aurc, reps, seed) if sel else (float("nan"),) * 3
        out[v] = {"n": len(sel), "delta_aurc": d, "ci95": [lo, hi],
                  "pass": bool(np.isfinite(d) and d < 0 and hi < 0)}
    out["pass"] = all(out[v]["pass"] for v in bins)
    return out


def _head_coverage(rows, key, r_max=RISK_TARGET):
    if not rows:
        return 0.0
    s = np.array([r[key] for r in rows], float)
    y = np.array([r["correct"] for r in rows], float)
    tau = threshold_at_risk(s, y, r_max)
    return float((s >= tau).mean()) if np.isfinite(tau) else 0.0


def _rule_coverage(rows, r_max=RISK_TARGET):
    """按加速倍数一刀切：取风险仍 <= r_max 的最宽松截断，返回它保留的覆盖率。"""
    if not rows:
        return 0.0
    for k in range(len(VIEW_ORDER), 0, -1):
        keep = set(VIEW_ORDER[:k])
        sel = [r for r in rows if r["view"] in keep]
        if sel and 1.0 - float(np.mean([r["correct"] for r in sel])) <= r_max:
            return len(sel) / len(rows)
    return 0.0


def h3(rows, reps=REPS, seed=0, r_max=RISK_TARGET):
    """固定风险下，学习闸门的覆盖率是否高于一刀切规则；同时报告 spec 3.2 的最小值基线。"""
    groups = _by_patient(rows)
    res = {"risk_target": r_max,
           "coverage_head": _head_coverage(rows, "head_score", r_max),
           "coverage_rule": _rule_coverage(rows, r_max),
           "coverage_min_baseline": _head_coverage(rows, "min_lesion", r_max)}
    for name, fn in (("vs_rule", lambda rs: _head_coverage(rs, "head_score", r_max) - _rule_coverage(rs, r_max)),
                     ("vs_min", lambda rs: _head_coverage(rs, "head_score", r_max)
                      - _head_coverage(rs, "min_lesion", r_max))):
        d, lo, hi = bootstrap_delta(groups, fn, reps, seed)
        res[name] = {"delta_coverage": d, "ci95": [lo, hi], "pass": bool(d > 0 and lo > 0)}
    res["pass"] = res["vs_rule"]["pass"]
    return res
```

`h3` 的 `pass` 只看与一刀切规则的比较（这是 spec 3.4 的 H3）；与最小值基线的比较（spec 3.2）一并报告，用来说明扫描级头相对"把逐病灶可靠性取最小值"是否值得，但不参与判定。

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_gate.py -q -p no:cacheprovider
```
预期：4 passed

- [ ] **Step 5: 写 `scripts/eval_gate.py` 并加一个端到端测试**

```python
#!/usr/bin/env python
# scripts/eval_gate.py
"""leg 2 的闸门报告与 H2、H3。

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_gate.py \
      --out docs/verification/2026-09-14/gate --figs ~/figs/anatobind_leg2
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import EXPORT_ROOT  # noqa: E402
from anatobind.eval.gate import VIEW_ORDER, h2, h3, risk_coverage  # noqa: E402

VIEW_LABELS = {"clean": "clean", "noise_q1": "noise\nq1", "noise_q2": "noise\nq2", "noise_q3": "noise\nq3",
               "us4": "us\n4x", "us8": "us\n8x", "us16": "us\n16x"}
COLORS = {"head": "#2a78d6", "peak": "#eb6834", "rule": "#1baf7a", "min": "#eda100"}


def read_scores(path, export_root):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            meta = json.loads((Path(export_root) / r["file"] / "meta.json").read_text())
            rows.append({**r, "fold": int(r["fold"]), "correct": float(r["correct"]),
                         "head_score": float(r["head_score"]), "peak_score": float(r["peak_score"]),
                         "min_lesion": float(r["min_lesion"]) if r["min_lesion"] else 0.0,
                         "patient": meta["patient_id"]})
    return rows


def fig_risk_coverage(scan_rows, path):
    fig, axes = plt.subplots(1, len(VIEW_ORDER), figsize=(2.0 * len(VIEW_ORDER), 2.6), sharey=True)
    for ax, v in zip(axes, VIEW_ORDER):
        sel = [r for r in scan_rows if r["view"] == v]
        for key, label in (("head_score", "scan head"), ("peak_score", "peak score")):
            if not sel:
                continue
            cov, risk = risk_coverage([r[key] for r in sel], [r["correct"] for r in sel])
            ax.plot(cov, risk, lw=2, color=COLORS["head" if key == "head_score" else "peak"], label=label)
        ax.axhline(0.05, ls="--", lw=1, color="0.4")
        ax.set_title(VIEW_LABELS[v], fontsize=9)
        ax.set_xlabel("coverage")
    axes[0].set_ylabel("risk among accepted scans")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_coverage_bars(res, path):
    names = [("coverage_head", "learned scan gate", "head"), ("coverage_rule", "acceleration rule", "rule"),
             ("coverage_min_baseline", "min of lesion scores", "min")]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar([n[1] for n in names], [res[n[0]] for n in names], color=[COLORS[n[2]] for n in names], width=0.6)
    for i, n in enumerate(names):
        ax.text(i, res[n[0]] + 0.01, f"{res[n[0]]:.2f}", ha="center", fontsize=9)
    ax.set_ylabel(f"coverage at risk <= {res['risk_target']:.2f}")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("runs/reliability_scores.csv"))
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figs", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    a.figs.mkdir(parents=True, exist_ok=True)

    rows = read_scores(a.scores, a.export_root)
    lesion_rows = [r for r in rows if r["kind"] == "lesion"]
    scan_rows = [r for r in rows if r["kind"] == "scan"]
    summary = {"n_lesion_rows": len(lesion_rows), "n_scan_rows": len(scan_rows),
               "n_patients": len({r["patient"] for r in rows}),
               "h2": h2(lesion_rows, reps=a.reps), "h3": h3(scan_rows, reps=a.reps)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(a.out / "records.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("H2 (per-lesion reliability against the peak score, within each view):")
    for v in VIEW_ORDER:
        d = summary["h2"][v]
        print(f"  {v:9} n={d['n']:5d}  delta AURC {d['delta_aurc']:+.4f}  "
              f"95% CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  {'pass' if d['pass'] else ''}")
    print(f"H2 VERDICT: {'PASS' if summary['h2']['pass'] else 'FAIL'}")
    r = summary["h3"]
    print(f"\nH3 at risk <= {r['risk_target']}: learned gate {r['coverage_head']:.3f}, "
          f"acceleration rule {r['coverage_rule']:.3f}, min baseline {r['coverage_min_baseline']:.3f}")
    print(f"  vs rule: {r['vs_rule']['delta_coverage']:+.3f} CI {r['vs_rule']['ci95']}")
    print(f"  vs min : {r['vs_min']['delta_coverage']:+.3f} CI {r['vs_min']['ci95']}")
    print(f"H3 VERDICT: {'PASS' if r['pass'] else 'FAIL'}")

    fig_risk_coverage(scan_rows, a.figs / "risk_coverage.png")
    fig_coverage_bars(r, a.figs / "coverage_at_risk.png")
    print(f"\nfigures: {a.figs}/risk_coverage.png, {a.figs}/coverage_at_risk.png")


if __name__ == "__main__":
    main()
```

端到端测试 `tests/test_eval_gate_script.py`：

```python
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_gate.py"
    spec = importlib.util.spec_from_file_location("eval_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path, informative):
    """informative=True: head_score 与正确性强相关，peak_score 与之无关。"""
    from anatobind.data_engine.fastmri_knee import VIEWS
    rng = np.random.default_rng(0)
    root = tmp_path / "leg2"
    rows = []
    for i in range(60):
        f = f"file{i}"
        (root / f).mkdir(parents=True)
        (root / f / "meta.json").write_text(json.dumps({"patient_id": f"p{i // 2}"}))
        for v in VIEWS:
            correct = float(rng.integers(0, 2))
            peak = float(rng.uniform(0, 1))
            head = correct * 0.5 + rng.uniform(0, 0.5) if informative else peak
            for kind in ("lesion", "scan"):
                rows.append({"kind": kind, "file": f, "view": v, "fold": i % 5, "patient": "",
                             "head_score": head, "peak_score": peak, "correct": correct,
                             "min_lesion": peak})
    path = tmp_path / "scores.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path, root


def test_h2_passes_when_the_head_knows_something_the_peak_score_does_not(tmp_path):
    scores, root = _write(tmp_path, informative=True)
    out = tmp_path / "report"
    _load().main(["--scores", str(scores), "--export-root", str(root), "--out", str(out),
                  "--figs", str(tmp_path / "figs"), "--reps", "200"])
    s = json.loads((out / "summary.json").read_text())
    assert s["h2"]["pass"] is True
    assert (tmp_path / "figs/risk_coverage.png").exists()
    assert (tmp_path / "figs/coverage_at_risk.png").exists()


def test_h2_fails_when_the_head_only_repeats_the_peak_score(tmp_path):
    scores, root = _write(tmp_path, informative=False)
    out = tmp_path / "report"
    _load().main(["--scores", str(scores), "--export-root", str(root), "--out", str(out),
                  "--figs", str(tmp_path / "figs"), "--reps", "200"])
    assert json.loads((out / "summary.json").read_text())["h2"]["pass"] is False
```

- [ ] **Step 6: 跑评估，看图，再引用**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_gate.py \
  --out docs/verification/$(date +%Y-%m-%d)/gate --figs ~/figs/anatobind_leg2
```

**渲染后先把两张 PNG 读出来看一眼**，修掉标签碰撞或被裁掉的文字再引用。

- [ ] **Step 7: 提交**

```bash
git add anatobind/eval/gate.py scripts/eval_gate.py tests/test_gate.py docs/verification/
git commit -m "Evaluation: risk-coverage, AURC and the H2 and H3 criteria"
```

---

### Task 13: 报告与交接

**Files:**
- Create: `docs/verification/<日期>/REPORT.md`
- Modify: `STATUS.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: Task 1–12 的全部产物

- [ ] **Step 1: 写报告**

每节按"结论 → 可粘贴的命令 → 原始输出 → 支撑了哪个决策"写，覆盖：数据引擎的计数、检测器的时间与检测指标、H1 的判定与 `SCORE_MIN`、H2 的逐档位数字、H3 的覆盖率对比、四条已知弱点（spec §7）。

- [ ] **Step 2: 按五段体例重写 `STATUS.md`**

已完成且已验证 / 待用户拍板 / 下一步 / 坑与别重做 / 关键决定的为什么。

- [ ] **Step 3: 更新 `CLAUDE.md`**

补上 leg 2 的导出根、新模块清单、新的测试数目，以及一行当前状态指向。

- [ ] **Step 4: 跑全套测试**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
```

- [ ] **Step 5: 提交、推送、打 tag**

```bash
git add -A
git commit -m "Report and handoff for leg 2"
git push origin main
git tag -f handoff/$(date +%Y-%m-%d) -m "Leg 2 complete"
git push -f origin handoff/$(date +%Y-%m-%d)
```
