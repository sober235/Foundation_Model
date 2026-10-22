# AUR 判门层与 fastMRI 批处理 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 2026-10-22 验收规格 §6.1 到 §6.4 做出判门层（A/U/R，2D 与 3D）与 fastMRI 4469 序列的像素网格批处理，用合成数据与小模型验收；不训练任何模型。

**Architecture:** 判门层不改现有评价器的开发口径，而是在 `anatobind/aur/unified_evaluation.py` 上加一种"门观察"（3D 整卷 + 2D 固定步长层 + 病灶质心层）和三项逐观察附加量（匹配实例 Dice、宿主占比、层内关系真值），再由一个纯函数模块 `anatobind/aur/gate.py` 按登记常数判定。历史 test 角色做成 `final_v3` 合同的只读后继目录（新模块 `anatobind/aur/test_role.py`），**不改 `data_contract.py`**，因为它的字节被 `final_v3` 的 live 校验钉死。S7 参照在模型的 1 mm 规范网格上重新打分。fastMRI 批走一条单独的像素网格 DICOM 入口，两种模式都先重采样到 1 mm 再映回原网格，侧别按屏幕左右。

**Tech Stack:** Python 3 / PyTorch 2.5.1 / nibabel / scipy / pydicom，环境 `~/anaconda3/envs/nvgen`；pytest。

**Spec:** `docs/superpowers/specs/2026-10-22-aur-three-subgoal-acceptance-design.md`（§3 用户决定 Q31–Q34，§6 门，§9 裁定清单）。本计划只覆盖规格 §10 的第 1 块；第 2 块（从 LR5 Stage I 起跑正式 II/III）另写计划。

## Global Constraints

- 数据只读 `/data2/congcong/data/FM_data`；所有医学数组、逐例记录只写 `/home/congcongliu/outputs/<run>/`，经 `anatobind.aur.data_contract.validate_private_output` 校验（家目录内、不在 Git 里）。Git 只收代码、测试、匿名计数与哈希。
- 任何工具输出、异常文本、日志、匿名摘要不得含病人标识、文件名或路径（`AGENTS.md`）；测试一律断言 `'synthetic_private' not in` 公开文本。
- 测试命令：`CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest <目标> -q -p no:cacheprovider`；全套为 `tests/ -q -rs`，当前基线 1820 passed / 5 skipped（LR5 分支计数；main 上以首次全套为准）。
- **不修改 `anatobind/aur/data_contract.py`**：`final_v3` 的 live 校验把它的 sha256 写进了合同，改一个字节训练加载就拒绝。需要的功能通过导入其函数在新模块里实现。
- 规格的门值全部来自 `anatobind/aur/gate.py::GATE_V1`，写死，看任何真实结果前不改；空分母只能得 `unavailable`，不能 `pass`。所有数字标 `NOT_EVIDENCE`。
- 2D 的 R 参照用 `host_targets` 在层内重跑（规格 §6.3），不用 `planar_relations._qualify_geometry`。
- fastMRI 像素网格入口只在显式调用时使用；现有严格入口 `anatobind/aur/dicom.py` 不动。输出永远带 `geometry='assumed'`，不产生患者坐标。
- 本计划不读历史 test 的任何数组：test 角色模块只用合成数据验收，真实 test 后继目录在第 2 块计划的最终判门步骤才建。
- 提交：作者用仓库本地配置（Congcong Liu），消息英文、句首大写、描述做了什么，不写任何 AI 痕迹；每个任务一个提交。GPU 不需要；CPU 任务 `nice -n 19`。

## Review Focus

- 空分母：某来源没有 S7 参照、某类在全部 test 卷里无真值、R 没有匹配上的实例 → 对应项必须是 `unavailable`，整体不能 `pass`（Task 5 `test_verdict_never_passes_on_empty_denominators`）。
- 监督缺失的行：`a_supervised=False` 或 `u_supervised=False` 的 test 行从对应门的分母里剔除，不算 fail（Task 5 `test_unsupervised_rows_leave_denominators`）。
- 一个坏文件拖垮整序列：fastMRI 某序列里一张 DICOM 形状不一致或读不出 → 整个序列跳过并按原因计数，不能部分叠栈（Task 7 `test_batch_skips_bad_series_and_counts_by_reason`）。
- 病灶质心层越界或单体素病灶：质心取整后必须落在 `[0, depth-1]`，单体素病灶也要有目标层（Task 1 `test_lesion_centroid_planes_rounds_and_clips`）。
- fastMRI+ 标注对不上序列：CSV 的 `file` 没有对应序列目录，或 `slice` 超过序列层数 → 计入 unmatched，不抛异常（Task 8 `test_unmatched_annotations_are_counted_not_raised`）。

---

### Task 1: 判门度量的纯函数

**Files:**
- Create: `anatobind/aur/gate_measures.py`
- Test: `tests/test_aur_gate_measures.py`

**Interfaces:**
- Consumes: `anatobind.aur.targets.host_targets(inst, seg, spacing)`、`anatobind.aur.labels.NO_HOST`。
- Produces:
  ```python
  PRESENCE_FRACTION = .05; VISIBLE_AREA_MM2 = 20.; STRIDE_MM = 8.; AMBIGUOUS_SHARE = .6
  def class_presence(supports, *, fraction=PRESENCE_FRACTION) -> list          # 逐类 True/False；truth==0 → None
  def visible_classes(supports, pixel_area_mm2, *, min_area_mm2=VISIBLE_AREA_MM2) -> list[bool]
  def observation_host_macro(hosts) -> float | None                            # 只平均 truth>0 的宿主 dice
  def host_dice_by_host(hosts) -> list                                          # 13 项；truth==0 → None
  def matched_instance_dice(pred_instances, gt_instances, matches) -> dict[int, float]   # gt id → Dice
  def host_shares(instances, seg_synthseg, spacing) -> np.ndarray              # (n, 14) = host_targets(...)['probs']
  def top_share(probs) -> np.ndarray                                            # (n,)
  def stride_planes(depth, slice_spacing_mm, *, stride_mm=STRIDE_MM) -> list[int]
  def lesion_centroid_planes(instances) -> dict[int, int]
  ```
  `supports`/`hosts` 是 `unified_evaluation.anatomy_metrics` 产出的 `{'labels','truth','predicted','intersection','dice'}` 字典。

- [ ] **Step 1: 写失败的测试**

```python
def test_class_presence_marks_absent_truth_none_and_applies_fraction():
    s = dict(labels=[2,3,4], truth=[100,0,20], predicted=[5,7,0], intersection=[0,0,0])
    assert class_presence(s) == [True, None, False]

def test_visible_classes_uses_truth_area():
    s = dict(labels=[2,3,4], truth=[100,0,20], predicted=[0,0,0], intersection=[0,0,0])
    assert visible_classes(s, .25) == [True, False, False]      # 25 mm², 0, 5 mm²

def test_observation_host_macro_averages_only_hosts_with_truth():
    h = dict(labels=['a','b','c'], truth=[10,0,5], predicted=[9,3,5], intersection=[8,0,2], dice=[.8,0.,.4])
    assert observation_host_macro(h) == pytest.approx(.6)
    assert host_dice_by_host(h) == [.8, None, .4]
    assert observation_host_macro(dict(h, truth=[0,0,0])) is None

def test_matched_instance_dice_only_for_matched_pairs():
    gt = np.zeros((1,4,4), np.int32); gt[0,0:2,0:2] = 1; gt[0,3,3] = 2
    pred = np.zeros((1,4,4), np.int32); pred[0,0:2,0:1] = 7
    assert matched_instance_dice(pred, gt, {1: 7}) == {1: pytest.approx(2*2/(4+2))}

def test_host_shares_in_plane_uses_nearest_within_10mm():
    seg = np.zeros((1,40,40), np.int16); seg[0,:,:20] = 2        # 左白质 SynthSeg 2
    inst = np.zeros((1,40,40), np.int32); inst[0,5,25] = 1; inst[0,5,39] = 2   # 5 mm 内 / 19 mm 外
    p = host_shares(inst, seg, (1.,1.,1.))
    assert p.shape == (2,14) and p[0].argmax() == 0 and p[1].argmax() == NO_HOST
    np.testing.assert_allclose(top_share(p), [1.,1.])

def test_stride_planes_from_header_spacing():
    assert stride_planes(16, 5.) == [0,2,4,6,8,10,12,14]
    assert stride_planes(160, 1.)[:3] == [0,8,16] and stride_planes(160, 1.)[-1] == 152
    assert stride_planes(3, 20.) == [0,1,2]

def test_lesion_centroid_planes_rounds_and_clips():
    inst = np.zeros((6,4,4), np.int32); inst[2:6,0,0] = 1; inst[0,3,3] = 2
    assert lesion_centroid_planes(inst) == {1: 4, 2: 0}       # 均值 3.5 → floor(4.0)=4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `… -m pytest tests/test_aur_gate_measures.py -q -p no:cacheprovider`
Expected: FAIL，`ModuleNotFoundError: anatobind.aur.gate_measures`

- [ ] **Step 3: 实现 `anatobind/aur/gate_measures.py`**

`stride_planes`：`k = max(1, int(round(stride_mm / slice_spacing_mm)))`，返回 `list(range(0, depth, k))`，只用头信息。`lesion_centroid_planes`：对每个 id 取体素 z 均值，`int(np.clip(np.floor(mean + .5), 0, depth-1))`。`host_shares` 直接返回 `host_targets(instances, seg_synthseg, spacing)['probs']`。`matched_instance_dice` 的 Dice 定义同 `unified_evaluation._with_dice`（`2|A∩B|/(|A|+|B|)`）。

- [ ] **Step 4: 跑测试确认通过**

Run: 同上。Expected: 7 passed。

- [ ] **Step 5: 提交**

```bash
git add anatobind/aur/gate_measures.py tests/test_aur_gate_measures.py
git commit -m "AUR gate: per-class presence, per-observation host Dice, matched-instance Dice, host shares, stride and lesion-centroid planes (spec 2026-10-22 section 6)"
```

---

### Task 2: 评价器的门观察与附加量

**Files:**
- Modify: `anatobind/aur/unified_evaluation.py`（`_freeze_verified` 第 54–75 行、`_evaluate_observation` 第 202–243 行、`_binding_metrics` 第 173–191 行、`run_evaluation` 之后新增函数）
- Test: `tests/test_aur_gate_evaluation.py`（新建；现有 `tests/test_aur_unified_evaluation.py` 必须原样通过）

**Interfaces:**
- Consumes: Task 1 全部；`anatobind.aur.unified_data.load_unified_volume(row)`（键 `raw_image, affine, entity, a_ignore, instance, small, hosts{'host','probs'}, spacing, voxel_mm3`）；`tests/test_aur_unified_infer.tiny_model()`、`certified_export()`。
- Produces:
  ```python
  ROLES = {'development_validation': 'val', 'historical_test': 'test'}
  GATE_VERSION = 'aur_gate_evaluation_v1'
  def freeze_gate_observations(rows, modes=('2d','3d'), *, stride_mm=STRIDE_MM) -> dict
      # 返回 {version, evidence, selection='gate_v1_stride_and_lesion_centroid', modes, rows, observations}
      # 观察字段：row_index, row_sha256, input_mode, plane_index, plane_kinds (['volume'] | 子集 of ['stride','lesion_centroid']),
      #          target_instances (list[int]，只有 lesion_centroid 层非空), shape, source_shape, affine_ras
  def run_gate_evaluation(rows, parent_contract_sha256, export_path, out_dir, *, device='cpu',
                          modes=('2d','3d'), crops=None, score_thresholds, stride_mm=STRIDE_MM) -> dict
      # score_thresholds: {source: float}；写 plan_private.json / records_private.json / anonymous_summary.json / complete.json
  ```
  逐观察记录新增：`record['u']['matched_dice']: {gt_id: dice}`；2D 的 `record['u']['source_ids']: list`（平面 gt 序号-1 → 源 3D 实例 id 或 None）；3D 的 `record['r']['instances'][i]['top_share']`；2D 的 `record['r_inplane'] = {'eligible','matched','hits','instances':[{instance, truth, top_share, detected_instance, end_to_end, end_to_end_b0}], 'evidence'}`。

- [ ] **Step 1: 写失败的测试**

文件头导入 `from test_aur_unified_infer import tiny_model, certified_export, supervised_boundaries`、`from anatobind.aur import data_contract as dc`、`from anatobind.aur.data_contract import json_digest`，`ue = anatobind.aur.unified_evaluation`。夹具 `gate_row(tmp_path, monkeypatch)`：仿 `test_aur_unified_evaluation.qualified` 写 image/anatomy/lesion 三个 NIfTI，形状 `(16,16,8)`、仿射 `diag(1,1,1,1)`，`anatomy` 全 2，`lesion[3:6,3:6,2:5]=1`（27 mm³ ≥ 10），行字段 `role='historical_test', split='test'`，其余同 `qualified`；`monkeypatch` 掉 `data_contract.private_output` 与 `source_path`。

```python
def test_gate_freeze_has_volume_stride_and_lesion_centroid_observations(gate_row):
    _, row = gate_row
    plan = ue.freeze_gate_observations([row])
    kinds = {(o['input_mode'], o['plane_index'], tuple(o['plane_kinds'])) for o in plan['observations']}
    assert ('3d', None, ('volume',)) in kinds
    assert ('2d', 0, ('stride',)) in kinds                       # 深度 8、间距 1 → 步长 8 → 只有第 0 层
    assert ('2d', 3, ('lesion_centroid',)) in kinds              # z 2..4 → 均值 3
    centroid = next(o for o in plan['observations'] if 'lesion_centroid' in o['plane_kinds'])
    assert centroid['target_instances'] == [1]
    assert plan['selection'] == 'gate_v1_stride_and_lesion_centroid' and plan['evidence'] == 'NOT_EVIDENCE'

def test_gate_evaluation_accepts_historical_test_role_and_rejects_split_mismatch(gate_row):
    _, row = gate_row
    obs = ue.freeze_gate_observations([row], modes=('3d',))['observations'][0]
    assert ue.evaluate_observation(tiny_model().eval(), row, obs, crop=(4,16,16))['input_mode'] == '3d'
    bad = dict(row, split='val'); obs['row_sha256'] = json_digest(bad)
    with pytest.raises(ValueError, match='observation'):
        ue.evaluate_observation(tiny_model().eval(), bad, obs, crop=(4,16,16))

def test_records_carry_matched_dice_source_ids_and_top_share(gate_row):
    _, row = gate_row
    plan = ue.freeze_gate_observations([row])
    model = tiny_model().eval()
    recs = {(o['input_mode'], o['plane_index']): ue.evaluate_observation(model, row, o,
             crop=(1 if o['input_mode']=='2d' else 4, 16, 16)) for o in plan['observations']}
    vol, plane = recs[('3d', None)], recs[('2d', 3)]
    assert set(vol['u']['matched_dice']) <= set(vol['u']['matches'])
    assert all(v is None or v == 1 for v in plane['u']['source_ids'])
    assert all(0 <= r['top_share'] <= 1 for r in vol['r']['instances'])
    assert plane['r_inplane']['eligible'] == len(plane['r_inplane']['instances']) >= 1
    assert all(0 <= r['truth'] <= 13 and 0 <= r['top_share'] <= 1 for r in plane['r_inplane']['instances'])
    assert vol['r']['instances'][0]['top_share'] == pytest.approx(1.)     # 病灶全在白质里

def test_run_gate_evaluation_uses_per_source_threshold_and_private_outputs(gate_row, monkeypatch, tmp_path, supervised_boundaries):
    home, row = gate_row
    identity = dict(supervised_boundaries[0]['data_contract'], contract_sha256='0'*64)
    exported = certified_export(tmp_path, monkeypatch, supervised_boundaries,
                                lambda s, f: [st.__setitem__('data_contract', identity) for st in (s, f)])
    path = home/'certified.pt'; torch.save(exported, path)
    monkeypatch.setattr(ue, 'validate_private_output', lambda p: dc.validate_private_output(p, private_root=home))
    summary = ue.run_gate_evaluation([row], '0'*64, path, home/'gate', crops={'2d': (1,16,16), '3d': (4,16,16)},
                                     score_thresholds={'isles': .4})
    plan = json.loads((home/'gate/plan_private.json').read_text())
    assert plan['configuration']['score_thresholds'] == {'isles': .4} and plan['version'] == ue.GATE_VERSION
    assert (home/'gate/complete.json').is_file() and summary['gate_version'] == ue.GATE_VERSION
    assert 'synthetic_private' not in json.dumps(summary)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `… -m pytest tests/test_aur_gate_evaluation.py -q -p no:cacheprovider`
Expected: FAIL，`AttributeError: … has no attribute 'freeze_gate_observations'`

- [ ] **Step 3: 实现**

- `_evaluate_observation` 的角色检查改为 `if ROLES.get(row.get('role')) != row.get('split'): raise ValueError()`；开发口径的 `_freeze_verified` 不动。
- `freeze_gate_observations`：复用 `_header_grid`；层间距 = `np.linalg.norm(affine[:3,2])`；stride 层来自 `stride_planes(shape[0], spacing)`；`u_supervised` 的行用 `load_unified_volume(row)['instance']` 取 `lesion_centroid_planes`；同一层合并 `plane_kinds` 与 `target_instances`；2D 观察的 `affine_ras` 平移同 `_freeze_verified`。
- 在 `_evaluate_observation` 里：U 之后加 `record['u']['matched_dice']`（2D 用平面 `instances`，3D 用 `instances`）和 2D 的 `record['u']['source_ids']`；`_binding_metrics` 加参数 `shares`（`(n,14)` 或 None），每行写 `top_share`；2D 在 R 之后（不论 `r_supervised`，只要 `a_supervised and u_supervised`）加 `record['r_inplane'] = _inplane_relation(instances, entity_to_synthseg(volume['entity'][sl]), volume['spacing'], record['u']['matches'], prediction.lesions, predicted_b0_plane)`，其中 `predicted_b0_plane = host_targets(prediction.instances, pred_seg, volume['spacing'])['host']`。
- `run_gate_evaluation` 照 `run_evaluation` 的骨架（私有输出目录校验、导出身份校验 `identity['contract_sha256'] == parent_contract_sha256`、评价器源码快照、前后复核），`configuration` 多记 `score_thresholds` 与 `stride_mm`，summary 加 `gate_version`。

- [ ] **Step 4: 跑测试确认通过**

Run: `… -m pytest tests/test_aur_gate_evaluation.py tests/test_aur_unified_evaluation.py -q -p no:cacheprovider`
Expected: 全部 passed（旧文件计数不变）。

- [ ] **Step 5: 提交**

```bash
git add anatobind/aur/unified_evaluation.py tests/test_aur_gate_evaluation.py
git commit -m "AUR evaluator: gate observations (volume, stride planes, lesion-centroid planes), historical test role, matched-instance Dice, host top share and in-plane relation reference, per-source score thresholds"
```

---

### Task 3: S7 参照在 1 mm 网格上重新打分

**Files:**
- Create: `anatobind/aur/reference_1mm.py`、`scripts/aur_reference_1mm.py`
- Test: `tests/test_aur_reference_1mm.py`

**Interfaces:**
- Consumes: `anatobind.aur.eval.REFERENCE`、`sensitivity_at_threshold`、`instance_rows`、`ignored_rows`；`anatobind.infer.brain_disease.detections(pred, probs, voxel_mm3, family)`；`anatobind.eval.lesion_boxes.load_nnunet_probabilities(npz, label_map_xyz)`；`anatobind.nnunet.brain_disease.DISEASES, fold_dir`；`anatobind.aur.dataset.canonical(img)`；`nibabel.processing.resample_from_to`；`anatobind.aur.unified_data.load_unified_volume`。
- Produces:
  ```python
  NNUNET_ROOT = Path('/data2/congcong/data/FM_data/derived/nnunet')
  def reference_probabilities_on_grid(prediction_path, probability_path, image_path) -> np.ndarray   # (C, z, y, x) float32，目标网格 = canonical(nib.load(image_path))
  def reference_scan_1mm(row, volume, prediction_path, probability_path, disease) -> dict             # {'case': row_sha256 前 12 位, 'gt': [...], 'dets': [...]}，family 一律 'lesion'
  def reference_rows(source, rows) -> list                                                           # sequence == REFERENCE[source]['sequence'] 且 u_supervised
  def reference_summary_1mm(source, rows, *, nnunet_root=NNUNET_ROOT, load_volume=load_unified_volume) -> dict
      # {'source','disease','grid':'aur_1mm_canonical','thr','sensitivity','fp_per_scan','n_scans','n_gt','n_hit','n_fp',
      #  'excluded_cases': int, 'evidence':'NOT_EVIDENCE'}
  ```
  1 mm 标签图 = 重采样后概率的 `argmax(0) == 1`，不重采样 nnU-Net 的标签图（否则 `detections` 的"组件内前景概率 ≥ 0.5"检查会炸）。case 到折：`splits_final.json`，不在任何折 → `excluded_cases += 1`，不补推理。

- [ ] **Step 1: 写失败的测试**

合成：`tmp/nnunet/preprocessed/<name>/splits_final.json`（一折，val 含 `'case_a'`）、`tmp/nnunet/results/.../fold_0/validation/case_a.nii.gz` 与 `.npz`（`probabilities` 形状 `(2, Z, Y, X)`，SimpleITK 顺序，前景在一个 3×3×3 块上 0.9）；`monkeypatch` `fold_dir` 返回该目录；行 `source='isles', case='case_a', sequence='DWI', u_supervised=True, image=<2 mm 网格的 NIfTI>`；`load_volume` 桩返回 `instance`（1 mm 网格、与块重叠的实例）、`small` 全零、`voxel_mm3=1.`。

```python
def test_reference_probabilities_are_resampled_to_the_image_grid(tmp_nnunet):
    probs = reference_probabilities_on_grid(tmp_nnunet.nii, tmp_nnunet.npz, tmp_nnunet.image_1mm)
    assert probs.shape[1:] == tmp_nnunet.shape_1mm_zyx and probs.dtype == np.float32
    assert (probs.argmax(0) == 1).sum() > 0

def test_summary_scores_on_aur_instances_and_counts_missing_folds(tmp_nnunet, monkeypatch):
    rows = [tmp_nnunet.row, dict(tmp_nnunet.row, case='case_missing')]
    s = reference_summary_1mm('isles', rows, nnunet_root=tmp_nnunet.root, load_volume=tmp_nnunet.load_volume)
    assert s['n_scans'] == 1 and s['excluded_cases'] == 1 and s['sensitivity'] == 1. and s['fp_per_scan'] == 0.
    assert s['thr'] == .5 and s['grid'] == 'aur_1mm_canonical' and s['evidence'] == 'NOT_EVIDENCE'
    assert 'case_a' not in json.dumps(s)

def test_reference_rows_keep_only_the_gated_sequence():
    rows = [dict(source='pdgm', sequence='FLAIR', u_supervised=True), dict(source='pdgm', sequence='T1', u_supervised=True),
            dict(source='pdgm', sequence='FLAIR', u_supervised=False)]
    assert reference_rows('pdgm', rows) == rows[:1]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `… -m pytest tests/test_aur_reference_1mm.py -q -p no:cacheprovider`。Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现模块与脚本**

`scripts/aur_reference_1mm.py --contract <合同目录> [--test-role <后继目录>] --qualified-source <归档根> --out <私有目录> [--sources pdgm bmsr isles]`：没有 `--test-role` 时用 `read_contract(..., source_mode='evaluation')['validation']`，有则用 Task 4 的 `read_test_role(...)['rows']`；写 `reference_private.json`（逐 scan）与 `reference_anonymous.json`（三来源 summary）；stdout 只打印每来源 `{source, n_scans, excluded_cases, sensitivity, fp_per_scan}`。`--test-role` 分支在 Task 4 完成前先 `raise SystemExit('test role reader not available')`，Task 4 接上。

- [ ] **Step 4: 跑测试确认通过**

Run: 同上。Expected: 3 passed。

- [ ] **Step 5: 提交**

```bash
git add anatobind/aur/reference_1mm.py scripts/aur_reference_1mm.py tests/test_aur_reference_1mm.py
git commit -m "AUR U gate: rescore the S7 out-of-fold reference on the model's 1 mm canonical grid against the same lesion instances; missing folds are counted, never re-inferred"
```

---

### Task 4: 历史 test 角色的只读后继目录

**Files:**
- Create: `anatobind/aur/test_role.py`
- Modify: `scripts/aur_reference_1mm.py`（接上 `--test-role`）
- Test: `tests/test_aur_test_role.py`

**Interfaces:**
- Consumes（只导入，不改）: `anatobind.aur.data_contract` 的 `DATA_ROOT, verify_contract, read_contract, read_json, write_private_json, private_output, source_path, file_sha256, json_digest, hash_files, inspect_geometry, inspect_label, supervised_row, patient_group, identity, coverage, decoder_environment, REPO`；`anatobind.aur.resample._target_grid`；`anatobind.aur.labels.discrete_labels`；`nibabel.processing.resample_from_to`。
- Produces:
  ```python
  VERSION = 'aur_historical_test_role_v1'; ROLE = 'historical_test'; ROLE_FILE = 'historical_test.json'
  SNAPSHOT_FILES = ('anatobind/aur/test_role.py','anatobind/aur/data_contract.py','anatobind/aur/labels.py',
                    'anatobind/aur/resample.py','anatobind/aur/samples.py')
  def test_rows_from_current(current_rows, native_rows) -> list[dict]      # split=='test'；supervised_row(..., split='test') 再 role=ROLE
  def repair_isles_test_label(row, native_row, out_dir, ledger) -> dict      # 同 data_contract.qualify 的 validation 分支规则；修复文件写 out_dir/'test_repair_v1/label_NNNN.nii.gz'
  def build_historical_test_role(parent_dir, out_dir, *, current=DATA_ROOT/'derived/aur/samples_1mm_v1.json',
                                 native=DATA_ROOT/'derived/aur/samples.json', qualified_source_root=None, workers=2) -> dict
  def read_test_role(out_dir, *, qualified_source_root=None) -> dict        # {'rows', 'contract_sha256', 'parent_contract_sha256'}
  ```
  `build` 写：`historical_test.json`、`policy.json`（`{'historical_test_dense':'opened_for_final_gate_only','label_arrays_read_for':'isles_repair_and_byte_hashing','model_evaluations_allowed':1}`）、`anonymous_summary.json`（`coverage(rows)` + 修复计数）、`contract.json`（`version, parent{path, sha256, complete_sha256}, source_files, outputs, source_snapshot, environment`）、`complete.json`。`read` 校验：`complete/contract` 摘要、父目录即记录路径且 sha 相符、`verify_contract(parent, source_mode='evaluation', …)` 通过、每个输出字节相符、每行 image/anatomy/lesion 当前 sha == `source_hashes`、环境相符；任一不符 → `ValueError('Historical test role verification failed')`，不打印路径。

- [ ] **Step 1: 写失败的测试**

夹具 `parent(tmp_path, monkeypatch)`：照 `test_aur_unified_evaluation.qualified` 建父合同（`validation.json` 含病人 `pa`），并 `monkeypatch` `dc.private_output`、`dc.source_path`、`dc.REPO`（指向临时目录，内含 `SNAPSHOT_FILES` 的占位文件）；`current.json` 含 `pa` 的 train 行与病人 `pb` 的 test 行（`source='isles', sequence='DWI'`，derived `lesion` **全零**（模拟整数截断丢掉的标签；`.99999994` 会被 `discrete_labels` 解码成 1，触发不了修复），native `lesion` 在 2 mm 网格上有一块整数 1，anatomy 为 1 mm 网格），`native.json` 对应。

```python
def test_build_writes_role_policy_contract_and_repairs_truncated_isles(parent):
    home, parent_dir, current, native = parent
    s = build_historical_test_role(parent_dir, home/'test_role_v1', current=current, native=native)
    rows = json.loads((home/'test_role_v1/historical_test.json').read_text())
    assert [r['role'] for r in rows] == ['historical_test'] and rows[0]['split'] == 'test'
    assert s['repair']['isles_repaired_cases'] == 1 and rows[0]['lesion'].endswith('test_repair_v1/label_0000.nii.gz')
    policy = json.loads((home/'test_role_v1/policy.json').read_text())
    assert policy['historical_test_dense'] == 'opened_for_final_gate_only'
    assert 'synthetic_private' not in json.dumps(s)

def test_read_verifies_chain_and_refuses_tampering(parent):
    home, parent_dir, current, native = parent
    build_historical_test_role(parent_dir, home/'t', current=current, native=native)
    got = read_test_role(home/'t')
    assert len(got['rows']) == 1 and got['parent_contract_sha256'] == dc.file_sha256(parent_dir/'contract.json')
    member = home/'t/historical_test.json'; text = member.read_text()
    member.write_text(text.replace('"split": "test"', '"split": "val"'))
    with pytest.raises(ValueError, match='verification failed'):
        read_test_role(home/'t')

def test_build_refuses_test_patient_present_in_parent_roles(parent):
    home, parent_dir, current, native = parent
    rows = json.loads(Path(current).read_text()); rows[1]['patient'] = rows[0]['patient']
    Path(current).write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='intersect'):
        build_historical_test_role(parent_dir, home/'x', current=current, native=native)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `… -m pytest tests/test_aur_test_role.py -q -p no:cacheprovider`。Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现模块，接上脚本的 `--test-role`**

修复规则逐字照 `data_contract.qualify` 第 637–671 行（`_target_grid(native_anatomy)`、`resample_from_to(order=0)`、`discrete_labels`、`inspect_label(destination, image)`、`chmod 0o600`），只是输出子目录名为 `test_repair_v1`，账本只记计数。父合同的 train/validation 病人集合用 `read_contract(parent_dir, verify=False)` 的四个角色文件取 `patient_group`。

- [ ] **Step 4: 跑测试确认通过**

Run: 同上。Expected: 3 passed。

- [ ] **Step 5: 提交**

```bash
git add anatobind/aur/test_role.py scripts/aur_reference_1mm.py tests/test_aur_test_role.py
git commit -m "AUR gate: historical test role as a verified read-only successor of a qualified contract (ISLES labels repaired the validation way), leaving data_contract.py untouched"
```

---

### Task 5: 判门与裁决脚本

**Files:**
- Create: `anatobind/aur/gate.py`、`scripts/aur_gate.py`
- Test: `tests/test_aur_gate.py`

**Interfaces:**
- Consumes: Task 1 的 `class_presence, visible_classes, observation_host_macro, host_dice_by_host, AMBIGUOUS_SHARE`；Task 2 记录格式；Task 3 的 summary 格式；`anatobind.aur.eval.REFERENCE`（三来源）；`anatobind.aur.data_contract.json_digest`。
- Produces:
  ```python
  GATE_V1 = {'version': 'aur_gate_v1', 'evidence': 'NOT_EVIDENCE',
             'a': {'presence_rate_min': .95, 'presence_fraction': .05, 'host_macro_min': .80, 'host_min': .60, 'visible_area_mm2': 20.},
             'u': {'margin': .05, 'matched_dice_median_min': .5, 'bbox_iou': .1, 'sources': ('pdgm','bmsr','isles')},
             'r': {'agreement_min': .90, 'ambiguous_share': .6},
             'two_d': {'stride_mm': 8., 'u_margin_vs_3d': .05}}
  def judge_a(records, mode, registration=GATE_V1) -> dict
  def judge_u_3d(records, reference, registration=GATE_V1) -> dict         # reference: {source: summary}
  def judge_u_2d(records_2d, records_3d, registration=GATE_V1) -> dict
  def judge_r(records, mode, registration=GATE_V1) -> dict
  def verdict(records, reference, *, registration=GATE_V1) -> dict
      # {'version','evidence','registration_sha256','modes':{'3d':{'a','u','r'},'2d':{'a','u','r'}},'overall'}；每项 status ∈ {'pass','fail','unavailable'}
  ```
  规则：A 的 `presence_rate` 逐类 = True 数 / 非 None 数（2D 只算可见类），`rate_min` 取最小；`host_macro` = 逐观察 `observation_host_macro` 的均值；`host_min` = 每宿主先对观察取均值再取 13 个里最小；任一分母为空 → `unavailable`。U 3D 按来源：`fp_per_scan ≤ ref.fp_per_scan + 1e-9`、`sensitivity ≥ ref.sensitivity − margin`、`matched_dice` 中位数 ≥ .5，三来源都 pass 才 pass；缺记录或缺参照的来源 → `unavailable`。U 2D：对每个 `(row_sha256, k ∈ target_instances)`，`hit2d` = 该层某平面 gt 的 `source_ids` 为 k 且在 `matches` 里；`hit3d` = 同 row 3D 记录的 `matches` 含 k；两者在同一集合上算灵敏度，`sens2d ≥ sens3d − margin` 且 Dice 中位数 ≥ .5；`fp_per_image` 只报告。R：3D 用 `record['r']['instances']`、2D 用 `record['r_inplane']['instances']`，只取 `detected_instance is not None` 且 `top_share ≥ .6` 的行，`agreement = (end_to_end == truth) / n`，`n == 0` → `unavailable`；同表报 `end_to_end_b0` 一致率与模棱两可计数。`overall` = 六项全 pass 才 pass，有 unavailable 则 unavailable，否则 fail。

- [ ] **Step 1: 写失败的测试**

用字典字面量构造记录（辅助 `rec(mode, a=None, u=None, r=None, r_inplane=None, source='pdgm', row='r1', plane_kinds=…, target_instances=…)`）。

```python
def test_a_fails_when_a_class_is_never_predicted_despite_host_macro():
    hosts = dict(labels=list(HOST_NAMES), truth=[10]*13, predicted=[10]*13, intersection=[9]*13, dice=[.9]*13)
    ents = dict(labels=list(ENTITY_LABELS), truth=[100]*32, predicted=[100]*31+[0], intersection=[90]*32)
    out = judge_a([rec('3d', a=dict(entities=ents, hosts=hosts))], '3d')
    assert out['host_macro'] == pytest.approx(.9) and out['presence_rate_min'] == 0. and out['status'] == 'fail'

def test_a_is_unavailable_when_a_host_never_has_truth():
    hosts = dict(labels=list(HOST_NAMES), truth=[10]*12+[0], predicted=[10]*13, intersection=[9]*13, dice=[.9]*12+[0.])
    ents = dict(labels=list(ENTITY_LABELS), truth=[100]*32, predicted=[100]*32, intersection=[90]*32)
    assert judge_a([rec('3d', a=dict(entities=ents, hosts=hosts))], '3d')['status'] == 'unavailable'

def test_u_3d_requires_every_reference_source():
    ref = {'pdgm': dict(sensitivity=.8, fp_per_scan=.4), 'bmsr': dict(sensitivity=.7, fp_per_scan=.6)}
    recs = [rec('3d', source=s, u=dict(n_gt=10, n_hit=8, n_fp=0, matches={1:1}, matched_dice={1:.7})) for s in ('pdgm','bmsr','isles')]
    out = judge_u_3d(recs, ref)
    assert out['sources']['pdgm']['status'] == 'pass' and out['sources']['isles']['status'] == 'unavailable' and out['status'] == 'unavailable'

def test_u_3d_blob_within_budget_but_poor_dice_fails():
    ref = {s: dict(sensitivity=.5, fp_per_scan=1.) for s in ('pdgm','bmsr','isles')}
    recs = [rec('3d', source=s, u=dict(n_gt=2, n_hit=2, n_fp=0, matches={1:1,2:2}, matched_dice={1:.2,2:.3})) for s in ref]
    assert judge_u_3d(recs, ref)['status'] == 'fail'

def test_u_2d_pairs_each_lesion_with_its_own_3d_hit():
    vol = rec('3d', row='r1', u=dict(n_gt=2, n_hit=2, n_fp=0, matches={1:5, 2:6}, matched_dice={1:.6, 2:.6}))
    p1 = rec('2d', row='r1', plane_kinds=['lesion_centroid'], target_instances=[1],
             u=dict(n_gt=1, n_hit=1, n_fp=0, matches={1:3}, matched_dice={1:.6}, source_ids=[1]))
    p2 = rec('2d', row='r1', plane_kinds=['lesion_centroid'], target_instances=[2],
             u=dict(n_gt=1, n_hit=0, n_fp=2, matches={}, matched_dice={}, source_ids=[2]))
    out = judge_u_2d([p1, p2], [vol])
    assert out['sources']['pdgm']['sensitivity_2d'] == .5 and out['sources']['pdgm']['sensitivity_3d'] == 1.
    assert out['sources']['pdgm']['status'] == 'fail' and out['fp_per_image'] == 1.

def test_r_excludes_unmatched_and_ambiguous_instances():
    rows = [dict(instance=1, truth=0, top_share=.9, detected_instance=1, end_to_end=0, end_to_end_b0=1),
            dict(instance=2, truth=3, top_share=.5, detected_instance=2, end_to_end=4, end_to_end_b0=4),
            dict(instance=3, truth=5, top_share=.9, detected_instance=None, end_to_end=None, end_to_end_b0=None)]
    out = judge_r([rec('3d', r=dict(instances=rows))], '3d')
    assert out['n'] == 1 and out['agreement'] == 1. and out['ambiguous'] == 1 and out['b0_agreement'] == 0.

def test_unsupervised_rows_leave_denominators():
    out = verdict([rec('3d'), rec('2d', plane_kinds=['stride'])], {})
    assert all(v['status'] == 'unavailable' for m in out['modes'].values() for v in m.values())

def test_verdict_never_passes_on_empty_denominators():
    out = verdict([], {})
    assert out['overall'] == 'unavailable' and out['registration_sha256'] == json_digest(GATE_V1)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `… -m pytest tests/test_aur_gate.py -q -p no:cacheprovider`。Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现 `gate.py` 与 `scripts/aur_gate.py`**

脚本：`--records <records_private.json> --reference <reference_anonymous.json> --out <私有目录>` → `verdict_anonymous.json` 与 `verdict.md`（模式 × 任务一张表，含分母、阈值、`registration_sha256`）；stdout 一行 JSON `{'overall', 'modes': {mode: {task: status}}}`。脚本测试：`runpy` 跑 main，断言两个文件存在、stdout 可解析、不含 `synthetic_private`。

- [ ] **Step 4: 跑测试确认通过**

Run: 同上。Expected: 9 passed（含脚本测试）。

- [ ] **Step 5: 提交**

```bash
git add anatobind/aur/gate.py scripts/aur_gate.py tests/test_aur_gate.py
git commit -m "AUR gate: registered GATE_V1 thresholds and the A/U/R verdict for both input modes - per-class presence, same-grid S7 budget with matched-instance Dice, end-to-end host agreement with ambiguous lesions excluded; empty denominators are unavailable, never pass"
```

---

### Task 6: fastMRI 的像素网格 DICOM 入口

**Files:**
- Create: `anatobind/aur/dicom_pixelgrid.py`
- Test: `tests/test_aur_dicom_pixelgrid.py`（复用 `tests/test_aur_dicom.make_mr`）

**Interfaces:**
- Consumes: `anatobind.aur.dicom` 的 `DicomInputError, SourceRef, FrameMapping, _read_header, _read_pixels, _Header, _hash_source, verify_sources, _private_intake, CLASSIC_MR`。
- Produces:
  ```python
  ASSUMED_IOP = np.array([1.,0.,0.,0.,1.,0.])
  @dataclass(frozen=True)
  class PixelGridInput:
      image_dhw: np.ndarray; affine_ras: np.ndarray; input_mode: str; source_refs: tuple; frame_map: tuple
      measurement_unit: str; voxel_volume_mm3: float | None; pixel_area_mm2: float
      geometry: str = 'assumed'; slice_order: str = 'filename_index'
  def read_pixel_grid_series(files, input_mode=None) -> PixelGridInput
  ```
  规则：经典 MR SOP class、`Modality=='MR'`、`MONOCHROME2`、`SamplesPerPixel==1`、单帧、`Rows/Columns/PixelSpacing/SeriesInstanceUID` 全序列一致（否则 `geometry_unsupported` / `mixed_series`）；3D 还要 `SpacingBetweenSlices` 存在、为正、一致；层序 = 文件名末尾整数（`re.search(r'(\d+)\s*$', path.stem)`），缺失或重复 → `filename_order_unsupported`；**IPP/IOP 即使存在也不读**；`affine_lps` 列向 `(ps_col,0,0)`、行向 `(0,ps_row,0)`、深度 `(0,0,sbs 或 1.0)`、原点 0，`affine_ras = diag(-1,-1,1,1) @ affine_lps`；像素经 `_read_pixels`（`RescaleSlope/Intercept` 照用）；读完 `verify_sources`。错误全部是 `DicomInputError(code)`。

- [ ] **Step 1: 写失败的测试**

```python
def test_single_plane_affine_from_spacing_only(tmp_path):
    p = make_mr(tmp_path, ImagePositionPatient=None, ImageOrientationPatient=None)
    x = read_pixel_grid_series([p])
    assert x.input_mode == '2d' and x.geometry == 'assumed' and x.pixel_area_mm2 == 6 and x.voxel_volume_mm3 is None
    np.testing.assert_allclose(x.affine_ras, [[-3,0,0,0],[0,-2,0,0],[0,0,1,0],[0,0,0,1]])
    np.testing.assert_array_equal(x.image_dhw, [[[0,1,2],[3,4,5]]])

def test_stack_orders_by_filename_index_and_uses_spacing_between_slices(tmp_path):
    p2 = make_mr(tmp_path, 2, ImagePositionPatient=None, ImageOrientationPatient=None, pixels=np.arange(10,16).reshape(2,3))
    p1 = make_mr(tmp_path, 1, ImagePositionPatient=None, ImageOrientationPatient=None)
    x = read_pixel_grid_series([p2, p1])
    assert x.input_mode == '3d' and x.affine_ras[2,2] == 9 and x.voxel_volume_mm3 == 54
    np.testing.assert_array_equal(x.image_dhw[:,0,0], [0, 10])
    assert [(m.output_depth_index, m.source_ref_index) for m in x.frame_map] == [(0,1),(1,0)]

def test_present_positions_are_ignored(tmp_path):
    p = make_mr(tmp_path, ImageOrientationPatient=[0,1,0,.6,0,.8], ImagePositionPatient=[10,20,30])
    np.testing.assert_allclose(read_pixel_grid_series([p]).affine_ras[:3,:3], np.diag([-3,-2,1]))

@pytest.mark.parametrize('changes,code', [
    (dict(Rows=4, pixels=np.zeros((4,3), np.uint16)), 'geometry_unsupported'),
    (dict(SpacingBetweenSlices=None), 'geometry_unsupported'),
    (dict(SeriesInstanceUID='1.2.826.0.1.3680043.10.987.9'), 'mixed_series')])
def test_stack_rejections(tmp_path, changes, code):
    p1 = make_mr(tmp_path, 1, ImagePositionPatient=None, ImageOrientationPatient=None)
    p2 = make_mr(tmp_path, 2, ImagePositionPatient=None, ImageOrientationPatient=None, **changes)
    with pytest.raises(DicomInputError) as e: read_pixel_grid_series([p1, p2])
    assert e.value.code == code

def test_empty_and_unordered_names(tmp_path):
    with pytest.raises(DicomInputError) as e: read_pixel_grid_series([])
    assert e.value.code == 'empty_input'
    p = make_mr(tmp_path, ImagePositionPatient=None, ImageOrientationPatient=None)
    q = p.with_name('noindex.dcm'); q.write_bytes(p.read_bytes())
    with pytest.raises(DicomInputError) as e: read_pixel_grid_series([q])
    assert e.value.code == 'filename_order_unsupported'

def test_public_repr_hides_private_values(tmp_path):
    p = make_mr(tmp_path, ImagePositionPatient=None, ImageOrientationPatient=None)
    x = read_pixel_grid_series([p]); public = repr(x) + repr(x.source_refs)
    assert 'SYNTHETIC_PRIVATE' not in public and str(tmp_path) not in public
```

- [ ] **Step 2: 跑测试确认失败**

Run: `… -m pytest tests/test_aur_dicom_pixelgrid.py -q -p no:cacheprovider`。Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现**

`_Header` 的 `position` 填零、`orientation` 填 `ASSUMED_IOP.reshape(2,3)`，这样 `_read_pixels` 原样复用。

- [ ] **Step 4: 跑测试确认通过**

Run: 同上，并跑 `tests/test_aur_dicom.py` 确认严格入口未变。Expected: 全 passed。

- [ ] **Step 5: 提交**

```bash
git add anatobind/aur/dicom_pixelgrid.py tests/test_aur_dicom_pixelgrid.py
git commit -m "AUR: pixel-grid DICOM entry for series without position or orientation - assumed standard axial frame, filename slice order, geometry tagged as assumed; the strict entry is unchanged"
```

---

### Task 7: fastMRI 批处理

**Files:**
- Create: `anatobind/aur/fastmri_batch.py`、`scripts/aur_fastmri_batch.py`
- Test: `tests/test_aur_fastmri_batch.py`

**Interfaces:**
- Consumes: Task 6 `read_pixel_grid_series`；`anatobind.aur.unified_infer.infer_input, model_from_unified_export, UnifiedPrediction`；`anatobind.aur.labels.HOST_SIDE, HOST_NAMES, NO_HOST`；`anatobind.aur.data_contract.validate_private_output, write_private_json, file_sha256`；`nibabel.processing.resample_from_to`；`anatobind.aur.crops.to_zyx`。
- Produces:
  ```python
  SEQUENCE_PATTERN = re.compile(r'file_brain_(AX[A-Z0-9]+)_')
  def series_dirs(root) -> list[Path]
  def sequence_of(series_dir) -> str | None
  def resample_to_1mm(image_dhw, affine_ras, mode) -> tuple[np.ndarray, np.ndarray]    # (image_1mm_dhw, affine_1mm)；3d 各向同性 1 mm，2d 面内 1 mm、深度 1
  def map_back(labels_1mm_dhw, affine_1mm, native_shape_dhw, native_affine) -> np.ndarray   # 最近邻
  def screen_side(host_index) -> str | None     # 'left'→'screen_right'，'right'→'screen_left'，脑干/无宿主→None
  def lesion_record(lesion, mode, slice_index=None) -> dict   # {'mode','slice','box'(原网格 zyx 半开),'host','host_name','screen_side','host_probs','score', 面积或体积}
  def run_series(model, series_dir, *, modes, crops, score_threshold, device) -> tuple[dict, dict]   # (record, arrays)；record 无路径
  def run_batch(root, export_path, out_dir, *, modes=('2d','3d'), crops=None, score_threshold=.5, device='cpu', limit=None) -> dict
  ```
  `run_batch`：序列目录按名排序、`limit` 截断；每个序列 `try: read_pixel_grid_series(sorted(dir.glob('*.dcm')))`，`DicomInputError` → `skipped[code] += 1`；3D 模式：重采样 1 mm → `infer_input` → `anatomy/instances` 映回原 16 层网格；2D 模式：逐层 `read_pixel_grid_series([file])` → 面内重采样 → `infer_input` → 映回；写 `out/series_<序号 05d>/{arrays.npz(anatomy_3d, instances_3d, anatomy_2d, instances_2d), record_private.json}`；已有 `record_private.json` 的序列跳过（续跑）；摘要 `anonymous_summary.json`：`n_series, n_ok, skipped{code:n}, by_sequence{seq:n}, export_sha256, score_threshold, modes, geometry='assumed', patient_coordinates='not_produced', evidence='NOT_EVIDENCE'`。

- [ ] **Step 1: 写失败的测试**

夹具：`tiny_model()` 导出为 pilot 格式（`certified_export` 的 format1 路径或直接 `torch.save` 一个 `format_version=1` 的字典，按 `model_from_unified_export` 的字段要求）；合成序列：`make_mr` 生成 3 层、`Rows=Columns=16`、`PixelSpacing=[1,1]`、`SpacingBetweenSlices=1`、无 IPP/IOP，放在 `root/file_brain_AXFLAIR_200_000001/`；再建一个空目录 `file_brain_AXT2_200_000002/` 和一个第二层 `Rows=8` 的坏序列。

```python
def test_screen_side_mapping_under_assumed_orientation():
    left = HOST_SIDE.index('left'); right = HOST_SIDE.index('right'); stem = HOST_SIDE.index(None)
    assert screen_side(left) == 'screen_right' and screen_side(right) == 'screen_left'
    assert screen_side(stem) is None and screen_side(NO_HOST) is None

def test_resample_round_trip_is_identity_on_a_1mm_grid():
    img = np.random.default_rng(0).random((3,16,16)).astype(np.float32)
    aff = np.diag([-1.,-1.,1.,1.])
    out, aff2 = resample_to_1mm(img, aff, '3d')
    assert out.shape == img.shape and np.allclose(aff2, aff)
    lab = np.zeros((3,16,16), np.int16); lab[1,4:8,4:8] = 5
    np.testing.assert_array_equal(map_back(lab, aff2, img.shape, aff), lab)

def test_batch_skips_bad_series_and_counts_by_reason(batch_root, pilot_export, private_home):
    s = run_batch(batch_root, pilot_export, private_home/'batch', crops={'2d': (1,16,16), '3d': (2,16,16)})
    assert s['n_series'] == 3 and s['n_ok'] == 1
    assert s['skipped'] == {'empty_input': 1, 'geometry_unsupported': 1} and s['by_sequence'] == {'AXFLAIR': 1}
    rec = json.loads(next((private_home/'batch').glob('series_*/record_private.json')).read_text())
    assert rec['sequence'] == 'AXFLAIR' and rec['geometry'] == 'assumed' and set(rec['modes']) == {'2d','3d'}
    arrays = np.load(next((private_home/'batch').glob('series_*/arrays.npz')))
    assert arrays['anatomy_3d'].shape == (3,16,16) and arrays['anatomy_2d'].shape == (3,16,16)
    assert 'file_brain' not in json.dumps(s) and 'SYNTHETIC_PRIVATE' not in json.dumps(s)

def test_batch_resume_skips_completed_series(batch_root, pilot_export, private_home, monkeypatch):
    run_batch(batch_root, pilot_export, private_home/'b', crops={'2d': (1,16,16), '3d': (2,16,16)})
    monkeypatch.setattr(fb, 'run_series', lambda *a, **k: pytest.fail('completed series was recomputed'))
    s = run_batch(batch_root, pilot_export, private_home/'b', crops={'2d': (1,16,16), '3d': (2,16,16)})
    assert s['n_ok'] == 1 and s['resumed'] == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `… -m pytest tests/test_aur_fastmri_batch.py -q -p no:cacheprovider`。Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现模块与脚本**

`scripts/aur_fastmri_batch.py --root <序列根> --export <模型导出> --out <私有目录> [--modes 2d 3d] [--score-threshold .5] [--limit N] [--device cpu] [--crops-2d 1 160 160] [--crops-3d 128 160 160]`；stdout 只打印摘要 JSON。重采样：把 `image_dhw` 转成 `Nifti1Image(image.transpose(2,1,0), affine)`，目标网格由 `affine` 的方向向量归一化成 1 mm、形状按 FOV 四舍五入；`resample_from_to(order=1)`；映回用 `order=0`。

- [ ] **Step 4: 跑测试确认通过**

Run: 同上。Expected: 4 passed。

- [ ] **Step 5: 提交**

```bash
git add anatobind/aur/fastmri_batch.py scripts/aur_fastmri_batch.py tests/test_aur_fastmri_batch.py
git commit -m "AUR: fastMRI series batch runner over the pixel-grid entry - both modes resampled to 1 mm and mapped back to the native grid, screen-side host reporting, per-series skip accounting, resumable private outputs"
```

---

### Task 8: fastMRI+ 框约定审计与命中率报告

**Files:**
- Create: `anatobind/aur/fastmri_plus_report.py`、`scripts/aur_fastmri_plus_report.py`
- Test: `tests/test_aur_fastmri_plus_report.py`

**Interfaces:**
- Consumes: `anatobind.data_engine.fastmri.read_fastmri_plus_rows, convert_box_csv_to_rss, box_iou_2d`；Task 7 的 `record_private.json`（`lesions[*].mode/slice/box`）与 `series_dirs/sequence_of`；Task 6 `read_pixel_grid_series`。
- Produces:
  ```python
  def boxes_by_file(csv_path) -> dict[str, dict[int, list[dict]]]        # 文件干名 → 层 → [{x,y,width,height,label}]
  def box_rows(box, n_rows, convention) -> tuple[int, int]              # 'top': (y, y+height)；'bottom': convert_box_csv_to_rss 的行区间
  def box_contrast(pixels_hw, boxes, convention) -> float | None        # 框内均值 / 框外均值
  def convention_audit(images, boxes_by_file, *, limit=20) -> dict      # images: 可迭代 (file_stem, slice, pixels_hw)；{'top','bottom','n_images','chosen'}
  def slice_hit_rate(detections_by_slice, boxes_by_slice, n_rows, *, convention, iou=.1) -> dict   # {'n_boxes','n_hit','n_fp','n_images'}
  def series_stem(series_dir) -> str                                      # 目录名即 CSV 的 file 干名
  def report_batch(batch_dir, csv_path, *, series_depths=None) -> dict    # {'n_boxes','n_hit','n_fp','n_images','unmatched_files','unmatched_slices', 按序列类型与模式分的子表}
  ```
  匹配：每层把检测的 zyx 框投到 (x,y,w,h)，与标注框按 IoU 贪心一对一（IoU ≥ .1）；未匹配的检测计 FP。CSV `file` 没有对应序列目录、或 `slice ≥ 序列层数` → `unmatched_files / unmatched_slices` 计数。

- [ ] **Step 1: 写失败的测试**

```python
def test_convention_audit_prefers_the_frame_where_boxes_are_bright():
    img = np.zeros((20,20), np.float32); img[2:6, 3:9] = 100.             # 框画在顶部坐标系 y=2,h=4
    rows = {'f': {0: [dict(x=3, y=2, width=6, height=4, label='x')]}}
    a = convention_audit([('f', 0, img)], rows)
    assert a['top'] > a['bottom'] and a['chosen'] == 'top' and a['n_images'] == 1

def test_slice_hit_rate_matches_one_to_one_and_counts_fp():
    dets = {0: [[0,2,3,1,6,9], [0,15,15,1,18,18]]}                        # zyx 半开框
    boxes = {0: [dict(x=3, y=2, width=6, height=4, label='x')]}
    assert slice_hit_rate(dets, boxes, 20, convention='top') == dict(n_boxes=1, n_hit=1, n_fp=1, n_images=1)

def test_unmatched_annotations_are_counted_not_raised(tmp_path):
    csv = tmp_path/'brain.csv'
    csv.write_text('file,slice,study_level,x,y,width,height,label\nfile_brain_AXFLAIR_200_000009,99,No,1,1,2,2,Edema\n')
    out = report_batch(tmp_path/'nonexistent_batch', csv, series_depths={'file_brain_AXFLAIR_200_000001': 16})
    assert out['unmatched_files'] == 1 and out['n_boxes'] == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `… -m pytest tests/test_aur_fastmri_plus_report.py -q -p no:cacheprovider`。Expected: FAIL。

- [ ] **Step 3: 实现模块与脚本**

`report_batch(batch_dir, csv_path, *, series_depths=None) -> dict` 读 Task 7 的记录（目录名↔干名的映射写在 `record_private.json` 里，匿名摘要不含干名）；脚本 `scripts/aur_fastmri_plus_report.py --batch <私有批目录> --annotations <csv> --dicom-root <根> --out <私有目录> [--audit-limit 20]` 写 `convention_audit_anonymous.json` 与 `hit_rate_anonymous.json`（按序列类型分、按模式分）。

- [ ] **Step 4: 跑测试确认通过**

Run: 同上。Expected: 3 passed。

- [ ] **Step 5: 提交**

```bash
git add anatobind/aur/fastmri_plus_report.py scripts/aur_fastmri_plus_report.py tests/test_aur_fastmri_plus_report.py
git commit -m "AUR: fastMRI+ box convention audit and per-slice hit rate against the batch records, reported only, never gated"
```

---

### Task 9: 真实数据上的只读试跑与记录

不训练、不碰历史 test、不占 GPU。目的：证明三条链在真实文件上跑得通，并把计数与哈希落盘。

**Files:**
- Create: `docs/verification/2026-10-22/aur_gate_layer/RESULTS.md`、`docs/verification/2026-10-22/aur_gate_layer/RUN.json`
- Modify: `STATUS.md`（顶部一节）、`CLAUDE.md`（测试计数一行）

- [ ] **Step 1: 全套 CPU 回归**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -rs -p no:cacheprovider 2>&1 | tail -5`
Expected: 0 failed；记录 passed/skipped 计数与秒数（新测试约 +30）。

- [ ] **Step 2: S7 参照在开发验证集上重打分（CPU，不读 test）**

`qualified_source_root` 取 `docs/verification/2026-10-21/aur_evaluation_interfaces/RUN.json` 里记录的归档根。
Run: `… nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_reference_1mm.py --contract ~/outputs/aur_qualified_training_20261019/data_contract/final_v3 --qualified-source <归档根> --out ~/outputs/aur_gate_layer_20261022/reference_val_v1`
Expected: 三来源各一行 `{source, n_scans, excluded_cases, sensitivity, fp_per_scan}`；`excluded_cases` 记进 RESULTS；数字标 `NOT_EVIDENCE`，只说明链路通。

- [ ] **Step 3: fastMRI 批跑 3 个序列（pilot 模型，只证明跑通）**

Run: `… nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_fastmri_batch.py --root /data2/congcong/data/FM_data/fastMRI_lh_brain_knee/brain/multicoil_train --export ~/outputs/aur_unified_input_20261019/pilot_fixed24_v1/stage_III_export.pt --out ~/outputs/aur_gate_layer_20261022/fastmri_smoke_v1 --limit 3 --device cpu`
Expected: 摘要 `n_ok=3`（或跳过原因计数）；逐序列墙钟记进 RUN.json，用来估全量 4469 的 CPU/GPU 时间。模型是 pilot（`pilot_only=True`），输出只证明接口，不作任何质量说明。

- [ ] **Step 4: fastMRI+ 框约定审计（像素，不用模型）**

Run: `… python scripts/aur_fastmri_plus_report.py --batch ~/outputs/aur_gate_layer_20261022/fastmri_smoke_v1 --annotations /data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/brain.csv --dicom-root /data2/congcong/data/FM_data/fastMRI_lh_brain_knee/brain/multicoil_train --out ~/outputs/aur_gate_layer_20261022/fastmri_plus_audit_v1 --audit-limit 20`
Expected: `convention_audit_anonymous.json` 给出 `chosen`；写进 RESULTS 作为后续命中率报告的固定约定。

- [ ] **Step 5: 写 RESULTS.md 与 RUN.json，更新 STATUS/CLAUDE**

RESULTS 五段：做了什么、每条命令与匿名输出、哈希（代码提交、导出 sha、合同 sha、输出 complete.json sha）、没做什么（无监督模型、未开 test、无质量结论）、下一步 = 第 2 块计划。STATUS 顶部加一节指向它；CLAUDE.md 测试计数行更新。

- [ ] **Step 6: 提交并推送**

```bash
git add docs/verification/2026-10-22/aur_gate_layer STATUS.md CLAUDE.md
git commit -m "Docs: gate layer and fastMRI batch runner verified on synthetic data and smoke-tested on real files (S7 reference rescored on validation, three fastMRI series, box convention audit); no supervised model, historical test untouched"
https_proxy=http://127.0.0.1:7897 git push origin main
```

---

## 自检记录

- 规格覆盖：§6.1 → Task 1、2、5；§6.2 → Task 2、3、5；§6.3 → Task 1、2、5；§6.4 → Task 6、7、8；§6 通用规则的 test 角色 → Task 4（真实构建留给第 2 块）；§6.5 不涉及；§8 的评价器两处扩展 → Task 2。
- 接口一致性：`matched_dice`、`source_ids`、`top_share`、`r_inplane`、`plane_kinds`、`target_instances` 在 Task 2 产出、Task 5 消费，字段名相同；`reference_summary_1mm` 的 `sensitivity/fp_per_scan` 键与 Task 5 `judge_u_3d` 读的键相同；Task 7 `record['lesions'][*]['box']` 与 Task 8 的 zyx 半开框约定相同。
- 未在本计划里做的：真实 test 角色构建、最终判门、正式 II/III 训练（第 2 块）。
