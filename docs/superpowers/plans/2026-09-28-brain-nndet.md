# Brain Small-Lesion Detector, Second Arm: nnDetection fold 0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up nnDetection (Retina U-Net) on the same 253 fastMRI+ brain FLAIR volumes and patient folds as S2, train fold 0, decide rule A (nnDetection's fold-0 hits at ≤ 2 FP per volume ≥ nnU-Net 2d's fold-0 hits + ceil(0.05 × n_gt)), and ship this arm's inference entry for S5.

**Architecture:** nnDetection runs untouched (commit 97a58f3) in its own conda env `nndet` (python 3.8, torch 1.11). A thin runner script in that env turns nnDetection's saved ensembler states into JSON boxes (default postprocessing, nnDetection's one-voxel margin undone, restored to the original image array). The repository (env nvgen) reads that JSON, reorders axes, and scores it with S2's detection metrics against the registry boxes. The nnDetection task is built from Dataset903 (same images, same folds), one instance per registry lesion painted largest-first.

**Tech Stack:** nnDetection 97a58f3 (PyTorch 1.11 + cu113, PyTorch Lightning ≤ 1.4.2, nnU-Net v1 1.7.1, SimpleITK 2.0) in `~/anaconda3/envs/nndet`; repository side Python 3.11, numpy 2.4, nibabel, scipy, pytest in `~/anaconda3/envs/nvgen`.

**Spec:** `docs/superpowers/specs/2026-09-28-brain-nndet-design.md` (decisions N1–N14). Read it before any task.

**Precondition:** S2 is merged to main; branch `build/brain-nndet` from main; worktree `../foundation_model-nndet`. The controller commits the spec and this plan as the branch's first commit.

**Plan dry run (2026-09-28, throwaway):** the nvgen-side code and tests of Tasks 1–8 were written verbatim into a scratch export of `build/brain-detector` 3e915fb and run: the new tests 45 passed, 1 skipped (the nndet-env module); the full suite 740 passed, 1 skipped. Nothing of the nndet-env side (Task 3's `tests/test_nndet_runner.py`, the runner's nnDetection calls) could run, because the env does not exist yet. Implementers still run every step themselves; these numbers are a hint, not evidence.

## Global Constraints

- nvgen tests: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest <paths> -q -p no:cacheprovider`. Tests never read `/data2`.
- nndet-env tests: `bash -c 'source scripts/nndet_env.sh && python -m pytest tests/test_nndet_runner.py -q -p no:cacheprovider --noconftest'` (`--noconftest`: `tests/conftest.py` imports repository fixtures that do not belong in the nndet env).
- The Bash tool runs zsh. Every command that sources a script or activates conda is wrapped in `bash -c '…'` or `bash -ex <<'EOF' … EOF`.
- The two environments never import each other: `scripts/nndet_runner.py` imports only the standard library and numpy at module level (nnDetection, torch, omegaconf only inside functions) and never imports `anatobind`; repository code never imports `nndet`. They exchange JSON only; `splits_final.pkl` holds plain Python lists (N10).
- Never source `scripts/nndet_env.sh` and `scripts/nnunet_env.sh` in the same shell.
- Logs: the controller's environment install is in `~/logs/nndet_install/0{1,2,3}_env.{sh,log}`; every task log in this plan goes to the worktree's untracked `logs/nndet_install/` or `logs/brain_nndet/` (never `git add` `logs/`).
- Paths: `det_data=/data2/congcong/data/FM_data/derived/nndet`, `det_models=/data2/congcong/data/FM_data/derived/nndet_models`, runner outputs `/data2/congcong/data/FM_data/derived/nndet_runs/`, smoke root `/data2/congcong/data/FM_data/derived/nndet_smoke/{data,models,runner}`. No `.` in any nnDetection path. nnDetection source `~/src/nnDetection` detached at `97a58f31`; never modify its tracked files.
- Task `Task903_FastMRIBrainSmallLesion`, id 903, one class `{"0": "small_lesion"}`, modality `{"0": "FLAIR"}`, plan `D3V001_3d`, model `RetinaUNetV001_D3V001_3d`, training dir `${det_models}/Task903_FastMRIBrainSmallLesion/RetinaUNetV001_D3V001_3d/fold0`.
- Training (N7): default `v001` (50 + 10 SWA epochs × 2500 batches, fp16) with `--sweep`. Rule A reads only boxes extracted from `sweep_predictions` with `BoxEnsemblerSelective.get_default_parameters()`; the swept extraction is NOT_GATE; only `model_last` is used.
- Box conversion (N8): nnDetection box `[lo_a, lo_b, hi_a, hi_b, lo_c, hi_c]` with lo = min − 1, hi = max + 1 → low ends += 1 in the preprocessed space → `restore_detection` → original array axes (slice, row, col) as `[s_lo, r_lo, s_hi, r_hi, c_lo, c_hi]` (the runner's JSON layout) → repository corner box `(x0, y0, z0, x1, y1, z1) = (c_lo, r_lo, s_lo, c_hi, r_hi, s_hi)`.
- Metrics unchanged: `anatobind.eval.detection_metrics` (IoU 0.1 one-to-one, thresholds 0.05–0.95 step 0.05, FP_MAX 2, gate 0.5); truth = registry boxes via `anatobind.nnunet.brain_lesion.gt_boxes`.
- Rule A (N3): continue to five folds iff nnDetection's hits at its ≤ 2 FP per volume operating point on the evaluated folds ≥ nnU-Net 2d's hits at its own operating point on the same folds + ceil(0.05 × n_gt) (probe: 92 + 14 = 106 of 280 on fold 0). Fold-subset numbers are never called the D1 gate.
- Compute (N9, N14): one GPU, idle-checked with nvidia-smi, never a card another process uses; nice 19; `det_num_threads=8`; ≤ 48 CPU threads in total; projected training + sweep > 24 h → report to the user and ask (training keeps running meanwhile).
- Nothing is overwritten or deleted: every output directory or file that already exists is refused; nnDetection's training dir must not exist before launch (its "overwrite" mode would reuse it). Deletions are only listed for the user.
- Commits: repository-local author, English messages, no Co-Authored-By or any AI trace (the user's rule overrides the harness reminder). No push, no merge.

## Review Focus

1. A lesion that loses every voxel to later-painted instances — expected: the build refuses and names the lesion (Task 2 test `test_a_lesion_covered_by_later_instances_is_refused`).
2. `splits_final.pkl` missing or replaced by nnDetection's auto-generated KFold split — expected: the launcher refuses when the pickle differs from Dataset903's splits (Task 7 test `test_preflight_refusals`).
3. A fold case missing from the runner JSON — expected: evaluation refuses and names the case instead of scoring fewer volumes (Task 4 test `test_nndet_scans_refuse_a_missing_case`).
4. Runner JSON with a foreign layout string or a non-zero class — expected: the repository reader refuses (Task 4 test `test_reader_refuses_foreign_layout_and_other_classes`).
5. A predicted box thinner than one voxel, partly outside, or fully outside the grid — expected: inference rows keep at least one voxel, are clipped, and a box fully outside is dropped (Task 4 test `test_lesion_rows_round_half_up_keep_one_voxel_clip_drop_and_sort`).

---

### Task 1: nnDetection environment, `scripts/nndet_env.sh`, toy smoke run

**Files:**
- Create: `scripts/nndet_env.sh`
- Create: `tests/test_nndet_env.py`
- Create: `docs/nndet_install.md`

**Interfaces:**
- Produces: `scripts/nndet_env.sh`; conda env `~/anaconda3/envs/nndet`; toy training dir `/data2/congcong/data/FM_data/derived/nndet_smoke/models/Task000D3_Example/RetinaUNetV001_D3V001_3d/fold0` containing `sweep_predictions/*_boxes.pt`, `val_predictions/`, `plan.pkl`, `plan_inference.pkl`, `config.yaml`, `model_last.ckpt` (Task 3 integration uses it).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nndet_env.py
import os
import subprocess
from pathlib import Path

ENV_SH = Path(__file__).resolve().parents[1] / "scripts/nndet_env.sh"


def _env_after_source(extra):
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"], **extra}
    out = subprocess.run(["bash", "-c", f"source {ENV_SH} && env"], capture_output=True, text=True, check=True,
                         env=env).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def test_nndet_env_sets_paths_threads_and_hides_nnunet():
    e = _env_after_source({"nnUNet_preprocessed": "/x", "nnUNet_raw": "/y", "nnUNet_results": "/z"})
    assert e["det_data"] == "/data2/congcong/data/FM_data/derived/nndet"
    assert e["det_models"] == "/data2/congcong/data/FM_data/derived/nndet_models"
    assert "." not in e["det_data"] and "." not in e["det_models"]
    assert e["OMP_NUM_THREADS"] == "1" and e["det_num_threads"] == "8" and e["PYTHONNOUSERSITE"] == "1"
    assert e["PATH"].split(":")[0] == os.path.expanduser("~/anaconda3/envs/nndet/bin")
    assert not any(k.startswith("nnUNet_") for k in e)
```

- [ ] **Step 2: Run to verify it fails** — `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_nndet_env.py -q -p no:cacheprovider` → FAIL (`bash: …/scripts/nndet_env.sh: No such file or directory`).

- [ ] **Step 3: Implement** `scripts/nndet_env.sh`:

```bash
# scripts/nndet_env.sh — nnDetection environment (spec 2026-09-28 brain-nndet §3). Source it with bash; never in the
# same shell as scripts/nnunet_env.sh (nnU-Net v1 inside nnDetection and nnU-Net v2 both read nnUNet_preprocessed).
unset nnUNet_raw nnUNet_preprocessed nnUNet_results nnUNet_raw_data_base RESULTS_FOLDER
export PATH="$HOME/anaconda3/envs/nndet/bin:$PATH"
export PYTHONNOUSERSITE=1
export det_data=/data2/congcong/data/FM_data/derived/nndet
export det_models=/data2/congcong/data/FM_data/derived/nndet_models
export OMP_NUM_THREADS=1
export det_num_threads=8
export det_verbose=1
```

- [ ] **Step 4: Run** the test → 1 passed.

- [ ] **Step 5: The conda env — installed by the controller ahead of the branch (2026-09-29, done 02:21); do not reinstall.** Every script and log is in `~/logs/nndet_install/` (read them for Step 8):
  - `01_env.sh/.log` — the README route: `conda create -n nndet python=3.8` succeeded (python 3.8.20 from conda-forge); the classic solver then ran 28 min on `gxx_linux-64==9.3.0` against the full Tsinghua conda-forge repodata (`~/.condarc`: Tsinghua mirror, strict channel priority) and was stopped.
  - `02_env.sh/.log` — `--override-channels -c defaults gxx_linux-64=9.3.0`: unsatisfiable (the env's python/libgcc come from conda-forge). No conda compiler is used from here on: the system has gcc-10/g++-10, which nvcc 11.3 accepts (host gcc ≤ 10; the default g++ is 11.4).
  - `03_env.sh/.log` — `conda install -y --override-channels -c nvidia/label/cuda-11.3.1 cuda` succeeded (nvcc 11.3.122 in the env). Its next step, a bare `pip`, resolved to `/usr/bin/pip` (system python 3.10): on this machine `conda activate` puts the env's `bin` after `~/.local/bin` and `/usr/bin`. It was stopped while downloading; nothing was installed. From here on every command names the env's python by absolute path.
  - `04_env.sh/.log` — the 1.6 GB torch wheel download through the proxy was cut at 0.9 GB and failed its sha256 check.
  - `05_env.sh/.log` — aria2c got HTTP 403 from the Aliyun mirror with its default user agent.
  - `06_env.sh/.log` — succeeded up to the build: the wheel fetched with `aria2c -U curl/7.81.0 -x 8 -s 8` from `https://mirrors.aliyun.com/pytorch-wheels/cu113/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl` and checked against the official sha256 `b6a799bdb6ee3d914e5e62bddb4276d4a10248c1af4f2d217738e5f9ee27485b` (`sha256sum -c`: OK); then `python -m pip install <wheel> torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`, `-r requirements.txt`, `hydra-core --upgrade --pre`, `pytorch_model_summary`, `pytest`; `import torch` → `1.11.0+cu113 11.3`. The build failed: `cannot import name 'packaging' from 'pkg_resources'` (setuptools 70.3.0 is too new for torch 1.11's cpp_extension).
  - `07_build.sh/.log` — the working finish:

```bash
E=$HOME/anaconda3/envs/nndet
export PATH="$E/bin:$PATH"
export PYTHONNOUSERSITE=1
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
"$E/bin/python" -m pip install setuptools==59.5.0
cd ~/src/nnDetection
CC=gcc-10 CXX=g++-10 CUDA_HOME="$E" FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 "$E/bin/python" -m pip install -v -e .
git status --short --untracked-files=no          # printed nothing
echo "ENV_SETUP_DONE $(date '+%F %T')"            # ENV_SETUP_DONE 2026-09-29 02:21:04
```

  Controller's check after 07 (`CUDA_VISIBLE_DEVICES=3`, from `/tmp`): `import torch, nndet._C, nndet` → `1.11.0+cu113 11.3 True`; `pytorch_lightning 1.4.2`, `SimpleITK 2.0.2`, `numpy 1.24.4`.

Precondition for the implementer: `grep ENV_SETUP_DONE ~/logs/nndet_install/07_build.log` prints the line above and `bash -c 'cd ~/src/nnDetection && git rev-parse --short HEAD && git status --short --untracked-files=no'` prints `97a58f3` and nothing else. If not, stop and report (BLOCKED); do not install anything yourself.

- [ ] **Step 6: Import check** (pick an idle GPU first: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits` and `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader`; a card with < 1000 MiB and no compute app):

```bash
bash -c 'source scripts/nndet_env.sh && cd /tmp && CUDA_VISIBLE_DEVICES=<idle gpu> python -c "import torch, nndet._C, nndet; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"'
```

Expected: `1.11.0 11.3 True`. If it fails with `module 'distutils' has no attribute 'version'`, run `bash -c 'source scripts/nndet_env.sh && pip install setuptools==59.5.0'`, record it, and repeat the check.

- [ ] **Step 7: Toy smoke run** (refuse if the smoke root exists: `test ! -e /data2/congcong/data/FM_data/derived/nndet_smoke`):

```bash
bash -ex <<'EOF' > logs/nndet_install/02_smoke.log 2>&1
source scripts/nndet_env.sh
export det_data=/data2/congcong/data/FM_data/derived/nndet_smoke/data
export det_models=/data2/congcong/data/FM_data/derived/nndet_smoke/models
mkdir -p "$det_data" "$det_models"
cd "$det_data"
nndet_example
nice -n 19 nndet_prep 000 -np 4 -npp 4
nice -n 19 nndet_unpack "$det_data/Task000D3_Example/preprocessed/D3V001_3d/imagesTr" 4
CUDA_VISIBLE_DEVICES=<idle gpu> nice -n 19 nndet_train 000 -o train=smoke --sweep
EOF
```

Expected: exit 0. Then confirm and record each output:

```bash
T=/data2/congcong/data/FM_data/derived/nndet_smoke/models/Task000D3_Example/RetinaUNetV001_D3V001_3d/fold0
ls $T; ls $T/sweep_predictions | head; ls $T/sweep_predictions/*_boxes.pt | wc -l
bash -c "source scripts/nndet_env.sh && python -c \"import pickle; p = pickle.load(open('$T/plan.pkl', 'rb')); print('inference_plan' in p, p['data_identifier'], p['planner_id'], p['transpose_backward']); q = pickle.load(open('$T/plan_inference.pkl', 'rb')); print(sorted(q['inference_plan']))\""
grep -n -E 'Predict cases with default settings|Start parameter sweep|Found inference plan' $T/train.log | head
```

Record the wall time of the sweep (timestamps of "Predict cases with default settings" and the end of `train.log`) and the number of validation cases it predicted.

- [ ] **Step 8: Write `docs/nndet_install.md`** with: every install run of Step 5 (01–07; each script verbatim and the last 15 lines of its log, and why it was abandoned); the commands of Steps 6–7; `bash -c 'source scripts/nndet_env.sh && pip freeze'`; `~/anaconda3/envs/nndet/bin/nvcc -V`; `g++-10 --version | head -1`; the Step 6 output; the Step 7 confirmations (file listing, the plan line, the swept parameter names, sweep wall time and case count). Paste real outputs only.

- [ ] **Step 9: Commit** — `git add scripts/nndet_env.sh tests/test_nndet_env.py docs/nndet_install.md && git commit -m "nnDetection environment: env file, install record and toy smoke run with sweep"`

---

### Task 2: Task builder module `anatobind/nndet/brain_task.py`

**Files:**
- Create: `anatobind/nndet/__init__.py`, `anatobind/nndet/brain_task.py`
- Test: `tests/test_nndet_task.py`

**Interfaces:**
- Produces:
  - constants `TASK_ID = 903`, `TASK_NAME = "Task903_FastMRIBrainSmallLesion"`, `PLAN_ID = "D3V001_3d"`, `MODEL_ID = "RetinaUNetV001_D3V001_3d"`, `CLASS_ID = 0`, `DATASET_META`
  - `member_mask(members, shape) -> bool array (col, row, slice)`; members = `[{"x", "y", "width", "height", "slice"}]` in the RSS frame (same as S2's `paint_members`)
  - `paint_instances(members_of, shape) -> (uint16 instance map, {lesion_id: instance})`, raises `ValueError` naming a lesion that keeps no voxel
  - `tight_box(mask) -> (c_min, c_max, r_min, r_max, s_min, s_max)` inclusive; `changed_boxes(members_of, inst, instance_of) -> sorted lesion ids`
  - `check_same_support(inst, binary, case)` raises `ValueError` starting with `"{case}:"`
  - `instances_json(instance_of) -> {"instances": {"1": 0, ...}}`
  - `write_instance_label(inst, image_path, out_path)`
  - `build_task(det_data, cases, write_case) -> Path`, `write_case(case, image_out_path, labels_dir)`; refuses an existing task
  - `splits_payload(splits_json) -> [{"train": [str], "val": [str]}]`; `write_splits_pkl(preprocessed_dir, splits_json) -> Path` (refuses an existing file, needs the directory); `read_splits_pkl(path) -> list`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nndet_task.py
import json

import nibabel as nib
import numpy as np
import pytest

from anatobind.nndet.brain_task import (
    DATASET_META, TASK_NAME, build_task, changed_boxes, check_same_support, instances_json, paint_instances,
    read_splits_pkl, splits_payload, write_instance_label, write_splits_pkl,
)


def _box(x, y, w, h, s):
    return {"x": x, "y": y, "width": w, "height": h, "slice": s}


def test_largest_first_keeps_a_small_lesion_inside_a_big_one():
    members_of = {7: [_box(2, 2, 10, 10, 1)], 3: [_box(4, 4, 2, 2, 1)]}       # lesion 3 sits inside lesion 7
    inst, instance_of = paint_instances(members_of, (20, 20, 3))
    assert instance_of == {7: 1, 3: 2}
    assert inst.dtype == np.uint16
    assert int((inst == 2).sum()) == 4 and int((inst == 1).sum()) == 100 - 4
    assert changed_boxes(members_of, inst, instance_of) == []                 # 7 keeps its corners


def test_ties_paint_the_smaller_lesion_id_first():
    members_of = {5: [_box(0, 0, 2, 2, 0)], 2: [_box(10, 10, 2, 2, 0)]}
    _, instance_of = paint_instances(members_of, (20, 20, 1))
    assert instance_of == {2: 1, 5: 2}


def test_a_lesion_covered_by_later_instances_is_refused():
    members_of = {1: [_box(0, 0, 2, 2, 0)], 2: [_box(0, 0, 1, 2, 0)], 3: [_box(1, 0, 1, 2, 0)]}
    with pytest.raises(ValueError, match="lesion 1 "):
        paint_instances(members_of, (4, 4, 1))


def test_changed_boxes_lists_a_lesion_whose_edge_was_overwritten():
    members_of = {7: [_box(2, 2, 4, 4, 0)], 3: [_box(2, 2, 1, 4, 0)]}        # 3 covers 7's whole first column
    inst, instance_of = paint_instances(members_of, (10, 10, 1))
    assert changed_boxes(members_of, inst, instance_of) == [7]


def test_support_must_equal_the_binary_label():
    inst = np.zeros((4, 4, 1), np.uint16)
    inst[1, 1, 0] = 1
    binary = (inst > 0).astype(np.uint8)
    check_same_support(inst, binary, "c")
    binary[2, 2, 0] = 1
    with pytest.raises(ValueError, match="^c:"):
        check_same_support(inst, binary, "c")


def test_instances_json_maps_every_instance_to_class_zero():
    assert instances_json({7: 1, 3: 2}) == {"instances": {"1": 0, "2": 0}}
    assert instances_json({}) == {"instances": {}}


def test_instance_label_shares_the_image_affine_and_refuses_a_shape_mismatch(tmp_path):
    aff = np.diag([0.6875, 0.6875, 5.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 2), np.float32), aff), str(tmp_path / "img.nii.gz"))
    inst = np.zeros((4, 5, 2), np.uint16)
    inst[1, 2, 1] = 3
    write_instance_label(inst, tmp_path / "img.nii.gz", tmp_path / "lab.nii.gz")
    lab = nib.load(str(tmp_path / "lab.nii.gz"))
    assert np.allclose(lab.affine, aff) and lab.get_data_dtype() == np.uint16
    assert np.asarray(lab.dataobj)[1, 2, 1] == 3
    with pytest.raises(ValueError, match="shape"):
        write_instance_label(np.zeros((4, 4, 2), np.uint16), tmp_path / "img.nii.gz", tmp_path / "lab2.nii.gz")


def test_build_task_layout_and_refusal(tmp_path):
    seen = []

    def write_case(case, img, labels_dir):
        seen.append((case, img.name, labels_dir.name))

    base = build_task(tmp_path, ["a", "b"], write_case)
    assert base == tmp_path / TASK_NAME
    assert (base / "raw_splitted/imagesTr").is_dir() and (base / "raw_splitted/labelsTr").is_dir()
    assert seen == [("a", "a_0000.nii.gz", "labelsTr"), ("b", "b_0000.nii.gz", "labelsTr")]
    assert json.loads((base / "dataset.json").read_text()) == DATASET_META
    assert DATASET_META["dim"] == 3 and DATASET_META["labels"] == {"0": "small_lesion"}
    with pytest.raises(FileExistsError):
        build_task(tmp_path, ["a"], write_case)


def test_splits_pickle_is_plain_lists_and_refuses_overwrite(tmp_path):
    sj = [{"train": ["b", "c"], "val": ["a"]}, {"train": ["a"], "val": ["b", "c"]}]
    with pytest.raises(FileNotFoundError):
        write_splits_pkl(tmp_path / "missing", sj)
    p = write_splits_pkl(tmp_path, sj)
    assert b"numpy" not in p.read_bytes()
    assert read_splits_pkl(p) == splits_payload(sj) == sj
    with pytest.raises(FileExistsError):
        write_splits_pkl(tmp_path, sj)
```

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_nndet_task.py -q -p no:cacheprovider` → `ModuleNotFoundError: No module named 'anatobind.nndet'`.

- [ ] **Step 3: Implement.** `anatobind/nndet/__init__.py`:

```python
"""nnDetection second arm of the brain small-lesion detector (spec 2026-09-28 brain-nndet)."""
```

`anatobind/nndet/brain_task.py`:

```python
"""nnDetection task for the brain small-lesion detector's second arm (spec 2026-09-28 brain-nndet §4).

Same 253 cases, images and folds as Dataset903. Each registry lesion becomes one instance of class 0, painted
largest-first (ties: smaller lesion id first) so a lesion sharing voxels with a bigger one keeps its own; the
instance map's non-zero voxels equal Dataset903's binary label. Nothing here imports nnDetection.
"""
import json
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np

TASK_ID = 903
TASK_NAME = f"Task{TASK_ID}_FastMRIBrainSmallLesion"
PLAN_ID = "D3V001_3d"
MODEL_ID = f"RetinaUNetV001_{PLAN_ID}"
CLASS_ID = 0
DATASET_META = {"task": TASK_NAME, "name": "FastMRIBrainSmallLesion", "dim": 3, "test_labels": False,
                "labels": {"0": "small_lesion"}, "modalities": {"0": "FLAIR"}}


def member_mask(members, shape):
    m = np.zeros(shape, bool)
    for b in members:
        m[b["x"]:b["x"] + b["width"], b["y"]:b["y"] + b["height"], b["slice"]] = True
    return m


def paint_instances(members_of, shape):
    """Instance map (uint16, (col, row, slice)) and {lesion_id: instance}; instances are numbered 1.. in painting
    order, largest painted-voxel count first, so smaller lesions overwrite the voxels they share."""
    masks = {lid: member_mask(ms, shape) for lid, ms in members_of.items()}
    order = sorted(masks, key=lambda lid: (-int(masks[lid].sum()), lid))
    inst = np.zeros(shape, np.uint16)
    instance_of = {}
    for k, lid in enumerate(order, start=1):
        inst[masks[lid]] = k
        instance_of[lid] = k
    for lid, k in instance_of.items():
        if not (inst == k).any():
            raise ValueError(f"lesion {lid} lost every voxel to later instances")
    return inst, instance_of


def tight_box(mask):
    idx = np.nonzero(mask)
    return tuple(int(v) for a in idx for v in (a.min(), a.max()))


def changed_boxes(members_of, inst, instance_of):
    """Lesion ids whose tight box in the instance map differs from the tight box of their own members."""
    return sorted(lid for lid, k in instance_of.items()
                  if tight_box(inst == k) != tight_box(member_mask(members_of[lid], inst.shape)))


def check_same_support(inst, binary, case):
    binary = np.asarray(binary)
    if inst.shape != binary.shape or not np.array_equal(inst > 0, binary > 0):
        raise ValueError(f"{case}: instance map support differs from the Dataset903 binary label")


def instances_json(instance_of):
    return {"instances": {str(k): CLASS_ID for k in sorted(instance_of.values())}}


def write_instance_label(inst, image_path, out_path):
    img = nib.load(str(image_path))
    if tuple(img.shape) != tuple(inst.shape):
        raise ValueError(f"label shape {inst.shape} != image shape {img.shape} ({image_path})")
    out = nib.Nifti1Image(np.asarray(inst, np.uint16), img.affine)
    out.set_data_dtype("uint16")
    out.header.set_xyzt_units("mm")
    nib.save(out, str(out_path))


def build_task(det_data, cases, write_case):
    """<det_data>/<TASK_NAME>/raw_splitted/{imagesTr,labelsTr} and dataset.json; never rebuilds an existing task."""
    base = Path(det_data) / TASK_NAME
    if base.exists():
        raise FileExistsError(f"{base} exists; the task is never rebuilt in place")
    (base / "raw_splitted" / "imagesTr").mkdir(parents=True)
    (base / "raw_splitted" / "labelsTr").mkdir()
    for c in cases:
        write_case(c, base / "raw_splitted" / "imagesTr" / f"{c}_0000.nii.gz", base / "raw_splitted" / "labelsTr")
    (base / "dataset.json").write_text(json.dumps(DATASET_META, indent=1))
    return base


def splits_payload(splits_json):
    """nnU-Net splits_final.json content -> nnDetection splits of plain lists (numpy 1 must read the pickle)."""
    return [{"train": [str(c) for c in s["train"]], "val": [str(c) for c in s["val"]]} for s in splits_json]


def write_splits_pkl(preprocessed_dir, splits_json):
    p = Path(preprocessed_dir) / "splits_final.pkl"
    if p.exists():
        raise FileExistsError(f"{p} exists")
    p.write_bytes(pickle.dumps(splits_payload(splits_json), protocol=4))
    return p


def read_splits_pkl(path):
    return pickle.loads(Path(path).read_bytes())
```

- [ ] **Step 4: Run** `tests/test_nndet_task.py` → 9 passed.

- [ ] **Step 5: Commit** — `git add anatobind/nndet/__init__.py anatobind/nndet/brain_task.py tests/test_nndet_task.py && git commit -m "nnDetection task builder: largest-first instance painting, support check against Dataset903, plain-list splits pickle"`

---

### Task 3: nnDetection runner `scripts/nndet_runner.py`

**Files:**
- Create: `scripts/nndet_runner.py`
- Test: `tests/test_nndet_runner_pure.py` (nvgen), `tests/test_nndet_runner.py` (nndet env; skipped in nvgen)
- Modify: `docs/nndet_install.md` (append the toy runner record)

**Interfaces:**
- Consumes (nndet env only, imported inside functions): `nndet.inference.restore.restore_detection`, `nndet.inference.ensembler.detection.BoxEnsemblerSelective` (`get_default_parameters`, `get_case_ids`, `from_checkpoint`, `update_parameters`, `get_case_result`, `.properties`), `nndet.inference.helper.predict_dir`, `nndet.inference.loading.get_loader_fn`, `nndet.planning.PLANNER_REGISTRY` (`run_preprocessing_test`), `omegaconf.OmegaConf`; test only: `nndet.io.transforms.instances.instances_to_boxes_np`.
- Produces:
  - `LAYOUT` (string, also in `anatobind/nndet/boxes.py`, Task 4)
  - `undo_margin(boxes) -> (N, 6) float`, `to_original(boxes_prep, properties, transpose_backward) -> (N, 6)`
  - `extract_cases(state_dir, params) -> {case: {"boxes", "scores", "labels"}}`, `gt_cases(prep_dir, plan_id="D3V001_3d") -> {case: {"boxes", "scores", "instances"}}`, `predict_case(image, train_dir, work) -> {"case": {...}}`
  - CLI `extract --state DIR --params default|swept [--train-dir DIR] --out JSON`, `gt --prep DIR --out JSON`, `predict --image NII --train-dir DIR --work DIR --out JSON`; output `{"layout": LAYOUT, "source": {...}, "cases": {...}}`; refuses an existing `--out`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_nndet_runner_pure.py
import importlib.util
from pathlib import Path

import numpy as np
import pytest

RUNNER = Path(__file__).resolve().parents[1] / "scripts/nndet_runner.py"


def _runner():
    spec = importlib.util.spec_from_file_location("nndet_runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_undo_margin_shifts_only_the_low_ends():
    r = _runner()
    b = np.array([[0.0, 1.0, 5.0, 6.0, 2.0, 4.0]])
    assert r.undo_margin(b).tolist() == [[1.0, 2.0, 5.0, 6.0, 3.0, 4.0]]
    assert b.tolist() == [[0.0, 1.0, 5.0, 6.0, 2.0, 4.0]]
    assert r.undo_margin(np.zeros((0, 6))).shape == (0, 6)


def test_runner_imports_neither_anatobind_nor_nndetection_at_module_level():
    top = [l for l in RUNNER.read_text().splitlines() if l.startswith(("import ", "from "))]
    assert not any(w in l for l in top for w in ("anatobind", "nndet", "torch", "omegaconf"))


def test_cli_refuses_an_existing_out_before_touching_nndetection(tmp_path):
    out = tmp_path / "x.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        _runner().main(["gt", "--prep", str(tmp_path), "--out", str(out)])
```

```python
# tests/test_nndet_runner.py — run in the nndet env (see Global Constraints); skipped elsewhere
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nndet")
from nndet.io.transforms.instances import instances_to_boxes_np  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

RUNNER = Path(__file__).resolve().parents[1] / "scripts/nndet_runner.py"
spec = importlib.util.spec_from_file_location("nndet_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _original():
    inst = np.zeros((6, 40, 30), np.int32)          # original array axes (slice, row, col)
    inst[2, 10:12, 5:7] = 1                          # single-slice 2 x 2 lesion
    inst[1:4, 20:25, 12:20] = 2                      # three-slice lesion
    return inst


def _tight(inst, k):
    s, r, c = np.nonzero(inst == k)
    return np.array([s.min(), r.min(), s.max() + 1, r.max() + 1, c.min(), c.max() + 1], float)


def test_roundtrip_with_crop_and_no_resampling_is_exact():
    inst = _original()
    crop = [(1, 5), (5, 35), (2, 28)]
    boxes, ids = instances_to_boxes_np(inst[1:5, 5:35, 2:28])
    props = {"original_spacing": np.array([5.0, 0.6875, 0.6875]),
             "spacing_after_resampling": np.array([5.0, 0.6875, 0.6875]), "crop_bbox": crop}
    got = runner.to_original(boxes, props, [0, 1, 2])
    for row, k in zip(got, ids):
        assert np.allclose(row, _tight(inst, k))


def test_roundtrip_with_transpose_and_resampling_is_within_one_voxel():
    inst = _original()
    tf = [2, 0, 1]
    tb = [int(i) for i in np.argsort(tf)]
    orig_sp = np.array([5.0, 0.86, 0.86])            # original axis order
    target_sp = np.array([5.0, 0.6875, 0.6875])
    res = zoom(inst.transpose(tf), orig_sp[tf] / target_sp[tf], order=0)
    boxes, ids = instances_to_boxes_np(res)
    props = {"original_spacing": orig_sp, "spacing_after_resampling": target_sp[tf],
             "crop_bbox": [(0, 6), (0, 40), (0, 30)]}
    got = runner.to_original(boxes, props, tb)
    assert sorted(int(k) for k in ids) == [1, 2]
    for row, k in zip(got, ids):
        assert np.all(np.abs(row - _tight(inst, k)) <= 1.0 + 1e-9)
```

- [ ] **Step 2: Run to verify they fail** — nvgen: `… -m pytest tests/test_nndet_runner_pure.py tests/test_nndet_runner.py -q -p no:cacheprovider` → pure tests fail (`FileNotFoundError` on the runner path), `test_nndet_runner.py` skipped; nndet env (Global Constraints command) → collection error, runner missing.

- [ ] **Step 3: Implement** `scripts/nndet_runner.py`:

```python
#!/usr/bin/env python
# scripts/nndet_runner.py
"""nnDetection side of the brain second arm (spec 2026-09-28 brain-nndet §5, §7, §8).

Runs ONLY in the nndet env (python 3.8, torch 1.11, nnDetection 97a58f3) and never imports anatobind. It writes
{"layout", "source", "cases"} JSON; boxes are nnDetection's layout on the ORIGINAL image array axes (slice, row, col),
[s_lo, r_lo, s_hi, r_hi, c_lo, c_hi], half-open floats, after undoing nnDetection's one-voxel margin.

  bash -c 'source scripts/nndet_env.sh && python scripts/nndet_runner.py extract --state DIR --params default --out J'
  ... extract --state DIR --params swept --train-dir DIR --out J
  ... gt --prep ${det_data}/Task903_FastMRIBrainSmallLesion/preprocessed --out J
  ... predict --image NII --train-dir DIR --work DIR --out J
"""
import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

LAYOUT = "nndet [s_lo, r_lo, s_hi, r_hi, c_lo, c_hi] on original array axes (slice, row, col), half-open"
LOW_ENDS = [0, 1, 4]            # x1, y1, z1 of nnDetection's (x1, y1, x2, y2, z1, z2)


def undo_margin(boxes):
    """nnDetection boxes are [min - 1, max + 1] per axis (instances_to_boxes_np); shifting the low ends by +1 gives
    the half-open [min, max + 1). Applies in the preprocessed space, before restoring."""
    b = np.array(boxes, dtype=float).reshape(-1, 6)
    b[:, LOW_ENDS] += 1.0
    return b


def to_original(boxes_prep, properties, transpose_backward):
    b = undo_margin(boxes_prep)
    if len(b) == 0:
        return b
    from nndet.inference.restore import restore_detection
    return np.asarray(restore_detection(b, transpose_backward=list(transpose_backward),
                                        original_spacing=properties["original_spacing"],
                                        spacing_after_resampling=properties["spacing_after_resampling"],
                                        crop_bbox=properties["crop_bbox"]), float)


def _np(x):
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x)


def default_parameters():
    from nndet.inference.ensembler.detection import BoxEnsemblerSelective
    return BoxEnsemblerSelective.get_default_parameters()


def extract_cases(state_dir, params):
    """Saved ensembler states (sweep_predictions, or predict's state dir) -> postprocess with params, restore=False,
    then undo the margin and restore with each case's own properties."""
    from nndet.inference.ensembler.detection import BoxEnsemblerSelective
    out = {}
    for case_id in sorted(BoxEnsemblerSelective.get_case_ids(state_dir)):
        ens = BoxEnsemblerSelective.from_checkpoint(base_dir=state_dir, case_id=case_id)
        ens.update_parameters(**params)
        res = ens.get_case_result(restore=False)
        boxes = _np(res["pred_boxes"]).reshape(-1, 6)
        p = ens.properties
        out[case_id] = {"boxes": to_original(boxes, p, p["transpose_backward"]).tolist(),
                        "scores": [float(s) for s in _np(res["pred_scores"]).reshape(-1)],
                        "labels": [int(v) for v in _np(res["pred_labels"]).reshape(-1)]}
    return out


def gt_cases(prep_dir, plan_id="D3V001_3d"):
    """Ground-truth boxes after nnDetection preprocessing, through the same conversion as predictions (spec §5 check
    two), with the instance ids still present after resampling."""
    prep_dir = Path(prep_dir)
    plan = pickle.loads((prep_dir / f"{plan_id}.pkl").read_bytes())
    stage = prep_dir / plan["data_identifier"] / "imagesTr"
    out = {}
    for f in sorted(stage.glob("*_boxes.pkl")):
        case_id = f.name[:-len("_boxes.pkl")]
        cand = pickle.loads(f.read_bytes())
        props = pickle.loads((stage / f"{case_id}.pkl").read_bytes())
        ids = [int(i) for i in cand["instances"]]
        boxes = np.asarray(cand["boxes"], float).reshape(-1, 6) if ids else np.zeros((0, 6))
        if len(boxes) != len(ids):
            raise ValueError(f"{case_id}: {len(boxes)} boxes for {len(ids)} instances")
        out[case_id] = {"boxes": to_original(boxes, props, plan["transpose_backward"]).tolist(),
                        "scores": [1.0] * len(ids), "instances": ids}
    return out


def predict_case(image, train_dir, work, num_processes=0):
    """One FLAIR NIfTI -> detections with model_last, 8 mirror TTA and default postprocessing (spec §8)."""
    from nndet.inference.helper import predict_dir
    from nndet.inference.loading import get_loader_fn
    from nndet.planning import PLANNER_REGISTRY
    from omegaconf import OmegaConf
    work, train_dir = Path(work), Path(train_dir)
    if work.exists():
        raise FileExistsError(f"{work} exists")
    raw = work / "raw_splitted" / "imagesTs"
    raw.mkdir(parents=True)
    shutil.copyfile(image, raw / "case_0000.nii.gz")
    plan = pickle.loads((train_dir / "plan.pkl").read_bytes())
    plan["inference_plan"] = {}                     # empty -> BoxEnsembler.from_case falls back to its defaults
    cfg = OmegaConf.load(str(train_dir / "config.yaml"))
    cfg.merge_with_dotlist(["host.parent_data=${oc.env:det_data}", "host.parent_results=${oc.env:det_models}"])
    cfg = OmegaConf.to_container(cfg, resolve=True)
    PLANNER_REGISTRY.get(plan["planner_id"]).run_preprocessing_test(
        preprocessed_output_dir=work / "preprocessed", splitted_4d_output_dir=work / "raw_splitted", plan=plan,
        num_processes=num_processes)
    state = work / "state"
    state.mkdir()
    predict_dir(source_dir=work / "preprocessed" / plan["data_identifier"] / "imagesTs", target_dir=state, cfg=cfg,
                plan=plan, source_models=train_dir, num_models=1, num_tta_transforms=None,
                model_fn=get_loader_fn(mode="last"), restore=False, save_state=True)
    return extract_cases(state, default_parameters())


def main(argv=None):
    ap = argparse.ArgumentParser(description="nnDetection runner for the brain second arm (nndet env only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--state", type=Path, required=True)
    e.add_argument("--params", choices=("default", "swept"), required=True)
    e.add_argument("--train-dir", type=Path)
    e.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("gt")
    g.add_argument("--prep", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("predict")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists")
    if a.cmd == "extract":
        if a.params == "default":
            params = default_parameters()
        else:
            if a.train_dir is None:
                ap.error("--params swept needs --train-dir")
            params = pickle.loads((a.train_dir / "plan_inference.pkl").read_bytes())["inference_plan"]
        cases, source = extract_cases(a.state, params), {"state": str(a.state), "params": a.params}
    elif a.cmd == "gt":
        cases, source = gt_cases(a.prep), {"prep": str(a.prep)}
    else:
        cases = predict_case(a.image, a.train_dir, a.work)
        source = {"image": str(a.image), "train_dir": str(a.train_dir)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"layout": LAYOUT, "source": source, "cases": cases}))
    print(f"wrote {a.out}: {len(cases)} cases, {sum(len(c['boxes']) for c in cases.values())} boxes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** — nvgen: `tests/test_nndet_runner_pure.py tests/test_nndet_runner.py` → 3 passed, 1 skipped (module-level skip counts as 1). nndet env: `bash -c 'source scripts/nndet_env.sh && python -m pytest tests/test_nndet_runner.py -q -p no:cacheprovider --noconftest'` → 2 passed.

- [ ] **Step 5: Toy integration** (nndet env; uses Task 1's smoke outputs; `<idle gpu>` as in Task 1 Step 6):

```bash
bash -ex <<'EOF' > logs/nndet_install/03_runner_toy.log 2>&1
source scripts/nndet_env.sh
export det_data=/data2/congcong/data/FM_data/derived/nndet_smoke/data
export det_models=/data2/congcong/data/FM_data/derived/nndet_smoke/models
T=$det_models/Task000D3_Example/RetinaUNetV001_D3V001_3d/fold0
W=/data2/congcong/data/FM_data/derived/nndet_smoke/runner
python scripts/nndet_runner.py extract --state $T/sweep_predictions --params default --out $W/extract_default.json
python scripts/nndet_runner.py extract --state $T/sweep_predictions --params swept --train-dir $T --out $W/extract_swept.json
python scripts/nndet_runner.py gt --prep $det_data/Task000D3_Example/preprocessed --out $W/gt.json
IMG=$(ls $det_data/Task000D3_Example/raw_splitted/imagesTs/*_0000.nii.gz | head -1)
CUDA_VISIBLE_DEVICES=<idle gpu> python scripts/nndet_runner.py predict --image $IMG --train-dir $T --work $W/predict_work --out $W/predict.json
EOF
```

Then check against nnDetection's own outputs (nndet env) and paste the printout:

```bash
bash -c 'source scripts/nndet_env.sh && python - <<"PY"
import json, pickle
from pathlib import Path
import numpy as np
import SimpleITK as sitk
D = Path("/data2/congcong/data/FM_data/derived/nndet_smoke/data/Task000D3_Example")
T = Path("/data2/congcong/data/FM_data/derived/nndet_smoke/models/Task000D3_Example/RetinaUNetV001_D3V001_3d/fold0")
W = Path("/data2/congcong/data/FM_data/derived/nndet_smoke/runner")
plan = pickle.load(open(D / "preprocessed/D3V001_3d.pkl", "rb"))
stage = D / "preprocessed" / plan["data_identifier"] / "imagesTr"
tb = list(plan["transpose_backward"])
# 1) swept extraction == nnDetection val_predictions (restored), high ends equal, low ends shifted by the spacing ratio
sw = json.load(open(W / "extract_swept.json"))["cases"]
worst = 0.0
for case, rec in sw.items():
    theirs = pickle.load(open(T / "val_predictions" / f"{case}_boxes.pkl", "rb"))
    tbx = np.asarray(theirs["pred_boxes"], float).reshape(-1, 6)
    ours = np.asarray(rec["boxes"], float).reshape(-1, 6)
    props = pickle.load(open(stage / f"{case}.pkl", "rb"))
    scale = np.asarray(props["spacing_after_resampling"])[tb] / np.asarray(props["original_spacing"])
    assert ours.shape == tbx.shape and np.allclose(rec["scores"], np.asarray(theirs["pred_scores"]).reshape(-1))
    assert np.allclose(ours[:, [2, 3, 5]], tbx[:, [2, 3, 5]])
    assert np.allclose(ours[:, [0, 1, 4]] - tbx[:, [0, 1, 4]], scale[[0, 1, 2]])
    worst = max(worst, float(np.abs(ours - tbx).max()) if len(ours) else 0.0)
print("swept vs val_predictions: cases", len(sw), "all consistent; max abs diff", worst)
# 2) gt conversion == tight boxes of the raw instance labels (exact when the case was not resampled)
gt = json.load(open(W / "gt.json"))["cases"]
n, errs = 0, []
for case, rec in gt.items():
    lab = sitk.GetArrayFromImage(sitk.ReadImage(str(D / "raw_splitted/labelsTr" / f"{case}.nii.gz")))
    for k, b in zip(rec["instances"], rec["boxes"]):
        s, r, c = np.nonzero(lab == k)
        tight = np.array([s.min(), r.min(), s.max() + 1, r.max() + 1, c.min(), c.max() + 1], float)
        errs.append(float(np.abs(np.asarray(b) - tight).max())); n += 1
print("gt instances", n, "max coordinate error", max(errs), "exact", sum(e < 1e-6 for e in errs))
pr = json.load(open(W / "predict.json"))["cases"]
print("predict cases", list(pr), "boxes", len(pr["case"]["boxes"]), "top scores", sorted(pr["case"]["scores"])[-3:])
PY'
```

Expected: the swept-vs-nnDetection assertions hold; gt max coordinate error ≤ 1.0 (0 where not resampled); predict returns one case `case`. Also `grep -n "Found inference plan" logs/nndet_install/03_runner_toy.log` must show `Found inference plan: {} for prediction` for the predict call (default postprocessing, spec N11). If any of this fails, stop and report (BLOCKED) — nothing downstream may run.

Append to `docs/nndet_install.md` a section "Runner on the toy task" with the commands and the printout.

- [ ] **Step 6: Commit** — `git add scripts/nndet_runner.py tests/test_nndet_runner_pure.py tests/test_nndet_runner.py docs/nndet_install.md && git commit -m "nnDetection runner: default-parameter extraction, margin undo and restore to the original array, ground-truth and single-volume prediction paths"`

---

### Task 4: Repository-side reader and evaluation helpers

**Files:**
- Create: `anatobind/nndet/boxes.py`, `anatobind/eval/brain_nndet.py`
- Modify: `anatobind/eval/brain_detector.py` (add `strata_maps`; nothing else changes)
- Test: `tests/test_nndet_boxes.py`, `tests/test_nndet_eval.py`

**Interfaces:**
- Consumes: `anatobind.eval.brain_detector.scan_record`, `anatobind.eval.detection_metrics.{IOU, match_scan, operating_point}`, `anatobind.eval.matching.iou3d`, `anatobind.eval.lesion_boxes.{load_label_map, load_nnunet_probabilities}`, `anatobind.nnunet.brain_lesion.{FAMILY, gt_boxes, validation_path, validation_npz_path}`.
- Produces:
  - `anatobind/nndet/boxes.py`: `LAYOUT`, `to_corner_box(b) -> (x0, y0, z0, x1, y1, z1)`, `load_runner_json(path) -> dict`, `detections(case_rec) -> [{"box", "score", "family"}]`, `lesion_rows(dets, shape) -> [{"z0", "z1", "score", "boxes"}]`
  - `anatobind/eval/brain_detector.py`: `strata_maps(registry) -> {"band", "n_slices", "inplane_tertile", "stratum_geometry"}` (each `{lesion_id: stratum}`)
  - `anatobind/eval/brain_nndet.py`: `RULE_A_MARGIN = 0.05`, `gt_of_cases(case_kinds, registry)`, `nndet_scans(cases_json, case_ids, gt_of_case)`, `nnunet_scans(results_root, config, splits, folds, gt_of_case)`, `rule_a(rows_nndet, rows_base, n_gt, margin=RULE_A_MARGIN)`, `found_lesions(scans, thr)`, `paired_table(scans_a, thr_a, scans_b, thr_b)`, `gt_check(cases_json, instance_map, gt_of_case)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_nndet_boxes.py
import importlib.util
import json
from pathlib import Path

import pytest

from anatobind.nndet.boxes import LAYOUT, detections, lesion_rows, load_runner_json, to_corner_box


def test_corner_box_reorders_the_runner_layout():
    assert to_corner_box([1, 2, 3, 4, 5, 6]) == (5.0, 2.0, 1.0, 6.0, 4.0, 3.0)


def test_runner_and_reader_share_the_layout_string():
    path = Path(__file__).resolve().parents[1] / "scripts/nndet_runner.py"
    spec = importlib.util.spec_from_file_location("nndet_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.LAYOUT == LAYOUT


def test_reader_refuses_foreign_layout_and_other_classes(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"layout": "other", "cases": {}}))
    with pytest.raises(ValueError, match="layout"):
        load_runner_json(p)
    with pytest.raises(ValueError, match="class"):
        detections({"boxes": [[0, 0, 1, 1, 0, 1]], "scores": [0.5], "labels": [1]})
    with pytest.raises(ValueError, match="length"):
        detections({"boxes": [[0, 0, 1, 1, 0, 1]], "scores": []})
    assert detections({"boxes": [[0, 1, 2, 3, 4, 5]], "scores": [0.25]}) == [
        {"box": (4.0, 1.0, 0.0, 5.0, 3.0, 2.0), "score": 0.25, "family": "small_lesion"}]


def test_lesion_rows_round_half_up_keep_one_voxel_clip_drop_and_sort():
    dets = [{"box": (2.5, 1.4, 0.6, 4.4, 3.5, 1.2), "score": 0.4, "family": "small_lesion"},    # sub-voxel in z
            {"box": (-3.0, 10.0, 1.0, 2.0, 14.0, 3.0), "score": 0.9, "family": "small_lesion"},   # left part outside
            {"box": (30.0, 0.0, 0.0, 34.0, 2.0, 1.0), "score": 0.7, "family": "small_lesion"}]    # fully outside
    assert lesion_rows(dets, (20, 16, 4)) == [
        {"z0": 1, "z1": 2, "score": 0.9, "boxes": {"1": [[10, 14, 0, 2]], "2": [[10, 14, 0, 2]]}},
        {"z0": 1, "z1": 1, "score": 0.4, "boxes": {"1": [[1, 4, 3, 4]]}},
    ]
    assert lesion_rows([], (20, 16, 4)) == []
```

```python
# tests/test_nndet_eval.py
import nibabel as nib
import numpy as np
import pytest

from anatobind.eval.brain_detector import strata_maps
from anatobind.eval.brain_nndet import (
    gt_check, gt_of_cases, nndet_scans, nnunet_scans, paired_table, rule_a,
)
from anatobind.nnunet.brain_lesion import DATASET_NAME, TRAINER


def _reg(lid, file, x0, y0, z0, x1, y1, z1, **kw):
    return {"lesion_id": lid, "file": file, "x0": x0, "y0": y0, "z0": z0, "x1": x1, "y1": y1, "z1": z1, **kw}


def test_strata_maps_follow_the_s2_definitions():
    reg = [{"lesion_id": i, "band": b, "n_slices": n, "inplane_mm": mm, "stratum_geometry": g}
           for i, (b, n, mm, g) in enumerate([("0", 1, 1.0, "a"), ("0-2", 2, 2.0, "a"), ("2-4", 1, 3.0, "b"),
                                              (">4", 3, 4.0, "b"), ("0", 1, 5.0, "c"), ("0", 1, 6.0, "c")])]
    m = strata_maps(reg)
    assert m["inplane_tertile"] == {0: "tertile_1", 1: "tertile_1", 2: "tertile_1", 3: "tertile_2", 4: "tertile_2",
                                    5: "tertile_3"}
    assert m["n_slices"] == {0: "1", 1: ">1", 2: "1", 3: ">1", 4: "1", 5: "1"}
    assert m["band"][3] == ">4" and m["stratum_geometry"][5] == "c"


def test_gt_of_cases_uses_registry_boxes_and_refuses_a_lesion_case_without_rows():
    reg = [_reg(0, "A", 5, 5, 10, 10, 10, 10)]
    g = gt_of_cases({"A": "lesion", "N": "normal"}, reg)
    assert g == {"A": [{"lesion_id": 0, "family": "small_lesion", "box": (5, 5, 10, 10, 10, 11)}], "N": []}
    with pytest.raises(ValueError, match="B"):
        gt_of_cases({"A": "lesion", "B": "lesion"}, reg)


def test_nndet_scans_refuse_a_missing_case():
    with pytest.raises(KeyError, match="N"):
        nndet_scans({"A": {"boxes": [], "scores": []}}, ["A", "N"], {"A": [], "N": []})


def _rows(*triples):
    return [{"thr": t, "n_hit": h, "sensitivity_family": h / 280, "fp_per_scan": fp} for t, h, fp in triples]


def test_rule_a_needs_ceil_margin_times_lesions_more_hits_than_the_baseline():
    base = _rows((0.3, 99, 2.5), (0.6, 92, 1.5))                 # operating point: 92 hits at <= 2 FP
    assert rule_a(_rows((0.5, 105, 1.0)), base, 280)["pass"] is False
    r = rule_a(_rows((0.2, 130, 3.0), (0.5, 106, 1.9)), base, 280)
    assert r["pass"] is True and r["required"] == 106 and r["nndet_hits"] == 106 and r["baseline_thr"] == 0.6
    assert rule_a(_rows((0.5, 150, 2.5)), base, 280)["pass"] is False   # no nnDetection threshold at <= 2 FP


def _scan(case, lids, found, thr_score=0.9):
    gt = [{"lesion_id": l, "family": "small_lesion", "box": (10 * l, 0, 0, 10 * l + 5, 5, 1)} for l in lids]
    dets = [{"box": (10 * l, 0, 0, 10 * l + 5, 5, 1), "score": thr_score, "family": "small_lesion"} for l in found]
    return {"case": case, "gt": gt, "dets": dets}


def test_paired_table_counts_the_four_groups():
    a = [_scan("A", [0, 1, 2, 3], [0, 1])]
    b = [_scan("A", [0, 1, 2, 3], [1, 2])]
    assert paired_table(a, 0.5, b, 0.5) == {"both": 1, "only_a": 1, "only_b": 1, "neither": 1, "n_gt": 4}
    with pytest.raises(ValueError):
        paired_table(a, 0.5, [_scan("A", [0, 1], [])], 0.5)


def test_gt_check_reports_iou_lost_and_below_threshold_instances():
    gt_of_case = {"A": [{"lesion_id": 10, "family": "small_lesion", "box": (5, 6, 1, 9, 10, 2)},
                        {"lesion_id": 11, "family": "small_lesion", "box": (20, 20, 0, 24, 24, 1)},
                        {"lesion_id": 12, "family": "small_lesion", "box": (40, 40, 0, 44, 44, 1)}],
                  "N": []}
    cases = {"A": {"boxes": [[1, 6, 2, 10, 5, 9], [0, 50, 1, 54, 50, 54]], "scores": [1.0, 1.0], "instances": [1, 2]},
             "N": {"boxes": [], "scores": [], "instances": []}}
    res = gt_check(cases, {"A": {"1": 10, "2": 11, "3": 12}, "N": {}}, gt_of_case)
    assert res["iou"][10] == pytest.approx(1.0) and res["iou"][11] == 0.0
    assert res["lost"] == [12] and res["below_iou"] == [11]
    with pytest.raises(ValueError, match="instance map"):
        gt_check({"A": {"boxes": [[1, 6, 2, 10, 5, 9]], "scores": [1.0], "instances": [4]}},
                 {"A": {"1": 10, "2": 11, "3": 12}}, {"A": gt_of_case["A"]})
    with pytest.raises(ValueError, match="registry"):
        gt_check(cases, {"A": {"1": 10, "2": 11}, "N": {}}, gt_of_case)


def test_nnunet_scans_read_only_the_requested_folds(tmp_path):
    splits = [{"train": ["B"], "val": ["A"]}, {"train": ["A"], "val": ["B"]}]
    shape = (32, 32, 4)
    d = tmp_path / DATASET_NAME / f"{TRAINER}__nnUNetPlans__2d" / "fold_1" / "validation"
    d.mkdir(parents=True)
    lab = np.zeros(shape, np.uint8)
    lab[5:10, 5:10, 2] = 1
    probs = np.zeros((2,) + shape, np.float32)
    probs[1][lab == 1] = 0.8
    nib.save(nib.Nifti1Image(lab, np.eye(4)), str(d / "B.nii.gz"))
    np.savez_compressed(d / "B.npz", probabilities=np.ascontiguousarray(probs.transpose(0, 3, 2, 1)))
    gt = {"A": [], "B": [{"lesion_id": 3, "family": "small_lesion", "box": (5, 5, 2, 10, 10, 3)}]}
    scans = nnunet_scans(tmp_path, "2d", splits, [1], gt)
    assert [s["case"] for s in scans] == ["B"] and len(scans[0]["dets"]) == 1
    with pytest.raises(FileNotFoundError, match="fold 0 case A"):
        nnunet_scans(tmp_path, "2d", splits, [0], gt)
```

- [ ] **Step 2: Run to verify they fail** — `… -m pytest tests/test_nndet_boxes.py tests/test_nndet_eval.py -q -p no:cacheprovider` → ImportErrors (`anatobind.nndet.boxes`, `strata_maps`, `anatobind.eval.brain_nndet`).

- [ ] **Step 3: Implement.** `anatobind/nndet/boxes.py`:

```python
"""Read the nndet runner's JSON into the repository's boxes and lesion tables (spec 2026-09-28 brain-nndet §5, §8)."""
import json
from pathlib import Path

import numpy as np

from anatobind.nnunet.brain_lesion import FAMILY

LAYOUT = "nndet [s_lo, r_lo, s_hi, r_hi, c_lo, c_hi] on original array axes (slice, row, col), half-open"


def to_corner_box(b):
    """[s_lo, r_lo, s_hi, r_hi, c_lo, c_hi] -> (x0, y0, z0, x1, y1, z1) on the (col, row, slice) grid, half-open."""
    s0, r0, s1, r1, c0, c1 = (float(v) for v in b)
    return (c0, r0, s0, c1, r1, s1)


def load_runner_json(path):
    d = json.loads(Path(path).read_text())
    if d.get("layout") != LAYOUT:
        raise ValueError(f"{path}: layout {d.get('layout')!r} is not the runner's {LAYOUT!r}")
    return d


def detections(case_rec):
    boxes, scores = case_rec["boxes"], case_rec["scores"]
    labels = case_rec.get("labels", [0] * len(boxes))
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError(f"boxes, scores and labels differ in length ({len(boxes)}, {len(scores)}, {len(labels)})")
    if any(int(v) != 0 for v in labels):
        raise ValueError(f"only class 0 is expected, found class {sorted({int(v) for v in labels})}")
    return [{"box": to_corner_box(b), "score": float(s), "family": FAMILY} for b, s in zip(boxes, scores)]


def _half_up(v):
    return int(np.floor(v + 0.5))


def _span(lo, hi, n):
    a, b = _half_up(lo), _half_up(hi)
    b = max(b, a + 1)
    a, b = max(0, a), min(n, b)
    return (a, b) if b > a else None


def lesion_rows(dets, shape):
    """Scored 3D boxes -> S2 lesions.json rows: z0..z1 inclusive, per-slice [row0, row1, col0, col1] = the box's
    in-plane rectangle on every slice; coordinates rounded half up, at least one voxel per axis, clipped to the
    (col, row, slice) grid; a box fully outside is dropped; rows sorted by score, highest first."""
    nc, nr, ns = shape
    out = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        x0, y0, z0, x1, y1, z1 = d["box"]
        cs, rs, ss = _span(x0, x1, nc), _span(y0, y1, nr), _span(z0, z1, ns)
        if cs is None or rs is None or ss is None:
            continue
        out.append({"z0": ss[0], "z1": ss[1] - 1, "score": float(d["score"]),
                    "boxes": {str(s): [[rs[0], rs[1], cs[0], cs[1]]] for s in range(ss[0], ss[1])}})
    return out
```

Add to `anatobind/eval/brain_detector.py` (after `strata_sensitivity`, nothing else changes):

```python
def strata_maps(registry):
    """The four report-only strata of spec 2026-09-28 §4 by lesion id, as scripts/eval_brain_detector.py builds them:
    d_interface band, 1 vs >1 slices, in-plane size tertile over the registry, measured geometry stratum."""
    vals = sorted(r["inplane_mm"] for r in registry)
    t33, t67 = vals[len(vals) // 3], vals[2 * len(vals) // 3]

    def tertile(mm):
        return "tertile_1" if mm <= t33 else ("tertile_2" if mm <= t67 else "tertile_3")

    return {"band": {r["lesion_id"]: r["band"] for r in registry},
            "n_slices": {r["lesion_id"]: "1" if r["n_slices"] == 1 else ">1" for r in registry},
            "inplane_tertile": {r["lesion_id"]: tertile(r["inplane_mm"]) for r in registry},
            "stratum_geometry": {r["lesion_id"]: r["stratum_geometry"] for r in registry}}
```

`anatobind/eval/brain_nndet.py`:

```python
"""nnDetection second arm evaluation (spec 2026-09-28 brain-nndet §5, §7): runner JSON -> scans in S2's format, rule A
against nnU-Net on the same folds, the paired table, and the ground-truth-as-prediction coordinate check."""
import math

from anatobind.eval.brain_detector import scan_record
from anatobind.eval.detection_metrics import IOU, match_scan, operating_point
from anatobind.eval.lesion_boxes import load_label_map, load_nnunet_probabilities
from anatobind.eval.matching import iou3d
from anatobind.nndet.boxes import detections, to_corner_box
from anatobind.nnunet.brain_lesion import gt_boxes, validation_npz_path, validation_path

RULE_A_MARGIN = 0.05


def gt_of_cases(case_kinds, registry):
    by_file = {}
    for r in registry:
        by_file.setdefault(r["file"], []).append(r)
    out = {}
    for case, kind in case_kinds.items():
        if kind == "lesion":
            if case not in by_file:
                raise ValueError(f"lesion case {case} has no registry rows")
            out[case] = gt_boxes(by_file[case])
        else:
            out[case] = []
    return out


def nndet_scans(cases_json, case_ids, gt_of_case):
    missing = [c for c in case_ids if c not in cases_json]
    if missing:
        raise KeyError(f"runner JSON lacks {len(missing)} cases: {missing[:5]}")
    return [{"case": c, "gt": gt_of_case[c], "dets": detections(cases_json[c])} for c in case_ids]


def nnunet_scans(results_root, config, splits, folds, gt_of_case):
    scans = []
    for f in folds:
        for case in splits[f]["val"]:
            nii, npz = validation_path(results_root, config, f, case), validation_npz_path(results_root, config, f, case)
            if not nii.exists() or not npz.exists():
                raise FileNotFoundError(f"fold {f} case {case}: missing {nii if not nii.exists() else npz}")
            lab = load_label_map(nii)
            scans.append(scan_record(case, gt_of_case[case], lab, load_nnunet_probabilities(npz, lab)))
    return scans


def rule_a(rows_nndet, rows_base, n_gt, margin=RULE_A_MARGIN):
    """Spec N3: continue iff nnDetection's hits at its <= 2 FP/scan operating point reach the baseline's hits at its
    own operating point plus ceil(margin * n_gt)."""
    op_n, op_b = operating_point(rows_nndet), operating_point(rows_base)
    base_hits = op_b["n_hit"] if op_b else 0
    need = base_hits + math.ceil(margin * n_gt - 1e-9)
    hits = op_n["n_hit"] if op_n else 0
    return {"pass": bool(hits >= need), "nndet_hits": hits, "baseline_hits": base_hits, "required": need, "n_gt": n_gt,
            "margin": margin, "nndet_thr": op_n["thr"] if op_n else None, "baseline_thr": op_b["thr"] if op_b else None}


def found_lesions(scans, thr):
    out = set()
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        out |= {s["gt"][g]["lesion_id"] for g in match_scan(s["gt"], dets)}
    return out


def paired_table(scans_a, thr_a, scans_b, thr_b):
    ids = {r["lesion_id"] for s in scans_a for r in s["gt"]}
    if ids != {r["lesion_id"] for s in scans_b for r in s["gt"]}:
        raise ValueError("the two scan sets do not hold the same lesions")
    a, b = found_lesions(scans_a, thr_a), found_lesions(scans_b, thr_b)
    return {"both": len(a & b), "only_a": len(a - b), "only_b": len(b - a), "neither": len(ids - a - b),
            "n_gt": len(ids)}


def gt_check(cases_json, instance_map, gt_of_case):
    """Spec §5 check two: every instance present after preprocessing must overlap its own registry box with
    IoU >= IOU. Returns {"iou": {lesion_id: IoU}, "lost": [...], "below_iou": [...]}."""
    ious, lost, below = {}, [], []
    for case, gts in gt_of_case.items():
        rec = cases_json.get(case, {"boxes": [], "scores": [], "instances": []})
        present = {int(k): b for k, b in zip(rec["instances"], rec["boxes"])}
        lid_of = {int(k): int(v) for k, v in instance_map.get(case, {}).items()}
        extra = sorted(set(present) - set(lid_of))
        if extra:
            raise ValueError(f"{case}: instances {extra} are not in the instance map")
        reg = {r["lesion_id"]: r["box"] for r in gts}
        if sorted(reg) != sorted(lid_of.values()):
            raise ValueError(f"{case}: registry lesions {sorted(reg)} != instance map lesions {sorted(lid_of.values())}")
        for k, lid in lid_of.items():
            if k not in present:
                lost.append(lid)
                continue
            ious[lid] = float(iou3d([to_corner_box(present[k])], [reg[lid]])[0, 0])
            if ious[lid] < IOU:
                below.append(lid)
    return {"iou": ious, "lost": sorted(lost), "below_iou": sorted(below)}
```

- [ ] **Step 4: Run** `tests/test_nndet_boxes.py tests/test_nndet_eval.py tests/test_brain_detector_eval.py tests/test_brain_detector_eval_script.py` → all pass.

- [ ] **Step 5: Commit** — `git add anatobind/nndet/boxes.py anatobind/eval/brain_nndet.py anatobind/eval/brain_detector.py tests/test_nndet_boxes.py tests/test_nndet_eval.py && git commit -m "nnDetection arm: runner JSON reader, lesion rows, rule A, paired table and ground-truth coordinate check; strata maps shared with S2"`

---

### Task 5: Evaluation script `scripts/eval_brain_nndet.py`

**Files:**
- Create: `scripts/eval_brain_nndet.py`
- Test: `tests/test_nndet_eval_script.py`

**Interfaces:**
- Consumes: Task 4's functions; `anatobind.eval.brain_detector.{evaluate, normal_fp_per_scan, strata_sensitivity, strata_maps}`; `anatobind.eval.detection_metrics.operating_point`; `anatobind.level_r.registry.{load_registry, BANDS}`.
- Produces: `main(argv=None)`; module constants `CASES_JSON`, `SPLITS_JSON`, `NNUNET_RESULTS`, `INSTANCE_MAP`, `EXPECTED_N_GT = 1297`, `EXPECTED_N_SCANS = 253`, `BASELINE = "2d"`, `REPORT_ONLY = ("3d_fullres",)`. Mode `--gt-check JSON --out DIR` writes `gt_check.md`, `gt_check.json`, exits 1 when any instance is below IoU 0.1 or any false positive appears. Mode `--dets JSON [--swept JSON] --folds F.. --out DIR` writes `REPORT.md`, `froc.csv`, `output.txt`, `rule_a.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nndet_eval_script.py
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from anatobind.nndet.boxes import LAYOUT
from anatobind.nnunet.brain_lesion import DATASET_NAME, TRAINER

REG = [
    {"lesion_id": 0, "file": "A", "band": "0", "n_slices": 1, "inplane_mm": 2.5, "stratum_geometry": "g1",
     "x0": 5, "y0": 5, "x1": 10, "y1": 10, "z0": 10, "z1": 10},
    {"lesion_id": 1, "file": "A", "band": ">4", "n_slices": 2, "inplane_mm": 3.5, "stratum_geometry": "g2",
     "x0": 20, "y0": 20, "x1": 25, "y1": 25, "z0": 15, "z1": 16},
    {"lesion_id": 2, "file": "B", "band": "2-4", "n_slices": 1, "inplane_mm": 4.5, "stratum_geometry": "g1",
     "x0": 30, "y0": 30, "x1": 35, "y1": 35, "z0": 8, "z1": 8},
]
# fold 0 = {A (lesions 0, 1), N (normal)}, fold 1 = {B}
SPLITS = [{"train": ["B"], "val": ["A", "N"]}, {"train": ["A", "N"], "val": ["B"]}]
INFO = {"A": {"patient_id": "p1", "kind": "lesion"}, "B": {"patient_id": "p2", "kind": "lesion"},
        "N": {"patient_id": "p3", "kind": "normal"}}
# runner layout [s_lo, r_lo, s_hi, r_hi, c_lo, c_hi]
BOX0, BOX1, BOX2 = [10, 5, 11, 10, 5, 10], [15, 20, 17, 25, 20, 25], [8, 30, 9, 35, 30, 35]


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_brain_nndet.py"
    spec = importlib.util.spec_from_file_location("eval_brain_nndet", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runner_json(path, cases):
    path.write_text(json.dumps({"layout": LAYOUT, "source": {}, "cases": cases}))
    return path


def _nnunet_outputs(results, config):
    """fold 0 validation: A has only lesion 0 predicted (prob 0.8); N has nothing."""
    d = results / DATASET_NAME / f"{TRAINER}__nnUNetPlans__{config}" / "fold_0" / "validation"
    d.mkdir(parents=True)
    shape = (64, 64, 20)
    for case in ("A", "N"):
        lab = np.zeros(shape, np.uint8)
        probs = np.zeros((2,) + shape, np.float32)
        if case == "A":
            lab[5:10, 5:10, 10] = 1
            probs[1][5:10, 5:10, 10] = 0.8
        nib.save(nib.Nifti1Image(lab, np.eye(4)), str(d / f"{case}.nii.gz"))
        np.savez_compressed(d / f"{case}.npz", probabilities=np.ascontiguousarray(probs.transpose(0, 3, 2, 1)))


def _patched(mod, tmp_path, instance_map=None):
    (tmp_path / "cases.json").write_text(json.dumps(INFO))
    (tmp_path / "splits.json").write_text(json.dumps(SPLITS))
    (tmp_path / "instances.json").write_text(json.dumps(instance_map or {"A": {"1": 0, "2": 1}, "B": {"1": 2}, "N": {}}))
    return [patch.object(mod, "CASES_JSON", tmp_path / "cases.json"),
            patch.object(mod, "SPLITS_JSON", tmp_path / "splits.json"),
            patch.object(mod, "NNUNET_RESULTS", tmp_path / "results"),
            patch.object(mod, "INSTANCE_MAP", tmp_path / "instances.json"),
            patch.object(mod, "EXPECTED_N_GT", 3), patch.object(mod, "EXPECTED_N_SCANS", 3),
            patch.object(mod, "load_registry", return_value=REG)]


def _enter(ps):
    for p in ps:
        p.start()


def _exit(ps):
    for p in ps:
        p.stop()


def test_fold_report_rule_a_paired_table_and_refusals(tmp_path):
    mod = _load()
    _nnunet_outputs(tmp_path / "results", "2d")
    _nnunet_outputs(tmp_path / "results", "3d_fullres")
    # nnDetection: lesion 0 at 0.9, lesion 1 at 0.6, one false positive on N at 0.3
    dets = _runner_json(tmp_path / "d.json", {"A": {"boxes": [BOX0, BOX1], "scores": [0.9, 0.6], "labels": [0, 0]},
                                              "N": {"boxes": [[5, 40, 6, 45, 40, 45]], "scores": [0.3], "labels": [0]}})
    ps = _patched(mod, tmp_path)
    _enter(ps)
    try:
        out = tmp_path / "rep"
        mod.main(["--dets", str(dets), "--swept", str(dets), "--folds", "0", "--out", str(out)])
        ra = json.loads((out / "rule_a.json").read_text())
        # nnDetection: sensitivity 1.0 for every threshold <= 0.60 with 0 FP above 0.30 -> operating threshold 0.60
        # nnU-Net 2d: sensitivity 0.5 for thresholds <= 0.80 -> operating threshold 0.80, 1 hit
        # required = 1 + ceil(0.05 * 2) = 2
        assert ra == {"pass": True, "nndet_hits": 2, "baseline_hits": 1, "required": 2, "n_gt": 2, "margin": 0.05,
                      "nndet_thr": 0.6, "baseline_thr": 0.8}
        rep = (out / "REPORT.md").read_text()
        assert "NOT the D1 gate" in rep
        assert "| both | 1 |" in rep and "| only nnDetection | 1 |" in rep and "| only nnU-Net 2d | 0 |" in rep
        assert "| neither | 0 |" in rep
        assert "NOT_GATE" in rep and "3d_fullres" in rep
        assert "| >4 | 1 | 1 |" in rep                                   # band stratum of lesion 1
        assert "Normal FP per volume at 0.60: 0.0000" in (out / "output.txt").read_text()
        assert (out / "froc.csv").read_text().splitlines()[0] == "thr,n_hit,sensitivity,fp_per_scan"
        with pytest.raises(FileExistsError):
            mod.main(["--dets", str(dets), "--folds", "0", "--out", str(out)])
        short = _runner_json(tmp_path / "short.json", {"A": {"boxes": [], "scores": []}})
        with pytest.raises(KeyError, match="N"):
            mod.main(["--dets", str(short), "--folds", "0", "--out", str(tmp_path / "rep2")])
    finally:
        _exit(ps)


def test_gt_check_passes_on_exact_boxes_and_fails_on_a_displaced_one(tmp_path):
    mod = _load()
    good = _runner_json(tmp_path / "gt.json", {
        "A": {"boxes": [BOX0, BOX1], "scores": [1.0, 1.0], "instances": [1, 2]},
        "B": {"boxes": [BOX2], "scores": [1.0], "instances": [1]},
        "N": {"boxes": [], "scores": [], "instances": []}})
    ps = _patched(mod, tmp_path)
    _enter(ps)
    try:
        mod.main(["--gt-check", str(good), "--out", str(tmp_path / "ok")])
        res = json.loads((tmp_path / "ok" / "gt_check.json").read_text())
        assert res["pass"] is True and res["lost"] == [] and res["below_iou"] == [] and res["n_fp"] == 0
        assert res["n_present"] == 3 and res["min_iou"] == pytest.approx(1.0)
        bad = _runner_json(tmp_path / "bad.json", {
            "A": {"boxes": [BOX0, BOX1], "scores": [1.0, 1.0], "instances": [1, 2]},
            "B": {"boxes": [[0, 50, 1, 55, 50, 55]], "scores": [1.0], "instances": [1]},
            "N": {"boxes": [], "scores": [], "instances": []}})
        with pytest.raises(SystemExit):
            mod.main(["--gt-check", str(bad), "--out", str(tmp_path / "no")])
        assert json.loads((tmp_path / "no" / "gt_check.json").read_text())["below_iou"] == [2]
    finally:
        _exit(ps)
```

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_nndet_eval_script.py -q -p no:cacheprovider` → FileNotFoundError on the script path.

- [ ] **Step 3: Implement** `scripts/eval_brain_nndet.py`:

```python
#!/usr/bin/env python
# scripts/eval_brain_nndet.py
"""nnDetection second arm: coordinate check or fold evaluation (spec 2026-09-28 brain-nndet §5, §7).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_nndet.py --gt-check GT.json --out DIR
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_nndet.py \
      --dets fold0_default.json --swept fold0_swept.json --folds 0 --out DIR
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.brain_detector import evaluate, normal_fp_per_scan, strata_maps, strata_sensitivity  # noqa: E402
from anatobind.eval.brain_nndet import (  # noqa: E402
    gt_check, gt_of_cases, nndet_scans, nnunet_scans, paired_table, rule_a,
)
from anatobind.eval.detection_metrics import operating_point  # noqa: E402
from anatobind.level_r.registry import BANDS, load_registry  # noqa: E402
from anatobind.nndet.boxes import load_runner_json  # noqa: E402
from anatobind.nndet.brain_task import TASK_NAME  # noqa: E402
from anatobind.nnunet.brain_lesion import DATASET_NAME  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
CASES_JSON = FM / "derived/nnunet/raw" / DATASET_NAME / "cases.json"
SPLITS_JSON = FM / "derived/nnunet/preprocessed" / DATASET_NAME / "splits_final.json"
NNUNET_RESULTS = FM / "derived/nnunet/results"
INSTANCE_MAP = FM / "derived/nndet" / TASK_NAME / "instances.json"
EXPECTED_N_GT = 1297
EXPECTED_N_SCANS = 253
BASELINE = "2d"
REPORT_ONLY = ("3d_fullres",)
STRATA_ORDER = {"band": list(BANDS), "n_slices": ["1", ">1"], "inplane_tertile": ["tertile_1", "tertile_2", "tertile_3"]}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_gt_check(a, gt_of_case, info):
    cases = load_runner_json(a.gt_check)["cases"]
    res = gt_check(cases, json.loads(Path(INSTANCE_MAP).read_text()), gt_of_case)
    scans = nndet_scans(cases, sorted(info), gt_of_case)
    row = evaluate(scans, {c for c, v in info.items() if v["kind"] == "normal"})["rows"][0]
    ious = sorted(res["iou"].values())
    summary = {"pass": not res["below_iou"] and row["n_fp"] == 0, "n_gt": row["n_gt"], "n_present": len(ious),
               "lost": res["lost"], "below_iou": res["below_iou"], "min_iou": ious[0] if ious else None,
               "median_iou": ious[len(ious) // 2] if ious else None, "n_iou_ge_0_99": sum(i >= 0.99 for i in ious),
               "n_hit_at_0_05": row["n_hit"], "n_fp": row["n_fp"], "input": str(a.gt_check),
               "input_sha256": sha256(a.gt_check)}
    a.out.mkdir(parents=True)
    (a.out / "gt_check.json").write_text(json.dumps(summary, indent=1))
    lines = ["# nnDetection coordinate check: ground truth as prediction (spec §5 check two)\n\n",
             f"Pass: {summary['pass']}\n\n", "```json\n", json.dumps(summary, indent=1), "\n```\n\n",
             "## Command\n\n```\n", " ".join(sys.argv), "\n```\n"]
    (a.out / "gt_check.md").write_text("".join(lines))
    print(json.dumps(summary))
    if not summary["pass"]:
        raise SystemExit(1)


def _gate_block(title, ev, normal):
    g, thr = ev["gate"], ev["gate"]["thr"]
    nfp = normal_fp_per_scan(ev["scans"], normal, thr) if thr is not None else None
    return [f"## {title}\n\n```json\n", json.dumps(g, indent=1), "\n```\n\n",
            f"Normal-volume FP per volume at the operating threshold: {'n/a' if nfp is None else f'{nfp:.4f}'}\n\n"]


def run_folds(a, gt_of_case, info, registry):
    splits = json.loads(Path(SPLITS_JSON).read_text())
    cases = [c for f in a.folds for c in splits[f]["val"]]
    normal = {c for c, v in info.items() if v["kind"] == "normal"}

    def ev_of(scans):
        ev = evaluate(scans, normal)
        ev["scans"] = scans
        return ev

    nd = ev_of(nndet_scans(load_runner_json(a.dets)["cases"], cases, gt_of_case))
    base = ev_of(nnunet_scans(NNUNET_RESULTS, BASELINE, splits, a.folds, gt_of_case))
    others = {cfg: ev_of(nnunet_scans(NNUNET_RESULTS, cfg, splits, a.folds, gt_of_case)) for cfg in REPORT_ONLY}
    swept = ev_of(nndet_scans(load_runner_json(a.swept)["cases"], cases, gt_of_case)) if a.swept else None
    ra = rule_a(nd["rows"], base["rows"], nd["n_gt"])
    op_n, op_b = operating_point(nd["rows"]), operating_point(base["rows"])
    paired = paired_table(nd["scans"], op_n["thr"], base["scans"], op_b["thr"]) if op_n and op_b else None
    five = sorted(a.folds) == [0, 1, 2, 3, 4]

    a.out.mkdir(parents=True)
    L = [f"# nnDetection second arm, folds {a.folds}\n\n",
         ("Five folds: the nnDetection gate line below is the D1 gate.\n\n" if five else
          "Fold subset: these numbers decide rule A only; they are NOT the D1 gate.\n\n"),
         f"Inputs: `{a.dets}` sha256 {sha256(a.dets)}" + (f"; swept `{a.swept}` sha256 {sha256(a.swept)}" if a.swept else "") + "\n\n",
         "## Rule A (spec N3)\n\n```json\n", json.dumps(ra, indent=1), "\n```\n\n"]
    L += _gate_block("nnDetection, default postprocessing", nd, normal)
    L += ["### FROC\n\n| thr | n_hit | sensitivity | FP per volume |\n|---|---|---|---|\n"]
    L += [f"| {r['thr']:.2f} | {r['n_hit']} | {r['sensitivity']:.4f} | {r['fp_per_scan']:.4f} |\n" for r in nd["rows"]]
    L += ["\n"]
    L += _gate_block(f"nnU-Net {BASELINE}, same folds (rule A baseline)", base, normal)
    for cfg, ev in others.items():
        L += _gate_block(f"nnU-Net {cfg}, same folds (report only)", ev, normal)
    L += ["## Paired table (each model at its own operating point, not gated)\n\n"]
    if paired:
        L += ["| group | lesions |\n|---|---|\n", f"| both | {paired['both']} |\n",
              f"| only nnDetection | {paired['only_a']} |\n", f"| only nnU-Net {BASELINE} | {paired['only_b']} |\n",
              f"| neither | {paired['neither']} |\n\n"]
    else:
        L += ["No operating point for one of the two models.\n\n"]
    if op_n:
        maps = strata_maps(registry)
        L += ["## Strata (nnDetection at its operating point, not gated)\n\n"]
        for name, m in maps.items():
            st = strata_sensitivity(nd["scans"], op_n["thr"], m)
            keys = [k for k in STRATA_ORDER.get(name, sorted(st)) if k in st]
            L += [f"### {name}\n\n| stratum | N (GT) | N (hit) | sensitivity |\n|---|---|---|---|\n"]
            L += [f"| {k} | {st[k]['n_gt']} | {st[k]['n_hit']} | {st[k]['sensitivity']:.4f} |\n" for k in keys]
            L += ["\n"]
    if swept:
        L += _gate_block("Swept postprocessing (NOT_GATE: parameters tuned on these validation cases)", swept, normal)
    L += ["## Command\n\n```\n", " ".join(sys.argv), "\n```\n"]
    (a.out / "REPORT.md").write_text("".join(L))
    (a.out / "rule_a.json").write_text(json.dumps(ra, indent=1))
    (a.out / "froc.csv").write_text("thr,n_hit,sensitivity,fp_per_scan\n" + "".join(
        f"{r['thr']:.2f},{r['n_hit']},{r['sensitivity']:.6f},{r['fp_per_scan']:.6f}\n" for r in nd["rows"]))
    out = [f"Folds: {a.folds}; scans {nd['n_scans']}; lesions {nd['n_gt']}\n", f"Rule A pass: {ra['pass']}\n"]
    if op_n:
        out.append(f"nnDetection operating threshold {op_n['thr']:.2f}: hits {op_n['n_hit']}, "
                   f"sensitivity {op_n['sensitivity']:.4f}, FP per volume {op_n['fp_per_scan']:.4f}\n")
        out.append(f"Normal FP per volume at {op_n['thr']:.2f}: {normal_fp_per_scan(nd['scans'], normal, op_n['thr']):.4f}\n")
    if op_b:
        out.append(f"nnU-Net {BASELINE} operating threshold {op_b['thr']:.2f}: hits {op_b['n_hit']}, "
                   f"sensitivity {op_b['sensitivity']:.4f}, FP per volume {op_b['fp_per_scan']:.4f}\n")
    (a.out / "output.txt").write_text("".join(out))
    print("".join(out), end="")


def main(argv=None):
    ap = argparse.ArgumentParser(description="nnDetection second arm: coordinate check or fold evaluation")
    ap.add_argument("--out", type=Path, required=True, help="output directory (must not exist)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gt-check", type=Path, help="runner gt JSON")
    mode.add_argument("--dets", type=Path, help="runner extract JSON, default parameters (the only input rule A reads)")
    ap.add_argument("--swept", type=Path, help="runner extract JSON, swept parameters (NOT_GATE)")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"--out {a.out} already exists")
    info = json.loads(Path(CASES_JSON).read_text())
    registry = load_registry()
    gt_of_case = gt_of_cases({c: v["kind"] for c, v in info.items()}, registry)
    n_gt = sum(len(v) for v in gt_of_case.values())
    if n_gt != EXPECTED_N_GT or len(info) != EXPECTED_N_SCANS:
        raise AssertionError(f"expected {EXPECTED_N_GT} lesions in {EXPECTED_N_SCANS} scans, got {n_gt} in {len(info)}")
    if a.gt_check:
        run_gt_check(a, gt_of_case, info)
    else:
        run_folds(a, gt_of_case, info, registry)


if __name__ == "__main__":
    main()
```

Note: `evaluate` (S2) returns rows whose dicts carry `n_hit`, `sensitivity`, `sensitivity_family`, `fp_per_scan`, `n_fp`; `ev["scans"]` is added here only for the report.

- [ ] **Step 4: Run** `tests/test_nndet_eval_script.py tests/test_nndet_eval.py` → all pass. If a hand-computed number disagrees, recheck the fixture comments against `detection_metrics.operating_point` (ties go to the higher threshold) before touching the code; do not loosen an assertion.

- [ ] **Step 5: Commit** — `git add scripts/eval_brain_nndet.py tests/test_nndet_eval_script.py && git commit -m "nnDetection arm evaluation script: ground-truth coordinate check and fold report with rule A, baselines, paired table and strata"`

---

### Task 6: Build Task903, preprocess, splits, ground-truth coordinate check (real data)

**Files:**
- Create: `scripts/nndet_prepare.py`
- Create: `docs/verification/2026-09-29/brain_nndet/task_build.txt`, `plan.txt`, `gt_check/` (from the runs)

**Interfaces:**
- Consumes: Task 2's builder; `anatobind.level_r.export.{match_registry, merged_lesions, small_lesion_rows}`, `anatobind.data_engine.fastmri.read_fastmri_plus_rows`, `anatobind.level_r.registry.load_registry`; Task 3's runner `gt`; Task 5's `--gt-check`.
- Produces: `${det_data}/Task903_FastMRIBrainSmallLesion/{dataset.json, raw_splitted/, instances.json, build_report.json}`; `preprocessed/{D3V001_3d.pkl, D3V001_3d/imagesTr, splits_final.pkl}`; `/data2/congcong/data/FM_data/derived/nndet_runs/gt.json`.

- [ ] **Step 1: Write** `scripts/nndet_prepare.py`:

```python
#!/usr/bin/env python
# scripts/nndet_prepare.py
"""Build nnDetection's Task903 from Dataset903 (spec 2026-09-28 brain-nndet §4), or write its splits.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/nndet_prepare.py --stage task
  bash -c 'source scripts/nndet_env.sh && nice -n 19 nndet_prep 903 -np 4 -npp 4 --full_check'
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/nndet_prepare.py --stage splits
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import read_fastmri_plus_rows  # noqa: E402
from anatobind.level_r.export import match_registry, merged_lesions, small_lesion_rows  # noqa: E402
from anatobind.level_r.registry import load_registry  # noqa: E402
from anatobind.nndet.brain_task import (  # noqa: E402
    TASK_NAME, build_task, changed_boxes, check_same_support, instances_json, paint_instances, write_instance_label,
    write_splits_pkl,
)
from anatobind.nnunet.brain_lesion import DATASET_NAME  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
NNUNET_RAW = FM / "derived/nnunet/raw" / DATASET_NAME
NNUNET_SPLITS = FM / "derived/nnunet/preprocessed" / DATASET_NAME / "splits_final.json"
DET_DATA = FM / "derived/nndet"
N_LESIONS = 1297


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_task():
    info = json.loads((NNUNET_RAW / "cases.json").read_text())
    reg_by_file, csv_by_file = {}, {}
    for r in load_registry():
        reg_by_file.setdefault(r["file"], []).append(r)
    for r in small_lesion_rows(read_fastmri_plus_rows(CSV)):
        csv_by_file.setdefault(r["file"], []).append(r)
    instance_map, changed = {}, {}

    def write_case(case, img_out, labels_dir):
        src = NNUNET_RAW / "imagesTr" / f"{case}_0000.nii.gz"
        shutil.copyfile(src, img_out)
        if sha256(src) != sha256(img_out):
            raise ValueError(f"{case}: the copied image differs from {src}")
        binary = np.asarray(nib.load(str(NNUNET_RAW / "labelsTr" / f"{case}.nii.gz")).dataobj)
        shape = binary.shape                                   # (col, row, slice); shape[1] = RSS rows
        if info[case]["kind"] == "lesion":
            matched = match_registry(reg_by_file[case], merged_lesions(csv_by_file.get(case, []), shape[1]))
            members_of = {lid: L["members"] for lid, L in matched.items()}
            inst, instance_of = paint_instances(members_of, shape)
            ch = changed_boxes(members_of, inst, instance_of)
            if ch:
                changed[case] = ch
        else:
            inst, instance_of = np.zeros(shape, np.uint16), {}
        check_same_support(inst, binary, case)
        write_instance_label(inst, img_out, labels_dir / f"{case}.nii.gz")
        (labels_dir / f"{case}.json").write_text(json.dumps(instances_json(instance_of)))
        instance_map[case] = {str(k): int(lid) for lid, k in instance_of.items()}
        print(f"{len(instance_map)}/{len(info)} {case} {info[case]['kind']} {len(instance_of)} instances", flush=True)

    base = build_task(DET_DATA, sorted(info), write_case)
    n = sum(len(v) for v in instance_map.values())
    (base / "instances.json").write_text(json.dumps(instance_map, indent=1))
    (base / "build_report.json").write_text(json.dumps({"n_cases": len(info), "n_instances": n,
                                                        "changed_boxes": changed}, indent=1))
    if n != N_LESIONS:
        raise AssertionError(f"expected {N_LESIONS} instances, got {n}")
    print(f"wrote {base}: {len(info)} cases, {n} instances; lesions whose box changed by overwrite: {changed}")


def stage_splits():
    prep = DET_DATA / TASK_NAME / "preprocessed"
    if not prep.is_dir():
        raise FileNotFoundError(f"{prep} missing: run nndet_prep 903 first")
    splits = json.loads(NNUNET_SPLITS.read_text())
    info = json.loads((NNUNET_RAW / "cases.json").read_text())
    val = sorted(c for s in splits for c in s["val"])
    if len(splits) != 5 or val != sorted(info) or len(val) != 253:
        raise AssertionError("Dataset903 splits do not cover the 253 cases exactly once")
    p = write_splits_pkl(prep, splits)
    print(f"wrote {p}: validation cases per fold {[len(s['val']) for s in splits]}")


def main():
    ap = argparse.ArgumentParser(description="Build nnDetection Task903 or write its splits")
    ap.add_argument("--stage", choices=("task", "splits"), required=True)
    a = ap.parse_args()
    stage_task() if a.stage == "task" else stage_splits()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build the task** (≈ 10 min; background if needed):

```bash
mkdir -p docs/verification/2026-09-29/brain_nndet
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/nndet_prepare.py --stage task 2>&1 | tee docs/verification/2026-09-29/brain_nndet/task_build.txt
```

Expected last line: `wrote …/Task903_FastMRIBrainSmallLesion: 253 cases, 1297 instances; lesions whose box changed by overwrite: {…}` (the probe found lesion 815 only). Any exception: stop and report; the partial task directory stays (list it for the user; never delete it).

- [ ] **Step 3: Preprocess** (nndet env, CPU, background; poll the log):

```bash
bash -c 'source scripts/nndet_env.sh && nice -n 19 nndet_prep 903 -np 4 -npp 4 --full_check' > logs/nndet_install/04_prep903.log 2>&1
bash -c 'source scripts/nndet_env.sh && python -c "
import pickle; p = pickle.load(open(\"/data2/congcong/data/FM_data/derived/nndet/Task903_FastMRIBrainSmallLesion/preprocessed/D3V001_3d.pkl\", \"rb\"))
print(\"data_identifier\", p[\"data_identifier\"]); print(\"target_spacing\", p[\"target_spacing\"]); print(\"transpose_forward\", p[\"transpose_forward\"], \"transpose_backward\", p[\"transpose_backward\"]); print(\"patch_size\", p[\"patch_size\"], \"batch_size\", p[\"batch_size\"])
"' | tee docs/verification/2026-09-29/brain_nndet/plan.txt
ls /data2/congcong/data/FM_data/derived/nndet/Task903_FastMRIBrainSmallLesion/preprocessed | tee -a docs/verification/2026-09-29/brain_nndet/plan.txt
```

Expected: the log ends without a traceback; `D3V001_3d.pkl` and `D3V001_3d/` exist (a low-resolution plan, if listed, is recorded and unused).

- [ ] **Step 4: Splits and unpack:**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/nndet_prepare.py --stage splits | tee -a docs/verification/2026-09-29/brain_nndet/plan.txt
bash -c 'source scripts/nndet_env.sh && nice -n 19 nndet_unpack "$det_data/Task903_FastMRIBrainSmallLesion/preprocessed/D3V001_3d/imagesTr" 8' > logs/nndet_install/05_unpack903.log 2>&1
```

Expected: `validation cases per fold [51, 51, 51, 50, 50]`.

- [ ] **Step 5: Ground-truth coordinate check** (spec §5 check two; training must not start unless it passes):

```bash
bash -c 'source scripts/nndet_env.sh && nice -n 19 python scripts/nndet_runner.py gt --prep "$det_data/Task903_FastMRIBrainSmallLesion/preprocessed" --out /data2/congcong/data/FM_data/derived/nndet_runs/gt.json'
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_nndet.py --gt-check /data2/congcong/data/FM_data/derived/nndet_runs/gt.json --out docs/verification/2026-09-29/brain_nndet/gt_check
```

Expected: exit 0, `"pass": true`, `"n_fp": 0`, `"below_iou": []`; `lost` lists the lesions whose instance vanished in resampling (record them; they stay in the denominator). If it exits 1: stop, report the JSON, do not train.

- [ ] **Step 6: Commit** — `git add scripts/nndet_prepare.py docs/verification/2026-09-29/brain_nndet && git commit -m "nnDetection Task903: build from Dataset903, preprocessing plan, splits and ground-truth coordinate check on real data"`

---

### Task 7: Launcher `scripts/nndet_train.py` and the fold-0 launch

**Files:**
- Create: `scripts/nndet_train.py`
- Test: `tests/test_nndet_train.py`
- Create: `docs/verification/2026-09-29/brain_nndet/launch.md`

**Interfaces:**
- Consumes: `scripts/brain_detector_train.py::{idle_gpus, query_nvidia_smi, query_busy_pids}`; Task 2's `TASK_ID, TASK_NAME, PLAN_ID, MODEL_ID, read_splits_pkl, splits_payload`.
- Produces: `train_dir(det_models, fold) -> Path`, `preflight(det_data, det_models, fold, nnunet_splits, log_dir) -> Path (log)`, `command(fold, gpu, log, env_sh) -> list[str]`, `main(argv=None)`; constants `DET_DATA`, `DET_MODELS`, `NNUNET_SPLITS`, `LOG_DIR`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nndet_train.py
import importlib.util
import json
import pickle
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("nndet_train", REPO / "scripts/nndet_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_paths_match_the_env_file():
    mod = _load()
    text = (REPO / "scripts/nndet_env.sh").read_text()
    assert f"det_data={mod.DET_DATA}\n" in text and f"det_models={mod.DET_MODELS}\n" in text
    assert mod.train_dir(Path("/m"), 0) == Path("/m/Task903_FastMRIBrainSmallLesion/RetinaUNetV001_D3V001_3d/fold0")


def _prep(root, splits):
    prep = root / "Task903_FastMRIBrainSmallLesion" / "preprocessed"
    (prep / "D3V001_3d" / "imagesTr").mkdir(parents=True)
    (prep / "splits_final.pkl").write_bytes(pickle.dumps(splits, protocol=4))


def test_preflight_refusals(tmp_path):
    mod = _load()
    models, logs = tmp_path / "models", tmp_path / "logs"
    sj = [{"train": ["b"], "val": ["a"]}, {"train": ["a"], "val": ["b"]}]
    (tmp_path / "splits.json").write_text(json.dumps(sj))
    with pytest.raises(FileNotFoundError, match="nndet_prep"):
        mod.preflight(tmp_path / "empty", models, 0, tmp_path / "splits.json", logs)
    _prep(tmp_path / "auto", [{"train": ["a"], "val": ["b"]}])         # e.g. nnDetection's own KFold split
    with pytest.raises(ValueError, match="differs"):
        mod.preflight(tmp_path / "auto", models, 0, tmp_path / "splits.json", logs)
    data = tmp_path / "data"
    _prep(data, sj)
    assert mod.preflight(data, models, 0, tmp_path / "splits.json", logs) == logs / "fold0.log"
    mod.train_dir(models, 0).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        mod.preflight(data, models, 0, tmp_path / "splits.json", logs)
    logs.mkdir()
    (logs / "fold1.log").write_text("")
    with pytest.raises(FileExistsError, match="fold1.log"):
        mod.preflight(data, models, 1, tmp_path / "splits.json", logs)


def test_command_pins_the_gpu_and_quotes_paths():
    cmd = _load().command(0, 3, Path("/tmp/a b/fold0.log"), Path("/r/scripts/nndet_env.sh"))
    assert cmd[:3] == ["setsid", "bash", "-c"]
    assert "source /r/scripts/nndet_env.sh && CUDA_VISIBLE_DEVICES=3 nice -n 19 nndet_train 903 -o exp.fold=0 --sweep" in cmd[3]
    assert cmd[3].endswith("> '/tmp/a b/fold0.log' 2>&1")
```

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_nndet_train.py -q -p no:cacheprovider` → FileNotFoundError on the script path.

- [ ] **Step 3: Implement** `scripts/nndet_train.py`:

```python
#!/usr/bin/env python
# scripts/nndet_train.py
"""Launch nnDetection training (with its sweep) of one Task903 fold on one idle GPU (spec 2026-09-28 brain-nndet §6).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/nndet_train.py --fold 0 --gpus 3 6 7
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain_detector_train import idle_gpus, query_busy_pids, query_nvidia_smi  # noqa: E402
from anatobind.nndet.brain_task import MODEL_ID, PLAN_ID, TASK_ID, TASK_NAME, read_splits_pkl, splits_payload  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
DET_DATA = FM / "derived/nndet"
DET_MODELS = FM / "derived/nndet_models"
NNUNET_SPLITS = FM / "derived/nnunet/preprocessed/Dataset903_FastMRIBrainSmallLesion/splits_final.json"
REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "logs/brain_nndet"


def train_dir(det_models, fold):
    return Path(det_models) / TASK_NAME / MODEL_ID / f"fold{fold}"


def preflight(det_data, det_models, fold, nnunet_splits, log_dir):
    """Refuse unless the preprocessed plan data exist, splits_final.pkl equals Dataset903's splits (nnDetection makes
    its own KFold split when the file is missing), and neither the training dir nor the log exists yet."""
    prep = Path(det_data) / TASK_NAME / "preprocessed"
    if not (prep / PLAN_ID / "imagesTr").is_dir():
        raise FileNotFoundError(f"{prep / PLAN_ID / 'imagesTr'} missing: run nndet_prep and nndet_unpack first")
    if read_splits_pkl(prep / "splits_final.pkl") != splits_payload(json.loads(Path(nnunet_splits).read_text())):
        raise ValueError(f"{prep / 'splits_final.pkl'} differs from Dataset903's splits")
    td = train_dir(det_models, fold)
    if td.exists():
        raise FileExistsError(f"{td} exists; nnDetection's overwrite mode would reuse it")
    log = Path(log_dir) / f"fold{fold}.log"
    if log.exists():
        raise FileExistsError(f"{log} exists")
    return log


def command(fold, gpu, log, env_sh):
    inner = (f"source {shlex.quote(str(env_sh))} && CUDA_VISIBLE_DEVICES={int(gpu)} nice -n 19 "
             f"nndet_train {TASK_ID} -o exp.fold={int(fold)} --sweep > {shlex.quote(str(log))} 2>&1")
    return ["setsid", "bash", "-c", inner]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Launch nnDetection training of one Task903 fold")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpus", type=int, nargs="+", required=True, help="candidate GPUs; the first idle one is used")
    a = ap.parse_args(argv)
    log = preflight(DET_DATA, DET_MODELS, a.fold, NNUNET_SPLITS, LOG_DIR)
    idle = idle_gpus(query_nvidia_smi(), a.gpus, query_busy_pids())
    if not idle:
        raise RuntimeError(f"none of GPUs {a.gpus} is idle")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(command(a.fold, idle[0], log, REPO / "scripts/nndet_env.sh"), cwd=str(REPO),
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"launched fold {a.fold} on GPU {idle[0]} (pid {proc.pid}); log {log}; "
          f"training dir {train_dir(DET_MODELS, a.fold)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** `tests/test_nndet_train.py tests/test_brain_detector_train.py` → all pass.

- [ ] **Step 5: Commit** — `git add scripts/nndet_train.py tests/test_nndet_train.py && git commit -m "nnDetection launcher: preflight on plan data, Dataset903 splits and fresh training dir; one idle GPU per fold"`

- [ ] **Step 6: Launch fold 0.** Run `nvidia-smi` first and pass the idle cards as candidates (at 2026-09-28 20:45 GPUs 3, 6, 7 were idle):

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/nndet_train.py --fold 0 --gpus 3 6 7
```

- [ ] **Step 7: Timing projection (N9).** After about 20 minutes:

```bash
tail -c 2000 logs/brain_nndet/fold0.log | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, *[0-9.]+(it/s|s/it)\]' | tail -3
```

Projection: with `R` it/s, one epoch ≈ `2500 / R` s (+ 100 validation batches); total ≈ 60 epochs × epoch time + the sweep, where the sweep is estimated as Task 1's toy sweep time × (51 / number of toy validation cases). Write `docs/verification/2026-09-29/brain_nndet/launch.md` with the nvidia-smi snapshot, the launcher line, the rate lines and the projection arithmetic. If the projection exceeds 24 h, the controller tells the user and asks before anything else (training keeps running; resuming later is `nndet_train 903 -o exp.fold=0 train.mode=resume --sweep`).

- [ ] **Step 8: Commit** — `git add docs/verification/2026-09-29/brain_nndet/launch.md && git commit -m "nnDetection fold 0 launched: GPU, rate and wall-clock projection"`

---

### Task 8: Inference entry for S5 (code and unit tests; the real run is in Task 9)

**Files:**
- Create: `anatobind/infer/brain_nndet.py`, `scripts/infer_brain_lesions_nndet.py`
- Test: `tests/test_nndet_infer.py`

**Interfaces:**
- Consumes: `anatobind.data_engine.fastmri.rss_h5_to_nifti`; Task 4's `detections`, `lesion_rows`, `load_runner_json`; Task 2's `TASK_NAME`, `MODEL_ID`; the runner's `predict`.
- Produces: `run_runner(image, train_dir, work, out_json, gpu)`, `run(h5_path, out_dir, train_dir, gpu, predict=run_runner) -> rows` writing `<out>/lesions.json` (S2's row format).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nndet_infer.py
import json

import nibabel as nib
import numpy as np
import pytest

import anatobind.infer.brain_nndet as B
from anatobind.nndet.boxes import LAYOUT


def _fake_rss(h5, out, pad_to_slices):
    nib.save(nib.Nifti1Image(np.zeros((16, 12, 3), np.float32), np.eye(4)), str(out))


def _predict_with(cases):
    def fake(image, train_dir, work, out_json, gpu):
        out_json.write_text(json.dumps({"layout": LAYOUT, "source": {}, "cases": cases}))
    return fake


def test_run_writes_rows_from_runner_json_and_refuses_an_existing_out(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "rss_h5_to_nifti", _fake_rss)
    fake = _predict_with({"case": {"boxes": [[1, 2, 2, 5, 3, 6]], "scores": [0.7], "labels": [0]}})
    rows = B.run(tmp_path / "x.h5", tmp_path / "out", tmp_path / "train", 0, predict=fake)
    assert rows == [{"z0": 1, "z1": 1, "score": 0.7, "boxes": {"1": [[2, 5, 3, 6]]}}]
    assert json.loads((tmp_path / "out" / "lesions.json").read_text()) == rows
    with pytest.raises(FileExistsError):
        B.run(tmp_path / "x.h5", tmp_path / "out", tmp_path / "train", 0, predict=fake)


def test_run_handles_no_detections_and_refuses_other_case_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "rss_h5_to_nifti", _fake_rss)
    assert B.run(tmp_path / "x.h5", tmp_path / "o1", tmp_path / "t", 0,
                 predict=_predict_with({"case": {"boxes": [], "scores": [], "labels": []}})) == []
    with pytest.raises(ValueError, match="expected"):
        B.run(tmp_path / "x.h5", tmp_path / "o2", tmp_path / "t", 0,
              predict=_predict_with({"other": {"boxes": [], "scores": []}}))


def test_run_runner_pins_the_gpu_and_calls_predict(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(B.subprocess, "run", lambda cmd, check: calls.append(cmd))
    B.run_runner(tmp_path / "a b.nii.gz", tmp_path / "t", tmp_path / "w", tmp_path / "p.json", 5)
    cmd = calls[0]
    assert cmd[:2] == ["bash", "-c"]
    assert "CUDA_VISIBLE_DEVICES=5 nice -n 19 python" in cmd[2] and "nndet_runner.py predict" in cmd[2]
    assert f"--image '{tmp_path / 'a b.nii.gz'}'" in cmd[2]
```

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_nndet_infer.py -q -p no:cacheprovider` → `ModuleNotFoundError: anatobind.infer.brain_nndet`.

- [ ] **Step 3: Implement** `anatobind/infer/brain_nndet.py`:

```python
"""Brain small-lesion inference with the nnDetection second arm, for S5 (spec 2026-09-28 brain-nndet §8): a fastMRI
FLAIR h5 -> RSS NIfTI -> the nndet runner (model_last, default postprocessing) -> S2-format lesion rows."""
import json
import shlex
import subprocess
from pathlib import Path

import nibabel as nib

from anatobind.data_engine.fastmri import rss_h5_to_nifti
from anatobind.nndet.boxes import detections, lesion_rows, load_runner_json

REPO = Path(__file__).resolve().parents[2]


def run_runner(image, train_dir, work, out_json, gpu):
    q = lambda p: shlex.quote(str(p))  # noqa: E731
    inner = (f"source {q(REPO / 'scripts/nndet_env.sh')} && CUDA_VISIBLE_DEVICES={int(gpu)} nice -n 19 python "
             f"{q(REPO / 'scripts/nndet_runner.py')} predict --image {q(image)} --train-dir {q(train_dir)} "
             f"--work {q(work)} --out {q(out_json)}")
    subprocess.run(["bash", "-c", inner], check=True)


def run(h5_path, out_dir, train_dir, gpu, predict=run_runner):
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    (out / "input").mkdir(parents=True)
    img = out / "input" / "case_0000.nii.gz"
    rss_h5_to_nifti(h5_path, img, pad_to_slices=0)
    predict(img, train_dir, out / "work", out / "pred.json", gpu)
    cases = load_runner_json(out / "pred.json")["cases"]
    if set(cases) != {"case"}:
        raise ValueError(f"runner returned cases {sorted(cases)}, expected ['case']")
    rows = lesion_rows(detections(cases["case"]), nib.load(str(img)).shape)
    (out / "lesions.json").write_text(json.dumps(rows, indent=1))
    return rows
```

`scripts/infer_brain_lesions_nndet.py`:

```python
#!/usr/bin/env python
# scripts/infer_brain_lesions_nndet.py
"""Brain small-lesion inference with the nnDetection second arm (spec 2026-09-28 brain-nndet §8).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_lesions_nndet.py \
      --h5 FILE.h5 --out DIR --fold 0 --gpu 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.infer.brain_nndet import run  # noqa: E402
from anatobind.nndet.brain_task import MODEL_ID, TASK_NAME  # noqa: E402

DET_MODELS = Path("/data2/congcong/data/FM_data/derived/nndet_models")


def main():
    ap = argparse.ArgumentParser(description="Brain small-lesion inference, nnDetection arm")
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory (must not exist)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    train_dir = DET_MODELS / TASK_NAME / MODEL_ID / f"fold{a.fold}"
    if not (train_dir / "model_last.ckpt").exists():
        raise FileNotFoundError(f"{train_dir / 'model_last.ckpt'} missing")
    rows = run(a.h5, a.out, train_dir, a.gpu)
    print(f"{len(rows)} lesions -> {a.out / 'lesions.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** `tests/test_nndet_infer.py tests/test_brain_detector_infer.py` → all pass.

- [ ] **Step 5: Commit** — `git add anatobind/infer/brain_nndet.py scripts/infer_brain_lesions_nndet.py tests/test_nndet_infer.py && git commit -m "nnDetection arm inference entry: h5 to S2-format lesion rows through the nndet runner"`

---

### Task 9: After training — extraction, fold-0 report, inference smoke

**Precondition:** training and its sweep finished: `train.log` in the training dir ends with the analysis lines (`run_analysis_suite … Found 51 predictions for analysis`), and `T=${det_models}/Task903_FastMRIBrainSmallLesion/RetinaUNetV001_D3V001_3d/fold0` holds `model_last.ckpt`, `plan_inference.pkl` and 51 `sweep_predictions/*_boxes.pt`.

**Files:**
- Create: `docs/verification/2026-09-29/brain_nndet/fold0/` (REPORT.md, froc.csv, output.txt, rule_a.json), `docs/verification/2026-09-29/brain_nndet/training.txt`, `docs/verification/2026-09-29/brain_nndet/infer_smoke.md`

- [ ] **Step 1: Confirm the run**

```bash
T=/data2/congcong/data/FM_data/derived/nndet_models/Task903_FastMRIBrainSmallLesion/RetinaUNetV001_D3V001_3d/fold0
{ ls $T; ls $T/sweep_predictions/*_boxes.pt | wc -l; head -3 $T/train.log; tail -5 $T/train.log; tail -c 1500 logs/brain_nndet/fold0.log; } | tee docs/verification/2026-09-29/brain_nndet/training.txt
```

Expected: 51 state files; `train.log` ends with the analysis lines. The launcher log may end with batchgenerators' teardown error `RuntimeError: One or more background workers are no longer alive` — that is printed after the work is done and is not a failure (seen in Task 1's toy run); any other traceback is. nnDetection may also hang at exit after that error (toy run: 6 threads in futex/poll wait, GPU memory still held, outputs already complete): once the outputs above are complete, the controller stops it with `kill -TERM <pid of nndet_train>` and records the pid, the time and the exit code in `training.txt`. Otherwise stop and report.

- [ ] **Step 2: Extract** (nndet env, CPU is enough):

```bash
bash -c 'source scripts/nndet_env.sh && T=$det_models/Task903_FastMRIBrainSmallLesion/RetinaUNetV001_D3V001_3d/fold0 && R=/data2/congcong/data/FM_data/derived/nndet_runs && nice -n 19 python scripts/nndet_runner.py extract --state $T/sweep_predictions --params default --out $R/fold0_default.json && nice -n 19 python scripts/nndet_runner.py extract --state $T/sweep_predictions --params swept --train-dir $T --out $R/fold0_swept.json'
```

- [ ] **Step 3: Fold-0 report**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_nndet.py \
  --dets /data2/congcong/data/FM_data/derived/nndet_runs/fold0_default.json \
  --swept /data2/congcong/data/FM_data/derived/nndet_runs/fold0_swept.json \
  --folds 0 --out docs/verification/2026-09-29/brain_nndet/fold0
```

Check: the nnU-Net 2d line reproduces the probe (hits 92 of 280 at threshold 0.60, 1.549 FP per volume); if it differs, the recomputed value rules (spec §2) and the difference is written into the report's commit message. Rule A is whatever `rule_a.json` says.

- [ ] **Step 4: Inference smoke.** Use the h5 named in S2's inference smoke record `docs/verification/2026-09-28/brain_detector/infer_smoke.md` (one of the 130 small-vessel-disease FLAIR studies without boxes), on an idle GPU:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_lesions_nndet.py --h5 <the h5 path from that record> --out /data2/congcong/data/FM_data/derived/nndet_runs/infer_smoke --fold 0 --gpu <idle gpu>
```

Write `docs/verification/2026-09-29/brain_nndet/infer_smoke.md`: the command, the printed line, the number of rows, the rows at or above the fold-0 operating threshold, the first three rows verbatim, and a check that every row has exactly the keys `z0, z1, score, boxes`.

- [ ] **Step 5: Index with known deviations and timing.** Write `docs/verification/2026-09-29/brain_nndet/README.md`: one line per record in this folder (what it is, the command that made it); "Known deviations" = the lesions whose box changed by overwrite (`build_report.json` via `task_build.txt`), the lesions lost in resampling and the IoU summary (`gt_check/gt_check.json`), the task-903 labels that are not painted (other fastMRI+ labels in lesion volumes, as S2 D10); "Timing" = the preprocessing, training and sweep wall clock read from the logs and `train.log` timestamps. Every number is copied from those files, none typed from memory.

- [ ] **Step 6: Commit** — `git add docs/verification/2026-09-29/brain_nndet && git commit -m "nnDetection fold 0: default-parameter report with rule A, swept version as NOT_GATE, inference smoke, record index"`

- [ ] **Step 7: Controller** reports rule A to the user. If it passes, report the five-fold wall-clock estimate and ask before any further training (N12); if it fails, stop (D9 of S2).

---

### Task 10: Documentation, final review, merge

**Files:**
- Modify: `CLAUDE.md` (the nndet env in the Python line; the status sentence; test count), `STATUS.md` (full rewrite, five sections)

- [ ] **Step 1:** nvgen full suite: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider` → record the count; nndet env: the Global Constraints command → 2 passed.
- [ ] **Step 2:** `CLAUDE.md`: add to the Python line "nnDetection 用 env `nndet`(py3.8/torch 1.11,`source scripts/nndet_env.sh`,不与 `nnunet_env.sh` 同 shell)"; update the status sentence to point at `docs/verification/2026-09-29/brain_nndet/fold0/REPORT.md` and this plan; update the test count. Change nothing else.
- [ ] **Step 3:** `STATUS.md` rewritten with the five fixed sections (verified with commands and raw outputs; decisions for the user: rule A outcome and five folds, push; next steps; pitfalls: two envs, splits auto-generation, margin, `train.mode`; the why of N1–N14).
- [ ] **Step 4:** Commit — `git add CLAUDE.md STATUS.md && git commit -m "Docs: nnDetection second arm fold 0 status and code map"`.
- [ ] **Step 5:** Whole-branch review on the most capable model; fix; re-review; then the controller merges to main and tags `handoff/<date>-brain-nndet-fold0` (no push).
