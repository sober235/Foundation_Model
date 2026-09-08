# M1 Plan 2a: Arm B Minimal Trainable Path

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the shortest end-to-end trainable path for arm B of the M1 bet experiment — `image → backbone → A masks + U_B boxes → relation token r_ij → main-host CE against the annotator's tissue_id` — and use it to prove that the M1 export is geometrically and semantically aligned with the model's assumptions. This plan produces **no experimental evidence** and **no M1 numbers**. Its only deliverables are working code, passing structural tests, and a human-verified overlay figure.

**Non-goal:** This is not §8 stage training and not the M1 gate run. See "What this plan deliberately does not do" below.

**Architecture:** New `anatobind/model/` package (backbone, decoders, relation, losses, assembly) and `anatobind/train/` (dataset), each unit-tested on synthetic tensors, plus two thin scripts: a fold-0 smoke trainer and an overlay renderer for human verification. The M1 export at `/data2/congcong/data/FM_data/derived/skmtea/m1/` is read-only input.

**Tech Stack:** Python 3.11 in `~/anaconda3/envs/nvgen` (torch 2.5.1+cu121, monai 1.5.2, einops 0.8.2, numpy 2.4, nibabel 5.4, scipy 1.17, pytest 9.1). Always run with `PYTHONNOUSERSITE=1` — `~/.local` holds a torch 2.11/cu130 install that shadows the env otherwise. Single GPU, pinned externally with `CUDA_VISIBLE_DEVICES=0`.

**Spec:** `RESEARCH_PLAN.md` v2.1 §4.1–4.3 (preprocessing, backbone, identity-anchored A), §4.5 (U_B), §4.6 (relation token, existence gating), §5.1 (relation truth is `tissue_id`, **never** overlap), §5.3 (loss composition), §8 stage III (relation binding), §9.1 (M1 gate). Read §5.1 and §9.1 before starting.

---

## Decision record (2026-09-07, user-confirmed unless marked)

| # | Decision | Rationale |
|---|---|---|
| A1 | Target is the **M1 bet experiment (§9.1)**, not the §8 four-stage training | §12 gates M2 on M1; running §8 first removes the falsification gate |
| A2 | Deliver a **minimal trainable path first**, then stop for review | First contact between real exported data and a model reliably exposes coordinate/label errors; finding them at step 200 is cheaper than after the full stack is written |
| A3 | Arm B first; **arm A (nnU-Net) is still required** and comes later | All technical risk is in arm B; arm A is a mature framework and mostly format conversion. Without arm A there is no M1 verdict |
| A4 | **Both arms train from scratch.** No SSL pretraining, no external weights | Otherwise arm B's win is a data/weights advantage, not a relation-modelling advantage |
| A5 | Backbone is a **shrunk** 3D Swin, ~3M params, patch 64×128×128 — a throwaway scaffold | Minimal path is for finding alignment bugs, not for numbers. **The M1 gate run must use a scale matched between arms; this scale must not silently become the final config** |
| A6 | Heads in scope: **A + U_B + R**. Out: S, U_Q, E, E\* gating | S degenerates on SKM-TEA (single-protocol 3T qDESS); U_Q has no signal on clean images; E requires degraded pairs and belongs to stage IV |
| A7 | **Clean images only** (`image_clean_e1`), fold 0, the 311 in-segmentation instances | Degraded views pull in the intervention engine, U_Q, E and E\* gating — four subsystems that would confound "is the plumbing correct?" |
| A8 | Model code is **test-first**, structural properties only | Structural tests are the only automatic guard against "loss is falling but the labels are wrong". No test asserts convergence |
| A9 | M1 degradation is **noise + undersampling only; no motion** | Motion simulation is unimplemented and blocked on a non-technical item ("once authorship is decided"). Motion is an Exp 4 held-out generalisation condition and does not belong in M1 training. **Project risk, flagged** |
| A10 | Backbone uses **MONAI `SwinViT`**, configured to the plan's shrunk geometry, rather than a hand-written Swin — *decided during implementation, reversible* | ~10 lines instead of ~250, on the least risky component, so effort goes to the risky parts (mask decoding, mm-geometry, host mapping, crop bookkeeping). Cost: MONAI has no `M_valid` key masking and no physical-coordinate RoPE. Both are deferred — see "Deferred from spec" |

### Deferred from spec (must be revisited before the M1 gate run)

- **`M_valid` attention masking** (§4.1, §4.2). Crops here are always taken fully inside the volume, so nothing is padded and `M_valid` is all ones. The variable-size claim is untested by this plan. Required for sliding-window inference and cross-dataset volumes.
- **Physical-coordinate 3D RoPE** (§4.2). MONAI's Swin uses relative position bias. §4.2 itself flags the physical-PE claim as "断言不是证据,须消融", so it is an ablation, not a prerequisite.
- **Hard negatives** `L_hard^0` (§5.3). `λ_h = 0` here.
- **Predicted-geometry relation input.** See the warning below.

### ⚠ Warning: this configuration cannot produce M1 evidence

Three teacher-forcing simplifications are used to make the smoke run stable:

1. Existence gating uses **ground-truth** A presence, not predicted presence.
2. The geometry features `G_ij` are computed from **ground-truth** A masks, not predicted masks.
3. The box fed to `G_ij` for a matched query is the **ground-truth** box, not the predicted one.

With ground-truth masks feeding `IoA(B_j, M_i)`, the relation head can reach a high main-host accuracy by learning approximately `argmax_i IoA` — which is precisely the seg-then-lookup baseline. **Any accuracy number produced by this plan is meaningless as evidence for or against the M1 proposition.** §9.1 already states the comparison is only meaningful under degradation and combination transfer. The M1 gate run must switch both to predicted quantities.

Note that `IoA` is a legitimate *input feature* per §4.6 — what §5.1 forbids is deriving the relation *truth* from overlap. Truth here is `host_label`, written by the annotator as `tissue_id` and side-resolved by the data engine. Do not "simplify" it to an overlap argmax.

### What this plan deliberately does not do

Arm A (nnU-Net seg-then-lookup) · degraded views · motion simulation · S head · U_Q branch · E head and `E*` gating · five-fold cross-validation · significance testing · the effusion (116) and ligament (38) strata · any claim about the M1 proposition.

---

## Global Constraints

- Data is read only from `/data2/congcong/data/FM_data/derived/skmtea/m1/`. **Never write into that tree** — a detached export job (`scripts/build_skmtea_m1.py --workers 4`, PPID 1) is writing there until roughly 02:30 on 2026-09-08. Training outputs go to `/data0/congcong/code/Project_Doing/foundation_model/runs/`.
- Only scans with a `manifest.csv` row and `status == ok` may be used. A scan directory that exists without a `boxes.csv` is mid-export; skip it.
- **Array axis order.** The export writes `(X, Y, Z) = (256, 256, 160)` with in-plane X, Y at 0.625 mm and slice Z at 0.8 mm. §4.2's token budget is stated for `(D, H, W) = (160, 256, 256)`; stem `(2, 4, 4)` gives `80 × 64 × 64 = 327,680` S1 tokens, which matches §4.2 exactly. **The model therefore works in `(Z, Y, X)` and every loaded array, mask and box must be permuted accordingly**: `(x0,y0,z0,x1,y1,z1) → (z0,y0,x0,z1,y1,x1)`, spacing `(sx,sy,sz) → (sz,sy,sx)`. This is the single most likely silent failure in the plan; Task 1 tests it directly.
- **Spacing varies per scan** (observed `0.6249083, 0.625, 0.8006518`). Always read `nibabel` zooms per scan. Never hard-code 0.625.
- **Intensity is unnormalised** (observed min 2.3e4, max 5.3e7 on one scan). Normalise per volume before cropping: clip to the [0.5, 99.5] percentiles of the non-zero voxels, then z-score. §4.1.
- **Box coordinates** in `boxes.csv` (`x0..z1`) index the exported 0.625 mm grid; `x0_full..z1_full` index the original 512-grid and are `start = floor(full/2)`, `end = ceil(full/2)` in-plane with Z unchanged. Use the un-suffixed columns; ignore `*_full`.
- **Relation truth** is the `host_label` column, an integer in 1..6 matching the `seg.nii.gz` label of the host structure. Rows used by this plan are exactly those with `layer == "in_seg"`.
- Segmentation labels: 1 patellar cartilage, 2 femoral cartilage, 3 tibial cartilage medial, 4 tibial cartilage lateral, 5 meniscus medial, 6 meniscus lateral. `K = 6` identity-anchored A queries, query `k-1` is permanently label `k`.
- In-segmentation supercategories are `Meniscal Tear` and `Cartilage Lesion` → `U_B` class set of 2 plus no-object.
- Determinism: seed everything (`torch`, `numpy`, `random`) from one `--seed`; log the seed.
- Tests: `cd /data0/congcong/code/Project_Doing/foundation_model && PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q`. Every function is written test-first. Tests are CPU-only, synthetic, and must stay under ~30 s total.
- Git: branch `plan-v5`, author identity from the repo config (`Congcong Liu`), **no AI trailers in commit messages**.

---

## File Structure

| File | Responsibility |
|---|---|
| `anatobind/train/dataset.py` | Read the M1 export: manifest/split filtering, per-volume normalisation, `(X,Y,Z)→(Z,Y,X)` permutation, lesion-centred cropping, target assembly |
| `anatobind/model/backbone.py` | MONAI `SwinViT` configured to the shrunk geometry; returns `F1..F4` |
| `anatobind/model/decoders.py` | `ADecoder` (identity-anchored, K=6, no Hungarian) and `UBDecoder` (DETR-style, M=8) |
| `anatobind/model/relation.py` | Geometry features in mm, `φ`, `Transformer_R`, `h_R`, main-host head with existence gating |
| `anatobind/model/losses.py` | `L_A`, `L_UB` with Hungarian matching, `L_rel` (main-host CE + relation BCE) |
| `anatobind/model/armb.py` | `ArmBMinimal`: assembly, forward, loss aggregation |
| `scripts/train_armb_minimal.py` | Fold-0 smoke trainer: GPU 0, bf16 autocast, checkpoint + jsonl log under `runs/` |
| `scripts/render_binding_overlay.py` | Step-5 verification: renders box · host mask · `host_label` · side overlays to PNG |
| `tests/test_armb_dataset.py`, `test_armb_backbone.py`, `test_armb_decoders.py`, `test_armb_relation.py`, `test_armb_losses.py`, `test_armb_model.py` | Structural unit tests on synthetic data |

---

### Task 1: Dataset — export reader, axis permutation, lesion-centred crops

**Files:**
- Create: `anatobind/train/__init__.py`, `anatobind/train/dataset.py`
- Test: `tests/test_armb_dataset.py`

**Interfaces:**
- `list_ready_scans(root: Path) -> list[str]` — manifest rows with `status == "ok"` **and** an existing `boxes.csv`.
- `load_fold(root: Path, fold: int) -> tuple[list[str], list[str]]` — `(train_ids, val_ids)` from `splits.json`, intersected with `list_ready_scans`.
- `normalise_volume(vol: np.ndarray) -> np.ndarray` — [0.5, 99.5] percentile clip over non-zero voxels, then z-score.
- `to_model_frame(arr: np.ndarray) -> np.ndarray` — `(X,Y,Z) → (Z,Y,X)`.
- `box_to_model_frame(box: tuple[int,...]) -> tuple[int,...]` — `(x0,y0,z0,x1,y1,z1) → (z0,y0,x0,z1,y1,x1)`.
- `class SkmteaArmBDataset(scan_ids, root, patch=(64,128,128), train=True, seed=0)` yielding a dict with
  `image (1,D,H,W) float32` · `seg (D,H,W) int64` · `boxes (N,6) float32` in model-frame **voxel** coords relative to the crop · `box_classes (N,) int64` in `{0: Meniscal Tear, 1: Cartilage Lesion}` · `host_label (N,) int64` in `1..6` · `spacing_mm (3,) float32` in `(z,y,x)` order · `present (K,) bool` — which of the 6 labels occur in the crop · `scan_id`, `crop_origin`.
- Cropping: pick one `in_seg` box uniformly, centre the patch on its centre with uniform jitter of ±1/4 patch per axis, then clamp so the crop lies fully inside the volume. Keep every box whose centre falls inside the crop. Validation uses jitter 0 and the first box.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_armb_dataset.py
import numpy as np
import nibabel as nib
import pytest

from anatobind.train.dataset import (
    box_to_model_frame, normalise_volume, to_model_frame,
    list_ready_scans, load_fold, SkmteaArmBDataset,
)


def test_to_model_frame_maps_xyz_to_zyx():
    a = np.zeros((4, 6, 8), dtype=np.float32)
    a[1, 2, 3] = 1.0
    b = to_model_frame(a)
    assert b.shape == (8, 6, 4)
    assert b[3, 2, 1] == 1.0


def test_box_permutation_is_the_inverse_of_the_array_permutation():
    # a voxel inside the box in (X,Y,Z) must stay inside the permuted box in (Z,Y,X)
    arr = np.zeros((10, 12, 14), dtype=np.float32)
    x0, y0, z0, x1, y1, z1 = 2, 3, 4, 6, 9, 11
    arr[x0:x1, y0:y1, z0:z1] = 1.0
    m = to_model_frame(arr)
    bz0, by0, bx0, bz1, by1, bx1 = box_to_model_frame((x0, y0, z0, x1, y1, z1))
    assert m[bz0:bz1, by0:by1, bx0:bx1].min() == 1.0
    assert m.sum() == (bz1 - bz0) * (by1 - by0) * (bx1 - bx0)


def test_normalise_is_scale_and_offset_invariant_and_zero_mean():
    rng = np.random.default_rng(0)
    v = rng.gamma(2.0, 1e6, size=(8, 8, 8)).astype(np.float32)
    a, b = normalise_volume(v), normalise_volume(v * 7.0)
    assert abs(float(a.mean())) < 1e-3 and abs(float(a.std()) - 1.0) < 0.05
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_list_ready_scans_skips_rows_without_boxes_csv(tmp_path):
    # a manifest row whose directory is still mid-export must not be offered
    ...


def test_crop_is_inside_the_volume_and_contains_the_seed_box(tmp_path):
    # synthetic 3-scan export; every sample's boxes lie inside [0, patch)
    ...


def test_kept_boxes_carry_their_host_label_and_class(tmp_path):
    # host_label survives cropping unchanged; box_classes maps the two supercategories
    ...
```

- [ ] **Step 2: Implement** `anatobind/train/dataset.py` until the tests pass.
- [ ] **Step 3: Verify** — `pytest tests/test_armb_dataset.py -q` green, and a one-off read of a real completed scan prints shapes `(1,64,128,128)`, `spacing_mm ≈ (0.80, 0.625, 0.625)`, and at least one box with `host_label ∈ 1..6`.

---

### Task 2: Backbone — shrunk 3D Swin

**Files:** Create `anatobind/model/__init__.py`, `anatobind/model/backbone.py`; Test `tests/test_armb_backbone.py`

**Interfaces:**
- `class Backbone(embed_dim=32, depths=(2,2,6,2), num_heads=(2,4,8,16), patch_size=(2,4,4), window_size=(4,8,8))`, wrapping `monai.networks.nets.swin_unetr.SwinTransformer`; `forward(x: (B,1,D,H,W)) -> tuple[F1..F4]` with channel dims `(C, 2C, 4C, 8C)` and strides `(2,4,4) · (4,8,8) · (8,16,16) · (16,32,32)`.
- `Backbone.num_parameters() -> int`.

- [ ] **Step 1: Write the failing tests** — forward on `(2,1,64,128,128)` returns four maps of shape `(2,32,32,32,32)`, `(2,64,16,16,16)`, `(2,128,8,8,8)`, `(2,256,4,4,4)`; parameter count is between 1M and 6M (A5's "throwaway scale" is asserted, so a silent upgrade to the default 12M fails the suite); a window larger than the feature map does not raise.
- [ ] **Step 2: Implement.** If MONAI's `SwinTransformer` rejects the non-cubic `patch_size=(2,4,4)` or the window clipping, record the exact error in this file and fall back to a hand-written Swin block — that fallback is the only sanctioned path back to A10's alternative.
- [ ] **Step 3: Verify** — `pytest tests/test_armb_backbone.py -q` green; print the parameter count into the plan's execution ledger.

---

### Task 3: Decoders — identity-anchored A, DETR-style U_B

**Files:** Create `anatobind/model/decoders.py`; Test `tests/test_armb_decoders.py`

**Interfaces:**
- `class ADecoder(d_model=128, K=6, layers=3)` — `K` learned queries, masked cross-attention over `F2..F4`, mask logits by dot product with `F1` upsampled to crop resolution. Returns `masks (B,K,D,H,W)`, `presence (B,K)`, `embed (B,K,d)`. **No Hungarian anywhere in this class.**
- `class UBDecoder(d_model=128, M=8, layers=3, num_classes=2)` — returns `logits (B,M,num_classes+1)`, `boxes (B,M,6)` as `(cz,cy,cx,dz,dy,dx)` normalised to `[0,1]` by the crop extent, `embed (B,M,d)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_decoder_query_index_is_the_class_index():
    # query k must be permanently label k+1: swapping two GT labels must change
    # which query's mask matches, i.e. there is no permutation invariance
    ...

def test_a_decoder_has_no_hungarian_dependency():
    import inspect, anatobind.model.decoders as d
    assert "linear_sum_assignment" not in inspect.getsource(d.ADecoder)

def test_ub_boxes_are_normalised_and_sizes_positive():
    ...

def test_shapes():
    ...
```

- [ ] **Step 2: Implement.**
- [ ] **Step 3: Verify** — tests green.

---

### Task 4: Relation module — mm geometry, φ, Transformer_R, main-host head

**Files:** Create `anatobind/model/relation.py`; Test `tests/test_armb_relation.py`

**Interfaces:**
- `geometry_features(a_masks, u_boxes, spacing_mm, crop_shape) -> (B,K,M,5)` — `Δz, Δy, Δx` in **millimetres** between the A mask centroid and the U box centre, `‖Δ‖`, and `IoA = |B_j ∩ M_i| / |B_j|`. Computed under `torch.no_grad()`; laterality and hierarchy are omitted because they are already encoded in the identity-anchored query index.
- `class RelationModule(d_model=128, geo_dim=64, layers=2, heads=8)` — `φ: [A_i ; U_j ; g(G_ij)] → d`, self-attention over valid pairs, then `R_ij = σ(h_R(r_ij))` and main-host logits `(B,M,K+1)` with the last column = `none`.
- Existence gating: a mask `(B,K)` of present A queries; absent queries get `-inf` in the host softmax and are excluded from the pair self-attention.

- [ ] **Step 1: Write the failing tests**

```python
def test_geometry_delta_is_in_millimetres_not_voxels():
    # anisotropic spacing (0.8, 0.625, 0.625): a 10-voxel offset along z must be
    # 8.0 mm while the same offset along x is 6.25 mm
    ...

def test_ioa_is_intersection_over_box_area_not_over_union():
    # a box fully inside a large mask has IoA == 1 regardless of the mask's size
    ...

def test_absent_queries_are_masked_out_of_the_host_softmax():
    # host probabilities over absent queries are exactly zero and the rest sum to 1
    ...

def test_relation_output_is_permutation_consistent_over_u_queries():
    # permuting U queries permutes the host logits identically
    ...
```

- [ ] **Step 2: Implement.**
- [ ] **Step 3: Verify** — tests green.

---

### Task 5: Losses

**Files:** Create `anatobind/model/losses.py`; Test `tests/test_armb_losses.py`

**Interfaces:**
- `a_loss(masks, presence, seg, present) -> dict` — per-query mask BCE + soft Dice on present structures, presence BCE on all K. No matching.
- `hungarian_match(logits, boxes, tgt_classes, tgt_boxes) -> (idx_pred, idx_tgt)` — cost `2·(1 - p_class) + 5·L1 + 2·(1 - GIoU3D)`.
- `ub_loss(...) -> dict` — class CE (no-object weight 0.1), box L1, GIoU3D.
- `rel_loss(host_logits, R, matched_host_label, present) -> dict` — main-host CE over `K+1`, plus relation BCE with target 1 at the host and 0 at the other present queries. `λ_h = 0`.
- `total_loss(...) -> (scalar, dict)` — `L_A + L_UB + λ_R · L_rel`, `λ_R = 1.0` per §5.3.

- [ ] **Step 1: Write the failing tests**

```python
def test_each_loss_is_zero_on_a_perfect_prediction():
    # saturated logits + exact boxes + exact masks -> every term < 1e-4
    ...

def test_hungarian_recovers_a_known_permutation():
    ...

def test_giou3d_matches_iou_for_identical_boxes_and_is_negative_when_disjoint():
    ...

def test_host_ce_target_uses_host_label_not_max_ioa():
    # construct a case where the maximum-IoA structure is NOT the annotated host;
    # the loss must be driven by host_label. Guards §5.1.
    ...
```

`test_host_ce_target_uses_host_label_not_max_ioa` is the single most important test in this plan. It is the automatic guard against a future "simplification" that derives the relation truth from overlap and thereby hands the M1 comparison to the baseline.

- [ ] **Step 2: Implement.**
- [ ] **Step 3: Verify** — tests green.

---

### Task 6: Assembly

**Files:** Create `anatobind/model/armb.py`; Test `tests/test_armb_model.py`

**Interfaces:**
- `class ArmBMinimal(nn.Module)` — holds backbone, both decoders, relation module; `forward(batch) -> dict`; `compute_loss(out, batch) -> (scalar, dict)`.
- Geometry and existence gating consume **ground-truth** masks and presence (see the warning above); the call sites carry a `# MINIMAL PATH:` comment naming what must change for the gate run.

- [ ] **Step 1: Write the failing tests**

```python
def test_forward_backward_reaches_the_backbone():
    # after loss.backward(), every backbone parameter has a finite, non-zero grad
    ...

def test_no_nan_under_bf16_autocast():
    ...

def test_a_sample_with_no_boxes_produces_a_finite_loss():
    # empty-target batches must not divide by zero
    ...
```

- [ ] **Step 2: Implement.**
- [ ] **Step 3: Verify** — full suite green: `pytest tests/ -q` (previous 67 + the new ones).

---

### Task 7: Fold-0 smoke trainer

**Files:** Create `scripts/train_armb_minimal.py`

**Interfaces:** `--fold 0 --steps 300 --batch 2 --lr 3e-4 --seed 0 --out runs/armb_minimal_<stamp>`. AdamW, cosine schedule, bf16 autocast, gradient clipping at 1.0. Writes `config.json`, `metrics.jsonl` (step, each loss term, grad-norm, lr, seconds), `last.pt`. Evaluates on fold 0's validation scans every 50 steps and logs main-host accuracy **labelled in the log as `NOT-EVIDENCE`**.

- [ ] **Step 1: Implement** (no unit test; this is a script).
- [ ] **Step 2: Verify on synthetic data first** — `--steps 5` against a tiny synthetic export in `tmp`, confirming the script runs without touching `/data2`.
- [ ] **Step 3: Verify on real data** — only after `manifest.csv` reaches 155 `ok` rows. `CUDA_VISIBLE_DEVICES=0`, 300 steps. Gate: no NaN, no shape error, total loss and each of `L_A`, `L_UB`, `L_rel` lower at step 300 than at step 10, peak GPU memory recorded.

---

### Task 8: Overlay verification figure

**Files:** Create `scripts/render_binding_overlay.py`

**Interfaces:** `--scans MTR_xxx,... --out ~/figs/anatobind_m1/`. For each scan, one PNG per `in_seg` box: the axial slice at the box centre with the box outline, the `host_label` mask contour, and a caption carrying `scan_id · ann_id · supercategory · tissue_id · host_label · host_side · host_ratio`. Rendered from the **model-frame** arrays, i.e. after `to_model_frame`, so the figure verifies the permutation the model actually sees.

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Render 3–5 scans** spanning both supercategories and both `host_side` values.
- [ ] **Step 3: Human gate** — present each figure as a `http://localhost:8765/...` URL line followed by its absolute path line. **The user confirms that box, host mask, `host_label` and side agree.** This gate cannot be passed by the agent alone.

---

## Exit gate for this plan

All of the following, in order:

1. `pytest tests/ -q` green, including the 67 pre-existing data-engine tests.
2. `manifest.csv` has 155 rows, all `ok` (the detached export finishes ~02:30 on 2026-09-08).
3. 300 steps on fold 0 with no NaN and every loss term falling.
4. Overlay figures reviewed and confirmed by the user.

Only then does the next plan (arm B at full scale, then arm A, then the evaluation harness) get written. **If step 4 fails, stop and fix the data engine — do not tune the model.**

## Findings from execution (2026-09-07)

Recorded here because they change what the data means, not just how it is read.

### F1. `host_side == "unresolved"` leaves `host_label` empty

`read_in_seg_boxes` crashed on `int("")`.  Semantics: the annotator gave the
tissue (meniscus), but the medial/lateral side could not be decided from mask
overlap, so **the relation truth is unknown** — not absent, and certainly not
some default label.  Implemented as `UNKNOWN_HOST = 0`: the box still supervises
`U_B`, and the instance is dropped from the host CE (`ignore_index`) and from
the relation BCE mask.  Guarding this matters because the previous code path
computed `host_label - 1 = -1`, which silently indexes the **last** query — it
would have trained a fabricated binding onto meniscus-lateral.

### F2. The `single_no_overlap` instances are the sharpest cases in the M1 bet

Final counts over the completed export (155/155 scans, 311 in-segmentation
instances, matching §9.1 exactly): `medial 76 · lateral 68 · single 160 ·
single_no_overlap 5 · unresolved 2`.  So the edge cases are **7 of 311 (2.3%)**,
not the ≈5% extrapolated from the first ~122 boxes.  The six visible while the
export was still running:

| scan · ann | tissue_id → host | labels inside the box | gap box→host |
|---|---|---|---|
| MTR_020 · 66 | 4 → 2 | background only | 19 vox |
| MTR_020 · 67 | 4 → 2 | **4 and 6** | 7 vox |
| MTR_040 · 74 | 1 → unresolved | background only | 3 vox to 5, 26 to 6 |
| MTR_052 · 80 | 1 → unresolved | 2 | 13 vox to 5, 5 to 6 |
| MTR_095 · 79 | 4 → 2 | background only | 7 vox |
| MTR_101 · 241 | 5 → 1 | 2 | 15 vox |

Not a coordinate bug: a frame error would give large random gaps and would break
the other ~116 boxes too; these gaps are 3–19 voxels (2–12 mm), i.e. boxes drawn
beside a thin structure.  **MTR_020 · 67 is the canonical case for §9.1**: the box
contains labels 4 and 6 while the annotated host is 2, so overlap-argmax — the
seg-then-lookup baseline — is wrong by construction and only relation modelling
can recover the truth.  These instances must be kept, and the M1 analysis should
report them as their own stratum.

### F3. Open decision — resolve the side by proximity when overlap is zero

Spec 5.1 decides medial/lateral by mask overlap ratio.  When the overlap is
zero the engine gives up, but distance separates the two cleanly (3 vs 26, and
5 vs 13 voxels above).  A nearest-structure fallback would recover both
instances.  This still takes the *tissue* from the annotator and only
disambiguates the *side*, so it does not violate 5.1 — but it changes rule D5
and therefore needs an explicit decision before the gate run.

The completed export settles the size of the prize: **exactly 2 unresolved
instances out of 311 (0.6%)**.  Changing a confirmed truth rule for two
instances is not worth the risk; the recommendation is to keep D5 as it stands,
report the abstention in the coverage numbers, and revisit only if the M1
analysis shows the abstention actually moves the verdict.

### F4. Fold-0 smoke run: gate passed, and one number worth keeping

`runs/armb_fold0_first`, 300 steps, batch 2, patch 64×128×128, 682 s on one
A800 shared with another user's job.  No non-finite value at any step, and all
eight loss terms lower over steps 281–300 than over 1–20 (total 11.94 → 8.43;
`host_ce` 0.96 → 0.61; `L1` 0.98 → 0.54).  **Gate passed: the plumbing agrees
with itself.**

The model's host accuracy is **0.652** over 46 scored validation instances and
is flat from step 50 onward.  As stated above this is not evidence — but the
comparison run alongside it is worth recording, because it does not depend on
our model at all:

| binder on fold 0 validation | accuracy |
|---|---|
| argmax IoA against **ground-truth** masks (seg-then-lookup, no learning) | **0.804** |
| the 300-step model | 0.652 |
| majority class | 0.413 |

So **0.804 is an oracle ceiling for the binding step of any seg-then-lookup
arm** — arm A will use predicted masks and can only do worse.  Equivalently,
measured directly over the fold-0 validation boxes, **8 of 57 instances (14%)
have their annotated host different from the structure their box overlaps
most**.  That 14–20% is the headroom the relation model has to convert, and
§9.1's 10-point threshold sits inside it.

Every one of the eight is an anatomically coherent adjacency confusion, not a
data error: meniscal tears whose boxes overlap the tibial plateau or femoral
condyle cartilage they sit against (MTR_069, MTR_104, MTR_163, MTR_236), and
patellofemoral cartilage lesions where the two facing cartilages swap
(MTR_052 ann 82 and 83 are a mirrored pair).  This is the physical basis of the
bet: at a joint, the structures a lesion touches are exactly the structures
overlap cannot tell apart.

**One case still needs a human call: MTR_110 ann 15**, a meniscal tear with
*zero* overlap with its host in 3D.  Measured rather than eyeballed (an earlier
note in this file called it "intercondylar" from a single slice; that reading
was wrong):

- box `(152,109,89)-(160,137,104)`, 5.0 × 17.5 × 12.0 mm, containing only
  femoral cartilage (305 voxels);
- the nearest medial-meniscus voxel is **1.25 mm away** (2 voxels), and the box
  lies entirely inside the medial meniscus's own bounding box;
- Z is the medial/lateral axis here (medial meniscus z 82–113, lateral z 32–60),
  and the box's z 89–104 agrees with its `side=medial`;
- **`ann 16` in the same knee carries the same host 5 and sits squarely on the
  meniscus** (772 voxels, gap 0.00 mm), so the segmentation is sound and ann 15
  is the outlier, not the mask.

So it is a near miss, not a gross error: the box is displaced by about its own
height away from the segmented meniscus, toward the femoral side.  Tear signal
extending past the segmented structure, a generously drawn box, or a genuine
mis-annotation all remain possible.  Diagnostic figure:
`~/figs/anatobind_m1_fold0/MTR_110_ann15_vs_ann16_diagnostic.png`.

**Recommendation: keep it.**  Its host is 1.25 mm away while its box contains
only femoral cartilage, so overlap says femoral cartilage and the radiologist
says medial meniscus — precisely the instance the relation model has to get
right.  Dropping the instances where overlap fails would quietly build a
benchmark that favours the baseline.  One instance changes nothing
statistically; the principle governs the other seven.

### F5. Effusion and ligament boxes are currently trained as background

`read_in_seg_boxes` keeps only the `in_seg` layer, so the 116 effusion and 38
ligament boxes are invisible to `U_B` — i.e. the detector is being taught that
a visible effusion is nothing.  Harmless for a plumbing check, but a real
modelling decision for the gate run: §9.1 treats effusion as the abstention
analysis and ligament as the extension analysis, and neither is "background".

## Known risks carried into this plan

1. **155 scans from scratch may simply not train arm B.** That is a real result, not a bug. Rescuing it with pretrained weights would destroy the bet experiment (A4).
2. **`MTR_150`** fails the absolute frame-gate threshold (1.408 < 1.5) while passing the transform ranking; the threshold is confounded by coil group and disease severity and is deferred "to be fixed together with the confound after the run". It is included here; revisit before the gate run.
3. **fastMRI brain left/right handedness is unknown** (assumed radiological). Irrelevant to SKM-TEA, relevant to any later brain-side binding.
4. **`RESEARCH_PLAN.md` §12 is stale** — M0 still lists the SynthSeg pipeline as the blocking item although it completed 2026-09-06 for all five brain sets.
