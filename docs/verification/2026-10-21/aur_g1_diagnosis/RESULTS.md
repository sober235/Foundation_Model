# Preserved-artifact 2D G1 diagnosis — 2026-10-21

The 2D adapters and shared backbone did update, and the two 2D feature caches are finite and nonconstant with all 13 hosts supported. This rules out several gross execution failures; it does **not** establish a cause of the weak 2D readout or justify a training change. Original selected step 1792, completed step 2048 and G1 **2D FAIL / 3D PASS** remain unchanged. All anatomy references are pseudo-labels: **NOT_EVIDENCE**.

Only CPU artifact analysis was performed: 0 new model constructions, 0 forwards, 0 fits, 0 source medical-image/label reads and 0 GPU work. The audit inspected saved feature/target/validity derivatives, not source arrays or private selection fields. The two 47.26GB 3D feature caches were neither hashed nor numerically read; 3D context below comes solely from the existing anonymous report.

## Historical boundaries and parameter evidence

An actual saved step 0 checkpoint exists. No warm-start reconstruction was needed or performed. Its 176 inherited tensors and 4 seeded adapter tensors independently reproduce the lineage state hashes recorded in that checkpoint; seed 20261019 and inherited legacy step 8500 are recorded. This verifies saved-state/lineage consistency, not a new rerun of the old initialization. The original 145-file Python snapshot matches immutable checkout `0c0f2f8b5efec2a25dfb65b95890d77827cad491` byte-for-byte. Selected export model tensors match the selected checkpoint exactly.

| Parameter group | Tensors | Step0→1792 ΔL2 | Relative ΔL2 |1792→2048 ΔL2 | Step0→2048 ΔL2 | Optimizer steps selected/final |
|---|---:|---:|---:|---:|---:|---:|
| input_2d_adapter | 2 | 0.083319583 | 1.73531% | 0.000596692 | 0.083485425 | 896/1024 |
| output_2d_adapter | 2 | 0.196535162 | 8.39409% | 0.000753651 | 0.197009443 | 896/1024 |
| shared_backbone | 158 | 6.927997285 | 4.89790% | 0.097944598 | 6.931119329 | 1792/2048 |
| shared_decoder | 10 | 0.696834181 | 5.93102% | 0.006977556 | 0.698017857 | 1792/2048 |
| input_3d_adapter | 2 | 0.037892750 | 0.90461% | 0.000840369 | 0.037714155 | 896/1024 |
| output_3d_adapter | 2 | 0.109336944 | 3.07375% | 0.000956410 | 0.109712442 | 896/1024 |
| projector | 4 | 0.067305472 | 0.50949% | 0.000271899 | 0.067576454 | 1792/2048 |

All 2D adapter elements changed; all 158 shared-backbone tensors changed. Adapter/shared optimizer moments are finite and have nonzero second moments. The saved step 0 optimizer is empty; selected/final states contain 180 parameters. Mapping uses the frozen source's parameter traversal and verifies complete IDs and moment shapes: optimizer parameter names were not saved. It is a source-derived mapping, not separately named optimizer telemetry. Contrast weight is 0: the projector has zero moments despite parameter changes, consistent with AdamW decay; parameter movement alone is not evidence of learned contrastive signal. The projector is not used in the G1 backbone features.

## Feature integrity and support

Each cache is 501,716,282 bytes, audited through read-only memory mapping in crop-sized chunks. Features are float16,896 channels on a 1×20×20 grid. Both arms have 96 host-calibration, 48 host-validation, 48 lesion-calibration and 48 lesion-validation observations. All 86,016,000 feature elements per arm are finite; all 896 channels have positive valid-cell variance in every split. Every crop has nonconstant valid features. Target and validity tensors are exactly equal between random and SSL arms; they reproduce the original 13-host validation supports.

| Arm/split | Valid cells | Minimum channel std | Median channel std | Zero-variance channels | Nonfinite elements |
|---|---:|---:|---:|---:|---:|
| random/cal | 38190 | 0.161765 | 0.397729 | 0 | 0 |
| random/val | 19078 | 0.160067 | 0.398869 | 0 | 0 |
| random/lcal | 18731 | 0.153348 | 0.362258 | 0 | 0 |
| random/lval | 18399 | 0.130801 | 0.329789 | 0 | 0 |
| ssl/cal | 38190 | 0.231738 | 0.708842 | 0 | 0 |
| ssl/val | 19078 | 0.223932 | 0.711766 | 0 | 0 |
| ssl/lcal | 18731 | 0.232575 | 0.686828 | 0 | 0 |
| ssl/lval | 18399 | 0.216515 | 0.672785 | 0 | 0 |

Nonzero variance rules out exact cache collapse; it does not establish semantic separability, effective rank, adequate class-conditioned variation or readout adequacy. The SSL host-calibration set has 27,623/38,190 background cells (72.33%); left/right thalamus have 64/74 cells. This imbalance and unweighted cross-entropy are a plausible readout limitation, not a demonstrated cause.

## Which classes account for the 2D result?

The original macro gain is 0.043545226 against the frozen 0.05 threshold, a shortfall of 0.006454774. The table averages the original 3 readout seeds and inspects existing scores; no heads were refitted. Calibration counts are recomputed from saved SSL target/validity tensors.

| Host index/name | Calibration cells | Validation cells | Random Dice | SSL Dice | Gain | Contribution to macro gain |
|---|---:|---:|---:|---:|---:|---:|
| 0 white_matter_left | 1864 | 1102 | 0.529722 | 0.622782 | +0.093060 | +0.007158 |
| 1 white_matter_right | 1986 | 1054 | 0.553459 | 0.596329 | +0.042870 | +0.003298 |
| 2 cortex_left | 2261 | 1162 | 0.491621 | 0.556417 | +0.064795 | +0.004984 |
| 3 cortex_right | 2391 | 1239 | 0.547244 | 0.549201 | +0.001957 | +0.000151 |
| 4 thalamus_left | 64 | 34 | 0.000000 | 0.000000 | +0.000000 | +0.000000 |
| 5 thalamus_right | 74 | 26 | 0.046798 | 0.000000 | -0.046798 | -0.003600 |
| 6 basal_ganglia_left | 82 | 30 | 0.000000 | 0.127108 | +0.127108 | +0.009778 |
| 7 basal_ganglia_right | 95 | 30 | 0.000000 | 0.042319 | +0.042319 | +0.003255 |
| 8 brainstem | 276 | 79 | 0.202265 | 0.211306 | +0.009041 | +0.000695 |
| 9 cerebellum_left | 637 | 181 | 0.233103 | 0.502159 | +0.269056 | +0.020697 |
| 10 cerebellum_right | 595 | 178 | 0.414646 | 0.354519 | -0.060127 | -0.004625 |
| 11 other_deep_grey_left | 121 | 36 | 0.000000 | 0.049320 | +0.049320 | +0.003794 |
| 12 other_deep_grey_right | 121 | 42 | 0.050331 | 0.023817 | -0.026513 | -0.002039 |

Both thalami have zero SSL Dice across all 3 seeds despite 34/26 validation cells and 64/74 calibration cells. Left thalamus also has zero random Dice and therefore contributes zero gain; it alone does not explain the missed gain threshold. Right thalamus, right cerebellum and right other deep grey have negative gains; left cerebellum supplies the largest positive contribution. These are descriptive contributions, not counterfactual effects of fixing a class. Host index 0 is valid. Host targets use map 0=background and map 1..13 = the 13 hosts; that background is not the binding label NO_HOST = 13.

## Training/probe transform comparison

Frozen source comparison found agreement in raw-plane percentile normalization before cropping, true depth 1, the mode-specific stem, invalid fill−1, source-affine patient coordinates, and zero local through-plane coordinate. Relevant historical sources are `ssl/unified_data.py:48`, `unified_data.py:137`, `crops.py:79`, and `ssl/unified_eval.py:147` under `anatobind/aur/`. The exact bytes are saved in the private 145-file source snapshot.

Observation distributions differ: image-only SSL uses foreground-biased plane/crop draws, rotation, independent intensity views and masked reconstruction; host G1 uses uniform unaugmented plane/windows with unmasked backbone features, while lesion G1 uses lesion-centred observations. Original `ssl/eval.py:126` standardizes each arm on its calibration statistics and fits an unweighted 14-class linear cross-entropy head for 300 epochs. These differences are expected implementation behavior; their quantitative effect was not measured. No static intensity/dimensional/coordinate mismatch was found, and no equivalence of actual train/probe sampled distributions is claimed.

## Scope and next measurement

| Input | A coverage | U coverage | R coverage |
|---|---|---|---|
| 2D | Existing G1 host-feature gain FAIL; final A32 quality unmeasured | Existing lesion-feature discrimination condition passes; instance detection/segmentation unmeasured | No binding measurement |
| 3D | Existing G1 host-feature gain PASS; final A32 quality unmeasured | Existing lesion-feature discrimination condition passes; instance detection/segmentation unmeasured | No binding measurement |

The audit cannot separate representation weakness from inherited initialization, class imbalance, sampling or finite-readout limitations. Current random-vs-selected G1 does not isolate the benefit of the new training interval. More epochs, a balanced readout or adapter/SSL changes are not established fixes.

The proposed finite **separate descriptive diagnostic** compares the actual saved step 0 with selected step 1792 on the same 96+48 frozen 2D host observations. All 144 frozen crop files exist and were hashed without deserializing their arrays. Reuse the selected feature cache; create only the 144 step 0 backbone features with the original F2+F3+F4/float16 recipe, and fit exactly 6 linear heads (2 arms×seeds 0/1/2,300 epochs, Adam lr 0.01/weight_decay 0.0001, original standardization and unweighted cross-entropy). Refit selected heads to recover paired crop counts and predeclare original-score reproduction tolerances/backend; stop on a mismatch. Report paired 48-group macro/per-host differences, confusion counts and 1000-bootstrap intervals with seeds 0/1/2. No sweep, continuation, reselection or revised G1 gate. This measures added 1792 steps, not the completed 2048 endpoint; adding final features/heads needs separate registration. No part of this control was executed here.

## Replay and limits

The audit and complete replay exited 0 in 12.921 s and 13.161 s. Replay reproduced the numeric JSON exactly and rehashed all 13 committed inputs and the 145-file historical/original source closures before and after. Cache hashes in [RUN.json](RUN.json) are **audit-time commitments**, not historical freezes. The private script/source snapshot, frozen questions, commitments, availability hashes and numeric results remain outside Git.

Run from `/home/congcongliu/src/Foundation_Model-conformance-work`:

```sh
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python /home/congcongliu/outputs/aur_g1_diagnosis_20261021/v1/audit.py replay
```

[Anonymous numeric evidence](aggregate_anonymous.json) and [execution/commitments](RUN.json) contain exact norms, supports and hashes. [Original G1 report](../../2026-10-20/aur_qualified_training/RESULTS.md) remains authoritative for the failed gate. The task worker performed a separate self-review; fresh controller review is separate. No clinical, final A/U/R, independent-cohort or 3D cache numeric claim is made.
