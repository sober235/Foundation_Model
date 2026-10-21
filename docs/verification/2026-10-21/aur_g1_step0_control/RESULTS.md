# Saved step 0 versus selected step 1792: 2D host readout

The selected checkpoint's mean macro Dice is **0.279637**, versus **0.248889** at the actual saved step 0, a descriptive difference of **+0.030747**. All three readout seeds have positive point estimates, but **all three paired patient-bootstrap 95% intervals cross zero**. These results do not establish a robust patient-level gain, an undertraining diagnosis or a causal component fix. The original **2D G1 FAIL / 3D G1 PASS** is unchanged; this control does not authorize Stage II or III. Pseudo-anatomy references remain **NOT_EVIDENCE**.

## Result and uncertainty

| Readout seed | Saved step 0 | Selected step 1792 | Difference | Paired bootstrap 95% interval |
|---|---:|---:|---:|---:|
| 0 | 0.247850448 | 0.280238539 | +0.032388091 | [-0.002451326, +0.062932548] |
| 1 | 0.245786607 | 0.280458808 | +0.034672201 | [-0.000216815, +0.069412890] |
| 2 | 0.253030807 | 0.278212845 | +0.025182039 | [-0.006752436, +0.062170694] |

Each interval uses 1,000 paired resamples of the same 48 source-qualified validation patient groups, with bootstrap seeds 0/1/2. The three seeds vary readout initialization; they are not independent backbone-training runs or new cohorts. The confidence intervals condition on the frozen calibration set and fitted heads. The patients belong to an already exposed development evaluation, and the anatomy labels are pseudo-references.

This isolates a *descriptive readout contrast* along the same training trajectory: actual saved step 0 versus original selected step 1792. It does not measure the completed step 2048 endpoint. It also differs from original G1's selected-versus-random comparison, whose gain remains 0.043545 < 0.05. No threshold, checkpoint selection or gate verdict was revised.

| Host | Validation cells | Step 0 mean Dice | Selected mean Dice | Difference |
|---|---:|---:|---:|---:|
| white_matter_left | 1102 | 0.581014 | 0.622782 | +0.041767 |
| white_matter_right | 1054 | 0.583274 | 0.596329 | +0.013055 |
| cortex_left | 1162 | 0.529516 | 0.556417 | +0.026900 |
| cortex_right | 1239 | 0.527929 | 0.549201 | +0.021273 |
| thalamus_left | 34 | 0.000000 | 0.000000 | +0.000000 |
| thalamus_right | 26 | 0.018018 | 0.000000 | -0.018018 |
| basal_ganglia_left | 30 | 0.000000 | 0.127108 | +0.127108 |
| basal_ganglia_right | 30 | 0.049593 | 0.042319 | -0.007275 |
| brainstem | 79 | 0.244054 | 0.211306 | -0.032748 |
| cerebellum_left | 181 | 0.341638 | 0.502159 | +0.160521 |
| cerebellum_right | 178 | 0.243282 | 0.354519 | +0.111237 |
| other_deep_grey_left | 36 | 0.092075 | 0.049320 | -0.042756 |
| other_deep_grey_right | 42 | 0.025166 | 0.023817 | -0.001349 |

Both cerebella and left basal ganglia contribute positive changes, while brainstem and left other deep grey decline. Left thalamus is zero at both boundaries across all seeds; selected right thalamus is also zero. The pattern does not support describing the new training interval as uniformly beneficial. Nonzero support and earlier finite-cache/update checks rule out gross absence or nonexecution, not class-imbalance, representation or readout limitations.

## Controlled execution and verification

The approved v2 attempt completed once in **35.611 seconds**, within its 900-second computation limit, with exactly **144 step-0 backbone forwards and six readout fits**. It used the same 96 calibration and 48 validation frozen 2D observations. Selected features were reused. All three selected heads were fitted first with the original settings; every macro Dice and all 13 per-host Dice scores reproduced the original values **exactly** (maximum absolute error 0.0; registered tolerance 1e-6). Only then was the step-0 backbone constructed and evaluated. There were no random refits, no final-2048 arm, no source medical-image/label reads and no extra training.

The readouts use original per-arm CPU float32 mean/sample standard deviation with a 1e-3 floor, CUDA training, CPU scoring, unweighted 14-class cross-entropy, Adam learning rate 0.01/weight decay 1e-4, 300 epochs and seeds 0/1/2. Features use the original F2 plus trilinearly upsampled F3/F4 recipe, saved in float16. The backbone remained unchanged and was restored/verified against its original state hash. Input, model-checkpoint, control-source and historical-source commitments passed before and after execution.

The authoritative evidence is the atomic `completion.json` with PASS, verified input/source and backbone integrity, and its embedded verified metrics. Standalone `metrics_anonymous.json` deliberately remains PROVISIONAL; it is not successful evidence. The parent finalizer cannot publish PASS after a timeout/error or missing integrity record.

An independent CPU-only implementation recomputed pooled-count Dice, 14-class confusion-table consistency, per-host differences and all 3×1,000 paired bootstrap summaries using NumPy float64. It matched the original torch float32 arithmetic within **3.881e-8**, below a predeclared arithmetic tolerance of 1e-7. That arithmetic tolerance did not change the selected-head reproduction gate. The replay rehashed all 150 scientific inputs, 145 historical sources, frozen control code and completion/model records; no hashes changed. No additional fits or forwards were performed during postprocessing.

The original prepared v1 control was rejected before execution for two preflight defects: scheduler/visibility intersection and premature success publication. It is preserved and unexecuted; this was a preflight rejection, not a measured scientific failure. v2 fixed both and passed 25 synthetic preparation tests. Those tests are code checks, separate from the six real readout fits.

## Resource and backend limits

The run selected an idle **NVIDIA A800 80GB PCIe, physical ordinal 0** after inspecting visibility, scheduler flags, utilization, memory and compute workloads. Its selection snapshot showed 2 MiB used, 0% utilization and no other compute processes; no scheduler allocation or explicit CUDA visibility restricted the device set. The worker peaked at 210,857,984 allocated and 260,046,848 reserved CUDA bytes. Anonymous before/after snapshots are included in the JSON evidence; workloads were left running.

Runtime matched the recorded checkpoint backend: torch 2.5.1+cu121, CUDA 12.1, cuDNN 90701, capability 8.0, two CPU/inter-op threads, highest float32 matmul precision, matmul TF32 off, cuDNN TF32 on, benchmark/deterministic flags off, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The CPU processes ran at nice 19. Separate backend telemetry for the original G1 had not been saved; exact selected-score reproduction is the decisive direct check. These settings do not establish deterministic future CUDA training.

## Remaining scope and one next candidate

| Input | A coverage in this control | U coverage | R coverage |
|---|---|---|---|
| 2D | Thirteen-host pseudo-anatomy feature readout only; full A32 quality unestablished | No instance-quality measurement | No binding measurement |
| 3D | No new measurement; original G1 PASS unchanged | No new measurement | No new measurement |

One bounded next candidate is a **separately registered 2D-adapter-only 5× learning-rate branch**. Restart from the same original warm-start weights and seeded new 2D adapters, keep the original seed 20261019, alternating `[2d, 3d]` cycle and **2,048-step budget** (1,024 updates per mode), and change only the LR multiplier for the four new adapter tensors: `backbone.patch_embed_2d` weight/bias and `decoder.output_2d` weight/bias. Their base LR would be 5e-4 rather than 1e-4; shared and 3D parameters retain 1e-4. Apply the same original warmup/cosine factors, architecture, masking/objective, optimizer/betas/weight decay, data and crop rules. Retain the original validation boundaries through completed **step 2048** and the official **best-by-equal-mode-validation-loss selection policy**; perform official dual-mode G1 on that preregistered selected export with the original thresholds. Additionally preregister a descriptive comparison at **fixed step 1792 against original selected step 1792**. No reselection of old results, post-hoc metric choice, multiplier/seed sweep, automatic budget extension or Stage II/III release is proposed.

The rationale is limited: the four new 2D tensors began seeded while shared weights inherited 8,500 legacy steps, and the observed input/output adapter relative L2 changes were 1.735%/8.394%. This motivates one bounded optimization hypothesis; neither those norms nor this uncertain readout gain diagnose insufficient LR, establish 5× as optimal or show that it will improve quality. Larger adapter updates can cause instability or degrade shared/3D behavior. A single new trajectory also cannot establish robustness across training seeds, and the reused development cohort remains a limitation.

The candidate is **PROPOSED and not ready to launch**. It requires a focused implementation, optimizer-group/serialization/resume review, and new source/qualification registration; the current formal Stage I runner creates a single SSL parameter group. Before execution, register a finite resource/wall budget, numerical stopping rules, the retained official selection/validation policy, additional fixed descriptive endpoint and dual-mode rejection criteria. Preserve original history and hash commitments, and distinguish initialization of a new optimizer configuration from exact historical optimizer resume. No candidate code, recipe or training was implemented here.

An exposure-dose alternative was assessed but is not the primary proposal: frozen `0c0f2f8`, `anatobind/aur/unified_run.py:47`, accepts only the exact cycle `[2d, 3d]` and rejects `[2d, 2d, 2d, 3d]`. It therefore cannot be described as runnable with the unchanged qualified runner; substituting the permissive technical pilot would change the formal contract. Such a ratio also changes interleaving, total shared updates and the global-step LR schedule, even if the number of 3D exposures is held fixed.

## Reuse

The one execution registration is consumed. Do not rerun its fits or forwards. From the private v2 directory, recompute arithmetic and integrity only:

```sh
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python postprocess.py
```

[Anonymous results](aggregate_anonymous.json) and [execution/commitments](RUN.json) document the source, registration and completion hashes. The [preserved-artifact audit](../aur_g1_diagnosis/RESULTS.md) and [original G1 report](../../2026-10-20/aur_qualified_training/RESULTS.md) retain their original claims. Private features, heads, counts and exact path/hash records remain outside Git. The task worker performed a separate self-review; independent controller review is separate.
