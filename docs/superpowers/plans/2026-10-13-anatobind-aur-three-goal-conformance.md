# AnatoBind-Brain A/U/R 三目标一致性实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute task-by-task. User selected native inline execution; no subagents.

**Goal:** 修正 C2 的 Q25 门控旁路和 U 未匹配 query 空掩膜监督，同时严格保持 A/U/R 三个已批准目标及其输出定义。

**Architecture:** 将 Q25 阈值判定做成纯可测逻辑，C2 链先停在 step 1000，通过 gate 后才从同一 checkpoint 恢复；失败时 fail-closed。U loss 只给未匹配 queries 增加空 mask BCE，首轮不改推理解码。所有代码和训练输出走隔离分支，不触碰 main checkout 上正在运行的 C0。

**Tech Stack:** Python, PyTorch, Bash, pytest, torchrun.

**Spec:** `docs/superpowers/specs/2026-10-13-anatobind-aur-three-goal-conformance-design.md`

**Review status:** 2026-10-13 plan review split the legacy Q25 gate from the new U-loss profile; awaiting user review of this corrected plan.

**Scope note (2026-10-14):** This remains the 3D model-correction subplan. The complete user-facing target—single-slice / 3D-series brain MRI DICOM input, A/U/R display, and a public interactive HTTPS URL—is specified in [the DICOM/web product plan](2026-10-14-anatobind-brain-dicom-web-execution.md). Passing this subplan alone does not deliver that product.

## Global Constraints

- 仅保留三个模型目标：A 输出 32 个 anatomy masks（13-host macro Dice 为门，32 类均报告）；U 将所有来源已标注病灶并为一个 abnormality 类并输出实例 masks；R 为每个 U instance 输出一个 13 类分侧 host 或 no-host。
- 不增加疾病分类 D、E、邻近/累及关系、其他器官或新的主要终点。
- A 门保持 13-host macro Dice ≥ 0.80；U 门保持同来源、S7 同误报工作点、参照灵敏度 −0.05；R G3 保持 ABA(R) ≥ B0。
- G1/Q25 是预训练/执行门，不是第四目标；G1 通过不代表 G2-A 通过。
- U loss profile 与 Q25 gate 分开配置并记录：`U_MASK_PROFILE=legacy|unmatched_empty_v1`。Q25 gate 只用于 legacy C2；readout ≥ 0.65 且 A point Dice > 0.045。缺数据、评估失败和解析失败均视为 FAIL。
- `unmatched_empty_v1` 使用 `NO_OBJECT_WEIGHT = 0.1`，不继承 legacy 的 step-1000 早停阈值；按既定 G2-A/G2-U/G3 最终门评估。任何新早停门须在 run 前另行预注册。
- 训练数据根固定为 `FM_DATA_ROOT=/data2/congcong/data/FM_data`；样本清单为 `$FM_DATA_ROOT/derived/aur/samples_1mm_v1.json`，验证患者清单为 `$FM_DATA_ROOT/derived/aur/ssl_manifest_v1/val_patients.json`。配对实验两臂使用相同文件，并在启动前记录两份清单的 SHA-256。
- 训练阈值来自验证集；test 只评一次。
- 不停止、重启、覆盖或暂存 live main checkout 的 C0、checkpoint、盲读密钥或逐例数据。
- 新 C0/C2 配对训练仅在当前 C0 结束、卡资源确认可用后启动；两臂使用完全相同的新 U loss 与 Stage II/III recipe。
- 所有 SynthSeg/病灶伪标签结果继续标记 `NOT_EVIDENCE`；Level R 人工真值需要真实读者，不能用伪标签替代。

## Review Focus

1. Q25 的边界条件：readout 恰为 0.65 通过；A Dice 恰为 0.045 失败；缺失/非有限指标 fail-closed。
2. `U_MASK_PROFILE=legacy` 的 Q25 C2 gate 失败时保留 step-1000 checkpoint、验证记录和失败报告；不恢复 Stage II、不进入 Stage III、不做最终验证集阈值选择或 test evaluation。step-1000 验证记录是 gate 输入，不能跳过；`unmatched_empty_v1` 不调用旧 Q25 gate。
3. 续跑必须保留 optimizer、scheduler、数据游标、global batch 和总预算；恢复调用不再传 `--stop-after`。
4. U mask loss 只惩罚已监督样本里的未匹配 queries；`u_supervised=False` 与 `point_weight=0` 不产生 U mask 梯度。
5. 报告严格覆盖目标输出：A 32 类分项与 13-host gate；U 各来源按 S7 一对一 IoU≥0.1 matching 在参照 FP/scan 工作点评估；R 每个 U instance 一个 14 类概率向量（13 host + no-host），报告 G3 controlled ABA。C0/C2 配对 recipe、患者划分和验证阈值选择相同。

---

### Task 1: Q25 gate 判定函数

**Files:**
- Create: `anatobind/aur/gates.py`
- Create: `tests/test_aur_gates.py`

**Interfaces:**
- Produces: `q25_step1000_gate(readout: float | None, a_dice: float | None) -> dict`
- Exact return keys: `passed: bool`, `readout_passed: bool`, `a_dice_passed: bool`, `failures: list[str]`, and `thresholds: {readout_min: 0.65, a_dice_min_exclusive: 0.045}`.
- `None`, NaN, infinity, or malformed values fail closed and produce a failure reason.

- [ ] **Step 1: Write failing boundary tests**

Add tests for: `(0.65, 0.0451) -> PASS`; `(0.65, 0.045) -> FAIL`; `(0.6499, 0.046) -> FAIL`; missing/non-finite inputs -> FAIL.

- [ ] **Step 2: Run gate tests and confirm expected failures**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_gates.py -q -p no:cacheprovider`

Expected: fail because `q25_step1000_gate` is not implemented.

- [ ] **Step 3: Implement the pure gate function**

Use inclusive `readout >= 0.65`; use strict `a_dice > 0.045`; reject non-finite and missing values. Do not add configurable thresholds in this Q25 helper.

- [ ] **Step 4: Re-run gate tests**

Run the same command. Expected: all gate boundary tests pass.

- [ ] **Step 5: Commit**

`git add anatobind/aur/gates.py tests/test_aur_gates.py && git commit -m "feat: add fail-closed Q25 gate decision"`

### Task 2: Parameterized Q25 checkpoint checker

**Files:**
- Modify: `docs/verification/2026-10-09/anatobind_brain_ssl_first/ops/c2v2_check.sh`
- Modify: `anatobind/aur/gates.py`
- Modify: `tests/test_aur_gates.py`

**Interfaces:**
- Checker environment inputs: required `STAGE2_DIR`, `SAMPLES`, `VAL_PATIENTS`, `PROBE_GPU`; optional `PY` defaults to the repo's nvgen Python. `GATE_OUT` is fixed to `$STAGE2_DIR/gate_step1000`; all outputs must stay under that run directory.
- Checker outputs: durable `gate_step1000/report.json`, frozen-readout artifacts, and exit status `0=PASS`, non-zero=FAIL/error.
- All artifacts live below the current run directory; no fixed `/tmp` scratch path.

- [ ] **Step 1: Add failing tests for checker result serialization**

Test PASS/FAIL JSON fields, absent `resume_step1000.pt`, missing validation step 1000, invalid G1 report, and non-finite metrics. Each invalid state must serialize a failure and return a non-passing decision.

- [ ] **Step 2: Run the new checker tests and confirm expected failures**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_gates.py -q -p no:cacheprovider`

Expected: fail on missing result loading/serialization behavior.

- [ ] **Step 3: Refactor the Q25 checker**

Read `$STAGE2_DIR/resume_step1000.pt` and `$STAGE2_DIR/val.jsonl`; require `SAMPLES`, `VAL_PATIENTS`, and `PROBE_GPU`. Use optional `PY` or default to the repo's nvgen Python. Export the backbone to a fresh path under `$GATE_OUT`, run the existing three-seed host-only probe on `PROBE_GPU`, read `a_dice` at step 1000, call `q25_step1000_gate`, write `$GATE_OUT/report.json`, and exit non-zero on every FAIL or operational error.

- [ ] **Step 4: Run checker tests**

Run the same command. Expected: all report and error-path tests pass without requiring a GPU.

- [ ] **Step 5: Commit**

`git add docs/verification/2026-10-09/anatobind_brain_ssl_first/ops/c2v2_check.sh anatobind/aur/gates.py tests/test_aur_gates.py && git commit -m "fix: make Q25 checker fail closed"`

### Task 3: Wire the gate into Stage II → Stage III chain

**Files:**
- Modify: `scripts/stage23_chain.sh`
- Create: `tests/test_aur_stage23_gate.py`

**Interfaces:**
- U loss profile is explicit and required by the chain: `U_MASK_PROFILE=legacy|unmatched_empty_v1`; both C0 and C2 in a comparison use the same loss profile. The chain passes it to `scripts/aur_train.py` and each run config records it. Direct trainer invocations default to `legacy` for backward compatibility.
- Q25 recipe is `ARM=c2`, `U_MASK_PROFILE=legacy`, `II_BB=5e-5`; it requires `C2_STEP1000_GATE=1` or exits before training. C0 never enters this gate.
- `U_MASK_PROFILE=unmatched_empty_v1` rejects `C2_STEP1000_GATE=1`; it uses final G2-A/G2-U/G3 gates unless a separately reviewed early-stop contract is added.
- The run script resolves repository root from its own location and accepts `FM_DATA_ROOT`, `PY`, `TORCHRUN`, `SEED`, `PROBE_CARD`, and `Q25_CHECKER` overrides, so branch code can run against the shared data mount and a temporary test harness without changing live main. It passes `SEED` and `U_MASK_PROFILE` through to both stages.
- Test harness supplies temporary repo/data roots and stubbed trainer/checker commands; production defaults remain the actual trainer and Q25 checker.

- [ ] **Step 1: Write failing chain-order tests**

Test that `U_MASK_PROFILE=legacy` C2 with the Q25 recipe calls Stage II with `--stop-after 1000`, produces the step-1000 validation record and `resume_step1000.pt`, then invokes the checker with `STAGE2_DIR`, `SAMPLES`, `VAL_PATIENTS`, `PROBE_GPU`, `PY`, and `GATE_OUT=$STAGE2_DIR/gate_step1000`. On FAIL, preserve those artifacts and exit before resumed Stage II, Stage III, final validation threshold selection, or test evaluation. On PASS, resume from `resume_step1000.pt` without `--stop-after`, then reach Stage III/evaluation. Test that legacy Q25 C2 without the required gate flag exits before training; `unmatched_empty_v1` rejects the Q25 flag; C0 never invokes the Q25 checker; and both arms pass the requested seed and U profile into trainer run config.

- [ ] **Step 2: Run chain-order tests and confirm expected failures**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_stage23_gate.py -q -p no:cacheprovider`

Expected: fail because the existing chain proceeds directly from complete Stage II to Stage III and has no Q25 gate mode.

- [ ] **Step 3: Implement the gated path**

For `ARM=c2`, apply the step-1000 gate only when `U_MASK_PROFILE=legacy` and the Q25 recipe is selected; stop at 1000, invoke `Q25_CHECKER` with `STAGE2_DIR`, `SAMPLES`, `VAL_PATIENTS`, `PROBE_GPU`, `PY`, and `GATE_OUT="$STAGE2_DIR/gate_step1000"`, exit immediately on non-zero, and resume the same Stage II run only on PASS. Reject that gate flag for `unmatched_empty_v1`. Keep smoke, optimizer state, scheduler schedule, data cursor, global batch, and Stage III settings unchanged. Resolve paths from the active branch and preserve existing `FM_DATA_ROOT` default.

- [ ] **Step 4: Run chain-order tests and shell syntax check**

Run the same pytest command, then `bash -n scripts/stage23_chain.sh` and `bash -n docs/verification/2026-10-09/anatobind_brain_ssl_first/ops/c2v2_check.sh`.

Expected: PASS; failure-path test observes no Stage III or test-evaluation command.

- [ ] **Step 5: Commit**

`git add scripts/stage23_chain.sh tests/test_aur_stage23_gate.py && git commit -m "fix: stop C2 chain when Q25 gate fails"`

### Task 4: Empty-mask supervision for unmatched U queries

**Files:**
- Modify: `anatobind/aur/losses.py`
- Modify: `anatobind/aur/train.py`
- Modify: `scripts/aur_train.py`
- Modify: `tests/test_aur_losses.py`

**Interfaces:**
- Exact loss API: `event_loss(presence, mask_logits, targets, point_weight, u_supervised, u_mask_profile="legacy") -> (loss_dict, matches)`. It keeps the current return keys and matching output; reject profile values outside `legacy|unmatched_empty_v1`.
- Add `--u-mask-profile {legacy,unmatched_empty_v1}` to `scripts/aur_train.py`; `scripts/stage23_chain.sh` requires `U_MASK_PROFILE` and passes the profile explicitly. Record the selected value at `run_config.json["config"]["u_mask_profile"]`.
- `legacy` preserves current loss exactly. `unmatched_empty_v1` uses `u_mask = existing_matched_mask_loss + NO_OBJECT_WEIGHT * mean_unmatched_empty_mask_bce`; compute per supervised sample, mean BCE over valid points per unmatched query, then mean over that sample's unmatched queries and supervised samples.

- [ ] **Step 1: Write failing loss tests**

Test that `U_MASK_PROFILE=legacy` is unchanged; `unmatched_empty_v1` gives a high-logit unmatched mask a positive gradient (gradient descent lowers it); matched lesion masks preserve existing loss behavior; `u_supervised=False` samples have zero U loss; `point_weight=0` points receive zero gradient; a supervised sample with zero lesion instances still suppresses all query masks.

- [ ] **Step 2: Run the loss tests and confirm expected failures**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_losses.py -q -p no:cacheprovider`

Expected: unmatched mask gradient assertion fails under current loss.

- [ ] **Step 3: Implement the minimal unmatched-mask BCE**

Thread `cfg["u_mask_profile"]` through `train.py` into the final `u_mask_profile` argument of `event_loss(...)`; serialize it under `run_config.json["config"]["u_mask_profile"]`. For `unmatched_empty_v1`, calculate BCE-to-zero for unmatched query masks. Mask invalid points with `point_weight`; average point loss per query, then across unmatched queries per sample, then across supervised samples; add with coefficient `NO_OBJECT_WEIGHT`. For `U_MASK_PROFILE=legacy`, preserve current loss exactly. Leave presence loss, matching, matched-query BCE/Dice, inference merging, and thresholds unchanged.

- [ ] **Step 4: Run loss tests**

Run the same command. Expected: all matched/unmatched/supervision-mask assertions pass.

- [ ] **Step 5: Commit**

`git add anatobind/aur/losses.py tests/test_aur_losses.py && git commit -m "fix: supervise unmatched U queries as empty masks"`

### Task 5: Synchronize current status with the three-goal contract

**Files:**
- Modify: `STATUS.md`

- [ ] Rewrite `STATUS.md`'s five current-state sections using the latest observed C0 state; distinguish the live old-loss C0 from any new-loss paired runs and retain historical records.
- [ ] Record Q25 fail-closed semantics, the historical C2 stop-rule deviation, the U loss change, and the pseudolabel `NOT_EVIDENCE` boundary.
- [ ] Preserve historical records and all untracked case/key files; do not stage run outputs.
- [ ] Review `git diff -- STATUS.md`; verify no secrets, case tables, or reader key files are included.
- [ ] Commit: `git add STATUS.md && git commit -m "docs: sync AUR status with approved goals"`

### Task 6: Verify the code branch

**Files:**
- Run existing test suite; no additional source changes unless a failure requires a focused fix.

- [ ] Run targeted tests from Tasks 1–4, verify the R output/inference contracts (`tests/test_aur_relation.py`, `tests/test_aur_infer.py`, `tests/test_aur_eval.py`), then the full CPU suite:

`CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`

- [ ] Run `git diff --check` and inspect the final diff; confirm only implementation, tests, and status files are staged.
- [ ] Commit any test fix separately; do not include generated model outputs or private evaluation files.

### Task 7: Run the matched U-loss experiment after resource release

**Files:**
- Create/modify run records under `docs/verification/2026-10-09/anatobind_brain_ssl_first/` only.
- Training outputs remain under the configured data `ssl_runs/`; do not commit checkpoints or case-level evaluation files.

- [ ] Confirm the existing C0 chain has completed and required cards are idle; never take cards from another process. Use two cards per arm; run in parallel only with four idle cards, otherwise run sequentially with the same recipe.
- [ ] Freeze and record this exact pair recipe before launch:
  - Data root: `FM_DATA_ROOT=/data2/congcong/data/FM_data`; samples: `$FM_DATA_ROOT/derived/aur/samples_1mm_v1.json`; validation patients: `$FM_DATA_ROOT/derived/aur/ssl_manifest_v1/val_patients.json`; run outputs: `$FM_DATA_ROOT/derived/aur/ssl_runs`. Record SHA-256 for both manifests and the C2 backbone before either arm starts; both arms must read those same bytes. Create one unique `PAIR_ID=$(date +%Y%m%d_%H%M%S)`; record `git rev-parse HEAD`, the hashes, and selected card IDs in a new pair contract record.
  - Shared training settings: seed 0; two cards; global batch 24; microbatch 12/card; grad accumulation 1; crop `128x160x160`; 2 crops/volume; 24 workers/rank; Stage II 240,000 seen crops; Stage III 80,000 seen crops; validation every 500 steps over 32 volumes. Use the same source code SHA and settings for both arms.
  - Stage II recipe: `II_BB=5e-5` as the script-scale backbone rate; with global batch 24 the script applies `sqrt(24/16)`, giving effective backbone/head rates 6.124e-5 / 6.124e-4 and a 667-step warmup. Stage III effective backbone/head/relation rates are 6.124e-5 / 3.062e-4 / 6.124e-4, with a 133-step warmup.
  - C0: random initialization. C2: initialize from `/data2/congcong/data/FM_data/derived/aur/ssl_runs/stage1_r2_320k_20261010_1303/ssl_stage1_best.pt`; use `docs/verification/2026-10-09/anatobind_brain_ssl_first/g1/r2_run_20261010_1942/g1_report.json`.
  - Set `U_MASK_PROFILE=unmatched_empty_v1` and `C2_STEP1000_GATE=0` for both arms. The only intentional difference is initialization.
- [ ] Launch both arms from the active branch root through `scripts/stage23_chain.sh`; confirm card IDs are idle first and export `C0_CARDS` / `C2_CARDS` to the verified IDs. For sequential runs, set both variables to the same verified pair and wait for C0 to finish before starting C2. For parallel runs, use four idle cards, set the variables to disjoint pairs, start each command in a separate session, and wait for both chains to finish:

```bash
export FM_DATA_ROOT=/data2/congcong/data/FM_data
export PY=/home/congcongliu/anaconda3/envs/nvgen/bin/python
export TORCHRUN=/home/congcongliu/anaconda3/envs/nvgen/bin/torchrun
export U_MASK_PROFILE=unmatched_empty_v1 C2_STEP1000_GATE=0
export GLOBAL=24 MB=12 WORKERS=24 II_BB=5e-5 SEED=0
export PAIR_ID=$(date +%Y%m%d_%H%M%S)
PAIR_DIR=docs/verification/2026-10-09/anatobind_brain_ssl_first/pair_unmatched_empty_$PAIR_ID
mkdir "$PAIR_DIR"
git rev-parse HEAD > "$PAIR_DIR/code_sha.txt"
sha256sum "$FM_DATA_ROOT/derived/aur/samples_1mm_v1.json" "$FM_DATA_ROOT/derived/aur/ssl_manifest_v1/val_patients.json" /data2/congcong/data/FM_data/derived/aur/ssl_runs/stage1_r2_320k_20261010_1303/ssl_stage1_best.pt > "$PAIR_DIR/input_sha256.txt"
# Export C0_CARDS and C2_CARDS to the verified card IDs before running these commands.
printf 'pair_id=%s\nC0_CARDS=%s\nC2_CARDS=%s\nprofile=%s\nGLOBAL=%s\nMB=%s\nWORKERS=%s\nII_BB=%s\nSEED=%s\n' "$PAIR_ID" "$C0_CARDS" "$C2_CARDS" "$U_MASK_PROFILE" "$GLOBAL" "$MB" "$WORKERS" "$II_BB" "$SEED" > "$PAIR_DIR/contract.txt"

ARM=c0 CARDS="$C0_CARDS" bash scripts/stage23_chain.sh
ARM=c2 CARDS="$C2_CARDS" INIT_BACKBONE=/data2/congcong/data/FM_data/derived/aur/ssl_runs/stage1_r2_320k_20261010_1303/ssl_stage1_best.pt G1_REPORT=docs/verification/2026-10-09/anatobind_brain_ssl_first/g1/r2_run_20261010_1942/g1_report.json bash scripts/stage23_chain.sh
```

- [ ] Do not apply the legacy Q25 step-1000 gate to the new U-loss pair. Apply only the predeclared final G2-A/G2-U/G3 gates; a separate early-stop gate requires a separate preregistration before launch.
- [ ] Complete both Stage III runs; select U thresholds on validation only; evaluate the full test split once. For G3 controlled ABA, provide each ground-truth test lesion instance to R and compare against B0 using the model's predicted anatomy; keep predicted-U end-to-end R results report-only. Report all 32 A structures plus the 13-host gate; per-source U sensitivity using S7 matching/FP operating points; one 14-class R host/no-host probability vector per instance. Mark all pseudolabel metrics `NOT_EVIDENCE`.
- [ ] After both test evaluations, set `C2_TEST_DIR` and `C0_TEST_DIR` to the exact test-output paths recorded in their chain logs. Set `COMPARE_DIR=docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/compare_unmatched_empty_$PAIR_ID`, fail if it already exists, then run `$PY scripts/aur_compare.py --a "$C2_TEST_DIR" --b "$C0_TEST_DIR" --name-a C2 --name-b C0 --n-boot 2000 --out "$COMPARE_DIR"`. It uses 2,000 patient-level bootstrap draws (seed 0); treat it as descriptive only. The comparison's U rows use each arm's own validation-selected threshold; the G2-U verdict remains the separate S7 reference-FP operating-point gate.
- [ ] After both arms and the paired comparison finish, if any final A/U/R gate fails, preserve the complete reports, mark that goal NOT PASSED, and stop this plan. Do not tune on test or make an unregistered change; a follow-up requires a new reviewed plan.
- [ ] Preserve the old-loss C0 run as historical comparison; do not claim it is matched to the new-loss pair.
- [ ] Update status with actual run IDs, gate verdicts, and verified metrics. Do not claim R0/R1 or clinical validation; Level R awaits human readers.

## Execution Handoff

Native inline execution already selected by the user. After plan approval, execute tasks in order on `codex/aur-three-goal-conformance-2026-10-13`; keep the active `main` C0 untouched.
