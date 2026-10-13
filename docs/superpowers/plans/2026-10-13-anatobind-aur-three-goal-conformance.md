# AnatoBind-Brain A/U/R 三目标一致性实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute task-by-task. User selected native inline execution; no subagents.

**Goal:** 修正 C2 的 Q25 门控旁路和 U 未匹配 query 空掩膜监督，同时严格保持 A/U/R 三个已批准目标及其输出定义。

**Architecture:** 将 Q25 阈值判定做成纯可测逻辑，C2 链先停在 step 1000，通过 gate 后才从同一 checkpoint 恢复；失败时 fail-closed。U loss 只给未匹配 queries 增加空 mask BCE，首轮不改推理解码。所有代码和训练输出走隔离分支，不触碰 main checkout 上正在运行的 C0。

**Tech Stack:** Python, PyTorch, Bash, pytest, torchrun.

**Spec:** `docs/superpowers/specs/2026-10-13-anatobind-aur-three-goal-conformance-design.md`

## Global Constraints

- 仅保留三个模型目标：A anatomy masks；U generic abnormality instances；R 每个 U instance 的单一主宿主。
- 不增加疾病分类 D、E、邻近/累及关系、其他器官或新的主要终点。
- A 门保持 13-host macro Dice ≥ 0.80；U 门保持同来源、S7 同误报工作点、参照灵敏度 −0.05；R G3 保持 ABA(R) ≥ B0。
- Q25 gate 只用于显式采用 Q25 的 C2 配置；readout ≥ 0.65 且 A point Dice > 0.045。缺数据、评估失败和解析失败均视为 FAIL。
- U 未匹配 query mask loss 使用 `NO_OBJECT_WEIGHT = 0.1`；训练阈值来自验证集；test 只评一次。
- 不停止、重启、覆盖或暂存 live main checkout 的 C0、checkpoint、盲读密钥或逐例数据。
- 新 C0/C2 配对训练仅在当前 C0 结束、卡资源确认可用后启动；两臂使用完全相同的新 U loss 与 Stage II/III recipe。
- 所有 SynthSeg/病灶伪标签结果继续标记 `NOT_EVIDENCE`；Level R 人工真值需要真实读者，不能用伪标签替代。

## Review Focus

1. Q25 的边界条件：readout 恰为 0.65 通过；A Dice 恰为 0.045 失败；缺失/非有限指标 fail-closed。
2. C2 gate 失败时不能进入 Stage II resume、Stage III 或 test evaluation。
3. 续跑必须保留 optimizer、scheduler、数据游标、global batch 和总预算；恢复调用不再传 `--stop-after`。
4. U mask loss 只惩罚已监督样本里的未匹配 queries；`u_supervised=False` 与 `point_weight=0` 不产生 U mask 梯度。
5. C0/C2 配对 recipe、患者划分、验证阈值选择相同；C2 未通过 early gate 时不报告 G2/G3 test 结果。

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
- Checker inputs: `STAGE2_DIR`, samples manifest, validation-patient manifest, probe GPU, and gate output directory.
- Checker outputs: durable `gate_step1000/report.json`, frozen-readout artifacts, and exit status `0=PASS`, non-zero=FAIL/error.
- All artifacts live below the current run directory; no fixed `/tmp` scratch path.

- [ ] **Step 1: Add failing tests for checker result serialization**

Test PASS/FAIL JSON fields, absent `resume_step1000.pt`, missing validation step 1000, invalid G1 report, and non-finite metrics. Each invalid state must serialize a failure and return a non-passing decision.

- [ ] **Step 2: Run the new checker tests and confirm expected failures**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_gates.py -q -p no:cacheprovider`

Expected: fail on missing result loading/serialization behavior.

- [ ] **Step 3: Refactor the Q25 checker**

Accept the Stage II run path and manifests as inputs. Export the backbone from `resume_step1000.pt`, run the existing three-seed host-only probe on the requested probe card, read `a_dice` from validation step 1000, call `q25_step1000_gate`, write the report under the run directory, and exit non-zero on every FAIL or operational error.

- [ ] **Step 4: Run checker tests**

Run the same command. Expected: all report and error-path tests pass without requiring a GPU.

- [ ] **Step 5: Commit**

`git add docs/verification/2026-10-09/anatobind_brain_ssl_first/ops/c2v2_check.sh anatobind/aur/gates.py tests/test_aur_gates.py && git commit -m "fix: make Q25 checker fail closed"`

### Task 3: Wire the gate into Stage II → Stage III chain

**Files:**
- Modify: `scripts/stage23_chain.sh`
- Create: `tests/test_aur_stage23_gate.py`

**Interfaces:**
- Q25 C2 run enables `C2_STEP1000_GATE=1`; C0 never enters this gate.
- Q25 recipe is `ARM=c2` with `II_BB=5e-5`; that combination requires `C2_STEP1000_GATE=1` or exits before training. C0 never enters this gate.
- The run script resolves repository root from its own location and accepts `FM_DATA_ROOT`, `PY`, `TORCHRUN`, and checker path overrides, so branch code can run against the shared data mount and a temporary test harness without changing live main.
- Test harness supplies temporary repo/data roots and stubbed trainer/checker commands; production defaults remain the actual trainer and Q25 checker.

- [ ] **Step 1: Write failing chain-order tests**

Test that Q25 C2 calls Stage II with `--stop-after 1000`, then the checker; FAIL exits before resume/Stage III/validation/test; PASS resumes from `resume_step1000.pt` without `--stop-after`, then reaches Stage III/evaluation. Test that Q25 C2 without the required gate flag exits before training. Test that C0 does not invoke the Q25 checker.

- [ ] **Step 2: Run chain-order tests and confirm expected failures**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_stage23_gate.py -q -p no:cacheprovider`

Expected: fail because the existing chain proceeds directly from complete Stage II to Stage III and has no Q25 gate mode.

- [ ] **Step 3: Implement the gated path**

For `ARM=c2` with Q25 gate enabled, stop at 1000, invoke the checker, exit immediately on non-zero, and resume the same Stage II run only on PASS. Keep smoke, optimizer state, scheduler schedule, data cursor, global batch, and Stage III settings unchanged. Resolve paths from the active branch and preserve existing `FM_DATA_ROOT` default.

- [ ] **Step 4: Run chain-order tests and shell syntax check**

Run the same pytest command, then `bash -n scripts/stage23_chain.sh` and `bash -n docs/verification/2026-10-09/anatobind_brain_ssl_first/ops/c2v2_check.sh`.

Expected: PASS; failure-path test observes no Stage III or test-evaluation command.

- [ ] **Step 5: Commit**

`git add scripts/stage23_chain.sh tests/test_aur_stage23_gate.py && git commit -m "fix: stop C2 chain when Q25 gate fails"`

### Task 4: Empty-mask supervision for unmatched U queries

**Files:**
- Modify: `anatobind/aur/losses.py`
- Modify: `tests/test_aur_losses.py`

**Interfaces:**
- `event_loss(...)` keeps its current return keys and matching output.
- `u_mask = existing_matched_mask_loss + NO_OBJECT_WEIGHT * mean_unmatched_empty_mask_bce` over U-supervised samples and valid weighted points only.

- [ ] **Step 1: Write failing loss tests**

Test that a high-logit unmatched mask receives a positive gradient (gradient descent lowers it); matched lesion masks preserve existing loss behavior; `u_supervised=False` samples have zero U loss; `point_weight=0` points receive zero gradient; a supervised sample with zero lesion instances still suppresses all query masks.

- [ ] **Step 2: Run the loss tests and confirm expected failures**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_losses.py -q -p no:cacheprovider`

Expected: unmatched mask gradient assertion fails under current loss.

- [ ] **Step 3: Implement the minimal unmatched-mask BCE**

For each supervised sample, calculate BCE-to-zero for unmatched query masks. Mask invalid points with `point_weight`; average point loss per query and then across unmatched queries; add with coefficient `NO_OBJECT_WEIGHT`. Leave presence loss, matching, matched-query BCE/Dice, inference merging, and thresholds unchanged.

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

- [ ] Run targeted tests from Tasks 1–4, then the full CPU suite:

`CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`

- [ ] Run `git diff --check` and inspect the final diff; confirm only implementation, tests, and status files are staged.
- [ ] Commit any test fix separately; do not include generated model outputs or private evaluation files.

### Task 7: Run the matched U-loss experiment after resource release

**Files:**
- Create/modify run records under `docs/verification/2026-10-09/anatobind_brain_ssl_first/` only.
- Training outputs remain under the configured data `ssl_runs/`; do not commit checkpoints or case-level evaluation files.

- [ ] Confirm the existing C0 chain has completed and required cards are idle; never take cards from another process. Use two cards per arm; run in parallel only with four idle cards, otherwise run sequentially with the same recipe.
- [ ] Run matched C0 (random init) and C2 (G1 init) with the new U loss, same two-card global batch, crops, Stage II/III budgets, validation split, and evaluation settings.
- [ ] Enable Q25 fail-closed gate for C2. On gate failure, stop and report; do not run Stage III or test evaluation.
- [ ] If C2 passes, complete both Stage III runs; select U thresholds on validation only; evaluate the full test split once; report A, U, and R, all pseudolabel metrics as `NOT_EVIDENCE`.
- [ ] Preserve the old-loss C0 run as historical comparison; do not claim it is matched to the new-loss pair.
- [ ] Update status with actual run IDs, gate verdicts, and verified metrics. Do not claim R0/R1 or clinical validation; Level R awaits human readers.

## Execution Handoff

Native inline execution already selected by the user. After plan approval, execute tasks in order on `codex/aur-three-goal-conformance-2026-10-13`; keep the active `main` C0 untouched.
