# AUR Residual Confidence Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; the user continuation authorizes the already recommended diagnostic. One independent whole-change code reviewer and one scientific result reviewer; no per-task implementer agents. Follow project review/fix limits of three rounds and existing localmain/tag handoff.

**Goal:** Directly measure final U-child confidence under frozen query_cc_v1 ownership, preserving A32/U/R14 behavior and the existing failed full-val verdict.
**Architecture:** An independent observer streams first-pass hashes into the unchanged QueryCCDecoder; a second identical prediction pass measures final child probability sums and verifies replay hashes. The CLI freezes original-six provenance, verifies A/U/controlledR, and emits only anonymous public aggregates.
**Tech Stack:** Existing Python/PyTorch/numpy/scipy/pytest in nvgen; no dependencies.
**Spec:** `docs/superpowers/specs/2026-10-17-aur-residual-confidence-design.md`.

## Global Constraints

- Exact previous sixval selection, C2 checkpoint38f637... and original input hashes; crop128x160x160,batch2,presence>.5,mask>=.3,26CC,10mm3,maskNMS>=.5,ownership/IDs unchanged.
- No training, production score change, new decoder profile, newtest inference, calibration or threshold sweep. Diagnostic score variants parent_score/child_raw_mean/child_fused_mean only at source floors BMSR.70/ISLES.60/PDGM.75.
- All present float32 query planes/presence/validity/window membership must replay exactly. Integrity hashes use the JSON wire representation so integer-key serialization cannot change resume checks. Reproduce parent means, original query rows/stats and legacy U; A saved scores/returned maps and55controlledR outputs/origins fail closed.
- infer.py/query_decoder.py/eval.py/model/train remain unchanged. GT is absent from observer APIs; used only after measurement for matching/control reports.
- No all-window probability caches. Private0700outputs, atomic per-row results, immutable resume contract; all raw inputs including unfinished rows checked against original frozen hashes before model loading.
- CPU nice19/threads1,decoder<=4workers; one fresh-selected compatible idleGPU under standing authorization;1800s default hard OS deadline with outer timeout,max7200s. A/Rpseudo references NOT_EVIDENCE.

## Review Focus

1. Negative/clipped windows and valid holes: child IDs and parent masks must map to the same global voxels without padding contributions.
2. Same query number in different windows, repeated callbacks and missing replay windows: preserve origin identity and fail closed.
3. Wrong parent/child masks or changed float32 values: do not silently report score deltas from a different voxel set or cast float16 cache values.
4. Resume/provenance: changed original raw bytes on an unfinished row and changed reference content must stop before inference.
5. Official ignored-GT matching and score-filter counterfactuals: masks/NMS/ownership stay fixed; predictions excluded at the original point are a separate descriptive group, not FP.

### Task1: Streaming confidence measurement

Files: create `anatobind/aur/confidence_diagnostics.py`, `tests/test_aur_confidence_diagnostics.py` only.

Interfaces:
- `ResidualConfidenceAudit(shape,voxel_mm3,workers=4)` owns an unchanged QueryCCDecoder and its failure-path cleanup (context manager).
- `.observe_first(**payload)` accepts the existing predict_volume observer keyword payload, hashes clipped float32 U planes/presence/validity, feeds the decoder without mutation.
- `.decode(fused_probability)->(instance_map,rows,stats)` finishes the original decoder and prepares canonical active-parent lookup and child fused means; returned rows/IDs remain original.
- `.observe_replay(**payload)` verifies the same observed windows/planes and measures each child's raw float64 bincount sum plus exact parent mean; validates containment/counts.
- `.finish_replay()->dict` returns `children` (instance,parent_score,child_raw_mean,child_fused_mean,parent_voxels,n_voxels,retained_fraction), `parents` (rank,counts,retained/removed raw means), and `checks` (window/plane counts, exact replay/parent/map checks). No GT fields or calibration.
- Pure helpers for scoring copies and anonymous summarization may live in the same diagnostic module; they use the three exact score names above and official matching after fixed filtering, without altering original rows.

- [x] Write hand-checked failing fixtures: a higher-score central mask removes the strongest voxels of a lower-score connected parent, leaving two children; require independently calculated parent/raw-child/fused-child means and removed-parent mean. Add full-retention/float32precision fixtures and source-score matching with ignored GT.
- [x] Run new tests RED; implement only the declared observer/state/math; run GREEN.
- [x] Add/run RED→GREEN error cases for clipped/negative window and valid holes, dtype/shape/nonfinite, duplicate/missing/changed plane/presence/validity, wrong child counts/parent containment, no children, reused query IDs across windows, and rows/map immutability. Verify observer versus direct production decoder on the same small real fixtures.
- [x] Run focused tests with CUDA hidden/nice19, inspect diff, commit owned module/tests. This is a task-level result; full project suite follows Task2 before completion claims.

### Task2: Frozen replay CLI and anonymous analysis

Files: create `scripts/aur_diagnose_confidence.py`, `tests/test_aur_confidence_replay.py`, and the fixed original-source hash manifest `docs/superpowers/specs/2026-10-17-aur-residual-confidence-sources.json`.

Consumes: Task1 class/method/result interfaces exactly. Reuse existing `diagnostics.identity/select_cases`, original frozen input-manifest construction, `freeze_contract/write/hard_deadline`, original `dataset.load_volume`, `I.predict_volume/bind_instances` and official E matching. New CLI: `--out NEW --gpu logical_index [--reference-six DIR --resume --max-seconds 1800]`; default reference-six is prior `six_1874af2`.

Produces: private contract/selection/manifests/reference content hashes/atomic row records; complete-only anonymous `aggregate.json` and COMPLETE.json with resource/verification counts. No per-case/child/origin arrays in public aggregate. First/second pass ordinary map hashes must match; old A/query/legacy/GT plus exact controlledR probabilities/choices/zero/IoU/share/origins must match reference. Skip all predicted FP binding work. Summary includes original-fixed-point TP/FP/ignored/below-threshold groups, score delta distributions, source-floor crossings, parent decomposition and three hypothetical post-ownership score filters.

- [x] Write failing tests for changed unfinished raw inputs, changed reference/selection/record hashes, mixed code/rules/checkpoint resume, exact controls including actual R origins, missing/extra child IDs, original-row immutability under score replacement, official filtered real-first/ignored-second matching, and aggregate identity-sentinel exclusion.
- [x] Run RED; implement minimal frozen prep/replay/aggregation/resume; run GREEN including Task1 tests.
- [x] Exercise runner with a small fake model only at the expensive model/I/O boundary while keeping real observer/decoder/matching/control checks; verify complete output and resumed no-inference behavior. Test hard deadline through existing tested helper rather than duplicating its implementation.
- [ ] Run project fullCPU suite once after integrated code; create exact diff/review package and independent whole-change review. Resolve substantive findings with focused regressions/re-review,max3rounds; broaden suite only after changes/failures.
- [ ] Commit final tested code before GPU freeze. Record code SHA and validate product inference/model/training files remain unchanged from starting revision.

### Task3: Execute, interpret and hand off

Files: anonymous report/data/commands under `docs/verification/2026-10-17/aur_residual_confidence/`; update STATUS.md/CLAUDE.md and this plan only after real outcomes.

- [ ] Fresh GPU/utilization/memory/workload/allocation check; launch one bounded detached replay under approved nvgen interpreter, CUDA pinning and private output. No additional GPU permission question under standing authorization.
- [ ] Require6/6original cases,83windows perpass,55controlledR checks and every raw-hash/numeric control passing; stop diagnosis on any mismatch, never relax tolerances after results.
- [ ] Analyze score inheritance hypothesis from actual raw child values, removed-parent values and predefined fused control; report counterfactual counts as six-case diagnostics only. State failures/nulls honestly; do not promote a score/profile from this probe.
- [ ] Independent scientific report review, anonymous-ID/link/hash checks, final focused validation; no new fullval/test/training unless a separate prospective contract is accepted.
- [ ] Commit outcomes, fast-forward authoritative localmain, add distinct handoff tag; preserve private outputs and existing worktree convention.
