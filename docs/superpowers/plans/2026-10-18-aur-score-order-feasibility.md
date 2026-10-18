# U score-order feasibility implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Root implements the coupled archive/evaluator interfaces; independent metadata/method audits and one whole-change review provide separate checks. User continuation authorizes bounded U research, with original A/U/R outputs and acceptance unchanged.

**Goal:** Measure an optimistic archived score-order ceiling before fitting a calibrator.
**Architecture:** A CPU-only diagnostic validates the frozen six-case corpus, reproduces its published aggregate, then rematches every exact retained-score set using the unchanged official evaluator. It emits anonymous count/sensitivity ceilings without thresholds or deployed settings.
**Tech stack:** Existing nvgen Python/numpy/scipy/pytest; no dependencies.
**Spec:** `docs/superpowers/specs/2026-10-18-aur-score-order-feasibility-design.md`.

## Global constraints

- Source corpus6b26137, original model/inference1874af2 and38source hashes unchanged; current baselinef46c019. Original405valFAIL/defaultlegacy_max preserved.
- Same six exposeddev cases,55realGT/7528children/2285parents; no training, new inference, historical-test inspection or new split.
- Scores parent_score/child_raw_mean/child_fused_mean; original order/IDs/masks/boxes/ownership/R unchanged. Inclusive cuts, equal-score groups, empty set; official real-first/ignored-second rematching at every set, no monotonic-hit assumption.
- Same-six S7 FP/scan budgets .5/1.5/0 for registered BMSR/T1c,ISLES/DWI,PDGM/FLAIR; no transported complete405budgets. Expected17,350sets includes9empty sets.
- No selected threshold, per-cut curve, score promotion or fitted parameter output. NOT_EVIDENCE and exposed-development interpretation throughout.
- CUDAhidden,CPU1/nice19,600sinner+outerdeadline,fresh0700output,atomicfiles,nooverwrite. Raw images and checkpoint tensors need not be opened.

## Review focus

1. Equal-score groups cannot be split by a stable-sort tie breaker; zero scores and the empty set must be counted.
2. Official summed-IoU Hungarian can lose admissible hits when lower-score detections are added; never reuse fixed labels or stop after one budget exceedance.
3. Source S7 counts/budgets must match the frozen six cases and committed registration, not a larger evaluation cohort.
4. Archive checksums alone are insufficient: cached control semantics, original membership and aggregate reproduction must also hold.
5. No reported threshold/identity/origin and no claims of A/R improvement or independent calibration.

### Task1: Pure bound plus frozen archive CLI

Files: create `scripts/aur_audit_score_order.py` and `tests/test_aur_score_order.py` only. Existing production/measurement/evaluation files remain unchanged.

Interfaces:
- `score_order_bound(scans, fp_budget)->dict`: scans follow E.scan; returns scans/real+ignoredGT/detection/unique-score/operating-set/feasible-set counts,integerFPcap,max_observed_hits_within_budget and optimistic sensitivity; never thresholds. Reject nonfinite/out-of-range scores or invalid budgets, handle empty scans, preserve inputs.
- `load_archive(run, anchor_path=PUBLIC_CONFIDENCE_RUN, query_reference=QUERY_SIX, query_summary=PUBLIC_QUERY_SUMMARY)->dict`: verifies hashes/selection/membership/payloads/oldreference/sourcecode and reproduces original aggregate. Returns private payloads, validated same-six reference and anonymous provenance. No model/volume entry points.
- `analyze(prepared)->dict`: evaluates all3scores/source groups using existing score_copies and official evaluator; includes original controls and explicit limitations.
- CLI `--run PATH --out NEW [--max-seconds 600]`: atomic contract/aggregate/COMPLETE only after successful preparation and counts, no resume/overwrite. Outputs anonymous schema only.

- [ ] Write and observe RED hand fixtures: TP+FP equal-score tie at zeroFP budget; empty-set/emptyinputs/zero/one boundaries; ignoredGT real-first matching; monotone scalar map preserves ceiling; nondecreasing tie merging cannot improve ceiling; input order/rows immutable. Include analytical Hungarian case GTz[0,21],[89,100],Pz[0,2.31],[0,100],[0,200], y/x[0,1], where adding lower-score P3 reduces hits2→1.
- [ ] Implement minimal pure bound and run GREEN using real official matching.
- [ ] Write RED archive fixtures using the existing synthetic six-case confidence fixture at expensive-forward boundary only: intact complete load; changed record/hash/membership/referencebudget; checksum-consistent measurement/control corruption; reproduced fixedpoint mismatch; no identity/threshold leakage and no overwrite.
- [ ] Implement archive validation/analysis/CLI; run focused GREEN and fullCPU suite. Independent whole-change code review, resolve substantive findings with RED/GREEN,maximum3rounds. Commit before analyzing actual ceiling outcomes.

### Task2: Run, interpret and hand off

Files: report/data/commands under `docs/verification/2026-10-18/aur_score_order/`; update STATUS.md/CLAUDE.md/this plan with actual results. No model or threshold setting changes.

- [ ] Readonly metadata audit and independent method audit incorporated; buildercheck says nothingimportantleft. State choices as agent choices; do not imply optimal research priority.
- [ ] Execute boundedCPU analysis from frozen code and same archived inputs; require6/55/7528/2285 and17,350sets with every reproduction/control check passing. Preserve originalrunbytes.
- [ ] Report each source and all3scores, same-six reference budgets/sensitivity, optimistic observed ceilings, calibration limitations and data exposure. Do not promote an observed optimum or infer new mask/R quality.
- [ ] Independent scientific review; JSON/hash/schema/known-identity/link checks; document test and coverage evidence.
- [ ] Commit,localmainfast-forward,distincthandofftag;preserveworktree/privateartifactsanddo notpublishremote.
