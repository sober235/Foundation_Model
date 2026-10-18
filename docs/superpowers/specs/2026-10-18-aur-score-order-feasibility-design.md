# U score-order feasibility before calibration

## Intent and authorization

The user asked to continue after the three-goal review. Continue the bounded U scoring investigation while retaining A32 identity masks, unified annotated-abnormality instance masks, and per-instance R13 sided host plus no-host. These implementation choices are the agent's calls, not user-selected hyperparameters. No model training, new inference, production profile, patient split change or new test inspection.

Starting revision f46c019, clean main and existing diagnostic worktree; no CodeGraph index. Reuse the worktree. Existing complete405val FAIL and default legacy_max remain. Independent review found no goal-definition drift; choosing a U-only controlled experiment does not imply A/R have improved.

## Question and alternatives

Can the existing score ordering attain useful detection coverage within the frozen FP budget on the already exposed six development cases? A strictly increasing transformation of a final scalar preserves all threshold-retained sets. A nondecreasing transformation can merge levels and remove sets, but creates none. This reasoning is per source, and excludes voxel-level probability transformations, new features and nonmonotone models.

1. Fitting a calibrator immediately would require TRAIN outputs and a defensible evaluation contract; it cannot fix the final-scalar ordering by itself.
2. **Chosen:** enumerate the archived score-retained sets and calculate their optimistic observed hit ceiling under the matching six-case S7 budgets. This can reject a scalar-order explanation cheaply, without new data or parameters.
3. A/R diagnostics remain separate candidate experiments; this choice does not establish that U scoring has the globally optimal priority.

This is explicitly threshold-set enumeration for descriptive feasibility, not threshold selection for deployment. No selected threshold, per-cut curve, optimum score choice or fitted calibration parameter is saved. Six exposed cases cannot establish efficacy, calibration or generalization. All results remain NOT_EVIDENCE.

## Frozen inputs and controls

- Archived observations: `/home/congcongliu/outputs/aur_residual_confidence_20261017/replay_6b26137`; original inference code1874af2, measurement code6b26137, unchanged C2 checkpoint38f637... . Public immutable anchor: residual-confidence RUN.json and its aggregate hash.
- Verify original contract/completion/manifest/aggregate hashes, original selection commitment, exact6record membership, record corpus and payload checksums. Validate cached A/U/R/window/parent controls using the recorded measurement contract and original query-six reference records. This reads historical observations; it never invokes the replay/resume entry point or changes its frozen contract.
- Verify the38frozen model/inference/measurement source hashes. Rebuild the previous anonymous aggregate from records and exactly reproduce every declared original fixed-point parent/raw/fused result before computing new outcomes.
- Exact6cases,55nonignoredGT,7528children,2285activeparents,83windows/815presentplanes per historical pass and55controlledR are expected. Child identities are unique within each case; do not deduplicate case-local IDs globally.
- Frozen reference: original `six_1874af2/reference_frozen.json`, SHA256 `0c3ef43fda4237800d0cb6835210ef34bd98fe47de8c45e46c12820ff2a74312`. Anchor chain: committed `six_probe_accelerated.json` → original contract → reference hash; reference summaries must also equal the committed per-source S7 blocks. Use the same selected patients, not transported complete405budgets.

| Source/sequence | Scans/GT | S7 FP/scan budget | Integer FP cap | S7 sensitivity |
|---|---:|---:|---:|---:|
| BMSR/T1c | 2/13 | 0.5 | 1 | 8/13 |
| ISLES/DWI | 2/40 | 1.5 | 3 | 18/40 |
| PDGM/FLAIR | 2/2 | 0.0 | 0 | 2/2 |

## Exact enumeration

Use all three already declared scores: parent_score, child_raw_mean, child_fused_mean. Copy rows with existing score_copies; preserve order, original instance ID, box, voxel/physical volume and detector_origin. Do not use filter_instances, which compacts IDs. Geometry/NMS/ownership/volume filtering and all A/R values remain archived constants.

For each source and score, evaluate the inclusive `score >= threshold` set at every exact distinct finite score in [0,1], plus an empty-set point above the maximum. Equal scores are inseparable. For each point recompute official real-first/ignored-second Hungarian bbox-IoU>=0.1 matching. Do not relabel fixed TP/FP groups, interpolate, stop at the first budget exceedance or assume monotonic hits/FP: official Hungarian maximizes summed IoU before rejecting sub-0.1 pairs.

Return counts of scores/operating sets, scans, real/ignored GT, detections and feasible sets, plus maximum observed hits/sensitivity within the source budget. Never return a maximizing threshold or selected detections. Exactly17,350 source/score sets including9empty sets are expected from the frozen corpus. Empty cases and score-zero/one edges remain valid.

## Patient exposure and interpretation

Metadata audit: U-supervised TRAIN761unique patients/1216rows, val86/143, historical test212/354; PDGM FLAIR/T1c share patients. TRAIN calibration folds constrain only the calibrator because the base model already saw TRAIN. Val/six are exposed development data; historical test has been inspected. Existing metadata establishes source-qualified split disjointness, not cross-source biological identity. Planned SibBMS10 includes7TRAIN/3historicalTEST patients; no fresh independent evaluation cohort was established.

Document the metadata sources/hashes and these limits. A low ceiling rules out only monotone remapping of that archived scalar on this sample; it does not rule out better masks, changed rankings or calibration utility at fixed numerical thresholds. A high ceiling is optimistic and does not validate a new operating point. No rescored end-to-end R results are produced.

## Resources and acceptance

CPU only, CUDA hidden, nice19, numerical threads1, no new dependencies, default600s outer+inner deadline. Atomic output in a fresh0700private directory; no overwrite/resume facility needed for this small analysis. Report only anonymous aggregates, source hashes and counts; never patient/case/GT/query/origin tables or threshold values.

Tests must protect equal-score groups, empty-set/empty-input behavior, ignored GT, inclusive cutoff edges, immutable input rows, strict monotone and tied nondecreasing maps, and actual nonmonotone official matching. Archive tests reject changed hashes, changed membership and checksum-consistent semantic corruption before analysis, and require published fixed-point reproduction. Full CPU suite and independent code/scientific review before localmain/tag handoff, maximum3review/fix rounds per scope. A null or adverse result is a completed diagnostic, not an adopted repair.
