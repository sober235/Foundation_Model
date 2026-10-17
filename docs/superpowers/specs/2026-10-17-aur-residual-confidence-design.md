# AUR residual-instance confidence diagnostic

## Authorization and scope

The user said `continue` after the completed query-CC comparison and its recommendation to investigate U residual confidence. Continue that bounded diagnostic under the existing A32/U-instance/R14 architecture. Data/method details below are agent implementation choices, not new user-selected hyperparameters. No training, production score/profile change, threshold search, new test inference, or expansion to UI/API work.

Starting revision: 2d97f56; both linked worktrees clean, no CodeGraph index. Reuse the existing diagnostic worktree. Previous complete405val remains a failed fixed-profile result; this six-case mechanism replay cannot replace it or establish population utility. All outputs remain NOT_EVIDENCE.

## Question and hypothesis

Does inheriting a connected parent's score numerically overstate its surviving child masks' own probabilities, or do the residual FP masks already have high raw probabilities? Small retained fraction alone cannot answer this. Measure the actual float32 probabilities on the exact final child voxels, with no GT choosing query/window/mask.

Primary comparisons for every final query_cc_v1 child, before source-score filtering:

1. Original parent component score (float32 sigmoid values, float64 original reduction).
2. Child raw probability mean from that same natural window and query (float32 input, float64 bincount reduction).
3. Reference control: mean of the unchanged legacy fused probability map on those exact child voxels (float32 input, float64 bincount reduction).

Do not change candidate priority, NMS, ownership, floor, masks, IDs, provenance or R. Secondary hypothetical score filtering compares these three fixed scores at the existing BMSR0.70/ISLES0.60/PDGM0.75 points, with official one-to-one bbox-IoU>=0.1 real-first/ignored-second matching. No re-NMS/backfill/reordering or threshold sweep. This is a diagnostic counterfactual after frozen ownership, not a new deployable decoder or calibrated model.

## Frozen data and baselines

- Replay exactly the original six validation rows in `/home/congcongliu/outputs/aur_query_cc_20261017/six_1874af2`: two distinct patients each for BMSR/T1c, ISLES/DWI, PDGM/FLAIR. No reselection using new outcomes.55nonignoredGT/83naturalwindows are the expected previous totals.
- Verify the source COMPLETE/contract/selection/row-record set, its original manifest digest and every selected current raw input against the original raw hashes BEFORE loading a model. Validate these rows against the original saved405val and the frozen sample/split hashes.
- Checkpoint SHA25638f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde; sample table SHA256fb96acd741aa6c0ee689eb772a4332304501c8d63ebd7ccddcd657f41eb86d5d. Raw data only under /data2/congcong/data/FM_data.
- Freeze new code SHA, checkpoint/sample/split/raw hashes, prior six contract/record content hashes, rules, score points, crop128x160x160, batch2, window order, and method choices in a new private output contract. Resume requires exactly the same contract; completed row records are atomic. Never overwrite previous runs.
- Previous floating caches are insufficient for exact child means; raw per-query float32 values must be observed again. These are development-val mechanism observations, not fresh validation, calibration data or independent holdout evidence.

## Two-pass measurement

Use existing inference observers and QueryCCDecoder unchanged; add a separate diagnostic module/CLI only.

Pass1: call ordinary predict_volume with an external observer that feeds the existing QueryCCDecoder and hashes each natural window's presence vector, clipped valid mask and every present query's float32 probability plane. Retain hashes, packed candidate masks and the completed global instance map/rows; do not retain all window probability planes. Finish using the exact existing decoder. Build a canonical-rank to active parent lookup from the decoder's frozen candidate list only after pending workers drain.

Verify exact query rows/stats, legacy detections, A saved scores and window count against the previous six query replay. GT is used only for later matching/reporting and controlled R, never by the measurement observer. Ordinary A/legacy-U map hashes and sequence/presence outputs are retained to check pass2.

Pass2: rerun the same volume/window order/batch2/autocast, hash every corresponding observed input again and fail on any missing/duplicate/changed window, presence, validity or query probability plane. For each active parent, verify the original parent mask's raw mean exactly reproduces its original score. Use global child IDs within the clipped window and float64 bincount sums to measure each child's mean. Confirm complete child counts, positive volumes, containment in the original parent/window/valid region, and unchanged input rows/map. Every retained child must be measured exactly once.

Additionally decompose each active parent's probabilities into all final child voxels versus removed voxels. Removed combines ownership loss and post-ownership floor loss; do not claim these two causes are separated. Parent-level counts must not be confused with child-weighted summaries.

Pass2 must reproduce pass1's returned A label map, returned A max-probability map, legacy U map, sequence and entity-presence values exactly. This checks only returned objects, not unreturned A32 probability tensors. Rebind the55controlled GT instances once using the unchanged bind_instances path; compare saved host probabilities, choices, IoU/zero/share plus actual query/window origins to the prior query replay. Do not rerun thousands of predicted FP bindings or claim new end-to-end R performance.

## Measurement outputs and interpretation

- Private per-child measurements/origins/GT associations and per-window hashes stay local under a0700run directory. Do not save image or all-query probability arrays. Public files contain anonymous source/class aggregates and hashes only.
- Primary grouping uses the original parent-score operating point: TP, FP, ignored-excused, below-original-threshold. Descriptive retained-fraction bins<=.1,(.1,.5],>.5 and original physical size strata are frozen. Parent counts are nonexclusive if their children have mixed outcomes.
- Report score differences (parent minus raw child, parent minus fused child), count/mean/median/range/quantiles, original-floor crossing counts, parent retained/removed probability decomposition and the three fixed-point U counterfactuals. No grouping or threshold selected to maximize a result.
- A high parent/child correlation or an improved six-case counterfactual does not prove formal U repair. A null parent-to-child gap weakens score-inheritance as a numerical explanation, but does not eliminate mask quality or calibration problems. Fused-map scoring is a predefined measurement control, not an adopted scoring rule. Raw probability means are not calibrated correctness probabilities.
- Preserve the prior three-goal verdict and A/R limitations, including pseudo-reference status, nonrepresentative six-case sample, and missing baseline voxel maps where applicable. The new replay can verify pass1/pass2 maps exactly but cannot retrospectively compare unrecorded old maps.

## Resources, interfaces and acceptance

- One currently idle compatible GPU, selected after fresh utilization/memory/workload and scheduler-visibility checks. Existing workloads remain running. CPU nice19, OMP/BLAS/MKL1, decoder at most4workers. Default1800s hard OS deadline including setup; external timeout as well, maximum7200s. Bounded per-query copies; no arbitrary candidate cap.
- New pure diagnostic module owns measurement state/math; new CLI owns frozen provenance, model replay, controls, atomic records and anonymous aggregation. infer.py/query_decoder.py/eval.py and model/training behavior remain unchanged.
- Meaningful synthetic RED/GREEN checks: parent/child score differences after ownership, clipping/valid holes, fixed full-retention scores, mixed same-query/window identities, float32/64 precision, map/row immutability, missing/duplicate/drifting replay inputs, parent/child containment/count failures, exact original-hash provenance on resume, and source-score matching without GT in inference.
- Independent builder check before implementation, focused independent code review before GPU execution, final independent scientific result review, at most3automatic fix rounds per scope. Existing user continuation and standing GPU authorization cover this work; only genuinely missing scope decisions require clarification.
- Completion requires6/6rows,55controlledbindings,83windows per pass, all declared fail-closed reproduction checks passing, measured score comparisons, resource/hash/command records, updated STATUS/CLAUDE and localmain/tag handoff. A failing score hypothesis is a valid outcome; tooling alone is not completion.
