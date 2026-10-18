# Paired TRAIN U background-allocation tangent experiment

Status: proposed on 2026-10-18; implementation and execution await approval of this complete description. This document does not authorize training or change a model verdict.

Builder review: a fresh-context ML research/software engineer read this complete design and relevant source on 2026-10-18, with no private-file access. Outcome: "nothing important left". This is a buildability review, not implementation, numerical verification or approval by the user.

## Purpose and bounded question

Continue the completed TRAIN U supervision diagnostic from `778c14b9c067a0e71220de975f1c0fa504ecdc8d`. Its fixed twelve crops produced 113 FP components, including 100 from ISLES. Most ISLES FP component-draw records came from matched queries. Sampled negative voxels already had suppressive logit gradients, so adding an already-present negative loss or measuring its sign again would not resolve the next question.

Test one specific allocation choice: **with the same 16384 U points and per-instance focus quota, does reallocating the existing 20% boundary-ring quota to global uniform sampling give a better local shared-mask-head direction on the existing matched-query FP background?** Measure the accompanying assigned-lesion and A responses. A null result concerns this ring-to-global contrast only; it cannot reject background coverage, harder negative selection, representation, or optimization explanations generally.

Preserve A32 anatomy masks, unified abnormality U instance masks, and R13 sided host plus no-host. S stays internal context. This is a frozen-feature, restricted-parameter tangent experiment on already-used TRAIN patients, not a learned repair, historical-gradient reconstruction, independent evaluation, or clinical evidence. Default `legacy_max`, complete405val `query_cc_v1` FAIL, the A missing-class/test gap, and controlled R FAIL remain unchanged.

All choices below are the agent's proposed defaults, not previously user-selected hyperparameters. Approval covers implementing, testing and running this bounded protocol and reporting either outcome; it does not authorize a training campaign.

## Why this experiment

Source inspection establishes:

- `losses.sample_points` already exposes `focus_share` and `boundary_share`. It first draws all points uniformly, overwrites equal per-instance focus quotas, then overwrites a boundary quota when a valid ring exists. It returns the unshuffled slots. Setting only `boundary_share=0.0` therefore restores those ring slots to the same initial uniform draws for batch1 and the same seed. This needs no new mining rule.
- Production `train._forward_losses` shares points between A and U, and uses U's Hungarian assignment for Stage III R supervision. A production sampler edit would couple several mechanisms. This experiment changes only the diagnostic U point set and conditions its primary comparison on the original assignment.
- `MaskHead.embed_full` is a shared 256-to-256-to-32 MLP, 74016 parameters at the frozen default configuration. Both A and U full-resolution masks use it. R `bind` uses the separate `embed_coarse`, frozen tokens/pixels and presence. A restricted `embed_full` tangent can expose shared A effects while leaving fixed-query R structurally unchanged.
- The prior archive has sampled logits/points and component maps, but no full pixel features or query tokens. It cannot supply new-point logits or shared-parameter derivatives. A supplemental frozen forward is necessary; old completed statistics are reused as controls, not reported as newly discovered measurements.

Alternatives considered: more coverage-only draws can describe a sampling probability but cannot reveal cross-voxel parameter response; prediction-guided hard-background mining adds a new spatial weighting rule; immediate fine-tuning adds optimizer/history and held-out evaluation choices. The existing ring-to-global allocation gives the smallest directly implementable comparison. Its modest change from approximately30% to50% uniform points is an explicit limitation.

## Frozen cohort and input commitments

Reuse exactly the prior six TRAIN rows and twelve augmented crops: two BMSR/T1c, two ISLES/DWI and two PDGM/FLAIR patients, two crops each. ISLES is the primary descriptive stratum; BMSR and PDGM are separate source controls. No patient or crop is selected or replaced using this experiment's outcomes. The two ISLES patients cannot support population inference.

Reuse the approved selection rule, original full-TRAIN row positions, seed20261018, epoch0, crop128x160x160, augmentation, lesion_share0.5 and existing volume-floor/valid/ignored semantics from [the completed design](2026-10-18-aur-u-train-supervision-design.md). Data are read-only under `/data2/congcong/data/FM_data`; reject resolved paths outside it and never access the cold backup.

- Samples SHA256: `fb96acd741aa6c0ee689eb772a4332304501c8d63ebd7ccddcd657f41eb86d5d`.
- Val-patient metadata SHA256: `475f91d97d8b03f312df560f98f39ab81c068c39d0a24c632d8efa5fd52d9d34`.
- C2 StageIII step3000 / 72000crops checkpoint SHA256: `38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde`.
- Model-state anchor: `5fbb0d6f9513fdebf66831033c6a8d32710a05ce2ec900656c062a61546e482f`.
- Private selection SHA256: `7f542122cc1dd1623f616681d47b96c7003e1488718ab33f964533bfb5a42bfb`; original TRAIN position digest: `719779c337e9240e828abb1db94cb48c42e8c1c1f305624620974252de084858`.
- Previous complete archive: `/home/congcongliu/outputs/aur_u_train_supervision_20261018/run_1a7de61`. Validate its contract, COMPLETE marker and all member hashes before reuse; preserve it byte-for-byte. Recheck source-qualified TRAIN eligibility and zero val/test membership without printing private records.

Before forward measurement require the regenerated crop tensor hashes to equal the previous archive. Freeze the new execution revision, source/design hashes, environment, inputs, selection, and previous-archive member commitments. Original production model/loss/sampler/trainer/decoder files remain unchanged.

## Two paired U point sets

Each crop has the same two original draw seeds: `20261018 + 1000*selection_ordinal + 10*crop_ordinal + draw_ordinal`, with ordinals0..5,0..1,0..1. Each arm receives a fresh CPU generator reset to that seed, batch1 and the same original valid/instance maps.

| Arm | Points | Focus share | Boundary share | Other sampler behavior |
|---|---:|---:|---:|---|
| Control | 16384 | 0.5 | 0.2 | Existing implementation |
| Ring-to-global | 16384 | 0.5 | 0.0 | Existing implementation |

Control points must equal the archived indices exactly. Independently verify that focus-quota slots and all non-ring-designated slots match between arms; modified slots must be the initial uniform draws restored by omitting ring overwrite. Do not merely assume seed equality implies this property. On no-instance/no-ring crops, the two arms must be identical. A synthetic fixture must exercise each such case even if this cohort has none.

Equal focus quotas do not imply equal total positive counts: restored uniform slots can fall on lesions, ignored regions or valid-background locations. Record realized positive, ring, distant-background, ignored, effective and duplicate counts. Preserve the actual point weights and normalization; do not filter, refill or importance-weight points. This contrast changes the spatial weighting of the sampled U objective, not just the number of unique covered voxels.

The original A evaluation points remain the control points. No production global sampler setting is changed. No threshold, score, loss coefficient, number of points, patient, crop, checkpoint or optimizer is varied.

## Supplemental forward and native-value control

One frozen eval forward per crop, all64event/32anatomy queries, original CUDA bf16 autocast, original weights and no model-parameter gradients. Transiently retain detached pixel features and event/entity tokens. Evaluate each arm's sampled logits through the original sampled-mask path, then convert to float32 before any loss/sigmoid operations. Dense-gather logits are not a substitute for the sampled GEMM path.

Reproduce the previous control presence and sampled logits with `rtol=1e-5, atol=1e-5`, and require exactly equal official assignments and archived crop-level component map/rows using the unchanged decoder/source cutoffs. Record maximum numerical error. An equality failure is an incomplete control, not permission to adjust tolerance or redefine the cohort after inspection. Raw logits, features and derived quantities must be finite.

The repeated decoder is a provenance check for newly obtained features. Reuse the prior FP/TP identities and matching convention, including score-first real-first/ignored-second evaluation, original IDs, ignored instance=None, duplicate handling and crop-local targets. No new threshold sweep, candidate selection, full-volume inference, val/test image load or performance gate is included.

## Loss conditioning and the head tangent

Let phi be functional float32 copies of all four parameter tensors of `model.masks.embed_full`, initialized from the checkpoint; verify the expected74016 values. All actual model tensors remain frozen, unchanged and gradient-free. Detached full pixel features and query tokens are held fixed. No optimizer, parameter update or checkpoint export exists.

Native bf16 logits and a float32 head recomputation differ. Define an explicit native-anchored local surrogate separately at each arm's sampled points:

```text
z(phi) = z_native + [h32(phi; frozen tokens, frozen pixels)
                    - stop_gradient(h32(phi0; frozen tokens, frozen pixels))]
```

At phi0, logits are exactly the native float32-converted values; derivatives follow the float32 shared head. Implement a numerically exact zero-valued residual at phi0, verify it, and describe this as a **float32 frozen-feature head tangent anchored at native outputs**, never the historical or full bf16 training gradient. Reference CPU reconstruction must use this same defined function.

For each control draw, obtain the baseline query/target pairs from the unmodified official `event_loss` using native control logits. Hold these pairs fixed in both primary arms. Reconstruct the complete scalar U objective with those pairs: original positive/no-object presence BCE, matched weighted BCE and Dice, unmatched-empty BCE0.1, original ignored handling and matched/unmatched/sample normalization. Nothing else is dropped or reweighted. On the control, value, constituent losses and logit derivatives must agree with original `event_loss`; fixed-match gradient equivalence requires tests, not a copied formula assumed correct.

Differentiate this scalar only with respect to phi, yielding `g_U_control` and `g_U_global`. Also compute `g_A` from the unchanged original-point A objective described below, zero when A supervision is disabled. Primary directions use `g_control = g_A + g_U_control` and `g_global = g_A + g_U_global`, with the original unit A/U coefficients. This conditions on the same A contribution rather than pretending the actual shared head receives only U loss. S and fixed-query R contribute zero in this restricted subspace. Report U-only gradients/responses separately for interpretation.

Report raw gradients/norms/cosine and both raw negative-gradient and unit-L2-negative-gradient directions. A zero norm is explicitly undefined for normalized responses. Primary comparisons use unit directions to separate alignment from magnitude; raw responses remain mandatory. Because phi excludes presence parameters, its presence derivative is zero even though the scalar includes presence loss.

Separately run official matching/loss for the intervention's native points and report target-to-query switches and zero-positive/overflow counts. This secondary result must not silently replace the fixed pairs in the primary tangent, A control, R query list, or ROI definitions. A real trainer change would include these assignment effects; this experiment deliberately does not estimate them jointly.

## Response measurements and denominators

Compute the MLP-output JVP for event and anatomy tokens along each negative-gradient direction. With frozen pixel features, a voxel's first-order logit response is the embedding JVP dotted with that voxel's features. Stream reductions; retain no dense query-by-volume autograd graph. A native-anchored mean-logit response needs no finite step size, optimizer, threshold crossing or hypothetical updated decoder.

Primary regions are the **effective true-background voxels** (`valid>0.5`, `point_weight>0`, `instance==0`) inside the existing official FP components whose detector query is baseline-training-matched, excluding the union of both arms' sampled voxel indices for that paired draw. This fixed unsampled-voxel panel tests spatial transfer rather than just a response at newly supervised locations. Define it independently for both baseline point draws. Also report all eligible FP-background voxels as a secondary panel and explicit empty-panel counts; do not drop tiny components that lose all support. Query-associated assigned-target voxels are a separate region, never silently classified as background. Other-GT and ignored voxels are counted separately; an FP bbox does not imply an empty foreground mask.

For each query/draw, union its eligible FP-background voxels before averaging logit responses. This query-weighted result is primary so a fragmented query does not gain many independent votes. Also report component-weighted results and support counts. Average query values within crop/draw and retain paired crop/draw tables privately. Public source summaries expose distributions and anonymous support/sign counts only, with the two draws identified as repeated measurements. No component-level significance tests or confidence intervals pretending independent patients.

For direction d, let D_FP be the mean FP-background logit derivative and D_target the mean logit derivative on the assigned target's effective voxels. Negative D_FP indicates local suppression; negative D_target indicates local lesion weakening. Report paired differences (global minus control), fractions with each sign, undefined counts, and all denominators. Report target response for all matched targets, plus separately queries with FP-background regions and TP-associated queries. No support or score-based query exclusion is added after inspection.

A check: use the original control points and original `entity_loss`, supervision flags, all32identity queries, a_ignore and present/absent handling, with the same native-anchored tangent for A sampled logits. Differentiate the A mask loss with respect to the same phi and report its directional derivative along both combined directions and the separate U-only directions, overall and by supported query identity. Positive means local A loss increase. U-unsupervised/A-unsupervised strata must never become negative examples. Include negative controls with no enabled A supervision. This is a local loss response, not a new A Dice or missing-class repair.

R check: fixed original query lists, tokens, coarse pixels/masks, gates and geometry must yield identical logits before/after observation and in both arm contexts. Show the separate intervention-rematch count as a potential real-training R pairing change. R's zero direct phi response is a restricted-path control, not proof that a full training intervention would protect R. S/presence and the real model state must also stay unchanged.

## Interpretation fixed before results

Report counts and paired magnitudes rather than tune a success threshold from these crops. A favorable local pattern requires more suppressive unit-direction FP-background response without greater assigned-target weakening or greater A-loss increase. Summarize signs per crop/draw and per source; do not hide mixed results in a pooled mean dominated by ISLES fragments. With only two primary-source patients, even a uniform favorable sign is a candidate for a separately designed learning experiment, not adoption.

- Better coverage without a better shared-head direction: this specific reallocation did not improve the measured local optimization direction.
- Better FP suppression with target/A harm: a tradeoff, not a repair; do not select an intermediate allocation fraction in this run.
- Better restricted direction and no observed target/A harm: motivates a separately frozen model-level experiment including actual A/U/R validation and optimizer controls.
- Null, mixed or undefined support: completes this bounded experiment; it does not disprove coverage generally or authorize more patients/arms.

No result changes the existing complete-val/test decisions. All outputs remain NOT_EVIDENCE, and A32/R geometry investigations remain independent open tasks. No follow-on training, calibration, decoder change or held-out inference is included in this approval.

## Privacy, archive and resource bounds

Only the controller's fixed scripts may read private selections/manifests. Never print their values or keys; patient filenames can be keys. Subagents may review source, synthetic fixtures and anonymous public aggregates, but receive no unrestricted private-file exploration task. Any independent numeric checker uses a specified numeric-cache schema and fixed anonymous output, without opening selections/manifests. This directly addresses the two prior tool-output incidents.

Use a fresh0700directory under `/home/congcongliu/outputs/aur_u_background_allocation_20261018/`, files0600. Refuse overwrite or partial-cache resume. Preserve old archives. Per-patient/crop/query/component records, point indices, sampled logits/features, tokens, head parameter copies, ROI sufficient statistics and gradients stay private. Exceptions and progress have a fixed anonymous vocabulary; no raw arrays/identifiers in tool outputs. Public outputs are a fixed whitelist of source-level summaries, anonymous support/sign counts, numerical checks, hashes, environment, timings and limitations under `docs/verification/2026-10-18/aur_u_background_allocation/`.

Archive enough float32 numerical inputs to reconstruct native-anchored losses, both gradients/JVPs, ROI mean-logit and A responses on CPU without another image/model forward or checkpoint deserialization. Include original numeric/control and reference-archive commitments, strict schema/member hashes, finite checks and provenance validation. Feature-mean ROI sufficient statistics may replace full dense features because the declared primary response is mean logit. Validate these statistics independently against streamed dense reductions before completion. Private CPU replay must not mutate any archive bytes; no partial result earns a COMPLETE marker.

Use nvgen absolute Python, PYTHONNOUSERSITE=1 and explicit PYTHONPATH, nice19, numerical threads2, decoder workers<=2. Before a GPU launch respect scheduler/device visibility and inspect utilization, memory and active compute jobs; use a compatible idle/lightest eligible device and preserve existing jobs. Standing GPU authorization applies. No packages are installed.

Budget: one supplemental forward per twelve fixed crops, no image-based repeats after a valid complete archive. First crop is the reusable resource/control pilot. Internal deadline plus external1800second timeout cover extraction, tangent measurement and archive completion. CPU tests/replay are separate. Record actual memory/time; no source-only runtime claim. On nonfinite values, state/input/cache/control mismatch, OOM or timeout stop incomplete, without smaller crops, query truncation, fallback precision, patient replacement or new arms.

## Acceptance and handoff

1. All original/new commitments, TRAIN exclusions, twelve crops and unchanged model/old archive verified; no private identifiers in new public artifacts or fixed runner stdout. Do not call this a session-wide privacy guarantee.
2. Paired sampler equality properties verified, including no-ring, no-instance, quota rounding, ignored points and duplicates. Synthetic tests confirm that restored uniform slots may add positives.
3. Control native values, official matches, prior component map/rows and fixed-loss values/logit derivatives reproduce. Tests cover empty GT, U/A-disabled rows, overflow, unmatched-empty normalization, assignment switches, ignored/zero-weight and duplicate-point cases.
4. Float32 tangent has exact native values at phi0. Independent synthetic central finite differences of the defined surrogate validate parameter directional derivatives, shared A responses and ROI/JVP reductions; R and real model state stay equal. Test zero gradients/undefined ROIs and source/query weighting.
5. Run focused new tests plus the actual project CPU suite from CLAUDE.md, reporting environment skips. Do not repeat unchanged historical nndet/CUDA supplemental tests just to change the skip count. Pure CPU complete-cache replay reconstructs results and leaves member hashes unchanged.
6. Fresh-context source review and independent numerical/claims checks resolve substantive findings, at most three review/fix rounds. Report any block. Private input-membership verification remains the controller's fixed-script check and must not be called an independent audit.
7. Write anonymous RESULTS/RUN/aggregate, preserve raw evidence, update STATUS/CLAUDE with observed outcome and unchanged A/U/R decisions. Follow the existing local commit/main/tag handoff; no remote push is authorized by this design.

After approval of this description, write the implementation plan. Add narrow diagnostic helpers/entry point under `scripts/` and focused tests, without changing original production files or the prior diagnostic's source contract. The old runner dynamically hashes all tracked `anatobind/**/*.py`; adding even an unused module there would expand that contract. Preserve that tracked file set as well as its contents. The builder chooses exact new script/test names and numerical implementation details within these interfaces and checks.
