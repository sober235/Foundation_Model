Read-only coarse-mask supervision trace, 2026-10-17

Scope and verification
Repository /home/congcongliu/src/Foundation_Model-aur-diagnostic at
1874af20a460cf86c46029cc83d2df0801a10910. No GPU, raw volumes,
model forward/backward, training, or repository edits were used. The pre-existing
modified plan and untracked diagnostic directory were preserved.
All source line references below are relative to this repository.

Conclusion
There is no direct A/U mask supervision, distillation, or consistency target for
masks.embed_coarse. It is a separate learned projection, frozen at its random
initialization throughout Stage II, then trained by R classification/hinge
gradients in Stage III. Its spatial logits are not constrained to agree with the
supervised dense mask logits. This is a verified structural mismatch, not proof
that it causes all R errors. Do not call the shared coarse pixel pyramid untrained:
it does receive A/U loss gradients through the dense branch.

Interfaces and loss path
- heads.py:93-101: embed_full and embed_coarse are separate MLPs. The full MLP
  ends in 32 channels, the coarse MLP in 64. Neither shares its MLP weights.
- heads.py:103-111: pix.coarse is p, an FPN-like lateral/smooth combination of
  all four backbone levels; pix.full is fuse(concat(up(p), img(image))). Thus
  A/U supervision of pix.full also optimizes p, lateral, smooth and backbone.
- heads.py:113-124: coarse logits are embed_coarse(query) dot p; dense/point
  logits are embed_full(query) dot pix.full. Coarse is not downsampled dense.
- model.py:41-45 and train.py:175-185: both supervised A and U mask losses use
  full_masks at sampled full-resolution voxel points. No coarse mask enters them.
- losses.py:75-98: A BCE over all entities, Dice for present entities, presence
  BCE; lesion voxels ignored and a_supervised gates the sample.
- losses.py:101-120: U Hungarian assignment uses full-mask point logits and
  event presence under no_grad. Assignment itself has no gradient.
- losses.py:123-164: U presence over all queries; matched dense masks get BCE
  and Dice; actual unmatched_empty_v1 additionally gives unmatched dense masks
  NO_OBJECT_WEIGHT=0.1 BCE to empty on U-supervised samples. This change does
  not supervise any coarse mask.
- train.py:188-201: Stage III invokes bind only on Hungarian-matched queries
  of r_supervised samples with nonempty matches. Since U-unsupervised samples
  produce matches=None, they also get no R loss. Presence threshold is not
  used to decide training R queries.
- losses.py:171-186: R is host cross-entropy plus weighted hard-negative hinge;
  total is A + S + U + lambda_r*(R_host + lambda_h*R_hard). There is no other
  coarse, dense/coarse consistency, or distillation loss in this total.
- model.py:47-59: bind computes coarse entity sigmoid probabilities, multiplies
  entity-presence sigmoid, max-merges entities into host masks, computes coarse
  event probabilities for selected queries, applies F1 valid mask, computes geo,
  and sends geo plus the unchanged host/event/sequence embeddings to relation.
- relation.py:26-41,90-99: host tokens use mean entity embeddings; host masks
  use max probabilities; CandidateCompetition concatenates host/event/sequence
  embeddings and geometry before learned competition.

Stage/optimizer semantics
- Stage I is StageOne, containing backbone plus masked-patch decoder/projector,
  not AnatoBindBrain's A/U/R heads (ssl/model.py:23-38). Objective is masked
  reconstruction plus contrast_weight*contrast (ssl/model.py:40-55).
  Actual Stage I r2 run_config.json:15 has contrast_weight=0.0. Thus this
  backbone lineage has masked-image learning but zero-weight InfoNCE semantic
  contribution. ssl/train.py:140 uses AdamW(model.parameters()).
- Only backbone is exported/transferred (ssl/checkpoint.py:67-86;
  train.py:133-135,358-361). Stage I has no embed_coarse/full to transfer.
- Stage II: train.py:75-90 freezes relation, backbone.mask_token, embed_coarse;
  frozen parameters are omitted from optimizer and from zero-touch. Other
  backbone/head parameters train. Backbone is learning-rate-scaled, not frozen.
- Stage III: train.py:91-96 enables requires_grad for every parameter, with
  backbone, heads, relation groups. embed_coarse is in heads, not relation group.
  train.py:153-157,362-369 loads the whole Stage II model strictly, then creates
  a new AdamW optimizer (the inter-stage load is not optimizer-state resume).
- train.py:203-204 adds p*0 for all trainable parameters. This produces zero
  gradients for otherwise unused parameters, not mask supervision. AdamW can
  still apply decay to these trainable parameters. A non-None gradient therefore
  does not demonstrate a meaningful task gradient. train.py:442-454 backprops,
  clips to 1.0 and steps optimizer/scheduler.
- backbone.mask_token is unused in II/III forward: model.py:35 provides no
  visible argument; swin.py:183-191 uses it only when visible is provided.

Which semantic R gradients can flow (nonzero values are not guaranteed)
  R -> CandidateCompetition -> geo -> embed_coarse + pix.coarse -> shared
  lateral/smooth pyramid -> backbone.
  R -> geo -> entity presence gate -> entities.presence and entity decoder.
  R -> geo -> selected event embeddings -> event decoder.
  R -> host/event/seq token inputs -> entity/event decoders and sequence.proj
  -> backbone. Query self-attention can couple queries; do not interpret
  selected-query input as only its learned query parameter receiving gradients.
  No direct R semantic gradient goes through embed_full, masks.up/img/fuse,
  events.presence, or sequence.cls. Those train through A/U/S respectively;
  shared downstream parameter updates can still alter their later outputs.
  Geometry is only partly differentiable (relation.py:50-73): soft overlap,
  centroids, displacement and side offset differentiate; host membership
  h>0.5, host present h_mass>0.5, the overlap>0.05 distance-zero switch and
  conditional branches do not provide gradients through their decisions.
  Distance can differentiate through the event centroid, but the thresholded
  set of host voxel coordinates is not differentiable with respect to host mass.
  Entity-to-host amax routes gradients to maximizing entities.

Actual C2 config and checkpoint agreement
Let RUNS=/data2/congcong/data/FM_data/derived/aur/ssl_runs.
II=RUNS/c2_stage2_240k_20261014_1344;
III=RUNS/c2_stage3_80k_20261014_1344.
II/run_config.json:3-42 records stage II, seed 0, unmatched_empty_v1, 240,000
crops, backbone LR 6.124e-5 and heads 6.124e-4, warmup and backbone warm
667 steps, backbone warm scale 0.1. Lines 47-54 record world 2/global24,
10,000 steps, total22,719,625/trainable20,589,832. Its selected Stage II
checkpoint is step10,000/240,000 crops.
III/run_config.json:3-43 records stage III, same profile/seed, stage-II best
input, 80,000-crop budget, LR backbone6.124e-5, heads3.062e-4,
relation6.124e-4, warmup133, no extra backbone warm phase. Lines48-55 show
world2/global24 and all22,719,625 parameters trainable. Config.init='backbone'
is an inherited CLI field only: run_config.json:70-71 and model loading in
train.py:362-363 correctly identify the actual initialization as stage2.
These rates agree with scripts/stage23_chain.sh:75-80 using sqrt(24/16)
scaling, II_BB=5e-5, and warmup exposure scaling. They differ from bare
Python defaults by explicit supported CLI overrides, not a config/code conflict.
III/summary.json:3-4 records completed3334 steps/80,016 crops. best.json:3-4
and actual checkpoint metadata select step3000/72,000 crops, not final3334.

CPU checkpoint inspection (CUDA_VISIBLE_DEVICES='', no forward/backward):
- II SHA256 c78413dcaac18de3d5b09e30c868ced346b3207c95097620dce35d76affd733d
- III SHA256 38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde
  matches frozen diagnostic RUN.json checksum.
- II embed_coarse: all4 tensors bitwise identical to freshly constructed
  seed0 AnatoBindBrain(use_checkpoint=False); relation all37 also identical.
- III embed_coarse differs from II; max absolute tensor change0.04983770475.
  Therefore frozen-in-II and trained-in-III are supported by actual weights.
- embed_full/lateral/smooth changed in II and again in III.
- Param-group counts regenerated from actual checkpoint config:
  II backbone12,929,792 + heads7,660,040 =20,589,832.
  III backbone12,929,856 + heads7,742,280 + relation2,047,489 =22,719,625.
- II and III checkpoint metadata code_sha is77fd6ebc81ec945f77f7c9316f05119294b1aca7.
  git diff77fd6ebc..HEAD over model.py,heads.py,train.py,losses.py,relation.py,
  ssl/model.py,ssl/checkpoint.py and stage23_chain.sh returned no changes.

Focused next counterfactual
Use the exact frozen crop/out/selected query and fixed model. Change only
the geometry-source factor: current coarse A/U probabilities versus predicted
dense A/U probabilities for that same query mapped to the same F1 grid.
Preserve entity-presence gate, host merging, valid mask, physical coordinates,
all host/event/sequence token embeddings and CandidateCompetition weights.
Use fixed patch-aligned average pooling (patch2x4x4) for dense probabilities,
not nearest hard-mask subsampling that can erase small U. Treat the pooling
rule as fixed, document it, and do not retune thresholds or query selection.
Recompute geo only and pass original embeddings plus swapped geo to relation.
Record paired host logits/probabilities, choice flips, and individual geometry
features. This joint geometry-source swap tests whether the dense perception
representation changes R for fixed tokens; it does not separate A versus U
effects. If separation is needed afterward, keep the same protocol and swap
A-only or U-only in separate one-factor runs. A null result cannot by itself
exclude geometry importance because a frozen head was trained on coarse
geometry; a positive result does not establish clinical correctness.
Use no GT mask/host as model input. Existing55 pseudo-host references can be
used only for descriptive diagnostic agreement, explicitly NOT_EVIDENCE.
No changes to U decoding, loss, training or weights belong in this test.

Reproducible focused verification
Script: docs/verification/2026-10-17/aur_query_cc/r_coarse_supervision/verify_cpu.py
Anonymous structured result:
  /tmp/aur-r-coarse-supervision-trace-20261017/verification.json
Exact successful command (run from any directory):
  nice -n 19 env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/nvgen/bin/python docs/verification/2026-10-17/aur_query_cc/r_coarse_supervision/verify_cpu.py --repo /home/congcongliu/src/Foundation_Model-aur-diagnostic --out /tmp/aur-r-coarse-supervision-trace-20261017/verification.json
The JSON is written only after every assertion passes. No patient/sample
identifiers, hostname, volume content, model inputs, forward or backward pass
are emitted. torch intra/inter-op threads are each explicitly bounded to one.
Checkpoint config and run JSON values are compared after JSON normalization:
the checkpoint represents crop as a tuple and the JSON sidecar as a list.
This is a serialization difference, not a training configuration disagreement.
Historical-source equality is checked against current HEAD, and the relevant
source files must also have no uncommitted differences. Unrelated working-tree
changes are neither rejected nor modified.
