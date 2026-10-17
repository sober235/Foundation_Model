# A supervision audit, 2026-10-17

Evidence status: NOT_EVIDENCE. This is a bounded training-label audit against SynthSeg pseudo-labels. No model was constructed, no inference or training was run, and the repository was not edited. Starting revision: `bbfc4ba6bfbc63f089f99a0919bea6a488d756d5`.

## Finding

The shared A/U sampler can starve small anatomy structures of positive supervision. This is a confirmed mechanism in the code and fixed TRAIN-crop measurements. It is **not an established causal explanation** for the checkpoint's five missing A labels, and the measurements specifically fail to explain the left 18 / right 54 asymmetry.

## Source trace

All source locations refer to `/home/congcongliu/src/Foundation_Model-aur-diagnostic` at the revision above.

- `anatobind/eval/lookup.py:92` and `anatobind/aur/labels.py:13,27-41`: the A label order is a fixed tuple of 32 SynthSeg labels, mapped to consecutive entity values. Labels 5, 44, 18, 54, 26, 58 are respectively zero-based query indices 6, 7, 25, 26, 28, 29. They are not dropped or merged by this A mapping.
- `anatobind/aur/heads.py:60-72`: the entity decoder has 32 fixed identity-anchored queries. `anatobind/aur/losses.py:82` compares query k directly to entity value k+1. Hungarian matching exists for U (`losses.py:101-164`), not A. Therefore U matching changes are not an A identity fix.
- `anatobind/aur/train.py:438-442` and `176-184`: one point set is sampled using the lesion instance map and reused by both A and U.
- `anatobind/aur/losses.py:29-59`: on crops with retained lesion instances, approximately 50% of the 16,384 points are split equally among lesion instances, 20% sample their immediate ring, and approximately 30% remain uniform. No entity has a guaranteed positive point quota. With no retained instances the entire point set remains uniform.
- `anatobind/aur/dataset.py:76` and `100-116`: A ignores all lesion values, including labels outside the U-visible subset. A presence is computed from the entire augmented crop before applying the lesion ignore mask.
- `anatobind/aur/losses.py:82-90`: ignored points carry no A mask loss. BCE averages every entity over the surviving points; Dice weights every entity present in the dense crop even if it has zero surviving sampled positives. Such a present entity then receives a sampled all-zero target, and both its mask BCE and Dice favor an empty mask for that draw. Presence BCE still has a positive target.
- `anatobind/aur/infer.py:150-170`: ordinary dense A inference averages entity mask probabilities and takes the winning entity above the entity threshold. It does not gate them by entity presence. The separate R coarse-geometry path does gate by presence (`model.py:52-56`). Thus low A presence alone cannot explain an entity missing from the ordinary dense A output.

## Frozen TRAIN measurement

The sample-table SHA256 and validation-patient SHA256 match the frozen diagnostic manifest: `fb96acd741aa6c0ee689eb772a4332304501c8d63ebd7ccddcd657f41eb86d5d` and `475f91d97d8b03f312df560f98f39ab81c068c39d0a24c632d8efa5fd52d9d34`.

Selection was frozen before opening label volumes: seed 20261017, two distinct TRAIN patients per source, A-supervised rows, U-visible where available, validation patients excluded. This yields eight rows, 16 actual augmented training crops of 128×160×160 voxels, 12 crops with retained lesion instances, and two fixed point draws per crop. The same crops were sampled with the current shared sampler and, as a measurement control only, uniform sampling. No sampler or loss in the repository was changed. All raw paths were checked to remain under `/data2/congcong/data/FM_data`.

Every label below was present in all 16 crops, with nonignored voxels in all of them. Across 32 point draws:

| SynthSeg label | Median nonignored crop voxels | Shared mean positive points | Shared zero-positive draws | Uniform mean positive points | Uniform zero-positive draws |
|---|---:|---:|---:|---:|---:|
| 5 | 335 | 0.938 | 14/32 | 2.406 | 6/32 |
| 44 | 341 | 1.000 | 14/32 | 1.875 | 7/32 |
| 18 | 1895.5 | 5.000 | 1/32 | 9.875 | 0/32 |
| 54 | 2141.5 | 5.188 | 1/32 | 10.625 | 0/32 |
| 26 | 661 | 1.562 | 12/32 | 3.469 | 2/32 |
| 58 | 665 | 1.750 | 5/32 | 3.656 | 0/32 |

Shared sampling spent 42.282% of all sampled points on A-ignored voxels versus 1.350% with uniform sampling. A normalizes BCE by its surviving points, so this is a reduction in spatial coverage and a change in sampling distribution, not a proportional 42% reduction of the A loss scalar.

For the 12 crops with instances alone, both 18 and 54 had **exactly the same mean 2.542 positive points and 1/24 zero-positive draws** under shared sampling. Uniform means were 9.042 and 9.792 respectively, with no zero-positive draws. This makes shared sampling a weak explanation for the observed unilateral missing label 18. A no-instance crop receives full uniform coverage and partly compensates; the audit intentionally favors U-visible rows and is not an unbiased estimate of the whole training stream.

A small synthetic call to the actual `entity_loss`, without a model, isolates the zero-positive case. Dense presence is true for query 25 (label 18), sampled targets are all zero, other query logits remain fixed at -4. Varying only this present query's logits from -4 to 0 to +4 gives A mask losses 0.5533, 1.0089, 1.1275. Its summed mask-logit gradients are positive at all three settings (0.24485, 0.03032, 0.03097), so gradient descent lowers its mask logits when the sampler missed all positives. This verifies the loss mechanism, not its historical contribution to a trained checkpoint.

## Bounded next diagnostic contract

Keep the current U decoder experiment separate. The current label-only audit is sufficient to establish a supervision candidate; it does not justify changing A loss or retraining alongside U decoding.

If a later A model audit is authorized, freeze the exact TRAIN row/crop recipe and checkpoint first, then measure all 32 queries' dense logits and sampled logits on these same crops. For 18 versus 54, report: GT positive probability, sampled positive coverage, probability on background/neighboring entities, winning-class margin at GT voxels, and fixed-query logit gradients under the original loss. This separates (a) missed positive supervision, (b) query identity/optimization collapse, and (c) winner-take-all competition. No validation or test outcomes should select crops, point quotas or loss weights. Only after that distinction should a single A sampling or loss variable be proposed, with a separate prospective comparison.

## Scope and verification

- CPU only, CUDA hidden, `nice -n 19`, two PyTorch/BLAS threads; 33.93 seconds for the eight-volume audit.
- Dataset `entity_present` was checked against dense entity counts for every measured crop; all selected rows were A-supervised TRAIN rows with zero validation-patient intersection.
- No crop arrays, image arrays, row identities, patient identities or raw-data paths are emitted in the aggregate output. Public artifacts contain aggregates and cryptographic hashes only.
- Outputs: `aggregate.json`, `audit_train_coverage.py`, this report. The aggregate includes every entity, source strata and instance/no-instance strata.
- Limitations: eight TRAIN rows / 16 crops / two point draws; U-visible selection is deliberate; no checkpoint behavior, parameter gradients, dense fit, learning dynamics or causal training effect was measured; all anatomy targets remain pseudo-labels.

Reproduction from the repository root:

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python docs/verification/2026-10-17/aur_query_cc/a_supervision_audit/audit_train_coverage.py
```
