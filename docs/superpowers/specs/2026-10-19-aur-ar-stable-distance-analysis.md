# Frozen A/R cache: stable-distance CPU analysis contract

This is a numerical instrumentation follow-up under the user's 2026-10-19 authorization. It neither changes the production relation geometry nor trains/chooses a model. The original three TRAIN forwards have finished; their native float32 replay remains FAIL and is not reclassified.

## Observed failure and hypothesis

Only nearest-host-distance/100 (geometry feature4) exceeded the predeclared CPU/CUDA replay tolerance in three elements of one crop: maximum absolute difference 3.95291e-5. All stored R logits matched that tolerance (maximum difference5.48363e-6). On the same stored float32 centroids, the CPU MM distance expansion differs from float64 direct differences by2.36878e-5; direct computation differs by5.37504e-8. This identifies cancellation in squared-norm expansion, not a token, weight or GT change. These numbers are parsed from the private CPU debug record; the original cache remains immutable.

## Frozen follow-up method

- No new medical image/checkpoint load, no backbone forward, no GPU, and no parameter update. Read only the already captured probability, token, validity, coordinate and frozen relation tensors.
- Snapshot collection-time source bytes and verify their old hashes before editing analysis code. Preserve all nine native members and their hashes, the failed native check, and native geo/logits exactly.
- Create a new private output directory and a separately versioned derived-analysis contract. It references the original source commitments and explicitly says `native_replay_passed=false`. An absent original COMPLETE marker must never be converted into an original successful run.
- Retain the original centroid/probability/valid/presence definitions. Replace only the algebraically equivalent Euclidean-distance evaluation with explicit float64 coordinate differences and norm, then cast the distance feature back to the original R-head dtype. Apply the same stable computation to both coarse and dense geometries.
- Keep the same tokens, fixed all-query set and R weights. Record numerical deltas from the native measurements and the coarse-to-dense contrast separately. Do not tune the method based on host correctness or downstream score.
- Do not change any predeclared tolerance or default production behavior. New CPU replay must reproduce the derived results without images/models or CUDA. Validate the direct distance against hand-computed nearby points at large absolute coordinates, corruption/missing-member refusal, and exact token/weight/source invariance.
- Only anonymous summaries may enter Git. Per-crop/query arrays stay under protected home directories. Conclusions remain local TRAIN mechanism observations and NOT_EVIDENCE; no new A/U/R performance verdict.
