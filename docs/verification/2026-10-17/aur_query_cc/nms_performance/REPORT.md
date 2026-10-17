# Exact NMS vector-bound probe

Only temporary copies in this directory were changed. No raw data, model inference,
GPU operations, repository edits, or commits were used. CPU execution was nice 19,
PyTorch/OMP/MKL/OpenBLAS threads limited to at most four. Each benchmark invocation
had a 115-second signal alarm and completed well within that limit.

## Finding

The scalar finish loop invokes `_mask_iou` for every previous survivor, even when
the existing safe bbox/volume bound immediately rejects that pair. Applying that
same bound in a vector operation eliminates most Python calls when bounding boxes
are spatially separated. This is a measured synthetic optimization opportunity,
not proof that this loop explains the live validation wall time.

Latest finish timings (seconds; single runs on a shared host):

| Synthetic geometry | Candidates | Survivors | Scalar | Vector | Speedup | Exact-IoU helper calls, scalar → vector |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Disjoint solid | 2400 | 2400 | 5.273 | 0.375 | 14.07× | 2878800 → 0 |
| Clustered solid | 1200 | 1168 | 1.458 | 0.163 | 8.93× | 694456 → 32 |
| Clustered sparse | 1200 | 1196 | 1.661 | 0.405 | 4.10× | 716106 → 1247 |
| Nested solid | 1200 | 10 | 0.072 | 0.077 | 0.93× | 3994 → 1190 |
| Equal-box connected sparse | 1000 | 965 | 11.162 | 11.231 | 0.99× | 473124 → 473124 |

All five scenarios produced exactly equal suppression traces (including exact
IoUs and earliest suppressor ranks), final instance arrays, rows, and statistics.
The last case deliberately defeats bbox pruning. An earlier equally sized
equal-box case with nearly identical masks suppressed to one survivor and took
0.027 → 0.047 seconds: vector overhead can dominate very cheap NMS workloads.

## Patch and semantics

`vector_bound.patch` changes only finish's suppression loop. Canonical sorting,
original packed masks, `_mask_iou`, inclusive .5 suppression, ownership, residual
splitting/flooring, scoring, profiles, and candidate counts are unchanged.
`np.flatnonzero` preserves ascending survivor indices, so the earliest qualifying
suppressor is unchanged. Pairs whose safe upper bound is exactly the threshold
are retained for the original exact computation.

Empty candidate lists and empty survivor slices work: vector arrays have empty
leading dimensions and no scalar pairs are visited. Nonempty candidates come
from positive-sized connected components satisfying a positive voxel floor;
zero-voxel candidates are outside the existing candidate-generation contract.
Zero bbox intersections produce zero upper bounds and cannot suppress positive
volume candidates.

All geometry and volume arrays use int64 rather than int32. Like the original
`np.prod` bound, this assumes physically representable in-memory masks. More
precisely the proposed bound's `3 * upper` and volume sum must fit int64; the
former permits upper bounds up to 3.07e18 voxels. This is vastly beyond practical
in-memory medical volumes, but no arbitrary-integer overflow guard was added.
The patch is not an arbitrary-precision geometry API. Extra persistent geometry
storage is 56 bytes per candidate (about 0.8 MB at 14286 candidates), plus
temporary vectors proportional to the number of survivors. Pairwise complexity
remains quadratic in the worst case; exact mask comparisons remain unaccelerated
when bounding boxes and volume ratios cannot rule them out.

The copied original file is byte-identical to revision `4f9fdbf`:
SHA256 `1af288b3d526bc179b608124ec4f186032494a49beca5c9431a7acb5e5d5c7a2`.
`git apply --check vector_bound.patch` passed against an isolated matching tree.

## Verification

The draft regression contains an independent copied scalar finish oracle, three
randomized candidate fixtures with tied scores and exact-half/earliest-suppressor
fixtures, and a deterministic work-pruning assertion for 80 disjoint candidates.
That assertion failed as expected against the original decoder; it passes with
the vector source. The full existing decoder tests plus the draft passed:
**27 passed in 3.40s**. Pytest emitted one harmless cache-write warning because
its inferred root cache location was read-only; test execution succeeded.

The draft is self-contained and imports the normal project decoder by default.
For isolated use set `TEST_DECODER_SOURCE` to the temporary vector source.
The existing suite was run using temporary import substitution, without editing
the repository source.

Reproduce CPU synthetic benchmark:

```sh
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python docs/verification/2026-10-17/aur_query_cc/nms_performance/benchmark_standalone.py --repo /home/congcongliu/src/Foundation_Model-aur-diagnostic --output /tmp/aur-nms-reproduction.json
```

Machine-readable measurements: `benchmark_results.json`.
Regression draft: `test_vector_nms_regression_draft.py`.
Reviewable source: `query_decoder_vector.py`.

Acceptance on actual validation data still requires a subsequent live run or
saved-candidate replay. This probe establishes exact equivalence on synthetic
fixtures and strong acceleration for separated boxes, not end-to-end gains.

The archived standalone benchmark loads the scalar source from git revision 4f9fdbf and imports the current checkout decoder. It preserves all frozen synthetic fixtures, refuses output overwrite, and sets a 115-second bound. Compilation and CLI checks passed; timings above are the original measured runs, not an additional rerun. Scratch patch/module files remain local; the implemented equivalent bound is commit1874af2.
