# Paired frozen-decoder analysis contract

Parent question: at frozen weights and the pre-fixed numeric scores, does query_cc_v1 alter detection coverage and false positives, and which eligible GT gain or lose end-to-end correct pseudo-host assignments? This is an auxiliary modeling experiment, NOT_EVIDENCE. A32/U/R13+no-host targets are unchanged.

The script reads saved JSON only. It does not load medical images, build a model, run inference, access GPUs, train, scan score thresholds, or launch full validation. Use nvgen Python, hidden GPUs, nice 19, and four CPU threads. Its output is a new aggregate JSON file; no patient/case identifiers, boxes, paths to input records, origins, or per-GT tables are serialized. Existing outputs cannot be overwritten.

Inputs must be completed runner directories with COMPLETE.json, contract.json, private selection/input manifests, reference_frozen.json, and all private row_records. It rejects incomplete runs, mixed frozen code/rules/reference contracts, duplicate membership, changed input manifests, wrong baseline hashes, and unregistered binding floors. Complete val requires the exact full saved-val row membership; supply every completed shard using repeated --run. The six probe requires its six completed rows. This script never analyzes incomplete source outcomes.

U uses the official one-to-one bbox-IoU >=0.1 matching after inclusive score filtering. Ignore GT are excluded from sensitivity and excuse matched leftover predictions under the official rule. The fixed scores are BMSR 0.70, ISLES 0.60, PDGM 0.75. Per-source/sequence output contains hits, FP, ignored GT, retained/excused predictions, sensitivity, FP/scan, paired rescue/harm counts, and the repository's volume-defined size strata. S7 budgets apply only to BMSR/T1c, ISLES/DWI, PDGM/FLAIR and their frozen complete scan sets. Other sequences are descriptive. This compares the same pre-fixed numeric thresholds, not equal observed FP: raw parent-component scores and fused-map means have different calibration. No threshold selection or sweep is performed. Failure at these fixed scores is not evidence that every calibration of query-preserving decoding must fail.

A invariance means exact JSON numerical equality (same-position NaNs equal, zero tolerance) of saved A dictionaries, not unrecorded voxelwise maps. Controlled R compares the exact saved instance membership and all baseline fields except decoder-dependent matched_det/matched_score and newly added binding_origin. Legacy cases.json lacks query/window origins, so exact controlled-origin invariance cannot be claimed. A or controlled-R exactness failure gives exit code 2 and invalidates causal interpretation; the anonymous failure counts remain visible.

R always reuses legacy predictions through the runner's fixed_legacy_e2e, which changes only cached GT associations after fixed-score filtering. The script independently recomputes official associations and verifies cached truth for every retained binding. The all-eligible-controlled-GT denominator includes detection misses and unbound matched GT as end-to-end failures. Coverage, bound coverage, and correct-host success over all eligible GT are explicit. Paired all-GT success transitions measure decoder-system rescue/harm. The intersection matched and bound by both decoders has separate paired correct-host transitions and R-versus-B0 counts. The intersection is conditional and must not be used to claim unconditional R accuracy improvement. No GT assignment is inferred from the predicted host.

Resource outputs contain aggregate candidate/NMS/ownership/residual counts, retained fractions, recorded row time distribution, total windows, maximum recorded CUDA allocated memory, and CPU analysis elapsed time/RSS. These are query-replay resources. Historical saved legacy timing is not a paired timing comparator; summed shard row times are not wall time. CUDA allocated peak excludes reserved memory and total device footprint.

Stop after the frozen completed comparison. Do not tune to six or val outcomes. Six-case results support mechanism and feasibility only; they cannot establish source-wide utility. Full validation still uses pseudo references, including approximate AUR-resampled versus native S7 grid comparisons for BMSR/ISLES. No Level R or clinical claim follows.

## Exact six invocation

```sh
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=/home/congcongliu/src/Foundation_Model-aur-diagnostic OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python docs/verification/2026-10-17/aur_query_cc/analysis/paired_analysis.py --repo /home/congcongliu/src/Foundation_Model-aur-diagnostic --run /home/congcongliu/outputs/aur_query_cc_20261017/six_4f9fdbf --baseline-cases /data0/congcong/code/Project_Doing/foundation_model-conformance/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_val_20261014_1344/cases.json --out /tmp/aur-query-paired-analysis-20261017/six_4f9fdbf_final_aggregate.json
```

For a later complete run, preserve the command's environment and replace --run with that run's completed directory. For full val shards, repeat --run once per complete shard and use a new output filename. Shards with different code SHA are rejected even when their algorithm is claimed equivalent.

## Synthetic acceptance check

```sh
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=/home/congcongliu/src/Foundation_Model-aur-diagnostic OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python docs/verification/2026-10-17/aur_query_cc/analysis/paired_analysis.py --repo /home/congcongliu/src/Foundation_Model-aur-diagnostic --self-test
```

Synthetic checks cover low-score perfect-box stealing fixed by rematching, threshold equality, ignored-GT excusal, true FP accounting, different detected subsets, all-GT versus common-match R rescue/harm, 1e-12 exactness failures, binding-floor mismatch rejection, and identity sentinels absent from serialized aggregates. The complete six acceptance check verifies six exact A score records and all 55 shared controlled-R records; origins remain unverified because absent in baseline.
