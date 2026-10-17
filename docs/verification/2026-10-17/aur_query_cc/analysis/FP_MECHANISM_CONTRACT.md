# Optional exploratory FP morphology diagnostic

This is post-hoc exploratory error analysis, NOT_EVIDENCE. It asks whether retained FP children are concentrated in small physical-size/low-retained-fraction groups or remain large components. Associations do not establish ownership as the cause, and the script does not select scores, change inference, or recommend a loss change.

`fp_mechanism.py` imports the sibling reviewed `paired_analysis.py` loader. It first requires completed, reproduction-passing full-val shard metadata totaling 405 rows, before opening private records. The loader then verifies exact saved-val membership, hashes, inputs, rules, fixed scores, and consistent code/reference contracts. It rejects six/probe or partial-val inputs. It reads JSON only and outputs one new anonymous aggregate JSON. Existing outputs cannot be overwritten. No medical images, inference, GPU work, or threshold sweep occurs.

At BMSR/ISLES/PDGM scores 0.70/0.60/0.75, official bbox-IoU >=0.1 matching first assigns nonignored GT; official second-stage ignored-GT matching excuses leftover detections. This yields TP, FP, ignored-excused labels, cross-checked against official TP/FP counts. Groups are source/sequence, with the registered S7 sequence flagged separately.

Each outcome group reports distributions of final child voxels/mm3, parent voxels, retained fraction, and inherited score. Child-weighted and distinct-parent-weighted distributions are labeled. Predetermined retained-fraction bins are <=0.1, (0.1,0.5], >0.5; physical child size uses the original equivalent-diameter strata <5, 5–10, >=10 mm. A cross-table reports counts and shares within each outcome class. No additional score bin or operating point is selected.

Parent identities `(row, natural window, query, component)` remain in memory only. Output contains distinct parent counts, final surviving child multiplicity, same-outcome child multiplicity, and counts of parents with mixed TP/FP/excused outcomes. Outcome-specific parent counts are nonexclusive. All siblings inherit one parent score, so active parents must retain all their post-ownership, post-volume-floor final children; the script verifies metadata and that invariant.

These are final surviving children only: rejected/tiny fragments and actual pre-ownership parent shapes are unavailable. No saved post-ownership child probability mean exists, so low retained fraction cannot establish a low-probability residual or scoring error. Detection matching can assign siblings of one parent to different outcome classes. This diagnostic complements the fixed-score budget comparison and cannot establish clinical utility or causal mechanisms.

## Invocation after both shards are complete and the script is reviewed

```sh
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=/home/congcongliu/src/Foundation_Model-aur-diagnostic OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python docs/verification/2026-10-17/aur_query_cc/analysis/fp_mechanism.py --repo /home/congcongliu/src/Foundation_Model-aur-diagnostic --run /home/congcongliu/outputs/aur_query_cc_20261017/val_1874af2_s0of2 --run /home/congcongliu/outputs/aur_query_cc_20261017/val_1874af2_s1of2 --baseline-cases /data0/congcong/code/Project_Doing/foundation_model-conformance/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_val_20261014_1344/cases.json --out /tmp/aur-query-paired-analysis-20261017/val_1874af2_fp_mechanism_aggregate.json
```

This command has been prepared but not run against real validation data.

## Synthetic validation

```sh
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=/home/congcongliu/src/Foundation_Model-aur-diagnostic OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python docs/verification/2026-10-17/aur_query_cc/analysis/fp_mechanism.py --repo /home/congcongliu/src/Foundation_Model-aur-diagnostic --self-test
```

Checks cover inclusive fixed-score filtering, real-first/ignored-second matching, TP/FP/excused accounting, mixed-outcome siblings, distinct parents across rows, fraction-bin endpoints, original physical-size bins, empty detections, inconsistent sibling-score rejection, no serialized identity sentinels, 405 synthetic rows, and completion metadata accepting 203+202 while rejecting partial membership/markers. No real partial validation records are consumed by these checks.
