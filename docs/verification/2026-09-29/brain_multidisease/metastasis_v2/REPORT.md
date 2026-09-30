# Brain multi-disease detector: metastasis (Dataset905_BMSRMetastasis), folds [0, 1, 2, 3, 4]

All five folds: the line below is the gate (spec M2).

## Verdict

```json
{
 "kind": "gate",
 "folds": [
  0,
  1,
  2,
  3,
  4
 ],
 "pass": true,
 "operating_point": true,
 "sensitivity": 0.7474402730375427,
 "thr": 0.65,
 "fp_per_scan": 0.5748373101952278,
 "beyond_budget": null,
 "stop_remaining_folds": false,
 "early_stop_undecided": false
}
```

Scans 461; ground-truth lesions counted 3809; ignored (< 10 mm3) 531.

Scores are mean foreground probabilities over components of the argmax map, so they exceed 0.5 by construction: the rows up to threshold 0.50 are identical and the budget of 2 false positives per scan may not be reached.

## FROC

| thr | n_hit | sensitivity | FP per scan |
|---|---|---|---|
| 0.05 | 2847 | 0.7474 | 0.5813 |
| 0.10 | 2847 | 0.7474 | 0.5813 |
| 0.15 | 2847 | 0.7474 | 0.5813 |
| 0.20 | 2847 | 0.7474 | 0.5813 |
| 0.25 | 2847 | 0.7474 | 0.5813 |
| 0.30 | 2847 | 0.7474 | 0.5813 |
| 0.35 | 2847 | 0.7474 | 0.5813 |
| 0.40 | 2847 | 0.7474 | 0.5813 |
| 0.45 | 2847 | 0.7474 | 0.5813 |
| 0.50 | 2847 | 0.7474 | 0.5813 |
| 0.55 | 2847 | 0.7474 | 0.5813 |
| 0.60 | 2847 | 0.7474 | 0.5792 |
| 0.65 | 2847 | 0.7474 | 0.5748 |
| 0.70 | 2846 | 0.7472 | 0.5683 |
| 0.75 | 2844 | 0.7467 | 0.5575 |
| 0.80 | 2839 | 0.7453 | 0.5358 |
| 0.85 | 2822 | 0.7409 | 0.5119 |
| 0.90 | 2769 | 0.7270 | 0.4403 |
| 0.95 | 2370 | 0.6222 | 0.2408 |

## Dice (report only, nnU-Net summary.json, cases with ground truth)

```json
{
 "n_cases": 461,
 "mean": 0.8071727177852703,
 "median": 0.848458904109589
}
```

## False positives per scan at the operating threshold (report only)

```json
{
 "n_scans": 461,
 "median": 0.0,
 "max": 8,
 "n_scans_over_budget": 26
}
```

## Strata at the operating threshold (report only)

### Equivalent diameter (mm)

| stratum | N (GT) | N (hit) | sensitivity |
|---|---|---|---|
| <5 | 1936 | 1108 | 0.5723 |
| 5-10 | 1283 | 1166 | 0.9088 |
| >=10 | 590 | 573 | 0.9712 |

### Prior craniotomy, biopsy or resection

| stratum | N (GT) | N (hit) | sensitivity |
|---|---|---|---|
| no | 2839 | 2128 | 0.7496 |
| yes | 970 | 719 | 0.7412 |

## Binding agreement (NOT_EVIDENCE: the anatomy is a SynthSeg pseudo-label)

```json
{
 "n_pairs": 2847,
 "host_agreement": 0.9771689497716894,
 "side_agreement": 0.9968387776606955,
 "host_side_agreement": 0.9985950122936424,
 "n_detections": 3156,
 "no_host_rate": 0.0,
 "nearest_rate": 0.0028517110266159697,
 "unlocated_rate": 0.0009505703422053232
}
```

## Command

```
scripts/eval_brain_disease.py --disease metastasis --folds 0 1 2 3 4 --workers 8 --out docs/verification/2026-09-29/brain_multidisease/metastasis_v2 --records /data2/congcong/data/FM_data/derived/brain_disease/metastasis/records_v2
```

Code: commit c8b1a11
