# Brain multi-disease detector: infarct (Dataset906_ISLESInfarct), folds [0]

Fold subset: an early reading, NOT the gate (spec M4).

## Verdict

```json
{
 "kind": "early_reading",
 "folds": [
  0
 ],
 "pass": null,
 "operating_point": true,
 "sensitivity": 0.5677570093457944,
 "thr": 0.6,
 "fp_per_scan": 1.66,
 "beyond_budget": null,
 "stop_remaining_folds": false,
 "early_stop_undecided": false
}
```

Scans 50; ground-truth lesions counted 428; ignored (< 10 mm3) 33.

Scores are mean foreground probabilities over components of the argmax map, so they exceed 0.5 by construction: the rows up to threshold 0.50 are identical and the budget of 2 false positives per scan may not be reached.

## FROC

| thr | n_hit | sensitivity | FP per scan |
|---|---|---|---|
| 0.05 | 243 | 0.5678 | 1.6800 |
| 0.10 | 243 | 0.5678 | 1.6800 |
| 0.15 | 243 | 0.5678 | 1.6800 |
| 0.20 | 243 | 0.5678 | 1.6800 |
| 0.25 | 243 | 0.5678 | 1.6800 |
| 0.30 | 243 | 0.5678 | 1.6800 |
| 0.35 | 243 | 0.5678 | 1.6800 |
| 0.40 | 243 | 0.5678 | 1.6800 |
| 0.45 | 243 | 0.5678 | 1.6800 |
| 0.50 | 243 | 0.5678 | 1.6800 |
| 0.55 | 243 | 0.5678 | 1.6600 |
| 0.60 | 243 | 0.5678 | 1.6600 |
| 0.65 | 242 | 0.5654 | 1.6000 |
| 0.70 | 241 | 0.5631 | 1.5800 |
| 0.75 | 237 | 0.5537 | 1.4400 |
| 0.80 | 233 | 0.5444 | 1.3800 |
| 0.85 | 230 | 0.5374 | 1.1000 |
| 0.90 | 218 | 0.5093 | 1.0200 |
| 0.95 | 181 | 0.4229 | 0.7400 |

## Dice (report only, nnU-Net summary.json, cases with ground truth)

```json
{
 "n_cases": 49,
 "mean": 0.753277521477103,
 "median": 0.8115631691648822
}
```

## False positives per scan at the operating threshold (report only)

```json
{
 "n_scans": 50,
 "median": 1.0,
 "max": 12,
 "n_scans_over_budget": 14
}
```

## Strata at the operating threshold (report only)

### Equivalent diameter (mm)

| stratum | N (GT) | N (hit) | sensitivity |
|---|---|---|---|
| <5 | 193 | 60 | 0.3109 |
| 5-10 | 152 | 107 | 0.7039 |
| >=10 | 83 | 76 | 0.9157 |

## Binding agreement (NOT_EVIDENCE: the anatomy is a SynthSeg pseudo-label)

```json
{
 "n_pairs": 243,
 "host_agreement": 0.9465020576131687,
 "side_agreement": 0.9876543209876543,
 "n_detections": 326,
 "no_host_rate": 0.0,
 "nearest_rate": 0.003067484662576687
}
```

## Command

```
scripts/eval_brain_disease.py --disease infarct --folds 0 --workers 2 --out docs/verification/2026-09-29/brain_multidisease/infarct_fold0
```
