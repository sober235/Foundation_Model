# S4 brain anatomy on fastMRI FLAIR: evaluation (NOT_EVIDENCE: agreement with SynthSeg pseudo-labels)

Verdict (spec A11): **fail** — host agreement 0.9215481171548117, mean host Dice 0.2664787808434745, outline Dice 0.9780942600738576 (gates 0.9 / 0.8 / 0.97).

The final judgement of S4 waits for the Level R reader labels (A12); the lowest two slices are reported, not judged.

## fastMRI stacks (reliable slices only)

```json
{
 "n_stacks": 433,
 "host_agreement": {
  "n_lesions": 1267,
  "n_evaluated": 956,
  "n_agree": 881,
  "n_outside_reliable": 311,
  "rate": 0.9215481171548117
 },
 "mean_host_dice": 0.2664787808434745
}
```

| class | mean Dice over stacks |
|---|---|
| white_matter_left | 0.8334 |
| white_matter_right | 0.8360 |
| cortex_left | 0.7798 |
| cortex_right | 0.7787 |
| thalamus_left | 0.0240 |
| thalamus_right | 0.0031 |
| basal_ganglia_left | 0.1345 |
| basal_ganglia_right | 0.0748 |
| brainstem | 0.0000 |
| cerebellum_left | 0.0000 |
| cerebellum_right | 0.0000 |
| other_deep_grey_left | 0.0000 |
| other_deep_grey_right | 0.0000 |

## Outline model (its test stacks)

```json
{
 "n_test_stacks": 88,
 "mean": 0.9780942600738576,
 "min": 0.827219457376786
}
```

## Lowest two slices (student output, median labelled area in cm2; no reliable reference there)

```json
{
 "0": 145.4,
 "1": 149.3
}
```

## Simulated test set (A13, report only)

```json
{
 "per_class": {
  "white_matter_left": 0.8495822296944994,
  "white_matter_right": 0.8467430224673661,
  "cortex_left": 0.7887509861937266,
  "cortex_right": 0.7908076867515893,
  "thalamus_left": 0.8051652758311318,
  "thalamus_right": 0.8255732086335263,
  "basal_ganglia_left": 0.7796113622029323,
  "basal_ganglia_right": 0.7966750538716206,
  "brainstem": 0.3535009767164611,
  "cerebellum_left": 0.4521404146315384,
  "cerebellum_right": 0.4078847338195166,
  "other_deep_grey_left": 0.43893067020226817,
  "other_deep_grey_right": 0.45882774036984525
 },
 "mean_host_dice": 0.6610917970296939,
 "n_cases": 285,
 "ventricles": 0.8485568585364752
}
```

## Command

```
scripts/eval_brain_anatomy.py --out docs/verification/2026-10-02/brain_anatomy_flair/eval --work /data2/congcong/data/FM_data/derived/brain_anatomy/eval_20261004_0233 --gpu 7
```

Code: commit f220389
