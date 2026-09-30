# nnDetection arm: inference smoke (plan Task 9 Step 4)

Volume: `file_brain_AXFLAIR_201_6002917`, the h5 of S2's inference smoke (`docs/verification/2026-09-28/brain_detector/infer_smoke.md`):
one of the 130 FLAIR studies whose fastMRI+ rows carry the small-vessel-disease global label but no small-lesion boxes. It is
not one of Task903's 253 cases, so fold 0 never saw it. Model: fold 0, `model_last.ckpt`, default postprocessing
(`plan_inference.pkl`). GPU 7 was idle when the run started (2026-09-30 09:57).

## Command

```
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_lesions_nndet.py \
  --h5 /data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/brain/multicoil_train/file_brain_AXFLAIR_201_6002917.h5 \
  --out /data2/congcong/data/FM_data/derived/nndet_runs/infer_smoke --fold 0 --gpu 7
```

Exit code 0 at 09:57:32. Last lines of the run (nnDetection's own progress bars omitted):

```
2026-09-30 09:57:30.670 | INFO     | nndet.inference.predictor:predict_case:187 - Prediction took 10.056577494833618 s
wrote /data2/congcong/data/FM_data/derived/nndet_runs/infer_smoke/pred.json: 1 cases, 156 boxes
156 lesions -> /data2/congcong/data/FM_data/derived/nndet_runs/infer_smoke/lesions.json
```

## Table check (`infer_smoke/lesions.json`)

Read back with:

```
PYTHONNOUSERSITE=1 ~/anaconda3/envs/nvgen/bin/python - <<'PY'
import json
rows = json.load(open('/data2/congcong/data/FM_data/derived/nndet_runs/infer_smoke/lesions.json'))
print('rows', len(rows), 'type', type(rows).__name__)
print('key sets', sorted({tuple(sorted(r.keys())) for r in rows}))
print('score>=0.5', sum(1 for r in rows if r['score'] >= 0.5), 'max score', max(r['score'] for r in rows), 'min', min(r['score'] for r in rows))
for r in rows[:3]:
    print(json.dumps(r))
PY
```

Output, unedited:

```
rows 156 type list
key sets [('boxes', 'score', 'z0', 'z1')]
score>=0.5 11 max score 0.9999440908432007 min 0.027374267578125
{"z0": 5, "z1": 7, "score": 0.9999440908432007, "boxes": {"5": [[101, 198, 117, 137]], "6": [[101, 198, 117, 137]], "7": [[101, 198, 117, 137]]}}
{"z0": 3, "z1": 5, "score": 0.7039624452590942, "boxes": {"3": [[89, 123, 180, 221]], "4": [[89, 123, 180, 221]], "5": [[89, 123, 180, 221]]}}
{"z0": 8, "z1": 8, "score": 0.5, "boxes": {"8": [[169, 178, 192, 200]]}}
```

- 156 rows; every row has exactly the keys `boxes, score, z0, z1` (one key set over all rows).
- Rows at or above the fold-0 operating threshold 0.50 (`fold0/rule_a.json`, `nndet_thr`): 11.
- The box format is S2's Level R format (`[row0, row1, col0, col1]` per slice, RSS frame); a 3D box is repeated on
  each of its slices, so the first row spans slices 5–7 with one in-plane box.

There is no truth for this volume, so the 156 rows (11 at the threshold) are not evidence of anything. The entry runs end
to end on an unseen study in the nndet environment and writes the same table format as S2's `infer_brain_lesions.py`
(17 rows on the same study with the 2d five-fold ensemble, all 17 above its threshold 0.55).
