# S2 inference smoke (2d, five-fold ensemble)

Volume: `file_brain_AXFLAIR_201_6002917` — one of the 130 FLAIR studies whose fastMRI+ rows carry
"Global label: Small vessel chronic white matter ischemic change" but no small-lesion boxes (150 such studies,
130 without registry lesions; the first in sorted order). It is not one of Dataset903's 253 cases, so no fold saw it.

Command (GPU 3 idle at 2026-09-28 21:50; outputs under the git-ignored `runs/`):

```
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_lesions.py \
  --h5 /data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/brain/multicoil_train/file_brain_AXFLAIR_201_6002917.h5 \
  --out runs/brain_infer_smoke --folds 0 1 2 3 4 --gpu 3
```

Last lines of the run:

```
Segmentation export complete.
Found 17 lesions. Output: runs/brain_infer_smoke
... 81.08s user 7.41s system 396% cpu 22.316 total
```

Table check (`runs/brain_infer_smoke/lesions.json`):

```
rows 17 keys ['boxes', 'score', 'z0', 'z1']
scores [0.965, 0.941, 0.938, 0.935, 0.918, 0.914, 0.909, 0.88, 0.879, 0.878, 0.869, 0.836, 0.792, 0.754, 0.745, 0.721, 0.702]
rows with score >= 0.55 (2d five-fold operating threshold): 17
n slices per row [1, 4, 4, 1, 2, 1, 3, 4, 1, 1, 1, 1, 1, 2, 1, 1, 1]
first row: {"z0": 5, "z1": 5, "score": 0.9646308422088623, "boxes": {"5": [[98, 113, 210, 222]]}}
```

The entry point runs end to end on an unseen study and writes the Level R box format (`[row0, row1, col0, col1]`
per slice, RSS frame). There is no truth for this volume, so the 17 detections are not evidence of anything; the
five-fold gate result (sensitivity 0.366 at 1.64 FP per volume) says roughly how many are real.
