# Brain multi-disease detectors (S7): five-fold verdicts, cross runs, inference smoke — record index

Spec `docs/superpowers/specs/2026-09-29-brain-multidisease-design.md` (decisions M1–M11), plan
`docs/superpowers/plans/2026-09-29-brain-multidisease.md` (13 tasks). Three nnU-Net v2 `3d_fullres` detectors
(`nnUNetTrainer_250epochs`, patient-level five folds, out-of-fold validation predictions), one per disease, each on its
dataset's native channels; lesions are the 26-connected components of the argmax foreground, scored by the mean foreground
probability; a SynthSeg pseudo-label map on the same grid gives every lesion a host structure and a side; each study gets a
record with a sentence "在xxx解剖上存在xxx异常，疑似xxx疾病". Every number below is copied from the file named beside it;
the three verdict reports and the three cross-run reports carry `Code: commit 9a89ff1` (the two single-fold readings
were made earlier: the infarct fold 0 reading before the stamp existed, the metastasis fold 1 reading at 7c4ae27).

| disease | nnU-Net dataset | source | channels | scans | patients |
|---|---|---|---|---|---|
| glioma | Dataset904_PDGMGlioma | UCSF-PDGM v5 | T1, T1c, T2, FLAIR | 501 | 495 |
| metastasis | Dataset905_BMSRMetastasis | UCSF-BMSR | T1pre, T1post, FLAIR | 461 | 314 |
| infarct | Dataset906_ISLESInfarct | ISLES 2022 | DWI, ADC | 250 | 250 |

(`build/{glioma,metastasis,infarct}_raw.txt`, last line of each.)

## Verdicts (gate M2, same rule as S2's D1: sensitivity ≥ 0.5 at ≤ 2 false positives per scan)

Matching: 3D box IoU ≥ 0.1, one to one; ground-truth components under 10 mm³ are ignored (not in the denominator, a
detection on them is excused); predicted components under 10 mm³ are dropped; the operating point is the threshold of the
grid 0.05 … 0.95 with ≤ 2 false positives per scan and the highest sensitivity (ties → highest threshold). Out-of-fold
over all five folds, pooled (`<disease>/verdict.json`, `<disease>/REPORT.md`, `<disease>/froc.csv`).

| model | scans | GT lesions counted (ignored < 10 mm³) | threshold | sensitivity | FP per scan | gate | Dice, mean (cases) | FP per scan: median / max / scans over 2 | sensitivity <5 / 5–10 / ≥10 mm (N GT) |
|---|---|---|---|---|---|---|---|---|---|
| glioma | 501 | 677 (259) | 0.60 | 0.8109 (549) | 0.349 | **pass** | 0.928 (501) | 0 / 7 / 10 | 0.0625 (64) / 0.1389 (36) / 0.9359 (577) |
| metastasis | 461 | 3809 (531) | 0.65 | 0.7474 (2847) | 0.575 | **pass** | 0.807 (461) | 0 / 8 / 26 | 0.5723 (1936) / 0.9088 (1283) / 0.9712 (590) |
| infarct | 250 | 2111 (238) | 0.50 | 0.5722 (1208) | 1.556 | **pass** | 0.785 (247) | 1 / 15 / 54 | 0.3363 (1026) / 0.7449 (682) / 0.8809 (403) |

Metastasis by prior craniotomy, biopsy or resection: no 0.7496 (2839 lesions), yes 0.7412 (970).

Beside the small-lesion detectors of S2 on fastMRI+ (same matching rule, 1297 lesions, 253 volumes;
`docs/verification/2026-09-28/brain_detector/README.md`, `docs/verification/2026-09-29/brain_nndet/fold0/REPORT.md`):

| detector | data | folds | sensitivity | FP per volume | threshold | status |
|---|---|---|---|---|---|---|
| nnU-Net 2d | fastMRI+ small lesions | 5 (gate) | 0.3662 (475/1297) | 1.636 | 0.55 | D1 fail |
| nnU-Net 3d_fullres | fastMRI+ small lesions | 5 (report only) | 0.3678 (477/1297) | 1.431 | 0.60 | report only |
| nnDetection, default postprocessing | fastMRI+ small lesions | fold 0 (rule A, not the gate) | 0.0607 (17/280) | 0.7255 | 0.50 | rule A fail, arm stopped |
| nnDetection, swept postprocessing | fastMRI+ small lesions | fold 0 | 0.2429 | 1.8431 | 0.85 | NOT_GATE |

The three disease datasets are not comparable with fastMRI+ in lesion size: the fastMRI+ lesions are mostly single-slice
small boxes, while 577 of the 677 counted glioma lesions and 403 of 2111 infarct lesions are ≥ 10 mm equivalent
diameter, and every sensitivity above falls steeply below 5 mm (the strata columns).

Scores exceed 0.5 by construction, so the grid rows up to 0.50 are identical and the false-positive budget is never
reached: at the lowest row glioma has 0.3932 and metastasis 0.5813 false positives per scan, infarct 1.556. The glioma and
metastasis operating thresholds (0.60, 0.65) are the highest rows that keep the un-thresholded sensitivity; the infarct
threshold 0.50 is the lowest row (0.55 already loses two lesions).

## Cross false-alarm runs (report only, spec M11; `crossrun/<model>_on_<data>/{REPORT.md,crossrun.json}`)

One disease's five-fold average model on another disease's fold 0 validation cases, at the model's operating threshold,
on the channels the two datasets share (glioma → metastasis takes BMSR's synthetic T2 as the T2 channel). Counting is
overlap: a detection is "on GT" when it shares a voxel with any labelled voxel (fragments under 10 mm³ included); a
counted GT lesion (≥ 10 mm³) is "claimed" when any detection shares a voxel with it. These are not sensitivities,
precisions or false-positive rates, and they describe these pairs of datasets (different scanners, resolutions,
preprocessing), not the diseases.

| model → data | cases | threshold | detections | detections on GT | detections per scan | scans with any detection | GT lesions counted | GT lesions claimed |
|---|---|---|---|---|---|---|---|---|
| infarct → glioma (DWI, ADC) | 100 | 0.50 | 46 | 25 | 0.46 | 36 | 144 | 16 |
| metastasis → glioma (T1, T1c, FLAIR) | 100 | 0.65 | 167 | 153 | 1.67 | 88 | 144 | 95 |
| glioma → metastasis (T1pre, T1post, T2Synth, FLAIR) | 105 | 0.60 | 378 | 164 | 3.60 | 105 | 729 | 222 |

Reading: the metastasis model fires on most gliomas (153 of its 167 detections lie on tumour); the glioma model fires
on every metastasis scan (3.6 detections per scan, 164 of 378 on labelled tissue, 222 of the 729 counted metastases
touched); the infarct model fires on few gliomas. "疑似X" in a record therefore states which model ran, not a differential
diagnosis (spec §6); none of the models was trained or tested on normal brains, and the false-positive rates of the
verdict table are measured on diseased scans.

## Binding (NOT_EVIDENCE: the anatomy is a SynthSeg pseudo-label; `<disease>/REPORT.md`, "Binding agreement")

For every matched pair, the host structure and side looked up on the predicted component are compared with the lookup
on the ground-truth component, both on the same SynthSeg map. Agreement between two lookups on a pseudo-label says
nothing about anatomical truth; left and right are the headers' (spec M10).

| model | pairs | host agreement | side agreement | host+side agreement | detections | bound by the nearest rule | unlocated (> 10 mm) |
|---|---|---|---|---|---|---|---|
| glioma | 549 | 0.9654 | 0.9982 | 0.9982 | 725 | 0.0083 | 0 |
| metastasis | 2847 | 0.9772 | 0.9972 | 0.9989 | 3156 | 0.0029 | 0.0010 |
| infarct | 1208 | 0.9627 | 0.9934 | 0.9983 | 1601 | 0.0125 | 0 |

## Records (`/data2/congcong/data/FM_data/derived/brain_disease/<disease>/records/<case>.json`; 501 / 461 / 250)

Each record: `study`, `disease_model`, `model_folds` (the one fold that held the case out), `threshold` (the operating
threshold), `impression`, `lesions` (score, box, volume, host, host_rule, host_fractions, side, host_side, host_sides,
host_distance_mm), `sentence`, `anatomy_source`. Sentence rules (spec §6; the two marked 待用户确认 there are implemented
as recommended): the five largest lesions by volume are named, one clause each; the rest are counted, with the places not
already named in brackets ("另有 N 处同类异常（还见于…）"); a lesion overlapping no structure is written on the nearest one
("邻近…（未与任何结构重叠）") or, beyond 10 mm, as "未能定位的区域"; a study without a detection reads "本模型未检出<类型>（阈值 0.xx）。"
with the impression "未检出相关异常". Read on the written records (`checks/sentence_preview.py` for the fold readings; the
counts below from the 1212 files):

| disease | lesions per record: min / median / max | records without a lesion | records with more than five | records with 还见于 (most places named) | longest sentence | detections by the nearest rule |
|---|---|---|---|---|---|---|
| glioma | 1 / 1 / 10 | 0 | 2 | 1 (2) | 183 chars | 6 of 725 |
| metastasis | 0 / 4 / 69 | 3 | 174 | 127 (8) | 221 chars | 9 of 3156 |
| infarct | 0 / 4 / 48 | 5 | 101 | 62 (6) | 220 chars | 20 of 1601 |

## Inference smoke

`infer_smoke.md`: the entry `scripts/infer_brain_disease.py` on the first fold 0 validation case of each disease with the
fold 0 model; 1 / 11 / 26 lesions; boxes and scores agree with the out-of-fold records to 1e-4
(`checks/infer_smoke_consistency.txt`).

## Known deviations

1. **Three infarct cases have no labelled voxel** (`build/infarct_raw.txt`: "3 cases without label voxels"). They are
   scans (false-positive denominator) but contribute no lesion; the Dice summary covers the 247 cases with ground truth.
2. **Ignored fragments**: 259 / 531 / 238 labelled components under 10 mm³ are not in the sensitivity denominator, and a
   detection on one of them is excused (spec §5). For metastasis they are 12 % of all labelled components.
3. **Synthetic T2** in the glioma → metastasis cross run (channel 2 is BMSR's synthetic T2; `crossrun.json` `note`).
4. **Matching rule** as in S2: the assignment with the largest total IoU is taken and pairs under 0.1 are dropped
   afterwards; in 47 of 20 000 random crowded scans this gives fewer hits than the best one-to-one matching, never more
   (whole-branch review 1, `reviews/final-review-1-report.md`). Kept for
   comparability with S2.
5. **Thresholds are measured on single-fold models** (each case predicted by the fold that held it out) and applied to
   the five-fold average in the cross runs and, by default, in the inference entry; the averaged model's behaviour at
   these thresholds was not measured on its own data.
6. **The score is a mean probability** (> 0.5 by construction; the smallest components score highest; large false
   positives cannot be thresholded away). Kept for comparability with S2.
7. **The early-stop rule (M4) never fired**: every fold of every disease trained and was evaluated; the queue log holds
   `finished … with exit code 0` for all 15 jobs and ends with `queue empty, nothing running: done`.
8. **The inference entry names the study after its first image file** (`infer_smoke.md`); the evaluation's records use
   the case id.
9. **Other sessions' jobs shared the cards** from the afternoon of 2026-09-29; the per-fold hours below vary for that
   reason (glioma 4.95–11.08 h), not because of the data.

## Timing (`logs/brain_disease/queue.log`, not in the repository; the queue ran with the code before ecb172d)

Queue launched 2026-09-29 13:46:14 on the six idle GPUs 0/1/2/4/5/6, at most six jobs, `nnUNet_n_proc_DA=6`; last job
finished 2026-09-30 05:48:16; wall clock 16.03 h; 85.8 GPU-hours; 15 jobs, all exit code 0.

| disease | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| glioma | 11.08 h (GPU 0) | 8.17 h (4) | 5.80 h (6) | 5.43 h (3) | 4.95 h (0) |
| metastasis | 9.00 h (1) | 5.08 h (5) | 7.87 h (2) | 4.75 h (1) | 4.37 h (2) |
| infarct | 3.30 h (2) | 3.27 h (6) | 6.22 h (5) | 3.32 h (4) | 3.23 h (5) |

Evaluations (2026-09-30, CPU, `--workers 8`, the three in parallel): launched after 09:53, finished 09:55:28 (infarct),
09:57:03 (glioma) and 09:57:23 (metastasis), records included. Cross runs (one GPU each, launched 09:59): infarct → glioma
done 10:04:31, metastasis → glioma 10:15:06, glioma → metastasis 10:32:51. Inference smokes: under a minute each.

## Records in this folder

| path | what | made by |
|---|---|---|
| `build/` | dataset builds (`*_raw.txt`), BMSR patient grouping (`bmsr_subjects.txt`), nnU-Net plans (`plans.txt`), splits (`splits.txt`) | `scripts/brain_disease_prepare.py` and nnU-Net's planning, run by the Task 2 implementer (plan Task 2) |
| `launch.md` | GPU snapshot, queue launch, first rates and projections | controller (plan Task 3, Steps 6–7) |
| `checks/` | read-only checks on the real data with their outputs, `sentence_preview.py`, `infer_smoke_consistency.py` | controller (see `checks/README.md`) |
| `infarct_fold0/`, `metastasis_fold1_preliminary/` | single-fold early readings (not verdicts) | `scripts/eval_brain_disease.py --folds <f>` |
| `glioma/`, `metastasis/`, `infarct/` | five-fold verdicts: `REPORT.md`, `verdict.json`, `froc.csv`, `output.txt` | `scripts/eval_brain_disease.py --disease <d> --folds 0 1 2 3 4 --workers 8 --records …` (Task 12 Step 1) |
| `crossrun/<model>_on_<data>/` | cross false-alarm runs: `REPORT.md`, `crossrun.json` | `scripts/brain_disease_crossrun.py` (Task 12 Step 2) |
| `infer_smoke.md` | inference entry on one case per disease | `scripts/infer_brain_disease.py` (Task 12 Step 3) |
| `reviews/` | the two whole-branch review reports: review 1 at a22da8b (before any training finished, six round trips), review 2 at 1e2e95f (after the verdicts; independent recount of all 1212 scans, records and cross runs; mergeable as is, minor findings only) | most capable model, dispatched by the controller (Task 13) |
| `README.md` | this index | Task 12 Step 4 |

Outside the repository: nnU-Net raw/preprocessed/results under `/data2/congcong/data/FM_data/derived/nnunet/` (Dataset904–906),
records, cross-run work directories and smoke outputs under `/data2/congcong/data/FM_data/derived/brain_disease/`, the
queue and training logs under the worktree's `logs/brain_disease/`.
