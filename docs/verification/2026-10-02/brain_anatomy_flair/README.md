# S4 brain anatomy on fastMRI FLAIR: records index (spec `docs/superpowers/specs/2026-10-02-brain-anatomy-flair-design.md` A1–A17, plan `docs/superpowers/plans/2026-10-02-brain-anatomy-flair.md`)

Teacher = SynthSeg on 1 mm T1 of three sources (SibBMS, PDGM, BMSR); the 1 mm FLAIR of the same cases is simulated into
fastMRI-like 16 × 5 mm axial stacks and a nnU-Net student learns 16 compact classes on them; a 2d outline model trained on
the real fastMRI stacks strips the skull at inference. Every number below is copied from the file named beside it.
Everything that compares two pseudo-label maps is NOT_EVIDENCE; the final judgement waits for the Level R reader (A12).

## Records in this folder

| file | what it is | made by |
|---|---|---|
| `build/sources.txt` | the 1320 cases and the patient-level 80 / 20 split per source (BMSR 359 / 102 cases, PDGM 400 / 101, SibBMS 276 / 82; `cases.json`) | `scripts/brain_anatomy_prepare.py --stage sources` |
| `build/simulate.txt` | the simulation run: 4140 training samples (K = 4 per training case) and 285 test samples | `--stage simulate --workers 8` |
| `simulation/README.md`, `simulation/sim_*.png` | sampled parameters, six montages (image / student labels), coverage beside the real stacks and the open calibration question | `checks/sim_montage.py` (`build/sim_montage.txt`) |
| `build/dataset908.txt`, `build/plan908.txt` | Dataset908_FastMRIBrainOutline: 345 training stacks, 88 test stacks, 14 excluded (`excluded.json` in the raw folder); nnU-Net planning | `--stage dataset908`; `nnUNetv2_plan_and_preprocess -d 908 -c 2d` |
| `build/dataset907.txt`, `build/plan907.txt`, `build/plans.txt` | Dataset907_BrainAnatomyFLAIR: 4140 / 285; plan 3d_fullres spacing [5, 0.6875, 0.6875], patch [16, 320, 320], batch 2 | `--stage dataset907`; `nnUNetv2_plan_and_preprocess -d 907 -c 3d_fullres` |
| `build/splits908.txt`, `build/splits907.txt` | fold files by patient: 69 × 5 validation stacks; 872 / 856 / 836 / 788 / 788 samples, 795 patients, none in two folds | the script's split functions (see the note in each file) |
| `launch.md` | GPU snapshots, launch lines, rates; the mirroring defect of the first student run and the relaunch (A17) | controller |
| `training.txt` | how the three runs ended: epochs, losses, per-class validation Dice | controller, from the result folders |
| `eval/REPORT.md`, `eval/verdict.json`, `eval/per_stem.{csv,json}`, `eval_output.txt` | the A11 evaluation on the 433 real stacks and the simulated test set | `scripts/eval_brain_anatomy.py` |
| `eval/README.md` | definitions, denominators, the verdict, the post-hoc diagnosis and what was seen in the montages (USER_REPORTED) | controller |
| `eval/montage/*.png`, `eval/low_slices/*.png` | six stacks: slices 0, 1 and the two above the reliable range; the same six, slices 0–5 (stack / student / SynthSeg) | `checks/eval_montage.py`, `checks/eval_low_slices.py` |
| `eval/ref_vs_student.txt`, `eval/ref_vs_student_rows.json` | per structure group, what the SynthSeg reference and the student hold on the real stacks | `checks/eval_ref_vs_student.py` |
| `infer_smoke.md` | the inference chain on S2's smoke volume and on a registry lesion with its box, beside the SynthSeg map | `scripts/infer_brain_anatomy.py` |

Data: `/data2/congcong/data/FM_data/derived/brain_anatomy/` (`cases.json`, `sim/` 23 GB, `eval_20261004_0233/` 507 MB,
`infer_smoke/`, `preflight_best_20261003_2249/` — a probe, deletable); nnU-Net raw / preprocessed / results for
Dataset907 (preprocessed 29 GB) and Dataset908 under `/data2/congcong/data/FM_data/derived/nnunet/`.

## Verdict (spec A11; `eval/verdict.json`)

| gate | value | threshold | pass |
|---|---|---|---|
| host agreement of the registry lesion boxes (student lookup against SynthSeg lookup) | 0.9215 (881 / 956 evaluated; 1267 lesions in the 433 stacks, 311 outside the reliable slices) | ≥ 0.90 | yes |
| mean sided-host Dice on the reliable slices (13 host classes, equal weight) | 0.2665 | ≥ 0.80 | **no** |
| outline Dice on the outline model's 88 test stacks | 0.9781 (min 0.8272) | ≥ 0.97 | yes |

**Verdict: fail.** No tuning followed (plan Task 9 Step 11). Per class on the real stacks: white matter 0.8334 / 0.8360,
cortex 0.7798 / 0.7787, thalamus 0.0240 / 0.0031, basal ganglia 0.1345 / 0.0748, brainstem 0, cerebellum 0 / 0, other deep
grey 0 / 0. The diagnosis in `eval/README.md` (post hoc, 2026-10-08): inside the reliable slices the SynthSeg reference
holds a mean of 0.06 mL of thalamus per stack (in 79 of 433 stacks) and only stray voxels of brainstem, cerebellum and
other deep grey, so nine of the thirteen classes have a Dice near 0 by construction; in the lowest two slices the student
labels the ventricles, thalamus and basal ganglia where the images show them and the reference does not. Three readings
for the user are laid out at the end of `eval/README.md`.

Reported, not judged (`eval/REPORT.md`, `training.txt`): simulated test set (A13) mean host Dice 0.6611 (white matter
0.8496 / 0.8467, cortex 0.7888 / 0.7908, thalamus 0.8052 / 0.8256, basal ganglia 0.7796 / 0.7967, brainstem 0.3535,
cerebellum 0.4521 / 0.4079, other deep grey 0.4389 / 0.4588, ventricles 0.8486); student fold-0 validation (simulated)
0.7004; outline fold-0 validation 0.9704; median labelled area of the student in slices 0 / 1: 145.4 / 149.3 cm².

## Known deviations

- **Mirroring (A17, found during Task 8).** Neither the spec nor the plan said that nnU-Net's default mirroring makes sided
  classes unlearnable. The first student run (`nnUNetTrainer_250epochs`) is not used; its result folder is kept as the
  record of the defect (`launch.md`, `training.txt`: validation Dice 0.5587, cerebellum 0 / 0). The delivered student is
  `nnUNetTrainer_250epochs_NoMirroring`. The outline model has no side and was not touched.
- SibBMS: 9 sessions excluded (`anatobind/anatomy/sources.py::EXCLUDED_SIBBMS`: the SynthSeg failures and the one 2D T1w
  of `docs/verification/2026-09-30/s4_sibbms_synthseg/`) → 358 FLAIR sessions of 185 patients. SibBMS lesions are not set
  to ignore (no same-grid lesion masks); PDGM and BMSR lesions are.
- BMSR was resampled from 1.5 mm to the 1 mm teacher grid. SibBMS is in template space (already skull-stripped).
- fastMRI: 14 of 447 stacks are excluded from the outline model (SynthSeg outline under 300 mL); the 30 registry lesions
  in them are outside the host-agreement denominator.
- The student is trained on simulated stacks from 3D FLAIR and judged on real 2D FLAIR (A7); the simulation's calibration
  against the real stacks is an open question (`simulation/README.md`).
- The lowest two slices and the slices above the reliable range are reported, not judged (A12); the binder uses the
  reliable range. In the montages, slice 14 of some stacks shows brain tissue that does not look like the vertex (possibly
  slices wrapping to the other end of the head) — not examined.
- HD-BET (the spec's fallback for skull-stripping) was not run: the own outline model passed its gate.
- Left / right follow the NIfTI headers of the sources and the fastMRI RSS frame convention (`s4_probe3_frame`).
- The two diagnosis checks (`eval_low_slices.py`, `eval_ref_vs_student.py`) were written after the evaluation, to explain
  the number; they are not part of the pre-registered gate.
- The hook that forbids deletion kept the first student run alive (the signal to stop it was refused); it ran to its end
  on GPU 4 beside the two used runs.

## Timing (all on 2026-10-03/04 unless said; from the files named)

| step | when | duration | where |
|---|---|---|---|
| sources | 22:08:01 | seconds | `build/sources.txt` |
| simulate (8 workers, CPU) | 22:08:11 → 22:34:44 | 26.5 min | `build/simulate.txt` |
| dataset908 + plan 908 | 22:17:31 → 22:21:40 | 4 min | `build/dataset908.txt`, `build/plan908.txt` |
| dataset907 + plan 907 (8 processes) | 22:36:55 → 22:43:50 | 7 min | `build/dataset907.txt`, `build/plan907.txt` |
| outline training, 250 epochs, GPU 7 | 22:22:42 → 01:22:18 (validation included) | 3.0 h, 41.4 s / epoch | `training.txt` |
| student without mirroring, 250 epochs, GPU 5 | 22:56:51 → 02:32:53 (validation included) | 3.6 h, 49.0 s / epoch | `training.txt` |
| first student run (mirroring, unused), GPU 4 | 22:44:28 → 02:21:53 | 3.6 h, 48.5 s / epoch | `training.txt` |
| evaluation (433 + 433 + 285 predictions), GPU 7 | 02:33 → 02:49 | 16 min | `eval_output.txt` |
| montages, diagnosis, inference smoke | 2026-10-08 09:46 → 09:50 | 4 min | `eval/`, `infer_smoke.md` |
