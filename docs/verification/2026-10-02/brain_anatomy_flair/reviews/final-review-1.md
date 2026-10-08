# Final whole-branch review — S4 brain anatomy on fastMRI FLAIR

Range reviewed: 8e3989d..236b6e1 (25 commits, 71 files), read in the checkout. HEAD of `main` is now 3a2ea0c (one commit
past the range: STATUS.md, the unused training folder's size 1.4 GB -> 712 MB; verified with `du`: 712M). Reviewer:
read-only; ran the full test suite once (CPU, nice 19) and read-only Python snippets over the records and
`/data2/congcong/data/FM_data/derived/`. No file in the checkout was changed except this report.

**Verdict: needs fixes — records and spec text only. The code, the numbers and the fail verdict stand.** No file turns the
fail into a pass; every record (REPORT.md, verdict.json, README, eval/README.md, STATUS.md, CLAUDE.md) says fail; the
post-hoc readings are labelled post hoc and conditioned on a spec amendment. No deletion, rename or truncation call in the
range; every output is refused when it exists; nothing writes outside `derived/{brain_anatomy,nnunet}`, `docs/`, `logs/`.
Tests: `872 passed, 1 skipped in 84.70s` (my run; CLAUDE.md and STATUS.md say 872 / 1).

Critical: 0. Important: 1. Minor: 12.

## Critical

None.

## Important

**I1. Two records claim a safeguard the code does not have: "the binder uses the reliable range".**
`docs/verification/2026-10-02/brain_anatomy_flair/README.md:65-66` ("the lowest two slices … are reported, not judged (A12);
the binder uses the reliable range") and `eval/README.md:78` ("outside the reliable range the binder uses").
`anatobind/infer/brain_anatomy.py:106-110` binds any `--box` on the full `anatomy` with `BrainBinder(...).bind(sl, ones)`;
no slice of the box is checked against the reliable range, and `reliable_slices` (line 99) is only written into
`record.json`. Moreover that range is computed from the student's own output (the rule applied to `anatomy > 0`), not from
a SynthSeg map as in the evaluation, so it means "where the student's own outline is trusted by the rule", which the record
key does not say. Why it matters: a reader of the index takes it that a box in slices 0–1 (the slices the spec refuses to
judge, 20 % of the registry lesions) is not bound or is flagged; it is bound silently with labels nobody has judged. Fix
(records): replace both sentences with "the record reports the reliable range of the student's own outline; the binder is
not restricted to it". Optional (code, one line, test it): add `"box_in_reliable_slices": z0 >= reliable.start and
z1 <= reliable.stop` to the record so S5 can gate on it. Neither changes any number.

## Minor

**M1. STATUS.md:1 says the tag `handoff/2026-10-08-brain-anatomy-flair` exists; `git tag -l 'handoff/2026-10-0*'` lists
nothing.** The plan (Task 10 Step 6) makes the tag after this review, so the sentence is premature. Fix: write "tag to be
set after the final review" now, or set the tag and leave the text.

**M2. `eval/README.md:20-21` — the distribution of the reliable top is missing one stack.** It lists 11 (183), 12 (141),
10 (70), 9 (19), 13 (15), 8 (4) = 432; `per_stem.json` has one stack with top 14. Fix: add "14 (1)".

**M3. `eval/README.md:26` "Ventricles are reported but are not a host"** — the ventricle Dice on the real stacks is not in
REPORT.md / verdict.json (`class_dice` defaults to `HOST_IDS`; ventricles appear only in the simulated-set Dice and as
volumes in `ref_vs_student.txt`). Fix: "ventricle volumes are reported in ref_vs_student.txt; the ventricle Dice is reported
for the simulated set only". Note for the user: had ventricles been the 14th class of the gate (spec wording, M8), the mean
would be at most (0.2665 × 13 + 1) / 14 = 0.319 — still a fail.

**M4. `simulation/README.md:19-29` — the "240 simulated 16-slice stacks" coverage column has no script and no output
file** (only the fastMRI column traces to `s4_probe2/fastmri_flair_coverage.txt`). Which 240 stacks were used is not
recorded. I recomputed on the first 240 16-slice test samples: median area 159 159 158 153 146 136 123 107 92 74 56 37 15 2
(README 157 157 155 150 144 135 123 108 92 76 57 36 15 1), basal ganglia 1.00 .98 .88 .72 .47 .22 (README 1.00 .96 .90 .73
.51 .25), thalamus .95 .81 .59 .33 (README .92 .80 .60 .33), cerebellum .15 .03 .01 (.14 .03 .00), bottom share 1.00 (0.98),
height 65 mm (65). The numbers are plausible, not reproducible. Fix: add the snippet as `checks/sim_coverage.py` with its
output, or mark the column "controller's ad-hoc count, not reproducible".

**M5. `anatobind/anatomy/simulate.py:17` comment "1 / 19 / 157 / 191 / 63 / 14 of 447 stacks have 0 / 1 / 2 / 3 / 4 / 5+"
sums to 445.** `s4_probe2/fastmri_flair_coverage.txt`: {0: 1, 1: 19, 2: 157, 3: 191, 4: 63, 5: 10, 6: 4, 11: 1} → 5+ is 15.
The weights in `EMPTY_TOP` (0.045 / 0.35 / 0.43 / 0.14 / 0.035) match 20 / 157 / 191 / 63 / 15 of 447, so only the comment is
wrong. Fix the comment when the file is next touched.

**M6. Spec A2 and §3 still say SibBMS 362** (`docs/superpowers/specs/2026-10-02-brain-anatomy-flair-design.md:23,44`); the
real count is 358 cases / 185 patients (`build/sources.txt`, plan line 1953, README "Known deviations"). The ledger's ruling
(Task 2) said the spec table would be corrected; the spec diff in the range is A17 only. Fix: the two numbers, with a note
"four MS sessions with a teacher map have no FLAIR".

**M7. Spec §5 step 7 says a 3rd-order polynomial bias field; `simulate.py:152-159` builds a 2nd-order field with six terms
(X, Y, Z, X², Y², XY; no Z², XZ, YZ).** Not listed under Known deviations. No effect on the verdict (augmentation only).
Fix: amend the spec sentence or add the line to Known deviations.

**M8. Spec A11 ② and §7 (lines 32, 89) say "14 个分侧宿主类"; the label space, the plan ("the 13 sided host ids") and the
code (`HOST_IDS`, 13 = 6 × 2 + brainstem) use 13.** The verdict is the same under either count (M3). Fix: "13" in the spec,
or state that ventricles are a landmark and not in the gate.

**M9. Spec §5 step 6 tie rule ("忽略背景之外的平局按靠近样板中心的层") vs `simulate.py:96-118`**: the code breaks every tie,
background included, by distance to the slab centre. Defensible reading; the spec sentence is ambiguous. Fix: one clause in
the spec.

**M10. `README.md:86` Timing row "evaluation 02:33 → 02:49 … where: eval_output.txt"** — `eval_output.txt` carries no
timestamps; the start is the work-directory name (`eval_20261004_0233`), the end is the mtime of `per_stem.json` (02:49) and
the ledger. Fix: name those sources, or add `date` lines around the command in future runs.

**M11. Spec A11 says 447 stacks; the evaluation ran on the 433 of Dataset908** (the 14 stacks with a SynthSeg outline under
300 mL have no usable reference; their 30 registry lesions are outside the denominator). Documented in README "Known
deviations" and eval/README; the spec text is not amended. Fix: a parenthesis in A11, or leave as a recorded deviation.

**M12. `scripts/eval_brain_anatomy.py:62-63` writes `<stem>_mask.nii.gz` into `stripped/`, which is then the nnU-Net input
folder of the student.** It works because nnU-Net derives case ids by stripping 5 + 7 characters (`_mask.nii.gz` collapses
onto the stem's case id and only `_0000` is read); the real run completed with 433 predictions. Fragile against a nnU-Net
version that refuses unexpected files; the test's fake predictor globs `*_0000.nii.gz` and does not exercise it. Fix at the
next change: write the masks into a sibling folder (`masks/`).

Observations that are not findings: the student's gates ① ② are computed on all 433 stacks, 345 of which trained the
outline model (the student never saw fastMRI; only the stripping is optimistic on those stacks; the spec asks for all
stacks). `launch.md:54` quotes the outline's pseudo Dice at epoch 34 as 0.973; the log says 0.9742. The fastMRI RSS header's
A/P code is inverted relative to the content (cerebellum at low axis-1 indices under an (L, P, S) header; `fastmri.py:34`
notes the fastMRI+ up/down flip); the simulation matches the content frame by the probe, both header conventions are
identical for simulated and real stacks, and the binder uses L/R only — no effect on S4, already an open item in STATUS §2.6.

## Code review against the spec (what was checked and found consistent)

- Label space (`labels.py`): 16 classes, ignore = 15 = highest id, 13 host ids, representative values map back to their own
  class and to the same `host_class_map` class; CSF 24 → background; `check_consistency` pins every label against
  `geometry.HOST_CLASSES / LEFT_LABELS / RIGHT_LABELS / LANDMARKS`. `to_synthseg` refuses ignore.
- Frame (`simulate.to_fastmri_frame`): flip axis 0 of a RAS array; agrees with `s4_probe3_frame/array_frames.txt` (fastMRI
  LEFT − RIGHT along axis 0 = +69.9 voxels; RAS −57.5; axes 1, 2 same sign). Simulated affine = `rss_affine` (diag
  −s, −s, 5), the same the real RSS NIfTIs carry.
- Stack placement (`bottom_index`): top + 1 + 5e − 5n, clamped at 0; `slab_groups` empty beyond the volume; manifest shows
  0 clamped bottoms in 4425 samples. Parameters and weights follow A6 / A7 (except M7).
- Reliable-slice rule used identically in three places: `outline.supervised_slices(fill_and_keep_largest(seg > 0), area)`
  builds the outline labels (prepare), `eval.reliable_slices` is the same call (evaluation and the two check scripts), and
  the inference record calls it on the student's output (see I1). Area in mm², LOW = 2, TOP_MARGIN = 1, 5 cm².
- Gates (`eval.verdict`): all three must pass; a missing number fails; thresholds 0.90 / 0.80 / 0.97; the report line prints
  the numbers (pinned by test). Host agreement: boxes with any slice outside the reliable range are counted, not evaluated;
  `BrainLookup(..., BRAIN_PARENCHYMA)` on both maps, side-agnostic class. Dice: per stack over reliable slices, class empty
  in both maps skipped, per-class mean over stacks where it occurs, gate = mean of 13 per-class means (as eval/README says).
- Trainers: `STUDENT` = `nnUNetTrainer_250epochs_NoMirroring` (class exists in the env's nnunetv2), `OUTLINE` =
  `nnUNetTrainer_250epochs`; launcher and predictor read the same dicts; tests pin both command lines. `debug.json`:
  student (used) `inference_allowed_mirroring_axes = None`, outline `(0, 1)`, first student run `(0, 1, 2)`.
- Grid safety: `check_one_grid` (shape + affine ≤ 1e-3) before any resampling of a teacher map or lesion mask;
  `write_outline_case` refuses an RSS image off its SynthSeg grid.
- Sources / split: per-source patient-level split, `check_split` refuses a mixed patient; real counts bmsr 359/102 cases
  (251/63 patients), pdgm 400/101 (396/99), sibbms 276/82 (148/37) = 1320 cases, 185 / 495 / 314 patients.

## Safety

- Deletion / rename / truncation: scanned the 25 Python files of the range for `os.remove/unlink`, `Path.unlink`, `rmdir`,
  `rmtree`, `shutil.move`, `os.rename/replace`, `Path.rename/replace`, `truncate`, `open(..., "w")`: no removal or rename
  call. The only `"w"` open is `per_stem.csv` inside a `--out` directory that was refused if it existed and created just
  before. `rss_h5_to_nifti` (pre-existing code) would overwrite, but is only called inside a freshly created dataset root.
- Refusals before creation: infer `run` (out), eval `main` (out, work), prepare (`cases.json`, `sim/`, both dataset roots,
  `splits_final.json`, both datasets checked before either split is written), train (result folder and log), all four check
  scripts (output dir / rows file). `git diff --diff-filter=DR` over the range: no file deleted or renamed.
- Write roots: `WORK = FM/derived/brain_anatomy`, `NNUNET_ROOT = FM/derived/nnunet`, `LOG_DIR = repo/logs/brain_anatomy`,
  records under `docs/`; `--out` / `--work` of eval and infer are user-given (the runs used `docs/` and `derived/`).
  No reference to `/data0/congcong/data` in the range.
- Commits: author Congcong Liu on all 25; no Co-Authored-By / "Generated with" / AI trace in any message.
- Tests never read `/data2` (tmp_path trees and fakes throughout).

## Numbers checked (traced to a file; "✓" = equal)

- verdict.json / REPORT.md: host agreement 0.9215 = 881/956, n_lesions 1267, outside 311; mean host Dice 0.2665
  (recomputed from the 13 per-class values: 0.266479 ✓); outline mean 0.9781, min 0.8272, 88 test stacks ✓; per-class Dice ✓;
  low-slice medians 145.4 / 149.3 (recomputed from per_stem.json ✓); simulated set 0.6611, 285 cases, ventricles 0.8486 ✓;
  `eval_output.txt` last line prints the same three numbers and `pass False` ✓.
- eval/README.md: reliable top 11/183, 12/141, 10/70, 9/19, 13/15, 8/4 ✓ (+ 14/1 missing, M2); first reliable slice 2 in
  all 433 ✓; 4 outline stacks under 0.95 ✓; registry 1297 lesions in 165 stacks ✓ (`load_registry`); 30 lesions in the 14
  excluded stacks ✓ (`excluded.json`); by SynthSeg host WM 846/855, cortex 35/100, BG 0/1 ✓; disagreements 65 / 7 / 2 / 1 ✓;
  per-stack median 0.418, P10–P90 0.298–0.631, min 0.094, max 0.851 ✓; ref_vs_student table (8 groups, mL and counts) ✓
  against `ref_vs_student.txt`.
- README.md index: 1320 cases and the six split counts ✓ (`build/sources.txt`, `cases.json`); 4140 / 285 samples ✓
  (`simulate.txt`, manifest); Dataset908 345 / 88 / 14 ✓ (`dataset908.txt`, `cases.json`, `excluded.json`); plans spacing /
  patch / batch ✓ (`plans.txt`); splits 69 × 5 and 872 / 856 / 836 / 788 / 788, 795 patients ✓ (`splits90x.txt`); sizes sim
  23 GB, eval 507 MB, preprocessed 29 GB, unused run 712 MB ✓ (`du`); timing rows ✓ against `training.txt` and the build
  logs (except M10); 0.06 mL thalamus in 79 of 433 ✓; simulated-set and validation Dice ✓.
- simulation/README.md: 3953 / 472 slices, 3643 / 782 in-plane, 2630 / 958 / 837 matrices, empty top 206 / 1546 / 1948 /
  581 / 144, ignore in 3025 samples (PDGM 1643, BMSR 1382), no clamped bottom ✓ (manifest); fastMRI column ✓
  (`s4_probe2/fastmri_flair_coverage.txt`); simulated column not traceable (M4).
- launch.md: pseudo-Dice rows of epochs 5 / 6 / 8 / 9 (first run) and 1 / 4 / 6 / 7 (no-mirroring run), all 8 leading
  values and the ventricle value ✓ (training logs); epochs 0–3 49.04 / 47.23 / 47.11 / 48.41 s ✓; outline 0.9734 at epoch 21 ✓,
  0.9742 at epoch 34 (quoted 0.973); mirroring axes ✓; preflight WM L 19.2 / R 268.9 mL, cortex 0.1 / 255.5 ✓
  (`preflight_best_20261003_2249/svd/record.json`).
- training.txt: 250 epochs each; mean epoch 41.4 / 49.0 / 48.5 s ✓ (recomputed from the logs); validation Dice 0.9704 /
  0.7004 / 0.5587 ✓; all per-class validation values ✓ (`validation/summary.json`); 70 / 873 / 873 files ✓.
- infer_smoke.md: brain 728.1 / 778.9 mL, reliable [2, 12] / [2, 11], all class volumes ✓ (`record.json`); box values
  student {41: 30, 42: 12}, SynthSeg {41: 28, 42: 14} ✓ (read from the two maps); BrainBinder on the SynthSeg map
  white_matter / overlap / right 0.667 ✓ (recomputed).
- STATUS.md / CLAUDE.md: every S4 number above ✓; "(3) 白质+皮层四类均值 0.807" = 0.8070 recomputed ✓; 872 passed / 1
  skipped ✓ (my run 84.70 s).
- Not found anywhere: the simulated coverage column (M4); eval start/end timestamps in `eval_output.txt` (M10).

## Rulings on the deferred minors (ledger `progress.md`)

| # | task | deferred minor | ruling |
|---|---|---|---|
| 1 | 1 | HOST_CLASS_OF / SIDE_OF from name suffixes (labels.py:27-30) | drop — `check_consistency` pins every class against geometry's tables |
| 2 | 1 | bare `assert` in check_consistency (python -O) | park — nothing runs with -O; turn into `raise` when the file is next touched |
| 3 | 1 | no negative test of check_consistency | park |
| 4 | 1 | empty / float input, mask shape in to_student / to_synthseg / with_ignore | drop — internal callers, shapes checked upstream |
| 5 | 2 | pdgm_cases / bmsr_cases do not check file existence | drop — `check_one_grid` opens every file at build time; 1320 cases verified present |
| 6 | 2 | `ses-???` glob and cohort filter silent | park — count recorded (358); one docstring line when touched |
| 7 | 2 | unreachable collision check; no guard for share 0 / 1 / one patient | drop |
| 8 | 3 | `ndimage.zoom(grid_mode=False)` scale off by ≤ 0.3 % | park — ≤ 0.5 px; change to `grid_mode=True` only if the stacks are regenerated (reading 1); the existing 23 GB cannot be removed |
| 9 | 3 | `np.add.at` slow in vote_labels | drop — 26.5 min for 4425 samples |
| 10 | 3 | missing tests (rotation sign, clipped stack, no-brain volume) | park |
| 11 | 4 | Dice test tolerance abs = 0.02 | drop |
| 12 | 4 | no test of MIN_VOLUME_ML with a non-empty supervised range | park — the real build exercised it (14 excluded, volumes recorded) |
| 13 | 4 | postprocess treats > 0 as brain | drop — callers pass argmax maps |
| 14 | 5 | rects_of does not clip; None == None counts as agreement | park — registry boxes lie on the grid (1280/1297 hold host voxels, the rest resolve by nearest); add an `n_both_none` counter at the next eval change |
| 15 | 5 | mean_host_dice definition to be stated in the README | closed — eval/README.md:22-26 states it |
| 16 | 5 | no test of nearest fallback / pool_agreement without boxes | park |
| 17 | 6 | bare errors after a prediction (shape, save) | drop — the exists message on rerun explains |
| 18 | 6 | check_box after staging leaves `input_outline` | drop — the error says so |
| 19 | 7 | simulate_case has no exists-check; serial determinism test; 908 grid check after the image is written | park |
| 20 | 8 | `--jobs student student` starts two trainings in one folder | park — one line (`list(dict.fromkeys(a.jobs))`) at the next launch; no launch is planned unless reading 1 |
| 21 | 8 | exit code 0 when one of two starts; "not started" on stdout | drop |
| 22 | 8 | gap between log check and redirect; main untested | drop |
| 23 | 8 | prints the `setsid` wrapper pid; `setsid` redundant with `start_new_session` | park — documented in STATUS §4 and launch.md; drop `setsid` at the next launch |
| 24 | 9 | registry rows of unevaluated stacks dropped without a count | closed in the record (30 lesions, eval/README.md:27-29); park adding the count to verdict.json for a rerun |
| 25 | 9 | failure after predictions leaves `--work` without a message; no resume | park |
| 26 | 9 | < 2-slice stack bias; `sys.argv` as the recorded command; untested paths, eval_montage.py untested | drop (no stack has < 12 slices; the recorded command is complete) |
| 27 | A17 | run_nnunet with an unknown dataset id raises a bare KeyError | drop |

Nothing in the table should be fixed before the tag. The fixes to make before tagging are I1 (two sentences in two
records, optionally one record field) and the record / spec wording in M1, M2, M3, M6, M8; M4, M5, M7, M9, M10, M11, M12
can go with the next touch of their files.
