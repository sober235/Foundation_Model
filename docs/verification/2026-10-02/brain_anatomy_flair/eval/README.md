# S4 evaluation on real fastMRI FLAIR stacks: what the numbers mean and what was seen (plan Task 9 Steps 8–9)

Everything here compares the student against SynthSeg pseudo-labels on the same real stacks: NOT_EVIDENCE. The final
judgement of S4 waits for the Level R reader labels (spec A12). Nothing was tuned after this run (plan Task 9 Step 11).

Run: 2026-10-04 02:33–02:49, GPU 7, code f220389 (`eval_output.txt` is the raw output; `REPORT.md`, `verdict.json`,
`per_stem.csv/json` are the script's files; work directory `/data2/congcong/data/FM_data/derived/brain_anatomy/eval_20261004_0233`).

## Verdict by the pre-registered rule (spec A11): fail

| gate | value | threshold | pass |
|---|---|---|---|
| host agreement (lesion boxes, student lookup against SynthSeg lookup) | 0.9215 (881 / 956) | ≥ 0.90 | yes |
| mean sided-host Dice on reliable slices | 0.2665 | ≥ 0.80 | **no** |
| outline Dice on the outline model's test stacks | 0.9781 (88 stacks, min 0.8272, 4 under 0.95) | ≥ 0.97 | yes |

Definitions, so that the numbers can be re-derived (`anatobind/eval/brain_anatomy.py`):

- *Reliable slices*: the slices of a stack that the outline rule trusts on the SynthSeg map (`supervised_slices` of the
  filled largest component: area ≥ 500 mm², from slice 2 up to one below the top). Here every stack starts at slice 2; the
  top is 11 (183 stacks), 12 (141), 10 (70), 9 (19), 13 (15), 8 (4) or 14 (1).
- *mean sided-host Dice*: for each stack and each of the 13 host classes (sided white matter, cortex, thalamus, basal
  ganglia, cerebellum, other deep grey; unsided brainstem), the Dice between the student's labels and the SynthSeg labels
  mapped to the same 16 classes, over the reliable slices only; a class empty in both maps is skipped for that stack. The
  per-class mean is over the stacks where the class occurs in either map; the gate value is the mean of the 13 per-class
  means. Ventricles are not a host and not in the gate: their Dice is reported for the simulated test set only, their
  volumes on the real stacks in `ref_vs_student.txt`. (Had they been a 14th class, the mean could be at most
  (0.2665 × 13 + 1) / 14 = 0.319 — still a fail.)
- *Host agreement denominator*: the registry holds 1297 lesions in 165 stacks; 30 lie in the 14 stacks the outline model
  excluded (`excluded.json`, SynthSeg outline under 300 mL) → 1267; 311 have a slice outside the reliable range (slices 0, 1
  or above the top) and are counted but not evaluated → 956 evaluated, 881 agree. Host = `BrainLookup` with
  `BRAIN_PARENCHYMA` candidates on each map, side-agnostic class. By SynthSeg host: white matter 846 / 855 agree, cortex
  35 / 100, basal ganglia 0 / 1; the disagreements are cortex → white matter (65), white matter → cortex (7), white matter
  → basal ganglia (2), basal ganglia → white matter (1). No left/right swap among them.

Per-class Dice on the reliable slices (from `REPORT.md`): white matter L 0.8334 / R 0.8360, cortex L 0.7798 / R 0.7787,
thalamus L 0.0240 / R 0.0031, basal ganglia L 0.1345 / R 0.0748, brainstem 0.0000, cerebellum 0.0000 / 0.0000, other deep
grey 0.0000 / 0.0000. Per stack: median 0.418, 10th–90th percentile 0.298–0.631, min 0.094, max 0.851.

## Why the middle gate fails: the reference holds almost none of the deep structures (post-hoc diagnosis, 2026-10-08)

`checks/eval_ref_vs_student.py` → `ref_vs_student.txt` (+ `ref_vs_student_rows.json`, one row per stack). Inside the
reliable slices, mean per stack over the 433 stacks:

| group | SynthSeg reference, mL | student, mL | stacks where the reference has it | where the student has it |
|---|---|---|---|---|
| white matter | 212.8 | 236.2 | 433 | 433 |
| cortex | 252.2 | 247.5 | 433 | 433 |
| thalamus | 0.06 | 1.05 | 79 | 233 |
| basal ganglia | 1.30 | 3.38 | 370 | 408 |
| brainstem | 0.03 | 0.00 | 140 | 0 |
| cerebellum | 0.02 | 0.00 | 132 | 2 |
| other deep grey | 0.04 | 0.00 | 198 | 2 |
| ventricles | 7.03 | 15.21 | 367 | 417 |

Nine of the thirteen host classes carry a mean of 0.02–1.3 mL per stack in the reference, i.e. a few voxels in a few
slices; brainstem, cerebellum and the other deep grey structures are not in these stacks at all (the stacks cover the
top ~65–80 mm of the head), and what the reference labels as such are stray voxels. A class whose reference is a few
stray voxels yields a Dice near 0 wherever either map has any voxel of it, and the gate averages the 13 classes with
equal weight. For the two classes that both maps hold in every stack the Dice is 0.78–0.84 (cortex is itself under
0.80 on both sides). Nothing in this paragraph changes the verdict; it says what the number measures.

## What was seen in the montages (USER_REPORTED — my reading of the pictures, not a measurement)

`montage/` (6 stacks: slices 0, 1 and the two above the reliable range; `checks/eval_montage.py`) and `low_slices/`
(the same 6 stacks, slices 0–5; `checks/eval_low_slices.py`). Rows: stripped stack / student / SynthSeg.

- Slices 0 and 1 (the lowest two, not judged, A12): the images show the lateral ventricles (frontal horns, bodies), the
  thalami and the lentiform nuclei in all six stacks. The student labels them as ventricles, thalamus and basal ganglia
  with the expected shapes and the correct sides. The SynthSeg reference there has no ventricles and no thalamus, leaves
  grey unlabelled holes in the centre and spreads cortex over the posterior half. So in these slices the student looks
  right and the reference wrong.
- Slices 2–5 (judged): the student follows the ventricle bodies and the top of the caudate / thalamus where the image
  shows them; the reference labels the ventricles when they are large (file 203_6000922) and misses them when they are
  narrow (200_6002425, slices 2–3). White matter and cortex of both maps agree in shape, which is what the 0.78–0.84 say.
- The two slices above the reliable range (13 and 14 in the montages): slice 13 shows the vertex (two paramedian gyri);
  slice 14 shows a larger lobulated blob with a smaller round one beside it, which does not look like scalp above the
  vertex. It may be that the last slices of some fastMRI stacks wrap to the other end of the head (then the student's
  few brainstem / cerebellum voxels there would be right). Not examined further: these slices are outside every gate and
  outside the reliable range (which the inference record reports; the binder itself is not restricted to it). Worth a
  look when the Level R reader sees these stacks.
- Lesion 201_6002981 (a large mass with oedema): the student keeps the ventricle, thalamus and basal ganglia of the
  healthy side and labels the oedema as white matter; the reference labels part of the oedema as cortex. Consistent with
  the 65 cortex → white matter disagreements, where the reference calls a juxtacortical white matter lesion cortex.

## What this means for the next step (for the user to decide; no tuning was done)

The rule says fail, and the number that fails measures mostly the reference's lack of deep structures in the reliable
slices, not a visible defect of the student. Three readings are possible and the choice is the user's: (1) take the fail
at face value and iterate on the simulation (A6 / A7); (2) judge the deep classes only where a reference exists — the
Level R reader (A12), which was the plan's final judge anyway; (3) restrict the Dice gate to the classes the stacks
actually contain. Readings (2) and (3) are post-hoc and would have to be written into the spec as amendments before any
number is quoted as a pass.
