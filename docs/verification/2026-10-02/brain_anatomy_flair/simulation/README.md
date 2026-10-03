# Simulated fastMRI-style stacks: what was generated and what it looks like (2026-10-03)

`scripts/brain_anatomy_prepare.py --stage simulate --workers 8` (code 5e26cac, log `../build/simulate.txt`): 22:08:11–22:34:44,
1320 cases → 4140 training samples (K = 4 per training case) and 285 test samples (one per test case), 23 GB under
`/data2/congcong/data/FM_data/derived/brain_anatomy/sim/`.

Sampled parameters over the 4425 samples (`sim/manifest.json`): 16 slices in 3953, 14 in 472; in-plane 0.6875 mm in 3643,
0.86 mm in 782; matrix 320 × 320 in 2630, 276 × 276 in 958, 260 × 320 in 837; empty slices above the brain 1 / 2 / 3 / 4 / 5
in 206 / 1546 / 1948 / 581 / 144; ignore voxels (lesions) in 3025 samples (BMSR 1382, PDGM 1643; SibBMS has no same-grid
lesion masks). The bottom never had to be clamped to the volume's first slice.

## Montages (`sim_<source>_<sample>.png`, made by `../checks/sim_montage.py`; USER_REPORTED: looked at by the controller)

Image on top, student labels below (15 = ignore, light). Seen in the six montages: stacks run from the basal ganglia level
to above the vertex; the patient's left is on the image's right half (high column indices, as in the fastMRI RSS frame);
tumour and metastases are ignore; the last one or two slices are empty; the BMSR case (resampled from 1.5 mm) shows no
resampling artefact.

## Coverage beside the fastMRI stacks (`../../s4_probe2`), 240 simulated 16-slice stacks

| | fastMRI (SynthSeg pseudo-labels on the stacks) | simulated (teacher labels) |
|---|---|---|
| brain in the stack (slices with > 5 cm²) | 65 mm (median) | 65 mm (median) |
| empty slices above | 1 / 2 / 3 / 4: 19 / 157 / 191 / 63 of 447 | by construction the same distribution |
| median labelled area per slice (cm², slice 0 → 13) | 126 139 147 151 149 142 133 120 105 88 69 46 23 1 | 157 157 155 150 144 135 123 108 92 76 57 36 15 1 |
| basal ganglia present (slice 0 → 5) | 0.81 0.77 0.79 0.64 0.37 0.14 | 1.00 0.96 0.90 0.73 0.51 0.25 |
| thalamus present (slice 0 → 3) | 0.16 0.25 0.16 0.06 | 0.92 0.80 0.60 0.33 |
| cerebellum present (slice 0 → 2) | 0.59 0.21 0.18 | 0.14 0.03 0.00 |
| bottom slice area / largest slice area | 0.83 (median) | 0.98 (median) |

**Open calibration question (not resolved, reported as is).** The two columns cannot be reconciled by pseudo-label
statistics alone, because the fastMRI pseudo-labels are unreliable in the lowest slices and at the vertex — the very
reason S4 exists. From slice 4 upwards the fastMRI area profile equals the simulated one shifted by one slice
(fastMRI slice k ≈ simulated slice k − 1: 149/150, 142/144, 133/135, 120/123, 105/108, 88/92), which would mean the real
stacks sit about 5 mm lower on the brain than the simulated ones (more cerebellum in slice 0, as the fastMRI column
shows); the basal ganglia and ventricle columns point the other way (they end about one slice earlier in fastMRI),
which under-labelling in 5 mm slices also explains. The simulated bottoms vary over ± 10 mm (empty top 1–5 slices), so
either reading lies inside the training range, but the lowest real slice may be under-represented. This is the first
thing to look at if the evaluation shows the student failing in slices 0–1 (spec §11); it is not tuned now (A11: no
tuning to pass a gate, and no reference exists to tune against until the Level R labels).
