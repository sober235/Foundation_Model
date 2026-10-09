# AnatoBind brain A/U/R — Part 1 records (spec `docs/superpowers/specs/2026-10-08-anatobind-brain-aur-design.md` N1–N18, plan `docs/superpowers/plans/2026-10-08-anatobind-brain-aur-part1.md`)

Part 1 built everything a training run needs: the package `anatobind/aur/` (labels, sample table, targets, crops, RoPE,
Swin backbone, heads, relation, losses, model, dataset), the two scripts (`scripts/aur_prepare.py`, `scripts/aur_probe.py`)
and the pre-flight records below. Training, evaluation and inference are Part 2. Every number here is copied from a file
in `p0/` or from a printed output; comparisons between pseudo-label maps are NOT_EVIDENCE.

## Records in `p0/`

| file | what it is | made by |
|---|---|---|
| `samples.txt` | the sample table run: rows per source and split, with the U-supervised counts | `scripts/aur_prepare.py --stage samples --out /data2/congcong/data/FM_data/derived/aur/samples.json` (2026-10-09 11:35, code 7119383's parent 097c480) |
| `grids.txt` | every row's image, anatomy and lesion on one grid | `--stage grids` |
| `isles_check.txt`, `isles_check/host_volumes_median_ml.csv`, `isles_check/isles_*.png` | median host volumes per source (first 30 cases each) and six ISLES montages (DWI / SynthSeg host classes) | `--stage isles_check` |
| `isles_ruling.md` | the P1 ruling: ISLES keeps A supervision | controller |
| `sibbms_note.md` | why no SibBMS row supervises U (10 annotated subjects on a different grid) | controller |
| `probe_single.txt` | memory and seconds per step on one A800, batch 1 / 2 / 4 with and without checkpointing | `scripts/aur_probe.py --gpu 0 …` |
| `probe_ddp.txt` | the four-card DDP probe: the default NCCL run timed out, the `NCCL_P2P_DISABLE=1` run gives the schedule | `torchrun --nproc_per_node 4 scripts/aur_probe.py --ddp …` |
| `spacing.txt` | orientation, spacing and shape of every case's anatomy map per source (final review I1) | `checks/spacing_table.py` |
| `model_params.txt` | parameter counts of the default model (backbone 12 929 792, total 22 719 561) | one-liner, recorded |

## Sample table (`samples.txt`; `/data2/congcong/data/FM_data/derived/aur/samples.json`, 4960 rows)

| source | train cases / rows / rows with U | test cases / rows / rows with U |
|---|---|---|
| PDGM | 400 / 1600 / 800 | 101 / 404 / 202 |
| BMSR | 359 / 1077 / 359 | 102 / 306 / 102 |
| ISLES | 200 / 400 / 200 | 50 / 100 / 50 |
| SibBMS | 276 / 827 / 0 | 82 / 246 / 0 |

Training rows 3904 (the spec estimated ≈ 3900), of which 1359 supervise U (spec ≈ 1360). SibBMS gives 827 rows for 276
sessions: one session lacks a T2w (the table lists the sequences present, `sibbms_samples`). Splits: PDGM / BMSR / SibBMS
from `derived/brain_anatomy/cases.json` (the S4 test patients), ISLES by `split_by_patient(seed=0)` (50 of 250 cases).

## Grids (`grids.txt`)

`4960 rows checked, 0 off their grid` — every image and lesion map shares its SynthSeg map's shape and affine (within
`check_grid`'s 1e-3).

## ISLES SynthSeg on the DWI grid (`isles_check.txt`, `isles_ruling.md`)

Median host volumes (mL, first 30 cases) ISLES against PDGM: white matter 206.8 / 205.0 vs 226.8 / 223.2, cortex 234.1 /
228.6 vs 279.0 / 274.9, thalamus 5.9 / 6.1 vs 7.1 / 7.7, basal ganglia 9.5 / 9.8 vs 10.4 / 11.3, brainstem 20.5 vs 24.6,
cerebellum 58.4 / 57.4 vs 67.7 / 68.4, other deep grey 7.7 / 8.0 vs 10.1 / 9.7 — ratios 0.76–0.92, inside the plan's 30 %
band; the six montages place the hosts where the DWI shows them (USER_REPORTED). **Ruling: ISLES keeps A supervision.**

## Probes and the training schedule (`probe_single.txt`, `probe_ddp.txt`; spec §6 and N16 amended on 2026-10-09)

Crop 128 × 160 × 160, 22 719 561 parameters, bf16 autocast, the full A + S + U + R loss on a synthetic batch:

| configuration | s / step | peak GiB |
|---|---|---|
| 1 card, batch 1, checkpointing | 0.541 | 3.26 |
| 1 card, batch 2, checkpointing | 0.666 | 6.03 |
| 1 card, batch 4, checkpointing | 0.968 | 11.57 |
| 1 card, batch 2, no checkpointing | 0.533 | 9.41 |
| 1 card, batch 4, no checkpointing | 0.862 | 18.4 |
| 4 cards DDP, batch 4 each, no checkpointing, `NCCL_P2P_DISABLE=1` | 0.90 | 18.48 |

The four-card run with NCCL's default path hung in initialisation and hit the 600 s timeout (`exit 124`), as a dry run
had the day before; with `NCCL_P2P_DISABLE=1` it ran 100 steps (NCCL warns "Could not find a path for pattern 4, falling
back to simple order"). Schedule (spec §6): 4 cards × batch 4 = 16 crops per step; Stage II 240 000 crops = 15 000 steps ≈
3.8 h; Stage III 80 000 crops = 5 000 steps ≈ 1.3 h; lr 5e-4 (square-root scaling from 2e-4 at batch 2), 1 000 warm-up
steps. Single-card fallback ≈ 14 h + 4.8 h.

## Tests

`PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`
→ `910 passed, 1 skipped in 87.78s` after the final-review fix wave (872 before this plan; 38 new tests in
`tests/test_aur_*.py`: the plan's 36, plus the sliver test of the batch C fix round and the RAS reorientation test of the
fix wave; 909 at the first documentation commit).

## What changed against the plan's first text (review rounds; every change is in the plan and the dry-run copy)

- Relation geometry: the host side comes from the host's identity (-1 left, +1 right, 0 brainstem) and the event side from
  the midpoint of the left-host and right-host masses (0 when a side is absent), instead of a midline from the crop's
  mass centroid (batch B review).
- Backbone contract: invalid voxels hold the constant -1; a partly valid patch is a valid token that sees that constant
  like an image border (docstring; the dataset tests enforce the fill).
- Losses: half of the mask-loss points are drawn per lesion instance (equal quota), the rest uniformly
  (`losses.FOCUS_SHARE = 0.5`); entity BCE over every entity (absent entities learn an empty mask), Dice over the present
  ones; `bind` gates the entity masks by the entity presence (batch C review).
- Dataset: an instance whose part inside the crop is under the 10 mm³ floor is not an instance of the crop; its voxels
  join the no-loss mask; host targets of a cut instance are those of the whole instance (batch C review).
- The probe test's tautological assertion was replaced by two real ones (plan defect found by the controller).

## Orientation and spacing of the sources (`p0/spacing.txt`; final review C1 / I1)

Orientations as stored: PDGM LPS (501), ISLES LAS (250), SibBMS RAS (358), BMSR RAS (452) + LAS (9). Without a
reorientation the patient's left would lie at low x for some sources and high x for others — an implicit mirroring of
the sided labels between sources. The fix wave after the final review makes `dataset.load_volume` reorient every file
to RAS by axis flips / permutations (`nib.as_closest_canonical`, no resampling) before the (z, y, x) transpose; Part 2's
inference must do the same. Spacings: PDGM and SibBMS 1 mm isotropic; ISLES 2 mm isotropic in 196 cases but 54 cases
with 4.8 mm slices (1.15–1.8 mm in-plane); BMSR 0.43–1.17 mm in-plane and 1–5 mm slices (median 0.859 × 0.859 × 1.5;
9 cases with a 3 mm+ axis). The spec's "1 mm (ISLES 2 mm)" was therefore wrong for BMSR and part of ISLES, and a fixed
voxel crop of 128 × 160 × 160 spans 69–188 mm in-plane and 128–640 mm along z over the sources. The choice (voxel crops as
they are / offline resampling of BMSR and ISLES to a common spacing / per-source crop sizes) is the user's, listed in
`STATUS.md` §2 before the Part 2 plan; the spec §3.1 carries a dated note.

## Known deviations and notes

- SibBMS supervises A and S only: its lesion annotations exist for 10 subjects on a native grid (`sibbms_note.md`).
- ISLES anatomy labels are SynthSeg on 2 mm DWI (coarser than the 1 mm T1 labels of the other sources); kept (ruling above).
- `AnatoBindBrain.DEFAULTS` keeps `use_checkpoint = True`; the measured schedule trains without checkpointing (Part 2 passes
  `use_checkpoint=False`). All other defaults are the spec's (§5).
- The four-card training must set `NCCL_P2P_DISABLE=1` on this host (spec N16).
- `sample_points(..., instance=...)` has no caller yet: the Part 2 trainer passes the crop-local instance map; the crop
  dict carries `entity_present` (K,) for `entity_loss` since the fix wave.
- DDP (final review I2): a Stage II step (no `bind`) or a batch whose U is unsupervised everywhere leaves the relation
  head (and the event decoder) without gradients, and `DistributedDataParallel` raises on the next step. The Part 2
  trainer must freeze `model.relation` in Stage II and touch the event outputs with a zero weight when no sample of a
  rank's batch supervises U (or pass `find_unused_parameters=True` and re-measure: the 0.90 s per step above was measured
  with every part in the loss).
- Augmentation (`crops.augment`) follows S4's recipe in spirit but is a re-implementation on the [-1, 1] scale: bias
  amplitude 0.2 and gamma 0.7–1.4 as in S4, in-plane blur σ 0.3–0.8 (S4 0–0.7), Gaussian noise σ ~ U(0, 0.05) (S4 Rician at
  1–4 % of the brain median), no re-normalisation afterwards.
- The ±10° rotation turns the voxel grid about the array's z axis while the coordinate grids stay as they are (the same
  as rotating the patient in the scanner); the corners that rotate in become invalid (-1).
- ISLES volumes (and the 4.8 mm BMSR ones) are smaller than the crop in every axis: both crops of such a volume per epoch
  are the same centred window and differ by augmentation only; `lesion_centred` does nothing there.
- Deferred minors from the task reviews are listed in the SDD ledger (`.superpowers/sdd/2026-10-08-anatobind-brain-aur-part1/progress.md`)
  and triaged by the final review, kept as `reviews/final-review-1.md` (with the re-review of the fix wave beside it).
- The ISLES montage judgement is controller-viewed (2 of 6 montages); the user has not looked: provisional (`STATUS.md` §2).
- Two scratch paths left by an implementer outside the scratchpad are on the deletion list (`STATUS.md` §2).

## Timing

Plan and dry run 2026-10-08; execution 2026-10-09: first task commit 10:40, last code commit before the records 12:0x
(git log; four batches, two fix rounds), P0 records: samples 11:35:35 → 11:35:38 (3 s), grids 11:35:38 → 11:36:59 (81 s),
ISLES check 11:36:30 → 11:36:51 (21 s, from `p0/*.txt`), probes 12:32:11 → 12:45:44 after waiting for idle cards from
11:40 (`p0/probe_*.txt`).
