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
→ `909 passed, 1 skipped in 97.09s` (872 before this plan; 37 new tests in `tests/test_aur_*.py`; the plan said 36 — the
dataset test gained one function in the batch C fix round).

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

## Known deviations and notes

- SibBMS supervises A and S only: its lesion annotations exist for 10 subjects on a native grid (`sibbms_note.md`).
- ISLES anatomy labels are SynthSeg on 2 mm DWI (coarser than the 1 mm T1 labels of the other sources); kept (ruling above).
- `AnatoBindBrain.DEFAULTS` keeps `use_checkpoint = True`; the measured schedule trains without checkpointing (Part 2 passes
  `use_checkpoint=False`). All other defaults are the spec's (§5).
- The four-card training must set `NCCL_P2P_DISABLE=1` on this host (spec N16).
- `sample_points(..., instance=...)` has no caller yet: the Part 2 trainer passes the crop-local instance map.
- Deferred minors from the task reviews are listed in the SDD ledger (`.superpowers/sdd/2026-10-08-anatobind-brain-aur-part1/progress.md`)
  and triaged by the final review (`reviews/`).
- Two scratch paths left by an implementer outside the scratchpad are on the deletion list (`STATUS.md` §2).

## Timing

Plan and dry run 2026-10-08; execution 2026-10-09 10:50–12:50: twelve code tasks in four batches (about 1 h 20 min
including two fix rounds), P0 records 11:35–11:37 (samples 5 s, grids 81 s, ISLES check 72 s), probes 12:32–12:45
(waiting for idle cards from 11:40).
