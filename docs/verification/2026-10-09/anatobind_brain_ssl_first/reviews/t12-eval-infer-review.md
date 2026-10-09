# Independent review of the inference / evaluation batch (T12) — 2026-10-09

Reviewer: a fresh general-purpose agent (read-only; it ran the three test files on the CPU and two CPU probes). Verdict: one blocking defect, the rest reporting / semantics. Findings and what was done:

| # | Severity | Finding | Done |
|---|---|---|---|
| 1 | **blocking** | The mask heads and `bind` ran outside the bf16 autocast: on CUDA the first window raises a dtype error (tests could not see it, autocast is off on the CPU) | heads and `bind` run inside the same autocast as the backbone; verified by hand on a card (tiny and full-size model, `predict_volume` and `bind_instances`); a CUDA-only test keeps it (skips on CPU; CPU bf16 autocast lacks ops in the transformer layers, so a forced-CPU test is not a valid proxy) |
| 2 | should-fix | U pooled sequences with different lesion labels (PDGM FLAIR vs T1c) against one reference | U is reported per (source, sequence); the gate reads the sequence whose labels are the reference's (PDGM FLAIR, BMSR T1c, ISLES DWI) |
| 3 | should-fix | Grid mismatch (AUR 1 mm resampled vs native reference for BMSR / ISLES) not stated | stated per source in `REFERENCE["grid"]` and the report, with the reference's scan and lesion counts beside AUR's |
| 4 | should-fix | The AUR threshold was selected on the test set while S7's was fixed beforehand | `--u-threshold source=thr` for thresholds chosen on `--split val`; otherwise the number is labelled "test-selected (optimistic)" |
| 5 | should-fix | Sub-floor truth components were not `ignore` rows (a detection on them counted as a false positive, unlike S7) | `ignored_rows` from the volume's `small` mask |
| 6 | should-fix | B0* was the truth by construction | dropped; the report says the controlled-track truth is the SynthSeg lookup itself |
| 7 | should-fix | "Recognition–binding gap" was unconditioned; end-to-end R track, Level R sheet and report-only rows missing | gap conditioned on local 13-host Dice ≥ 0.8 in the bind window and a detection matched at the chosen threshold; end-to-end track (predicted lesions bound, matched to a truth at the threshold); `level_r_sheet.csv` (blind) + `level_r_key.json`; SibBMS / fastMRI rows listed as not implemented in the report |
| 8 | should-fix | The A gate included `a_supervised == False` rows | gated rows only; the others reported separately |
| 9 | should-fix | The reference was built after the hours-long loop | built before it |
| 10 | should-fix | Zero overlap bound query 0 | tie broken by the soft overlap and flagged `zero_overlap`; counted in the R block |
| 11 | should-fix | Deployed operating point (map 0.5, no score filter) ≠ evaluated one (map 0.3 + score threshold) | the inference script's map threshold is the evaluated one, `--score-threshold` added, both recorded in `record.json` |
| 12 | should-fix | Record boxes in the canonical frame, NIfTIs in the input's | `box_input` (the written file's axes) added next to `box`, with `box_frame` |
| 13 | should-fix | Full-resolution masks copied to the host per window; the accumulator copied twice | accumulation on the device (0.004 s per window instead of 0.5 s copy + 1.05 s numpy); `host_dice` by one joint bincount; `entity_prob` (32 × volume) no longer returned |
| 14–18 | nit | Gaussian centre, redundant weight, `entity_presence` naming, `_side` for brainstem vs no-host, 4-D inputs, implicit cuda:0 | `_side` fixed, 4-D singleton frames squeezed, `--gpu` / `--cpu` required; the Gaussian centre left as is |

Timing after the fix (GPU 5 while the Stage I run shares the card, so an upper bound): full-size model on a 176 × 256 × 256 volume, 18 windows, 26 s, peak 4.7 GiB; binding 2 instances 3.5 s. The remaining per-window cost is the host-side window preparation (coordinates); 1,056 test rows ≈ 8 h on one card, or split over cards by source.

Untested items the reviewer named and that were added: constant-model blending, the query choice in `bind_instances` (crafted masks), ignored truth rows, `select_rows` and the evaluation script end to end, the Level R export.
