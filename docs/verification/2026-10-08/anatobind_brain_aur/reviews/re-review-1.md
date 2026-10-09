# Scoped re-review of the final-review fix wave (2026-10-09; mid-tier reviewer, read-only)

Scope: `final-review-1.md` findings C1, I1, I2, M1–M13 over the range fa37b3c..c93ee5f (plan regeneration 820c27f, code
commit 47355a7, records / docs commit c93ee5f).

- C1 ADDRESSED: `dataset.load_volume` applies `canonical()` (`nib.as_closest_canonical`) to the image, the anatomy and
  the lesion map; the spacing comes from `spacing_of` on the reoriented affine; the lesion shape check still raises on a
  mismatch; `test_load_volume_reorients_every_source_to_ras` stores LPS copies of all three files (array axes 0 and 1
  flipped, `aff[0,0]`, `aff[1,1]` negated, translation set so every voxel keeps its position; `aff2axcodes == "LPS"`
  asserted) and compares image, entity, instance, small, a_ignore, spacing and host targets with the RAS load; the
  docstring, README, STATUS §3 / §4 and spec §6 state the RAS contract and the inference requirement.
- I1 ADDRESSED (records): `p0/spacing.txt` + `checks/spacing_table.py`; README section and spec §3.1 notes match the table;
  STATUS §2 item 1 puts the crop-scale choice to the user (a / b / c, recommendation b).
- I2 ADDRESSED: README "Known deviations" and STATUS §3 record the DDP remedy for the Part 2 trainer.
- M1–M10, M12, M13 ADDRESSED (details in the reviewer's report: isles_ruling wording and STATUS provisional item; sibbms
  261 / 97; timing from files; `p0/model_params.txt` and spec §5; augmentation note and spec §6 wording; `entity_present`
  from the rotated crop map, stacked by `collate`, asserted by the tests; README bullets M7 / M9; STATUS §3 P4 / P5;
  unused imports, dead variable, probe `t0` / `logged`; `match_events` docstring; `reviews/final-review-1.md` and the
  README pointer). M11 parked for Part 2.
- New breakage in the fix diff: none. `dataset.py` changes only `canonical`, `spacing_of`, the `load_volume` edits, the
  `N_ENTITIES` import, the `entity_present` line and the `collate` key; the other four code files change only as listed.
- Minor (not blocking): the "196 cases at 2 mm isotropic" wording (2 of them are 0.875 mm in-plane; corrected after this
  review); "12:0x" in the README's timing (replaced by the commit time); the LPS test flips axes only, no permutation
  (same code path); the tag line of STATUS is checked at the tag step.

**Approved.**
