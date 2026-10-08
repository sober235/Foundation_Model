# Scoped re-review of the final-review fix commit 01e01aa (2026-10-08; reviewer: mid-tier model, read-only)

Scope: `final-review-1.md` findings I1 and M2–M11 plus the launch.md number; diff `3a2ea0c..01e01aa` (9 files). M1 (the tag)
was resolved by setting the tag after this review; M12 (masks in the student's nnU-Net input folder) is parked.

### Spec Compliance
- ✅ Spec compliant. I1 and M2–M11 and the launch.md number are all addressed, and nothing outside them changed, apart from the extras listed below.

**I1: addressed.** `README.md`: the false sentence is gone; it now says the record reports `reliable_slices` and
`box_in_reliable_slices`, that the binder itself is not restricted to it, and that a box in slices 0–1 is bound on unjudged
labels so the caller must gate on the flag; it also says the range comes from the student's own outline. `eval/README.md`:
"outside the reliable range the binder uses" replaced by "which the inference record reports; the binder itself is not
restricted to it". Code: `anatobind/infer/brain_anatomy.py` adds
`record["box_in_reliable_slices"] = bool(len(reliable) and reliable.start <= z0 and z1 <= reliable.stop)` — correct:
`reliable.stop` is exclusive and the box occupies z0..z1-1; an empty range gives False (the `len` guard is redundant but
harmless); the flag sits inside `if box is not None:` before the bind call. Test asserts `is True` for box z 3–5 inside 2..6
(only the True case). Plan: identical code line, test line and the M5 comment fix.

**M2: addressed** (eval/README: "… 13 (15), 8 (4) or 14 (1)"). **M3: addressed** (ventricle Dice for the simulated set only,
volumes in `ref_vs_student.txt`, the 0.319 bound). **M4: addressed** (simulated coverage column marked ad hoc, the recount
listed; "within 1–3 units" is loose — basal ganglia slice 4 is .47 against .51, first-slice median area 159 against 157; the
table cells themselves are not annotated, only the note below). **M5: addressed** (14 → 15 in simulate.py and the plan).
**M6: addressed** (spec A2 358 cases / 185 patients with the note; §3 table 358). **M7: addressed** (spec §5 step 7, 2nd order,
six terms). **M8: addressed** (spec A11 ② and §7: 13 host classes, ventricles a landmark outside the gate). **M9: addressed**
(spec §5 step 6 tie clause). **M10: addressed** (timing row names the work-directory stamp and the per_stem.json mtime).
**M11: addressed** (spec A11 parenthesis: 433 of 447, 14 excluded, 30 boxes outside the denominator). **launch.md:
addressed** (0.9742).

Extras in the diff (not findings): spec A2 gains a pointer to `sources.py::EXCLUDED_SIBBMS`; spec A11 gains the "(原文误写 14)"
note; the README I1 bullet is one long line. No numbers or verdicts were altered.

### Issues
- Critical: none. Important: none.
- Minor: `tests/test_brain_anatomy_infer.py` asserts only the True case of `box_in_reliable_slices`; add a box with z0 < 2
  expecting False when the file is touched again.

### Assessment
**Task quality: Approved.** I1 and M2–M11 are fixed as claimed; the new flag expression is correct for normal and empty
ranges; code, plan and test agree.
