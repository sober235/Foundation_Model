# Validation of the saved-evaluation diagnostic

**NOT_EVIDENCE.** This records CPU checks of the diagnostic, not a new model evaluation.

The script was executed with `CUDA_VISIBLE_DEVICES=''`, `PYTHONNOUSERSITE=1`,
`PYTHONPATH=.`, `OMP_NUM_THREADS=1`, `nice -n19`, and the absolute `nvgen`
Python interpreter. It completed in approximately one second. The exact input
paths, source SHA256 values and a reproduction command using a fresh output
directory are in `REPORT.md` and `summary.json`.

Verified on both saved validation evaluations:

- All 405 row identities match, with no duplicate identities and identical saved GT definitions.
- All three gated source/sequence groups match: BMSR/T1c (41 scans), ISLES/DWI (20), and PDGM/FLAIR (41).
- For all six arm/group combinations, recomputed scan counts, counted/ignored GT counts, one-to-one hits, FP counts, sensitivity and FP/scan equal the saved aggregate values, using the repository's existing matching and ignored-GT rules.
- The output JSON parses, contains no case/patient table keys, and records the current script's correct SHA256.

A minimal in-memory synthetic check also passed:

- A unit GT box inside a `100 × 100 × 100` prediction box has box IoU `1e-6`; with saved component volume `100001 mm³`, the geometric containment count is one.
- Empty GT and empty detection sets produce defined empty/zero geometry results.
- A separate detection on an ignored GT is exempted from the FP count after counted GT matching.
- Nonfinite diagnostic values become JSON `null`.
- A new/old membership mismatch raises an error without emitting identifiers.

No model inference, checkpoint loading, image loading, GPU execution, sealed test rerun, or threshold sweep occurred. No persistent test files were added.
