# Independent review of Stage I repair round 1 (commit 61b915a) — 2026-10-10

Reviewer: a fresh general-purpose agent, read-only, CPU only (it ran the SSL tests, the new tests against the reverted decoder, two mutation tests, a per-stage gradient balance on 12 real crops for the round-1 model at init and the run-0 checkpoint at step 2000, a stage-1 reach measurement on real masks, the float16 cast check of the G1 features, the resume-to-backbone conversion, and the bash wait/kill semantics). Verdict: the decoder change is correct, has no leakage and is a fair single-change test; do not launch before fixing the early rule and the probe-failure handling.

| # | Severity | Finding | Done |
|---|---|---|---|
| 1 | ok | No path carries a hidden voxel into the reconstruction (whole-patch masks, stem replaces hidden voxels, decoder reads levels only); the leakage test covers all four levels (a 1 % stem-leak mutation fails it) | — |
| 2 | nit (pre-existing) | Blocks are intersected with the foreground of view 1: the mask layout carries a foreground silhouette of the hidden region | disclosed in G1.md; not changed in round 1 (a second change) |
| 3 | nit | The decoder leakage test had no positive control | added (a visible change changes the reconstruction) |
| 4 | ok | The defect is fixed; the design is in the SparK / FPN family | — |
| 5 | should-fix | Gradient balance: at init 0.1 × contrast is 3–5 × the reconstruction gradient at every stage; record per-stage balance and the decoder's level ablation at the probes | `anatobind/aur/ssl/attrib.py`, `scripts/aur_ssl_attrib.py`, run at each probe on the CPU |
| 6 | note | F1 alone can reach visible context for every hidden token (stage-1 reach ≈ 16 × 64 × 64 mm), so the deep stages may learn only residuals | the pre-registered round-2 choice includes "drop the F1 lateral" |
| 7 | nit | The no-learning baseline averaged the background into its prediction | baseline over the visible foreground only; disclosed that run 0's 0.361 used the old one |
| 8 | **blocking** | The early rule stops a flat run above random about half the time (single-seed noise ≈ 0.003) and always stops a run below random but rising | 3 seeds; stop only if step-2000 readout < random − 0.01 and rose < 0.01 since step 1000; G1.md amended before any run |
| 9 | should-fix | A probe failure killed the training | retry once, then skip; never stops the run |
| 10 | should-fix | The probe card defaulted to a training card | PROBE_CARD required and refused if inside CARDS |
| 11 | should-fix | With 1 or 2 cards the InfoNCE spans 12 crops, not 36 (a second change) | only 3 or 6 training cards |
| 12 | should-fix | Probe backbones (52 MB each) were written under docs/ | written into the run directory |
| 13 | nit | A 30 s sleep after the resume file appears | wait until rank 0 has logged step ≥ s + 20 |
| 14 | nit | The process-group comment was wrong (torchrun workers run in their own sessions; SIGTERM to the agent stops them) | comment corrected; Ctrl-C leaves the training running, documented |
| 15 | ok / nit | wait/kill semantics, stdout capture, conversion, the batch table, the G1 and C2 hand-off verified; no restart path | documented: resume by hand |
| 16 | nit | A resumed invocation rewrote `run_config.json` and a stopped one wrote `summary.json` | `run_config_resume_step<N>.json`; a stopped invocation writes `summary_stop_step<N>.json`, `summary.json` only when the run is done |
| 17 | should-fix | `stage23_chain.sh` never completed its smoke on real data | start C0 (whose chain begins with the smoke) before round 1's C2 hand-off |
| 18 | nit | C0 on 1 card vs C2 on 2 cards get different data streams | match the card counts when launching |
| 19 | should-fix | The gradient tests only checked "nonzero" (a 1e-6 scaling of the top-down path passed) | per-stage gradient RMS ≥ 1e-3 × stage 1; zeroing each deep level must move the reconstruction |
| 20 | nit | Blocks and merges were asserted together | asserted separately |
| 21 | — | Coverage when the fix is reverted: heads, model-gradient and host-only tests fail; the leakage test is a guard | — |

After the fixes: full suite 996 passed, 2 skipped (CPU).
