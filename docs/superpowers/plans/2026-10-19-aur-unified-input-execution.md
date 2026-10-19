# Unified 2D/3D AUR implementation and bounded readiness plan

**Authorization:** On 2026-10-19 the user requested `superpowers7-executing-plans` and execution of the proposed follow-up work. This authorizes the shared-body, dimension-specific stem/up, single-checkpoint route described in `../specs/2026-10-19-aur-unified-2d-3d-dicom-design.md`. Implementation details below are engineering choices, not additional user requirements.

**Starting revision:** `86e938e22aad9ef349bd3b36e558a10f836edea0`; existing linked worktree `Foundation_Model-aur-diagnostic`, initial tracked and untracked diff empty. Existing 3D A/U/R failures and NOT_EVIDENCE remain in force.

**Goal:** One model and checkpoint accepts true depth-one 2D and 3D inputs, with shared A32/U/R parameters; establish a reproducible TRAIN-only I/II/III technical pilot, geometry/provenance contracts, and separate A/R diagnostic and evaluation preparation. No UI, deployment, formal long training, test-selected model changes, or invented human labels.

**Constraints:** Read medical data only from `/data2/congcong/data/FM_data`; outputs and checkpoints in private directories under the user's home, never Git. Preserve all authoritative DICOM files and metadata. Report derived-plane tests separately from native DICOM acceptance. Keep legacy 3D defaults, parameter names, loss, sampling and R geometry unchanged. GPU selection uses live utilization/memory/processes and respects visibility/schedulers. CPU checks hide CUDA, use nice19 and two numeric threads.

## Task 1 — Shared model and strict migration

**Files:** `anatobind/aur/{swin,heads,model}.py`, `anatobind/aur/ssl/{model,heads}.py`, new `anatobind/aur/unified.py`, `tests/test_aur_unified_model.py`.

- Add opt-in `dual_input` and explicit `input_mode` dispatch; 2D requires D=1 and patch depth1; legacy 3D stem/up retains its names and parameters. Use the selected patch for valid/coordinate pooling, SSL foreground masks and reconstruction. Other layers are shared objects, not cloned models.
- Strictly migrate legacy model/backbone state only when the exact additional 2D adapter keys are absent. Reject missing shared keys, extra keys or shape differences. Never describe newly random adapters as pretrained.
- RED then GREEN: one instance produces A/U/R for both modes, finite backward into shared blocks and selected adapters, no inactive-adapter update, single-checkpoint round trip, malformed modes/depth refusal, exact legacy 3D eval output equivalence.
- Verify related backbone/head/SSL tests and record actual commands; inspect complete diff before Task 2.

## Task 2 — Geometry, derived planes and DICOM intake

**Files:** new `anatobind/aur/{unified_data,dicom,unified_infer}.py`, `tests/test_aur_{unified_data,dicom,unified_infer}.py`, `scripts/inspect_aur_dicom.py`.

- Derive TRAIN planes only after existing source-qualified patient splits. Record provenance/affine internally; keep image, valid, A, U and ignore aligned. True 2D stays depth1 and has area units, not invented volume.
- Source 3D instance qualification remains intact. Split visible planar instance components; use an explicit configurable minimum area (pilot default 1 mm2), ignore smaller components, refuse query overflow. Unknown/unreliable or out-of-plane R supervision is disabled per instance; never converted to no-host. Carry `r_eligible` aligned with component splitting and target renumbering; retain all U matches, then separately filter relation query/target indices by `r_eligible[ti]`. Initial conservative derived-plane pilot disables R unless explicitly supported by in-plane targets. Test mixed eligible/ineligible targets and all-ineligible cases without disabling their U supervision.
- Support classic MR single plane and regular same-series spatial stacks using physical positions; preserve source bytes by read-only reference/hash. Reject ambiguous temporal/echo/diffusion stacks, mixed series, duplicate positions, missing/nonfinite geometry and irregular grids. Enhanced objects may be explicitly unsupported until independently tested; no silent fallback.
- RED/GREEN fixtures cover LPS/RAS, oblique axes, anisotropy, ordering, rescale and source round trip. Inventory real supplied DICOM with bounded metadata-only inspection; if inaccessible, record the native-DICOM gap without claiming validation.
- Connect the validated input's original array/affine to mode-explicit sliding-window inference using the same model object. Keep legacy 3D decoder behavior; 2D instances use connected-plane area filtering and report area only. Bind decoded instances on the same observed grid and preserve source references privately. Test original-grid shape/geometry and both modes through a single loaded model.

## Task 3 — Recoverable alternating-mode training

**Files:** new `anatobind/aur/unified_train.py`, `scripts/aur_unified_pilot.py`, `tests/test_aur_unified_training.py`.

- Explicit single-process technical-pilot runner, one optimizer and fixed alternating homogeneous modes. Refuse distributed/formal use until those acceptance contracts exist; do not silently adapt the old DDP launchers.
- Stage I uses unified MIM (contrast weight0 for this technical pilot); II strictly inherits unified backbone and III the unified II state. Existing supervised losses and stage parameter groups are reused. Inactive adapters have grad=None before optimizer.step, including after historical optimizer moments exist.
- Versioned checkpoint stores model/config/stage, optimizer, scheduler, CPU/CUDA/Python/NumPy RNG, point/mask generators, next mode/data cursor and per-mode exposures; reject changed manifest hashes, crop/model/loss/schedule/seed contracts. Use fresh output names. Never load untrusted external pickle.
- RED/GREEN: CPU uninterrupted versus interrupted/resumed alternating steps produce exactly the same subsequent draws, loss, LR, exposure and parameters; legacy-to-unified initializer lineage, I→II→III strictness, supervision masks, capacity and immutable configuration. CUDA checkpoint restoration itself must be exact. Subsequent CUDA backward steps are not promised parameter-identical: actual same-seed uninterrupted controls drift, and strict deterministic mode fails at PyTorch2.5.1 trilinear backward. Keep this observed limitation, matching backend contracts, and the original loss tolerance; do not loosen a parameter tolerance to hide it.

## Task 4 — Bounded technical pilot and independent diagnostics

**Files:** `scripts/aur_unified_pilot.py`, new `scripts/aur_diagnose_a_queries.py`, `scripts/aur_diagnose_r_geometry.py`, focused regression tests, anonymous evidence in `docs/verification/2026-10-19/aur_unified_input/`.

- Run CPU synthetic integration first. Then check live GPU state twice and run a bounded TRAIN-only tiny-config real-data pilot (at most six source-qualified TRAIN patients, two modes, no validation/test tuning). Freeze seeds, crops, point count, steps and LR before launch; save source/input/code/config hashes and private logs. Declare tiny-pilot initialization and parameter scale honestly. A performance gate is not inferred from a successful optimizer step.
- A diagnostic reports anonymous per-class dense probabilities, positive coverage and winner margins on frozen TRAIN crops. R contrast fixes tokens/query/weights and swaps coarse versus aligned dense predicted A/U geometry; report logits/features, no new clinical claim. These diagnostics do not change default production paths.
- Record runtime, memory, finite losses, gradient participation, save/resume verification, and what remains before formal unified Stage I/G1 and native DICOM acceptance.

## Task 5 — Evaluation preparation, review and handoff

**Files:** new `docs/verification/2026-10-19/aur_unified_input/{RESULTS.md,RUN.json,evaluation_readiness.json}`, update `STATUS.md`, `CLAUDE.md` and this checklist.

- Prepare separate 2D/3D×A/U/R evaluation requirements and patient exposure inventory; historical test and derived views are not a new independent cohort. Level R tooling readiness does not replace actual reader labels; no messages sent to readers.
- Full CPU suite, affected environment/GPU checks where relevant, fresh-context full-change review and compatibility/security review of checkpoint and DICOM paths. Resolve substantive findings and rerun affected checks (maximum three review rounds).
- Inspect privacy-safe staged diff. Commit verified changes on the existing work branch; do not merge/push/deploy without completing relevant checks and a concrete handoff. Preserve recovery: opt-in unified runner; legacy 3D interface/checkpoints remain available.

## Validation commands and completion ledger

Environment prefix: `CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python`.

Targeted: prefix + `-m pytest tests/test_aur_unified_model.py tests/test_aur_unified_data.py tests/test_aur_dicom.py tests/test_aur_unified_training.py -q -rs -p no:cacheprovider` (only after each listed file exists). Full: prefix + `-m pytest tests/ -q -rs -p no:cacheprovider`. Pilot CLI help must be checked before GPU execution. Actual RED/GREEN outputs and any sandbox-related reruns are recorded in RUN.json.

- [x] Baseline suite observed: 1345 passed, 2 skipped, 177.86s outside sandbox; original sandbox attempt interrupted while blocked in local multiprocessing communication. Focused model/Swin/SSL baseline: 16 passed, 5.95s. Plan independently reviewed: one medium finding on per-instance R eligibility resolved in Task 2; no remaining plan blockers.
- [x] Task 1 core verified: 18 new cases plus existing model/backbone/heads/SSL/masking checks, 39 passed in 6.96s; legacy outputs exactly equal after explicit migration. Final focused extension recorded separately.
- [x] Task 2 verified: data/inference/classic DICOM integration passed; independent review's nested diffusion and ambiguous temporal-dimension cases were reproduced and fixed, with strict geometry/anonymous errors preserved.
- [x] Task 3 CPU and synthetic CLI verified; real TRAIN tiny pilot completed I/II/III24 steps each (72 total, each stage12/12 mode exposure). CUDA exact restoration/environment checks and final full-suite validation are recorded in final evidence.
- [x] Task 4 executed: real TRAIN tiny I/II/III pilot; three full-C2 standard-crop shared A/R forwards; original native replay FAIL retained and separately versioned stable CPU derivation/replay PASS. Original images, checkpoints and native archives unchanged; native-DICOM/planar-human-label limitations explicitly recorded.
- [x] Task 5 verified: final CPU1512 passed/5 skipped; separate CUDA4 passed; nnDetection dedicated environment2 passed. Review findings resolved, last anonymous exception boundary regression verified after the three-round cap; versioned results/readiness/recovery documents written. Source-control checkpoint follows final staged privacy/diff inspection.
