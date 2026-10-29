# Formal 128-query development evaluation and freeze

User authorized execution on 2026-10-29. Continue the completed formal
supervision plan's Task 3; never repeat training, G1 or S7 inference.

Spec: `docs/superpowers/specs/2026-10-22-aur-three-subgoal-acceptance-design.md`.
Baseline: main `4deeead`; only work on `work/aur-formal-supervised`.

## Global constraints and decisions

- Preserve the qualification contract, archived qualification source, frozen
  training runtimes, all exports, failures and historical reports.
- Development validation only. Do not build/read historical_test or start
  fastMRI until this development milestone is complete and reviewed.
- Use formal run_v5 Stage III selected2048 with verified certificate. Keep its
  training identity; snapshot current evaluation code separately.
- First evaluate only S7-covered source/sequence 3D development rows at score0,
  preserving all fixed0.3 connected components. This is a threshold-selection
  pass, not the full A/U/R evaluation. Recompute matching at each0.05..0.95
  cutoff using the existing sensitivity_at_budget rule (maximize sensitivity
  within reference FP/scan, tie highest cutoff; no feasible point is explicit).
- Then run the existing complete development gate observations in both modes
  at those three cutoffs. The initial head registration is diagnostic and
  records a real selection-pass digest; it is not the final deployment choice.
  Both head and lookup are measured by the same records. No cached low-cutoff
  mask Dice or R matches are reused at another cutoff.
- Select one global R implementation by the minimum agreement of the two
  modes, then equal-mode mean agreement, then lookup on ties. Both modes need
  nonzero eligible matched denominators; otherwise do not freeze deployment.
- Persist the original diagnostic registration and records. Final deployment
  registration binds the completed development evidence; never rewrite record
  registration fields or represent validation-selected scores as test results.
- Private records and model outputs stay outside Git; publish only explicit
  anonymous aggregates/hashes. Parser/errors must never echo private values.
- Reuse existing watchdog and UUID GPU admission. Total new evaluation cap12h,
  including admission/checks; stop on timeout/failure, no automatic retry.
  Per-observation progress is a durable anonymous count, not patient identity.

## Task 1: Development selection library

Files: new `anatobind/aur/development_selection.py`,
`tests/test_aur_development_selection.py`.

Interfaces: consumes qualified development contract, committed same-grid S7
reference, verified formal StageIII; produces threshold-selection evidence,
full development gate records, both-track R comparison and final registration.

Write tests first: correct reference intersection, higher-cutoff rematching,
no-feasible budget reporting, equal-mode R choice and unavailable refusal,
reference mutation, non-development membership and incomplete/pilot refusal.
Run focused tests RED, implement, run GREEN plus existing gate/evaluation tests.
Commit the implementation and evidence.

## Task 2: Bounded private controller

Files: new `scripts/aur_development_selection.py`, controller tests in Task1
test file. Bind source snapshot, input hashes, protocol, selected export and
output path before launch; exclusive output ownership; use watchdog parent
death protection and12h limit; status/progress contain anonymous fields only.
Tests must exercise parser redaction, existing-output preservation, source/input
mutation refusal and bounded worker termination. Commit after focused GREEN.

## Task 3: Review, register and execute

Fresh-context reviewer inspects complete change and exact private registration.
Resolve substantive findings. Run project CPU suite with CUDA hidden, nvgen,
nice19 and2 numerical threads; preserve skips. Check current physical UUID
visibility/allocation/load, prefer compatible idle GPU, record immediate
admission before initialization. Run a finite detached controller with a frozen
evaluation source tree. Never execute training or test-role creation here.

## Task 4: Publish observed state

Update PROJECT_CONTEXT and STATUS with actual state, evidence hashes and exact
remaining gaps; record either completed development/freeze or running/failed
status honestly. Inspect private prediction overlays before claiming visual
validation. Final historical-test and4469 fastMRI remain separate next steps.

## Review focus

Do not leak private manifest keys or raw errors. Verify S7 exact row intersection
and source/sequence; actual threshold rematching; ignored GT; strict formal
StageIII provenance; cross-mode R denominator and tie handling; no false final
freeze on empty supports; immutable ownership, input/source rechecks, watchdog
death, output commitments and diagnostic-vs-final registration distinction.
