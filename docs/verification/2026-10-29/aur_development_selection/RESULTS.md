# Formal 128-query development evaluation: running

2026-10-29. All results NOT_EVIDENCE. No new training, G1, S7 inference,
historical-test evaluation or full fastMRI execution was started.

## Observed execution

The finite development controller was launched at2026-10-29 07:46:38 UTC
(15:46:38 Beijing), host PID3975686/start ticks449332314. The host boundary
confirmed the detached controller alive. Current observed phase is admission:
the complete qualified data contract and frozen inputs are being verified.
This is not an observed model forward, threshold choice or quality result.

Physical GPU4, A80080GB, was resolved by full UUID and passed the launcher's
idle/no-compute-workload check. The worker independently repeats admission
before moving its verified formal StageIII model onto CUDA. Inherited CUDA/
Slurm GPU constraints were absent; CUDA_DEVICE_ORDER was preserved. Other
workloads were left running.

An earlier prelaunch attempt on physicalGPU2 was refused before any run
directory, controller log or process submission. Its original exception was
not captured. The later snapshot showed a compute workload and40800MiB on
GPU2, but does not establish the original refusal cause. Preserve
launch_refusal_v1.json; no model forwards or training updates occurred in
that attempt. The successful submission uses the same evaluation registration,
with GPU4 selected from the current idle inventory, not a restarted model run.

## Implementation and verification

Code commit `d2ac4b0` adds a development-only selection library, bounded
controller and incremental private observation records. The optional existing
gate-evaluator callback preserves observations when a later step fails.

- Selection/gate/development focused checks:39 passed in24.17s.
- Final controller checks, including real certified synthetic StageIII through
  selection/full development/freeze and StageII refusal:12 passed in12.98s.
- Fresh full project CPU suite on host: **2142 passed,5 skipped,370.93s,exit0**.
  Skips: nndet module unavailable1, CUDA inference1, CUDA optimizer/RNG restore3.
- Independent implementation review found one P2: unavailable R results can
  omit n/agreement. Its actual shape reproduced KeyError in RED; normalization
  now preserves completed development with no deployment freeze. Focused
  post-fix checks passed. No other actionable findings.
- Independent exact release review verified158 source commitments in both
  working branch and frozen evaluation tree, all7 registered input hashes,
  original qualification source commitments, model lineage, policy and crops;
  zero remaining findings.

The first sandbox full-suite attempt was interrupted after566.84s while
parallel-crop DataLoader workers were blocked:765 passed,1 then-unfixed test
fixture failure,2 skipped. Its log remains; it is not a passing suite. The
fixture needed its intended temporary private-root alias. A later assertion
incorrectly assumed random tiny-model detections; it now verifies actual
registration commitments. The host final suite above supersedes both failures,
without changing production behavior to satisfy a test assumption.

CPU command (CUDA hidden; nvgen Python; nice19):

```sh
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 \
  /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/ \
  -q -rs -p no:cacheprovider -o faulthandler_timeout=60
```

## Frozen workflow

1. Full data/source/certificate verification using the original qualification
   archive; preserve the trained export's runtime identity.
2. Score0 3D pass on exactly the completed S7 source/sequence intersection
   (expected41 BMSR,20 ISLES,41 PDGM). Fixed0.3 component geometry is unchanged.
   Recompute GT matching at each0.05..0.95 cutoff; maximize sensitivity inside
   the same-grid S7 FP/scan budget, tie highest cutoff. Infeasible budgets remain
   explicitly infeasible; no lower-quality measurement is relabeled success.
3. Complete405-row development gate observations, independently2D/3D, using
   the selected three source cutoffs. No cached low-cutoff Dice/R matches are
   reused at another cutoff. The initial head registration is diagnostic.
4. Compare head and lookup on the same matched support in each mode. Choose
   by maximum weaker-mode agreement, then equal-mode mean, then lookup on ties.
   Both modes need nonzero supported denominators; otherwise preserve complete
   development results with deployment_frozen=false.
5. Recheck inputs/sources, freeze final deployment registration from actual
   completed development evidence; preserve diagnostic registration/records.

The controller has a hard12h total cap including verification. Timeout/failure
stops the owned worker and retains records; no automatic retry. Watchdog worker
terminates when its parent dies. No training budget or checkpoint was changed.

Execution choices: the separate102-row 3D selection pass costs an additional
forward on those rows but avoids introducing mask-cache/remapping semantics.
The existing linked worktree was clean and switched to the registered work
branch, then fast-forwarded to main. Skill helper scripts were not executable;
the workspace helper ran through bash and command evidence was recorded
directly. Library/controller were committed together because their end-to-end
acceptance is shared. No minor review findings were deferred.

## Private handoff and remaining gaps

Private root: `~/outputs/aur_development_selection_20261029/`.
Frozen code: `execution_v1/runtime/`; registration: `registration_v1.json`;
launch identity: `launch_record_v1.json`; current output: `run_v1/`.
Read only whitelisted anonymous progress/status and hashes in tool output.
Do not dump record/manifest keys, patient identifiers, arrays or per-case data.

Use `run_v1/progress.json` for phase/completed counts, incremental
selection_records/development_records for durable private evidence, and
terminal.json plus complete.json/output hashes for actual completion. Check
controller liveness at the host boundary with PID and start ticks. Do not
duplicate launch if the job is still alive, even during long admission.

Pending: observed model execution, selected U cutoffs, both-mode A/U/R metrics,
head/lookup freeze, private overlay inspection, real historical_test successor
and one final evaluation, and4469-sequence fastMRI dual-mode execution.
Current128-query CUDA resume remains unverified; historical evidence does not
substitute. Development metrics will not themselves establish test passage,
clinical evidence or physician-level accuracy. Protocol and paper remain
deferred. This running milestone is local on the registered work branch;
main/remote were not merged or pushed as part of this continuation.
