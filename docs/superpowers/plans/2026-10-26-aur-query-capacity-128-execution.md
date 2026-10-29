# AUR query capacity 128 implementation plan

> Execute task by task with executing-plans, TDD and one fresh-context full-change review before release.

**Goal:** Preserve all qualified targets while training a registered128-query model from the original completed LR5 StageI.
**Architecture:** Parameterize the target capacity, add a narrow explicit StageI→II transfer across registered runtime identities, and extend the finite external controller. Existing64 default and strict resume remain.
**Tech Stack:** Python, PyTorch, NumPy/SciPy; existing nvgen environment.
**Spec:** ../specs/2026-10-26-aur-query-capacity-128-design.md (user continuation accepted).

## Global constraints

Keep oldruntime, alloldregistrations, run_v2/run_v3 and1536checkpoint immutable. No GT truncation/skipping, no changed crop/loss/sampler. NewformalII4096→III2048; no resume from failedsupervision. CPU nice19/threads2, no private records in stdout/Git. New probeII8→III8 explicitlyresource_probe with complete priorprobe binding; no formal certificate fabrication. Existing GPU authorization and idleUUID checks apply.

## Review focus

Cross-runtime transfer must not accept altered backbone code/weights or changeddata; changing a transfer JSON after loading must fail.128 must reach2D R as well as U. Old64 certificates and strictresume must remainvalid. Probe weights must never releaseformalIII. Saved versus loadedsource and frozenimport origins must match atlaunch andclose.

### Task 1: Capacity and loss regression

Files: unified_data.py, planar_relations.py, unified_run.py, eval.py; tests/test_aur_query_capacity.py and oldcapacity tests.
Interfaces: make_unified_crop(n_events=128) passes max_instances to qualify_planar_weak_relations; config validates1..128 integer preservingdefault64.
- [ ] Write tests preserving65/85/102/128 targets inbothmodes andR qualification;129 rejected; default64 rejects65; Hungarian matches every target.
- [ ] Run newtests RED, implement explicitcapacity propagation, run GREEN including legacytests.
- [ ] Report evaluation saturation capacity frommodelconfiguration; commit focusedchange.

### Task 2: Explicit StageI transfer

Files: new anatobind/aur/query_transfer.py, unified_run.py, unified_launcher.py; tests/test_aur_query_transfer.py.
Interface: UnifiedRun(..., initialization_transfer=None); run_stage(..., initialization_transfer=None) accepts immutableprivateJSONpath archived/hash-bound ininitialization_inputs. validate_transfer(parent,cfg,identity,gate,record) verifies originalparentcertificate/G1 underoriginalidentity, exactdata/geometry, onlyn_events64→128, bothsourceclosures and narrow changed-file allowlist, exactchildcfgdigest andparentexport/G1 hashes. Returns lineage metadata. Legacy paths unchanged.
- [ ] Write refusal tests for changeddata/geometry/model/backbone/source/parent/G1/recipe andmissingtransfer; actualsmall-model loading must preserveeverybackbonetensor andnewoptimizerstep0.
- [ ] Implement explicitbridge; no field rewriting. StageIII uses ordinarystricttransition. Newlineage includesoriginalparent andtransfer commitments.
- [ ] Verify real smallmodel II→III andsavedresume, existinglauncher/certificate tests; commit.

### Task 3: External controller and finite probe

Files: scripts/aur_formal_supervised.py and focusedtests; privateversionedlaunch/probescripts.
Interfaces: optionalregistration initialization_transfer {path,sha256}; finalrecipes derivedfromparent thenonlymodel.n_events=128. StageIIworker receivestransferpath, III no transfer. Admission verifies newruntime andoriginalparent independently, closing rechecks allcommitments.
- [ ] Tests fornewregistration andrecipe mismatch, worker propagation, oldregistrations, noIII onfailure.
- [ ] Freeze newruntime bycopyoriginal andoverlay onlyrevieweddependencyfiles; verify parent sources unchanged.
- [ ] FullCPU suite, independentcode/exactregistration review; resolvefindings.
- [ ] Register finite8+8 actualTRAIN probe including65/85 targets andeligibleR forbothmodes; measuredGPUresources andrestoration evidence. No useprobeweights forformal.

### Task 4: Formal release and handoff

- [ ] Verify budgets feasible usingprobe; freshGPU/source/input admission; launch detachedfiniteII4096→III2048 withnewprivateoutput.
- [ ] Confirm actualbothmodeupdates andcheckpoint; persist currentstate/evidence inPROJECT_CONTEXT/STATUS/RESULTS/RUN. Do notclaimcompletion beforecertificates/closingvalidation. ExistingfinalA/U/R evaluation and4469execution remain downstream ofcompleteformalIII.

## Commands and expected outcomes

Targeted: env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n19 /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest <tasktests> -q -p no:cacheprovider. Expected newregressions failbeforecode, passafter; existingtests pass.
Full: sameenvironment python -m pytest tests/ -q -rs -p no:cacheprovider (hostexecution forDataLoader). Preserve skippednndet/CUDA count, supplement applicableCUDAchecks separately. HistoricalfullCPU2080/5 isbaselineevidence, notfreshresult.

Execution choice: inline implementation with finalindependentreview; the user's repeatedcontinuation already authorizes this plan's concreteimplementation without another workflow-choice prompt.
