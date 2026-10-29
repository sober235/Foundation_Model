# 128-query release and formal continuation — 2026-10-28

Status: RUNNING (snapshot below). Engineering only; no paper work. All metrics NOT_EVIDENCE.

## Completed evidence

Implementation 506dbdc parameterizes capacity, introduces explicit original LR5 Stage I → 128-query Stage II initialization, and preserves old64/default certificates and strict transitions. The incremental probe-entry fix removes an accidental `out=None` before output-path validation; regression checks new owned output creation and existing-output refusal.

Completed probe_v5: II8 and III8, each mode4 observations, all losses finite; complete U matching including65/85 stress observations and eligible R in both modes. Read-only replay on2026-10-28 verified plan/audit binding, checkpoint/export/completion identities and weights, and opening/closing commitments. No new model forward or training was performed for recall.

Full CPU historical run:2129 passed/5 skipped in350.97s. Later focused90/46 checks are separate evidence, not added to the full count. Current incremental regression:4 passed in7.70s; independent reviewer4 passed in7.38s,0 actionable findings. Reviewer checked final controller hashes, transfer digest and145 parent/147 child source commitments. Parent completed checkpoint deserialization/verification separately.

## Resources and limits

Probe II training/validation/save44.90s, data verification415.21s; III45.96s and411.17s. Mean step time II2D1.084s/3D4.118s, III2D1.167s/3D4.138s. Maximum CUDA allocated about5.25GiB, reserved8.40GiB. These selected observations do not provide a reliable full-budget duration projection. Separate validation and save timings were not captured. This probe does not establish current128-query CUDA restoration; prior CUDA restore evidence and synthetic strict-resume checks remain historical/scoped evidence.

## Authorized bounded formal run

Use registration_v5 and its frozen runtime; original LR5 formal StageI parent (selected1792), never probe weights or old1536 supervised weights. Formal II4096 → III2048; stage time caps14400/7200s, controller28800s. Same data/crop/loss/sampler. Fresh source/data/parent verification is mandatory in the controller and each worker before updates; physical GPU idle checks repeat immediately before model initialization.

Keep caps as a bounded attempt, not a claim of completion feasibility. Stop on failure/timeout; preserve all outputs. No automatic additional budget. Formal III starts only after verifying formal II certificate. No duplicate launches. Oldrun_v2/run_v3, parent runtime and probe remain immutable.

## Remaining

Launch and confirm both-mode updates/checkpoint. Then formal completion/certificates, development workpoint and head/lookup freeze, qualified historical-test final A/U/R gates and4469 fastMRI execution. No heldout arrays are read during release. Protocol is not an explicit input; spatial DICOM geometry processing remains. A/U/R and2D/3D goals unchanged.

Private locator: `~/outputs/aur_lr5_formal_supervised_20261026/`; registration_v5.json, probe_v5/result.json, probe_v5_replay_verification_20261028.json. Anonymous release state/hashes in RUN.json.

## Launch snapshot

2026-10-28 09:53:02 UTC: run_v5 launched, host PID1743973, idle A800 physical2 selected by UUID. Runtime import origin verified. This is pending admission, not observed training progress; no completion claim. Private launch_record_v5.json preserves launch identity.

## First trained checkpoint verified

2026-10-28 10:09:26 UTC: Stage II267/4096, exposures2D134/3D133, all267 recorded losses finite. Durable checkpoint256 SHA256 verified; deserialized purpose=formal, stage=II, exact registered128 model/config/runtime identity and all floating model weights finite. This is saved-state validation, not a CUDA resume experiment or final quality score. StageIII not yet started. Private progress_snapshot_v5_first_checkpoint.json records the hash. StageII must complete and verify before automaticIII starts; final A/U/R gates and fullfastMRI remain outstanding.
