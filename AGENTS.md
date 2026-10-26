# Project instructions

Read `PROJECT_CONTEXT.md`, then the current sections of `STATUS.md` and `CLAUDE.md` for the project contract and state.

## Progress recall scope

- The user requires progress recall from `main` plus exactly one registered work branch: `work/aur-formal-supervised`. `PROJECT_CONTEXT.md` is the authoritative handoff index.
- Read the main handoff and linked current evidence, then only that work branch's delta from main. Do not enumerate/read every branch or reconstruct progress by searching all branch histories.
- Older branches are archives. Access a specific old commit only when the handoff explicitly references it, evidence is missing, or the user requests it. Do not delete archived branches as part of this policy.
- Keep the handoff and the single work-branch pointer current after each milestone; distinguish prepared, running and completed work. Do not repeat completed training/inference merely to restore context.

## Private diagnostic evidence

- Private manifest **keys can be patient filenames**. Never dump private selections/manifests, list their keys, or print their values in tool output. Parse them silently; emit only explicitly whitelisted anonymous counts, booleans and hashes.
- Keep medical arrays and per-patient/query/component records in protected private directories, outside Git and tool output. A schema inspection is not an exception to this rule.
- For the frozen TRAIN U diagnostic, use `scripts/aur_diagnose_u_supervision.py --out <private-run> --replay-only` for complete integrity/numeric checks; its output is deliberately fixed and anonymous. Follow the recorded contract instead of dumping archive contents or repeating GPU inference.
- For the completed paired background-allocation tangent, use `scripts/aur_diagnose_u_background.py --out <private-run> --replay-only`. This preserves the complete numeric archive and needs no GPU, model construction or checkpoint deserialization; do not repeat its twelve image forwards to recover already-cached measurements.
- Legacy tracked image filenames can also contain case identifiers. For history audits, use allowlisted Markdown documents and filtered scope quotations; do not dump broad repository filename inventories.
- Historical source/hash contracts remain immutable. Synthetic replay tests must freeze their own temporary source tree and manifest; run actual historical replays from their frozen checkout instead of updating historical hashes to accept feature code.
- Production medical-output CLIs must validate and write the same normalized path, outside Git and under the private home root. Argument errors must not echo raw private values.
- Private experiment controllers must also redact parser, plan-bootstrap and lock errors before a valid private diagnostic destination exists. Lock contention must not overwrite the owner's status.
- Decode categorical NIfTI arrays with `anatobind.aur.labels.discrete_labels`, including proxy scaling, before resampling or target construction. Direct integer casts can erase near-one lesion labels. Repair already-truncated derivatives into a new private data version; rounding an existing all-zero derivative or resuming its old export cannot recover lost labels. Disease-cohort empty masks need explicit negative provenance before they become supervised negatives.

## Scope and training records

- The final input scope includes standalone 2D and 3D MRI with complete A/U/R. Report coverage separately for both modes and all three tasks. Deferred work remains an explicit unfinished requirement; a 3D subplan never replaces the full goal.
- Distinguish user requirements, proposed architecture/recipes, and observed execution. Before a new formal run, record its initialization lineage, mode-specific data/splits/evaluation, budget and recovery; historical commands or passing code tests do not establish training readiness.
- Keep A decoding definitions explicit: production `infer.predict_volume` thresholds raw entity-mask probabilities and reports entity presence separately. A presence-times-mask diagnostic must be reported separately; never relabel it as production A quality or use it to rewrite an existing gate. Sampled U matching Dice likewise does not replace full-grid or decoded detection metrics.
- Distinguish exact checkpoint/RNG restoration from future CUDA parameter reproducibility. Record numerical backend settings; CUDA trilinear-backward nondeterminism is not removed by restoring RNG. Keep failed numerical controls and their declared tolerances.
- Resolve inherited CUDA/scheduler constraints to proven physical GPU UUIDs before replacing visibility; never equate numeric CUDA/Slurm identifiers with nvidia-smi indices. Preserve constraints and CUDA_DEVICE_ORDER, or stop for an unresolved namespace. Finite GPU workers must terminate when their watchdog parent dies, including nested workers.
- Check process liveness from the job's execution boundary; a missing sandbox-visible PID does not establish an external job's exit.
- Running stage `status.json` is not a per-step heartbeat. Read complete `steps.jsonl` records for the latest step; stage completion still requires its verified certificate and closing integrity records.
