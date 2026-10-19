# Project instructions

Read `CLAUDE.md` and `STATUS.md` for the current project contract and state.

## Private diagnostic evidence

- Private manifest **keys can be patient filenames**. Never dump private selections/manifests, list their keys, or print their values in tool output. Parse them silently; emit only explicitly whitelisted anonymous counts, booleans and hashes.
- Keep medical arrays and per-patient/query/component records in protected private directories, outside Git and tool output. A schema inspection is not an exception to this rule.
- For the frozen TRAIN U diagnostic, use `scripts/aur_diagnose_u_supervision.py --out <private-run> --replay-only` for complete integrity/numeric checks; its output is deliberately fixed and anonymous. Follow the recorded contract instead of dumping archive contents or repeating GPU inference.
- For the completed paired background-allocation tangent, use `scripts/aur_diagnose_u_background.py --out <private-run> --replay-only`. This preserves the complete numeric archive and needs no GPU, model construction or checkpoint deserialization; do not repeat its twelve image forwards to recover already-cached measurements.
- Legacy tracked image filenames can also contain case identifiers. For history audits, use allowlisted Markdown documents and filtered scope quotations; do not dump broad repository filename inventories.
- Historical source/hash contracts remain immutable. Synthetic replay tests must freeze their own temporary source tree and manifest; run actual historical replays from their frozen checkout instead of updating historical hashes to accept feature code.
- Production medical-output CLIs must validate and write the same normalized path, outside Git and under the private home root. Argument errors must not echo raw private values.
- Decode categorical NIfTI arrays with `anatobind.aur.labels.discrete_labels`, including proxy scaling, before resampling or target construction. Direct integer casts can erase near-one lesion labels. Repair already-truncated derivatives into a new private data version; rounding an existing all-zero derivative or resuming its old export cannot recover lost labels. Disease-cohort empty masks need explicit negative provenance before they become supervised negatives.

## Scope and training records

- The final input scope includes standalone 2D and 3D MRI with complete A/U/R. Report coverage separately for both modes and all three tasks. Deferred work remains an explicit unfinished requirement; a 3D subplan never replaces the full goal.
- Distinguish user requirements, proposed architecture/recipes, and observed execution. Before a new formal run, record its initialization lineage, mode-specific data/splits/evaluation, budget and recovery; historical commands or passing code tests do not establish training readiness.
- Distinguish exact checkpoint/RNG restoration from future CUDA parameter reproducibility. Record numerical backend settings; CUDA trilinear-backward nondeterminism is not removed by restoring RNG. Keep failed numerical controls and their declared tolerances.
