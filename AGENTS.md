# Project instructions

Read `CLAUDE.md` and `STATUS.md` for the current project contract and state.

## Private diagnostic evidence

- Private manifest **keys can be patient filenames**. Never dump private selections/manifests, list their keys, or print their values in tool output. Parse them silently; emit only explicitly whitelisted anonymous counts, booleans and hashes.
- Keep medical arrays and per-patient/query/component records in protected private directories, outside Git and tool output. A schema inspection is not an exception to this rule.
- For the frozen TRAIN U diagnostic, use `scripts/aur_diagnose_u_supervision.py --out <private-run> --replay-only` for complete integrity/numeric checks; its output is deliberately fixed and anonymous. Follow the recorded contract instead of dumping archive contents or repeating GPU inference.
- For the completed paired background-allocation tangent, use `scripts/aur_diagnose_u_background.py --out <private-run> --replay-only`. This preserves the complete numeric archive and needs no GPU, model construction or checkpoint deserialization; do not repeat its twelve image forwards to recover already-cached measurements.
