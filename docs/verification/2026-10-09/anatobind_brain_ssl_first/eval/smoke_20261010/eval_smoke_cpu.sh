#!/bin/bash
# Real-data smoke of the evaluation stack (2026-10-10), CPU only at the lowest priority, with C0's Stage II export at
# step 2000 (its best validation so far): the chain's own path (the validation split in 2 shards with 1 row per source,
# the merge, the U thresholds it chooses, the test split in 2 shards with those thresholds, the merge), then the fastMRI
# external A evaluation (1 stack) and the SibBMS U evaluation. Outputs go to a new directory on /data2.
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=""
PY="nice -n 19 $HOME/anaconda3/envs/nvgen/bin/python"
D=/data2/congcong/data/FM_data/derived/aur
CK=$D/ssl_runs/c0_stage2_240k_20261010_0903/aur_stage2_step2000.pt
SAMPLES=$D/samples_1mm_v1.json
VAL=$D/ssl_manifest_v1/val_patients.json
OUT=$D/eval_smoke_20261010
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
mkdir -p "$OUT"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/smoke.log"; }
evaluate() {   # evaluate <split> <out> [merge args]: 2 CPU shards in parallel, then one merge
  local split=$1 out=$2; shift 2
  local pids=()
  for k in 0 1; do
    $PY scripts/aur_eval.py --checkpoint "$CK" --samples "$SAMPLES" --split "$split" --val-patients "$VAL" --limit 1 \
      --out "$out.shard$k" --cpu --batch-size 2 --shard "$k/2" > "$out.shard$k.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || { say "a $split shard failed: see $out.shard*.log"; return 1; }; done
  $PY scripts/aur_eval_merge.py --shards "$out.shard0" "$out.shard1" --samples "$SAMPLES" --split "$split" --val-patients "$VAL" "$@" \
    --out "$out" > "$out.log" 2>&1 || { say "the $split merge failed: see $out.log"; return 1; }
}
say "smoke with $CK"
t=$(date +%s); evaluate val "$OUT/val" && say "val done in $(( $(date +%s) - t )) s"
THR=$($HOME/anaconda3/envs/nvgen/bin/python -c "import json; a=json.load(open('$OUT/val/aggregate.json')); print(' '.join(f'--u-threshold {s}={e[\"thr\"]}' for s,e in a['u']['per_source'].items() if e.get('thr') is not None))" 2>&1)
say "U thresholds from the validation split: $THR"
t=$(date +%s); evaluate test "$OUT/test" $THR && say "test done in $(( $(date +%s) - t )) s"
t=$(date +%s); $PY scripts/aur_eval_fastmri.py --checkpoint "$CK" --out "$OUT/fastmri" --limit 1 --cpu > "$OUT/fastmri.log" 2>&1 \
  && say "fastMRI done in $(( $(date +%s) - t )) s" || say "fastMRI failed: see $OUT/fastmri.log"
t=$(date +%s); $PY scripts/aur_eval_sibbms.py --checkpoint "$CK" --out "$OUT/sibbms" --cpu > "$OUT/sibbms.log" 2>&1 \
  && say "SibBMS done in $(( $(date +%s) - t )) s" || say "SibBMS failed: see $OUT/sibbms.log"
say "smoke finished"
