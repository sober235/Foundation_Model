#!/bin/bash
# Stage II -> Stage III -> evaluation for one arm (plan 2026-10-09 §1; decisions Q12, Q16, Q22):
#   ARM=c0  the random-initialisation control (Stage II from a random backbone)
#   ARM=c2  the main line (Stage II from the Stage I export; needs INIT_BACKBONE and a passing G1_REPORT)
# Batch (user, 2026-10-10: fill the cards' memory): MB crops per card (default 12, about 60 GB of an 80 GB card) and a
# global batch GLOBAL (default 24 = 12 x 2 cards; accumulation GLOBAL / (MB x cards) when fewer cards are given). The
# spec's rates are defined at a global batch of 16 (spec §6: Stage II 5e-4 with the backbone at a tenth for the first
# 1000 steps; Stage III 5e-5 / 2.5e-4 / 5e-4, warmup 200 steps): they are scaled by sqrt(GLOBAL / 16) and every warmup
# is kept in crops (steps x 16 / GLOBAL). C0 and C2 must run with the same MB and GLOBAL (PROPOSED, decision Q23).
# WORKERS loader processes per rank (default 24): a Stage II crop costs about 4 s of one CPU core (2026-10-10 profile:
# the in-plane rotation of six arrays, the bias field, the physical coordinates), so a process gives about 0.25 crops/s,
# and a rank needs about 6 crops/s whatever the card count; 8 left a one-card rank at about 12 s a step against 4 s of
# compute. The worker count changes speed only: every crop is seeded by its index.
# Order: a 100-step Stage II smoke that stops at step 50 and is resumed to 100 (T10 acceptance on real data), Stage II
# (240k crops), Stage III (80k crops), the evaluation on the validation split (chooses the U thresholds) and on the
# whole test split, each sharded over all CARDS (scripts/aur_eval.py --shard k/N) and merged once. RESUME_STAGE2 (a
# resume_step*.pt) with STAGE2_DIR (its run directory) continues a Stage II instead (no smoke): same global batch, so
# the same schedule, on any card count. MB defaults to the largest of 12, 8, 6 that divides GLOBAL over the cards.
# Every failure stops the chain and says so; nothing is deleted or overwritten. Run inside tmux:
#   ARM=c0 CARDS=3 tmux new-window -t anatobind -n c0 "bash scripts/stage23_chain.sh; exec bash"
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
: "${ARM:?ARM=c0|c2}" "${CARDS:?CARDS=comma-separated card indices}"
export PYTHONNOUSERSITE=1 PYTHONPATH=. NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
PY=~/anaconda3/envs/nvgen/bin/python
TORCHRUN=~/anaconda3/envs/nvgen/bin/torchrun
DATA=/data2/congcong/data/FM_data/derived/aur
RUNS=$DATA/ssl_runs
SAMPLES=$DATA/samples_1mm_v1.json
VAL=$DATA/ssl_manifest_v1/val_patients.json
REC=docs/verification/2026-10-09/anatobind_brain_ssl_first
STAMP=$(date +%Y%m%d_%H%M)
LOG=$RUNS/${ARM}_chain_$STAMP.log
GLOBAL=${GLOBAL:-24}

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
LOCK=$RUNS/${ARM}_chain.lock                                     # one chain per arm: a second launch of the same arm exits
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  say "another $ARM chain (pid $(cat "$LOCK")) is running: not starting a second one"; exit 3
fi
echo $$ > "$LOCK"
N=$(echo "$CARDS" | awk -F, '{print NF}')
if [ -z "${MB:-}" ]; then
  for m in 12 8 6; do [ $((GLOBAL % (m * N))) -eq 0 ] && { MB=$m; break; }; done
fi
[ -n "${MB:-}" ] && [ $((GLOBAL % (MB * N))) -eq 0 ] || { say "global batch $GLOBAL does not split over $N cards with 12, 8 or 6 crops per card"; exit 1; }
ACCUM=$((GLOBAL / (MB * N)))
SCALE=$($PY -c "import math; print(math.sqrt($GLOBAL / 16))")
lr() { $PY -c "print(f'{$1 * $SCALE:.3e}')"; }
steps() { $PY -c "print(max(1, round($1 * 16 / $GLOBAL)))"; }
II_LR=(--lr-backbone "$(lr 5e-4)" --lr-heads "$(lr 5e-4)" --warmup-steps "$(steps 1000)" --backbone-warm-steps "$(steps 1000)")
III_LR=(--lr-backbone "$(lr 5e-5)" --lr-heads "$(lr 2.5e-4)" --lr-relation "$(lr 5e-4)" --warmup-steps "$(steps 200)")
case "$ARM" in
  c0) INIT=(--init random) ;;
  c2) : "${INIT_BACKBONE:?}" "${G1_REPORT:?}"; INIT=(--init-backbone "$INIT_BACKBONE" --g1-report "$G1_REPORT") ;;
  *) say "ARM must be c0 or c2"; exit 1 ;;
esac
launch() {   # launch <out> <log> <stage args...>: one torchrun on CARDS (plain python on one card)
  local out=$1 log=$2; shift 2
  if [ "$N" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$CARDS $TORCHRUN --standalone --nproc_per_node=$N scripts/aur_train.py "$@" --out "$out" > "$log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$CARDS $PY scripts/aur_train.py "$@" --out "$out" > "$log" 2>&1
  fi
}
WORKERS=${WORKERS:-24}                                           # loader processes per rank (see the header)
COMMON=(--samples "$SAMPLES" --val-patients "$VAL" --microbatch $MB --grad-accum $ACCUM --workers $WORKERS)
say "$ARM chain started on cards $CARDS (world $N, microbatch $MB, accumulation $ACCUM, global $GLOBAL, $WORKERS loaders per rank; Stage II ${II_LR[*]}; Stage III ${III_LR[*]})"

if [ -n "${RESUME_STAGE2:-}" ]; then
  : "${STAGE2_DIR:?STAGE2_DIR=the run directory of RESUME_STAGE2}"
  STAGE2=$STAGE2_DIR
  say "Stage II ($ARM) resumed from $RESUME_STAGE2 in $STAGE2"
  launch "$STAGE2" "$STAGE2.resume_$STAMP.log" --stage II "${INIT[@]}" "${COMMON[@]}" "${II_LR[@]}" --val-every 500 --save-every 500 --log-every 20 --resume "$RESUME_STAGE2"
  [ -f "$STAGE2/aur_stage2_best.pt" ] || { say "Stage II failed: see $STAGE2.resume_$STAMP.log"; exit 1; }
  say "Stage II finished: $(tr -d '\n' < "$STAGE2/summary.json" | cut -c1-300)"
else
# ---- Stage II smoke: stop at 50, resume to 100 ----
SMOKE=$RUNS/${ARM}_stage2_smoke_$STAMP
SMOKE_ARGS=(--stage II "${INIT[@]}" "${COMMON[@]}" "${II_LR[@]}" --max-steps 100 --val-every 50 --save-every 50 --log-every 10 --val-volumes 8)
launch "$SMOKE" "$SMOKE.log" "${SMOKE_ARGS[@]}" --stop-after 50 || { say "Stage II smoke (first half) failed: see $SMOKE.log"; exit 1; }
[ -f "$SMOKE/resume_step50.pt" ] || { say "Stage II smoke wrote no resume_step50.pt: see $SMOKE.log"; exit 1; }
launch "$SMOKE" "${SMOKE}_resume.log" "${SMOKE_ARGS[@]}" --resume "$SMOKE/resume_step50.pt" || { say "Stage II smoke resume failed: see ${SMOKE}_resume.log"; exit 1; }
STEPS=$($PY -c "import json; print(json.load(open('$SMOKE/summary.json'))['steps'])")
[ "$STEPS" = "100" ] || { say "Stage II smoke ended at step $STEPS, not 100"; exit 1; }
say "Stage II smoke passed (stopped at 50, resumed to 100): $SMOKE"

# ---- Stage II ----
STAGE2=$RUNS/${ARM}_stage2_240k_$STAMP
say "Stage II ($ARM, 240k crops) -> $STAGE2"
launch "$STAGE2" "$STAGE2.log" --stage II "${INIT[@]}" "${COMMON[@]}" "${II_LR[@]}" --val-every 500 --save-every 500 --log-every 20
[ -f "$STAGE2/aur_stage2_best.pt" ] || { say "Stage II failed: see $STAGE2.log"; exit 1; }
say "Stage II finished: $(tr -d '\n' < "$STAGE2/summary.json" | cut -c1-300)"
fi

# ---- Stage III ----
STAGE3=$RUNS/${ARM}_stage3_80k_$STAMP
say "Stage III ($ARM, 80k crops) -> $STAGE3"
launch "$STAGE3" "$STAGE3.log" --stage III --resume-stage2 "$STAGE2/aur_stage2_best.pt" "${COMMON[@]}" "${III_LR[@]}" --val-every 500 --save-every 500 --log-every 20
[ -f "$STAGE3/aur_stage3_best.pt" ] || { say "Stage III failed: see $STAGE3.log"; exit 1; }
say "Stage III finished: $(tr -d '\n' < "$STAGE3/summary.json" | cut -c1-300)"

# ---- evaluation, sharded over all cards: the validation split chooses the U thresholds, then the whole test split ----
evaluate() {   # evaluate <split> <out> [extra args]: one shard per card, then one merge
  local split=$1 out=$2; shift 2
  local k=0 pids=() shards=()
  for card in ${CARDS//,/ }; do
    CUDA_VISIBLE_DEVICES=$card $PY scripts/aur_eval.py --checkpoint "$STAGE3/aur_stage3_best.pt" --samples "$SAMPLES" --split "$split" --val-patients "$VAL" \
      --out "$out.shard$k" --gpu 0 --batch-size 2 --shard "$k/$N" > "$out.shard$k.log" 2>&1 &
    pids+=($!); shards+=("$out.shard$k"); k=$((k + 1))
  done
  for p in "${pids[@]}"; do wait "$p" || { say "a shard of the $split evaluation failed: see $out.shard*.log"; return 1; }; done
  $PY scripts/aur_eval_merge.py --shards "${shards[@]}" --samples "$SAMPLES" --split "$split" --val-patients "$VAL" "$@" --out "$out" > "$out.log" 2>&1 \
    || { say "the merge of the $split evaluation failed: see $out.log"; return 1; }
}
EVAL_VAL=$REC/eval/${ARM}_val_$STAMP
say "evaluation on the validation split, $N shards on cards $CARDS -> $EVAL_VAL"
evaluate val "$EVAL_VAL" || exit 1
THR=$($PY -c "import json; a=json.load(open('$EVAL_VAL/aggregate.json')); print(' '.join(f'--u-threshold {s}={e[\"thr\"]}' for s,e in a['u']['per_source'].items() if e.get('thr') is not None))")
say "U thresholds from the validation split: $THR"
EVAL_TEST=$REC/eval/${ARM}_test_$STAMP
say "evaluation on the whole test split, $N shards on cards $CARDS -> $EVAL_TEST"
evaluate test "$EVAL_TEST" $THR || exit 1
say "evaluation finished: $EVAL_TEST/REPORT.md"
say "$ARM chain done"
