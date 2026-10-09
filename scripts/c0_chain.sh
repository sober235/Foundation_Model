#!/bin/bash
# The C0 chain (plan 2026-10-09 §1: the random-initialisation control; run while the Stage I line is blocked at G1):
# Stage II from a RANDOM backbone (a 100-step smoke with save / resume first), Stage III, the evaluation on the
# validation split (U thresholds) and on the whole test split (decision Q22). Every failure stops the chain and says
# so; nothing is deleted or overwritten. Meant to run inside tmux:
#   tmux new-window -t anatobind -n c0 "bash scripts/c0_chain.sh; echo '--- c0 chain exited ---'; exec bash"
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
export PYTHONNOUSERSITE=1 PYTHONPATH=. NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
PY=~/anaconda3/envs/nvgen/bin/python
TORCHRUN=~/anaconda3/envs/nvgen/bin/torchrun
DATA=/data2/congcong/data/FM_data/derived/aur
RUNS=$DATA/ssl_runs
SAMPLES=$DATA/samples_1mm_v1.json
VAL=$DATA/ssl_manifest_v1/val_patients.json
REC=docs/verification/2026-10-09/anatobind_brain_ssl_first
STAMP=$(date +%Y%m%d_%H%M)
LOG=$RUNS/c0_chain_$STAMP.log
CARDS_WANTED=${CARDS_WANTED:-3}

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
idle_cards() {                       # idle cards, 5,6,7 first, at most $1
  local lim=$1
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1500 {print $1}' \
    | awk '{o=($1==5||$1==6||$1==7)?0:1; print o, $1}' | sort -n | awk '{print $2}' | head -n "$lim" | paste -sd, -
}
count() { echo "$1" | awk -F, '{print NF}'; }
scaled_lr() { $PY -c "import math; print(f'{$1*math.sqrt($2/16):.2e}')"; }

say "C0 chain started (random backbone; Stage I is NOT used)"
MB=4

# ---- Stage II smoke: 100 steps with a save at 50 and a resume to 100 (the T10 acceptance record on real data) ----
CARDS=$(idle_cards $CARDS_WANTED); N=$(count "$CARDS"); [ "$N" -ge 1 ] || { say "no idle card"; exit 1; }
GB=$((MB * N)); LR2=$(scaled_lr 5e-4 $GB)
SMOKE=$RUNS/c0_stage2_smoke_$STAMP
say "Stage II smoke (random init) on cards $CARDS (world $N, microbatch $MB, global $GB, lr $LR2) -> $SMOKE"
CUDA_VISIBLE_DEVICES=$CARDS timeout 1800 $TORCHRUN --standalone --nproc_per_node=$N scripts/aur_train.py --stage II --init random --pilot \
  --samples "$SAMPLES" --val-patients "$VAL" --out "$SMOKE" --microbatch $MB --workers 8 --lr-backbone "$LR2" --lr-heads "$LR2" \
  --max-steps 100 --val-every 50 --save-every 50 --log-every 10 --val-volumes 8 > "$SMOKE.log" 2>&1
[ $? -eq 0 ] || { say "Stage II smoke failed: see $SMOKE.log"; exit 1; }
CUDA_VISIBLE_DEVICES=$CARDS timeout 1800 $TORCHRUN --standalone --nproc_per_node=$N scripts/aur_train.py --stage II --init random --pilot \
  --samples "$SAMPLES" --val-patients "$VAL" --out "$SMOKE" --resume "$SMOKE/resume_step50.pt" --microbatch $MB --workers 8 \
  --lr-backbone "$LR2" --lr-heads "$LR2" --max-steps 100 --val-every 50 --save-every 50 --log-every 10 --val-volumes 8 > "${SMOKE}_resume.log" 2>&1
[ $? -eq 0 ] || { say "Stage II smoke resume failed: see ${SMOKE}_resume.log"; exit 1; }
say "Stage II smoke passed (100 steps, resume from 50): $SMOKE"

# ---- Stage II (C0) ----
CARDS=$(idle_cards $CARDS_WANTED); N=$(count "$CARDS"); [ "$N" -ge 1 ] || { say "no idle card for Stage II"; exit 1; }
GB=$((MB * N)); LR2=$(scaled_lr 5e-4 $GB)
STAGE2=$RUNS/c0_stage2_240k_$STAMP
say "Stage II (C0, random init) on cards $CARDS (world $N, microbatch $MB, global $GB, lr $LR2, 240k crops) -> $STAGE2"
CUDA_VISIBLE_DEVICES=$CARDS $TORCHRUN --standalone --nproc_per_node=$N scripts/aur_train.py --stage II --init random --pilot \
  --samples "$SAMPLES" --val-patients "$VAL" --out "$STAGE2" --microbatch $MB --workers 8 --lr-backbone "$LR2" --lr-heads "$LR2" \
  --val-every 500 --save-every 500 --log-every 20 > "$STAGE2.log" 2>&1
[ $? -eq 0 ] && [ -f "$STAGE2/aur_stage2_best.pt" ] || { say "Stage II failed: see $STAGE2.log"; exit 1; }
say "Stage II finished: $(cat "$STAGE2/summary.json" | tr -d '\n' | cut -c1-300)"

# ---- Stage III (C0) ----
CARDS=$(idle_cards $CARDS_WANTED); N=$(count "$CARDS"); [ "$N" -ge 1 ] || { say "no idle card for Stage III"; exit 1; }
GB=$((MB * N))
STAGE3=$RUNS/c0_stage3_80k_$STAMP
say "Stage III (C0) on cards $CARDS (world $N, microbatch $MB, global $GB, 80k crops) -> $STAGE3"
CUDA_VISIBLE_DEVICES=$CARDS $TORCHRUN --standalone --nproc_per_node=$N scripts/aur_train.py --stage III --resume-stage2 "$STAGE2/aur_stage2_best.pt" \
  --samples "$SAMPLES" --val-patients "$VAL" --out "$STAGE3" --microbatch $MB --workers 8 \
  --lr-backbone "$(scaled_lr 5e-5 $GB)" --lr-heads "$(scaled_lr 2.5e-4 $GB)" --lr-relation "$(scaled_lr 5e-4 $GB)" \
  --val-every 500 --save-every 500 --log-every 20 > "$STAGE3.log" 2>&1
[ $? -eq 0 ] && [ -f "$STAGE3/aur_stage3_best.pt" ] || { say "Stage III failed: see $STAGE3.log"; exit 1; }
say "Stage III finished: $(cat "$STAGE3/summary.json" | tr -d '\n' | cut -c1-300)"

# ---- evaluation: the validation split chooses the U thresholds, then the whole test split ----
CARD=$(idle_cards 1); [ -n "$CARD" ] || { say "no idle card for the evaluation"; exit 1; }
EVAL_VAL=$REC/eval/c0_val_$STAMP
say "evaluation on the validation split (card $CARD) -> $EVAL_VAL"
CUDA_VISIBLE_DEVICES=$CARD $PY scripts/aur_eval.py --checkpoint "$STAGE3/aur_stage3_best.pt" --samples "$SAMPLES" --split val --val-patients "$VAL" --out "$EVAL_VAL" --gpu 0 --batch-size 2 > "$EVAL_VAL.log" 2>&1
[ $? -eq 0 ] || { say "validation-split evaluation failed: see $EVAL_VAL.log"; exit 1; }
THR=$($PY -c "import json; a=json.load(open('$EVAL_VAL/aggregate.json')); print(' '.join(f'--u-threshold {s}={e[\"thr\"]}' for s,e in a['u']['per_source'].items() if e.get('thr') is not None))")
say "U thresholds from the validation split: $THR"
EVAL_TEST=$REC/eval/c0_test_$STAMP
say "evaluation on the whole test split (card $CARD) -> $EVAL_TEST"
CUDA_VISIBLE_DEVICES=$CARD $PY scripts/aur_eval.py --checkpoint "$STAGE3/aur_stage3_best.pt" --samples "$SAMPLES" --split test --out "$EVAL_TEST" --gpu 0 --batch-size 2 $THR > "$EVAL_TEST.log" 2>&1
[ $? -eq 0 ] || { say "test-split evaluation failed: see $EVAL_TEST.log"; exit 1; }
say "evaluation finished: $EVAL_TEST/REPORT.md"
say "C0 chain done"
