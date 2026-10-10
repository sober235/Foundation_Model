#!/bin/bash
# Round 2 of Stage I (lambda_c 0) continued from a resume checkpoint (2026-10-10: the run of 13:03 was taken down at
# step ~3870 by the NCCL watchdog, one rank stalled for 17 min; its probes at steps 1000/2000 and the registered early
# rule ("go") are done and recorded in g1/G1.md). Same configuration and run directory; then G1 on PROBE_CARD and, if it
# passes, C2 (scripts/stage23_chain.sh ARM=c2) on C2_CARDS, exactly as scripts/stage1_r2_chain.sh does after its run.
#   CARDS=0,2,3 PROBE_CARD=5 C2_CARDS=0,2 RUN=<run dir> RESUME=<run dir>/resume_stepN.pt bash scripts/stage1_r2_resume_chain.sh
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
: "${CARDS:?}" "${PROBE_CARD:?}" "${C2_CARDS:?}" "${RUN:?}" "${RESUME:?}"
export PYTHONNOUSERSITE=1 PYTHONPATH=. NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
PY=~/anaconda3/envs/nvgen/bin/python
TORCHRUN=~/anaconda3/envs/nvgen/bin/torchrun
DATA=/data2/congcong/data/FM_data/derived/aur
RUNS=$DATA/ssl_runs
SAMPLES=$DATA/samples_1mm_v1.json
SSL_SAMPLES=$DATA/ssl_manifest_v1/samples_ssl.json
VAL=$DATA/ssl_manifest_v1/val_patients.json
REC=docs/verification/2026-10-09/anatobind_brain_ssl_first
STAMP=$(date +%Y%m%d_%H%M)
LOG=$RUNS/stage1_r2_resume_chain_$STAMP.log
N=$(echo "$CARDS" | awk -F, '{print NF}')
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
[ "$N" = "3" ] || { say "this continuation keeps the 3-card configuration (microbatch 12, global 36); got $N cards"; exit 1; }
case ",$CARDS," in *",$PROBE_CARD,"*) say "PROBE_CARD $PROBE_CARD is one of the training cards $CARDS"; exit 1 ;; esac
[ -f "$RESUME" ] && [ -d "$RUN" ] || { say "missing $RESUME or $RUN"; exit 1; }
ARGS=(--samples "$SSL_SAMPLES" --out "$RUN" --seen-crops 320000 --microbatch 12 --grad-accum 1 --workers 8 --lr 4.50e-04 --warmup-steps 444
      --val-every 500 --save-every 500 --log-every 20 --val-volumes 32 --mask-ratio 0.60 --contrast-weight 0.0 --resume "$RESUME")
say "Stage I r2 (lambda_c 0) continued from $RESUME on cards $CARDS (world 3, microbatch 12, global 36) in $RUN; G1 on card $PROBE_CARD; C2 on $C2_CARDS"
CUDA_VISIBLE_DEVICES=$CARDS $TORCHRUN --standalone --nproc_per_node=3 scripts/aur_ssl_train.py "${ARGS[@]}" > "$RUN.resume_$STAMP.log" 2>&1
[ -f "$RUN/ssl_stage1_best.pt" ] || { say "the run ended without ssl_stage1_best.pt: see $RUN.resume_$STAMP.log"; exit 1; }
say "Stage I r2 finished: $(tr -d '\n' < "$RUN/summary.json" | cut -c1-300)"

G1=$REC/g1/r2_run_$STAMP
say "G1 (round 2) on card $PROBE_CARD -> $G1"
CUDA_VISIBLE_DEVICES=$PROBE_CARD $PY scripts/aur_ssl_eval.py --checkpoint "$RUN/ssl_stage1_best.pt" --samples "$SAMPLES" --val-patients "$VAL" --out "$G1" > "$G1.log" 2>&1
[ -f "$G1/g1_report.json" ] || { say "G1 script failed (no report): see $G1.log"; exit 1; }
VERDICT=$($PY -c "import json; r=json.load(open('$G1/g1_report.json'))['g1']; print('pass' if r.get('pass') else 'fail', r.get('host_pass'), r.get('lesion_pass'), round(r['macro_dice_gain'], 4))")
say "G1 (round 2) verdict: $VERDICT (report $G1/g1_report.json)"
case "$VERDICT" in pass*) ;; *) say "G1 did not pass in round 2: both repair rounds have failed; the user decides (plan)"; exit 2;; esac
say "C2 on cards $C2_CARDS"
ARM=c2 CARDS=$C2_CARDS INIT_BACKBONE="$RUN/ssl_stage1_best.pt" G1_REPORT="$G1/g1_report.json" bash scripts/stage23_chain.sh
say "stage1 r2 resume chain done"
