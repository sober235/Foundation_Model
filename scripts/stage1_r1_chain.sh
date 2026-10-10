#!/bin/bash
# Stage I, repair round 1 (gate G1 failed on 2026-10-09; the plan allows two repair rounds, then the user): the same
# run as the first one (320k crops, global batch 36, lr 4.5e-4, warmup 444, mask 0.6, blocks 16-32 mm, tau 0.2,
# lambda_c 0.1; decision Q21) with the one change of the repair: the reconstruction decodes from all four levels.
# Early probes (pre-registered in g1/G1.md before the run): the host readout of the resume checkpoints at steps 1000
# and 2000 (seed 0, the G1 crops, both arms); the run continues only if at step 2000 the Stage I readout is at least
# the random one and higher than at step 1000; otherwise this chain stops its own run and says so. After the run:
# G1 (the gate as pre-registered); if it passes, C2 (scripts/stage23_chain.sh with ARM=c2) on C2_CARDS.
#   CARDS=5,6,7 [PROBE_CARD=4] [C2_CARDS=5,6] bash scripts/stage1_r1_chain.sh       (inside tmux)
# Global 36 needs 1 (microbatch 12, accumulation 3), 2 (6 x 3), 3 (12 x 1) or 6 (6 x 1) cards.
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
: "${CARDS:?CARDS=comma-separated card indices}"
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
LOG=$RUNS/stage1_r1_chain_$STAMP.log
RUN=$RUNS/stage1_r1_320k_$STAMP
N=$(echo "$CARDS" | awk -F, '{print NF}')
PROBE_CARD=${PROBE_CARD:-${CARDS%%,*}}
C2_CARDS=${C2_CARDS:-$(echo "$CARDS" | cut -d, -f1-2)}
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG" >&2; }      # stderr: probe() returns its numbers on stdout
case "$N" in
  1) MB=12; ACCUM=3 ;; 2) MB=6; ACCUM=3 ;; 3) MB=12; ACCUM=1 ;; 6) MB=6; ACCUM=1 ;;
  *) say "global batch 36 needs 1, 2, 3 or 6 cards, got $N"; exit 1 ;;
esac
ARGS=(--samples "$SSL_SAMPLES" --out "$RUN" --seen-crops 320000 --microbatch $MB --grad-accum $ACCUM --workers 8 --lr 4.50e-04 --warmup-steps 444
      --val-every 500 --save-every 500 --log-every 20 --val-volumes 32 --mask-ratio 0.60 --contrast-weight 0.10)
say "Stage I r1 on cards $CARDS (world $N, microbatch $MB, accumulation $ACCUM, global 36) -> $RUN; probes on card $PROBE_CARD"
set -m                                                            # the run gets its own process group (stop_run reaches every worker)
if [ "$N" -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=$CARDS $TORCHRUN --standalone --nproc_per_node=$N scripts/aur_ssl_train.py "${ARGS[@]}" > "$RUN.log" 2>&1 &
else
  CUDA_VISIBLE_DEVICES=$CARDS $PY scripts/aur_ssl_train.py "${ARGS[@]}" > "$RUN.log" 2>&1 &
fi
TRAIN_PID=$!
set +m
say "training pid $TRAIN_PID (its own process group)"

probe() {   # probe <step>: the host-only readout of resume_step<step>.pt; prints "ssl random"
  local s=$1 out=$REC/g1/r1_probe_step${1}_$STAMP
  until [ -f "$RUN/resume_step$s.pt" ]; do
    kill -0 $TRAIN_PID 2>/dev/null || { say "the run ended before step $s: see $RUN.log"; return 1; }
    sleep 60
  done
  sleep 30                                                        # let the save finish
  $PY - "$RUN/resume_step$s.pt" "$out.backbone.pt" <<'PYEOF' >> "$LOG" 2>&1 || return 1
import sys, torch
from anatobind.aur.ssl import checkpoint as CK
from anatobind.aur.ssl.model import StageOne
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = StageOne()
m.load_state_dict(ck["model"], strict=True)
CK.export_backbone(sys.argv[2], m.backbone, {**ck["meta"], "from_resume": sys.argv[1]})
PYEOF
  CUDA_VISIBLE_DEVICES=$PROBE_CARD $PY scripts/aur_ssl_eval.py --checkpoint "$out.backbone.pt" --samples "$SAMPLES" --val-patients "$VAL" --out "$out" --seeds 0 --host-only > "$out.log" 2>&1 \
    || { say "probe at step $s failed: see $out.log"; return 1; }
  $PY -c "import json; g=json.load(open('$out/g1_report.json'))['g1']; print(g['host_macro_ssl'], g['host_macro_random'])"
}
stop_run() { say "stopping the run (process group $TRAIN_PID): $*"; kill -TERM -- -$TRAIN_PID 2>/dev/null; wait $TRAIN_PID 2>/dev/null; }

P1=$(probe 1000) || { stop_run "probe 1000 failed"; exit 1; }
say "probe step 1000: Stage I / random host macro Dice = $P1"
P2=$(probe 2000) || { stop_run "probe 2000 failed"; exit 1; }
say "probe step 2000: Stage I / random host macro Dice = $P2"
GO=$($PY -c "a=[float(x) for x in '$P1'.split()]; b=[float(x) for x in '$P2'.split()]; print('go' if b[0] >= b[1] and b[0] > a[0] else 'stop')")
if [ "$GO" != "go" ]; then
  stop_run "early rule not met (step 2000 Stage I must be >= random and above step 1000)"
  say "round 1 stopped early; the plan's round 2 or the user decides"; exit 2
fi
say "early rule met: the run continues"
wait $TRAIN_PID
[ -f "$RUN/ssl_stage1_best.pt" ] || { say "the run ended without ssl_stage1_best.pt: see $RUN.log"; exit 1; }
say "Stage I r1 finished: $(tr -d '\n' < "$RUN/summary.json" | cut -c1-300)"
for i in $(seq 1 60); do pgrep -f "aur_ssl_train.py.*$RUN" >/dev/null || break; sleep 5; done

G1=$REC/g1/r1_run_$STAMP
say "G1 (round 1) on card $PROBE_CARD -> $G1"
CUDA_VISIBLE_DEVICES=$PROBE_CARD $PY scripts/aur_ssl_eval.py --checkpoint "$RUN/ssl_stage1_best.pt" --samples "$SAMPLES" --val-patients "$VAL" --out "$G1" > "$G1.log" 2>&1
[ -f "$G1/g1_report.json" ] || { say "G1 script failed (no report): see $G1.log"; exit 1; }
VERDICT=$($PY -c "import json; r=json.load(open('$G1/g1_report.json'))['g1']; print('pass' if r.get('pass') else 'fail', r.get('host_pass'), r.get('lesion_pass'), round(r['macro_dice_gain'], 4))")
say "G1 (round 1) verdict: $VERDICT (report $G1/g1_report.json)"
case "$VERDICT" in pass*) ;; *) say "G1 did not pass in round 1: the chain stops here"; exit 2;; esac

say "C2 on cards $C2_CARDS"
ARM=c2 CARDS=$C2_CARDS INIT_BACKBONE="$RUN/ssl_stage1_best.pt" G1_REPORT="$G1/g1_report.json" bash scripts/stage23_chain.sh
say "stage1 r1 chain done"
