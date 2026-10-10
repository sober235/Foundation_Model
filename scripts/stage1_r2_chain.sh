#!/bin/bash
# Stage I, repair round 2 (pre-registered in g1/G1.md: chosen from round 1's attribution; at step 1000 the reconstruction
# rested on levels 2-4 and the blocks of stages 3-4 were dominated by the contrastive gradient, MIM/contrast 0.34 and
# 0.11): round 1's run (320k crops, global batch 36, lr 4.5e-4, warmup 444, mask 0.6, blocks 16-32 mm, the four-level
# decoder) with the one change of round 2: no contrastive term (lambda_c = 0, pure masked reconstruction). Without
# the InfoNCE the micro-batch span no longer matters: 1, 2, 3 or 6 cards (global 36 by accumulation).
# Early probes (pre-registered in g1/G1.md, amended after the review of 2026-10-10, before any run): the host readout of
# the resume checkpoints at steps 1000 and 2000 (seeds 0 1 2, the G1 crops, both arms); the run is stopped only if at
# step 2000 the Stage I readout is below random - 0.01 AND it rose by less than 0.01 since step 1000. A probe that
# fails (out of memory, a crash) is retried once and otherwise skipped: it never stops the run. At each probe the
# attribution diagnostic (scripts/aur_ssl_attrib.py, on the CPU) is recorded. After the run: G1 (the gate as
# pre-registered); if it passes, C2 (scripts/stage23_chain.sh with ARM=c2) on C2_CARDS.
#   CARDS=4,5,7 PROBE_CARD=2 C2_CARDS=4,5 bash scripts/stage1_r2_chain.sh       (inside tmux)
# PROBE_CARD: a card outside CARDS. C2_CARDS: the C0 card count (2) for paired settings (decision Q23: 12 crops per card).
# The training runs in its own process group: Ctrl-C on this chain leaves it running; stop it with
# kill -TERM <pid> (the pid is in the chain log; torchrun's own handler then stops its workers, which run in sessions
# of their own). There is no restart path in the chain: resume a run by hand with scripts/aur_ssl_train.py --resume.
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
: "${CARDS:?CARDS=3 or 6 comma-separated card indices}" "${PROBE_CARD:?PROBE_CARD=a card outside CARDS}"
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
LOG=$RUNS/stage1_r2_chain_$STAMP.log
RUN=$RUNS/stage1_r2_320k_$STAMP
N=$(echo "$CARDS" | awk -F, '{print NF}')
C2_CARDS=${C2_CARDS:-$(echo "$CARDS" | cut -d, -f1-2)}
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG" >&2; }      # stderr: probe() returns its numbers on stdout
case ",$CARDS," in *",$PROBE_CARD,"*) say "PROBE_CARD $PROBE_CARD is one of the training cards $CARDS"; exit 1 ;; esac
case "$N" in
  1) MB=12; ACCUM=3 ;; 2) MB=6; ACCUM=3 ;; 3) MB=12; ACCUM=1 ;; 6) MB=6; ACCUM=1 ;;
  *) say "global batch 36 needs 1, 2, 3 or 6 cards, got $N"; exit 1 ;;
esac
ARGS=(--samples "$SSL_SAMPLES" --out "$RUN" --seen-crops 320000 --microbatch $MB --grad-accum $ACCUM --workers 8 --lr 4.50e-04 --warmup-steps 444
      --val-every 500 --save-every 500 --log-every 20 --val-volumes 32 --mask-ratio 0.60 --contrast-weight 0.0)
say "Stage I r2 (lambda_c 0) on cards $CARDS (world $N, microbatch $MB, accumulation $ACCUM, global 36) -> $RUN; probes on card $PROBE_CARD; C2 on $C2_CARDS"
set -m                                                            # the run gets its own process group: Ctrl-C here does not reach it
CUDA_VISIBLE_DEVICES=$CARDS $TORCHRUN --standalone --nproc_per_node=$N scripts/aur_ssl_train.py "${ARGS[@]}" > "$RUN.log" 2>&1 &
TRAIN_PID=$!
set +m
say "training pid $TRAIN_PID"

saved() {   # saved <step>: rank 0 has logged a step past <step>, so its synchronous save of resume_step<step>.pt is done
  [ -f "$RUN/resume_step$1.pt" ] && [ -f "$RUN/log_rank0.jsonl" ] && \
    [ "$(tail -n 1 "$RUN/log_rank0.jsonl" | $PY -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge $(( $1 + 20 )) ]
}
probe_once() {   # probe_once <step> <tag>: prints "ssl random" (3-seed means) on stdout
  local s=$1 out=$REC/g1/r2_probe_step${1}_$STAMP$2 bb=$RUN/probe_step${1}$2.backbone.pt
  $PY - "$RUN/resume_step$s.pt" "$bb" <<'PYEOF' >> "$LOG" 2>&1 || return 1
import sys, torch
from anatobind.aur.ssl import checkpoint as CK
from anatobind.aur.ssl.model import StageOne
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = StageOne()
m.load_state_dict(ck["model"], strict=True)
CK.export_backbone(sys.argv[2], m.backbone, {**ck["meta"], "from_resume": sys.argv[1]})
PYEOF
  CUDA_VISIBLE_DEVICES=$PROBE_CARD $PY scripts/aur_ssl_eval.py --checkpoint "$bb" --samples "$SAMPLES" --val-patients "$VAL" --out "$out" --seeds 0 1 2 --host-only > "$out.log" 2>&1 || return 1
  $PY -c "import json; g=json.load(open('$out/g1_report.json'))['g1']; print(g['host_macro_ssl'], g['host_macro_random'])"
}
probe() {   # probe <step>: waits for the save, records the attribution (CPU, background), probes (one retry); empty on failure
  local s=$1 r
  until saved $s; do
    kill -0 $TRAIN_PID 2>/dev/null || { say "the run ended before step $s: see $RUN.log"; return 1; }
    sleep 60
  done
  CUDA_VISIBLE_DEVICES="" nice -n 19 $PY scripts/aur_ssl_attrib.py --checkpoint "$RUN/resume_step$s.pt" --samples "$SSL_SAMPLES" \
    --out "$REC/g1/r2_attrib_step${s}_$STAMP.json" > "$REC/g1/r2_attrib_step${s}_$STAMP.log" 2>&1 &
  r=$(probe_once $s "") || r=$(probe_once $s "_retry") || { say "probe at step $s failed twice: see $REC/g1/r2_probe_step${s}_$STAMP*.log; the run continues"; echo ""; return 0; }
  echo "$r"
}

P1=$(probe 1000) || exit 1
say "probe step 1000: Stage I / random host macro Dice (3 seeds) = ${P1:-none}"
P2=$(probe 2000) || exit 1
say "probe step 2000: Stage I / random host macro Dice (3 seeds) = ${P2:-none}"
if [ -n "$P1" ] && [ -n "$P2" ]; then
  RULE=$($PY -c "a=[float(x) for x in '$P1'.split()]; b=[float(x) for x in '$P2'.split()]; print('stop' if (b[0] < b[1] - 0.01 and b[0] - a[0] < 0.01) else 'go')")
  if [ "$RULE" = "stop" ]; then
    say "early rule: stop (step 2000 Stage I below random - 0.01 and rose by less than 0.01 since step 1000); stopping pid $TRAIN_PID"
    kill -TERM $TRAIN_PID 2>/dev/null; wait $TRAIN_PID 2>/dev/null
    say "round 2 stopped early: both repair rounds have failed; the user decides (plan)"; exit 2
  fi
  say "early rule: go (the run continues)"
else
  say "early rule not evaluated (a probe was skipped); the run continues to G1"
fi
wait $TRAIN_PID
[ -f "$RUN/ssl_stage1_best.pt" ] || { say "the run ended without ssl_stage1_best.pt: see $RUN.log"; exit 1; }
say "Stage I r2 finished: $(tr -d '\n' < "$RUN/summary.json" | cut -c1-300)"
for i in $(seq 1 60); do pgrep -f "aur_ssl_train.py.*$RUN" >/dev/null || break; sleep 5; done

G1=$REC/g1/r2_run_$STAMP
say "G1 (round 2) on card $PROBE_CARD -> $G1"
CUDA_VISIBLE_DEVICES=$PROBE_CARD $PY scripts/aur_ssl_eval.py --checkpoint "$RUN/ssl_stage1_best.pt" --samples "$SAMPLES" --val-patients "$VAL" --out "$G1" > "$G1.log" 2>&1
[ -f "$G1/g1_report.json" ] || { say "G1 script failed (no report): see $G1.log"; exit 1; }
VERDICT=$($PY -c "import json; r=json.load(open('$G1/g1_report.json'))['g1']; print('pass' if r.get('pass') else 'fail', r.get('host_pass'), r.get('lesion_pass'), round(r['macro_dice_gain'], 4))")
say "G1 (round 2) verdict: $VERDICT (report $G1/g1_report.json)"
case "$VERDICT" in pass*) ;; *) say "G1 did not pass in round 2: both repair rounds have failed; the user decides (plan)"; exit 2;; esac

say "C2 on cards $C2_CARDS"
ARM=c2 CARDS=$C2_CARDS INIT_BACKBONE="$RUN/ssl_stage1_best.pt" G1_REPORT="$G1/g1_report.json" bash scripts/stage23_chain.sh
say "stage1 r2 chain done"
