#!/bin/bash
# One-shot (2026-10-11, decision Q25): hold cards 0 and 2, stop the spec-recipe C2 chain (tmux window r2res), and start
# C2 again from the G1 export with the Stage II backbone at a tenth (II_BB=5e-5) on the same cards; then start the card
# guard for the new chain.
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
PY=$HOME/anaconda3/envs/nvgen/bin/python
SCR=/tmp/claude-1002/-home-congcongliu/4c51bb86-6842-43b4-9b18-0dd733b5a3d4/scratchpad
RUNS=/data2/congcong/data/FM_data/derived/aur/ssl_runs
OLD="2830799,2830846,2830847,2793554,2208322"
INIT=$RUNS/stage1_r2_320k_20261010_1303/ssl_stage1_best.pt
G1R=docs/verification/2026-10-09/anatobind_brain_ssl_first/g1/r2_run_20261010_1942/g1_report.json
say() { echo "[$(date '+%F %T')] $*"; }
[ -f "$INIT" ] && [ -f "$G1R" ] || { say "missing $INIT or $G1R"; exit 1; }
tmux new-window -d -t anatobind -n hold02 "CUDA_VISIBLE_DEVICES=0,2 PYTHONNOUSERSITE=1 $PY $SCR/hold_swap.py 4 50000 3600 $OLD 0 2 2>&1 | tee $SCR/hold02.log; exec bash"
for i in $(seq 60); do grep -q holding $SCR/hold02.log 2>/dev/null && break; sleep 2; done
grep -q holding $SCR/hold02.log 2>/dev/null || { say "the hold did not start: nothing stopped"; exit 1; }
say "hold on; stopping the spec-recipe C2 (Ctrl-C in r2res)"
tmux send-keys -t anatobind:r2res C-c
for i in $(seq 90); do
  alive=0; for p in ${OLD//,/ }; do kill -0 $p 2>/dev/null && alive=1; done
  [ $alive -eq 0 ] && break; sleep 2
done
[ $alive -eq 0 ] || { say "the old C2 did not stop within 3 min: not starting the new one; see r2res"; exit 1; }
say "old C2 stopped"
tmux new-window -d -t anatobind -n c2v2 "cd /data0/congcong/code/Project_Doing/foundation_model && ARM=c2 CARDS=0,2 II_BB=5e-5 INIT_BACKBONE=$INIT G1_REPORT=$G1R bash scripts/stage23_chain.sh; echo '--- c2 v2 chain exited ---'; exec bash"
tmux set-option -w -t anatobind:c2v2 remain-on-exit on
say "new C2 chain started in window c2v2"
for i in $(seq 30); do L=$RUNS/c2_chain.lock; [ -f $L ] && kill -0 "$(cat $L)" 2>/dev/null && [ "$(cat $L)" != "2793554" ] && break; sleep 2; done
say "new C2 chain pid $(cat $RUNS/c2_chain.lock); starting its card guard (window guardc2v2)"
tmux new-window -d -t anatobind -n guardc2v2 "CUDA_VISIBLE_DEVICES=0,2 PYTHONNOUSERSITE=1 $PY $SCR/card_guard.py $RUNS/c2_chain.lock 0 2 2>&1 | tee -a $SCR/guard_c2v2.log; exec bash"
say "done"
