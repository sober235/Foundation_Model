#!/bin/bash
# Overnight watchdog for round 2 (2026-10-10): (1) if the resumed Stage I run is taken down again (its chain log says
# "ended without ssl_stage1_best.pt"), hold cards 0, 2, 3 at once and continue from the latest resume checkpoint, at most
# 3 times; (2) once the run has logged step 8800, hold cards 0 and 2 for C2 (hold_c2b.py) through the end of training,
# G1 on card 5 and C2's smoke; (3) stop when G1 has decided.
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
RUNS=/data2/congcong/data/FM_data/derived/aur/ssl_runs
RUN=$RUNS/stage1_r2_320k_20261010_1303
SCR=/tmp/claude-1002/-home-congcongliu/4c51bb86-6842-43b4-9b18-0dd733b5a3d4/scratchpad
PY=$HOME/anaconda3/envs/nvgen/bin/python
say() { echo "[$(date '+%F %T')] $*"; }
attempt=0; c2hold=0
LOGF=$(ls -t $RUNS/stage1_r2_resume_chain_*.log | head -1)
say "watching $LOGF"
while true; do
  if grep -qE "verdict|script failed|did not pass" "$LOGF"; then say "G1 has decided (or its script failed): watchdog done"; exit 0; fi
  if grep -q "ended without" "$LOGF"; then
    attempt=$((attempt + 1)); c2hold=0
    if [ $attempt -gt 3 ]; then say "taken down a fourth time: giving up"; exit 1; fi
    say "the run was taken down (attempt $attempt): holding cards 0, 2, 3"
    tmux new-window -d -t anatobind -n "hold023_$attempt" "CUDA_VISIBLE_DEVICES=0,2,3 PYTHONNOUSERSITE=1 $PY $SCR/hold_multi.py 4 50000 3600 0 2 3 2>&1 | tee $SCR/hold023_$attempt.log; exec bash"
    sleep 30
    LAST=$(ls $RUN/resume_step*.pt | sed 's/.*resume_step\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
    say "continuing from resume_step$LAST.pt"
    tmux new-window -d -t anatobind -n "r2res_$attempt" "cd /data0/congcong/code/Project_Doing/foundation_model && CARDS=0,2,3 PROBE_CARD=5 C2_CARDS=0,2 RUN=$RUN RESUME=$RUN/resume_step$LAST.pt bash scripts/stage1_r2_resume_chain.sh; echo '--- r2 resume chain exited ---'; exec bash"
    tmux set-option -w -t "anatobind:r2res_$attempt" remain-on-exit on
    sleep 90
    LOGF=$(ls -t $RUNS/stage1_r2_resume_chain_*.log | head -1)
    say "watching $LOGF"
    continue
  fi
  STEP=$(tail -n 1 $RUN/log_rank0.jsonl | $PY -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)
  if [ $c2hold -eq 0 ] && [ "$STEP" -ge 8800 ]; then
    say "step $STEP: holding cards 0 and 2 for C2"
    tmux new-window -d -t anatobind -n holdc2b "CUDA_VISIBLE_DEVICES=0,2 PYTHONNOUSERSITE=1 $PY $SCR/hold_c2b.py 2>&1 | tee -a $SCR/hold_c2b.log; exec bash"
    c2hold=1
  fi
  sleep 20
done
