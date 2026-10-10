#!/bin/bash
# One-shot (2026-10-10): once C0's Stage II has written resume_step3500.pt (validation and best export come first in
# the trainer), restart its chain on the same card from that checkpoint with the new scripts/stage23_chain.sh
# (32 loader processes for this one-card arm), holding 4 GiB of card 5 across the gap so that no other job takes the card.
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
PY=~/anaconda3/envs/nvgen/bin/python
SCR=/tmp/claude-1002/-home-congcongliu/4c51bb86-6842-43b4-9b18-0dd733b5a3d4/scratchpad
RUN=/data2/congcong/data/FM_data/derived/aur/ssl_runs/c0_stage2_240k_20261010_0903
CK=$RUN/resume_step3500.pt
TRAINER=1447388; CHAIN=1447361
say() { echo "[$(date '+%F %T')] $*"; }
say "waiting for $CK to be written and closed by the trainer $TRAINER"
until [ -f "$CK" ] && [ "$(stat -c %s "$CK")" -gt 250000000 ] && ! ls -l /proc/$TRAINER/fd 2>/dev/null | grep -q resume_step3500; do
  kill -0 $TRAINER 2>/dev/null || { say "the trainer $TRAINER is gone before step 3500: nothing done"; exit 1; }
  sleep 5
done
sleep 3
say "checkpoint written ($(stat -c %s "$CK") bytes); holding card 5"
tmux new-window -d -t anatobind -n hold5 "CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 $PY $SCR/hold_card.py 4 $TRAINER 5 50000 1800 2>&1 | tee $SCR/hold5.log; exec bash"
for i in $(seq 60); do grep -q holding $SCR/hold5.log 2>/dev/null && break; sleep 2; done
grep -q holding $SCR/hold5.log 2>/dev/null || { say "the hold did not start: C0 left running untouched"; exit 1; }
say "hold on; stopping the C0 chain (Ctrl-C in c0w)"
tmux send-keys -t anatobind:c0w C-c
for i in $(seq 90); do
  if ! kill -0 $TRAINER 2>/dev/null && ! kill -0 $CHAIN 2>/dev/null; then break; fi
  sleep 2
done
if kill -0 $TRAINER 2>/dev/null || kill -0 $CHAIN 2>/dev/null; then say "C0 did not stop within 3 min: nothing replaced, see c0w"; exit 1; fi
say "C0 stopped"
if ps -eo args | awk '$1=="bash" && $2=="scripts/stage23_chain.sh"' | grep -q .; then
  say "another stage23_chain.sh is running: the script is not replaced (C0 restarts with the old one)"
else
  cat $SCR/stage23_chain_workers.sh > scripts/stage23_chain.sh && say "scripts/stage23_chain.sh replaced (WORKERS, default 24 per rank)"
fi
tmux respawn-window -k -t anatobind:c0w "cd /data0/congcong/code/Project_Doing/foundation_model && ARM=c0 CARDS=5 WORKERS=32 RESUME_STAGE2=$CK STAGE2_DIR=$RUN bash scripts/stage23_chain.sh; echo '--- c0 chain exited ---'; exec bash"
say "C0 chain respawned in c0w from $CK"
