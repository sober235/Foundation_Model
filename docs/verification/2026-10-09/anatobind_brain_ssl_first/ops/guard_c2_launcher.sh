#!/bin/bash
# Start the card guard for C2 (cards 0 and 2) as soon as C2's chain has written its lock (2026-10-10).
L=/data2/congcong/data/FM_data/derived/aur/ssl_runs/c2_chain.lock
SCR=/tmp/claude-1002/-home-congcongliu/4c51bb86-6842-43b4-9b18-0dd733b5a3d4/scratchpad
echo "[$(date '+%F %T')] waiting for $L"
until [ -f "$L" ] && kill -0 "$(cat "$L")" 2>/dev/null; do sleep 20; done
echo "[$(date '+%F %T')] C2 chain pid $(cat "$L"): starting the guard on cards 0, 2"
CUDA_VISIBLE_DEVICES=0,2 PYTHONNOUSERSITE=1 $HOME/anaconda3/envs/nvgen/bin/python $SCR/card_guard.py "$L" 0 2
