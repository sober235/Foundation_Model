#!/bin/bash
# H1 rerun chain: train the given folds one after another on one GPU, caching each fold's held-out
# detections right after its training. A fold claimed by another chain (marker file) is skipped.
#   logs/train_chain_gate0.sh <gpu> <fold> [<fold> ...]
cd /data0/congcong/code/Project_Doing/foundation_model-aur || exit 1
G=$1; shift
PY=$HOME/anaconda3/envs/nvgen/bin/python
for f in "$@"; do
  d=runs/detector_gate0_fold$f
  if [ -e "$d/.claimed" ]; then echo "$(date '+%F %T') fold $f already claimed, skip"; continue; fi
  mkdir -p "$d"; touch "$d/.claimed"
  echo "$(date '+%F %T') fold $f on GPU $G: train"
  CUDA_VISIBLE_DEVICES=$G PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 $PY scripts/train_detector.py \
      --fold $f --out $d --resume > logs/detector_gate0_fold$f.log 2>&1
  echo "$(date '+%F %T') fold $f train rc=$?"
  echo "$(date '+%F %T') fold $f on GPU $G: cache detections"
  CUDA_VISIBLE_DEVICES=$G PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 $PY scripts/cache_detections.py \
      --fold $f --run $d --score-min 0.01 > logs/cache_gate0_fold$f.log 2>&1
  echo "$(date '+%F %T') fold $f cache rc=$?"
done
echo "$(date '+%F %T') chain on GPU $G done"
