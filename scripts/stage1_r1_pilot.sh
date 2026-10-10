#!/bin/bash
# A one-card pilot of Stage I repair round 1 (2026-10-10; NOT the pre-registered round 1, which needs 3 or 6 cards):
# the four-level decoder with the round-1 settings on one card for 36k source-crop exposures (microbatch 12, global 12,
# lr 2.6e-4 = 4.5e-4 x sqrt(12/36), warmup 1333 steps = 16k exposures, cosine to zero at 3000 steps), then the
# host-only readout probe (seeds 0 1 2, the G1 crops, both arms) and the attribution diagnostic of the checkpoints at
# steps 1500 and 3000 (18k and 36k exposures). Differences from round 1 that a reader must keep in mind: the InfoNCE
# spans 12 crops per micro-batch instead of 36, and the schedule anneals within the pilot. Its purpose: an early,
# cheap reading of whether the repaired objective makes the frozen F2-F4 readout rise above the random backbone (run 0
# was 0.300 at 18k exposures against 0.383). Then, if THEN_C0=1, C0 on the same card.
#   CARD=1 [THEN_C0=1] bash scripts/stage1_r1_pilot.sh
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
: "${CARD:?CARD=one card index}"
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=8
PY=~/anaconda3/envs/nvgen/bin/python
DATA=/data2/congcong/data/FM_data/derived/aur
RUNS=$DATA/ssl_runs
SAMPLES=$DATA/samples_1mm_v1.json
SSL_SAMPLES=$DATA/ssl_manifest_v1/samples_ssl.json
VAL=$DATA/ssl_manifest_v1/val_patients.json
REC=docs/verification/2026-10-09/anatobind_brain_ssl_first/g1
STAMP=$(date +%Y%m%d_%H%M)
RUN=$RUNS/stage1_r1_pilot36k_$STAMP
LOG=$RUNS/stage1_r1_pilot_$STAMP.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "pilot on card $CARD -> $RUN"
CUDA_VISIBLE_DEVICES=$CARD $PY scripts/aur_ssl_train.py --samples "$SSL_SAMPLES" --out "$RUN" --seen-crops 36000 --microbatch 12 --grad-accum 1 \
  --workers 8 --lr 2.60e-04 --warmup-steps 1333 --val-every 500 --save-every 1500 --log-every 20 --val-volumes 32 --mask-ratio 0.60 \
  --contrast-weight 0.10 > "$RUN.log" 2>&1
[ -f "$RUN/resume_step3000.pt" ] || { say "the pilot ended without resume_step3000.pt: see $RUN.log"; exit 1; }
say "pilot trained: $(tr -d '\n' < "$RUN/summary.json" | cut -c1-240)"
for s in 1500 3000; do
  BB=$RUN/probe_step$s.backbone.pt
  $PY - "$RUN/resume_step$s.pt" "$BB" <<'PYEOF' >> "$LOG" 2>&1
import sys, torch
from anatobind.aur.ssl import checkpoint as CK
from anatobind.aur.ssl.model import StageOne
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = StageOne()
m.load_state_dict(ck["model"], strict=True)
CK.export_backbone(sys.argv[2], m.backbone, {**ck["meta"], "from_resume": sys.argv[1]})
PYEOF
  OUT=$REC/pilot_r1_probe_step${s}_$STAMP
  CUDA_VISIBLE_DEVICES=$CARD $PY scripts/aur_ssl_eval.py --checkpoint "$BB" --samples "$SAMPLES" --val-patients "$VAL" --out "$OUT" --seeds 0 1 2 --host-only > "$OUT.log" 2>&1 \
    && say "probe step $s: $($PY -c "import json; g=json.load(open('$OUT/g1_report.json'))['g1']; print('Stage I', round(g['host_macro_ssl'], 4), 'random', round(g['host_macro_random'], 4), 'gain', round(g['macro_dice_gain'], 4))")" \
    || say "probe step $s failed: see $OUT.log"
  CUDA_VISIBLE_DEVICES="" nice -n 19 $PY scripts/aur_ssl_attrib.py --checkpoint "$RUN/resume_step$s.pt" --samples "$SSL_SAMPLES" \
    --out "$REC/pilot_r1_attrib_step${s}_$STAMP.json" > "$REC/pilot_r1_attrib_step${s}_$STAMP.log" 2>&1 \
    && say "attribution step $s: $(head -1 "$REC/pilot_r1_attrib_step${s}_$STAMP.log")" || say "attribution step $s failed"
done
say "pilot done"
if [ "${THEN_C0:-0}" = "1" ]; then
  say "C0 on card $CARD"
  ARM=c0 CARDS=$CARD bash scripts/stage23_chain.sh
fi
