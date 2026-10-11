#!/bin/bash
# Diagnostic (2026-10-11): does Stage II keep the features Stage I built? The frozen 13-host readout of G1 (host only,
# 3 seeds, the same calibration / validation crops) on the backbone of Stage II resume checkpoints: C2 at steps 500 and
# 1500, C0 at step 1500 and at the end of its Stage II (step 10000). C2 starts from the G1 export (0.724); C0 from a
# random backbone (0.383). One probe per free card, in parallel.
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=8
PY=$HOME/anaconda3/envs/nvgen/bin/python
D=/data2/congcong/data/FM_data/derived/aur
RUNS=$D/ssl_runs
SAMPLES=$D/samples_1mm_v1.json
VAL=$D/ssl_manifest_v1/val_patients.json
OUT=docs/verification/2026-10-09/anatobind_brain_ssl_first/stage2/frozen_probe
SCR=/tmp/claude-1002/-home-congcongliu/4c51bb86-6842-43b4-9b18-0dd733b5a3d4/scratchpad
C2=$RUNS/c2_stage2_240k_20261011_0216
C0=$RUNS/c0_stage2_240k_20261010_0903
mkdir -p "$OUT" "$SCR/bb"
say() { echo "[$(date '+%F %T')] $*"; }
export_bb() {   # export_bb <resume> <out>: the backbone of a Stage II resume checkpoint, in the Stage I export format
  $PY - "$1" "$2" <<'PYEOF'
import sys, torch
from anatobind.aur.ssl import checkpoint as CK
from anatobind.aur.model import AnatoBindBrain
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = AnatoBindBrain(use_checkpoint=False)
m.load_state_dict(ck["model"], strict=True)
CK.export_backbone(sys.argv[2], m.backbone, {**ck["meta"], "from_resume": sys.argv[1]})
PYEOF
}
probe() {   # probe <card> <tag> <resume>
  local card=$1 tag=$2 res=$3
  [ -f "$res" ] || { say "$tag: $res missing"; return 1; }
  export_bb "$res" "$SCR/bb/$tag.backbone.pt" || { say "$tag: export failed"; return 1; }
  CUDA_VISIBLE_DEVICES=$card $PY scripts/aur_ssl_eval.py --checkpoint "$SCR/bb/$tag.backbone.pt" --samples "$SAMPLES" --val-patients "$VAL" \
    --out "$OUT/$tag" --seeds 0 1 2 --host-only --workers 4 > "$OUT/$tag.log" 2>&1 || { say "$tag: probe failed, see $OUT/$tag.log"; return 1; }
  say "$tag: $($PY -c "import json; g=json.load(open('$OUT/$tag/g1_report.json'))['g1']; print('frozen host macro Dice', round(g['host_macro_ssl'], 4), 'random', round(g['host_macro_random'], 4))")"
}
CARD=${CARD:-5}          # 2026-10-11 03:58: no free card; the probes share card 5 with C0's Stage III (66 GB + 7-9 GB), one at a time
TAGS=(c2_step500 c2_step1500 c0_step1500 c0_step10000)
RES=("$C2/resume_step500.pt" "$C2/resume_step1500.pt" "$C0/resume_step1500.pt" "$C0/resume_step10000.pt")
say "probes on card $CARD, one at a time"
for i in "${!TAGS[@]}"; do
  probe "$CARD" "${TAGS[$i]}" "${RES[$i]}"
done
say "all probes done"
