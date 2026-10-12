#!/bin/bash
# Pre-registered check of the new-recipe C2 (decision Q25, stage2/STAGE2_V2.md): at step 1000, frozen 13-host readout
# >= 0.65 AND validation A point Dice > 0.045. Waits for the run's step-1000 validation and resume checkpoint, probes the
# backbone on card 5 (shared with C0), prints the verdict. Reports only: stopping is decided after the report.
set -u
cd /data0/congcong/code/Project_Doing/foundation_model
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=8
PY=$HOME/anaconda3/envs/nvgen/bin/python
RUNS=/data2/congcong/data/FM_data/derived/aur/ssl_runs
D=/data2/congcong/data/FM_data/derived/aur
OUT=docs/verification/2026-10-09/anatobind_brain_ssl_first/stage2/frozen_probe
SCR=/tmp/claude-1002/-home-congcongliu/4c51bb86-6842-43b4-9b18-0dd733b5a3d4/scratchpad
say() { echo "[$(date '+%F %T')] $*"; }
RUN=""
until [ -n "$RUN" ]; do RUN=$(ls -d $RUNS/c2_stage2_240k_20261011_05* 2>/dev/null | head -1); [ -n "$RUN" ] || sleep 30; done
say "new C2 Stage II run: $RUN"
until [ -f "$RUN/resume_step1000.pt" ] && grep -q '"step": 1000' "$RUN/val.jsonl" 2>/dev/null; do sleep 30; done
T=$(pgrep -f "^$PY -u scripts/aur_train.py --stage II --init-backbone" | head -1)
while ls -l /proc/$T/fd 2>/dev/null | grep -q resume_step1000; do sleep 5; done; sleep 5
say "step 1000 saved; exporting the backbone and probing on card 5"
$PY - "$RUN/resume_step1000.pt" "$SCR/bb/c2v2_step1000.backbone.pt" <<'PYEOF'
import sys, torch
from anatobind.aur.ssl import checkpoint as CK
from anatobind.aur.model import AnatoBindBrain
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = AnatoBindBrain(use_checkpoint=False)
m.load_state_dict(ck["model"], strict=True)
CK.export_backbone(sys.argv[2], m.backbone, {**ck["meta"], "from_resume": sys.argv[1]})
PYEOF
CUDA_VISIBLE_DEVICES=5 $PY scripts/aur_ssl_eval.py --checkpoint "$SCR/bb/c2v2_step1000.backbone.pt" --samples "$D/samples_1mm_v1.json" \
  --val-patients "$D/ssl_manifest_v1/val_patients.json" --out "$OUT/c2v2_step1000" --seeds 0 1 2 --host-only --workers 4 > "$OUT/c2v2_step1000.log" 2>&1 \
  || { say "probe failed: see $OUT/c2v2_step1000.log"; exit 1; }
$PY - "$OUT/c2v2_step1000/g1_report.json" "$RUN/val.jsonl" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1]))["g1"]
val = {json.loads(l)["step"]: json.loads(l) for l in open(sys.argv[2])}
readout, a1000 = g["host_macro_ssl"], val[1000]["a_dice"]
a500 = val.get(500, {}).get("a_dice")
ok1, ok2 = readout >= 0.65, a1000 > 0.045
print(f"frozen readout at step 1000: {readout:.4f} (>= 0.65: {ok1}); validation A point Dice: step 500 {a500}, step 1000 {a1000:.4f} (> 0.045: {ok2})")
print("VERDICT:", "PASS - continue" if ok1 and ok2 else "FAIL - stop and report")
for s in sorted(val):
    v = val[s]; print(" val", s, "a_dice %.4f a_mask %.4f u_matched %s s %.4f" % (v["a_dice"], v["a_mask"], v.get("u_matched"), v["s"]))
PYEOF
