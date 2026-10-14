#!/bin/bash
# Fail-closed check of decision Q25 (stage2/STAGE2_V2.md) at step 1000 of a Stage II run: frozen 13-host readout
# >= 0.65 AND validation A point Dice > 0.045 (anatobind/aur/gates.py). Environment: STAGE2_DIR, SAMPLES, VAL_PATIENTS,
# PROBE_GPU required; PY optional. Everything is written under $STAGE2_DIR/gate_step1000/ (report.json, the exported
# backbone, the probe); an existing verdict is never overwritten. Exit 0 only on PASS; FAIL and every error exit
# non-zero. The 2026-10-11 version only printed its verdict and the chain ran on past a FAIL; it is kept as run in
# stage2/frozen_probe/c2v2_step1000_check.sh.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
PY=${PY:-$HOME/anaconda3/envs/nvgen/bin/python}
[ -n "${STAGE2_DIR:-}" ] || { echo "Q25 check: STAGE2_DIR is not set" >&2; exit 2; }
GATE_OUT=$STAGE2_DIR/gate_step1000
mkdir "$GATE_OUT" || { echo "Q25 check: cannot create $GATE_OUT (an earlier verdict is never overwritten)" >&2; exit 2; }
cd "$REPO" || exit 2
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO" OMP_NUM_THREADS=8

ERRORS=()
for v in SAMPLES VAL_PATIENTS PROBE_GPU; do [ -n "${!v:-}" ] || ERRORS+=("env_missing:$v"); done
[ -n "${SAMPLES:-}" ] && [ ! -f "$SAMPLES" ] && ERRORS+=("samples_missing")
[ -n "${VAL_PATIENTS:-}" ] && [ ! -f "$VAL_PATIENTS" ] && ERRORS+=("validation_manifest_missing")
PROBE_ERROR=""
if [ ${#ERRORS[@]} -eq 0 ] && [ -f "$STAGE2_DIR/resume_step1000.pt" ]; then
  "$PY" - "$STAGE2_DIR/resume_step1000.pt" "$GATE_OUT/backbone_step1000.pt" > "$GATE_OUT/export.log" 2>&1 <<'PYEOF' || PROBE_ERROR="backbone export failed, see export.log"
import sys, torch
from anatobind.aur.ssl import checkpoint as CK
from anatobind.aur.model import AnatoBindBrain
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = AnatoBindBrain(use_checkpoint=False)
m.load_state_dict(ck["model"], strict=True)
CK.export_backbone(sys.argv[2], m.backbone, {**ck["meta"], "from_resume": sys.argv[1]})
PYEOF
  if [ -z "$PROBE_ERROR" ]; then
    CUDA_VISIBLE_DEVICES=$PROBE_GPU "$PY" scripts/aur_ssl_eval.py --checkpoint "$GATE_OUT/backbone_step1000.pt" \
      --samples "$SAMPLES" --val-patients "$VAL_PATIENTS" --out "$GATE_OUT/probe" --seeds 0 1 2 --host-only \
      --workers 4 > "$GATE_OUT/probe.log" 2>&1 || PROBE_ERROR="probe failed, see probe.log"
  fi
fi

"$PY" - "$STAGE2_DIR" "$GATE_OUT" "$PROBE_ERROR" "${ERRORS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
from anatobind.aur.gates import build_q25_step1000_report

run, out, probe_error, input_errors = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] or None, sys.argv[4:]


def read(path, parse):
    try:
        return parse(path.read_text())
    except (OSError, ValueError):
        return None


report = build_q25_step1000_report(
    read(out / "probe" / "g1_report.json", json.loads), read(run / "val.jsonl", str),
    checkpoint_exists=(run / "resume_step1000.pt").is_file(), probe_error=probe_error, input_errors=input_errors)
(out / "report.json").write_text(json.dumps(report, indent=1))
print("Q25 step 1000:", "PASS" if report["passed"] else "FAIL", json.dumps(report["metrics"]), report["failures"])
sys.exit(0 if report["passed"] else 1)
PYEOF
