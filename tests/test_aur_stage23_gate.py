"""scripts/stage23_chain.sh with the decision-Q25 gate (three-goal conformance plan, Task 3).

The chain runs from a copy in a temporary repository against a temporary data root; the trainer, the evaluation and
the Q25 checker are stubs that record their calls, so nothing here trains, reads real data or touches a GPU."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REC = "docs/verification/2026-10-09/anatobind_brain_ssl_first"

FAKE_TRAIN = '''
import json, sys
from pathlib import Path
args = sys.argv[1:]
opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
out = Path(opt("--out")); out.mkdir(parents=True, exist_ok=True)
resume = opt("--resume")
start = int(Path(resume).stem.split("_step")[1]) if resume else 0
if opt("--stop-after") is not None:
    step = start + int(opt("--stop-after"))
    (out / f"resume_step{step}.pt").write_bytes(b"stub")
    with open(out / "val.jsonl", "a") as f:
        f.write(json.dumps({"step": step, "a_dice": 0.05}) + "\\n")
    sys.exit(0)
(out / "summary.json").write_text(json.dumps({"steps": int(opt("--max-steps", "10000"))}))
(out / ("aur_stage2_best.pt" if opt("--stage") == "II" else "aur_stage3_best.pt")).write_bytes(b"stub")
'''
FAKE_EVAL = '''
import sys
from pathlib import Path
args = sys.argv[1:]
Path(args[args.index("--out") + 1]).mkdir(parents=True)
'''
FAKE_MERGE = '''
import json, sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--out") + 1]); out.mkdir(parents=True)
(out / "aggregate.json").write_text(json.dumps({"u": {"per_source": {"pdgm": {"thr": 0.6}}}}))
(out / "REPORT.md").write_text("stub")
'''
FAKE_PY = '''#!/bin/bash
echo "$*" >> "$STUB_LOG"
case "${{1:-}}" in
  scripts/aur_train.py) shift; exec {py} {stubs}/fake_train.py "$@" ;;
  scripts/aur_eval.py) shift; exec {py} {stubs}/fake_eval.py "$@" ;;
  scripts/aur_eval_merge.py) shift; exec {py} {stubs}/fake_merge.py "$@" ;;
  *) exec {py} "$@" ;;
esac
'''
FAKE_CHECKER = '''#!/bin/bash
echo "STAGE2_DIR=$STAGE2_DIR SAMPLES=$SAMPLES VAL_PATIENTS=$VAL_PATIENTS PROBE_GPU=$PROBE_GPU PY=$PY" >> "$CHECKER_LOG"
mkdir "$STAGE2_DIR/gate_step1000"
exit "$STUB_GATE_EXIT"
'''


class ChainRun:
    def __init__(self, result, tmp):
        self.returncode, self.stdout, self.stderr = result.returncode, result.stdout, result.stderr
        self.tmp = tmp

    def calls(self, script):
        log = self.tmp / "stub.log"
        lines = log.read_text().splitlines() if log.exists() else []
        return [line.split()[1:] for line in lines if line.split()[:1] == [script]]

    def checker_lines(self):
        log = self.tmp / "checker.log"
        return log.read_text().splitlines() if log.exists() else []


def opt(call, name):
    return call[call.index(name) + 1] if name in call else None


@pytest.fixture
def chain(tmp_path):
    repo, data, stubs = tmp_path / "repo", tmp_path / "data", tmp_path / "stubs"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "stage23_chain.sh", repo / "scripts" / "stage23_chain.sh")
    (repo / REC / "eval").mkdir(parents=True)
    (data / "derived/aur/ssl_runs").mkdir(parents=True)
    (data / "derived/aur/ssl_manifest_v1").mkdir(parents=True)
    (data / "derived/aur/samples_1mm_v1.json").write_text("[]")
    (data / "derived/aur/ssl_manifest_v1/val_patients.json").write_text("[]")
    stubs.mkdir()
    for name, text in (("fake_train.py", FAKE_TRAIN), ("fake_eval.py", FAKE_EVAL), ("fake_merge.py", FAKE_MERGE)):
        (stubs / name).write_text(text)
    (stubs / "py").write_text(FAKE_PY.format(py=sys.executable, stubs=stubs))
    (stubs / "checker.sh").write_text(FAKE_CHECKER)
    os.chmod(stubs / "py", 0o755)

    def run(**env):
        full = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "CARDS": "0", "WORKERS": "1",
                "PY": str(stubs / "py"), "TORCHRUN": str(stubs / "no_torchrun"), "FM_DATA_ROOT": str(data),
                "Q25_CHECKER": str(stubs / "checker.sh"), "STUB_LOG": str(tmp_path / "stub.log"),
                "CHECKER_LOG": str(tmp_path / "checker.log"), "STUB_GATE_EXIT": "0",
                "INIT_BACKBONE": str(tmp_path / "ssl_stage1_best.pt"), "G1_REPORT": str(tmp_path / "g1_report.json")}
        full.update(env)
        result = subprocess.run(["bash", str(repo / "scripts" / "stage23_chain.sh")], env=full,
                                capture_output=True, text=True, timeout=300)
        return ChainRun(result, tmp_path)

    run.data = data
    return run


def stage2_calls(run):
    return [c for c in run.calls("scripts/aur_train.py") if "_stage2_240k_" in opt(c, "--out")]


def test_q25_gate_failure_stops_the_chain_at_step_1000(chain):
    run = chain(ARM="c2", U_MASK_PROFILE="legacy", II_BB="5e-5", C2_STEP1000_GATE="1", STUB_GATE_EXIT="1")

    assert run.returncode != 0
    stage2 = stage2_calls(run)
    assert len(stage2) == 1 and opt(stage2[0], "--stop-after") == "1000"
    assert not [c for c in run.calls("scripts/aur_train.py") if opt(c, "--stage") == "III"]
    assert not run.calls("scripts/aur_eval.py") and not run.calls("scripts/aur_eval_merge.py")
    out = Path(opt(stage2[0], "--out"))
    assert run.checker_lines() == [f"STAGE2_DIR={out} SAMPLES={chain.data}/derived/aur/samples_1mm_v1.json "
                                   f"VAL_PATIENTS={chain.data}/derived/aur/ssl_manifest_v1/val_patients.json "
                                   f"PROBE_GPU=0 PY={run.tmp}/stubs/py"]
    assert (out / "resume_step1000.pt").exists() and (out / "val.jsonl").exists()


def test_q25_gate_pass_resumes_the_same_run_then_stage3_and_evaluation(chain):
    run = chain(ARM="c2", U_MASK_PROFILE="legacy", II_BB="5e-5", C2_STEP1000_GATE="1", STUB_GATE_EXIT="0")

    assert run.returncode == 0, run.stdout + run.stderr
    first, resumed = stage2_calls(run)
    out = opt(first, "--out")
    assert opt(first, "--stop-after") == "1000"
    assert opt(resumed, "--out") == out and opt(resumed, "--resume") == f"{out}/resume_step1000.pt"
    assert "--stop-after" not in resumed
    stage3 = [c for c in run.calls("scripts/aur_train.py") if opt(c, "--stage") == "III"]
    assert len(stage3) == 1 and opt(stage3[0], "--resume-stage2") == f"{out}/aur_stage2_best.pt"
    assert len(run.calls("scripts/aur_eval.py")) == 2 and len(run.checker_lines()) == 1


@pytest.mark.parametrize("gate", [None, "0"])
def test_the_q25_recipe_without_the_gate_exits_before_training(chain, gate):
    env = {"ARM": "c2", "U_MASK_PROFILE": "legacy", "II_BB": "5e-5"}
    if gate is not None:
        env["C2_STEP1000_GATE"] = gate
    run = chain(**env)

    assert run.returncode != 0
    assert run.calls("scripts/aur_train.py") == []


@pytest.mark.parametrize("arm", ["c0", "c2"])
def test_the_gate_is_rejected_outside_the_q25_recipe(chain, arm):
    run = chain(ARM=arm, U_MASK_PROFILE="unmatched_empty_v1", II_BB="5e-5", C2_STEP1000_GATE="1")

    assert run.returncode != 0
    assert run.calls("scripts/aur_train.py") == []


@pytest.mark.parametrize("profile", [None, "new"])
def test_the_chain_needs_a_known_u_mask_profile(chain, profile):
    env = {"ARM": "c0", "II_BB": "5e-5"}
    if profile is not None:
        env["U_MASK_PROFILE"] = profile
    run = chain(**env)

    assert run.returncode != 0
    assert run.calls("scripts/aur_train.py") == []


@pytest.mark.parametrize("arm", ["c0", "c2"])
def test_the_new_profile_runs_without_the_gate_and_every_stage_gets_seed_and_profile(chain, arm):
    run = chain(ARM=arm, U_MASK_PROFILE="unmatched_empty_v1", II_BB="5e-5", SEED="7")

    assert run.returncode == 0, run.stdout + run.stderr
    assert run.checker_lines() == []
    stage2 = stage2_calls(run)
    assert len(stage2) == 1 and "--stop-after" not in stage2[0]
    calls = run.calls("scripts/aur_train.py")
    assert len(calls) == 4                                   # smoke, smoke resume, Stage II, Stage III
    for call in calls:
        assert opt(call, "--seed") == "7" and opt(call, "--u-mask-profile") == "unmatched_empty_v1"
