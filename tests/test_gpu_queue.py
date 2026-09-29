# tests/test_gpu_queue.py
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("gpu_queue", REPO / "scripts/gpu_queue.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_jobs_go_out_fold_by_fold():
    q = _load()
    assert q.job_list(["glioma", "metastasis", "infarct"], [0, 1]) == [
        ("glioma", 0), ("metastasis", 0), ("infarct", 0), ("glioma", 1), ("metastasis", 1), ("infarct", 1)]


def test_launches_take_only_free_idle_gpus_up_to_the_cap():
    q = _load()
    pending = q.job_list(["glioma", "metastasis", "infarct"], [0, 1])
    assert q.plan_launches(pending, set(), [1, 2, 4], 6) == [(("glioma", 0), 1), (("metastasis", 0), 2), (("infarct", 0), 4)]
    assert q.plan_launches(pending, {1, 2}, [1, 2, 4, 5], 6) == [(("glioma", 0), 4), (("metastasis", 0), 5)]   # ours are not reused
    assert q.plan_launches(pending, {0, 1, 2, 4, 5}, [6, 7], 6) == [(("glioma", 0), 6)]                           # one slot left
    assert q.plan_launches(pending, {0, 1, 2, 4, 5, 6}, [7], 6) == []                                              # at the cap
    assert q.plan_launches(pending, set(), [], 6) == [] and q.plan_launches([], set(), [1], 6) == []


def test_refusal_names_existing_outputs(tmp_path):
    q = _load()
    job = ("infarct", 2)
    assert q.refusal(tmp_path / "res", tmp_path / "logs", job, "nnUNetTrainer_250epochs") is None
    log = q.log_path(tmp_path / "logs", job, "nnUNetTrainer_250epochs")
    assert log.name == "Dataset906_ISLESInfarct_nnUNetTrainer_250epochs_fold2.log"
    log.parent.mkdir()
    log.write_text("")
    assert "log" in q.refusal(tmp_path / "res", tmp_path / "logs", job, "nnUNetTrainer_250epochs")
    (tmp_path / "res/Dataset906_ISLESInfarct/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_2").mkdir(parents=True)
    assert "result folder" in q.refusal(tmp_path / "res", tmp_path / "logs", job, "nnUNetTrainer_250epochs")
    assert q.refusal(tmp_path / "res", tmp_path / "logs", job, "nnUNetTrainer_5epochs") is None


def test_command_pins_the_gpu_sets_the_workers_and_quotes_the_log():
    cmd = _load().command(("glioma", 3), 5, "nnUNetTrainer_250epochs", Path("/tmp/a b/x.log"), Path("/r/scripts/nnunet_env.sh"))
    assert cmd[:2] == ["bash", "-c"]
    # the redirect covers the whole group, so a failing `source` is written to the job's log too
    assert cmd[2] == ("{ source /r/scripts/nnunet_env.sh && export nnUNet_n_proc_DA=6 && CUDA_VISIBLE_DEVICES=5 nice -n 19 "
                      "nnUNetv2_train 904 3d_fullres 3 -tr nnUNetTrainer_250epochs --npz; } > '/tmp/a b/x.log' 2>&1")


def test_skip_files_name_diseases_by_dataset_id(tmp_path):
    q = _load()
    assert q.skipped_diseases(tmp_path) == set()
    (tmp_path / "skip_905").write_text("fold 0 sensitivity 0.21 < 0.3")
    assert q.skipped_diseases(tmp_path) == {"metastasis"}


def test_dry_run_prints_the_first_round_and_launches_nothing(tmp_path, monkeypatch, capsys):
    q = _load()
    monkeypatch.setattr(q, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))
    monkeypatch.setattr(q, "query_nvidia_smi", lambda: "0, 30000\n1, 14\n2, 14\n3, 18000\n")
    monkeypatch.setattr(q, "query_busy_pids", lambda: {0: [11], 3: [22]})
    monkeypatch.setattr(q.subprocess, "Popen", lambda *a, **k: pytest.fail("a dry run must not launch"))
    assert q.main(["--diseases", "glioma", "metastasis", "infarct", "--folds", "0", "--gpus", "0", "1", "2", "3", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would launch ('glioma', 0) on GPU 1" in out and "would launch ('metastasis', 0) on GPU 2" in out
    assert "('infarct', 0) on GPU" not in out                   # GPUs 0 and 3 are busy: no card for the third job


def test_a_failing_gpu_query_costs_one_round_not_the_queue(tmp_path, monkeypatch, capsys):
    q = _load()

    def boom():
        raise subprocess.CalledProcessError(9, ["nvidia-smi"])

    monkeypatch.setattr(q, "query_nvidia_smi", boom)
    monkeypatch.setattr(q, "query_busy_pids", lambda: {})
    assert q.idle_now([0, 1]) is None
    assert "GPU query failed (CalledProcessError" in capsys.readouterr().out
    monkeypatch.setattr(q, "query_nvidia_smi", lambda: "0, 14\n1, not-a-number\n")
    assert q.idle_now([0, 1]) is None and "GPU query failed (ValueError" in capsys.readouterr().out
    monkeypatch.setattr(q, "query_nvidia_smi", lambda: "0, 14\n1, 30000\n")
    assert q.idle_now([0, 1]) == [0]
    # the loop itself: a failing query starts nothing and does not raise
    monkeypatch.setattr(q, "query_nvidia_smi", boom)
    monkeypatch.setattr(q, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))
    assert q.main(["--diseases", "infarct", "--folds", "0", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "nothing starts in this round" in out and "would launch" not in out


def test_the_job_log_receives_a_failure_of_the_environment_script(tmp_path):
    q = _load()
    log = tmp_path / "job.log"
    cmd = q.command(("infarct", 0), 0, "nnUNetTrainer_250epochs", log, tmp_path / "no_such_env.sh")
    done = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert done.returncode != 0 and "no_such_env.sh" in log.read_text()


class _Proc:
    """A training that ends with `code` at the n-th poll."""
    pid = 4242

    def __init__(self, code, polls):
        self.code, self.polls, self.returncode = code, polls, None

    def poll(self):
        self.polls -= 1
        if self.polls <= 0:
            self.returncode = self.code
        return self.returncode


def _loop(q, tmp_path, monkeypatch, outcomes, clock_step=1.0, on_launch=None):
    """Run main() for the three diseases' fold 0 and 1 on two pretend cards; outcomes: job -> (exit code, polls)."""
    launched, now = [], [0.0]
    monkeypatch.setattr(q, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))
    monkeypatch.setattr(q, "idle_now", lambda gpus: [0, 1])

    def tick():
        now[0] += clock_step
        return now[0]

    def popen(cmd, **kw):
        job = next(j for j in q.job_list(["glioma", "metastasis", "infarct"], [0, 1])
                   if f"nnUNetv2_train {q.DISEASES[j[0]]['id']} 3d_fullres {j[1]} " in cmd[2])
        launched.append(job)
        if on_launch:
            on_launch(job)
        return _Proc(*outcomes.get(job, (0, 2)))

    monkeypatch.setattr(q, "time", SimpleNamespace(time=tick, sleep=lambda s: None))   # the queue's own clock only
    monkeypatch.setattr(q.subprocess, "Popen", popen)
    code = q.main(["--diseases", "glioma", "metastasis", "infarct", "--folds", "0", "1", "--gpus", "0", "1"])
    return code, launched


def test_the_loop_runs_every_job_and_counts_them(tmp_path, monkeypatch, capsys):
    q = _load()
    code, launched = _loop(q, tmp_path, monkeypatch, {})
    assert code == 0 and launched == q.job_list(["glioma", "metastasis", "infarct"], [0, 1])
    assert "done; succeeded 6, failed 0 [], refused 0 [], never started 0 []" in capsys.readouterr().out


def test_a_skip_file_drops_a_disease_and_a_stop_file_everything_that_has_not_started(tmp_path, monkeypatch, capsys):
    q = _load()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "skip_905").write_text("fold 0 sensitivity 0.21 < 0.3")
    code, launched = _loop(q, tmp_path, monkeypatch, {})
    assert launched == [("glioma", 0), ("infarct", 0), ("glioma", 1), ("infarct", 1)] and code == 1
    assert "never started 2 [('metastasis', 0), ('metastasis', 1)]" in capsys.readouterr().out
    q2, other = _load(), tmp_path / "second"
    code, launched = _loop(q2, other, monkeypatch, {}, on_launch=lambda job: (
        (other / "logs" / "stop").write_text("stop") if job == ("metastasis", 0) else None))
    assert launched == [("glioma", 0), ("metastasis", 0)] and code == 1
    out = capsys.readouterr().out
    assert "stop file found: 4 jobs will not start" in out and "succeeded 2, failed 0 [], refused 0 [], never started 4" in out


def test_a_training_that_fails_at_once_stops_new_starts_and_a_late_failure_does_not(tmp_path, monkeypatch, capsys):
    q = _load()
    code, launched = _loop(q, tmp_path, monkeypatch, {("glioma", 0): (3, 1)})
    assert launched == [("glioma", 0), ("metastasis", 0)] and code == 1
    out = capsys.readouterr().out
    assert "finished ('glioma', 0) on GPU 0 with exit code 3" in out
    assert "failed within 10 min of its start: nothing new starts; 4 jobs will not start" in out
    assert "succeeded 1, failed 1 [('glioma', 0)], refused 0 [], never started 4" in out
    q2 = _load()
    code, launched = _loop(q2, tmp_path / "second", monkeypatch, {("glioma", 0): (3, 1)}, clock_step=700.0)
    assert len(launched) == 6 and code == 1                      # after 700 s a failure is that training's own
    assert "succeeded 5, failed 1 [('glioma', 0)], refused 0 [], never started 0 []" in capsys.readouterr().out


def test_a_refused_job_is_counted_and_the_others_run(tmp_path, monkeypatch, capsys):
    q = _load()
    (tmp_path / "logs").mkdir()
    q.log_path(tmp_path / "logs", ("infarct", 0), "nnUNetTrainer_250epochs").write_text("an earlier start")
    code, launched = _loop(q, tmp_path, monkeypatch, {})
    assert ("infarct", 0) not in launched and len(launched) == 5 and code == 1
    assert "refused 1 [('infarct', 0)]" in capsys.readouterr().out
