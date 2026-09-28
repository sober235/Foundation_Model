import importlib.util
import shlex
from pathlib import Path

spec = importlib.util.spec_from_file_location("bdt", Path(__file__).resolve().parents[1] / "scripts/brain_detector_train.py")
bdt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdt)


def test_idle_gpus_parse_nvidia_smi_and_skip_busy_cards():
    csv = "0, 14\n1, 30000\n2, 14\n3, 14\n4, 14\n"
    assert bdt.idle_gpus(csv, [0, 1, 2, 3], busy_pids={}) == [0, 2, 3]
    assert bdt.idle_gpus(csv, [0, 2], busy_pids={2: [123]}) == [0]


def test_assign_folds_round_robin_and_command():
    assert bdt.assign([0, 1, 2, 3, 4], [0, 1, 2, 3]) == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0)]
    assert bdt.train_command("2d", 3, "nnUNetTrainer_250epochs") == ["nnUNetv2_train", "903", "2d", "3", "-tr", "nnUNetTrainer_250epochs", "--npz"]


def test_existing_logs_names_only_the_folds_that_would_be_overwritten(tmp_path):
    (tmp_path / "2d_nnUNetTrainer_250epochs_fold0.log").write_text("old run")
    assert bdt.existing_logs("2d", "nnUNetTrainer_250epochs", [0, 1], tmp_path) == [tmp_path / "2d_nnUNetTrainer_250epochs_fold0.log"]
    assert bdt.existing_logs("2d", "nnUNetTrainer_250epochs", [1, 2], tmp_path) == []
    assert bdt.existing_logs("3d_fullres", "nnUNetTrainer_250epochs", [0], tmp_path) == []
    assert bdt.existing_logs("2d", "nnUNetTrainer_5epochs", [0], tmp_path) == []


def test_build_chain_quotes_trainer_in_log_redirect(tmp_path):
    """An adversarial --trainer must not let its `;`/space escape the redirect's shell quoting.
    Constructs strings only; nothing here is executed."""
    trainer = "x; echo pwned #"
    chain = bdt.build_chain("2d", 0, [0], trainer, Path("/repo"), tmp_path)
    assert chain[:3] == ["setsid", "bash", "-c"]
    inner = chain[3]
    tokens = shlex.split(inner)
    expected_log = str(tmp_path / f"2d_{trainer}_fold0.log")
    assert expected_log in tokens
    assert not any(t == ";" for t in tokens)


def test_existing_results_returns_only_the_folds_that_exist_in_fold_order(tmp_path):
    base = tmp_path / "Dataset903_FastMRIBrainSmallLesion" / "nnUNetTrainer_250epochs__nnUNetPlans__2d"
    (base / "fold_1").mkdir(parents=True)
    (base / "fold_3").mkdir(parents=True)
    assert bdt.existing_results(tmp_path, "2d", "nnUNetTrainer_250epochs", [0, 1, 2, 3]) == [base / "fold_1", base / "fold_3"]
    assert bdt.existing_results(tmp_path, "2d", "nnUNetTrainer_250epochs", [0, 2]) == []
    assert bdt.existing_results(tmp_path, "3d_fullres", "nnUNetTrainer_250epochs", [1]) == []
    assert bdt.existing_results(tmp_path, "2d", "nnUNetTrainer_5epochs", [1]) == []


def test_main_refuses_when_nnunet_results_fold_dir_already_exists(tmp_path, monkeypatch, capsys):
    """The results-dir check must run before any nvidia-smi query, so a training run already sitting in that
    folder is never at risk of being started a second time."""
    trainer = "nnUNetTrainer_250epochs_TESTONLY"
    fold_dir = tmp_path / "Dataset903_FastMRIBrainSmallLesion" / f"{trainer}__nnUNetPlans__2d" / "fold_0"
    fold_dir.mkdir(parents=True)
    monkeypatch.setenv("nnUNet_results", str(tmp_path))

    def boom():
        raise AssertionError("query_nvidia_smi must not be called")

    monkeypatch.setattr(bdt, "query_nvidia_smi", boom)

    rc = bdt.main(["--config", "2d", "--folds", "0", "--gpus", "0", "--trainer", trainer])
    assert rc == 1
    assert str(fold_dir) in capsys.readouterr().err
