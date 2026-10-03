# tests/test_brain_anatomy_scripts.py
import importlib.util
from pathlib import Path

import pytest


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_launcher_refuses_existing_outputs_and_pins_one_job_per_card(tmp_path):
    t = _load("brain_anatomy_train")
    results, logs = tmp_path / "results", tmp_path / "logs"
    starts, reasons = t.plan(["student", "outline"], [3, 5], results, logs)
    assert starts == [("student", 3), ("outline", 5)] and reasons == {}
    t.result_dir(results, "student").mkdir(parents=True)
    starts, reasons = t.plan(["student", "outline"], [3], results, logs)
    assert starts == [("outline", 3)] and "result folder" in reasons["student"]
    logs.mkdir()
    t.log_path(logs, "outline").write_text("")
    starts, reasons = t.plan(["student", "outline"], [3, 5], results, logs)
    assert starts == [] and "log" in reasons["outline"]
    assert t.plan(["outline"], [], tmp_path / "r2", tmp_path / "l2") == ([], {"outline": "no idle GPU left"})
    # the student's classes have a side: no mirroring (spec A17); the outline keeps the default trainer
    assert t.train_command("student") == ["nnUNetv2_train", "907", "3d_fullres", "0", "-tr", "nnUNetTrainer_250epochs_NoMirroring"]
    assert t.train_command("outline") == ["nnUNetv2_train", "908", "2d", "0", "-tr", "nnUNetTrainer_250epochs"]
    cmd = t.launch_command("outline", 5, tmp_path, logs)
    assert cmd[:3] == ["setsid", "bash", "-c"] and "CUDA_VISIBLE_DEVICES=5 nice -n 19 nnUNetv2_train 908 2d 0" in cmd[3]
    assert "scripts/nnunet_env.sh" in cmd[3] and str(t.log_path(logs, "outline")) in cmd[3]
    assert t.result_dir(results, "outline").name == "fold_0" and "nnUNetTrainer_250epochs__nnUNetPlans__2d" in str(t.result_dir(results, "outline"))
    assert "nnUNetTrainer_250epochs_NoMirroring__nnUNetPlans__3d_fullres" in str(t.result_dir(results, "student"))
    assert "nnUNetTrainer_250epochs_NoMirroring" in t.log_path(logs, "student").name


def test_infer_entry_passes_the_arguments_through(tmp_path, capsys):
    s = _load("infer_brain_anatomy")
    seen = {}

    def fake_run(out, gpu, h5=None, nifti=None, box=None):
        seen.update(out=out, gpu=gpu, h5=h5, nifti=nifti, box=box)
        return {"brain_ml": 1234.5, "reliable_slices": [2, 12], "anatomy": str(out / "anatomy.nii.gz"), "box": list(box),
                "binding": {"host": "thalamus", "host_rule": "overlap", "host_side": "left", "host_fractions": {"thalamus": 1.0}}}

    s.main(["--h5", "/x/file.h5", "--out", str(tmp_path / "o"), "--gpu", "4", "--box", "1", "2", "3", "4", "5", "6"], run_fn=fake_run)
    assert seen == {"out": tmp_path / "o", "gpu": 4, "h5": Path("/x/file.h5"), "nifti": None, "box": (1, 2, 3, 4, 5, 6)}
    out = capsys.readouterr().out
    assert "brain 1234.5 mL" in out and "host thalamus (overlap), side left" in out
    with pytest.raises(SystemExit):
        s.parse(["--h5", "a", "--nifti", "b", "--out", "o", "--gpu", "0"])
